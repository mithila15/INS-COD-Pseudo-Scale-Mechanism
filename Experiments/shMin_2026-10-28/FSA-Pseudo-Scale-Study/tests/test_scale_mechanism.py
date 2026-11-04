"""Synthetic architecture, target isolation and orchestration checks; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from fsa_study.conditions import variant_name
from fsa_study.distillation import FrozenScaleTeacher, load_feature_teacher, scale_feature_loss
from fsa_study.engine import _model_config_digest, checkpoint_model_kwargs, study_loss
from fsa_study.model import StudyModel
from fsa_study.utilizers import build_utilizer
from plan_study import main as make_plan
from run_plan import read_plan
from run_study import build_parser

HERE = Path(__file__).resolve().parents[1]


def cnn(**options):
    return StudyModel(backbone="convnext_tiny", feature_mode="last", channels=8,
                      generator="native_conv", **options).eval()


def test_stride16_retains_all_stages_weights_and_decoder():
    low, high = cnn(output_stride=32), cnn(output_stride=16)
    assert low.state_dict().keys() == high.state_dict().keys()
    assert all(torch.equal(v, high.state_dict()[k]) for k, v in low.state_dict().items())
    observed = []
    hook = high.encoder.features[7].register_forward_hook(lambda module, inp, out: observed.append(out.shape))
    with torch.no_grad():
        a = low.encoder(torch.rand(1, 3, 64, 96))
        b = high.encoder(torch.rand(1, 3, 64, 96))
        prediction = high(torch.rand(1, 3, 64, 96))
    hook.remove()
    assert list(a) == list(b) == ["reduction_5"]
    assert a["reduction_5"].shape == (1, 768, 2, 3)
    assert b["reduction_5"].shape == (1, 768, 4, 6)
    assert observed == [(1, 768, 4, 6), (1, 768, 4, 6)]
    assert prediction["logits"].shape == (1, 1, 64, 96)
    assert len(high.last_decoder.upsampling) == len(low.last_decoder.upsampling) == 5


@pytest.mark.parametrize("stride", [16, 32])
def test_true_scale_reference_encodes_resized_rgb_three_times(stride):
    model = cnn(input_mode="image_pyramid", output_stride=stride)
    assert not hasattr(model, "generators")
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda m, x: calls.append(x[0].shape[-2:]))
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x, return_features=True)
    hook.remove()
    assert calls == [(64, 64), (96, 96), (32, 32)]
    bank = output["feature_banks"][0]
    with torch.no_grad():
        large = model.projected_features(F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False))[0]
    torch.testing.assert_close(bank[1], F.interpolate(large, size=bank[0].shape[-2:], mode="bilinear", align_corners=False))
    assert model.metadata["encoder_calls_per_image"] == 3
    assert variant_name(model.config) == "image_pyramid__mean"


def test_single_pass_and_old_configuration_compatibility():
    model = cnn()
    old_config = {k: v for k, v in model.config.items() if k not in {"output_stride", "input_mode"}}
    old = StudyModel(**checkpoint_model_kwargs(old_config, StudyModel)).eval()
    old.load_state_dict(model.state_dict(), strict=True)
    calls = []
    h = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x)["logits"], old(x)["logits"], rtol=0, atol=0)
    h.remove()
    assert calls == [1]


@pytest.mark.parametrize("options", [
    {"backbone": "pvt_v2_b2", "feature_mode": "last", "output_stride": 16},
    {"backbone": "convnext_tiny", "output_stride": 16},
    {"feature_mode": "last", "input_mode": "image_pyramid", "baseline": True},
    {"input_mode": "image_pyramid"},
])
def test_invalid_scale_combinations_fail(options):
    with pytest.raises(ValueError):
        StudyModel(**options)


def test_anchor_gate_keeps_direct_main_path_and_connects_auxiliaries():
    utilizer = build_utilizer("anchor_gated", 8)
    bank = [torch.randn(2, 8, 4, 4, requires_grad=True) for _ in range(3)]
    output = utilizer(bank)
    expected = bank[0] + .01 * utilizer.projection(torch.cat(bank[1:], dim=1))
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(bank[0].grad, torch.ones_like(bank[0]))
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in bank)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in utilizer.parameters())


def test_feature_targets_detached_and_branch_order_matters():
    main = torch.randn(2, 8, 4, 4)
    a, b = torch.randn_like(main), torch.randn_like(main)
    student = [[main, a.clone().requires_grad_(), b.clone().requires_grad_()]]
    teacher = [[main, a.clone().requires_grad_(), b.clone().requires_grad_()]]
    same = scale_feature_loss(student, teacher)
    swapped = scale_feature_loss(student, [[main, b, a]])
    assert abs(same.item()) < 1e-6 and swapped > .1
    same.backward()
    assert teacher[0][1].grad is None and teacher[0][2].grad is None


def test_frozen_teacher_training_loss_without_optimizer_updates():
    student = cnn(output_stride=16)
    teacher = FrozenScaleTeacher(cnn(input_mode="image_pyramid", output_stride=16), {})
    before = {k: v.clone() for k, v in teacher.model.state_dict().items()}
    x = torch.rand(2, 3, 64, 64)
    output = student(x, return_features=True)
    mask = (torch.rand(2, 1, 64, 64) > .5).float()
    supervised, _ = study_loss(output, mask, mask, .5)
    loss = supervised + .1 * teacher.loss(output["feature_banks"], x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.model.parameters())
    assert all(torch.equal(v, teacher.model.state_dict()[k]) for k, v in before.items())
    assert not teacher.model.training
    assert not any("teacher" in k for k in student.state_dict())


def _teacher_checkpoint():
    model = cnn(input_mode="image_pyramid")
    config = model.config
    return {"schema_version": 1, "model": model.state_dict(), "model_config": config,
            "model_config_sha256": _model_config_digest(config), "manifest_sha256": "fixture",
            "epoch": 1, "step": 1,
            "run_config": {"model_config": config, "manifest_sha256": "fixture", "epochs": 1,
                           "training_protocol": "full_train", "size": 64,
                           "checkpoint_selection": "fixed final epoch; no validation or test selection"}}


def test_teacher_checkpoint_and_student_standalone_roundtrip(tmp_path):
    checkpoint = _teacher_checkpoint()  # Synthetic initialized weights, no training performed.
    path = tmp_path / "teacher.pth"
    torch.save(checkpoint, path)
    student = cnn()
    teacher = load_feature_teacher(path, {"content_sha256": "fixture"}, student.config, "full_train", 64, "cpu")
    assert teacher.metadata["inference_teacher"] is False
    state = student.state_dict()
    path.unlink()
    restored = StudyModel(**student.config).eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        x = torch.rand(1, 3, 64, 64)
        torch.testing.assert_close(restored(x)["logits"], student(x)["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["manifest", "stride", "input_mode", "epoch", "size", "protocol"])
def test_invalid_teacher_rejected(tmp_path, monkeypatch, invalid):
    cp = _teacher_checkpoint()
    if invalid == "manifest": cp["manifest_sha256"] = "other-data"
    if invalid == "stride": cp["model_config"]["output_stride"] = 16
    if invalid == "input_mode": cp["model_config"]["input_mode"] = "single"
    if invalid == "epoch": cp["epoch"] = 0
    if invalid == "size": cp["run_config"]["size"] = 96
    if invalid == "protocol": cp["run_config"]["training_protocol"] = "holdout"
    cp["model_config_sha256"] = _model_config_digest(cp["model_config"])
    path = tmp_path / "fixture.pth"; path.write_bytes(b"fixture")
    monkeypatch.setattr(torch, "load", lambda *a, **k: cp)
    with pytest.raises(ValueError):
        load_feature_teacher(path, {"content_sha256": "fixture"}, cnn().config, "full_train", 64, "cpu")


@pytest.mark.parametrize("stage,count,stride", [("core", 4, 32), ("resolution", 4, 16), ("fusion", 2, 32), ("distill", 2, 32)])
def test_launcher_creates_only_requested_stage(tmp_path, stage, count, stride):
    manifest = tmp_path / "manifest.json"; manifest.write_text('{}')
    weights = tmp_path / "weights.pth"; weights.write_bytes(b'plan-only fixture')
    teacher = tmp_path / "teacher.pth"; teacher.write_bytes(b'plan-only fixture')
    output = tmp_path / "output"
    command = ["bash", str(HERE / "launch_scale_mechanism.sh"), "--prepare-only", stage, str(weights)]
    if stage == "distill": command.append(str(teacher))
    process = subprocess.run(command, env={**os.environ, "PY": sys.executable, "MANIFEST": str(manifest), "OUTPUT": str(output)},
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    plan = read_plan(output / "plans" / stage / "plan.json")
    assert plan["run_count"] == count and plan["seeds"] == [3407] and plan["output_stride"] == stride
    assert not (output / "runs").exists()
    if count == 4:
        assert [j["variant"] for j in plan["jobs"]] == ["single", "projection__mean", "native_conv__mean", "image_pyramid__mean"]
    for job in plan["jobs"]:
        args = build_parser().parse_args(job["train"][2:])
        assert args.output_stride == stride
        assert (args.distill_weight > 0) == (stage == "distill")


def test_teacher_arguments_cannot_silently_be_ignored(tmp_path):
    with pytest.raises(SystemExit):
        make_plan(["--phase", "selected", "--backbone", "convnext_tiny", "--feature-mode", "last",
                   "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "plan"),
                   "--results", str(tmp_path / "runs"), "--random-init", "--distill-weight", ".1"])
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("stride", [16, 32])
def test_image_pyramid_profile_counts_real_encoder_cost_and_diagnostics(tmp_path, stride):
    output = tmp_path / "profile.json"
    command = [sys.executable, str(HERE / "profile_study.py"), "--backbone", "convnext_tiny",
               "--feature-mode", "last", "--output-stride", str(stride), "--input-mode", "image_pyramid",
               "--size", "64", "--channels", "8", "--warmup", "0", "--repeats", "1", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["metadata"]["encoder_calls_per_image"] == 3
    assert report["protocol"]["output_stride"] == stride
    assert len(report["diagnostics"]) == 1 and len(report["diagnostics"][0]["branches"]) == 2
    assert not report["complexity"]["coverage"]["unsupported_operators"]
