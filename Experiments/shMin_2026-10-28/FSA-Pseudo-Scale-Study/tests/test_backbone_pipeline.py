"""Plan/report compatibility and inference checks; never run an optimizer."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fsa_study.data import sha256_file
from fsa_study.leaderboard import build_leaderboard
from plan_study import main as create_plan
from run_plan import completed, read_plan
from run_study import build_parser
from summarize_study import aggregate

HERE = Path(__file__).resolve().parents[1]


def _plan(tmp_path, *flags):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"content_sha256": "synthetic"}))
    output = tmp_path / "plan"
    create_plan(["--manifest", str(manifest), "--output", str(output),
                 "--results", str(tmp_path / "runs"), "--python", sys.executable,
                 "--random-init", "--device", "cpu", *flags])
    return output / "plan.json"


def test_selected_plan_is_one_case_one_seed_without_implicit_controls(tmp_path):
    path = _plan(tmp_path, "--phase", "selected", "--generator", "native_conv", "--utilizer", "sk",
                 "--backbone", "convnext_tiny", "--feature-mode", "last")
    plan = read_plan(path)
    assert plan["phase"] == "selected"
    assert plan["run_count"] == plan["variant_count"] == 1
    assert plan["seeds"] == [3407]
    job, = plan["jobs"]
    assert job["variant"] == "native_conv__sk"
    args = build_parser().parse_args(job["train"][2:])
    assert (args.backbone, args.feature_mode) == ("convnext_tiny", "last")
    assert not args.baseline and not args.legacy_reference
    assert not Path(job["run"]).exists()
    assert not (tmp_path / "runs").exists()
    for action in ("train", "evaluate", "profile", "run_all"):
        result = subprocess.run(["bash", "-n", str(path.parent / f"{action}.sh")],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_selected_baseline_does_not_add_a_second_case(tmp_path):
    plan = read_plan(_plan(tmp_path, "--phase", "selected", "--baseline", "--backbone", "resnet50", "--feature-mode", "last"))
    assert plan["run_count"] == 1
    assert plan["jobs"][0]["variant"] == "single"
    assert "--baseline" in plan["jobs"][0]["train"]


@pytest.mark.parametrize("flags", [[], ["--phase", "priority8", "--seeds", "3407"]])
def test_default_or_explicit_priority8_is_exactly_previous_shortlist_one_seed(tmp_path, flags):
    plan = read_plan(_plan(tmp_path, "--backbone", "convnext_tiny", "--feature-mode", "last", *flags))
    assert plan["phase"] == "priority8"
    assert plan["run_count"] == plan["variant_count"] == 8
    assert plan["seeds"] == [3407]
    assert [job["variant"] for job in plan["jobs"]] == [
        "single", "native_conv__asff", "native_conv__sk", "native_conv__mean",
        "projection__mean", "dysample__asff", "dysample__sk", "dysample__mean",
    ]
    for job in plan["jobs"]:
        args = build_parser().parse_args(job["train"][2:])
        assert args.seed == 3407 and args.backbone == "convnext_tiny" and args.feature_mode == "last"
        assert not args.legacy_reference
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("flags", [
    ["--phase", "cross", "--backbone", "convnext_tiny"],
    ["--phase", "generation", "--feature-mode", "last"],
    ["--phase", "priority8", "--baseline"],
])
def test_incompatible_reference_or_baseline_rejected_before_writing_plan(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        _plan(tmp_path, *flags)
    assert error.value.code == 2
    assert not (tmp_path / "plan").exists()
    assert not (tmp_path / "runs").exists()


def test_plan_metadata_must_match_command_backbone(tmp_path):
    path = _plan(tmp_path, "--backbone", "convnext_tiny", "--feature-mode", "last")
    plan = json.loads(path.read_text())
    plan["backbone"] = "resnet50"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="backbone disagrees"):
        read_plan(path)


def _mark_completed(job, historical=False):
    run = Path(job["run"])
    run.mkdir(parents=True)
    config = vars(build_parser().parse_args(job["train"][2:]))
    config["manifest_sha256"] = "synthetic"
    if historical:
        config.pop("backbone")
        config.pop("feature_mode")
        for key in ("output_stride", "input_mode", "distill_teacher", "distill_weight"):
            config.pop(key)
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"status": "training_complete"}))
    (run / "best.pth").write_bytes(b"synthetic checkpoint identity, not model weights")


def test_historical_pvt_completed_skip_still_works_and_rejects_cnn_change(tmp_path):
    plan = read_plan(_plan(tmp_path, "--phase", "selected"))
    job, = plan["jobs"]
    _mark_completed(job, historical=True)
    assert completed(job, "train")
    job["train"][job["train"].index("--backbone") + 1] = "convnext_tiny"
    with pytest.raises(ValueError, match="configuration disagrees"):
        completed(job, "train")


def test_completed_cnn_cannot_skip_with_different_feature_mode(tmp_path):
    job, = read_plan(_plan(tmp_path, "--phase", "selected", "--backbone", "convnext_tiny", "--feature-mode", "last"))["jobs"]
    _mark_completed(job)
    assert completed(job, "train")
    job["train"][job["train"].index("--feature-mode") + 1] = "multistage"
    with pytest.raises(ValueError, match="configuration disagrees"):
        completed(job, "train")


def _report_run(root, name, seed, **model_options):
    run = root / name
    run.mkdir(parents=True)
    model = {"baseline": False, "legacy_reference": False, "generator": "native_conv",
             "utilizer": "sk", "channels": 48, "seed": seed, **model_options}
    (run / "config.json").write_text(json.dumps({"model_config": model, "manifest_sha256": "synthetic"}))
    (run / "status.json").write_text(json.dumps({"status": "training_complete", "best_val_mae": .1}))
    (run / "best.pth").write_bytes(b"synthetic checksum fixture")
    checkpoint_hash = sha256_file(run / "best.pth")
    (run / "test").mkdir()
    (run / "test/metrics.json").write_text(json.dumps({
        "split": "test", "model_config": model, "manifest_sha256": "synthetic",
        "checkpoint_sha256": checkpoint_hash, "datasets": {"camo": {"count": 2, "S": .8}},
    }))
    (run / "profile.json").write_text(json.dumps({
        "checkpoint_sha256": checkpoint_hash, "parameters": 100,
        "latency_ms": {"median": 1.0}, "images_per_second": 1000.0,
        "protocol": {"device": "cpu", "dtype": "float32", "size": 384, "batch_size": 1},
        "environment": {"device": "synthetic"},
    }))


def test_backbone_feature_mode_and_architecture_have_separate_rank_cohorts(tmp_path):
    _report_run(tmp_path, "old-pvt", 3407)
    _report_run(tmp_path, "explicit-pvt", 3408, backbone="pvt_v2_b2", feature_mode="multistage", architecture="common")
    _report_run(tmp_path, "cnn-four", 3407, backbone="convnext_tiny", feature_mode="multistage")
    _report_run(tmp_path, "cnn-last", 3407, backbone="convnext_tiny", feature_mode="last")
    _report_run(tmp_path, "resnet-last", 3407, backbone="resnet50", feature_mode="last")
    _report_run(tmp_path, "other-decoder", 3407, architecture="future_decoder")
    report = aggregate(tmp_path)
    rows = [row for row in report["rows"] if row["dataset"] == "validation"]
    assert len(rows) == 5
    pvt = next(row for row in rows if row["protocol"]["backbone"] == "pvt_v2_b2"
               and row["protocol"]["architecture"] == "common")
    assert pvt["n"] == 2 and pvt["seeds"] == [3407, 3408]
    boards = build_leaderboard(report)["boards"]
    for dataset, metric in (("validation", "MAE"), ("camo", "S"), ("inference", "parameters")):
        matched = [b for b in boards if b["dataset"] == dataset and b["metric"] == metric]
        assert len(matched) == 5
        assert all(len(b["rows"]) == 1 for b in matched)


def test_scale_reference_has_own_variant_and_stride_teacher_have_own_cohorts(tmp_path):
    common = {"backbone": "convnext_tiny", "feature_mode": "last", "utilizer": "mean"}
    _report_run(tmp_path, "single", 3407, **common, baseline=True)
    _report_run(tmp_path, "pseudo", 3407, **common)
    _report_run(tmp_path, "reference", 3407, **common, input_mode="image_pyramid", generator="identity")
    _report_run(tmp_path, "dense", 3407, **common, output_stride=16)
    _report_run(tmp_path, "teacher", 3407, **common)
    config_path = tmp_path / "teacher/config.json"
    config = json.loads(config_path.read_text())
    config["feature_distillation"] = {"checkpoint_sha256": "teacher-identity", "weight": .1}
    config_path.write_text(json.dumps(config))
    boards = [b for b in build_leaderboard(aggregate(tmp_path))["boards"]
              if b["dataset"] == "camo" and b["metric"] == "S"]
    assert len(boards) == 3
    core = next(b for b in boards if b["protocol"]["output_stride"] == 32
                and b["protocol"]["feature_distillation"] is None)
    assert {row["variant"] for row in core["rows"]} == {"single", "native_conv__mean", "image_pyramid__mean"}
    assert all(len(b["rows"]) == 1 for b in boards if b is not core)


def test_completed_teacher_identity_is_checked_even_when_path_moves(tmp_path):
    teacher = tmp_path / "teacher.pth"
    teacher.write_bytes(b"synthetic teacher identity")
    path = _plan(tmp_path, "--phase", "selected", "--backbone", "convnext_tiny", "--feature-mode", "last",
                 "--generator", "native_conv", "--training-protocol", "full_train",
                 "--distill-teacher", str(teacher), "--distill-weight", ".1")
    job, = read_plan(path)["jobs"]
    _mark_completed(job)
    config_path = Path(job["run"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["feature_distillation"] = {"checkpoint_sha256": sha256_file(teacher)}
    config_path.write_text(json.dumps(config))
    assert completed(job, "train")
    relocated = tmp_path / "relocated.pth"
    teacher.rename(relocated)
    job["train"][job["train"].index("--distill-teacher") + 1] = str(relocated)
    assert completed(job, "train")
    relocated.write_bytes(b"different teacher identity")
    with pytest.raises(ValueError, match="feature teacher disagrees"):
        completed(job, "train")


def test_last_stage_profile_reports_final_endpoint_and_backbone(tmp_path):
    output = tmp_path / "profile.json"
    process = subprocess.run([
        sys.executable, str(HERE / "profile_study.py"), "--backbone", "convnext_tiny",
        "--feature-mode", "last", "--generator", "native_conv", "--utilizer", "sk",
        "--channels", "8", "--size", "64", "--device", "cpu", "--warmup", "0", "--repeats", "1",
        "--output", str(output),
    ], capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    report = json.loads(output.read_text())
    assert report["protocol"]["backbone"] == "convnext_tiny"
    assert report["protocol"]["feature_mode"] == "last"
    assert [row["level"] for row in report["diagnostics"]] == [5]
    assert report["parameters"] > 0
    assert report["protocol"]["weights"] == "random_initialization"
