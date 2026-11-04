"""Checkpoint and optimizer wiring checks; never perform optimizer steps."""
import pytest
import torch

from fsa_study.engine import (_freeze_encoder_bn, _model_config_digest, _optimizer,
                              checkpoint_model_kwargs, study_loss, training_model_config,
                              validate_checkpoint)
from fsa_study.model import StudyModel
from run_study import build_parser


@pytest.mark.parametrize("backbone", ["convnext_tiny", "resnet50"])
def test_cnn_loss_optimizer_and_checkpoint_roundtrip(backbone, tmp_path):
    args = build_parser().parse_args([
        "train", "--manifest", "/unused/manifest.json", "--output", str(tmp_path / "run"),
        "--backbone", backbone, "--feature-mode", "last", "--random-init",
        "--generator", "native_conv", "--utilizer", "sk", "--channels", "8"])
    config = training_model_config(args)
    assert config["backbone"] == backbone and config["feature_mode"] == "last"
    model = StudyModel(**config).eval()
    _freeze_encoder_bn(model)
    optimizer = _optimizer(model, 1e-4)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-4]
    assert not optimizer.state
    for stem in model.frozen_stem_modules:
        assert all(not p.requires_grad for p in stem.parameters())
    grouped = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}
    image = torch.rand(2, 3, 64, 64)
    mask, edge = (torch.rand(2, 1, 64, 64) > .5).float(), torch.zeros(2, 1, 64, 64)
    loss, terms = study_loss(model(image), mask, edge, .5)
    assert set(terms) == {"seg", "deep", "region", "edge", "ual"}
    loss.backward()  # Connectivity only: zero optimizer steps.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in grouped)
    assert not optimizer.state
    path = tmp_path / "initialization.pth"
    torch.save({"schema_version": 1, "model_config": config, "model": model.state_dict(),
                "model_config_sha256": _model_config_digest(config), "manifest_sha256": "fixture"}, path)
    checkpoint = torch.load(path, weights_only=True)
    validate_checkpoint(checkpoint, {"content_sha256": "fixture"})
    restored = StudyModel(**checkpoint_model_kwargs(checkpoint["model_config"], StudyModel)).eval()
    restored.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(image)["logits"], model(image)["logits"], rtol=0, atol=0)
