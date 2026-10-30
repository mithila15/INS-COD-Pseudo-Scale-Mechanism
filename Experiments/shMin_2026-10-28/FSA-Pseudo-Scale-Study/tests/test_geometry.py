"""Geometry regression checks use generated data and inference only, never train."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from fsa_study.data import CODDataset, audit_geometry, geometry_protocol, prepare_manifest, sha256_file
from fsa_study.engine import _model_config_digest, evaluate
from fsa_study.metrics import METRIC_KEYS
from run_study import build_parser
from test_data import fixture


@pytest.fixture
def geometry_data(tmp_path):
    root = tmp_path / "data"
    fixture(root, tests=("chameleon",))
    image = root / "CHAMELEON/Imgs/chameleon_test_0.png"
    mask = root / "CHAMELEON/GT/chameleon_test_0.png"
    Image.new("RGB", (1000, 628), (21, 42, 63)).save(image)
    pixels = np.zeros((628, 998), dtype=np.uint8)
    pixels[150:450, 240:740] = 255
    Image.fromarray(pixels).save(mask)
    path = tmp_path / "manifest.json"
    manifest = prepare_manifest(root, path, allow_nonstandard_counts=True)
    return root, path, manifest, image, mask


def test_default_strict_rejects_actual_reported_dimensions(geometry_data):
    root, _, manifest, _, _ = geometry_data
    with pytest.raises(ValueError, match=r"\(1000, 628\) != \(998, 628\)"):
        CODDataset(manifest, root, "test")[0]
    with pytest.raises(ValueError, match="1 mismatched test pairs"):
        audit_geometry(manifest, root)


@pytest.mark.parametrize("split", ["train", "val"])
def test_opt_in_cannot_relax_training_or_validation(geometry_data, split):
    root, _, manifest, _, _ = geometry_data
    with pytest.raises(ValueError, match="only for test"):
        CODDataset(manifest, root, split, geometry_policy="resize-prediction-to-gt")
    with pytest.raises(ValueError, match="only for test"):
        audit_geometry(manifest, root, split, "resize-prediction-to-gt")
    record = next(r for r in manifest["records"] if r["split"] == split)
    Image.new("RGB", (8, 4), (99, 123, 144)).save(root / record["image"])
    with pytest.raises(ValueError, match="geometry mismatch"):
        CODDataset(manifest, root, split)[0]


def test_audit_includes_all_selected_pairs_and_preserves_bytes(geometry_data):
    root, path, manifest, image, mask = geometry_data
    before = {p: sha256_file(p) for p in (path, image, mask)}
    report = audit_geometry(manifest, root, policy="resize-prediction-to-gt")
    assert report["dataset_counts"] == {"chameleon": 2}
    assert report["checked_pairs"] == 2 and report["mismatch_count"] == 1
    assert report["mismatches"][0]["image_size"] == [1000, 628]
    assert report["mismatches"][0]["mask_size"] == [998, 628]
    assert {p: sha256_file(p) for p in before} == before
    assert report["protocol"] == geometry_protocol("resize-prediction-to-gt")


def test_strict_preflight_fails_before_checkpoint_load_or_output(geometry_data, tmp_path):
    _, path, _, _, _ = geometry_data
    output = tmp_path / "strict-eval"
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(path), "--checkpoint", str(tmp_path / "missing.pth"),
        "--split", "test", "--output", str(output), "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="geometry mismatch"):
        evaluate(args)
    assert not output.exists()


class TinyPredictor(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer("template", torch.linspace(-4, 4, 64 * 64).reshape(1, 1, 64, 64))

    def forward(self, image):
        return {"logits": self.template.expand(image.shape[0], -1, -1, -1)}


def test_old_checkpoint_evaluation_saves_original_gt_size_and_all_metrics(geometry_data, tmp_path, monkeypatch):
    root, path, manifest, image, mask = geometry_data
    from fsa_study import model as model_module
    monkeypatch.setattr(model_module, "StudyModel", TinyPredictor)
    config = {"seed": 3407}
    checkpoint = tmp_path / "old-best.pth"
    torch.save({"schema_version": 1, "model_config": config,
                "model_config_sha256": _model_config_digest(config), "model": TinyPredictor().state_dict(),
                "manifest_sha256": manifest["content_sha256"], "run_config": {"size": 64}, "epoch": 150}, checkpoint)
    before = {p: sha256_file(p) for p in (path, image, mask, checkpoint)}
    partial = tmp_path / "old-test"
    partial.mkdir()
    (partial / "keep.txt").write_text("failed evaluation retained")
    output = tmp_path / "test-geometry-v1"
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(path), "--checkpoint", str(checkpoint),
        "--split", "test", "--output", str(output), "--device", "cpu", "--workers", "0",
        "--test-geometry", "resize-prediction-to-gt",
    ])
    evaluate(args)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["epoch"] == 150
    values = metrics["datasets"]["chameleon"]
    assert values["count"] == 2 and set(METRIC_KEYS).issubset(values)
    assert all(np.isfinite(values[key]) for key in METRIC_KEYS)
    sizes = []
    for saved in (output / "predictions").rglob("*.png"):
        with Image.open(saved) as handle:
            sizes.append(handle.size)
    assert sorted(sizes) == [(4, 4), (998, 628)]
    assert metrics["geometry_audit"]["mismatch_count"] == 1
    assert metrics["artifacts"]["sha256"]["geometry_audit.json"] == sha256_file(output / "geometry_audit.json")
    assert {p: sha256_file(p) for p in before} == before
    assert (partial / "keep.txt").read_text() == "failed evaluation retained"


def test_opt_in_leaves_matching_pair_tensors_unchanged(geometry_data):
    root, _, manifest, _, _ = geometry_data
    strict = CODDataset(manifest, root, "test", size=64)[1]
    allowed = CODDataset(manifest, root, "test", size=64, geometry_policy="resize-prediction-to-gt")[1]
    assert torch.equal(strict["image"], allowed["image"])
    assert strict["mask_path"] == allowed["mask_path"]
