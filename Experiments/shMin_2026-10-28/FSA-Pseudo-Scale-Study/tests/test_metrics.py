"""Metric checks use analytic masks/official implementations, never optimizer steps."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa_study.engine import evaluate_model
from fsa_study.metrics import (BINARY_KEYS, METRIC_KEYS, METRIC_PROTOCOL, MetricRecorder,
                               binary_metrics, boundary_mask, image_metrics, protocol_details)


def _pairs(count=3):
    rng = np.random.default_rng(812)
    return [(rng.integers(15, 220, (27, 35), dtype=np.uint8),
             (rng.random((27, 35)) > .7).astype(np.uint8) * 255) for _ in range(count)]


def test_six_original_scores_are_bit_exact_with_official_dataset_accumulation():
    from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure
    reference = {"S": Smeasure(), "wF": WeightedFmeasure(), "MAE": MAE(), "E": Emeasure()}
    recorder = MetricRecorder()
    for i, (prediction, target) in enumerate(_pairs()):
        for metric in reference.values():
            metric.step(prediction, target)
        recorder.step(prediction, target, "cod10k", str(i))
    scores = recorder.finish()[0]["cod10k"]
    e = reference["E"].get_results()["em"]
    expected = {"S": reference["S"].get_results()["sm"], "wF": reference["wF"].get_results()["wfm"],
                "MAE": reference["MAE"].get_results()["mae"], "E_adaptive": e["adp"],
                "E_mean": np.mean(e["curve"]), "E_max": np.max(e["curve"])}
    assert {key: scores[key] for key in expected} == expected


def test_fbeta_v2_matches_original_official_fmeasure_and_curves():
    from py_sod_metrics import Fmeasure
    with pytest.warns(UserWarning, match="removed"):
        reference = Fmeasure(beta=.3)
    recorder = MetricRecorder()
    for i, (prediction, target) in enumerate(_pairs()):
        reference.step(prediction, target)
        recorder.step(prediction, target, "cod10k", str(i))
    scores = recorder.finish()[0]["cod10k"]
    expected = reference.get_results()
    assert scores["Fbeta_adaptive"] == expected["fm"]["adp"]
    assert scores["Fbeta_mean"] == np.mean(expected["fm"]["curve"])
    assert scores["Fbeta_max"] == np.max(expected["fm"]["curve"])
    np.testing.assert_array_equal(np.mean([r["precision"] for r in recorder.curves["cod10k"]], axis=0),
                                  expected["pr"]["p"][::-1])
    np.testing.assert_array_equal(np.mean([r["recall"] for r in recorder.curves["cod10k"]], axis=0),
                                  expected["pr"]["r"][::-1])


def test_dataset_max_is_max_of_averaged_curve_not_average_of_image_maxima():
    recorder = MetricRecorder()
    for i, (prediction, target) in enumerate(_pairs(8)):
        recorder.step(prediction, target, "cod10k", str(i))
    scores = recorder.finish()[0]["cod10k"]
    for metric in ("E", "Fbeta"):
        mean_curve = np.mean([row[metric] for row in recorder.curves["cod10k"]], axis=0)
        assert scores[f"{metric}_max"] == np.max(mean_curve)
        assert scores[f"{metric}_max"] < np.mean([row[f"{metric}_max"] for row in recorder.samples["cod10k"]])


@pytest.mark.parametrize("shape", [(8, 10), (100, 200), (513, 711)])
@pytest.mark.parametrize("kind", ["empty", "full", "random", "border"])
def test_boundary_geometry_matches_original_author_code(shape, kind):
    rng = np.random.default_rng(27)
    mask = {"empty": np.zeros(shape, np.uint8), "full": np.ones(shape, np.uint8),
            "random": rng.integers(0, 2, shape, dtype=np.uint8),
            "border": np.pad(np.ones((shape[0], shape[1] // 2), np.uint8),
                             ((0, 0), (0, shape[1] - shape[1] // 2)))}[kind]
    radius = max(1, int(round(.02 * np.sqrt(shape[0] ** 2 + shape[1] ** 2))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=radius)[1:shape[0] + 1, 1:shape[1] + 1]
    np.testing.assert_array_equal(boundary_mask(mask), (mask - eroded).astype(bool))


def test_binary_perfect_inverted_disjoint_and_empty_conventions():
    target = np.zeros((20, 24), dtype=bool)
    target[3:9, 4:12] = True
    perfect = binary_metrics(target, target)
    for key in BINARY_KEYS:
        assert perfect[key] == (0. if key in {"BER_0_5", "FPR_0_5", "FNR_0_5"} else 1.)
    inverted = binary_metrics(~target, target)
    for key in BINARY_KEYS:
        assert inverted[key] == (1. if key in {"BER_0_5", "FPR_0_5", "FNR_0_5"} else 0.)
    shifted = np.roll(target, 12, axis=1)
    disjoint = binary_metrics(shifted, target)
    assert disjoint["IoU_0_5"] == disjoint["Dice_0_5"] == disjoint["BoundaryIoU_0_5"] == 0
    empty = np.zeros_like(target)
    assert binary_metrics(empty, empty) == perfect
    assert binary_metrics(empty, target)["Precision_0_5"] == 0
    assert binary_metrics(empty, target)["Recall_0_5"] == 0
    assert binary_metrics(target, empty)["Recall_0_5"] == 1
    assert binary_metrics(np.ones_like(target), np.ones_like(target))["Specificity_0_5"] == 1


def test_analytic_partial_overlap_binary_metrics():
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    scores = binary_metrics(prediction, target)
    assert scores["IoU_0_5"] == pytest.approx(1 / 3)
    assert all(scores[key] == .5 for key in BINARY_KEYS if key not in {"IoU_0_5", "BoundaryIoU_0_5"})


def test_threshold_curves_ascending_and_png_reproduces_scores(tmp_path):
    prediction, target = _pairs(1)[0]
    recorder = MetricRecorder(tmp_path)
    recorder.step(prediction, target, "cod10k", "source/image", "original-mask.png")
    scores, artifacts = recorder.finish()
    rows = list(csv.DictReader((tmp_path / artifacts["per_image_metrics"]).open()))
    assert len(rows) == 1 and rows[0]["id"] == "source/image"
    with Image.open(tmp_path / rows[0]["prediction_path"]) as image:
        saved = np.asarray(image)
        assert image.size == (target.shape[1], target.shape[0])
    np.testing.assert_array_equal(saved, prediction)
    recomputed, _ = image_metrics(saved, target)
    assert {key: scores["cod10k"][key] for key in METRIC_KEYS} == recomputed
    curve_files = artifacts["curves"]["cod10k"]
    curve = list(csv.DictReader((tmp_path / curve_files["csv"]).open()))
    assert [int(row["threshold_uint8"]) for row in curve] == list(range(256))
    assert float(curve[0]["threshold_0_1"]) == 0 and float(curve[-1]["threshold_0_1"]) == 1
    assert float(curve[0]["recall"]) == 1  # threshold zero selects the entire image
    for kind in ("pr", "fbeta", "emeasure"):
        root = ET.parse(tmp_path / curve_files[kind]["svg"]).getroot()
        line = root.find("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}polyline")
        assert len(line.attrib["points"].split()) == 256
        with Image.open(tmp_path / curve_files[kind]["png"]) as plot:
            assert plot.size == (720, 520)
    for name, digest in artifacts["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


def test_artifact_names_are_collision_safe_and_do_not_escape_or_overwrite(tmp_path):
    prediction, target = _pairs(1)[0]
    recorder = MetricRecorder(tmp_path)
    for sample in ("first/image", "second/image", "../../image", "a b", "a_b"):
        recorder.step(prediction, target, "../cod10k", sample)
    _, artifacts = recorder.finish()
    paths = [row["prediction_path"] for row in recorder.rows]
    assert len(set(paths)) == 5
    assert all((tmp_path / path).resolve().is_relative_to(tmp_path) for path in paths)
    before = {path: (tmp_path / path).read_bytes() for path in artifacts["sha256"]}
    with pytest.raises(ValueError, match="Duplicate evaluation sample ID"):
        recorder.step(prediction, target, "../cod10k", "first/image")
    another = MetricRecorder(tmp_path)
    with pytest.raises(FileExistsError):
        another.step(prediction, target, "../cod10k", "first/image")
    assert all((tmp_path / path).read_bytes() == content for path, content in before.items())


def test_no_prediction_option_still_exports_all_scores_and_curves(tmp_path):
    prediction, target = _pairs(1)[0]
    recorder = MetricRecorder(tmp_path, save_predictions=False)
    recorder.step(prediction, target, "cod10k", "sample")
    scores, artifacts = recorder.finish()
    assert set(scores["cod10k"]) == {"count", *METRIC_KEYS}
    assert artifacts["predictions_saved"] is False and not (tmp_path / "predictions").exists()
    assert (tmp_path / artifacts["per_image_metrics"]).exists()
    assert recorder.rows[0]["prediction_path"] == ""


class _ToyModel(torch.nn.Module):
    def forward(self, image):
        return {"logits": image[:, :1]}


def test_evaluation_exports_metrics_and_preserves_validation_selection(tmp_path):
    target = np.zeros((18, 21), dtype=np.uint8)
    target[2:12, 4:17] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(target).save(mask_path)
    batch = {"image": torch.tensor(target > 0, dtype=torch.float32)[None, None] * 40 - 20,
             "mask_path": [str(mask_path)], "dataset": ["cod10k"], "id": ["cod10k/test"]}
    output = tmp_path / "evaluation"
    result = evaluate_model(_ToyModel(), [batch], "cpu", official=True, output=output)
    assert result["metric_protocol"] == METRIC_PROTOCOL
    assert result["metric_protocol_details"]["schema_version"] == 2
    assert (output / "per_image_metrics.csv").exists()
    validation_output = tmp_path / "validation"
    validation = evaluate_model(_ToyModel(), [batch], "cpu", output=validation_output)
    assert validation["metric_protocol"] == "float_MAE"
    assert set(validation["datasets"]["cod10k"]) == {"count", "MAE"}
    assert not validation_output.exists()
    assert result["sample_weighted_MAE"] == validation["sample_weighted_MAE"] == 0


def test_protocol_records_threshold_empty_masks_papers_and_library_version():
    details = protocol_details()
    assert details["binary_threshold"] == {"value": .5, "comparison": ">=", "input": "PySODMetrics prepared float64 map"}
    assert details["Fbeta"]["beta_squared"] == .3
    assert details["boundary"]["dilation_ratio"] == .02
    assert details["pysodmetrics_version"]
    assert "Boundary_IoU_author_code" in details["sources"]
    assert "binary_empty_masks" in details


def test_evaluate_cli_accepts_previous_checkpoint_schema_and_exports_all_artifacts(tmp_path):
    """Random model inference only: an old schema-1 checkpoint needs no new fields."""
    import json
    import os
    import subprocess
    from fsa_study.data import SOURCES, prepare_manifest, sha256_file
    from fsa_study.engine import _model_config_digest
    from fsa_study.model import StudyModel

    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    rng = np.random.default_rng(901)
    for source in ("cod10k_train", "camo_train", "cod10k_test"):
        dataset, split, image_rel, mask_rel, _ = SOURCES[source]
        for index in range(2 if split == "train" else 1):
            stem = ("COD10K-CAM-" if dataset == "cod10k" else "") + f"{source}-{index}"
            image_path, mask_path = data_root / image_rel / f"{stem}.png", data_root / mask_rel / f"{stem}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (27, 35, 3), dtype=np.uint8)).save(image_path)
            target = np.zeros((27, 35), dtype=np.uint8)
            target[3:19, 8:26] = 255
            Image.fromarray(target).save(mask_path)
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest = prepare_manifest(data_root, manifest_path, allow_nonstandard_counts=True, test_datasets=["cod10k"])
    model_config = {"generator": "legacy", "utilizer": "mean", "channels": 8, "seed": 19,
                    "baseline": True, "legacy_reference": False, "generator_options": {}, "utilizer_options": {}}
    model = StudyModel(**model_config)
    checkpoint_path = tmp_path / "synthetic-untrained-schema1.pth"
    # Deliberately no metric protocol, artifact settings, optimizer or new schema fields.
    torch.save({"schema_version": 1, "model": model.state_dict(), "model_config": model_config,
                "model_config_sha256": _model_config_digest(model_config),
                "manifest_sha256": manifest["content_sha256"], "epoch": 0,
                "run_config": {"size": 64, "determinism": "warn"}}, checkpoint_path)
    original_checkpoint_hash = sha256_file(checkpoint_path)
    del model
    output = tmp_path / "evaluation"
    process = subprocess.run([sys.executable, str(ROOT / "run_study.py"), "evaluate",
                              "--manifest", str(manifest_path), "--checkpoint", str(checkpoint_path),
                              "--split", "test", "--device", "cpu", "--batch-size", "1", "--workers", "0",
                              "--output", str(output)], capture_output=True, text=True, timeout=90,
                             env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "metrics.json").read_text())
    assert report["metric_protocol"] == METRIC_PROTOCOL
    assert report["datasets"]["cod10k"]["count"] == 1
    assert set(report["datasets"]["cod10k"]) == {"count", *METRIC_KEYS}
    assert len(METRIC_KEYS) == 21
    assert report["checkpoint_sha256"] == original_checkpoint_hash == sha256_file(checkpoint_path)
    assert report["epoch"] == 0
    artifacts = report["artifacts"]
    assert (output / artifacts["per_image_metrics"]).is_file()
    for kind in ("pr", "fbeta", "emeasure"):
        for extension in ("png", "svg"):
            assert (output / artifacts["curves"]["cod10k"][kind][extension]).is_file()
    rows = list(csv.DictReader((output / artifacts["per_image_metrics"]).open()))
    assert len(rows) == 1
    with Image.open(output / rows[0]["prediction_path"]) as prediction:
        assert prediction.size == (35, 27)  # Original GT resolution, not model resolution.
    assert not (output / "status.json").exists()  # No claim of training completion.
    assert not (output / "last.pth").exists()
