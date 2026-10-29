"""Workflow tests use synthetic artifacts and inference only; never optimize."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsa_study.data import sha256_file
from fsa_study.engine import evaluate_model
from summarize_study import aggregate


def _run(script, *arguments):
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                          capture_output=True, text=True, timeout=60)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fake_run(root, name, seed=1, manifest="data", metric_protocol="official_py_sod_metrics_uint8", complete=True):
    run = root / name
    model = {"baseline": False, "legacy_reference": False, "generator": "legacy", "utilizer": "mean",
             "channels": 48, "seed": seed, "generator_options": {}, "utilizer_options": {}}
    config = {"model_config": model, "manifest_sha256": manifest, "backbone_sha256": "weights",
              "size": 384, "epochs": 150, "lr": .0001, "batch_size": 4,
              "eval_interval": 1, "determinism": "warn", "source_sha256": {"code.py": "stable"},
              "loss_weights": {"seg": 1}, "augmentation": "same", "edge_protocol": "same",
              "environment": {"packages": {"torch": "same"}}}
    _write(run / "config.json", config)
    if not complete:
        return run
    _write(run / "status.json", {"status": "training_complete", "best_val_mae": .1 * seed})
    # These bytes are intentionally not a model; the summarizer only verifies artifact identity.
    (run / "best.pth").write_bytes(f"synthetic-selected-checkpoint-{seed}".encode())
    metrics = {"split": "test", "manifest_sha256": manifest, "model_config": model,
               "checkpoint_sha256": sha256_file(run / "best.pth"),
               "metric_protocol": metric_protocol, "prediction_protocol": "same",
               "environment": {"packages": {"pysodmetrics": "same"}},
               "datasets": {"cod10k": {"count": 2, "MAE": .1 * seed, "S": .9}}}
    _write(run / "test/metrics.json", metrics)
    return run


@pytest.mark.parametrize("phase,expected", [("generation", 42), ("utilization", 45), ("cross", 474)])
def test_plan_cli_counts_absolute_commands_and_shell_syntax(tmp_path, phase, expected):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")  # Plan generation intentionally does not touch dataset content.
    output, runs = tmp_path / "plan", tmp_path / "future-runs"
    result = _run("plan_study.py", "--phase", phase, "--manifest", manifest, "--output", output,
                  "--results", runs, "--python", sys.executable, "--random-init", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "plan.json").read_text())
    assert payload["run_count"] == expected
    assert payload["status"] == "commands_only_not_executed"
    assert not runs.exists()
    assert len({job["run"] for job in payload["jobs"]}) == expected
    for job in payload["jobs"]:
        assert Path(job["run"]).is_absolute()
        assert "--random-init" in job["train"]
        assert job["evaluate"][job["evaluate"].index("--split") + 1] == "test"
        assert job["profile"][job["profile"].index("--batch-size") + 1] == "1"
    for action in ("train", "evaluate", "profile"):
        parsed = subprocess.run(["bash", "-n", str(output / f"{action}.sh")], capture_output=True, text=True)
        assert parsed.returncode == 0, parsed.stderr


def test_plan_rejects_relative_destination(tmp_path):
    result = _run("plan_study.py", "--manifest", tmp_path / "m.json", "--output", "relative-plan",
                  "--results", tmp_path / "runs", "--random-init")
    assert result.returncode != 0
    assert "must be an absolute path" in result.stderr


def test_summary_mean_std_and_incomplete_reporting(tmp_path):
    _fake_run(tmp_path, "first", seed=1)
    _fake_run(tmp_path, "second", seed=2)
    incomplete = _fake_run(tmp_path, "incomplete", seed=3, complete=False)
    report = aggregate(tmp_path)
    row = next(row for row in report["rows"] if row["dataset"] == "cod10k" and row["metric"] == "MAE")
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(.15)
    assert row["sample_std"] == pytest.approx(np.std([.1, .2], ddof=1))
    assert report["incomplete_runs"] == [str(incomplete)]


def test_summary_separates_incompatible_data_and_metric_protocols(tmp_path):
    _fake_run(tmp_path, "a", seed=1, manifest="source-a")
    _fake_run(tmp_path, "b", seed=2, manifest="source-b")
    _fake_run(tmp_path, "c", seed=3, manifest="source-a", metric_protocol="other")
    report = aggregate(tmp_path)
    rows = [row for row in report["rows"] if row["dataset"] == "cod10k" and row["metric"] == "MAE"]
    assert len(rows) == 3
    assert all(row["n"] == 1 for row in rows)


def test_summary_duplicate_seed_rejected(tmp_path):
    _fake_run(tmp_path, "a", seed=4)
    _fake_run(tmp_path, "b", seed=4)
    with pytest.raises(ValueError, match="Repeated seed"):
        aggregate(tmp_path)


@pytest.mark.parametrize("corruption,message", [
    ("manifest", "Mismatched test metadata"),
    ("variant", "Mismatched test metadata"),
    ("checkpoint", "selected best checkpoint"),
    ("split", "Expected test split"),
])
def test_summary_corrupted_evaluation_identity_rejected(tmp_path, corruption, message):
    run = _fake_run(tmp_path, "run")
    path = run / "test/metrics.json"
    metrics = json.loads(path.read_text())
    if corruption == "manifest":
        metrics["manifest_sha256"] = "wrong"
    elif corruption == "variant":
        metrics["model_config"]["utilizer"] = "signed"
    elif corruption == "checkpoint":
        metrics["checkpoint_sha256"] = "different-checkpoint"
    else:
        metrics["split"] = "val"
    _write(path, metrics)
    with pytest.raises(ValueError, match=message):
        aggregate(tmp_path)


class ToyPredictor(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("template", logits)

    def forward(self, image):
        return {"logits": self.template.expand(image.shape[0], -1, -1, -1)}


def _toy_batch(tmp_path):
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 255
    paths = []
    for index in range(2):
        path = tmp_path / f"mask_{index}.png"
        Image.fromarray(mask).save(path)
        paths.append(str(path))
    batch = {"image": torch.zeros(2, 3, 8, 10), "mask_path": paths, "dataset": ["cod10k", "camo"]}
    logits = torch.tensor(mask > 0, dtype=torch.float32)[None, None] * 40 - 20
    return batch, logits


def test_official_metrics_perfect_and_inverted_toy_predictions(tmp_path):
    pytest.importorskip("py_sod_metrics")
    batch, logits = _toy_batch(tmp_path)
    perfect = evaluate_model(ToyPredictor(logits), [batch], "cpu", official=True)
    inverted = evaluate_model(ToyPredictor(-logits), [batch], "cpu", official=True)
    assert perfect["metric_protocol"] == "official_py_sod_metrics_uint8_v2"
    assert perfect["sample_weighted_MAE"] == pytest.approx(0, abs=1e-6)
    assert inverted["sample_weighted_MAE"] > .99
    for values in perfect["datasets"].values():
        assert values["count"] == 1
        assert values["S"] > .99 and values["wF"] > .99
        assert all(np.isfinite(value) for value in values.values())
        assert {"E_adaptive", "E_mean", "E_max"}.issubset(values)
    floating = evaluate_model(ToyPredictor(logits), [batch], "cpu", official=False)
    assert floating["sample_weighted_MAE"] == pytest.approx(0, abs=1e-6)


def test_evaluation_rejects_nonfinite_logits(tmp_path):
    batch, logits = _toy_batch(tmp_path)
    with pytest.raises(FloatingPointError, match="Nonfinite evaluation logits"):
        evaluate_model(ToyPredictor(torch.full_like(logits, float("nan"))), [batch], "cpu")


def test_evaluate_and_profile_do_not_overwrite_existing_artifacts(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserved")
    evaluation = _run("run_study.py", "evaluate", "--manifest", tmp_path / "nonexistent-manifest.json",
                      "--checkpoint", tmp_path / "nonexistent.pth", "--split", "test",
                      "--device", "cpu", "--output", existing)
    assert evaluation.returncode != 0 and "Output already exists" in evaluation.stderr
    profile = _run("profile_study.py", "--output", sentinel)
    assert profile.returncode != 0 and "output already exists" in profile.stderr
    assert sentinel.read_text() == "preserved"
    assert list(existing.iterdir()) == [sentinel]


def test_profile_rejects_corrupt_variant_before_output_creation(tmp_path):
    checkpoint = tmp_path / "corrupt.pth"
    torch.save({"schema_version": 1, "model_config": {"generator": "legacy", "utilizer": "mean"},
                "model_config_sha256": "invalid", "manifest_sha256": "data"}, checkpoint)
    output = tmp_path / "profile.json"
    result = _run("profile_study.py", "--checkpoint", checkpoint, "--output", output,
                  "--size", 64, "--device", "cpu", "--warmup", 0, "--repeats", 1)
    assert result.returncode != 0
    assert "variant metadata checksum mismatch" in result.stderr
    assert not output.exists()


def _fake_profile(run, *, device="cpu", threads=2, median=10, cuda_memory=False):
    profile = {
        "checkpoint_sha256": sha256_file(run / "best.pth"), "image_sha256": None,
        "parameters": 1000, "latency_ms": {"median": median},
        "images_per_second": 1000 / median,
        "cuda_memory": {"peak_allocated_bytes": 4096} if cuda_memory else None,
        "environment": {"torch": "same", "python": "same", "device": "synthetic-device"},
        "protocol": {"size": 384, "batch_size": 1, "threads": threads, "warmup": 10,
                     "repeats": 50, "device": device, "dtype": "float32",
                     "input": "synthetic_uniform_RGB", "seed": 3407,
                     "timed_scope": "all heads, no IO"},
    }
    _write(run / "profile.json", profile)


@pytest.mark.parametrize("cuda_memory", [False, True])
def test_summary_collects_matching_profile_costs_and_optional_cuda_memory(tmp_path, cuda_memory):
    a, b = _fake_run(tmp_path, "a", seed=1), _fake_run(tmp_path, "b", seed=2)
    _fake_profile(a, median=10, cuda_memory=cuda_memory)
    _fake_profile(b, median=20, cuda_memory=cuda_memory)
    rows = [row for row in aggregate(tmp_path)["rows"] if row["dataset"] == "inference"]
    values = {row["metric"]: row for row in rows}
    assert len(values) == (4 if cuda_memory else 3)
    assert all(row["n"] == 2 for row in rows)
    assert values["parameters"]["mean"] == 1000
    assert values["latency_median_ms"]["mean"] == 15
    assert values["images_per_second"]["mean"] == 75
    if cuda_memory:
        assert values["cuda_peak_allocated_bytes"]["mean"] == 4096
    else:
        assert "cuda_peak_allocated_bytes" not in values


def test_summary_rejects_profile_from_different_checkpoint(tmp_path):
    run = _fake_run(tmp_path, "run")
    _fake_profile(run)
    path = run / "profile.json"
    profile = json.loads(path.read_text())
    profile["checkpoint_sha256"] = "different-checkpoint"
    _write(path, profile)
    with pytest.raises(ValueError, match="Profile did not use this run's selected best checkpoint"):
        aggregate(tmp_path)


@pytest.mark.parametrize("difference", [{"device": "cuda:0"}, {"threads": 8}])
def test_summary_separates_profile_device_and_thread_conditions(tmp_path, difference):
    a, b = _fake_run(tmp_path, "a", seed=1), _fake_run(tmp_path, "b", seed=2)
    _fake_profile(a)
    _fake_profile(b, **difference)
    rows = [row for row in aggregate(tmp_path)["rows"]
            if row["dataset"] == "inference" and row["metric"] == "latency_median_ms"]
    assert len(rows) == 2
    assert all(row["n"] == 1 for row in rows)
    field = next(iter(difference))
    assert len({row["protocol"]["profile"][field] for row in rows}) == 2
