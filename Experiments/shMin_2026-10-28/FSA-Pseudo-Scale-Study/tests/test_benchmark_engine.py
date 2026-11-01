"""Inference-only integration and pure checkpoint policy checks; never optimize."""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from fsa_study import benchmark
from fsa_study.benchmark import extend_test_manifest
from fsa_study.data import manifest_digest, prepare_manifest, sha256_file
from fsa_study.engine import (checkpoint_model_kwargs, _model_config_digest,
                              _select_checkpoint, _validation_due, evaluate)
from fsa_study.metrics import METRIC_KEYS
from run_study import build_parser
from test_data import fixture
from test_geometry import TinyPredictor


@pytest.fixture(autouse=True)
def restore_evaluation_random_state():
    """Evaluation seeds a process; isolate its settings from later unit tests."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


@pytest.fixture
def extension_data(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "EXPECTED_TEST_COUNTS", {name: 2 for name in benchmark.EXPECTED_TEST_COUNTS})
    root = tmp_path / "data"
    fixture(root, tests=("camo", "cod10k", "chameleon", "nc4k"))
    original_path = tmp_path / "training.json"
    original = prepare_manifest(root, original_path, allow_nonstandard_counts=True,
                                test_datasets=["camo", "chameleon", "nc4k"])
    extended_path = tmp_path / "extended.json"
    extended = extend_test_manifest(original_path, extended_path)
    config = {"seed": 3407, "architecture": "common"}
    checkpoint = tmp_path / "best.pth"
    torch.save({"schema_version": 1, "model_config": config,
                "model_config_sha256": _model_config_digest(config), "model": TinyPredictor().state_dict(),
                "manifest_sha256": original["content_sha256"], "run_config": {"size": 64}, "epoch": 150}, checkpoint)
    from fsa_study import model
    monkeypatch.setattr(model, "StudyModel", TinyPredictor)
    return original_path, extended_path, checkpoint, original, extended


def _arguments(data, output, *extra):
    original, extended, checkpoint, _, _ = data
    return build_parser().parse_args([
        "evaluate", "--manifest", str(extended), "--training-manifest", str(original),
        "--checkpoint", str(checkpoint), "--output", str(output), "--split", "test",
        "--device", "cpu", "--workers", "0", "--test-geometry", "resize-prediction-to-gt", *extra,
    ])


def test_old_checkpoint_evaluates_all_four_without_modifying_training(extension_data, tmp_path):
    original, extended, checkpoint, before_manifest, _ = extension_data
    hashes = {path: sha256_file(path) for path in (original, extended, checkpoint)}
    output = tmp_path / "new-test"
    evaluate(_arguments(extension_data, output))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["benchmark_coverage"]["complete"]
    assert metrics["overlap_audit"]["policy"] == "strict"
    assert metrics["overlap_audit"]["disjoint"]
    assert metrics["overlap_audit"]["affected_test_count"] == 0
    assert set(metrics["datasets"]) == {"camo", "cod10k", "chameleon", "nc4k"}
    assert all(set(METRIC_KEYS).issubset(values) and values["count"] == 2
               for values in metrics["datasets"].values())
    assert metrics["training_manifest_sha256"] == before_manifest["content_sha256"]
    assert metrics["manifest_sha256"] != metrics["training_manifest_sha256"]
    assert metrics["model_config"]["architecture"] == "common"
    for name in ("manifest.json", "training_manifest.json"):
        assert metrics["artifacts"]["sha256"][name] == sha256_file(output / name)
    assert {path: sha256_file(path) for path in hashes} == hashes


def test_acknowledged_inference_records_overlap_without_dropping_scores(extension_data, tmp_path, monkeypatch):
    from fsa_study import overlap
    original_path, _, checkpoint, original, extended = extension_data
    root = Path(original["data_root"])
    training = next(r for r in original["records"] if r["source"] == "cod10k_train" and r["split"] == "train")
    testing = next(r for r in extended["records"] if r["source"] == "cod10k_test")
    (root / testing["image"]).write_bytes((root / training["image"]).read_bytes())
    # Tiny synthetic PNGs stand in for the exact server-evidence pair. The
    # production acknowledgement list is never changed on disk.
    monkeypatch.setattr(overlap, "KNOWN_COD10K_OVERLAPS", ((Path(training["image"]).stem,
                        Path(testing["image"]).stem, training["image_sha256"]),))
    acknowledged_path = tmp_path / "acknowledged.json"
    acknowledged = extend_test_manifest(original_path, acknowledged_path, acknowledge_cod10k=True)
    before = {path: sha256_file(path) for path in (original_path, checkpoint, acknowledged_path)}
    data = original_path, acknowledged_path, checkpoint, original, acknowledged
    output = tmp_path / "acknowledged-test"
    evaluate(_arguments(data, output))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["overlap_audit"] == overlap.overlap_metadata(acknowledged)
    assert not metrics["overlap_audit"]["disjoint"]
    assert metrics["overlap_audit"]["train_test_count"] == 1
    assert metrics["overlap_audit"]["val_test_count"] == 0
    assert metrics["benchmark_coverage"]["complete"]
    assert all(set(METRIC_KEYS).issubset(values) and values["count"] == 2
               for values in metrics["datasets"].values())
    assert {path: sha256_file(path) for path in before} == before


def test_extension_without_original_manifest_is_rejected(extension_data, tmp_path):
    args = _arguments(extension_data, tmp_path / "bad-test")
    args.training_manifest = None
    with pytest.raises(ValueError, match="different dataset manifest"):
        evaluate(args)
    assert not (tmp_path / "bad-test").exists()


def test_changed_training_record_rejected_before_model_or_output(extension_data, tmp_path):
    _, path, _, _, manifest = extension_data
    changed = copy.deepcopy(manifest)
    record = next(r for r in changed["records"] if r["split"] == "train")
    record["id"] += "-changed"
    changed["content_sha256"] = manifest_digest(changed)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="original records"):
        evaluate(_arguments(extension_data, tmp_path / "bad-test"))
    assert not (tmp_path / "bad-test").exists()


def test_extension_cannot_be_used_for_validation(extension_data, tmp_path):
    args = _arguments(extension_data, tmp_path / "bad-val", "--split", "val")
    with pytest.raises(ValueError, match="only for test"):
        evaluate(args)


def test_full_training_never_validates_or_selects_an_early_epoch():
    for epoch in range(1, 151):
        assert not _validation_due("full_train", epoch, 150, 1)
        for improved in (True, False):
            assert _select_checkpoint("full_train", epoch, 150, improved) == (epoch == 150)
    assert _validation_due("holdout", 150, 150, 7)
    assert not _validation_due("holdout", 2, 150, 7)
    assert _select_checkpoint("holdout", 2, 150, True)
    assert not _select_checkpoint("holdout", 150, 150, False)


def test_common_architecture_tag_does_not_change_saved_metadata():
    class Common:
        def __init__(self, channels):
            pass
    class Explicit:
        def __init__(self, channels, architecture):
            pass
    config = {"architecture": "common", "channels": 48}
    assert checkpoint_model_kwargs(config, Common) == {"channels": 48}
    assert checkpoint_model_kwargs(config, Explicit) == config
    assert config == {"architecture": "common", "channels": 48}
    with pytest.raises(ValueError, match="original architecture"):
        checkpoint_model_kwargs({**config, "architecture": "pcd"}, Common)


def test_prepare_cli_requires_four_tests_by_default():
    args = build_parser().parse_args(["prepare", "--data-root", "/data", "--manifest", "/new.json"])
    assert set(args.test_datasets) == {"camo", "chameleon", "cod10k", "nc4k"}
    assert args.training_protocol == "holdout"


def test_priority8_plan_full_training_commands_do_not_execute(tmp_path):
    from test_workflow import _run
    plan_dir = tmp_path / "plan"
    result = _run("plan_study.py", "--phase", "priority8", "--seeds", "3407",
                  "--training-protocol", "full_train", "--manifest", tmp_path / "full.json",
                  "--output", plan_dir, "--results", tmp_path / "runs", "--random-init")
    assert result.returncode == 0, result.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["run_count"] == 8 and plan["jobs"][0]["variant"] == "single"
    assert not (tmp_path / "runs").exists()
    for job in plan["jobs"]:
        args = build_parser().parse_args(job["train"][2:])
        assert args.training_protocol == "full_train"
