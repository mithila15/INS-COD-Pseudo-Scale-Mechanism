"""Synthetic report artifacts only: no dataset downloads or optimizer steps."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa_study.data import manifest_digest, prepare_manifest, sha256_file
from fsa_study.benchmark import EXPECTED_TEST_COUNTS, extend_test_manifest, validate_test_extension
from fsa_study.leaderboard import build_leaderboard, write_leaderboard
from summarize_study import aggregate


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _run(root, variant, seed, *, value=.1, test=None, protocol="v2", data="manifest", subdir="test", options=None):
    run = root / f"{variant}-{seed}"
    model = {"baseline": variant == "single", "legacy_reference": variant == "legacy_full",
             "generator": "legacy", "utilizer": variant if variant not in {"single", "legacy_full"} else "mean",
             "channels": 48, "seed": seed, "generator_options": options or {}, "utilizer_options": {}}
    config = {"model_config": model, "manifest_sha256": data, "backbone_sha256": "weights", "epochs": 150,
              "size": 384, "lr": .0001, "batch_size": 4, "source_sha256": {"engine.py": "same"},
              "environment": {"packages": {"torch": "same"}}}
    _write(run / "config.json", config)
    _write(run / "status.json", {"status": "training_complete", "best_val_mae": value})
    (run / "best.pth").write_bytes(f"synthetic:{variant}:{seed}".encode())
    if test is not None:
        _write(run / subdir / "metrics.json", {"split": "test", "manifest_sha256": data, "model_config": model,
                                               "checkpoint_sha256": sha256_file(run / "best.pth"),
                                               "metric_protocol": protocol, "metric_protocol_details": {"threshold": .5},
                                               "prediction_protocol": "same", "datasets": {"cod10k": {"count": 2, **test}}})
    return run


def _plan(root, variants, seeds):
    return {"phase": "synthetic", "run_count": len(variants) * len(seeds), "variant_count": len(variants),
            "seeds": seeds, "jobs": [{"variant": variant if variant in {"single", "legacy_full"} else f"legacy__{variant}",
                                      "seed": seed, "run": str(root / f"{variant}-{seed}")}
                                     for variant in variants for seed in seeds]}


def _boards(report, dataset="validation", metric="MAE"):
    return [board for board in report["boards"] if board["dataset"] == dataset and board["metric"] == metric]


def test_three_seed_means_ranks_compare_control_and_variant(tmp_path):
    for variant, values in {"single": [.2, .3, .4], "mean": [.1, .2, .3], "legacy_full": [.3, .4, .5]}.items():
        for seed, value in enumerate(values, 1):
            _run(tmp_path, variant, seed, value=value, test={"S": 1 - value, "BER_0_5": value})
    result = build_leaderboard(aggregate(tmp_path, plan=_plan(tmp_path, ["single", "mean", "legacy_full"], [1, 2, 3])))
    board, = _boards(result)
    assert [row["variant"] for row in board["rows"]] == ["legacy__mean", "single", "legacy_full"]
    assert [row["rank"] for row in board["rows"]] == [1, 2, 3]
    assert board["rows"][0]["mean"] == pytest.approx(.2)
    assert board["rows"][0]["sample_std"] == pytest.approx(.1)
    assert all(row["completion"] == "planned_seeds_complete" for row in board["rows"])
    assert _boards(result, "cod10k", "S")[0]["rows"][0]["variant"] == "legacy__mean"
    assert _boards(result, "cod10k", "BER_0_5")[0]["rows"][0]["variant"] == "legacy__mean"
    assert result["progress"]["training_complete"] == 9
    assert result["progress"]["evaluation_complete"] == 9
    assert result["progress"]["profile_complete"] == 0


@pytest.mark.parametrize("metric", ["BER_0_5", "FPR_0_5", "FNR_0_5", "MAE"])
def test_error_metrics_rank_lower_with_stable_competition_ties(tmp_path, metric):
    for variant, value in [("signed", .2), ("mean", .1), ("single", .1)]:
        _run(tmp_path, variant, 1, test={metric: value})
    board, = _boards(build_leaderboard(aggregate(tmp_path)), "cod10k", metric)
    assert [row["variant"] for row in board["rows"]] == ["legacy__mean", "single", "legacy__signed"]
    assert [row["rank"] for row in board["rows"]] == [1, 1, 3]
    assert all(row["completion"] == "provisional_single_seed" for row in board["rows"])


def test_different_seed_sets_or_training_protocol_never_share_rank(tmp_path):
    for seed in [1, 2, 3]:
        _run(tmp_path, "mean", seed)
    for seed in [1, 2]:
        _run(tmp_path, "signed", seed)
    for seed in [1, 2, 3]:
        _run(tmp_path, "single", seed, data="different-data")
    result = build_leaderboard(aggregate(tmp_path, plan=_plan(tmp_path, ["mean", "signed", "single"], [1, 2, 3])))
    boards = _boards(result)
    assert len(boards) == 3
    assert all(len(board["rows"]) == 1 for board in boards)
    partial = next(board["rows"][0] for board in boards if board["rows"][0]["variant"] == "legacy__signed")
    assert partial["completion"] == "provisional_missing_seeds" and partial["missing_seeds"] == [3]
    assert result["progress"]["not_started"] == 1


def test_generation_options_define_variant_not_comparison_protocol(tmp_path):
    _run(tmp_path / "a", "mean", 1, options={"scale": 1.5})
    _run(tmp_path / "b", "mean", 1, options={"scale": 2})
    board, = _boards(build_leaderboard(aggregate(tmp_path)))
    assert len(board["rows"]) == 2
    assert len({row["variant_label"] for row in board["rows"]}) == 2
    assert "generator_options" not in board["protocol"]


def test_metric_versions_and_per_dataset_counts_remain_separate(tmp_path):
    _run(tmp_path, "mean", 1, test={"S": .8}, protocol="v1")
    _run(tmp_path, "signed", 1, test={"S": .9}, protocol="v2")
    run = _run(tmp_path, "single", 1, test={"S": .9}, protocol="v2")
    path = run / "test/metrics.json"
    payload = json.loads(path.read_text())
    payload["datasets"]["cod10k"]["count"] = 3
    _write(path, payload)
    report = build_leaderboard(aggregate(tmp_path))
    assert len(_boards(report, "cod10k", "S")) == 3
    assert len(_boards(report)) == 1


def test_custom_evaluation_subdirectory_and_missing_metric(tmp_path):
    _run(tmp_path, "mean", 1, test={"Fbeta_max": .8}, subdir="retest-v2")
    assert not _boards(build_leaderboard(aggregate(tmp_path)), "cod10k", "Fbeta_max")
    report = build_leaderboard(aggregate(tmp_path, "retest-v2"), test_metric="Fbeta_max")
    assert len(_boards(report, "cod10k", "Fbeta_max")) == 1
    assert report["evaluation_subdir"] == "retest-v2"
    for invalid in ["../escape", "/absolute", ""]:
        with pytest.raises(ValueError, match="relative path"):
            aggregate(tmp_path, invalid)


@pytest.mark.parametrize("field", ["variant", "seed"])
def test_plan_identity_must_match_actual_run(tmp_path, field):
    _run(tmp_path, "mean", 1)
    plan = _plan(tmp_path, ["mean"], [1])
    plan["jobs"][0][field] = "wrong" if field == "variant" else 9
    with pytest.raises(ValueError, match="Mismatched plan"):
        aggregate(tmp_path, plan=plan)


def test_plan_rejects_unknown_runs_and_duplicate_seeds(tmp_path):
    _run(tmp_path, "mean", 1)
    with pytest.raises(ValueError, match="not in the supplied plan"):
        aggregate(tmp_path, plan=_plan(tmp_path, ["signed"], [1]))
    plan = _plan(tmp_path, ["mean"], [1])
    plan["jobs"].append(dict(plan["jobs"][0]))
    with pytest.raises(ValueError, match="Repeated plan"):
        aggregate(tmp_path, plan=plan)


def test_completed_status_without_selected_checkpoint_cannot_rank(tmp_path):
    run = _run(tmp_path, "mean", 1)
    (run / "best.pth").unlink()
    with pytest.raises(ValueError, match="missing selected best checkpoint"):
        aggregate(tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "0.2", True])
def test_nonfinite_and_nonnumeric_scores_cannot_receive_ranks(tmp_path, value):
    _run(tmp_path, "mean", 1, test={"S": value})
    with pytest.raises(ValueError, match="finite numeric"):
        aggregate(tmp_path)


def _profile(run, *, device="cpu", dtype="float32", classified=True):
    _write(run / "profile.json", {
        "checkpoint_sha256": sha256_file(run / "best.pth"), "parameters": 1000,
        "latency_ms": {"median": 2}, "images_per_second": 500,
        "environment": {"device": device}, "protocol": {"device": device, "dtype": dtype, "size": 384,
                    "batch_size": 1, "complexity_protocol": {"convention": "two-flop-mac"}},
        "complexity": {"macs": 100, "flops": 200, "macs_per_image": 100, "flops_per_image": 200,
                       "estimated_total_flops": 250 if classified else None,
                       "estimated_total_flops_per_image": 250 if classified else None,
                       "scope": "all heads", "convention": "two-flop-mac", "version": "v1",
                       "coverage": {"all_observed_ops_classified": classified}}})


def test_costs_include_complexity_but_never_rank_mixed_hardware_dtype_coverage(tmp_path):
    for variant, settings in [("mean", {}), ("single", {"device": "cuda"}),
                              ("signed", {"dtype": "float16"}), ("absolute", {"classified": False})]:
        _profile(_run(tmp_path, variant, 1), **settings)
    report = build_leaderboard(aggregate(tmp_path))
    assert len(_boards(report, "inference", "flops_per_image")) == 4
    assert len(_boards(report, "inference", "estimated_total_flops_per_image")) == 3
    assert all(board["rows"][0]["mean"] == 200 for board in _boards(report, "inference", "flops_per_image"))


def test_custom_profile_path_preserves_original_and_checks_checkpoint_identity(tmp_path):
    run = _run(tmp_path, "mean", 1)
    _profile(run)
    original = (run / "profile.json").read_bytes()
    updated = json.loads(original)
    updated["latency_ms"]["median"] = 7
    _write(run / "profiles/v2.json", updated)
    report = build_leaderboard(aggregate(tmp_path, profile_name="profiles/v2.json"))
    board, = _boards(report, "inference", "latency_median_ms")
    assert board["rows"][0]["mean"] == 7
    assert report["profile_name"] == "profiles/v2.json"
    assert (run / "profile.json").read_bytes() == original
    for invalid in ["../escape", "/absolute", ""]:
        with pytest.raises(ValueError, match="profile_name.*relative path"):
            aggregate(tmp_path, profile_name=invalid)
    updated["checkpoint_sha256"] = "other"
    _write(run / "profiles/v2.json", updated)
    with pytest.raises(ValueError, match="selected best checkpoint"):
        aggregate(tmp_path, profile_name="profiles/v2.json")


def test_empty_plan_generates_all_reports_without_creating_results(tmp_path):
    results = tmp_path / "future-results"
    plan = tmp_path / "plan.json"
    _write(plan, _plan(results, ["single", "mean"], [1, 2, 3]))
    output = tmp_path / "reports"
    args = [sys.executable, str(ROOT / "summarize_study.py"), "--results", str(results), "--plan", str(plan), "--output", str(output)]
    process = subprocess.run(args, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert not results.exists()
    report = json.loads((output / "leaderboard.json").read_text())
    assert report["progress"]["not_started"] == 6
    assert report["boards"] == []
    assert {file.name for file in output.iterdir()} == {"summary.json", "summary.csv", "leaderboard.json", "leaderboard.csv", "leaderboard.md", "leaderboard.html"}
    sentinel = output / "preserve.txt"
    sentinel.write_text("keep me")
    rejected = subprocess.run(args, capture_output=True, text=True)
    assert rejected.returncode != 0 and "output already exists" in rejected.stderr
    refreshed = subprocess.run([*args, "--overwrite"], capture_output=True, text=True)
    assert refreshed.returncode == 0, refreshed.stderr
    assert sentinel.read_text() == "keep me"


def test_output_csv_json_and_html_preserve_scores_and_inert_labels(tmp_path):
    for seed in [1, 2, 3]:
        _run(tmp_path / "runs", "mean", seed, value=.1 * seed, test={"S": .8})
    report = build_leaderboard(aggregate(tmp_path / "runs"))
    report["boards"][0]["rows"][0]["variant_label"] = "</script><script>alert(1)</script>"
    output = tmp_path / "report"
    output.mkdir()
    write_leaderboard(output, report)
    csv_rows = list(csv.DictReader((output / "leaderboard.csv").open()))
    assert len(csv_rows) == 2
    assert float(csv_rows[0]["mean"]) == pytest.approx(.2)
    assert "</script><script>alert(1)</script>" not in (output / "leaderboard.html").read_text()
    assert "textContent" in (output / "leaderboard.html").read_text()
    assert json.loads((output / "leaderboard.json").read_text()) == report


def test_unknown_test_metric_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown test ranking"):
        build_leaderboard(aggregate(tmp_path), "arbitrary_composite")


def test_eight_executions_complete_does_not_hide_missing_cod10k(tmp_path):
    variants = ["single", "mean", "signed", "absolute", "relative", "normalized", "adjacent", "concat"]
    for variant in variants:
        run = _run(tmp_path / "runs", variant, 3407, test={"S": .8})
        _profile(run)
        path = run / "test/metrics.json"
        payload = json.loads(path.read_text())
        payload["datasets"] = {name: {"count": count, "S": .8, "wF": .7, "MAE": .06}
                               for name, count in EXPECTED_TEST_COUNTS.items() if name != "cod10k"}
        _write(path, payload)
    summary = aggregate(tmp_path / "runs")
    report = build_leaderboard(summary)
    assert report["progress"]["execution_complete"] == 8
    assert report["progress"]["all_stages_complete"] == 8
    assert report["progress"]["benchmark_complete"] == 0
    assert summary["benchmark_coverage"] == report["benchmark_coverage"]
    assert report["benchmark_coverage"]["missing_datasets"] == ["cod10k"]
    assert report["benchmark_coverage"]["datasets"]["cod10k"] == {
        "expected_count": 2026, "observed_counts": [], "evaluated_runs": 0,
        "standard_count_runs": 0, "missing_runs": 8}
    assert all(run["benchmark_coverage"]["missing_datasets"] == ["cod10k"] for run in report["runs"])
    output = tmp_path / "report"
    output.mkdir()
    write_leaderboard(output, report)
    markdown = (output / "leaderboard.md").read_text()
    assert "Standard four-dataset coverage: 0/8 runs complete" in markdown
    assert "| cod10k | 2026 | not evaluated | 0/8 | 0/8 |" in markdown
    html = (output / "leaderboard.html").read_text()
    assert 'id="coverage"' in html and "Execution status" in html
    assert '"missing_datasets":["cod10k"]' in html


def test_standard_coverage_requires_each_run_not_union_of_datasets(tmp_path):
    first = _run(tmp_path, "single", 1, test={"S": .8})
    second = _run(tmp_path, "mean", 1, test={"S": .9})
    for run, omitted in [(first, "cod10k"), (second, "camo")]:
        path = run / "test/metrics.json"
        payload = json.loads(path.read_text())
        payload["datasets"] = {name: {"count": count, "S": .8}
                               for name, count in EXPECTED_TEST_COUNTS.items() if name != omitted}
        _write(path, payload)
    report = build_leaderboard(aggregate(tmp_path))
    coverage = report["benchmark_coverage"]
    assert coverage["complete_runs"] == 0 and not coverage["complete"]
    assert set(coverage["missing_datasets"]) == {"camo", "cod10k"}
    assert coverage["datasets"]["cod10k"]["observed_counts"] == [2026]
    assert coverage["datasets"]["cod10k"]["missing_runs"] == 1


@pytest.mark.parametrize("count", [0, 2025])
def test_zero_or_nonstandard_count_does_not_complete_benchmark(tmp_path, count):
    run = _run(tmp_path, "single", 1, test={"S": .8})
    path = run / "test/metrics.json"
    payload = json.loads(path.read_text())
    payload["datasets"] = {name: {"count": count if name == "cod10k" else expected, "S": .8}
                           for name, expected in EXPECTED_TEST_COUNTS.items()}
    _write(path, payload)
    report = build_leaderboard(aggregate(tmp_path))
    assert report["progress"]["evaluation_complete"] == 1
    assert report["progress"]["benchmark_complete"] == 0
    coverage = report["benchmark_coverage"]
    if count == 0:
        assert coverage["missing_datasets"] == ["cod10k"]
    else:
        assert coverage["missing_datasets"] == []
        assert coverage["nonstandard_counts"][0]["counts"] == {"cod10k": {"expected": 2026, "observed": 2025}}


def test_four_standard_counts_complete_and_all_metric_rows_retained(tmp_path):
    from fsa_study.metrics import METRIC_KEYS
    run = _run(tmp_path, "single", 1, test={"S": .8})
    path = run / "test/metrics.json"
    payload = json.loads(path.read_text())
    payload["datasets"] = {name: {"count": count, **{key: .5 for key in METRIC_KEYS}}
                           for name, count in EXPECTED_TEST_COUNTS.items()}
    _write(path, payload)
    report = build_leaderboard(aggregate(tmp_path))
    assert report["progress"]["benchmark_complete"] == 1
    assert report["benchmark_coverage"]["complete"]
    assert len([board for board in report["boards"] if board["kind"] == "descriptive_test"]) == 4 * len(METRIC_KEYS)


@pytest.fixture
def extension_report(tmp_path, monkeypatch):
    from test_data import fixture
    root = tmp_path / "data"
    fixture(root, tests=tuple(EXPECTED_TEST_COUNTS))
    for dataset in EXPECTED_TEST_COUNTS:
        monkeypatch.setitem(EXPECTED_TEST_COUNTS, dataset, 2)
    original_path = tmp_path / "original.json"
    original = prepare_manifest(root, original_path, allow_nonstandard_counts=True,
                                test_datasets=["camo", "chameleon", "nc4k"])
    extended = extend_test_manifest(original_path, tmp_path / "extended.json")
    run = _run(tmp_path / "runs", "single", 1, data=original["content_sha256"], test={"S": .8})
    _write(run / "manifest.json", original)
    folder = run / "test"
    _write(folder / "training_manifest.json", original)
    _write(folder / "manifest.json", extended)
    path = folder / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics.update({"manifest_sha256": extended["content_sha256"],
                    "training_manifest_sha256": original["content_sha256"],
                    "manifest_extension": validate_test_extension(original, root, extended, root),
                    "datasets": {name: {"count": 2, "S": .8} for name in EXPECTED_TEST_COUNTS},
                    "artifacts": {"sha256": {name: sha256_file(folder / name)
                                             for name in ("manifest.json", "training_manifest.json")}}})
    _write(path, metrics)
    return run, path, metrics, root


def test_verified_extension_report_is_portable_and_uses_actual_test_manifest(extension_report):
    run, path, metrics, root = extension_report
    before = {file: sha256_file(file) for file in run.rglob("*") if file.is_file()}
    shutil.rmtree(root)  # Reporting needs saved proof, not the remote dataset disk.
    summary = aggregate(run.parent)
    board, = _boards(build_leaderboard(summary), "cod10k", "S")
    assert board["protocol"]["manifest_sha256"] == metrics["training_manifest_sha256"]
    assert board["protocol"]["evaluation_manifest_sha256"] == metrics["manifest_sha256"]
    assert summary["benchmark_coverage"]["complete_runs"] == 1
    assert {file: sha256_file(file) for file in before} == before


@pytest.mark.parametrize("tamper", ["missing_proof", "wrong_training_hash", "missing_snapshot", "snapshot_bytes",
                                   "forged_proof", "changed_training_records", "wrong_metric_count"])
def test_manifest_extension_metadata_is_not_accepted_without_proof(extension_report, tamper):
    run, path, metrics, _ = extension_report
    if tamper == "missing_proof":
        metrics.pop("manifest_extension")
    elif tamper == "wrong_training_hash":
        metrics["training_manifest_sha256"] = "wrong"
    elif tamper == "missing_snapshot":
        (path.parent / "training_manifest.json").unlink()
    elif tamper == "snapshot_bytes":
        snapshot = path.parent / "manifest.json"
        snapshot.write_text(snapshot.read_text() + "\n")
    elif tamper == "forged_proof":
        metrics["manifest_extension"]["added_records"] += 1
    elif tamper == "changed_training_records":
        snapshot = path.parent / "manifest.json"
        payload = json.loads(snapshot.read_text())
        payload["records"][0]["mask_sha256"] = "0" * 64
        payload["content_sha256"] = manifest_digest(payload)
        _write(snapshot, payload)
        metrics["manifest_sha256"] = payload["content_sha256"]
        metrics["manifest_extension"]["evaluation_manifest_sha256"] = payload["content_sha256"]
        metrics["artifacts"]["sha256"]["manifest.json"] = sha256_file(snapshot)
    elif tamper == "wrong_metric_count":
        metrics["datasets"]["cod10k"]["count"] = 1
    _write(path, metrics)
    with pytest.raises(ValueError):
        aggregate(run.parent)


def _set_full_train(run):
    path = run / "config.json"
    config = json.loads(path.read_text())
    config.update({"training_protocol": "full_train",
                   "training_split_counts": {"train": {"camo": 1000, "cod10k": 3040}, "val": {}},
                   "training_sample_count": 4040, "validation_sample_count": 0,
                   "checkpoint_selection": "fixed final epoch; no validation or test selection"})
    _write(path, config)
    status_path = run / "status.json"
    status = json.loads(status_path.read_text())
    status["best_val_mae"] = None
    _write(status_path, status)


def test_full_train_four_dataset_report_has_test_and_cost_ranks_without_validation(tmp_path):
    variants = ["single", "mean", "signed", "absolute", "relative", "normalized", "adjacent", "concat"]
    for variant in variants:
        run = _run(tmp_path, variant, 3407, test={"S": .8})
        _set_full_train(run)
        _profile(run)
        path = run / "test/metrics.json"
        payload = json.loads(path.read_text())
        payload["datasets"] = {name: {"count": count, "S": .8, "wF": .7, "MAE": .06}
                               for name, count in EXPECTED_TEST_COUNTS.items()}
        _write(path, payload)
    report = build_leaderboard(aggregate(tmp_path))
    assert not _boards(report, "validation", "MAE")
    assert all(board["kind"] != "selection" for board in report["boards"])
    assert len(_boards(report, "cod10k", "S")[0]["rows"]) == 8
    assert len(_boards(report, "inference", "latency_median_ms")[0]["rows"]) == 8
    assert report["progress"]["execution_complete"] == report["progress"]["benchmark_complete"] == 8
    assert all(run["training_protocol"] == "full_train" for run in report["runs"])
    assert all(board["protocol"]["training_sample_count"] == 4040 for board in report["boards"])
    assert "fixed final epoch" in report["selection_rule"]


def test_holdout_and_full_train_test_scores_never_share_cohort(tmp_path):
    _run(tmp_path, "single", 1, test={"S": .8})
    full = _run(tmp_path, "mean", 1, test={"S": .9})
    _set_full_train(full)
    report = build_leaderboard(aggregate(tmp_path))
    assert len(_boards(report, "cod10k", "S")) == 2
    assert [row["variant"] for row in _boards(report)[0]["rows"]] == ["single"]


@pytest.mark.parametrize("tamper", ["selection", "validation_mae", "unknown_protocol"])
def test_invalid_full_train_selection_cannot_produce_ranks(tmp_path, tamper):
    run = _run(tmp_path, "single", 1, test={"S": .8})
    _set_full_train(run)
    if tamper == "validation_mae":
        path = run / "status.json"
        payload = json.loads(path.read_text())
        payload["best_val_mae"] = .1
    else:
        path = run / "config.json"
        payload = json.loads(path.read_text())
        if tamper == "selection":
            payload["checkpoint_selection"] = "best test MAE"
        else:
            payload["training_protocol"] = "anything"
    _write(path, payload)
    with pytest.raises(ValueError):
        aggregate(tmp_path)
