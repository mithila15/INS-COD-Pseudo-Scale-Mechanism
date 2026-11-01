"""Portable metadata/report checks only; no training or dataset downloads."""
import copy
import csv
import json

import pytest

from fsa_study.data import manifest_digest
from fsa_study.metrics import METRIC_KEYS
from fsa_study.overlap import KNOWN_COD10K_OVERLAPS, acknowledge_cod10k_overlap, overlap_metadata
from fsa_study.leaderboard import build_leaderboard, write_leaderboard
from summarize_study import aggregate
from test_leaderboard import _run, _write, _boards, extension_report
from test_complete_benchmark import original, _write_completed_evaluation
from complete_benchmark import complete_benchmark_plan
from run_plan import completed


def acknowledged_report(root, val_count=1):
    manifest = {"data_root": "/metadata-only-fixture", "records": []}
    for index, (train, test, digest) in enumerate(KNOWN_COD10K_OVERLAPS):
        for source, stem, split in (("cod10k_train", train, "val" if index < val_count else "train"),
                                    ("cod10k_test", test, "test")):
            manifest["records"].append({"id": f"{source}/{stem}", "source": source, "dataset": "cod10k",
                                        "split": split, "image": f"{source}/Image/{stem}.jpg",
                                        "mask": f"{source}/GT/{stem}.png", "image_sha256": digest})
    audit = acknowledge_cod10k_overlap(manifest)
    manifest["content_sha256"] = manifest_digest(manifest)
    run = _run(root, "single", 3407, data=manifest["content_sha256"],
               test=dict.fromkeys(METRIC_KEYS, .5))
    _write(run / "manifest.json", manifest)
    path = run / "test/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["datasets"]["cod10k"]["count"] = 7
    metrics["overlap_audit"] = audit
    _write(path, metrics)
    return run, path, metrics


@pytest.mark.parametrize("val_count", [0, 1])
def test_known_overlap_preserves_all_scores_and_exports_visible_integrity(tmp_path, val_count):
    run, _, _ = acknowledged_report(tmp_path / "runs", val_count)
    summary = aggregate(run.parent)
    report = build_leaderboard(summary)
    assert len([b for b in report["boards"] if b["kind"] == "descriptive_test"]) == len(METRIC_KEYS)
    assert report["data_integrity"]["overlapping_runs"] == 1
    assert report["data_integrity"]["disjoint_runs"] == 0
    board, = _boards(report, "cod10k", "S")
    audit = board["protocol"]["overlap_audit"]
    assert audit["train_test_count"] == 7 - val_count
    assert audit["val_test_count"] == val_count and audit["affected_test_count"] == 7
    output = tmp_path / "report"
    output.mkdir()
    write_leaderboard(output, report)
    saved = list(csv.DictReader((output / "leaderboard.csv").open()))
    row = next(r for r in saved if r["dataset"] == "cod10k" and r["metric"] == "S")
    assert row["overlap_policy"] == "acknowledged-cod10k-v1"
    assert row["split_disjoint"] == "False"
    assert int(row["train_test_count"]) == 7 - val_count
    assert int(row["val_test_count"]) == val_count and row["affected_test_count"] == "7"
    markdown = (output / "leaderboard.md").read_text()
    assert "0 disjoint, 1 overlapping, 0 unknown" in markdown
    assert f"acknowledged-cod10k-v1 | False | {7 - val_count} | {val_count} | 7" in markdown
    html = (output / "leaderboard.html").read_text()
    assert 'id="integrity"' in html and "retain the acknowledged overlapping images" in html


def test_historical_metrics_without_audit_stay_unknown(tmp_path):
    _run(tmp_path, "single", 1, test={"S": .8})
    report = build_leaderboard(aggregate(tmp_path))
    assert report["data_integrity"]["unknown_runs"] == 1
    assert report["data_integrity"]["disjoint_runs"] == 0
    board, = _boards(report, "cod10k", "S")
    assert board["rows"][0]["overlap_policy"] == "unknown"
    assert board["rows"][0]["split_disjoint"] is None


@pytest.mark.parametrize("tamper", ["absent", "counts", "policy", "missing_snapshot", "changed_snapshot"])
def test_missing_or_forged_nonextension_audit_is_rejected(tmp_path, tamper):
    run, path, metrics = acknowledged_report(tmp_path)
    if tamper == "absent":
        metrics.pop("overlap_audit")
    elif tamper == "counts":
        metrics["overlap_audit"]["train_test_count"] = 0
    elif tamper == "policy":
        metrics["overlap_audit"]["policy"] = "strict"
    elif tamper == "missing_snapshot":
        (run / "manifest.json").unlink()
    else:
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["records"][0]["split"] = "train"
        _write(run / "manifest.json", manifest)
    _write(path, metrics)
    with pytest.raises(ValueError, match="overlap audit|Overlap audit|manifest content"):
        aggregate(run.parent)


def test_extension_audit_verified_from_portable_snapshot(extension_report):
    run, path, metrics, _ = extension_report
    manifest = json.loads((path.parent / "manifest.json").read_text())
    metrics["overlap_audit"] = overlap_metadata(manifest)
    _write(path, metrics)
    assert aggregate(run.parent)["data_integrity"]["disjoint_runs"] == 1
    metrics["overlap_audit"]["affected_test_count"] = 7
    _write(path, metrics)
    with pytest.raises(ValueError, match="overlap audit disagrees"):
        aggregate(run.parent)


def test_overlap_policy_and_affected_split_are_comparison_protocol(tmp_path):
    run, _, _ = acknowledged_report(tmp_path)
    summary = aggregate(run.parent)
    source = next(r for r in summary["rows"] if r["dataset"] == "cod10k" and r["metric"] == "S")
    second = copy.deepcopy(source)
    second["variant"] = "native_conv__mean"
    second["protocol"]["overlap_audit"]["train_test_count"] = 7
    second["protocol"]["overlap_audit"]["val_test_count"] = 0
    summary["rows"] = [source, second]
    assert len(_boards(build_leaderboard(summary), "cod10k", "S")) == 2


def test_plan_completion_rejects_forged_overlap_audit(original, tmp_path):
    source, _, original_path, _ = original
    output = tmp_path / "new-plan"
    job = complete_benchmark_plan(source, output)["jobs"][0]
    target, metrics = _write_completed_evaluation(job, output / "manifest.json", original_path)
    manifest = json.loads((output / "manifest.json").read_text())
    metrics["overlap_audit"] = overlap_metadata(manifest)
    _write(target / "metrics.json", metrics)
    assert completed(job, "evaluate")
    metrics["overlap_audit"]["affected_test_count"] = 7
    _write(target / "metrics.json", metrics)
    with pytest.raises(ValueError, match="overlap audit disagrees"):
        completed(job, "evaluate")
