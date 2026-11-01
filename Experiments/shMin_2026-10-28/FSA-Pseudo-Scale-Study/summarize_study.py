#!/usr/bin/env python3
"""Summarize completed runs and rank comparable seed cohorts without test selection."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
from fsa_study.data import geometry_protocol, manifest_digest, sha256_file
from fsa_study.benchmark import benchmark_coverage, validate_test_extension
from fsa_study.overlap import overlap_metadata
from fsa_study.leaderboard import build_leaderboard, summarize_benchmark_coverage, summarize_data_integrity, write_leaderboard


def _variant(model):
    return "single" if model["baseline"] else "legacy_full" if model["legacy_reference"] else f"{model['generator']}__{model['utilizer']}"


def _number(value, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric metric at {source}: {value!r}")
    return value


def _validate_test_manifest(metrics, config, metric_path, run):
    """An extended test manifest needs reproducible proof, not a trusted flag."""
    proof = metrics.get("manifest_extension")
    if proof is None:
        if metrics["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError(f"Mismatched test metadata: {metric_path}")
        if metrics.get("training_manifest_sha256", config["manifest_sha256"]) != config["manifest_sha256"]:
            raise ValueError(f"Mismatched training manifest metadata: {metric_path}")
        anchor = run / "manifest.json"
        if metrics.get("overlap_audit") is not None:
            if not anchor.is_file():
                raise ValueError(f"Overlap audit requires the training manifest snapshot: {anchor}")
            manifest = json.loads(anchor.read_text())
            if (manifest.get("content_sha256") != config["manifest_sha256"]
                    or manifest_digest(manifest) != config["manifest_sha256"]):
                raise ValueError(f"Mismatched manifest content digest: {anchor}")
            return _verify_overlap(metrics, manifest, metric_path)
        if anchor.is_file():
            manifest = json.loads(anchor.read_text())
            if manifest.get("overlap_policy", "strict") != "strict":
                raise ValueError(f"Acknowledged overlap metrics require an overlap audit: {metric_path}")
        return None
    if metrics.get("training_manifest_sha256") != config["manifest_sha256"]:
        raise ValueError(f"Mismatched training manifest metadata: {metric_path}")
    snapshots = {}
    recorded_hashes = metrics.get("artifacts", {}).get("sha256", {})
    for name in ("manifest.json", "training_manifest.json"):
        path = metric_path.parent / name
        if not path.is_file() or recorded_hashes.get(name) != sha256_file(path):
            raise ValueError(f"Missing or mismatched test manifest snapshot: {path}")
        snapshots[name] = json.loads(path.read_text())
    anchor = run / "manifest.json"
    if not anchor.is_file():
        raise ValueError(f"Missing original training manifest snapshot: {anchor}")
    original = json.loads(anchor.read_text())
    training, evaluation = snapshots["training_manifest.json"], snapshots["manifest.json"]
    for payload, expected in ((original, config["manifest_sha256"]),
                              (training, config["manifest_sha256"]),
                              (evaluation, metrics["manifest_sha256"])):
        if payload.get("content_sha256") != expected or manifest_digest(payload) != expected:
            raise ValueError(f"Mismatched manifest content digest: {metric_path}")
    training_root = evaluation.get("test_extension", {}).get("training_data_root", training["data_root"])
    verified = validate_test_extension(training, training_root, evaluation,
                                       evaluation["data_root"], verify_files=False)
    if proof != verified:
        raise ValueError(f"Mismatched test manifest extension proof: {metric_path}")
    expected_counts = defaultdict(int)
    for record in evaluation["records"]:
        if record["split"] == "test":
            expected_counts[record["dataset"]] += 1
    actual_counts = {dataset: values.get("count") for dataset, values in metrics["datasets"].items()}
    if actual_counts != dict(expected_counts):
        raise ValueError(f"Test metric counts do not match the evaluation manifest: {metric_path}")
    return _verify_overlap(metrics, evaluation, metric_path)


def _verify_overlap(metrics, manifest, metric_path):
    expected = overlap_metadata(manifest)
    recorded = metrics.get("overlap_audit")
    if recorded is None and expected["policy"] == "strict":
        # Historical reports without an audit remain unknown, even when their
        # count-only coverage happens to match all four standard datasets.
        return None
    if recorded != expected:
        raise ValueError(f"Evaluation overlap audit disagrees with its manifest: {metric_path}")
    return expected


def _plan_jobs(plan, root):
    if plan is None:
        return {}, None
    payload = json.loads(Path(plan).read_text()) if isinstance(plan, (str, Path)) else plan
    jobs, identities = {}, set()
    for job in payload["jobs"]:
        run = Path(job["run"]).resolve()
        if not run.is_relative_to(root):
            raise ValueError(f"Plan run is outside results root: {run}")
        identity = (job["variant"], job["seed"])
        if run in jobs or identity in identities:
            raise ValueError(f"Repeated plan run or variant/seed: {run}")
        jobs[run] = job
        identities.add(identity)
    return jobs, payload


def aggregate(root, evaluation_subdir="test", plan=None, profile_name="profile.json"):
    root = Path(root).resolve()
    evaluation_path = Path(evaluation_subdir)
    if evaluation_path.is_absolute() or ".." in evaluation_path.parts or evaluation_path == Path("."):
        raise ValueError("evaluation_subdir must be a nonempty relative path inside each run")
    profile_relative = Path(profile_name)
    if profile_relative.is_absolute() or ".." in profile_relative.parts or profile_relative == Path("."):
        raise ValueError("profile_name must be a nonempty relative path inside each run")
    jobs, plan_payload = _plan_jobs(plan, root)
    groups = defaultdict(list)
    incomplete = []
    progress = {}
    for path, job in jobs.items():
        progress[str(path)] = {"run": str(path), "variant": job["variant"], "seed": job["seed"],
                               "status": "not_started", "training_complete": False,
                               "evaluation_complete": False, "profile_complete": False, "planned": True,
                               "benchmark_coverage": benchmark_coverage({})}
    for config_path in sorted(root.rglob("config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        if "model_config" not in config:
            continue
        model = config["model_config"]
        variant = _variant(model)
        if plan_payload is not None and run not in jobs:
            raise ValueError(f"Run is not in the supplied plan: {run}")
        if run in jobs and (jobs[run]["variant"] != variant or jobs[run]["seed"] != model["seed"]):
            raise ValueError(f"Mismatched plan variant/seed: {run}")
        state = {"run": str(run), "variant": variant, "seed": model["seed"], "status": "training_incomplete",
                 "training_complete": False, "evaluation_complete": False, "profile_complete": False,
                 "planned": run in jobs, "benchmark_coverage": benchmark_coverage({})}
        progress[str(run)] = state
        if not (run / "status.json").exists():
            incomplete.append(str(run))
            continue
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "training_complete":
            incomplete.append(str(run))
            continue
        if not (run / "best.pth").is_file():
            raise ValueError(f"Completed run missing selected best checkpoint: {run}")
        state.update(status="training_complete", training_complete=True)
        # Keep architecture, input, optimization, initial weights and data protocol
        # distinct; incompatible conditions are never silently averaged together.
        protocol = {key: config.get(key) for key in ("manifest_sha256", "backbone_sha256", "backbone_initialization", "size", "epochs", "lr", "batch_size", "eval_interval", "determinism", "source_sha256", "loss_weights", "augmentation", "edge_protocol", "checkpoint_selection")}
        protocol["packages"] = config.get("environment", {}).get("packages", {})
        protocol["channels"] = model["channels"]
        protocol["generator_options"] = model.get("generator_options", {})
        protocol["utilizer_options"] = model.get("utilizer_options", {})
        training_protocol = config.get("training_protocol", "holdout")
        if training_protocol not in {"holdout", "full_train"}:
            raise ValueError(f"Unknown training protocol: {config_path}")
        for key in ("training_protocol", "training_split_counts", "training_sample_count", "validation_sample_count"):
            if key in config:
                protocol[key] = config[key]
        condition = json.dumps(protocol, sort_keys=True)
        base = {"variant": variant, "seed": model["seed"], "run": str(run)}
        state["training_protocol"] = training_protocol
        if training_protocol == "full_train":
            if config.get("checkpoint_selection") != "fixed final epoch; no validation or test selection":
                raise ValueError(f"Full-train run must record fixed final epoch checkpoint selection: {config_path}")
            if status.get("best_val_mae") is not None:
                raise ValueError(f"Full-train run cannot contain a validation selection MAE: {run / 'status.json'}")
        else:
            groups[(variant, condition, "validation", "MAE")].append({**base, "value": _number(status["best_val_mae"], run / "status.json")})
        metric_path = run / evaluation_path / "metrics.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text())
            if metrics.get("split") != "test":
                raise ValueError(f"Expected test split: {metric_path}")
            if metrics["model_config"] != model:
                raise ValueError(f"Mismatched test metadata: {metric_path}")
            overlap = _validate_test_manifest(metrics, config, metric_path, run)
            if not (run / "best.pth").exists() or metrics.get("checkpoint_sha256") != sha256_file(run / "best.pth"):
                raise ValueError(f"Evaluation did not use this run's selected best checkpoint: {metric_path}")
            test_protocol = {**protocol, "metric_protocol": metrics.get("metric_protocol"),
                             "geometry_protocol": metrics.get("geometry_protocol", geometry_protocol("strict")),
                             "metric_protocol_details": metrics.get("metric_protocol_details"),
                             "prediction_protocol": metrics.get("prediction_protocol"),
                             "evaluation_source_sha256": metrics.get("source_sha256"),
                             "evaluation_packages": metrics.get("environment", {}).get("packages", {})}
            # Different overlap policies, affected identities or train/val
            # assignments are separate comparison cohorts.
            test_protocol["overlap_audit"] = overlap
            if metrics.get("manifest_extension") is not None:
                test_protocol["evaluation_manifest_sha256"] = metrics["manifest_sha256"]
            state["evaluation_complete"] = True
            state["overlap_audit"] = overlap
            state["benchmark_coverage"] = benchmark_coverage({
                dataset: values.get("count") for dataset, values in metrics["datasets"].items()})
            for dataset, values in metrics["datasets"].items():
                test_condition = json.dumps({**test_protocol, "evaluation_count": values.get("count")}, sort_keys=True)
                for metric, value in values.items():
                    if metric != "count":
                        groups[(variant, test_condition, dataset, metric)].append({**base, "value": _number(value, f"{metric_path}:{dataset}/{metric}")})
        profile_path = run / profile_relative
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            if not (run / "best.pth").exists() or profile.get("checkpoint_sha256") != sha256_file(run / "best.pth"):
                raise ValueError(f"Profile did not use this run's selected best checkpoint: {profile_path}")
            measurement = profile["protocol"]
            cost_protocol = {**protocol, "profile_environment": profile["environment"],
                             "profile_input_sha256": profile.get("image_sha256"),
                             "profile": {key: measurement.get(key) for key in
                                         ("size", "batch_size", "threads", "warmup", "repeats", "device", "dtype", "input", "seed", "timed_scope", "complexity_protocol")}}
            cost_condition = json.dumps(cost_protocol, sort_keys=True)
            costs = {"parameters": profile["parameters"], "latency_median_ms": profile["latency_ms"]["median"],
                     "images_per_second": profile["images_per_second"]}
            if profile.get("cuda_memory"):
                costs["cuda_peak_allocated_bytes"] = profile["cuda_memory"]["peak_allocated_bytes"]
            complexity = profile.get("complexity", {})
            for key in ("macs", "flops", "estimated_noncontraction_flops", "estimated_counted_flops", "estimated_total_flops",
                        "macs_per_image", "flops_per_image", "estimated_total_flops_per_image"):
                if complexity.get(key) is not None:
                    costs[key] = complexity[key]
            if complexity:
                cost_protocol["complexity"] = {key: complexity.get(key) for key in ("scope", "convention", "version")}
                cost_protocol["complexity"]["all_observed_ops_classified"] = complexity.get("coverage", {}).get("all_observed_ops_classified")
                cost_condition = json.dumps(cost_protocol, sort_keys=True)
            for metric, value in costs.items():
                groups[(variant, cost_condition, "inference", metric)].append({**base, "value": _number(value, f"{profile_path}:{metric}")})
            state["profile_complete"] = True
        if state["evaluation_complete"] and state["profile_complete"]:
            state["status"] = "complete"
        elif state["evaluation_complete"]:
            state["status"] = "evaluation_complete"
    rows = []
    for (variant, condition, dataset, metric), entries in sorted(groups.items()):
        entries.sort(key=lambda entry: entry["seed"])
        seeds = [entry["seed"] for entry in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated seed in {variant}/{dataset}/{metric}; select one experiment root")
        values = [entry["value"] for entry in entries]
        rows.append({"variant": variant, "dataset": dataset, "metric": metric,
                     "n": len(values), "seeds": seeds, "mean": statistics.mean(values),
                     "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                     "protocol": json.loads(condition), "runs": [e["run"] for e in entries]})
    expected = defaultdict(list)
    for job in jobs.values():
        expected[job["variant"]].append(job["seed"])
    return {"rows": rows, "incomplete_runs": incomplete, "runs": list(progress.values()),
            "benchmark_coverage": summarize_benchmark_coverage(list(progress.values())),
            "data_integrity": summarize_data_integrity(list(progress.values())),
            "expected_seeds": {variant: sorted(seeds) for variant, seeds in expected.items()},
            "plan": {key: plan_payload.get(key) for key in ("phase", "run_count", "variant_count", "seeds")} if plan_payload is not None else None,
            "evaluation_subdir": evaluation_subdir,
            "profile_name": profile_name,
            "interpretation": "Descriptive mean/sample SD only, no significance claim; test metrics are not model-selection criteria"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True, help="New directory for summary and leaderboard artifacts")
    parser.add_argument("--plan", help="Optional plan.json to report not-started runs and expected seeds")
    parser.add_argument("--evaluation-subdir", default="test", help="Relative evaluation directory inside each run")
    parser.add_argument("--profile-name", default="profile.json", help="Relative profile filename inside each run; use a new name for re-profiling")
    parser.add_argument("--test-metric", default="S", help="Default test metric shown in Markdown; HTML includes every metric")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace the six owned report files; preserve all other files")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error("output already exists; use a new directory or --overwrite for report refresh")
    filenames = ("summary.json", "summary.csv", "leaderboard.json", "leaderboard.csv", "leaderboard.md", "leaderboard.html")
    if output.exists() and (not output.is_dir() or any((output / name).is_dir() for name in filenames)):
        parser.error("output or owned report path is not a regular directory/file")
    report = aggregate(args.results, args.evaluation_subdir, args.plan, args.profile_name)
    leaderboard = build_leaderboard(report, test_metric=args.test_metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".leaderboard-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (staging / "summary.csv").open("w", newline="") as stream:
            columns = ["variant", "dataset", "metric", "n", "seeds", "mean", "sample_std", "protocol", "runs"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in report["rows"]:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        write_leaderboard(staging, leaderboard)
        output.mkdir(exist_ok=args.overwrite)
        for name in filenames:
            (staging / name).replace(output / name)
    print(json.dumps({"rows": len(report["rows"]), "incomplete": len(report["incomplete_runs"]),
                      "leaderboard": str(output / "leaderboard.html"), "progress": leaderboard["progress"],
                      "benchmark_coverage": leaderboard["benchmark_coverage"],
                      "data_integrity": leaderboard["data_integrity"]}))


if __name__ == "__main__":
    main()
