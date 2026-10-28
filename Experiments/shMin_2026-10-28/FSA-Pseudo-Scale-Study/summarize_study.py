#!/usr/bin/env python3
"""Aggregate comparable completed runs; no ranking or best selection on test sets."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from fsa_study.data import sha256_file


def aggregate(root):
    groups = defaultdict(list)
    incomplete = []
    for config_path in sorted(Path(root).rglob("config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        if "model_config" not in config:
            continue
        if not (run / "status.json").exists():
            incomplete.append(str(run))
            continue
        model = config["model_config"]
        variant = "single" if model["baseline"] else "legacy_full" if model["legacy_reference"] else f"{model['generator']}__{model['utilizer']}"
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "training_complete":
            incomplete.append(str(run))
            continue
        # Keep architecture, input, optimization, initial weights and data protocol
        # distinct; incompatible conditions are never silently averaged together.
        protocol = {key: config.get(key) for key in ("manifest_sha256", "backbone_sha256", "size", "epochs", "lr", "batch_size", "eval_interval", "determinism", "source_sha256", "loss_weights", "augmentation", "edge_protocol")}
        protocol["packages"] = config.get("environment", {}).get("packages", {})
        protocol["channels"] = model["channels"]
        protocol["generator_options"] = model.get("generator_options", {})
        protocol["utilizer_options"] = model.get("utilizer_options", {})
        condition = json.dumps(protocol, sort_keys=True)
        base = {"variant": variant, "seed": model["seed"], "run": str(run)}
        groups[(variant, condition, "validation", "MAE")].append({**base, "value": status["best_val_mae"]})
        metric_path = run / "test" / "metrics.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text())
            if metrics.get("split") != "test":
                raise ValueError(f"Expected test split: {metric_path}")
            if metrics["manifest_sha256"] != config["manifest_sha256"] or metrics["model_config"] != model:
                raise ValueError(f"Mismatched test metadata: {metric_path}")
            if not (run / "best.pth").exists() or metrics.get("checkpoint_sha256") != sha256_file(run / "best.pth"):
                raise ValueError(f"Evaluation did not use this run's selected best checkpoint: {metric_path}")
            test_protocol = {**protocol, "metric_protocol": metrics.get("metric_protocol"),
                             "prediction_protocol": metrics.get("prediction_protocol"),
                             "evaluation_packages": metrics.get("environment", {}).get("packages", {})}
            test_condition = json.dumps(test_protocol, sort_keys=True)
            for dataset, values in metrics["datasets"].items():
                for metric, value in values.items():
                    if metric != "count":
                        groups[(variant, test_condition, dataset, metric)].append({**base, "value": value})
        profile_path = run / "profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            if not (run / "best.pth").exists() or profile.get("checkpoint_sha256") != sha256_file(run / "best.pth"):
                raise ValueError(f"Profile did not use this run's selected best checkpoint: {profile_path}")
            measurement = profile["protocol"]
            cost_protocol = {**protocol, "profile_environment": profile["environment"],
                             "profile_input_sha256": profile.get("image_sha256"),
                             "profile": {key: measurement.get(key) for key in
                                         ("size", "batch_size", "threads", "warmup", "repeats", "device", "dtype", "input", "seed", "timed_scope")}}
            cost_condition = json.dumps(cost_protocol, sort_keys=True)
            costs = {"parameters": profile["parameters"], "latency_median_ms": profile["latency_ms"]["median"],
                     "images_per_second": profile["images_per_second"]}
            if profile.get("cuda_memory"):
                costs["cuda_peak_allocated_bytes"] = profile["cuda_memory"]["peak_allocated_bytes"]
            for metric, value in costs.items():
                groups[(variant, cost_condition, "inference", metric)].append({**base, "value": value})
    rows = []
    for (variant, condition, dataset, metric), entries in sorted(groups.items()):
        seeds = [entry["seed"] for entry in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated seed in {variant}/{dataset}/{metric}; select one experiment root")
        values = [entry["value"] for entry in entries]
        rows.append({"variant": variant, "dataset": dataset, "metric": metric,
                     "n": len(values), "seeds": seeds, "mean": statistics.mean(values),
                     "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                     "protocol": json.loads(condition), "runs": [e["run"] for e in entries]})
    return {"rows": rows, "incomplete_runs": incomplete,
            "interpretation": "Descriptive mean/sample SD only, no significance claim; test metrics are not model-selection criteria"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True, help="New directory for summary.json and summary.csv")
    args = parser.parse_args()
    report = aggregate(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        columns = ["variant", "dataset", "metric", "n", "seeds", "mean", "sample_std", "protocol", "runs"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    print(json.dumps({"rows": len(report["rows"]), "incomplete": len(report["incomplete_runs"])}))


if __name__ == "__main__":
    main()
