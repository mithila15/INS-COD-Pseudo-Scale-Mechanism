#!/usr/bin/env python3
"""Collect COD metrics from separated FSA case outputs.

The summary CSV uses publication-facing COD column names:
    Sα ↑, Eφ ↑, Fwβ ↑, M ↓

Internal metric keys in metrics_best.json remain:
    S, meanE, wF, MAE
where meanE is used as Eφ.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

METRIC_ORDER = ["S", "meanE", "wF", "MAE"]
METRIC_LABELS = {
    "S": "Sα↑",
    "meanE": "Eφ↑",
    "wF": "Fwβ↑",
    "MAE": "M↓",
}
# Fallbacks for older result files that may contain adpE/maxE but not meanE.
METRIC_FALLBACKS = {
    "meanE": ["meanE", "adpE", "maxE"],
}
DATASET_ORDER = ["chameleon", "camo_te", "cod10k_te", "nc4k"]


def _find_metric_files(case_dir: Path) -> List[Path]:
    best = sorted(case_dir.glob("**/metrics_best.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[:1]
    latest = sorted(case_dir.glob("**/metrics_latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return latest[:1]


def _metric_value(values: Dict[str, Any], metric: str) -> Optional[float]:
    keys = METRIC_FALLBACKS.get(metric, [metric])
    for key in keys:
        if key in values:
            try:
                return float(values[key])
            except Exception:
                return None
    return None


def _ordered_datasets(metrics: Dict[str, Any]) -> List[str]:
    known = [d for d in DATASET_ORDER if d in metrics]
    extra = sorted([d for d in metrics.keys() if d not in known and isinstance(metrics.get(d), dict)])
    return known + extra


def _mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def _flatten(case_id: str, metric_path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case": case_id, "metric_file": str(metric_path)}
    dataset_names = _ordered_datasets(metrics)

    # Average over all available evaluation datasets.
    for metric in METRIC_ORDER:
        avg = _mean_ignore_none(_metric_value(metrics[d], metric) for d in dataset_names if isinstance(metrics.get(d), dict))
        row[f"avg.{METRIC_LABELS[metric]}"] = _fmt(avg)

    # Dataset-wise values.
    for dataset in dataset_names:
        values = metrics.get(dataset, {})
        if not isinstance(values, dict):
            continue
        for metric in METRIC_ORDER:
            row[f"{dataset}.{METRIC_LABELS[metric]}"] = _fmt(_metric_value(values, metric))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--save-csv", required=True, type=Path)
    parser.add_argument("--save-paper-csv", type=Path, default=None, help="Optional duplicate paper-style CSV path for compatibility with scripts.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for case_dir in sorted(args.output_root.glob("case*")):
        if not case_dir.is_dir():
            continue
        metric_files = _find_metric_files(case_dir)
        if not metric_files:
            rows.append({"case": case_dir.name, "metric_file": "NOT_FOUND"})
            continue
        metric_path = metric_files[0]
        with metric_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(_flatten(case_id=case_dir.name, metric_path=metric_path, metrics=metrics))

    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        args.save_csv.write_text("case,metric_file\n", encoding="utf-8")
        print(f"No case metrics found under {args.output_root}")
        return

    columns: List[str] = []
    preferred_prefix = ["case", "metric_file"]
    for col in preferred_prefix:
        if any(col in row for row in rows):
            columns.append(col)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    def _write_csv(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {path}")

    _write_csv(args.save_csv)
    if args.save_paper_csv is not None:
        _write_csv(args.save_paper_csv)


if __name__ == "__main__":
    main()
