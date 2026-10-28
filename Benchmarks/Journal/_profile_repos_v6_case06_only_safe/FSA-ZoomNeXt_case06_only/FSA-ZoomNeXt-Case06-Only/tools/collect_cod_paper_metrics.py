#!/usr/bin/env python3
"""Collect COD paper-style metrics from case outputs.

Outputs columns for Sα ↑, Eφ ↑, Fwβ ↑, and M ↓. Eφ is mapped to
meanE by default; use --e-source adpE or --e-source maxE if needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

DATASET_ORDER = ["chameleon", "camo_te", "cod10k_te", "nc4k"]
DATASET_LABELS = {"chameleon": "CHAMELEON", "camo_te": "CAMO", "cod10k_te": "COD10K", "nc4k": "NC4K"}
METRIC_HEADERS = ["Sα↑", "Eφ↑", "Fwβ↑", "M↓"]


def _find_metric_file(case_dir: Path) -> Optional[Path]:
    best = sorted(case_dir.glob("**/metrics_best.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    latest = sorted(case_dir.glob("**/metrics_latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return latest[0] if latest else None


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def _mean_present(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(mean(vals)) if vals else None


def _values(metrics: Dict[str, Any], dataset: str, e_source: str) -> Dict[str, Optional[float]]:
    v = metrics.get(dataset, {})
    if not isinstance(v, dict):
        v = {}
    return {
        "Sα↑": _to_float(v.get("S")),
        "Eφ↑": _to_float(v.get(e_source)),
        "Fwβ↑": _to_float(v.get("wF")),
        "M↓": _to_float(v.get("MAE")),
    }


def _case_dirs(output_root: Path, cases: Optional[List[str]]) -> List[Path]:
    if cases:
        return [output_root / c for c in cases]
    return sorted([p for p in output_root.glob("case*") if p.is_dir()])


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row(case_id: str, metric_file: Optional[Path], metrics: Dict[str, Any], e_source: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case": case_id, "metric_file": "NOT_FOUND" if metric_file is None else str(metric_file), "Eφ_source": e_source}
    buf = {h: [] for h in METRIC_HEADERS}
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        vals = _values(metrics, dataset, e_source)
        for h in METRIC_HEADERS:
            row[f"{label}.{h}"] = _fmt(vals[h])
            buf[h].append(vals[h])
    for h in METRIC_HEADERS:
        row[f"AVG.{h}"] = _fmt(_mean_present(buf[h]))
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case", "metric_file", "Eφ_source"] + [f"AVG.{h}" for h in METRIC_HEADERS]
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        cols += [f"{label}.{h}" for h in METRIC_HEADERS]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case"] + [f"AVG.{h}" for h in METRIC_HEADERS]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("Eφ is selected with --e-source. Default: meanE.")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--save-csv", required=True, type=Path)
    parser.add_argument("--save-md", type=Path, default=None)
    parser.add_argument("--e-source", default="meanE", choices=["meanE", "adpE", "maxE"])
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for case_dir in _case_dirs(args.output_root, args.cases):
        mf = _find_metric_file(case_dir)
        metrics = _load(mf) if mf is not None else {}
        rows.append(_row(case_dir.name, mf, metrics, args.e_source))
    _write_csv(args.save_csv, rows)
    if args.save_md is not None:
        _write_md(args.save_md, rows)


if __name__ == "__main__":
    main()
