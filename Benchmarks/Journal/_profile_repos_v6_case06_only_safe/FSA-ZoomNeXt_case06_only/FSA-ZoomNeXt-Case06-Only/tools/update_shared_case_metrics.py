#!/usr/bin/env python3
"""Build one shared COD metric table from one or more experiment output roots.

The script scans case directories such as ``case02_av_stage5`` and picks the
newest ``metrics_best.json`` if available, otherwise the newest
``metrics_latest.json``. If the same case appears under multiple roots, the
newest metric file is used. The output file is rewritten atomically, so it is
safe to call after every case; it will not append duplicate rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATASET_ORDER = ["chameleon", "camo_te", "cod10k_te", "nc4k"]
DATASET_LABELS = {
    "chameleon": "CHAMELEON",
    "camo_te": "CAMO",
    "cod10k_te": "COD10K",
    "nc4k": "NC4K",
}
METRIC_HEADERS = ["Sα↑", "Eφ↑", "Fwβ↑", "M↓"]


def natural_key(text: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def find_metric_file(case_dir: Path) -> Optional[Path]:
    best = sorted(case_dir.glob("**/metrics_best.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    latest = sorted(case_dir.glob("**/metrics_latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return latest[0] if latest else None


def discover_cases(output_roots: Sequence[Path], cases: Optional[Sequence[str]]) -> Dict[str, Path]:
    selected: Dict[str, Tuple[float, Path]] = {}
    wanted = set(cases or [])
    for root in output_roots:
        if not root.exists():
            continue
        candidates = [root / c for c in wanted] if wanted else [p for p in root.glob("case*") if p.is_dir()]
        for case_dir in candidates:
            if not case_dir.is_dir():
                continue
            metric_file = find_metric_file(case_dir)
            if metric_file is None:
                continue
            mtime = metric_file.stat().st_mtime
            case_id = case_dir.name
            if case_id not in selected or mtime > selected[case_id][0]:
                selected[case_id] = (mtime, metric_file)
    return {case_id: mf for case_id, (_, mf) in selected.items()}


def to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def mean_present(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(mean(vals)) if vals else None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def dataset_values(metrics: Dict[str, Any], dataset: str, e_source: str) -> Dict[str, Optional[float]]:
    v = metrics.get(dataset, {})
    if not isinstance(v, dict):
        v = {}
    return {
        "Sα↑": to_float(v.get("S")),
        "Eφ↑": to_float(v.get(e_source)),
        "Fwβ↑": to_float(v.get("wF")),
        "M↓": to_float(v.get("MAE")),
    }


def row_from_metrics(case_id: str, metric_file: Path, e_source: str) -> Dict[str, Any]:
    metrics = load_json(metric_file)
    row: Dict[str, Any] = {
        "case": case_id,
        "metric_file": str(metric_file),
        "Eφ_source": e_source,
    }
    buckets = {h: [] for h in METRIC_HEADERS}
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        vals = dataset_values(metrics, dataset, e_source)
        for h in METRIC_HEADERS:
            row[f"{label}.{h}"] = fmt(vals[h])
            buckets[h].append(vals[h])
    for h in METRIC_HEADERS:
        row[f"AVG.{h}"] = fmt(mean_present(buckets[h]))
    return row


def columns() -> List[str]:
    cols = ["case", "metric_file", "Eφ_source"] + [f"AVG.{h}" for h in METRIC_HEADERS]
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        cols += [f"{label}.{h}" for h in METRIC_HEADERS]
    return cols


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns()
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print(f"[shared metrics] Saved CSV: {path}")


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["case"] + [f"AVG.{h}" for h in METRIC_HEADERS]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("Eφ is selected with --e-source. Default: meanE.")
    atomic_write_text(path, "\n".join(lines))
    print(f"[shared metrics] Saved MD : {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-roots", nargs="+", required=True, type=Path)
    parser.add_argument("--save-csv", required=True, type=Path)
    parser.add_argument("--save-md", type=Path, default=None)
    parser.add_argument("--e-source", default="meanE", choices=["meanE", "adpE", "maxE"])
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    case_to_metric = discover_cases(args.output_roots, args.cases)
    rows = [row_from_metrics(case_id, case_to_metric[case_id], args.e_source) for case_id in sorted(case_to_metric, key=natural_key)]
    write_csv(args.save_csv, rows)
    if args.save_md:
        write_md(args.save_md, rows)
    print(f"[shared metrics] Cases collected: {len(rows)}")


if __name__ == "__main__":
    main()
