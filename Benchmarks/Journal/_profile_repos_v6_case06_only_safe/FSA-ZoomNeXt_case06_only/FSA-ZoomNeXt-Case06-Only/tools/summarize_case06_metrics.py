#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
from statistics import mean

def find_metrics(root: Path):
    files = sorted(root.rglob("metrics_best.json"))
    if not files:
        files = sorted(root.rglob("metrics_latest.json"))
    return files

def fnum(x):
    try: return float(x)
    except Exception: return None

def get_metric(d, keys):
    if not isinstance(d, dict): return None
    lower = {str(k).lower(): k for k in d.keys()}
    for key in keys:
        real = lower.get(key.lower())
        if real is not None:
            val = fnum(d[real])
            if val is not None: return val
    return None

def find_ds(obj):
    ds_names = ["chameleon", "camo_te", "cod10k_te", "nc4k"]
    if not isinstance(obj, dict): return {}
    out = {ds: obj[ds] for ds in ds_names if isinstance(obj.get(ds), dict)}
    if out: return out
    for v in obj.values():
        if isinstance(v, dict):
            nested = find_ds(v)
            if nested: return nested
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs_case06")
    ap.add_argument("--save-csv", default=None)
    ap.add_argument("--save-md", default=None)
    args = ap.parse_args()
    root = Path(args.output_root).resolve()
    files = find_metrics(root)
    if not files:
        print(f"No metrics_best.json or metrics_latest.json found under {root}")
        return
    # keep the newest metrics file
    mf = max(files, key=lambda p: p.stat().st_mtime)
    data = json.loads(mf.read_text())
    ds_metrics = find_ds(data)
    rows=[]
    vals={"Sα":[],"Eφ":[],"Fwβ":[],"M":[]}
    for ds,m in ds_metrics.items():
        row={"dataset": ds}
        row["Sα ↑"] = get_metric(m, ["S", "S_alpha", "Salpha"])
        row["Eφ ↑"] = get_metric(m, ["meanE", "E", "E_phi", "Ephi", "adpE", "maxE"])
        row["Fwβ ↑"] = get_metric(m, ["wF", "Fwβ", "Fwbeta", "weightedF", "weighted_f"])
        row["M ↓"] = get_metric(m, ["MAE", "M", "mae"])
        for k,name in [("Sα ↑","Sα"),("Eφ ↑","Eφ"),("Fwβ ↑","Fwβ"),("M ↓","M")]:
            if row[k] is not None: vals[name].append(row[k])
        rows.append(row)
    avg={"dataset":"AVG", "Sα ↑": mean(vals["Sα"]) if vals["Sα"] else None, "Eφ ↑": mean(vals["Eφ"]) if vals["Eφ"] else None, "Fwβ ↑": mean(vals["Fwβ"]) if vals["Fwβ"] else None, "M ↓": mean(vals["M"]) if vals["M"] else None}
    rows.append(avg)
    csv_path = Path(args.save_csv) if args.save_csv else root/"case06_summary_paper_metrics.csv"
    md_path = Path(args.save_md) if args.save_md else root/"case06_summary_paper_metrics.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["dataset","Sα ↑","Eφ ↑","Fwβ ↑","M ↓"])
        w.writeheader(); w.writerows(rows)
    def fmt(x): return "-" if x is None else f"{x:.4f}"
    lines=["| dataset | Sα ↑ | Eφ ↑ | Fwβ ↑ | M ↓ |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {fmt(r['Sα ↑'])} | {fmt(r['Eφ ↑'])} | {fmt(r['Fwβ ↑'])} | {fmt(r['M ↓'])} |")
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nmetrics source: {mf}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD : {md_path}")
if __name__ == "__main__":
    main()
