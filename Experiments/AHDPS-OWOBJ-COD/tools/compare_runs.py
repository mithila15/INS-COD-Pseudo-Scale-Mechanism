from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.data.builders import build_records_from_split, ensure_split_file
from ahdps_cod.utils.common import save_csv_rows, save_json
from ahdps_cod.utils.config import load_config
from ahdps_cod.utils.reporting import split_alias_to_pretty_name
from ahdps_cod.utils.sample_metrics import compute_sample_metrics
from ahdps_cod.utils.visualization import (
    load_rgb_image,
    resize_gray_u8,
    save_bar_plot,
    save_compare_panel,
    save_hist_plot,
    save_scatter_plot,
)

PRED_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def _resolve_pred(root: Path, sample_id: str) -> Path:
    for ext in PRED_EXTS:
        path = root / f"{sample_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Prediction for sample '{sample_id}' not found under: {root}")


def _pick_active_root(pred_root: Path, split_alias: str) -> Path:
    candidates = [pred_root / split_alias_to_pretty_name(split_alias), pred_root / split_alias, pred_root]
    active_root = next((p for p in candidates if p.exists()), None)
    if active_root is None:
        raise FileNotFoundError(f"No prediction directory found for split '{split_alias}' under {pred_root}")
    return active_root


def _mean(rows, key: str) -> float:
    if len(rows) == 0:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--method-root", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--metric", type=str, default="dice", choices=["dice", "iou", "fbeta", "mae"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_aliases = args.splits or list(cfg["data"].get("eval_splits", []))
    baseline_root = Path(args.baseline_root)
    method_root = Path(args.method_root)
    compare_name = args.name or f"{method_root.name}_vs_{baseline_root.name}"

    results_root = Path(cfg["paths"]["results"].get("root", Path(cfg["project"]["root"]) / "results"))
    default_compare_root = Path(cfg["paths"]["results"].get("comparisons", results_root / "comparisons"))
    out_root = Path(args.out_root) if args.out_root is not None else default_compare_root / compare_name
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    baseline_label = baseline_root.name
    method_label = method_root.name

    summary_rows = []
    for split_alias in split_aliases:
        split_path = ensure_split_file(cfg, split_alias, force=False)
        records = build_records_from_split(cfg, split_path)
        dataset_name = split_alias_to_pretty_name(split_alias)
        baseline_active = _pick_active_root(baseline_root, split_alias)
        method_active = _pick_active_root(method_root, split_alias)

        rows = []
        for record in tqdm(records, desc=f"compare {split_alias}", leave=False):
            gt = np.asarray(Image.open(record.mask_path).convert("L"), dtype=np.uint8)
            gt_hw = (gt.shape[0], gt.shape[1])
            baseline_pred = np.asarray(Image.open(_resolve_pred(baseline_active, record.sample_id)).convert("L"), dtype=np.uint8)
            method_pred = np.asarray(Image.open(_resolve_pred(method_active, record.sample_id)).convert("L"), dtype=np.uint8)
            if baseline_pred.shape != gt.shape:
                baseline_pred = resize_gray_u8(baseline_pred, gt_hw)
            if method_pred.shape != gt.shape:
                method_pred = resize_gray_u8(method_pred, gt_hw)

            base_metrics = compute_sample_metrics(baseline_pred, gt)
            method_metrics = compute_sample_metrics(method_pred, gt)
            row = {
                "dataset_name": record.dataset_name,
                "sample_id": record.sample_id,
                "image_path": record.image_path,
                "mask_path": record.mask_path,
                "baseline_pred_path": str(_resolve_pred(baseline_active, record.sample_id)),
                "method_pred_path": str(_resolve_pred(method_active, record.sample_id)),
                "baseline_iou": base_metrics["iou"],
                "baseline_dice": base_metrics["dice"],
                "baseline_precision": base_metrics["precision"],
                "baseline_recall": base_metrics["recall"],
                "baseline_fbeta": base_metrics["fbeta"],
                "baseline_mae": base_metrics["mae"],
                "method_iou": method_metrics["iou"],
                "method_dice": method_metrics["dice"],
                "method_precision": method_metrics["precision"],
                "method_recall": method_metrics["recall"],
                "method_fbeta": method_metrics["fbeta"],
                "method_mae": method_metrics["mae"],
                "delta_iou": method_metrics["iou"] - base_metrics["iou"],
                "delta_dice": method_metrics["dice"] - base_metrics["dice"],
                "delta_fbeta": method_metrics["fbeta"] - base_metrics["fbeta"],
                "delta_mae": method_metrics["mae"] - base_metrics["mae"],
            }
            rows.append(row)

        rows_sorted = sorted(rows, key=lambda item: float(item[f"delta_{args.metric}"]), reverse=(args.metric != "mae"))
        dataset_root = out_root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        save_csv_rows(rows_sorted, dataset_root / "per_sample.csv")
        save_json(rows_sorted, dataset_root / "per_sample.json")

        summary[split_alias] = {
            "baseline_dice": _mean(rows, "baseline_dice"),
            "baseline_iou": _mean(rows, "baseline_iou"),
            "baseline_fbeta": _mean(rows, "baseline_fbeta"),
            "baseline_mae": _mean(rows, "baseline_mae"),
            "method_dice": _mean(rows, "method_dice"),
            "method_iou": _mean(rows, "method_iou"),
            "method_fbeta": _mean(rows, "method_fbeta"),
            "method_mae": _mean(rows, "method_mae"),
            "delta_dice": _mean(rows, "delta_dice"),
            "delta_iou": _mean(rows, "delta_iou"),
            "delta_fbeta": _mean(rows, "delta_fbeta"),
            "delta_mae": _mean(rows, "delta_mae"),
            "num_samples": len(rows),
        }
        summary_rows.append({"dataset": dataset_name, **summary[split_alias]})

        save_scatter_plot(
            dataset_root / f"scatter_{args.metric}.png",
            before_values=[float(row[f"baseline_{args.metric}"]) for row in rows],
            after_values=[float(row[f"method_{args.metric}"]) for row in rows],
            metric_name=args.metric,
            before_label=baseline_label,
            after_label=method_label,
            title=f"{dataset_name}: {method_label} vs {baseline_label} ({args.metric})",
        )
        save_hist_plot(
            dataset_root / f"delta_hist_{args.metric}.png",
            values=[float(row[f"delta_{args.metric}"]) for row in rows],
            metric_name=args.metric,
            title=f"{dataset_name}: Δ{args.metric} distribution",
        )
        save_bar_plot(
            dataset_root / "mean_metrics.png",
            labels=["Dice", "IoU", "Fβ"],
            before_values=[summary[split_alias]["baseline_dice"], summary[split_alias]["baseline_iou"], summary[split_alias]["baseline_fbeta"]],
            after_values=[summary[split_alias]["method_dice"], summary[split_alias]["method_iou"], summary[split_alias]["method_fbeta"]],
            before_label=baseline_label,
            after_label=method_label,
            title=f"{dataset_name}: mean metrics",
        )

        top_improved = rows_sorted[: max(0, args.topk)]
        reverse_for_regressed = (args.metric != "mae")
        regressed_sorted = sorted(rows, key=lambda item: float(item[f"delta_{args.metric}"]), reverse=not reverse_for_regressed)
        top_regressed = regressed_sorted[: max(0, args.topk)]

        for rank, row in enumerate(top_improved, start=1):
            gt = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.uint8)
            gt_hw = (gt.shape[0], gt.shape[1])
            image_rgb = load_rgb_image(row["image_path"], size_hw=gt_hw)
            baseline_pred = np.asarray(Image.open(row["baseline_pred_path"]).convert("L"), dtype=np.uint8)
            method_pred = np.asarray(Image.open(row["method_pred_path"]).convert("L"), dtype=np.uint8)
            if baseline_pred.shape != gt.shape:
                baseline_pred = resize_gray_u8(baseline_pred, gt_hw)
            if method_pred.shape != gt.shape:
                method_pred = resize_gray_u8(method_pred, gt_hw)
            save_compare_panel(
                out_path=dataset_root / "top_improved" / f"{rank:03d}_{row['sample_id']}.png",
                image_rgb=image_rgb,
                gt_u8=gt,
                baseline_pred=baseline_pred,
                method_pred=method_pred,
                summary=row,
                sample_id=row["sample_id"],
                dataset_name=row["dataset_name"],
                baseline_label=baseline_label,
                method_label=method_label,
            )

        for rank, row in enumerate(top_regressed, start=1):
            gt = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.uint8)
            gt_hw = (gt.shape[0], gt.shape[1])
            image_rgb = load_rgb_image(row["image_path"], size_hw=gt_hw)
            baseline_pred = np.asarray(Image.open(row["baseline_pred_path"]).convert("L"), dtype=np.uint8)
            method_pred = np.asarray(Image.open(row["method_pred_path"]).convert("L"), dtype=np.uint8)
            if baseline_pred.shape != gt.shape:
                baseline_pred = resize_gray_u8(baseline_pred, gt_hw)
            if method_pred.shape != gt.shape:
                method_pred = resize_gray_u8(method_pred, gt_hw)
            save_compare_panel(
                out_path=dataset_root / "top_regressed" / f"{rank:03d}_{row['sample_id']}.png",
                image_rgb=image_rgb,
                gt_u8=gt,
                baseline_pred=baseline_pred,
                method_pred=method_pred,
                summary=row,
                sample_id=row["sample_id"],
                dataset_name=row["dataset_name"],
                baseline_label=baseline_label,
                method_label=method_label,
            )

    save_json(summary, out_root / "summary.json")
    save_csv_rows(summary_rows, out_root / "summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
