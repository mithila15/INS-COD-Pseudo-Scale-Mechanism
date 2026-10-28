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
from ahdps_cod.metrics import CODMetricRecorder
from ahdps_cod.utils.common import save_csv_rows, save_json
from ahdps_cod.utils.config import load_config
from ahdps_cod.utils.reporting import split_alias_to_pretty_name, write_cod_results_txt
from ahdps_cod.utils.sample_metrics import compute_sample_metrics
from ahdps_cod.utils.visualization import load_rgb_image, resize_gray_u8, save_saved_pred_panel

PRED_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def _resolve_pred(root: Path, sample_id: str) -> Path:
    for ext in PRED_EXTS:
        path = root / f"{sample_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Prediction for sample '{sample_id}' not found under: {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pred-root", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--save-sample-metrics", action="store_true")
    parser.add_argument("--max-vis-per-split", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_aliases = args.splits or list(cfg["data"].get("eval_splits", []))
    pred_root = Path(args.pred_root)
    method_name = pred_root.name

    results_root = Path(cfg["paths"]["results"].get("root", Path(cfg["project"]["root"]) / "results"))
    visualizations_root = Path(cfg["paths"]["results"].get("visualizations", results_root / "visualizations")) / f"saved_preds_{method_name}"

    all_metrics = {}
    for split_alias in split_aliases:
        split_path = ensure_split_file(cfg, split_alias, force=False)
        records = build_records_from_split(cfg, split_path)

        candidate_roots = [
            pred_root / split_alias_to_pretty_name(split_alias),
            pred_root / split_alias,
            pred_root,
        ]
        active_root = next((p for p in candidate_roots if p.exists()), None)
        if active_root is None:
            raise FileNotFoundError(f"No prediction directory found for split '{split_alias}' under {pred_root}")

        recorder = CODMetricRecorder()
        dataset_name = split_alias_to_pretty_name(split_alias)
        analysis_root = visualizations_root / dataset_name
        if args.save_panels:
            (analysis_root / "panels").mkdir(parents=True, exist_ok=True)
        sample_rows = []
        vis_saved = 0

        for record in tqdm(records, desc=f"eval-preds {split_alias}", leave=False):
            gt = np.asarray(Image.open(record.mask_path).convert("L"), dtype=np.uint8)
            pred_path = _resolve_pred(active_root, record.sample_id)
            pred = np.asarray(Image.open(pred_path).convert("L"), dtype=np.uint8)
            if pred.shape != gt.shape:
                pred = resize_gray_u8(pred, (gt.shape[0], gt.shape[1]))
            recorder.step(pred=pred, gt=gt)

            metrics = compute_sample_metrics(pred, gt)
            row = {
                "dataset_name": record.dataset_name,
                "sample_id": record.sample_id,
                "image_path": record.image_path,
                "mask_path": record.mask_path,
                "pred_path": str(pred_path),
                "final_iou": metrics["iou"],
                "final_dice": metrics["dice"],
                "final_precision": metrics["precision"],
                "final_recall": metrics["recall"],
                "final_fbeta": metrics["fbeta"],
                "final_mae": metrics["mae"],
            }
            sample_rows.append(row)

            if args.save_panels and (args.max_vis_per_split <= 0 or vis_saved < args.max_vis_per_split):
                image_rgb = load_rgb_image(record.image_path, size_hw=(gt.shape[0], gt.shape[1]))
                save_saved_pred_panel(
                    out_path=analysis_root / "panels" / f"{record.sample_id}.png",
                    image_rgb=image_rgb,
                    gt_u8=gt,
                    pred_u8=pred,
                    metrics=row,
                    sample_id=record.sample_id,
                    dataset_name=record.dataset_name,
                )
                vis_saved += 1

        if args.save_sample_metrics:
            analysis_root.mkdir(parents=True, exist_ok=True)
            save_csv_rows(sample_rows, analysis_root / "sample_metrics.csv")
            save_json(sample_rows, analysis_root / "sample_metrics.json")
        all_metrics[split_alias] = recorder.get_results()

    metrics_root = Path(cfg["paths"]["results"]["metrics"])
    save_json(all_metrics, metrics_root / f"eval_saved_preds_{method_name}.json")
    write_cod_results_txt(metrics_root / f"eval_saved_preds_{method_name}.txt", method_name, all_metrics)
    print(all_metrics)


if __name__ == "__main__":
    main()
