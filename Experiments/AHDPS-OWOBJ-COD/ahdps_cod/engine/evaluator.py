from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ahdps_cod.metrics import CODMetricRecorder
from ahdps_cod.utils.common import save_csv_rows, save_json
from ahdps_cod.utils.sample_metrics import compute_sample_metrics
from ahdps_cod.utils.visualization import (
    load_rgb_image,
    make_change_rgb,
    make_error_rgb,
    prob_to_u8,
    resize_gray_u8,
    save_single_run_panel,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    save_dir: str | None = None,
    analysis_dir: str | None = None,
    save_debug_maps: bool = False,
    save_panels: bool = False,
    save_sample_metrics: bool = False,
    max_visualizations: int = 50,
) -> Dict[str, float]:
    model.eval()
    recorder = CODMetricRecorder()
    save_root = Path(save_dir) if save_dir is not None else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)

    analysis_root = Path(analysis_dir) if analysis_dir is not None else None
    if analysis_root is not None:
        analysis_root.mkdir(parents=True, exist_ok=True)
        if save_panels:
            (analysis_root / "panels").mkdir(parents=True, exist_ok=True)
        if save_debug_maps:
            for name in [
                "coarse",
                "final",
                "p3",
                "p4",
                "p5",
                "dense_prior",
                "objectness_prior",
                "hybrid_prior",
                "error_final",
                "change_coarse_to_final",
            ]:
                (analysis_root / "maps" / name).mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, float | str | int]] = []
    vis_saved = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)

        maps_cpu = {
            "final": torch.sigmoid(outputs["final_logits"]).detach().cpu(),
            "coarse": torch.sigmoid(outputs["coarse_logits"]).detach().cpu(),
            "p3": torch.sigmoid(outputs["aux_logits"]["p3"]).detach().cpu(),
            "p4": torch.sigmoid(outputs["aux_logits"]["p4"]).detach().cpu(),
            "p5": torch.sigmoid(outputs["aux_logits"]["p5"]).detach().cpu(),
            "dense_prior": outputs["dense_prior"].detach().cpu(),
            "objectness_prior": outputs["objectness_prior"].detach().cpu(),
            "hybrid_prior": outputs["hybrid_prior"].detach().cpu(),
        }

        sample_ids = list(batch["sample_id"])
        image_paths = list(batch["image_path"])
        mask_paths = list(batch["mask_path"])
        dataset_names = list(batch["dataset_name"])

        for idx, (sample_id, image_path, mask_path, dataset_name) in enumerate(zip(sample_ids, image_paths, mask_paths, dataset_names)):
            gt = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            gt_hw = (int(gt.shape[0]), int(gt.shape[1]))

            resized_maps = {}
            for key, tensor in maps_cpu.items():
                map_u8 = prob_to_u8(tensor[idx, 0].numpy())
                resized_maps[key] = resize_gray_u8(map_u8, gt_hw)

            final_pred = resized_maps["final"]
            recorder.step(pred=final_pred, gt=gt)

            final_metrics = compute_sample_metrics(final_pred, gt)
            coarse_metrics = compute_sample_metrics(resized_maps["coarse"], gt)
            dense_metrics = compute_sample_metrics(resized_maps["dense_prior"], gt)
            objectness_metrics = compute_sample_metrics(resized_maps["objectness_prior"], gt)
            hybrid_metrics = compute_sample_metrics(resized_maps["hybrid_prior"], gt)
            p3_metrics = compute_sample_metrics(resized_maps["p3"], gt)
            p4_metrics = compute_sample_metrics(resized_maps["p4"], gt)
            p5_metrics = compute_sample_metrics(resized_maps["p5"], gt)

            row = {
                "dataset_name": str(dataset_name),
                "sample_id": str(sample_id),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "final_iou": final_metrics["iou"],
                "final_dice": final_metrics["dice"],
                "final_precision": final_metrics["precision"],
                "final_recall": final_metrics["recall"],
                "final_fbeta": final_metrics["fbeta"],
                "final_mae": final_metrics["mae"],
                "coarse_iou": coarse_metrics["iou"],
                "coarse_dice": coarse_metrics["dice"],
                "coarse_fbeta": coarse_metrics["fbeta"],
                "coarse_mae": coarse_metrics["mae"],
                "dense_prior_iou": dense_metrics["iou"],
                "dense_prior_dice": dense_metrics["dice"],
                "dense_prior_fbeta": dense_metrics["fbeta"],
                "dense_prior_mae": dense_metrics["mae"],
                "objectness_prior_iou": objectness_metrics["iou"],
                "objectness_prior_dice": objectness_metrics["dice"],
                "objectness_prior_fbeta": objectness_metrics["fbeta"],
                "objectness_prior_mae": objectness_metrics["mae"],
                "hybrid_prior_iou": hybrid_metrics["iou"],
                "hybrid_prior_dice": hybrid_metrics["dice"],
                "hybrid_prior_fbeta": hybrid_metrics["fbeta"],
                "hybrid_prior_mae": hybrid_metrics["mae"],
                "p3_dice": p3_metrics["dice"],
                "p4_dice": p4_metrics["dice"],
                "p5_dice": p5_metrics["dice"],
                "delta_final_minus_coarse_iou": final_metrics["iou"] - coarse_metrics["iou"],
                "delta_final_minus_coarse_dice": final_metrics["dice"] - coarse_metrics["dice"],
                "delta_final_minus_coarse_fbeta": final_metrics["fbeta"] - coarse_metrics["fbeta"],
                "delta_final_minus_coarse_mae": final_metrics["mae"] - coarse_metrics["mae"],
                "delta_hybrid_minus_dense_iou": hybrid_metrics["iou"] - dense_metrics["iou"],
                "delta_hybrid_minus_dense_dice": hybrid_metrics["dice"] - dense_metrics["dice"],
                "delta_hybrid_minus_dense_fbeta": hybrid_metrics["fbeta"] - dense_metrics["fbeta"],
                "delta_hybrid_minus_dense_mae": hybrid_metrics["mae"] - dense_metrics["mae"],
            }
            sample_rows.append(row)

            if save_root is not None:
                Image.fromarray(final_pred).save(save_root / f"{sample_id}.png")

            should_visualize = analysis_root is not None and (save_panels or save_debug_maps)
            vis_allowed = should_visualize and (max_visualizations <= 0 or vis_saved < max_visualizations)
            if vis_allowed:
                image_rgb = load_rgb_image(image_path, size_hw=gt_hw)
                if save_debug_maps:
                    for map_name, map_arr in resized_maps.items():
                        Image.fromarray(map_arr).save(analysis_root / "maps" / map_name / f"{sample_id}.png")
                    error_final = make_error_rgb(gt, final_pred)
                    change_map = make_change_rgb(gt, resized_maps["coarse"], final_pred)
                    Image.fromarray(error_final).save(analysis_root / "maps" / "error_final" / f"{sample_id}.png")
                    Image.fromarray(change_map).save(analysis_root / "maps" / "change_coarse_to_final" / f"{sample_id}.png")
                if save_panels:
                    save_single_run_panel(
                        out_path=analysis_root / "panels" / f"{sample_id}.png",
                        image_rgb=image_rgb,
                        gt_u8=gt,
                        maps=resized_maps,
                        metrics=row,
                        sample_id=sample_id,
                        dataset_name=dataset_name,
                    )
                vis_saved += 1

    if analysis_root is not None and save_sample_metrics:
        save_csv_rows(sample_rows, analysis_root / "sample_metrics.csv")
        save_json(sample_rows, analysis_root / "sample_metrics.json")

    return recorder.get_results()
