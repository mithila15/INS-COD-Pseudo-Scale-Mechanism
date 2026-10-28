from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb_image(image_path: str | Path, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if size_hw is not None:
        image = image.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_gray_u8(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0.0, 255.0).astype(np.uint8)
    if tuple(arr.shape[:2]) == tuple(size_hw):
        return arr
    return np.asarray(Image.fromarray(arr).resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR), dtype=np.uint8)


def prob_to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return arr.clip(0.0, 255.0).astype(np.uint8)


def _mask_bool(arr: np.ndarray, threshold: int = 127) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = prob_to_u8(arr)
    return arr > int(threshold)


def make_error_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_b = _mask_bool(gt)
    pred_b = _mask_bool(pred)
    rgb = np.zeros(gt_b.shape + (3,), dtype=np.uint8)
    rgb[np.logical_and(gt_b, pred_b)] = (0, 200, 0)      # TP
    rgb[np.logical_and(gt_b, ~pred_b)] = (220, 30, 30)   # FN
    rgb[np.logical_and(~gt_b, pred_b)] = (30, 80, 220)   # FP
    return rgb


def make_change_rgb(gt: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    gt_b = _mask_bool(gt)
    before_b = _mask_bool(before)
    after_b = _mask_bool(after)

    before_correct = before_b == gt_b
    after_correct = after_b == gt_b
    improved = np.logical_and(~before_correct, after_correct)
    regressed = np.logical_and(before_correct, ~after_correct)

    rgb = np.zeros(gt_b.shape + (3,), dtype=np.uint8)
    rgb[np.logical_and(improved, gt_b)] = (0, 200, 0)         # recovered foreground
    rgb[np.logical_and(improved, ~gt_b)] = (0, 180, 200)      # removed false positives
    rgb[np.logical_and(regressed, gt_b)] = (220, 30, 30)      # lost foreground
    rgb[np.logical_and(regressed, ~gt_b)] = (255, 165, 0)     # introduced false positives
    return rgb


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    mask_b = _mask_bool(mask)
    out = image_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask_b] = (1.0 - alpha) * out[mask_b] + alpha * color_arr
    return out.clip(0.0, 255.0).astype(np.uint8)


def _imshow_gray(ax, arr: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_single_run_panel(
    out_path: str | Path,
    image_rgb: np.ndarray,
    gt_u8: np.ndarray,
    maps: Mapping[str, np.ndarray],
    metrics: Mapping[str, float],
    sample_id: str,
    dataset_name: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coarse = maps["coarse"]
    final = maps["final"]
    dense = maps["dense_prior"]
    obj = maps["objectness_prior"]
    hybrid = maps["hybrid_prior"]
    p3 = maps["p3"]
    p4 = maps["p4"]
    p5 = maps["p5"]
    change_rgb = make_change_rgb(gt_u8, coarse, final)
    final_error_rgb = make_error_rgb(gt_u8, final)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.reshape(3, 4)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Image", fontsize=10)
    axes[0, 0].axis("off")
    _imshow_gray(axes[0, 1], gt_u8, "GT mask")
    _imshow_gray(axes[0, 2], coarse, f"Coarse\nDice {metrics['coarse_dice']:.4f}")
    _imshow_gray(axes[0, 3], final, f"Final\nDice {metrics['final_dice']:.4f}")

    _imshow_gray(axes[1, 0], dense, f"Dense prior\nDice {metrics['dense_prior_dice']:.4f}", cmap="viridis")
    _imshow_gray(axes[1, 1], obj, f"Objectness prior\nDice {metrics['objectness_prior_dice']:.4f}", cmap="viridis")
    _imshow_gray(axes[1, 2], hybrid, f"Hybrid prior\nDice {metrics['hybrid_prior_dice']:.4f}", cmap="viridis")
    axes[1, 3].imshow(change_rgb)
    axes[1, 3].set_title("Change coarse → final\nGreen/Cyan: improved, Red/Orange: regressed", fontsize=10)
    axes[1, 3].axis("off")

    _imshow_gray(axes[2, 0], p3, f"P3\nDice {metrics['p3_dice']:.4f}")
    _imshow_gray(axes[2, 1], p4, f"P4\nDice {metrics['p4_dice']:.4f}")
    _imshow_gray(axes[2, 2], p5, f"P5\nDice {metrics['p5_dice']:.4f}")
    axes[2, 3].imshow(final_error_rgb)
    axes[2, 3].set_title("Final error map\nGreen TP / Red FN / Blue FP", fontsize=10)
    axes[2, 3].axis("off")

    fig.suptitle(
        f"{dataset_name} | {sample_id}\n"
        f"ΔDice(final-coarse)={metrics['delta_final_minus_coarse_dice']:+.4f}, "
        f"ΔIoU(final-coarse)={metrics['delta_final_minus_coarse_iou']:+.4f}, "
        f"Final MAE={metrics['final_mae']:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_compare_panel(
    out_path: str | Path,
    image_rgb: np.ndarray,
    gt_u8: np.ndarray,
    baseline_pred: np.ndarray,
    method_pred: np.ndarray,
    summary: Mapping[str, float],
    sample_id: str,
    dataset_name: str,
    baseline_label: str,
    method_label: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_error = make_error_rgb(gt_u8, baseline_pred)
    method_error = make_error_rgb(gt_u8, method_pred)
    change_rgb = make_change_rgb(gt_u8, baseline_pred, method_pred)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.reshape(2, 4)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Image", fontsize=10)
    axes[0, 0].axis("off")
    _imshow_gray(axes[0, 1], gt_u8, "GT mask")
    _imshow_gray(axes[0, 2], baseline_pred, f"{baseline_label}\nDice {summary['baseline_dice']:.4f}")
    _imshow_gray(axes[0, 3], method_pred, f"{method_label}\nDice {summary['method_dice']:.4f}")

    axes[1, 0].imshow(baseline_error)
    axes[1, 0].set_title(f"{baseline_label} error", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(method_error)
    axes[1, 1].set_title(f"{method_label} error", fontsize=10)
    axes[1, 1].axis("off")
    axes[1, 2].imshow(change_rgb)
    axes[1, 2].set_title("Change baseline → method", fontsize=10)
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    text = (
        f"ΔDice: {summary['delta_dice']:+.4f}\n"
        f"ΔIoU: {summary['delta_iou']:+.4f}\n"
        f"ΔFβ: {summary['delta_fbeta']:+.4f}\n"
        f"ΔMAE: {summary['delta_mae']:+.4f}\n"
        f"{baseline_label} Precision/Recall: {summary['baseline_precision']:.4f}/{summary['baseline_recall']:.4f}\n"
        f"{method_label} Precision/Recall: {summary['method_precision']:.4f}/{summary['method_recall']:.4f}"
    )
    axes[1, 3].text(0.02, 0.98, text, va="top", ha="left", fontsize=10)

    fig.suptitle(f"{dataset_name} | {sample_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_saved_pred_panel(
    out_path: str | Path,
    image_rgb: np.ndarray,
    gt_u8: np.ndarray,
    pred_u8: np.ndarray,
    metrics: Mapping[str, float],
    sample_id: str,
    dataset_name: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_rgb = make_error_rgb(gt_u8, pred_u8)
    overlay = make_overlay(image_rgb, pred_u8, color=(0, 255, 0), alpha=0.35)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.reshape(2, 3)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Image", fontsize=10)
    axes[0, 0].axis("off")
    _imshow_gray(axes[0, 1], gt_u8, "GT mask")
    _imshow_gray(axes[0, 2], pred_u8, f"Prediction\nDice {metrics['final_dice']:.4f}")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Prediction overlay", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(error_rgb)
    axes[1, 1].set_title("Error map\nGreen TP / Red FN / Blue FP", fontsize=10)
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    text = (
        f"IoU: {metrics['final_iou']:.4f}\n"
        f"Dice: {metrics['final_dice']:.4f}\n"
        f"Fβ: {metrics['final_fbeta']:.4f}\n"
        f"MAE: {metrics['final_mae']:.4f}\n"
        f"Precision/Recall: {metrics['final_precision']:.4f}/{metrics['final_recall']:.4f}"
    )
    axes[1, 2].text(0.02, 0.98, text, va="top", ha="left", fontsize=10)
    fig.suptitle(f"{dataset_name} | {sample_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_scatter_plot(
    out_path: str | Path,
    before_values: Sequence[float],
    after_values: Sequence[float],
    metric_name: str,
    before_label: str,
    after_label: str,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(before_values, after_values, s=8, alpha=0.7)
    axis_min = float(min([0.0, *before_values, *after_values]))
    axis_max = float(max([1.0, *before_values, *after_values]))
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1)
    ax.set_xlabel(f"{before_label} {metric_name}")
    ax.set_ylabel(f"{after_label} {metric_name}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_hist_plot(
    out_path: str | Path,
    values: Sequence[float],
    metric_name: str,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30)
    ax.axvline(float(np.mean(values) if len(values) else 0.0), linestyle="--", linewidth=1)
    ax.set_xlabel(f"Δ {metric_name}")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_bar_plot(
    out_path: str | Path,
    labels: Sequence[str],
    before_values: Sequence[float],
    after_values: Sequence[float],
    before_label: str,
    after_label: str,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, before_values, width=width, label=before_label)
    ax.bar(x + width / 2, after_values, width=width, label=after_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_train_history_plots(out_dir: str | Path, history: Sequence[Mapping[str, object]]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(history) == 0:
        return
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item.get("train_loss", 0.0)) for item in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_loss.png", dpi=160)
    plt.close(fig)

    metric_names = ["Smeasure", "wFmeasure", "MAE", "adpFm", "meanEm"]
    for metric_name in metric_names:
        values = [float(item.get("dev", {}).get(metric_name, np.nan)) for item in history]
        if all(np.isnan(v) for v in values):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, values, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Dev {metric_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"dev_{metric_name}.png", dpi=160)
        plt.close(fig)
