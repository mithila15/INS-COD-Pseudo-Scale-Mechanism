from __future__ import annotations

from typing import Dict

import numpy as np


def _as_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.clip(0.0, 1.0)


def _as_binary(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return _as_float01(arr) >= float(threshold)


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3, threshold: float = 0.5, eps: float = 1e-8) -> Dict[str, float]:
    pred_prob = _as_float01(pred)
    gt_bin = _as_binary(gt, threshold=threshold)
    pred_bin = pred_prob >= float(threshold)

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    tn = float(np.logical_and(~pred_bin, ~gt_bin).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    mae = float(np.mean(np.abs(pred_prob - gt_bin.astype(np.float32))))
    fbeta = ((1.0 + beta_sq) * precision * recall) / (beta_sq * precision + recall + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fbeta": float(fbeta),
        "mae": float(mae),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
