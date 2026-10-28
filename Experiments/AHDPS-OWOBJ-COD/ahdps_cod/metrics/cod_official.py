from __future__ import annotations

"""Official-style COD metrics.

This module contains a lightweight adaptation of the core MAE / F-measure /
S-measure / E-measure / weighted F-measure implementations from the MIT-licensed
PySODMetrics project. The code here is reduced to the metrics commonly reported
in COD papers and kept self-contained so this repository does not depend on an
external evaluation package at runtime.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

EPS = np.spacing(1)
TYPE = np.float64


def prepare_data(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gt = gt > 128
    pred = pred.astype(np.float64) / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def validate_and_normalize_input(
    pred: np.ndarray,
    gt: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch between prediction ({pred.shape}) and ground truth ({gt.shape})")

    if normalize:
        return prepare_data(pred, gt)

    if pred.dtype not in (np.float32, np.float64):
        raise TypeError(f"Prediction array must be float32 or float64, got {pred.dtype}")
    if not (0 <= pred.min() and pred.max() <= 1):
        raise ValueError("Prediction values must be in range [0, 1]")
    if gt.dtype != bool:
        raise TypeError(f"Ground truth must be boolean, got {gt.dtype}")
    return pred, gt


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class Fmeasure:
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True) -> None:
        pred, gt = validate_and_normalize_input(pred, gt, normalize)
        self.adaptive_fms.append(self.cal_adaptive_fm(pred=pred, gt=gt))
        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_prediction = pred >= adaptive_threshold
        area_intersection = binary_prediction[gt].sum()
        if area_intersection == 0:
            return 0.0
        pre = area_intersection / np.count_nonzero(binary_prediction)
        rec = area_intersection / np.count_nonzero(gt)
        return float((1 + self.beta) * pre * rec / (self.beta * pre + rec + EPS))

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray):
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        tps = fg_w_thrs
        ps = fg_w_thrs + bg_w_thrs
        ps[ps == 0] = 1
        total_fg = max(np.count_nonzero(gt), 1)
        precisions = tps / ps
        recalls = tps / total_fg
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, TYPE)) if self.adaptive_fms else 0.0
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=TYPE), axis=0) if self.changeable_fms else np.zeros(256, dtype=TYPE)
        precision = np.mean(np.array(self.precisions, dtype=TYPE), axis=0) if self.precisions else np.zeros(256, dtype=TYPE)
        recall = np.mean(np.array(self.recalls, dtype=TYPE), axis=0) if self.recalls else np.zeros(256, dtype=TYPE)
        return dict(fm=dict(adp=float(adaptive_fm), curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE:
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True) -> None:
        pred, gt = validate_and_normalize_input(pred, gt, normalize)
        self.maes.append(float(np.mean(np.abs(pred - gt))))

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, TYPE)) if self.maes else 0.0
        return dict(mae=float(mae))


class Smeasure:
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True) -> None:
        pred, gt = validate_and_normalize_input(pred, gt, normalize)
        self.sms.append(float(self.cal_sm(pred, gt)))

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> np.floating:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            object_score = self.object(pred, gt) * self.alpha
            region_score = self.region(pred, gt) * (1 - self.alpha)
            sm = np.maximum(0, object_score + region_score)
        return sm

    def s_object(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        std = np.std(x, ddof=1) if x.size > 1 else 0.0
        return float(2 * mean / (np.power(mean, 2) + 1 + std + EPS))

    def object(self, pred: np.ndarray, gt: np.ndarray) -> np.floating:
        gt_mean = np.mean(gt)
        fg_score = self.s_object(pred[gt]) * gt_mean
        bg_score = self.s_object((1 - pred)[~gt]) * (1 - gt_mean)
        return fg_score + bg_score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = gt.shape
        area = h * w
        if np.count_nonzero(gt) == 0:
            cy, cx = np.round(h / 2), np.round(w / 2)
        else:
            cy, cx = np.argwhere(gt).mean(axis=0).round()
        cy, cx = int(cy) + 1, int(cx) + 1

        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        score_lt = self.ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        score_rt = self.ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        score_lb = self.ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        score_rb = self.ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        return float(score_lt + score_rt + score_lb + score_rb)

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        n = h * w
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x = np.sum((pred - x) ** 2) / (n - 1 + EPS)
        sigma_y = np.sum((gt - y) ** 2) / (n - 1 + EPS)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1 + EPS)
        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)
        if alpha != 0:
            return float(alpha / (beta + EPS))
        if alpha == 0 and beta == 0:
            return 1.0
        return 0.0

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=TYPE)) if self.sms else 0.0
        return dict(sm=float(sm))


class Emeasure:
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []
        self.gt_fg_numel = 0
        self.gt_size = 0

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True) -> None:
        pred, gt = validate_and_normalize_input(pred, gt, normalize)
        self.gt_fg_numel = int(np.count_nonzero(gt))
        self.gt_size = int(gt.shape[0] * gt.shape[1])
        self.changeable_ems.append(self.cal_changeable_em(pred, gt))
        self.adaptive_ems.append(self.cal_adaptive_em(pred, gt))

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        return float(self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold))

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return self.cal_em_with_cumsumhistogram(pred, gt)

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)
        pred_fg_numel = fg_fg_numel + fg_bg_numel
        pred_bg_numel = self.gt_size - pred_fg_numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = pred_bg_numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = pred_fg_numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=pred_fg_numel,
                pred_bg_numel=pred_bg_numel,
            )
            results_parts = []
            for part_numel, combination in zip(parts_numel, combinations):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        return float(enhanced_matrix_sum / (self.gt_size - 1 + EPS))

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)
        pred_fg_numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        pred_bg_numel_w_thrs = self.gt_size - pred_fg_numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = pred_bg_numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = pred_fg_numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=pred_fg_numel_w_thrs,
                pred_bg_numel=pred_bg_numel_w_thrs,
            )
            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        return enhanced_matrix_sum / (self.gt_size - 1 + EPS)

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel
        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size
        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value
        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=TYPE)) if self.adaptive_ems else 0.0
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=TYPE), axis=0) if self.changeable_ems else np.zeros(256, dtype=TYPE)
        return dict(em=dict(adp=float(adaptive_em), curve=changeable_em))


class WeightedFmeasure:
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True) -> None:
        pred, gt = validate_and_normalize_input(pred, gt, normalize)
        if np.all(~gt):
            wfm = 0.0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(float(wfm))

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        dst, idx = bwdist(gt == 0, return_indices=True)
        e = np.abs(pred - gt)
        et = np.copy(e)
        et[gt == 0] = et[idx[0][gt == 0], idx[1][gt == 0]]
        k = self.matlab_style_gauss2d((7, 7), sigma=5)
        ea = convolve(et, weights=k, mode="constant", cval=0)
        min_e_ea = np.where(gt & (ea < e), ea, e)
        b = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt, dtype=np.float64))
        ew = min_e_ea * b
        tpw = np.sum(gt) - np.sum(ew[gt == 1])
        fpw = np.sum(ew[gt == 0])
        r = 1 - np.mean(ew[gt == 1])
        p = tpw / (tpw + fpw + EPS)
        return float((1 + self.beta) * r * p / (r + self.beta * p + EPS))

    @staticmethod
    def matlab_style_gauss2d(shape: tuple[int, int] = (7, 7), sigma: int = 5) -> np.ndarray:
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=TYPE)) if self.weighted_fms else 0.0
        return dict(wfm=float(weighted_fm))
