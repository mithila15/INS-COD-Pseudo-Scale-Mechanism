# -*- coding: utf-8 -*-
"""COD/SOD evaluation metrics.

The evaluator is designed for camouflaged object detection (COD) masks and
reports the metrics commonly used in COD/SOD papers:

- S-measure / S_alpha
- weighted F-measure / Fw_beta
- MAE
- E-measure: adaptive, mean, max
- F-measure: adaptive, mean, max
- Precision/Recall: adaptive, mean, max, plus PR curve and AUC-PR
- Optional segmentation-style metrics: IoU, Dice, F1, BER, Specificity,
  Overall Accuracy, Kappa, and MSIoU when requested

Expected input arrays are uint8 masks in [0, 255]. GT is binarized to {0, 255}.
"""
from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

try:
    import py_sod_metrics
except Exception as exc:  # pragma: no cover - depends on runtime environment
    py_sod_metrics = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Numerical metrics shown in summaries. The first block is the standard COD
# reporting block; the later blocks are optional/extra but useful for analysis.
_DEFAULT_ORDER = [
    "S",
    "wF",
    "MAE",
    "adpE",
    "meanE",
    "maxE",
    "adpF",
    "meanF",
    "maxF",
    "adpP",
    "meanP",
    "maxP",
    "adpR",
    "meanR",
    "maxR",
    "AUC_PR",
    "adpIoU",
    "meanIoU",
    "maxIoU",
    "adpDice",
    "meanDice",
    "maxDice",
]

_EXTRA_ORDER = [
    "adpF1",
    "meanF1",
    "maxF1",
    "adpBER",
    "meanBER",
    "maxBER",
    "adpSpec",
    "meanSpec",
    "maxSpec",
    "adpOA",
    "meanOA",
    "maxOA",
    "adpKappa",
    "meanKappa",
    "maxKappa",
    "MSIoU",
]

_ALL_ORDER = _DEFAULT_ORDER + _EXTRA_ORDER
_LOWER_IS_BETTER = {"MAE", "adpBER", "meanBER", "maxBER"}

_ALIAS = {
    # all/default
    "all": "all",
    "default": "default",
    "core": "core",
    # S-measure
    "s": "S",
    "sm": "S",
    "smeasure": "S",
    "s_measure": "S",
    "s_alpha": "S",
    "salpha": "S",
    "sα": "S",
    # weighted F-measure
    "wfm": "wF",
    "wf": "wF",
    "weighted_f": "wF",
    "weighted_fmeasure": "wF",
    "weighted_f_measure": "wF",
    "weightedf": "wF",
    "fw": "wF",
    "fw_beta": "wF",
    "fwβ": "wF",
    # MAE
    "mae": "MAE",
    "m": "MAE",
    # E-measure
    "e": "meanE",
    "em": "meanE",
    "ephi": "meanE",
    "e_phi": "meanE",
    "eφ": "meanE",
    "emeasure": "meanE",
    "e_measure": "meanE",
    "adpem": "adpE",
    "adpe": "adpE",
    "adaptive_e": "adpE",
    "adaptive_em": "adpE",
    "meanem": "meanE",
    "meane": "meanE",
    "avgem": "meanE",
    "avge": "meanE",
    "maxem": "maxE",
    "maxe": "maxE",
    # F-measure
    "f": "maxF",
    "fm": "maxF",
    "fmeasure": "maxF",
    "f_measure": "maxF",
    "fbeta": "maxF",
    "f_beta": "maxF",
    "fβ": "maxF",
    "adpfm": "adpF",
    "adpf": "adpF",
    "adaptive_f": "adpF",
    "adaptive_fm": "adpF",
    "meanfm": "meanF",
    "meanf": "meanF",
    "avgfm": "meanF",
    "avgf": "meanF",
    "maxfm": "maxF",
    "maxf": "maxF",
    # Precision
    "p": "meanP",
    "pre": "meanP",
    "precision": "meanP",
    "adpp": "adpP",
    "adppre": "adpP",
    "adpprecision": "adpP",
    "adaptive_precision": "adpP",
    "meanp": "meanP",
    "meanpre": "meanP",
    "meanprecision": "meanP",
    "avgp": "meanP",
    "avgpre": "meanP",
    "maxp": "maxP",
    "maxpre": "maxP",
    "maxprecision": "maxP",
    # Recall
    "r": "meanR",
    "rec": "meanR",
    "recall": "meanR",
    "adpr": "adpR",
    "adprec": "adpR",
    "adprecall": "adpR",
    "adaptive_recall": "adpR",
    "meanr": "meanR",
    "meanrec": "meanR",
    "meanrecall": "meanR",
    "avgr": "meanR",
    "avgrec": "meanR",
    "maxr": "maxR",
    "maxrec": "maxR",
    "maxrecall": "maxR",
    # AUC-PR
    "auc": "AUC_PR",
    "aucpr": "AUC_PR",
    "auc_pr": "AUC_PR",
    "aupr": "AUC_PR",
    "auprc": "AUC_PR",
    "pr_auc": "AUC_PR",
    # IoU
    "iou": "meanIoU",
    "miou": "meanIoU",
    "adpiou": "adpIoU",
    "adaptive_iou": "adpIoU",
    "meaniou": "meanIoU",
    "avgiou": "meanIoU",
    "maxiou": "maxIoU",
    # Dice
    "dice": "meanDice",
    "adpdice": "adpDice",
    "adaptive_dice": "adpDice",
    "meandice": "meanDice",
    "avgdice": "meanDice",
    "maxdice": "maxDice",
    # F1
    "f1": "meanF1",
    "adpf1": "adpF1",
    "meanf1": "meanF1",
    "avgf1": "meanF1",
    "maxf1": "maxF1",
    # BER
    "ber": "meanBER",
    "adpber": "adpBER",
    "meanber": "meanBER",
    "avgber": "meanBER",
    "maxber": "maxBER",
    # Specificity
    "spec": "meanSpec",
    "specificity": "meanSpec",
    "tnr": "meanSpec",
    "adpspec": "adpSpec",
    "meanspec": "meanSpec",
    "avgspec": "meanSpec",
    "maxspec": "maxSpec",
    # Overall Accuracy
    "oa": "meanOA",
    "overall_accuracy": "meanOA",
    "accuracy": "meanOA",
    "adpoa": "adpOA",
    "meanoa": "meanOA",
    "avgoa": "meanOA",
    "maxoa": "maxOA",
    # Kappa
    "kappa": "meanKappa",
    "adpkappa": "adpKappa",
    "meankappa": "meanKappa",
    "avgkappa": "meanKappa",
    "maxkappa": "maxKappa",
    # MSIoU
    "msiou": "MSIoU",
    "ms_iou": "MSIoU",
    "multi_scale_iou": "MSIoU",
}

_CURVE_NAMES = OrderedDict(
    [
        ("E", "E-measure"),
        ("F", "F-measure"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1"),
        ("IoU", "IoU"),
        ("Dice", "Dice"),
        ("BER", "BER"),
        ("Specificity", "Specificity"),
        ("OA", "Overall Accuracy"),
        ("Kappa", "Kappa"),
    ]
)


def _require_py_sod_metrics() -> None:
    if py_sod_metrics is None:
        raise ImportError(
            "py_sod_metrics is required for COD evaluation. Install it with `pip install pysodmetrics` "
            "or `pip install -r requirements.txt`."
        ) from _IMPORT_ERROR


def _normalize_key(name: str) -> str:
    return str(name).replace("-", "_").replace(" ", "_").strip().lower()


def canonical_metric_name(name: str) -> str:
    key = _normalize_key(name)
    return _ALIAS.get(key, name)


def canonical_metric_names(metric_names: Optional[Iterable[str]]) -> list[str]:
    if not metric_names:
        return list(_DEFAULT_ORDER)

    expanded: list[str] = []
    for name in metric_names:
        canonical = canonical_metric_name(name)
        if canonical == "all":
            candidates = list(_ALL_ORDER)
        elif canonical in {"default", "core"}:
            candidates = list(_DEFAULT_ORDER)
        else:
            candidates = [canonical]

        for candidate in candidates:
            if candidate not in _ALL_ORDER:
                raise KeyError(f"Unsupported COD metric: {name}. Supported: {_ALL_ORDER}")
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def metric_sort_key(name: str) -> int:
    try:
        return _ALL_ORDER.index(name)
    except ValueError:
        return len(_ALL_ORDER)


def is_better_metric(metric_name: str, new_value: float, best_value: Optional[float], mode: Optional[str] = None) -> bool:
    if best_value is None:
        return True
    metric_name = canonical_metric_name(metric_name.split(".")[-1])
    if mode is None or mode == "auto":
        mode = "min" if metric_name in _LOWER_IS_BETTER else "max"
    return new_value < best_value if mode == "min" else new_value > best_value


def _to_float(value) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.mean()) if arr.ndim else float(arr)


def _flip_curve(curve) -> np.ndarray:
    return np.flip(np.asarray(curve, dtype=np.float64))


def _dynamic_result(info: Mapping[str, Mapping], key: str) -> Optional[np.ndarray]:
    if key not in info:
        return None
    value = info[key].get("dynamic")
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _adaptive_result(info: Mapping[str, Mapping], key: str) -> Optional[float]:
    if key not in info:
        return None
    value = info[key].get("adaptive")
    if value is None:
        return None
    return _to_float(value)


def _summarize_dynamic(
    results: Dict[str, float],
    info: Mapping[str, Mapping],
    handler_key: str,
    output_prefix: str,
) -> None:
    curve = _dynamic_result(info, handler_key)
    if curve is not None:
        results[f"mean{output_prefix}"] = float(curve.mean())
        results[f"max{output_prefix}"] = float(curve.max())
    adaptive = _adaptive_result(info, handler_key)
    if adaptive is not None:
        results[f"adp{output_prefix}"] = adaptive


def _auc_pr_from_curves(precision_curve: np.ndarray, recall_curve: np.ndarray) -> float:
    precision = np.asarray(precision_curve, dtype=np.float64)
    recall = np.asarray(recall_curve, dtype=np.float64)
    valid = np.isfinite(precision) & np.isfinite(recall)
    if valid.sum() < 2:
        return 0.0
    precision = precision[valid]
    recall = recall[valid]
    order = np.argsort(recall, kind="stable")
    recall = recall[order]
    precision = precision[order]
    # Duplicate recall points can make trapz unstable across implementations.
    # Averaging precision per recall value gives a deterministic curve area.
    unique_recall, inverse = np.unique(recall, return_inverse=True)
    if unique_recall.size < 2:
        return 0.0
    precision_avg = np.zeros_like(unique_recall, dtype=np.float64)
    counts = np.zeros_like(unique_recall, dtype=np.float64)
    np.add.at(precision_avg, inverse, precision)
    np.add.at(counts, inverse, 1)
    precision_avg = precision_avg / np.maximum(counts, 1)
    return float(np.trapz(y=precision_avg, x=unique_recall))


def _as_jsonable_curves(curves: Mapping[str, np.ndarray], digits: Optional[int] = None) -> OrderedDict:
    output = OrderedDict()
    for key, value in curves.items():
        arr = np.asarray(value, dtype=np.float64)
        if digits is not None and arr.dtype.kind in {"f", "c"}:
            arr = np.round(arr, digits)
        output[key] = arr.tolist()
    return output



def _extract_msiou_value(msiou_obj):
    """Return one scalar MSIoU value across PySODMetrics API versions.

    Older code expected get_results()["msiou"]. Newer PySODMetrics returns keys
    such as "adaptive", "dynamic", and "binary". For the existing single
    "MSIoU" field, we use mean(dynamic) when available.
    """
    if msiou_obj is None:
        return None
    msiou_results = msiou_obj.get_results()
    if not isinstance(msiou_results, dict):
        return float(msiou_results)

    # Old or alternative APIs that already expose a scalar/array-like msiou value.
    for key in ("msiou", "MSIoU", "miou", "m_iou"):
        if key in msiou_results:
            value = np.asarray(msiou_results[key], dtype=np.float64)
            return float(np.nanmean(value))

    # Current PySODMetrics API: dynamic is the threshold curve. Use its mean
    # as the existing single MSIoU summary value.
    if "dynamic" in msiou_results:
        value = np.asarray(msiou_results["dynamic"], dtype=np.float64)
        if value.size > 0:
            return float(np.nanmean(value))

    # Fallbacks, in case only adaptive or binary variants were enabled.
    for key in ("adaptive", "binary"):
        if key in msiou_results:
            value = np.asarray(msiou_results[key], dtype=np.float64)
            return float(np.nanmean(value))

    raise KeyError(f"Cannot find MSIoU result key. Available keys: {list(msiou_results.keys())}")

class CODMetricRecorder:
    """Per-dataset COD metric recorder."""

    supported_metrics = list(_ALL_ORDER)

    def __init__(self, metric_names: Optional[Iterable[str]] = None, beta: float = 0.3) -> None:
        _require_py_sod_metrics()
        self.metric_names = canonical_metric_names(metric_names)
        self.beta = beta
        self.smeasure = py_sod_metrics.Smeasure()
        self.weighted_fmeasure = py_sod_metrics.WeightedFmeasure()
        self.mae = py_sod_metrics.MAE()
        self.emeasure = py_sod_metrics.Emeasure()
        self.fmeasure_v2 = py_sod_metrics.FmeasureV2()

        sample_gray = dict(with_dynamic=True, with_adaptive=True)
        handler_specs = OrderedDict(
            [
                ("fm", (py_sod_metrics.FmeasureHandler, dict(**sample_gray, beta=beta))),
                ("pre", (py_sod_metrics.PrecisionHandler, dict(**sample_gray))),
                ("rec", (py_sod_metrics.RecallHandler, dict(**sample_gray))),
                ("iou", (py_sod_metrics.IOUHandler, dict(**sample_gray))),
                ("dice", (py_sod_metrics.DICEHandler, dict(**sample_gray))),
                ("f1", (py_sod_metrics.FmeasureHandler, dict(**sample_gray, beta=1.0))),
                ("ber", (py_sod_metrics.BERHandler, dict(**sample_gray))),
                ("spec", (py_sod_metrics.SpecificityHandler, dict(**sample_gray))),
                ("oa", (py_sod_metrics.OverallAccuracyHandler, dict(**sample_gray))),
                ("kappa", (py_sod_metrics.KappaHandler, dict(**sample_gray))),
            ]
        )
        for handler_name, (handler_cls, kwargs) in handler_specs.items():
            self.fmeasure_v2.add_handler(handler_name=handler_name, metric_handler=handler_cls(**kwargs))

        self.msiou = None

        if hasattr(py_sod_metrics, "MSIoU"):

            try:

                self.msiou = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True, with_binary=True)

            except TypeError:

                try:

                    self.msiou = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True)

                except TypeError:

                    self.msiou = None

                    if hasattr(py_sod_metrics, "MSIoU"):

                        try:

                            self.msiou = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True, with_binary=True)

                        except TypeError:

                            try:

                                self.msiou = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True)

                            except TypeError:

                                self.msiou = py_sod_metrics.MSIoU()
    def step(self, pre: np.ndarray, gt: np.ndarray, gt_path: str = "") -> None:
        if pre.shape != gt.shape:
            raise AssertionError(f"Prediction/GT shape mismatch: {pre.shape} vs {gt.shape}. GT: {gt_path}")
        if pre.dtype != np.uint8 or gt.dtype != np.uint8:
            raise AssertionError(f"Prediction/GT dtype must be uint8: {pre.dtype}, {gt.dtype}. GT: {gt_path}")
        gt = gt.copy()
        gt[gt > 0] = 255
        for metric in (self.smeasure, self.weighted_fmeasure, self.mae, self.emeasure, self.fmeasure_v2):
            metric.step(pre, gt)
        if self.msiou is not None:
            self.msiou.step(pre, gt)

    def _collect_all_results(self) -> OrderedDict:
        results: Dict[str, float] = {}

        sm = self.smeasure.get_results()["sm"]
        wfm = self.weighted_fmeasure.get_results()["wfm"]
        mae = self.mae.get_results()["mae"]
        em_info = self.emeasure.get_results()["em"]
        fmv2_info = self.fmeasure_v2.get_results()

        em_curve = np.asarray(em_info["curve"], dtype=np.float64)
        results["S"] = float(sm)
        results["wF"] = float(wfm)
        results["MAE"] = float(mae)
        results["adpE"] = float(em_info["adp"])
        results["meanE"] = float(em_curve.mean())
        results["maxE"] = float(em_curve.max())

        _summarize_dynamic(results, fmv2_info, "fm", "F")
        _summarize_dynamic(results, fmv2_info, "pre", "P")
        _summarize_dynamic(results, fmv2_info, "rec", "R")
        _summarize_dynamic(results, fmv2_info, "iou", "IoU")
        _summarize_dynamic(results, fmv2_info, "dice", "Dice")
        _summarize_dynamic(results, fmv2_info, "f1", "F1")
        _summarize_dynamic(results, fmv2_info, "ber", "BER")
        _summarize_dynamic(results, fmv2_info, "spec", "Spec")
        _summarize_dynamic(results, fmv2_info, "oa", "OA")
        _summarize_dynamic(results, fmv2_info, "kappa", "Kappa")

        pre_curve = _dynamic_result(fmv2_info, "pre")
        rec_curve = _dynamic_result(fmv2_info, "rec")
        if pre_curve is not None and rec_curve is not None:
            results["AUC_PR"] = _auc_pr_from_curves(precision_curve=pre_curve, recall_curve=rec_curve)

        if self.msiou is not None:
            msiou_value = _extract_msiou_value(self.msiou)
            if msiou_value is not None:
                results["MSIoU"] = msiou_value
        ordered = OrderedDict()
        for name in _ALL_ORDER:
            if name in results:
                ordered[name] = float(results[name])
        return ordered

    def get_results(self, digits: Optional[int] = 4) -> OrderedDict:
        all_results = self._collect_all_results()
        ordered = OrderedDict()
        for name in _ALL_ORDER:
            if name in self.metric_names and name in all_results:
                value = all_results[name]
                ordered[name] = round(value, digits) if digits is not None else value
        return ordered

    def get_curves(self, digits: Optional[int] = None) -> OrderedDict:
        """Return threshold curves as threshold-ascending arrays.

        PySODMetrics stores dynamic threshold results as >=255, >=254, ..., >=0.
        For CSV/plot readability, this method flips them to threshold 0..255.
        """
        em_info = self.emeasure.get_results()["em"]
        fmv2_info = self.fmeasure_v2.get_results()
        curves = OrderedDict()
        curves["threshold"] = np.arange(256, dtype=np.int32)
        curves["E"] = _flip_curve(em_info["curve"])

        dynamic_map = {
            "F": "fm",
            "Precision": "pre",
            "Recall": "rec",
            "F1": "f1",
            "IoU": "iou",
            "Dice": "dice",
            "BER": "ber",
            "Specificity": "spec",
            "OA": "oa",
            "Kappa": "kappa",
        }
        for output_name, handler_key in dynamic_map.items():
            curve = _dynamic_result(fmv2_info, handler_key)
            if curve is not None:
                curves[output_name] = _flip_curve(curve)
        if digits is not None:
            return _as_jsonable_curves(curves, digits=digits)
        return curves


class GroupedCODMetricRecorder:
    """COD metric recorder grouped by dataset name."""

    supported_metrics = list(OrderedDict.fromkeys(list(_ALL_ORDER) + sorted(_ALIAS.keys())))

    def __init__(self, group_names: Optional[Iterable[str]] = None, metric_names: Optional[Iterable[str]] = None) -> None:
        self.group_names = list(group_names) if group_names is not None else None
        self.metric_names = canonical_metric_names(metric_names)
        self.metric_recorders: Dict[str, CODMetricRecorder] = {}
        if self.group_names is not None:
            for name in self.group_names:
                self.metric_recorders[name] = CODMetricRecorder(metric_names=self.metric_names)

    def step(self, group_name: str, pre: np.ndarray, gt: np.ndarray, gt_path: str = "") -> None:
        if group_name not in self.metric_recorders:
            self.metric_recorders[group_name] = CODMetricRecorder(metric_names=self.metric_names)
        self.metric_recorders[group_name].step(pre=pre, gt=gt, gt_path=gt_path)

    def show(self, digits: Optional[int] = 4, return_group: bool = False):
        group_results = OrderedDict(
            (name, recorder.get_results(digits=None)) for name, recorder in self.metric_recorders.items()
        )
        if not group_results:
            empty = OrderedDict()
            return (empty, empty) if return_group else empty

        avg = OrderedDict()
        for metric_name in self.metric_names:
            values = [metrics[metric_name] for metrics in group_results.values() if metric_name in metrics]
            if values:
                value = float(np.mean(values))
                avg[metric_name] = round(value, digits) if digits is not None else value

        if digits is not None:
            group_results = OrderedDict(
                (group, OrderedDict((k, round(v, digits)) for k, v in metrics.items()))
                for group, metrics in group_results.items()
            )
        return (avg, group_results) if return_group else avg

    def get_curves(self, digits: Optional[int] = None) -> OrderedDict:
        return OrderedDict(
            (group, recorder.get_curves(digits=digits)) for group, recorder in self.metric_recorders.items()
        )

    def save_curves(self, save_dir: str, digits: int = 6) -> None:
        """Save per-dataset curves to JSON, CSV, and PR-curve PNG.

        Files written for each dataset:
          - <dataset>_curves.json
          - <dataset>_curves.csv
          - <dataset>_pr_curve.png
          - <dataset>_f_e_curve.png
        """
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)

        for group_name, recorder in self.metric_recorders.items():
            curves = recorder.get_curves(digits=None)
            jsonable = _as_jsonable_curves(curves, digits=digits)
            json_path = os.path.join(save_dir, f"{group_name}_curves.json")
            csv_path = os.path.join(save_dir, f"{group_name}_curves.csv")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(jsonable, f, indent=2, ensure_ascii=False)

            keys = list(curves.keys())
            length = len(np.asarray(curves["threshold"]))
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for idx in range(length):
                    row = []
                    for key in keys:
                        value = np.asarray(curves[key])[idx]
                        if isinstance(value, np.generic):
                            value = value.item()
                        if isinstance(value, float):
                            value = round(value, digits)
                        row.append(value)
                    writer.writerow(row)

            self._save_curve_plots(group_name=group_name, curves=curves, save_dir=save_dir)

    @staticmethod
    def _save_curve_plots(group_name: str, curves: Mapping[str, np.ndarray], save_dir: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        # Precision-Recall curve.
        if "Precision" in curves and "Recall" in curves:
            precision = np.asarray(curves["Precision"], dtype=np.float64)
            recall = np.asarray(curves["Recall"], dtype=np.float64)
            plt.figure(figsize=(5, 4))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {group_name}")
            plt.grid(True, linewidth=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_pr_curve.png"), dpi=150)
            plt.close()

        # Threshold-dependent F/E curves.
        if "F" in curves or "E" in curves:
            threshold = np.asarray(curves["threshold"], dtype=np.float64)
            plt.figure(figsize=(5, 4))
            if "F" in curves:
                plt.plot(threshold, np.asarray(curves["F"], dtype=np.float64), label="F-measure")
            if "E" in curves:
                plt.plot(threshold, np.asarray(curves["E"], dtype=np.float64), label="E-measure")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"F/E Curves - {group_name}")
            plt.grid(True, linewidth=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_f_e_curve.png"), dpi=150)
            plt.close()
