# -*- coding: utf-8 -*-
"""COD evaluation metrics.

This recorder wraps the official/standard py_sod_metrics implementation and
returns the metrics most commonly reported for camouflaged object detection:
S_alpha, weighted F_beta, MAE, E-measure, and F-measure variants.

Expected input arrays are uint8 masks in [0, 255].
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Optional

import numpy as np

try:
    import py_sod_metrics
except Exception as exc:  # pragma: no cover - depends on runtime environment
    py_sod_metrics = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_ALIAS = {
    "s": "S",
    "sm": "S",
    "smeasure": "S",
    "s_alpha": "S",
    "salpha": "S",
    "wfm": "wF",
    "wf": "wF",
    "weighted_f": "wF",
    "weighted_fmeasure": "wF",
    "mae": "MAE",
    "m": "MAE",
    "adpem": "adpE",
    "adpe": "adpE",
    "adaptive_e": "adpE",
    "meanem": "meanE",
    "meane": "meanE",
    "avgem": "meanE",
    "avge": "meanE",
    "em": "meanE",
    "maxem": "maxE",
    "maxe": "maxE",
    "adpfm": "adpF",
    "adpf": "adpF",
    "adaptive_f": "adpF",
    "meanfm": "meanF",
    "meanf": "meanF",
    "avgfm": "meanF",
    "avgf": "meanF",
    "fmeasure": "maxF",
    "fm": "maxF",
    "maxfm": "maxF",
    "maxf": "maxF",
}

_DEFAULT_ORDER = ["S", "wF", "MAE", "adpE", "meanE", "maxE", "adpF", "meanF", "maxF"]
_LOWER_IS_BETTER = {"MAE"}


def _require_py_sod_metrics() -> None:
    if py_sod_metrics is None:
        raise ImportError(
            "py_sod_metrics is required for COD evaluation. Install it with `pip install pysodmetrics` "
            "or `pip install -r requirements_brg_lite.txt`."
        ) from _IMPORT_ERROR


def canonical_metric_name(name: str) -> str:
    key = str(name).replace("-", "_").strip().lower()
    return _ALIAS.get(key, name)


def canonical_metric_names(metric_names: Optional[Iterable[str]]) -> list[str]:
    if not metric_names:
        return list(_DEFAULT_ORDER)
    names: list[str] = []
    for name in metric_names:
        canonical = canonical_metric_name(name)
        if canonical not in _DEFAULT_ORDER:
            raise KeyError(f"Unsupported COD metric: {name}. Supported: {_DEFAULT_ORDER}")
        if canonical not in names:
            names.append(canonical)
    return names


def metric_sort_key(name: str) -> int:
    try:
        return _DEFAULT_ORDER.index(name)
    except ValueError:
        return len(_DEFAULT_ORDER)


def is_better_metric(metric_name: str, new_value: float, best_value: Optional[float], mode: Optional[str] = None) -> bool:
    if best_value is None:
        return True
    metric_name = canonical_metric_name(metric_name.split(".")[-1])
    if mode is None or mode == "auto":
        mode = "min" if metric_name in _LOWER_IS_BETTER else "max"
    return new_value < best_value if mode == "min" else new_value > best_value


class CODMetricRecorder:
    """Per-dataset COD metric recorder."""

    supported_metrics = list(_DEFAULT_ORDER)

    def __init__(self, metric_names: Optional[Iterable[str]] = None, beta: float = 0.3) -> None:
        _require_py_sod_metrics()
        self.metric_names = canonical_metric_names(metric_names)
        self.beta = beta
        self.smeasure = py_sod_metrics.Smeasure()
        self.weighted_fmeasure = py_sod_metrics.WeightedFmeasure()
        self.mae = py_sod_metrics.MAE()
        self.emeasure = py_sod_metrics.Emeasure()
        self.fmeasure_v2 = py_sod_metrics.FmeasureV2()
        self.fmeasure_v2.add_handler(
            handler_name="fm",
            metric_handler=py_sod_metrics.FmeasureHandler(
                with_dynamic=True,
                with_adaptive=True,
                with_binary=False,
                beta=beta,
            ),
        )

    def step(self, pre: np.ndarray, gt: np.ndarray, gt_path: str = "") -> None:
        if pre.shape != gt.shape:
            raise AssertionError(f"Prediction/GT shape mismatch: {pre.shape} vs {gt.shape}. GT: {gt_path}")
        if pre.dtype != np.uint8 or gt.dtype != np.uint8:
            raise AssertionError(f"Prediction/GT dtype must be uint8: {pre.dtype}, {gt.dtype}. GT: {gt_path}")
        gt = gt.copy()
        gt[gt > 0] = 255
        for metric in (self.smeasure, self.weighted_fmeasure, self.mae, self.emeasure, self.fmeasure_v2):
            metric.step(pre, gt)

    def get_results(self, digits: Optional[int] = 4) -> OrderedDict:
        results: Dict[str, float] = {}

        sm = self.smeasure.get_results()["sm"]
        wfm = self.weighted_fmeasure.get_results()["wfm"]
        mae = self.mae.get_results()["mae"]
        em_info = self.emeasure.get_results()["em"]
        fm_info = self.fmeasure_v2.get_results()["fm"]

        em_curve = np.asarray(em_info["curve"], dtype=np.float64)
        fm_curve = np.asarray(fm_info["dynamic"], dtype=np.float64)

        results["S"] = float(sm)
        results["wF"] = float(wfm)
        results["MAE"] = float(mae)
        results["adpE"] = float(em_info["adp"])
        results["meanE"] = float(em_curve.mean())
        results["maxE"] = float(em_curve.max())
        results["adpF"] = float(fm_info["adaptive"])
        results["meanF"] = float(fm_curve.mean())
        results["maxF"] = float(fm_curve.max())

        ordered = OrderedDict()
        for name in _DEFAULT_ORDER:
            if name in self.metric_names:
                value = results[name]
                ordered[name] = round(value, digits) if digits is not None else value
        return ordered


class GroupedCODMetricRecorder:
    """COD metric recorder grouped by dataset name."""

    supported_metrics = list(_DEFAULT_ORDER) + sorted(_ALIAS.keys())

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
