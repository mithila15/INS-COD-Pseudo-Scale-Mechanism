"""Auditable COD metrics and immutable per-image/curve artifacts.

The six original COD scores use the same PySODMetrics preparation and algorithms.
Extra binary metrics are sample means at threshold >= .5 on that prepared map.
Boundary extraction follows Cheng et al., CVPR 2021, official boundary-iou-api.
"""
from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

METRIC_PROTOCOL = "official_py_sod_metrics_uint8_v2"
BINARY_KEYS = ("IoU_0_5", "Dice_0_5", "F1_0_5", "Precision_0_5", "Recall_0_5",
               "Specificity_0_5", "BER_0_5", "FPR_0_5", "FNR_0_5", "Accuracy_0_5",
               "BalancedAccuracy_0_5", "BoundaryIoU_0_5")
METRIC_KEYS = ("S", "wF", "MAE", "E_adaptive", "E_mean", "E_max",
               "Fbeta_adaptive", "Fbeta_mean", "Fbeta_max", *BINARY_KEYS)
CURVE_KEYS = ("precision", "recall", "Fbeta", "E")


def protocol_details():
    return {
        "schema_version": 2,
        "pysodmetrics_version": importlib.metadata.version("pysodmetrics"),
        "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
        "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
        "binary_threshold": {"value": .5, "comparison": ">=", "input": "PySODMetrics prepared float64 map"},
        "Fbeta": {"beta_squared": .3, "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
                  "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion",
                  "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)"},
        "binary_empty_masks": {
            "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
            "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
            "recall": "no GT foreground -> 1; FNR -> 0",
            "specificity": "no GT background -> 1; FPR -> 0",
            "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
            "BalancedAccuracy": "1 - BER",
        },
        "boundary": {"dilation_ratio": .02, "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))",
                     "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
                     "empty_union": 1.0, "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP"},
        "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
        "sources": {
            "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
            "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
            "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
            "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html",
            "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
            "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
            "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
        },
    }


def official_metrics():
    try:
        from py_sod_metrics import (Emeasure, FmeasureHandler, FmeasureV2, MAE,
                                    PrecisionHandler, RecallHandler, Smeasure, WeightedFmeasure)
    except ImportError as error:
        raise RuntimeError("Test evaluation requires official py_sod_metrics (install the locked pysodmetrics package); no substitute metrics are used.") from error
    return {"S": Smeasure(), "wF": WeightedFmeasure(), "E": Emeasure(), "MAE": MAE(),
            "F": FmeasureV2({"Fbeta": FmeasureHandler(True, True, beta=.3),
                              "precision": PrecisionHandler(True, False), "recall": RecallHandler(True, False)})}


def boundary_mask(mask, dilation_ratio=.02):
    """Exact author-code erosion geometry, including objects touching the border."""
    import cv2
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("Boundary mask must be a nonempty 2D array")
    radius = max(1, int(round(dilation_ratio * math.hypot(*mask.shape))))
    padded = np.pad(mask, 1, mode="constant")
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=radius)[1:-1, 1:-1]
    return (mask - eroded).astype(bool)


def binary_metrics(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or target.ndim != 2 or not target.size:
        raise ValueError("Binary metrics require equal, nonempty 2D masks")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = target.size - tp - fp - fn
    union, foreground_sum = tp + fp + fn, 2 * tp + fp + fn
    iou = tp / union if union else 1.0
    dice = 2 * tp / foreground_sum if foreground_sum else 1.0
    precision = tp / (tp + fp) if tp + fp else float(not (tp + fn))
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    fnr, fpr = 1 - recall, 1 - specificity
    ber = .5 * (fnr + fpr)
    predicted_boundary, target_boundary = boundary_mask(prediction), boundary_mask(target)
    boundary_union = int(np.count_nonzero(predicted_boundary | target_boundary))
    boundary_iou = (int(np.count_nonzero(predicted_boundary & target_boundary)) / boundary_union
                    if boundary_union else 1.0)
    return dict(zip(BINARY_KEYS, (iou, dice, dice, precision, recall, specificity, ber, fpr, fnr,
                                  (tp + tn) / target.size, 1 - ber, boundary_iou)))


def image_metrics(prediction_uint8, target_uint8):
    """Each expensive official measure runs once; curves and per-image scores share it."""
    from py_sod_metrics.utils import prepare_data
    if prediction_uint8.dtype != np.uint8 or target_uint8.dtype != np.uint8:
        raise TypeError("Official metric input must use uint8 prediction/target")
    if prediction_uint8.shape != target_uint8.shape or prediction_uint8.ndim != 2 or not prediction_uint8.size:
        raise ValueError("Official metrics require equal nonempty 2D arrays")
    prediction, target = prepare_data(prediction_uint8, target_uint8)
    metrics = official_metrics()
    for metric in metrics.values():
        metric.step(prediction, target, normalize=False)
    e = metrics["E"].get_results()["em"]
    f = metrics["F"].get_results()
    # Library arrays are ordered 255..0. Public artifacts consistently use 0..255.
    curves = {"precision": f["precision"]["dynamic"][::-1], "recall": f["recall"]["dynamic"][::-1],
              "Fbeta": f["Fbeta"]["dynamic"][::-1], "E": e["curve"][::-1]}
    values = {"S": float(metrics["S"].get_results()["sm"]),
              "wF": float(metrics["wF"].get_results()["wfm"]),
              "MAE": float(metrics["MAE"].get_results()["mae"]),
              "E_adaptive": float(e["adp"]), "E_mean": float(np.mean(e["curve"])),
              "E_max": float(np.max(e["curve"])), "Fbeta_adaptive": float(f["Fbeta"]["adaptive"]),
              "Fbeta_mean": float(np.mean(f["Fbeta"]["dynamic"])),
              "Fbeta_max": float(np.max(f["Fbeta"]["dynamic"])),
              **binary_metrics(prediction >= .5, target)}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Official metric computation produced a nonfinite value")
    if any(curve.shape != (256,) or not np.isfinite(curve).all() for curve in curves.values()):
        raise FloatingPointError("Official metric curve must contain 256 finite thresholds")
    return values, curves


def _artifact_name(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:64] or "sample"
    return f"{label}-{hashlib.sha256(str(value).encode()).hexdigest()}"


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, x, y, title, xlabel, ylabel, xmax=1.0):
    """Standalone SVG + PNG; coordinates come directly from the exported CSV."""
    width, height = 720, 520
    left, top, right, bottom = 85, 55, 685, 445
    ymax = max(1., float(np.max(y)))
    points = [(left + float(a) / xmax * (right - left), bottom - float(b) / ymax * (bottom - top))
              for a, b in zip(x, y)]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#17202a">']
    picture = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(picture)
    def label(px, py, text, anchor="middle"):
        svg.append(f'<text x="{px}" y="{py}" text-anchor="{anchor}">{html.escape(str(text))}</text>')
        # Pillow's bundled font needs no OS font dependency.
        draw.text((px, py), str(text), fill="#17202a", anchor={"middle": "mm", "end": "rm", "start": "lm"}[anchor])
    for fraction in np.linspace(0, 1, 6):
        px = left + fraction * (right - left)
        py = bottom - fraction * (bottom - top)
        svg.append(f'<path d="M {px} {top} V {bottom} M {left} {py} H {right}" stroke="#e5e7eb" fill="none"/>')
        draw.line((px, top, px, bottom), fill="#e5e7eb")
        draw.line((left, py, right, py), fill="#e5e7eb")
        label(px, bottom + 24, f"{fraction * xmax:.2f}" if xmax == 1 else f"{fraction * xmax:.0f}")
        label(left - 12, py + 4, f"{fraction * ymax:.2f}", "end")
    svg.append(f'<path d="M {left} {top} V {bottom} H {right}" stroke="#17202a" fill="none"/>')
    draw.line((left, top, left, bottom, right, bottom), fill="#17202a", width=2)
    label((left + right) / 2, 27, title)
    label((left + right) / 2, height - 20, xlabel)
    label(left, top - 14, ylabel, "start")
    svg.append('<polyline points="' + " ".join(f"{a:.5f},{b:.5f}" for a, b in points) + '" fill="none" stroke="#1967d2" stroke-width="2"/>')
    draw.line(points, fill="#1967d2", width=2)
    svg.extend(["</g>", "</svg>"])
    with path.with_suffix(".svg").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(svg) + "\n")
    with path.with_suffix(".png").open("xb") as stream:
        picture.save(stream, format="PNG")


class MetricRecorder:
    """One evaluation's image scores and curves; artifact files are never overwritten."""
    def __init__(self, output=None, save_predictions=True):
        official_metrics()  # Check the dependency before any samples/artifacts.
        self.output = Path(output) if output is not None else None
        self.save_predictions = save_predictions
        self.samples, self.curves, self.rows, self.ids = {}, {}, [], set()

    def step(self, prediction, target, dataset, sample_id, mask_path=""):
        key = (dataset, sample_id)
        if key in self.ids:
            raise ValueError(f"Duplicate evaluation sample ID: {dataset}/{sample_id}")
        self.ids.add(key)
        values, curves = image_metrics(prediction, target)
        self.samples.setdefault(dataset, []).append(values)
        self.curves.setdefault(dataset, []).append(curves)
        prediction_path = ""
        if self.output is not None and self.save_predictions:
            relative = Path("predictions") / _artifact_name(dataset) / f"{_artifact_name(sample_id)}.png"
            path = self.output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                Image.fromarray(prediction).save(stream, format="PNG")
            prediction_path = relative.as_posix()
        self.rows.append({"dataset": dataset, "id": sample_id, "mask_path": str(mask_path),
                          "prediction_path": prediction_path, "height": target.shape[0], "width": target.shape[1], **values})

    def finish(self):
        if not self.samples:
            raise ValueError("Cannot finalize empty evaluation metrics")
        results, artifacts = {}, {"predictions_saved": self.output is not None and self.save_predictions,
                                  "per_image_metrics": None, "curves": {}}
        for dataset, samples in sorted(self.samples.items()):
            means = {key: float(np.mean([row[key] for row in samples], dtype=np.float64)) for key in METRIC_KEYS}
            curves = {key: np.mean(np.asarray([row[key] for row in self.curves[dataset]], dtype=np.float64), axis=0)
                      for key in CURVE_KEYS}
            # Protocol: max of dataset-averaged curve, never average of per-image maxima.
            for metric in ("E", "Fbeta"):
                means[f"{metric}_mean"] = float(curves[metric][::-1].mean())
                means[f"{metric}_max"] = float(curves[metric].max())
            results[dataset] = {"count": len(samples), **means}
            if self.output is None:
                continue
            relative = Path("curves") / _artifact_name(dataset)
            directory = self.output / relative
            _write_csv(directory / "threshold_curves.csv", ["threshold_uint8", "threshold_0_1", *CURVE_KEYS],
                       [{"threshold_uint8": t, "threshold_0_1": t / 255,
                         **{key: float(curves[key][t]) for key in CURVE_KEYS}} for t in range(256)])
            _plot(directory / "pr", curves["recall"], curves["precision"], f"{dataset}: precision-recall", "Recall", "Precision")
            _plot(directory / "fbeta", np.arange(256), curves["Fbeta"], f"{dataset}: F-beta (beta squared = 0.3)", "Threshold (uint8)", "F-beta", 255)
            _plot(directory / "emeasure", np.arange(256), curves["E"], f"{dataset}: E-measure", "Threshold (uint8)", "E-measure", 255)
            artifacts["curves"][dataset] = {
                "csv": (relative / "threshold_curves.csv").as_posix(),
                **{kind: {ext: (relative / f"{kind}.{ext}").as_posix() for ext in ("svg", "png")}
                   for kind in ("pr", "fbeta", "emeasure")}}
        if self.output is not None:
            _write_csv(self.output / "per_image_metrics.csv",
                       ["dataset", "id", "mask_path", "prediction_path", "height", "width", *METRIC_KEYS], self.rows)
            artifacts["per_image_metrics"] = "per_image_metrics.csv"
            files = [self.output / artifacts["per_image_metrics"]]
            files += [path for group in (self.output / "curves", self.output / "predictions") if group.exists()
                      for path in group.rglob("*") if path.is_file()]
            artifacts["sha256"] = {path.relative_to(self.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(files)}
        return results, artifacts
