from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from .cod_official import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure


class CODMetricRecorder:
    """Common COD summary recorder.

    Output keys intentionally follow the naming style widely used by COD repos:
    Smeasure / wFmeasure / MAE / adpEm / meanEm / maxEm / adpFm / meanFm / maxFm.
    """

    def __init__(self) -> None:
        self.fm = Fmeasure()
        self.wfm = WeightedFmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()
        self.mae = MAE()
        self.count = 0

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)
        self.fm.step(pred=pred, gt=gt)
        self.wfm.step(pred=pred, gt=gt)
        self.sm.step(pred=pred, gt=gt)
        self.em.step(pred=pred, gt=gt)
        self.mae.step(pred=pred, gt=gt)
        self.count += 1

    def get_results(self) -> Dict[str, float]:
        fm = self.fm.get_results()["fm"]
        wfm = self.wfm.get_results()["wfm"]
        sm = self.sm.get_results()["sm"]
        em = self.em.get_results()["em"]
        mae = self.mae.get_results()["mae"]
        return OrderedDict(
            Smeasure=float(sm),
            wFmeasure=float(wfm),
            MAE=float(mae),
            adpEm=float(em["adp"]),
            meanEm=float(np.mean(em["curve"])),
            maxEm=float(np.max(em["curve"])),
            adpFm=float(fm["adp"]),
            meanFm=float(np.mean(fm["curve"])),
            maxFm=float(np.max(fm["curve"])),
        )

    def get_raw_results(self) -> Dict[str, Any]:
        return {
            "fm": self.fm.get_results(),
            "wfm": self.wfm.get_results(),
            "sm": self.sm.get_results(),
            "em": self.em.get_results(),
            "mae": self.mae.get_results(),
            "count": self.count,
        }
