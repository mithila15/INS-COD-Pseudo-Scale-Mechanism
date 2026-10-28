from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningMetrics:
    mae_sum: float = 0.0
    iou_sum: float = 0.0
    dice_sum: float = 0.0
    fbeta_sum: float = 0.0
    count: int = 0

    def update(self, probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        probs = probs.detach().float().cpu()
        targets = targets.detach().float().cpu()
        preds = (probs >= threshold).float()

        self.mae_sum += torch.abs(probs - targets).mean().item()

        inter = (preds * targets).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        target_sum = targets.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - inter

        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)

        precision = (inter + 1e-6) / (pred_sum + 1e-6)
        recall = (inter + 1e-6) / (target_sum + 1e-6)
        beta2 = 0.3
        fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-6)

        self.iou_sum += iou.mean().item()
        self.dice_sum += dice.mean().item()
        self.fbeta_sum += fbeta.mean().item()
        self.count += 1

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"mae": 0.0, "iou": 0.0, "dice": 0.0, "fbeta": 0.0}
        return {
            "mae": self.mae_sum / self.count,
            "iou": self.iou_sum / self.count,
            "dice": self.dice_sum / self.count,
            "fbeta": self.fbeta_sum / self.count,
        }
