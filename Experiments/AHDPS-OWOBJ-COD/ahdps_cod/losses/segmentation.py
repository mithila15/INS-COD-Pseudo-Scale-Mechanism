from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    dice_target: torch.Tensor | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    if weight_map is None:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight_map)
    dice_targets = targets if dice_target is None else dice_target
    dice = dice_loss_from_logits(logits, dice_targets)
    return bce_weight * bce + dice_weight * dice
