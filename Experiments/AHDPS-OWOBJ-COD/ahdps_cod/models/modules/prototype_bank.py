from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DualPrototypeBank(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.95, eps: float = 1e-4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer("fg_mean", torch.zeros(self.dim))
        self.register_buffer("fg_var", torch.ones(self.dim))
        self.register_buffer("bg_mean", torch.zeros(self.dim))
        self.register_buffer("bg_var", torch.ones(self.dim))
        self.register_buffer("fg_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("bg_count", torch.tensor(0, dtype=torch.long))

    @property
    def is_ready(self) -> bool:
        return bool(self.fg_count.item() > 0 and self.bg_count.item() > 0)

    @torch.no_grad()
    def _update_slot(self, prefix: str, embeds: torch.Tensor) -> None:
        if embeds.numel() == 0:
            return
        mean = embeds.mean(dim=0)
        var = embeds.var(dim=0, unbiased=False).clamp_min(self.eps)

        mean_buf = getattr(self, f"{prefix}_mean")
        var_buf = getattr(self, f"{prefix}_var")
        count_buf = getattr(self, f"{prefix}_count")

        if int(count_buf.item()) == 0:
            mean_buf.copy_(mean)
            var_buf.copy_(var)
        else:
            mean_buf.mul_(self.momentum).add_(mean * (1.0 - self.momentum))
            var_buf.mul_(self.momentum).add_(var * (1.0 - self.momentum))
        count_buf.add_(int(embeds.shape[0]))

    @torch.no_grad()
    def update(self, fg_embeds: torch.Tensor | None = None, bg_embeds: torch.Tensor | None = None) -> None:
        if fg_embeds is not None and fg_embeds.numel() > 0:
            self._update_slot("fg", fg_embeds.detach())
        if bg_embeds is not None and bg_embeds.numel() > 0:
            self._update_slot("bg", bg_embeds.detach())

    def mahalanobis(self, embeds: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp_min(self.eps)
        return (((embeds - mean.view(1, 1, -1)) ** 2) / var.view(1, 1, -1)).mean(dim=-1)

    def distances(self, embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d_fg = self.mahalanobis(embeds, self.fg_mean, self.fg_var)
        d_bg = self.mahalanobis(embeds, self.bg_mean, self.bg_var)
        return d_fg, d_bg

    def export(self) -> Dict[str, torch.Tensor]:
        return {
            "fg_mean": self.fg_mean.detach().clone(),
            "fg_var": self.fg_var.detach().clone(),
            "bg_mean": self.bg_mean.detach().clone(),
            "bg_var": self.bg_var.detach().clone(),
            "fg_count": self.fg_count.detach().clone(),
            "bg_count": self.bg_count.detach().clone(),
        }
