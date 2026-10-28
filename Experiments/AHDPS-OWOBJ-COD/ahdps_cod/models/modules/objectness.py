from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ahdps_cod.utils.mask_targets import build_region_objectness_targets
from ahdps_cod.models.heads.mask_heads import ConvBlock

from .prototype_bank import DualPrototypeBank


class GridObjectnessBranch(nn.Module):
    def __init__(self, cfg_objectness: Dict, coarse_channels: int, aux_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.enabled = bool(cfg_objectness.get("enabled", False))
        self.grid_size = tuple(int(v) for v in cfg_objectness.get("grid_size", [8, 8]))
        self.embed_channels = int(cfg_objectness.get("embed_channels", 64))
        self.hidden_channels = int(cfg_objectness.get("hidden_channels", coarse_channels))
        self.temperature = float(cfg_objectness.get("temperature", 0.75))
        self.eps = float(cfg_objectness.get("eps", 1e-4))
        self.update_min_points = int(cfg_objectness.get("update_min_points", 4))

        self.embed = nn.Sequential(
            ConvBlock(coarse_channels + aux_channels, self.hidden_channels, dropout=dropout),
            nn.Conv2d(self.hidden_channels, self.embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_channels),
            nn.GELU(),
        )
        self.bank = DualPrototypeBank(
            dim=self.embed_channels,
            momentum=float(cfg_objectness.get("ema_momentum", 0.95)),
            eps=self.eps,
        )

        # cached config for region-target generation during prototype updates
        self.target_cfg = {
            "core_kernel": int(cfg_objectness.get("core_kernel", 7)),
            "halo_kernel": int(cfg_objectness.get("halo_kernel", 11)),
            "halo_target": float(cfg_objectness.get("halo_target", 0.2)),
            "pos_body_thresh": float(cfg_objectness.get("pos_body_thresh", 0.55)),
            "pos_core_thresh": float(cfg_objectness.get("pos_core_thresh", 0.10)),
            "neg_body_thresh": float(cfg_objectness.get("neg_body_thresh", 0.02)),
            "neg_halo_thresh": float(cfg_objectness.get("neg_halo_thresh", 0.05)),
            "amb_halo_thresh": float(cfg_objectness.get("amb_halo_thresh", 0.10)),
            "amb_boundary_thresh": float(cfg_objectness.get("amb_boundary_thresh", 0.10)),
            "amb_guidance_mix": float(cfg_objectness.get("amb_guidance_mix", 0.5)),
        }

    def _fallback_region_scores(self, coarse_prob_low: torch.Tensor, dense_prior_low: torch.Tensor) -> torch.Tensor:
        guidance = 0.5 * (coarse_prob_low.detach() + dense_prior_low.detach())
        pooled = F.adaptive_avg_pool2d(guidance, self.grid_size)
        return pooled.flatten(1).clamp(1e-4, 1.0 - 1e-4)

    @torch.no_grad()
    def _update_bank(
        self,
        region_embeds: torch.Tensor,
        gt_masks: torch.Tensor,
        coarse_prob_low: torch.Tensor,
        dense_prior_low: torch.Tensor,
    ) -> None:
        guidance = 0.5 * (coarse_prob_low.detach() + dense_prior_low.detach())
        target_info = build_region_objectness_targets(
            masks=gt_masks,
            grid_size=self.grid_size,
            guidance_map=guidance,
            **self.target_cfg,
        )
        pos_mask = target_info["positive"]
        neg_mask = target_info["negative"]

        pos_embeds = region_embeds.detach()[pos_mask]
        neg_embeds = region_embeds.detach()[neg_mask]
        if neg_embeds.shape[0] < self.update_min_points:
            relaxed_neg_mask = (~pos_mask) & (
                target_info["body"] <= self.target_cfg["neg_body_thresh"]
            ) & (
                target_info["boundary"] <= self.target_cfg["amb_boundary_thresh"]
            )
            relaxed_neg_embeds = region_embeds.detach()[relaxed_neg_mask]
            if relaxed_neg_embeds.shape[0] > neg_embeds.shape[0]:
                neg_embeds = relaxed_neg_embeds
        if pos_embeds.shape[0] >= self.update_min_points or neg_embeds.shape[0] >= self.update_min_points:
            self.bank.update(
                fg_embeds=pos_embeds if pos_embeds.shape[0] >= self.update_min_points else None,
                bg_embeds=neg_embeds if neg_embeds.shape[0] >= self.update_min_points else None,
            )

    def forward(
        self,
        coarse_feat: torch.Tensor,
        p3: torch.Tensor,
        coarse_prob_low: torch.Tensor,
        dense_prior_low: torch.Tensor,
        gt_masks: torch.Tensor | None = None,
        active: bool = True,
    ) -> Dict[str, torch.Tensor | Tuple[int, int] | bool]:
        p3_up = F.interpolate(p3, size=coarse_feat.shape[-2:], mode="bilinear", align_corners=False)
        embed_map = self.embed(torch.cat([coarse_feat, p3_up], dim=1))
        pooled = F.adaptive_avg_pool2d(embed_map, self.grid_size)
        region_embeds = pooled.flatten(2).transpose(1, 2).contiguous()  # [B, R, D]

        if active and self.bank.is_ready:
            d_fg, d_bg = self.bank.distances(region_embeds)
            region_logits = (d_bg - d_fg) / max(self.temperature, self.eps)
            region_scores = torch.sigmoid(region_logits)
            ready = True
        else:
            region_scores = self._fallback_region_scores(coarse_prob_low, dense_prior_low)
            region_logits = torch.logit(region_scores)
            d_fg = torch.zeros_like(region_scores)
            d_bg = torch.zeros_like(region_scores)
            ready = False

        objectness_grid = region_scores.view(region_scores.shape[0], 1, self.grid_size[0], self.grid_size[1])
        objectness_prior_low = F.interpolate(
            objectness_grid,
            size=coarse_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        energy = d_fg - d_bg

        if self.training and gt_masks is not None and active:
            self._update_bank(region_embeds, gt_masks, coarse_prob_low, dense_prior_low)

        stats = self.bank.export()
        return {
            "region_embeddings": region_embeds,
            "region_logits": region_logits,
            "region_scores": region_scores,
            "region_d_fg": d_fg,
            "region_d_bg": d_bg,
            "region_energy": energy,
            "objectness_prior_low": objectness_prior_low,
            "region_grid": self.grid_size,
            "prototype_ready": ready,
            "prototype_stats": stats,
        }
