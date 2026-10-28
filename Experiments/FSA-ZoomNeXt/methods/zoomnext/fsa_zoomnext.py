# -*- coding: utf-8 -*-
"""FSA-ZoomNeXt.

FSA = Feature-Space Scale Approximation.

The model keeps the ZoomNeXt motivation of zoom-in/zoom-out scale reasoning,
but removes explicit image_l/image_s encoder forward passes. Auxiliary scale
cues are approximated from the main-scale feature pyramid and corrected through
scale-residual priors.
"""
from __future__ import annotations

import abc
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .fsa_layers import (
    DSConvBNReLU,
    PseudoScaleGenerator,
    SafeSimpleASPP,
    ScaleAwareRegionRefinement,
    ScaleDiscrepancyPriorDecoder,
    ScaleResidualCompensationUnit,
    ScaleResidualIntegration,
    dice_loss_with_logits,
    weighted_bce_iou_loss,
)
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


class _FSAZoomNeXtBase(nn.Module):
    @staticmethod
    def get_coef(iter_percentage: float = 1.0, method: str = "cos", milestones: Tuple[float, float] = (0, 1)) -> float:
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0.0, 1.0
        if iter_percentage < min_point:
            return min_coef
        if iter_percentage > max_point:
            return max_coef
        if method == "linear":
            return (max_coef - min_coef) / (max_point - min_point) * (iter_percentage - min_point)
        if method == "cos":
            perc = (iter_percentage - min_point) / (max_point - min_point)
            return float((1 - np.cos(perc * np.pi)) / 2 * (max_coef - min_coef) + min_coef)
        return max_coef

    @abc.abstractmethod
    def body(self, data: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def forward(self, data: Dict[str, torch.Tensor], iter_percentage: float = 1.0, **kwargs):
        logits, aux = self.body(data=data)
        if not self.training:
            return logits

        mask = data["mask"]
        edge = data.get("edge", None)
        prob = logits.sigmoid()
        losses: List[torch.Tensor] = []
        loss_str: List[str] = []

        loss_seg = weighted_bce_iou_loss(logits, mask)
        losses.append(loss_seg)
        loss_str.append(f"seg: {loss_seg.item():.5f}")

        deep_weight = float(getattr(self, "deep_weight", 0.25))
        if deep_weight > 0:
            deep_losses = [weighted_bce_iou_loss(p, mask) for p in aux.get("deep_logits", [])]
            if deep_losses:
                loss_deep = deep_weight * sum(deep_losses) / len(deep_losses)
                losses.append(loss_deep)
                loss_str.append(f"deep: {loss_deep.item():.5f}")

        region_weight = float(getattr(self, "region_weight", 0.5))
        if region_weight > 0 and "po" in aux:
            loss_region = region_weight * weighted_bce_iou_loss(aux["po"], mask)
            losses.append(loss_region)
            loss_str.append(f"po: {loss_region.item():.5f}")

        edge_weight = float(getattr(self, "edge_weight", 1.0))
        if edge_weight > 0 and edge is not None and "pe" in aux:
            loss_edge = edge_weight * dice_loss_with_logits(aux["pe"], edge)
            losses.append(loss_edge)
            loss_str.append(f"edge: {loss_edge.item():.5f}")

        ual_weight = float(getattr(self, "ual_weight", 1.0))
        if ual_weight > 0:
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            loss_ual = ual_weight * ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(loss_ual)
            loss_str.append(f"ual_{ual_coef:.3f}: {loss_ual.item():.5f}")

        kd_weight = float(getattr(self, "kd_weight", 0.0))
        teacher_logits = data.get("teacher_logits", None)
        if kd_weight > 0 and teacher_logits is not None:
            if teacher_logits.shape[-2:] != logits.shape[-2:]:
                teacher_logits = resize_to(teacher_logits, tgt_hw=logits.shape[-2:])
            teacher_prob = teacher_logits.detach().sigmoid()
            loss_kd = kd_weight * F.binary_cross_entropy_with_logits(logits, teacher_prob)
            losses.append(loss_kd)
            loss_str.append(f"kd: {loss_kd.item():.5f}")

        return dict(
            vis=dict(
                sal=prob,
                po=resize_to(aux["po"].sigmoid(), tgt_hw=prob.shape[-2:]) if "po" in aux else prob,
                pe=resize_to(aux["pe"].sigmoid(), tgt_hw=prob.shape[-2:]) if "pe" in aux else prob,
            ),
            loss=sum(losses),
            loss_str=" ".join(loss_str),
        )

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class PvtV2B2_FSA_ZoomNeXt(_FSAZoomNeXtBase):
    """FSA-ZoomNeXt with a PVTv2-B2 backbone."""

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 1,  # kept for compatibility with the trainer
        input_norm: bool = True,
        mid_dim: int = 48,
        hmu_groups: int = 2,  # kept for compatibility with old configs/scripts
        use_checkpoint: bool = False,
        use_compensation: bool = True,
        residual_refine_weight: float = 0.5,
        ual_weight: float = 1.0,
        edge_weight: float = 1.0,
        region_weight: float = 0.5,
        deep_weight: float = 0.25,
        kd_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.embed_dims = self.encoder.embed_dims
        self.mid_dim = mid_dim
        self.use_compensation = use_compensation
        self.ual_weight = ual_weight
        self.edge_weight = edge_weight
        self.region_weight = region_weight
        self.deep_weight = deep_weight
        self.kd_weight = kd_weight

        self.tra_5 = SafeSimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        self.ps_5 = PseudoScaleGenerator(mid_dim)
        self.ps_4 = PseudoScaleGenerator(mid_dim)
        self.ps_3 = PseudoScaleGenerator(mid_dim)
        self.ps_2 = PseudoScaleGenerator(mid_dim)

        self.sri_5 = ScaleResidualIntegration(mid_dim)
        self.sri_4 = ScaleResidualIntegration(mid_dim)
        self.sri_3 = ScaleResidualIntegration(mid_dim)
        self.sri_2 = ScaleResidualIntegration(mid_dim)

        self.prior = ScaleDiscrepancyPriorDecoder(mid_dim, num_levels=4)

        if use_compensation:
            self.comp_5 = ScaleResidualCompensationUnit(mid_dim, num_levels=4)
            self.comp_4 = ScaleResidualCompensationUnit(mid_dim, num_levels=4)
            self.comp_3 = ScaleResidualCompensationUnit(mid_dim, num_levels=4)
            self.comp_2 = ScaleResidualCompensationUnit(mid_dim, num_levels=4)

        self.refine_5 = ScaleAwareRegionRefinement(mid_dim, residual_weight=residual_refine_weight)
        self.refine_4 = ScaleAwareRegionRefinement(mid_dim, residual_weight=residual_refine_weight)
        self.refine_3 = ScaleAwareRegionRefinement(mid_dim, residual_weight=residual_refine_weight)
        self.refine_2 = ScaleAwareRegionRefinement(mid_dim, residual_weight=residual_refine_weight)

        self.deep_pred_5 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_4 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_3 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_2 = nn.Conv2d(mid_dim, 1, 1)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DSConvBNReLU(mid_dim, mid_dim, 3, 1),
        )
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DSConvBNReLU(mid_dim, 32, 3, 1),
            nn.Conv2d(32, 1, 1),
        )
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.normalizer(x)
        features = self.encoder(x)
        return (
            features["reduction_2"],
            features["reduction_3"],
            features["reduction_4"],
            features["reduction_5"],
        )

    def _integrate(self, x: torch.Tensor, ps: PseudoScaleGenerator, sri: ScaleResidualIntegration):
        l, m, s = ps(x)
        return sri(l=l, m=m, s=s)

    def _compensate(self, ys: List[torch.Tensor], residuals: List[torch.Tensor], pe: torch.Tensor) -> List[torch.Tensor]:
        if not self.use_compensation:
            return ys
        y2, y3, y4, y5 = ys
        r2, r3, r4, r5 = residuals
        c2 = self.comp_2(y2, r2, [y3, y4, y5], pe)
        c3 = self.comp_3(y3, r3, [y2, y4, y5], pe)
        c4 = self.comp_4(y4, r4, [y2, y3, y5], pe)
        c5 = self.comp_5(y5, r5, [y2, y3, y4], pe)
        return [c2, c3, c4, c5]

    def body(self, data: Dict[str, torch.Tensor]):
        m_trans_feats = self.normalize_encoder(data["image_m"])

        f2 = self.tra_2(m_trans_feats[0])
        f3 = self.tra_3(m_trans_feats[1])
        f4 = self.tra_4(m_trans_feats[2])
        f5 = self.tra_5(m_trans_feats[3])

        y5, res5 = self._integrate(f5, self.ps_5, self.sri_5)
        y4, res4 = self._integrate(f4, self.ps_4, self.sri_4)
        y3, res3 = self._integrate(f3, self.ps_3, self.sri_3)
        y2, res2 = self._integrate(f2, self.ps_2, self.sri_2)

        po, pe = self.prior([y2, y3, y4, y5], [res2, res3, res4, res5])
        y2, y3, y4, y5 = self._compensate([y2, y3, y4, y5], [res2, res3, res4, res5], pe=pe)

        r5 = self.refine_5(y5, deeper=None, po=po, residual=res5)
        r4 = self.refine_4(y4, deeper=r5, po=po, residual=res4)
        r3 = self.refine_3(y3, deeper=r4, po=po, residual=res3)
        r2 = self.refine_2(y2, deeper=r3, po=po, residual=res2)

        x = self.tra_1(r2)
        logits = self.predictor(x)
        aux = {
            "po": po,
            "pe": pe,
            "deep_logits": [self.deep_pred_5(r5), self.deep_pred_4(r4), self.deep_pred_3(r3), self.deep_pred_2(r2)],
        }
        return logits, aux


class PvtV2B3_FSA_ZoomNeXt(PvtV2B2_FSA_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_FSA_ZoomNeXt(PvtV2B2_FSA_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_FSA_ZoomNeXt(PvtV2B2_FSA_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
