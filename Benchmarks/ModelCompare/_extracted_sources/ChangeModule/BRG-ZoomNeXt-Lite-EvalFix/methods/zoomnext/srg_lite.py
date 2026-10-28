# -*- coding: utf-8 -*-
"""SRG-ZoomNeXt-Lite: Scale-Residual Guided ZoomNeXt compression.

This variant implements the revised design discussed after the first BRG
prototype. It is meant to be a cleaner ZoomNeXt-specific compression model:
- encoder is evaluated only on image_m;
- image_l/image_s features are approximated by pseudo-scale generators;
- boundary prior Pe comes from pseudo-scale discrepancies, not F1/F4 concat;
- region prior Po comes after scale integration, not from raw backbone features;
- feature compensation uses scale-residual gates instead of ULCOD-style GMFFM.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .brg_lite import _BRGZoomNeXtLiteBase
from .brg_lite_layers import DSConvBNReLU, PseudoScaleGenerator, SafeSimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from .srg_lite_layers import (
    ScaleAwareRegionRefinement,
    ScaleDiscrepancyPriorDecoder,
    ScaleResidualCompensationUnit,
    ScaleResidualIntegration,
)


class PvtV2B2_SRG_ZoomNeXt_Lite(_BRGZoomNeXtLiteBase):
    """Recommended revised student model.

    SRG = Scale-Residual Guided. This model keeps the main contribution as
    feature-space scale approximation while reducing direct dependence on the
    ULCOD-Net LBRD/GMFFM formulation used in the first prototype.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 1,  # kept for compatibility with the trainer
        input_norm: bool = True,
        mid_dim: int = 48,
        hmu_groups: int = 2,  # kept for compatibility with old configs
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


class PvtV2B3_SRG_ZoomNeXt_Lite(PvtV2B2_SRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_SRG_ZoomNeXt_Lite(PvtV2B2_SRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_SRG_ZoomNeXt_Lite(PvtV2B2_SRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
