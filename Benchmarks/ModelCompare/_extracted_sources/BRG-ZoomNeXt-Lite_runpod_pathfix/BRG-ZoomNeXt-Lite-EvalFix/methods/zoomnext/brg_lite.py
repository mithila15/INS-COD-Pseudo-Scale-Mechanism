# -*- coding: utf-8 -*-
"""BRG-ZoomNeXt-Lite.

A practical lightweight ZoomNeXt variant based on the analysis in this chat:
- remove image_l/image_s encoder forward passes;
- approximate zoom-in/zoom-out features in feature space;
- add ULCOD-style boundary/region priors;
- use region-guided refinement or a reduced RGPU.
"""
from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .brg_lite_layers import (
    DSConvBNReLU,
    DifferenceGatedLiteSIU,
    LightweightBoundaryRegionDecoder,
    PeGuidedGMFFM,
    PseudoScaleGenerator,
    RegionGuidedRefinement,
    dice_loss_with_logits,
    weighted_bce_iou_loss,
)
from .layers import RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


class _BRGZoomNeXtLiteBase(nn.Module):
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

        # Final prediction.
        loss_seg = weighted_bce_iou_loss(logits, mask)
        losses.append(loss_seg)
        loss_str.append(f"seg: {loss_seg.item():.5f}")

        # Deep supervision for RFRM/decoder predictions.
        deep_weight = float(getattr(self, "deep_weight", 0.25))
        if deep_weight > 0:
            deep_losses = []
            for p in aux.get("deep_logits", []):
                deep_losses.append(weighted_bce_iou_loss(p, mask))
            if deep_losses:
                loss_deep = deep_weight * sum(deep_losses) / len(deep_losses)
                losses.append(loss_deep)
                loss_str.append(f"deep: {loss_deep.item():.5f}")

        # Region prior supervision.
        region_weight = float(getattr(self, "region_weight", 0.5))
        if region_weight > 0 and "po" in aux:
            loss_region = region_weight * weighted_bce_iou_loss(aux["po"], mask)
            losses.append(loss_region)
            loss_str.append(f"po: {loss_region.item():.5f}")

        # Boundary prior supervision.
        edge_weight = float(getattr(self, "edge_weight", 1.0))
        if edge_weight > 0 and edge is not None and "pe" in aux:
            loss_edge = edge_weight * dice_loss_with_logits(aux["pe"], edge)
            losses.append(loss_edge)
            loss_str.append(f"edge: {loss_edge.item():.5f}")

        # Original ZoomNeXt UAL-like uncertainty loss.
        ual_weight = float(getattr(self, "ual_weight", 1.0))
        if ual_weight > 0:
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            loss_ual = ual_weight * ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(loss_ual)
            loss_str.append(f"ual_{ual_coef:.3f}: {loss_ual.item():.5f}")

        # Optional teacher logits for response distillation.
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


class PvtV2B2_BRG_ZoomNeXt_Lite(_BRGZoomNeXtLiteBase):
    """Recommended student model.

    The network keeps only image_m as input. image_l/image_s are ignored even if
    the dataloader provides them.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 1,
        input_norm: bool = True,
        mid_dim: int = 48,
        hmu_groups: int = 2,
        use_checkpoint: bool = False,
        use_rfrm: bool = True,
        use_gmffm: bool = True,
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
        self.use_rfrm = use_rfrm
        self.use_gmffm = use_gmffm
        self.ual_weight = ual_weight
        self.edge_weight = edge_weight
        self.region_weight = region_weight
        self.deep_weight = deep_weight
        self.kd_weight = kd_weight

        # Transform PVT features to a common lightweight width.
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        # Feature-space scale approximation and lite scale integration.
        self.ps_5 = PseudoScaleGenerator(mid_dim)
        self.ps_4 = PseudoScaleGenerator(mid_dim)
        self.ps_3 = PseudoScaleGenerator(mid_dim)
        self.ps_2 = PseudoScaleGenerator(mid_dim)
        self.siu_5 = DifferenceGatedLiteSIU(mid_dim)
        self.siu_4 = DifferenceGatedLiteSIU(mid_dim)
        self.siu_3 = DifferenceGatedLiteSIU(mid_dim)
        self.siu_2 = DifferenceGatedLiteSIU(mid_dim)

        # ULCOD-style boundary/region priors.
        self.lbrd = LightweightBoundaryRegionDecoder(mid_dim)

        if use_gmffm:
            self.gmffm_5 = PeGuidedGMFFM(mid_dim, num_levels=4)
            self.gmffm_4 = PeGuidedGMFFM(mid_dim, num_levels=4)
            self.gmffm_3 = PeGuidedGMFFM(mid_dim, num_levels=4)
            self.gmffm_2 = PeGuidedGMFFM(mid_dim, num_levels=4)

        if use_rfrm:
            self.refine_5 = RegionGuidedRefinement(mid_dim)
            self.refine_4 = RegionGuidedRefinement(mid_dim)
            self.refine_3 = RegionGuidedRefinement(mid_dim)
            self.refine_2 = RegionGuidedRefinement(mid_dim)
        else:
            # Alternative: reduced RGPU keeps closer behavior to the original ZoomNeXt decoder.
            self.refine_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
            self.refine_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
            self.refine_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
            self.refine_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.deep_pred_5 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_4 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_3 = nn.Conv2d(mid_dim, 1, 1)
        self.deep_pred_2 = nn.Conv2d(mid_dim, 1, 1)

        # Original PVT ZoomNeXt used a hard-coded 64 here. This version is mid_dim-safe.
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
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def _gmffm(self, feats: List[torch.Tensor], pe: torch.Tensor) -> List[torch.Tensor]:
        if not self.use_gmffm:
            return feats
        f2, f3, f4, f5 = feats
        out2 = self.gmffm_2(f2, [f3, f4, f5], pe)
        out3 = self.gmffm_3(f3, [f2, f4, f5], pe)
        out4 = self.gmffm_4(f4, [f2, f3, f5], pe)
        out5 = self.gmffm_5(f5, [f2, f3, f4], pe)
        return [out2, out3, out4, out5]

    def _lite_siu(self, x: torch.Tensor, ps: PseudoScaleGenerator, siu: DifferenceGatedLiteSIU, pe: torch.Tensor) -> torch.Tensor:
        l, m, s = ps(x)
        return siu(l=l, m=m, s=s, pe=pe)

    def body(self, data: Dict[str, torch.Tensor]):
        # Only image_m is used. For compatibility, if the old dataloader only provides image_l/s/m,
        # this model still works by reading image_m.
        m_trans_feats = self.normalize_encoder(data["image_m"])

        f2 = self.tra_2(m_trans_feats[0])  # 1/4, e.g. 96x96 at 384 input
        f3 = self.tra_3(m_trans_feats[1])  # 1/8
        f4 = self.tra_4(m_trans_feats[2])  # 1/16
        f5 = self.tra_5(m_trans_feats[3])  # 1/32

        po, pe = self.lbrd([f2, f3, f4, f5])
        f2, f3, f4, f5 = self._gmffm([f2, f3, f4, f5], pe=pe)

        y5 = self._lite_siu(f5, self.ps_5, self.siu_5, pe)
        y4 = self._lite_siu(f4, self.ps_4, self.siu_4, pe)
        y3 = self._lite_siu(f3, self.ps_3, self.siu_3, pe)
        y2 = self._lite_siu(f2, self.ps_2, self.siu_2, pe)

        if self.use_rfrm:
            r5 = self.refine_5(y5, deeper=None, po=po)
            r4 = self.refine_4(y4, deeper=r5, po=po)
            r3 = self.refine_3(y3, deeper=r4, po=po)
            r2 = self.refine_2(y2, deeper=r3, po=po)
        else:
            r5 = self.refine_5(y5)
            r4 = self.refine_4(y4 + resize_to(r5, tgt_hw=y4.shape[-2:]))
            r3 = self.refine_3(y3 + resize_to(r4, tgt_hw=y3.shape[-2:]))
            r2 = self.refine_2(y2 + resize_to(r3, tgt_hw=y2.shape[-2:]))

        x = self.tra_1(r2)
        logits = self.predictor(x)
        aux = {
            "po": po,
            "pe": pe,
            "deep_logits": [self.deep_pred_5(r5), self.deep_pred_4(r4), self.deep_pred_3(r3), self.deep_pred_2(r2)],
        }
        return logits, aux


class PvtV2B3_BRG_ZoomNeXt_Lite(PvtV2B2_BRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_BRG_ZoomNeXt_Lite(PvtV2B2_BRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_BRG_ZoomNeXt_Lite(PvtV2B2_BRG_ZoomNeXt_Lite):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool) -> None:
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
