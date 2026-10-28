from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.simple_cnn import SimplePyramidBackbone
from .heads.mask_heads import ConvBlock, MaskHead
from .modules.objectness import GridObjectnessBranch
from .necks.fpn import FPN


class AHDPSCODNet(nn.Module):
    def __init__(self, cfg_model: dict) -> None:
        super().__init__()
        backbone_widths = tuple(cfg_model["backbone"]["widths"])
        neck_channels = int(cfg_model["neck"]["out_channels"])
        dec_channels = int(cfg_model["decoder"]["channels"])
        dropout = float(cfg_model["decoder"].get("dropout", 0.0))

        self.backbone = SimplePyramidBackbone(
            in_channels=int(cfg_model.get("in_channels", 3)),
            widths=backbone_widths,
        )
        self.neck = FPN(in_channels=backbone_widths, out_channels=neck_channels)

        self.coarse_fuse = ConvBlock(neck_channels * 4, dec_channels, dropout=dropout)
        self.coarse_head = nn.Conv2d(dec_channels, 1, kernel_size=1)

        self.aux_heads = nn.ModuleDict({
            "p3": MaskHead(neck_channels, dec_channels, dropout=dropout),
            "p4": MaskHead(neck_channels, dec_channels, dropout=dropout),
            "p5": MaskHead(neck_channels, dec_channels, dropout=dropout),
        })

        self.prior_logits = nn.Parameter(torch.zeros(3))

        hdps_cfg = dict(cfg_model.get("hdps", {}))
        obj_cfg = dict(cfg_model.get("objectness", {}))
        obj_cfg.setdefault("core_kernel", int(hdps_cfg.get("core_kernel", 7)))
        obj_cfg.setdefault("halo_kernel", int(hdps_cfg.get("halo_kernel", 11)))
        obj_cfg.setdefault("halo_target", float(hdps_cfg.get("halo_target", 0.20)))
        self.objectness_enabled = bool(obj_cfg.get("enabled", False))
        self.objectness_warmup_epochs = int(obj_cfg.get("warmup_epochs", 3))
        self.current_epoch = 0

        self.objectness = GridObjectnessBranch(
            cfg_objectness=obj_cfg,
            coarse_channels=dec_channels,
            aux_channels=neck_channels,
            dropout=dropout,
        ) if self.objectness_enabled else None

        self.prior_fusion_logits = nn.Parameter(torch.zeros(2))
        self.refine = ConvBlock(dec_channels + 4, dec_channels, dropout=dropout)
        self.final_head = nn.Conv2d(dec_channels, 1, kernel_size=1)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _objectness_active(self) -> bool:
        if not self.objectness_enabled:
            return False
        if not self.training:
            return True
        return self.current_epoch >= self.objectness_warmup_epochs

    def forward(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict | tuple | bool]:
        feats = self.backbone(x)
        pyramid = self.neck(feats)
        p2, p3, p4, p5 = pyramid["p2"], pyramid["p3"], pyramid["p4"], pyramid["p5"]

        p2_size = p2.shape[-2:]
        up_to_p2 = lambda t: F.interpolate(t, size=p2_size, mode="bilinear", align_corners=False)

        coarse_fused = torch.cat([p2, up_to_p2(p3), up_to_p2(p4), up_to_p2(p5)], dim=1)
        coarse_feat = self.coarse_fuse(coarse_fused)
        coarse_logits_low = self.coarse_head(coarse_feat)
        coarse_prob_low = torch.sigmoid(coarse_logits_low)

        aux_logits_low = {
            "p3": self.aux_heads["p3"](p3),
            "p4": self.aux_heads["p4"](p4),
            "p5": self.aux_heads["p5"](p5),
        }

        prior_weights = torch.softmax(self.prior_logits, dim=0)
        dense_prior_low = (
            prior_weights[0] * torch.sigmoid(up_to_p2(aux_logits_low["p3"])) +
            prior_weights[1] * torch.sigmoid(up_to_p2(aux_logits_low["p4"])) +
            prior_weights[2] * torch.sigmoid(up_to_p2(aux_logits_low["p5"]))
        )

        objectness_active = self._objectness_active()
        objectness_outputs: dict = {}
        objectness_prior_low = dense_prior_low.detach()
        hybrid_prior_low = dense_prior_low
        fusion_weights = torch.softmax(self.prior_fusion_logits, dim=0)

        if self.objectness is not None:
            objectness_outputs = self.objectness(
                coarse_feat=coarse_feat,
                p3=p3,
                coarse_prob_low=coarse_prob_low,
                dense_prior_low=dense_prior_low,
                gt_masks=masks if objectness_active else None,
                active=objectness_active,
            )
            raw_objectness_prior_low = objectness_outputs["objectness_prior_low"]
            if objectness_active:
                objectness_prior_low = raw_objectness_prior_low
            else:
                objectness_prior_low = dense_prior_low.detach()
            hybrid_prior_low = (
                fusion_weights[0] * dense_prior_low +
                fusion_weights[1] * objectness_prior_low
            )
        else:
            hybrid_prior_low = dense_prior_low

        refine_input = torch.cat(
            [coarse_feat, coarse_prob_low, dense_prior_low, objectness_prior_low, hybrid_prior_low],
            dim=1,
        )
        refine_feat = self.refine(refine_input)
        final_logits_low = self.final_head(refine_feat)

        out_size = x.shape[-2:]
        outputs = {
            "coarse_logits": F.interpolate(coarse_logits_low, size=out_size, mode="bilinear", align_corners=False),
            "final_logits": F.interpolate(final_logits_low, size=out_size, mode="bilinear", align_corners=False),
            "aux_logits": {
                "p3": F.interpolate(aux_logits_low["p3"], size=out_size, mode="bilinear", align_corners=False),
                "p4": F.interpolate(aux_logits_low["p4"], size=out_size, mode="bilinear", align_corners=False),
                "p5": F.interpolate(aux_logits_low["p5"], size=out_size, mode="bilinear", align_corners=False),
            },
            "dense_prior": F.interpolate(dense_prior_low, size=out_size, mode="bilinear", align_corners=False),
            "objectness_prior": F.interpolate(objectness_prior_low, size=out_size, mode="bilinear", align_corners=False),
            "hybrid_prior": F.interpolate(hybrid_prior_low, size=out_size, mode="bilinear", align_corners=False),
            "objectness_active": objectness_active,
            "objectness_enabled": self.objectness_enabled,
            "prior_weights": prior_weights.detach().cpu(),
            "prior_fusion_weights": fusion_weights.detach().cpu(),
        }
        outputs.update(objectness_outputs)
        return outputs
