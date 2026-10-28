# -*- coding: utf-8 -*-
"""Lightweight modules for BRG-ZoomNeXt-Lite.

The design goal is to keep the useful scale/refinement behavior of ZoomNeXt
while removing the expensive image_l/image_s encoder forward passes.

Main blocks:
- PseudoScaleGenerator: approximates large/small-scale features from main-scale features.
- DifferenceGatedLiteSIU: cheap replacement for MHSIU.
- LightweightBoundaryRegionDecoder: ULCOD-style Po/Pe prior generation.
- RegionGuidedRefinement: ULCOD-style region-constrained refinement.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import resize_to


class DSConvBNReLU(nn.Sequential):
    """Depthwise-separable Conv + BN + ReLU.

    This is used instead of the full 3x3 ConvBNReLU in high-resolution blocks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        modules = [
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            modules.append(nn.ReLU(inplace=True))
        super().__init__(*modules)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, padding: int = 0, act: bool = True) -> None:
        modules = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm2d(out_ch)]
        if act:
            modules.append(nn.ReLU(inplace=True))
        super().__init__(*modules)




class SafeSimpleASPP(nn.Module):
    """ASPP variant safe for batch_size=1 in training mode.

    The original SimpleASPP global pooling branch may pass a [B,C,1,1]
    tensor through BatchNorm2d. BatchNorm2d fails when B=1 because each
    channel has only one value. This Lite-only ASPP removes BatchNorm from
    that global branch while keeping the remaining branches unchanged.
    """

    def __init__(self, in_dim: int, out_dim: int, dilation: int = 3) -> None:
        super().__init__()
        from .ops import ConvBNReLU

        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1),
            ConvBNReLU(out_dim, out_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


class PseudoScaleGenerator(nn.Module):
    """Create pseudo large/small scale features from one main-scale feature.

    Mathematical approximation:
        E(S_alpha x) ~= R_alpha(E(x))

    We avoid explicitly evaluating the shared encoder on image_l and image_s.
    `large` keeps local detail through a depthwise branch; `small` injects
    low-frequency context by pooling and re-upsampling.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.large_proj = nn.Sequential(
            DSConvBNReLU(channels, channels, 3, 1),
            ConvBNAct(channels, channels, 1),
        )
        self.small_proj = nn.Sequential(
            DSConvBNReLU(channels, channels, 3, 1),
            ConvBNAct(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        large = self.large_proj(x)

        # Low-frequency / zoom-out approximation. Pooling size adapts to the feature map.
        pooled_h = max(1, h // 2)
        pooled_w = max(1, w // 2)
        small = F.adaptive_avg_pool2d(x, output_size=(pooled_h, pooled_w))
        small = resize_to(small, tgt_hw=(h, w))
        small = self.small_proj(small)
        return large, x, small


class DifferenceGatedLiteSIU(nn.Module):
    """Cheap scale integration unit.

    Original MHSIU estimates dense attention over encoded l/m/s features.
    Here we approximate scale selection with a difference form:

        Y = m + alpha_l * (l - m) + alpha_s * (s - m)

    The boundary prior Pe can optionally modulate the gate. This makes the
    removal of image_l less damaging around object boundaries.
    """

    def __init__(self, channels: int, gate_reduction: int = 2) -> None:
        super().__init__()
        hidden = max(8, channels // gate_reduction)
        self.gate = nn.Sequential(
            ConvBNAct(3 * channels, hidden, 1),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = DSConvBNReLU(channels, channels, 3, 1)

    def forward(
        self,
        l: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if l.shape[-2:] != m.shape[-2:]:
            l = resize_to(l, tgt_hw=m.shape[-2:])
        if s.shape[-2:] != m.shape[-2:]:
            s = resize_to(s, tgt_hw=m.shape[-2:])

        dl = l - m
        ds = s - m
        gate = self.gate(torch.cat([m, dl, ds], dim=1))
        gl, gs = gate.chunk(2, dim=1)
        y = m + gl * dl + gs * ds

        if pe is not None:
            pe = resize_to(pe, tgt_hw=m.shape[-2:]).sigmoid()
            y = y * (1.0 + 0.25 * pe)
        return self.out_proj(y)


class LightweightBoundaryRegionDecoder(nn.Module):
    """ULCOD-style initial region and boundary prior generator.

    Inputs are ordered from high resolution to low resolution, e.g.
        [F2(96x96), F3(48x48), F4(24x24), F5(12x12)] for PVT at 384x384.

    Outputs:
        Po: region prior logit at highest input feature resolution.
        Pe: boundary prior logit at highest input feature resolution.
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels != 4:
            raise ValueError("This implementation expects four feature levels.")
        self.align_12 = ConvBNAct(channels, channels, 1)
        self.align_23 = ConvBNAct(channels, channels, 1)
        self.align_34 = ConvBNAct(channels, channels, 1)
        self.region_pred = nn.Conv2d(channels, 1, kernel_size=1)

        self.boundary = nn.Sequential(
            ConvBNAct(2 * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        f1, f2, f3, f4 = feats
        f21 = self.align_12(f1) + resize_to(f2, tgt_hw=f1.shape[-2:])
        f32 = self.align_23(f21) + resize_to(f3, tgt_hw=f1.shape[-2:])
        f43 = self.align_34(f32) + resize_to(f4, tgt_hw=f1.shape[-2:])
        po = self.region_pred(f43)

        f4_up = resize_to(f4, tgt_hw=f1.shape[-2:])
        pe = self.boundary(torch.cat([f1, f4_up], dim=1))
        return po, pe


class PeGuidedGMFFM(nn.Module):
    """Boundary-guided multi-level fusion inspired by ULCOD-Net GMFFM.

    This is provided as a lightweight optional block. The default model uses
    LiteSIU + RFRM for minimal changes to ZoomNeXt. You can enable this module
    from the model constructor with use_gmffm=True.
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            DSConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(DSConvBNReLU(channels, channels, 3, 1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.aux_fuse = nn.Sequential(
            DSConvBNReLU((num_levels - 1) * channels, channels, 3, 1),
            ConvBNAct(channels, channels, 1),
        )
        self.out = DSConvBNReLU(channels, channels, 3, 1)

    def forward(self, x: torch.Tensor, others: Iterable[torch.Tensor], pe: torch.Tensor) -> torch.Tensor:
        pe = resize_to(pe, tgt_hw=x.shape[-2:]).sigmoid()
        x_pe = x * pe
        sa = self.spatial(x_pe)
        gate1 = self.gate(x_pe) * sa

        aux = [resize_to(o, tgt_hw=x.shape[-2:]) for o in others]
        aux = self.aux_fuse(torch.cat(aux, dim=1)) if aux else torch.zeros_like(x)
        gate2 = 1.0 - gate1
        return self.out(gate1 * x_pe + gate2 * aux + x_pe)


class RegionGuidedRefinement(nn.Module):
    """Region-constrained feature refinement block.

    It is a lightweight alternative to RGPU for top-down decoding.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = ConvBNAct(2 * channels, channels, 1)
        self.conv3 = DSConvBNReLU(channels, channels, 3, 1)
        self.conv5 = DSConvBNReLU(channels, channels, 5, 1)
        self.conv1 = ConvBNAct(channels, channels, 1)
        self.out = ConvBNAct(channels, channels, 1)

    def forward(self, x: torch.Tensor, deeper: Optional[torch.Tensor], po: torch.Tensor) -> torch.Tensor:
        if deeper is None:
            deeper = torch.zeros_like(x)
        else:
            deeper = resize_to(deeper, tgt_hw=x.shape[-2:])
        po = resize_to(po, tgt_hw=x.shape[-2:]).sigmoid()
        y = self.pre(torch.cat([x, deeper], dim=1))
        y = y * po
        y = self.conv3(y) + self.conv5(y) + self.conv1(y)
        return self.out(y)


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Weighted BCE + weighted IoU used by many COD methods."""
    target = target.float()
    if logits.shape[-2:] != target.shape[-2:]:
        logits = resize_to(logits, tgt_hw=target.shape[-2:])
    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + eps)

    pred = torch.sigmoid(logits)
    inter = ((pred * target) * weit).sum(dim=(2, 3))
    union = ((pred + target) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0 + eps)
    return (wbce + wiou).mean()


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    target = target.float()
    if logits.shape[-2:] != target.shape[-2:]:
        logits = resize_to(logits, tgt_hw=target.shape[-2:])
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0 + eps)).mean()
