# -*- coding: utf-8 -*-
"""Layers for FSA-ZoomNeXt.

FSA means Feature-Space Scale Approximation. The model removes explicit
image_l/image_s encoder forward passes and approximates their cues from the
main-scale feature pyramid.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ConvBNReLU, resize_to


class DSConvBNReLU(nn.Sequential):
    """Depthwise-separable Conv + BN + ReLU."""

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

    The original SimpleASPP global pooling branch may pass a [B,C,1,1] tensor
    through BatchNorm2d. That fails when B=1. This variant removes BN from the
    global branch while preserving the remaining ASPP branches.
    """

    def __init__(self, in_dim: int, out_dim: int, dilation: int = 3) -> None:
        super().__init__()
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

    def _branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return y0, y1, y2, y3, y4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat(self._branches(x), dim=1))


class SafeSimpleASPPWithContext(SafeSimpleASPP):
    """SafeSimpleASPP that also returns an A-context tensor.

    The branch features are the report's A values:
        A0: global scene context
        A1: local semantic context
        A2/A3/A4: progressively wider dilated contexts

    The returned context is a compact feature summary of [A0,A1,A2,A3,A4]. It is
    used by A-V experiments to weight the 1x1/3x3/5x5 refinement branches.
    """

    def __init__(self, in_dim: int, out_dim: int, dilation: int = 3) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, dilation=dilation)
        self.context_proj = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1),
            DSConvBNReLU(out_dim, out_dim, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branches = self._branches(x)
        branch_cat = torch.cat(branches, dim=1)
        fused = self.fuse(branch_cat)
        a_context = self.context_proj(branch_cat)
        return fused, a_context


class PseudoScaleGenerator(nn.Module):
    """Create pseudo large/small scale features from one main-scale feature.

    Approximation:
        E(S_alpha x) ~= R_alpha(E(x))
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
        pooled_h = max(1, h // 2)
        pooled_w = max(1, w // 2)
        small = F.adaptive_avg_pool2d(x, output_size=(pooled_h, pooled_w))
        small = resize_to(small, tgt_hw=(h, w))
        small = self.small_proj(small)
        return large, x, small


class ScaleResidualIntegration(nn.Module):
    """Pseudo-scale integration with explicit scale residual output.

        dl = l - m, ds = s - m
        y  = m + alpha_l * dl + alpha_s * ds
        r  = Psi(|dl|, |ds|, m)
    """

    def __init__(self, channels: int, gate_reduction: int = 2) -> None:
        super().__init__()
        hidden = max(8, channels // gate_reduction)
        self.gate = nn.Sequential(
            ConvBNAct(3 * channels, hidden, 1),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_proj = nn.Sequential(
            ConvBNAct(3 * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
        )
        self.out_proj = DSConvBNReLU(channels, channels, 3, 1)

    def forward(self, l: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l.shape[-2:] != m.shape[-2:]:
            l = resize_to(l, tgt_hw=m.shape[-2:])
        if s.shape[-2:] != m.shape[-2:]:
            s = resize_to(s, tgt_hw=m.shape[-2:])
        dl = l - m
        ds = s - m
        gate = self.gate(torch.cat([m, dl, ds], dim=1))
        gl, gs = gate.chunk(2, dim=1)
        y = m + gl * dl + gs * ds
        residual = self.residual_proj(torch.cat([dl.abs(), ds.abs(), m], dim=1))
        return self.out_proj(y), residual


class ScaleDiscrepancyPriorDecoder(nn.Module):
    """Generate Pe/Po from scale residuals and scale-integrated features.

    Pe is estimated from pseudo-scale discrepancies, while Po is estimated from
    scale-integrated features.
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels != 4:
            raise ValueError("ScaleDiscrepancyPriorDecoder expects four feature levels.")
        self.edge_fuse = nn.Sequential(
            ConvBNAct(num_levels * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.region_fuse = nn.Sequential(
            ConvBNAct(num_levels * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor], residuals: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(features) != 4 or len(residuals) != 4:
            raise ValueError("features/residuals must be four-level sequences: [2,3,4,5].")
        tgt_hw = features[0].shape[-2:]
        y_cat = torch.cat([resize_to(f, tgt_hw=tgt_hw) for f in features], dim=1)
        r_cat = torch.cat([resize_to(r, tgt_hw=tgt_hw) for r in residuals], dim=1)
        po = self.region_fuse(y_cat)
        pe = self.edge_fuse(r_cat)
        return po, pe


class ScaleResidualCompensationUnit(nn.Module):
    """Scale-residual guided feature compensation.

        c_i = Phi([r_i, Ctx_i])
        g_i = sigmoid(Gamma([y_i, r_i, Pe_i]))
        m_i = y_i + g_i * c_i
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        self.aux_fuse = nn.Sequential(
            ConvBNAct((num_levels - 1) * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
        )
        self.correction = nn.Sequential(
            ConvBNAct(2 * channels, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
        )
        self.gate = nn.Sequential(
            ConvBNAct(2 * channels + 1, channels, 1),
            DSConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = DSConvBNReLU(channels, channels, 3, 1)

    def forward(self, y: torch.Tensor, residual: torch.Tensor, others: Iterable[torch.Tensor], pe: torch.Tensor) -> torch.Tensor:
        pe_i = resize_to(pe, tgt_hw=y.shape[-2:]).sigmoid()
        residual = resize_to(residual, tgt_hw=y.shape[-2:])
        aux = [resize_to(o, tgt_hw=y.shape[-2:]) for o in others]
        aux = self.aux_fuse(torch.cat(aux, dim=1)) if aux else torch.zeros_like(y)
        correction = self.correction(torch.cat([residual, aux], dim=1))
        gate = self.gate(torch.cat([y, residual, pe_i], dim=1))
        out = y + gate * correction
        out = out * (1.0 + 0.15 * pe_i)
        return self.out(out)


class ScaleAwareRegionRefinement(nn.Module):
    """Region-guided refinement with optional scale residual injection."""

    def __init__(self, channels: int, residual_weight: float = 0.5) -> None:
        super().__init__()
        self.residual_weight = residual_weight
        self.pre = ConvBNAct(2 * channels, channels, 1)
        self.res_proj = ConvBNAct(channels, channels, 1)
        self.res_gate = nn.Sequential(
            ConvBNAct(2 * channels, channels, 1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.conv3 = DSConvBNReLU(channels, channels, 3, 1)
        self.conv5 = DSConvBNReLU(channels, channels, 5, 1)
        self.conv1 = ConvBNAct(channels, channels, 1)
        self.out = ConvBNAct(channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        deeper: Optional[torch.Tensor],
        po: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if deeper is None:
            deeper = torch.zeros_like(x)
        else:
            deeper = resize_to(deeper, tgt_hw=x.shape[-2:])
        po_i = resize_to(po, tgt_hw=x.shape[-2:]).sigmoid()
        y = self.pre(torch.cat([x, deeper], dim=1))
        y = y * po_i
        if residual is not None and self.residual_weight > 0:
            residual = resize_to(residual, tgt_hw=y.shape[-2:])
            gate = self.res_gate(torch.cat([y, residual], dim=1))
            y = y + self.residual_weight * gate * self.res_proj(residual)
        y = self.conv3(y) + self.conv5(y) + self.conv1(y)
        return self.out(y)


class AVGuidedRegionRefinement(ScaleAwareRegionRefinement):
    """A-guided V-branch refinement.

    Base refinement computes three V branches:
        V1 = 1x1 branch, V3 = 3x3 branch, V5 = 5x5 branch.

    The A-context from SafeSimpleASPP can be used to weight these branches:
        V = a1*V1 + a3*V3 + a5*V5

    gate_mode:
        none             : same as equal summation
        stage5           : use A-context only where it is explicitly passed
        global           : GAP(A) -> one 3-way gate per image
        spatial          : A -> spatial 3-way gate
        residual_spatial : [A, residual] -> spatial 3-way gate
    """

    def __init__(self, channels: int, residual_weight: float = 0.5, gate_mode: str = "spatial") -> None:
        super().__init__(channels=channels, residual_weight=residual_weight)
        valid_modes = {"none", "stage5", "global", "spatial", "residual_spatial"}
        if gate_mode not in valid_modes:
            raise ValueError(f"Unknown gate_mode={gate_mode}. Expected one of {sorted(valid_modes)}")
        self.gate_mode = gate_mode
        self.a_gate = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        self.ar_gate = nn.Sequential(
            ConvBNAct(2 * channels, channels, 1),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )

    def _branch_weights(
        self,
        y: torch.Tensor,
        a_context: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.gate_mode == "none" or a_context is None:
            return y.new_full((y.shape[0], 3, y.shape[2], y.shape[3]), 1.0 / 3.0)
        a_context = resize_to(a_context, tgt_hw=y.shape[-2:])
        if self.gate_mode == "global":
            gate_logits = self.a_gate(F.adaptive_avg_pool2d(a_context, output_size=1))
            gate = torch.softmax(gate_logits, dim=1)
            return gate.expand(-1, -1, y.shape[2], y.shape[3])
        if self.gate_mode == "residual_spatial" and residual is not None:
            residual = resize_to(residual, tgt_hw=y.shape[-2:])
            gate_logits = self.ar_gate(torch.cat([a_context, residual], dim=1))
        else:
            gate_logits = self.a_gate(a_context)
        return torch.softmax(gate_logits, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        deeper: Optional[torch.Tensor],
        po: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        a_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if deeper is None:
            deeper = torch.zeros_like(x)
        else:
            deeper = resize_to(deeper, tgt_hw=x.shape[-2:])
        po_i = resize_to(po, tgt_hw=x.shape[-2:]).sigmoid()
        y = self.pre(torch.cat([x, deeper], dim=1))
        y = y * po_i
        if residual is not None and self.residual_weight > 0:
            residual = resize_to(residual, tgt_hw=y.shape[-2:])
            gate = self.res_gate(torch.cat([y, residual], dim=1))
            y = y + self.residual_weight * gate * self.res_proj(residual)

        v1 = self.conv1(y)
        v3 = self.conv3(y)
        v5 = self.conv5(y)
        weights = self._branch_weights(y=y, a_context=a_context, residual=residual)
        y = weights[:, 0:1] * v1 + weights[:, 1:2] * v3 + weights[:, 2:3] * v5
        return self.out(y)


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Weighted BCE + weighted IoU for COD region supervision."""
    target = target.float()
    if logits.shape[-2:] != target.shape[-2:]:
        logits = resize_to(logits, tgt_hw=target.shape[-2:])
    weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + eps)

    pred = torch.sigmoid(logits)
    inter = ((pred * target) * weight).sum(dim=(2, 3))
    union = ((pred + target) * weight).sum(dim=(2, 3))
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
