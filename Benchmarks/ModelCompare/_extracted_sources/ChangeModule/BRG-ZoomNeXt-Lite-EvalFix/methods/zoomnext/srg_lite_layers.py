# -*- coding: utf-8 -*-
"""Scale-residual guided modules for SRG-ZoomNeXt-Lite.

This file contains the revised modules proposed after the first BRG prototype.
Compared with the earlier ULCOD-style Po/Pe branch, the priors here are tied
more directly to ZoomNeXt compression:

1) boundary prior Pe is estimated from pseudo-scale discrepancies
   |R_l(F)-F| and |R_s(F)-F|;
2) region prior Po is estimated after pseudo-scale integration rather than from
   raw backbone features;
3) multi-level compensation uses a scale-residual gate instead of the original
   GMFFM Gate1/Gate2 design;
4) refinement is still region-guided, but it additionally injects scale residual
   cues to compensate for the removed image_l/image_s encoders.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .brg_lite_layers import DSConvBNReLU, ConvBNAct
from .ops import resize_to


class ScaleResidualIntegration(nn.Module):
    """Pseudo-scale integration with an explicit scale-residual output.

    Given pseudo large/main/small features (l, m, s), this block computes

        dl = l - m, ds = s - m
        y  = m + alpha_l * dl + alpha_s * ds
        r  = Psi(|dl|, |ds|, m)

    where y is the scale-integrated feature and r is a compact scale-residual
    descriptor used to produce boundary priors and later compensation.
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

    def forward(
        self,
        l: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Unlike the first BRG prototype, Pe is not produced by simply concatenating
    shallow and deep backbone features. Instead:

        Pe = f({r2, r3, r4, r5}),

    where ri are pseudo-scale discrepancy descriptors. The region prior is also
    generated from scale-integrated features:

        Po = g({y2, y3, y4, y5}).
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels != 4:
            raise ValueError("ScaleDiscrepancyPriorDecoder currently expects four levels.")
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

    def forward(
        self,
        features: Sequence[torch.Tensor],
        residuals: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    This replaces the earlier GMFFM-like compensation. It does not use the
    complementary Gate2 = 1 - Gate1 formulation. Instead, it forms a correction
    feature from scale residuals and cross-level context, then injects only the
    necessary correction through a learned gate:

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

    def forward(
        self,
        y: torch.Tensor,
        residual: torch.Tensor,
        others: Iterable[torch.Tensor],
        pe: torch.Tensor,
    ) -> torch.Tensor:
        pe_i = resize_to(pe, tgt_hw=y.shape[-2:]).sigmoid()
        residual = resize_to(residual, tgt_hw=y.shape[-2:])
        aux = [resize_to(o, tgt_hw=y.shape[-2:]) for o in others]
        if aux:
            aux = self.aux_fuse(torch.cat(aux, dim=1))
        else:
            aux = torch.zeros_like(y)
        correction = self.correction(torch.cat([residual, aux], dim=1))
        gate = self.gate(torch.cat([y, residual, pe_i], dim=1))
        out = y + gate * correction
        out = out * (1.0 + 0.15 * pe_i)
        return self.out(out)


class ScaleAwareRegionRefinement(nn.Module):
    """Region-guided refinement with optional scale residual injection.

    This keeps the useful coarse-to-fine region constraint of RFRM, but uses
    scale residuals as an additional correction source:

        x_i = Conv([m_i, up(r_{i+1})]) * sigmoid(Po)
        x_i = x_i + eta * gate(x_i, residual_i) * residual_i
        R_i = Conv(DSConv3(x_i) + DSConv5(x_i) + Conv1(x_i))
    """

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
