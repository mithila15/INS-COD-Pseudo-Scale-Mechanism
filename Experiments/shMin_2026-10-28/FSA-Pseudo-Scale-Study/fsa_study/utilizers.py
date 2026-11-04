"""Controlled uses of aligned pseudo-scale features, with no auxiliary decoder.

Every bank is ordered ``[main, large_or_local, small_or_context]``.  All
features must already have the same B,C,H,W shape, dtype, and device.  Attention
``weights(bank)`` methods return fresh tensors and never cache live graphs.
The mathematical sources and deliberate adaptations are in docs/UTILIZATION.md.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


UTILIZER_NAMES = (
    "mean", "concat", "signed", "absolute", "relative", "normalized",
    "adjacent", "legacy_gated", "bifpn", "sk", "asff", "aff", "mhsiu",
)

# Explicit branch experiments; not silently added to the historical full grid.
EXPERIMENTAL_UTILIZER_NAMES = ("anchor_gated",)

_CITATIONS = {
    "sk": "https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html",
    "asff": "https://arxiv.org/abs/1911.09516",
    "bifpn": "https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html",
    "aff": "https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html",
    "mhsiu": "https://doi.org/10.1109/TPAMI.2024.3417329",
    "legacy_gated": "../../../FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py",
}


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_eps(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("eps must be finite and strictly positive")
    return value


def _local_norm(channels: int) -> nn.Module:
    """GN with at least two channels/group; safe at B=H=W=1.

    A single-channel local path uses Identity because no grouping can satisfy
    this condition. Pooled global paths deliberately do not use BN or GN.
    """
    if channels == 1:
        return nn.Identity()
    groups = max(g for g in range(1, min(8, channels // 2) + 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class _Utilizer(nn.Module):
    def __init__(self, name: str, channels: int, num_branches: int = 3) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.num_branches = _positive_int(num_branches, "num_branches")
        self.metadata: dict[str, Any] = {
            "name": name,
            "channels": channels,
            "num_branches": num_branches,
            "bank_order": ["main", "large_or_local", "small_or_context"],
            "citation": _CITATIONS.get(name),
            "checked_date": "2026-10-28",
            "adaptation": "Study control; no paper reproduction claim.",
            "normalization": "none",
            "output": "one fused B,C,H,W tensor; no residual head",
        }

    def _check(self, bank: Sequence[Tensor]) -> Sequence[Tensor]:
        if not isinstance(bank, (list, tuple)) or len(bank) != self.num_branches:
            raise ValueError(f"bank must contain exactly {self.num_branches} tensors")
        anchor = bank[0]
        if not isinstance(anchor, Tensor) or anchor.ndim != 4:
            raise ValueError("bank[0] must be a B,C,H,W tensor")
        if anchor.shape[1] != self.channels or any(d < 1 for d in anchor.shape):
            raise ValueError(f"expected nonempty bank with C={self.channels}, got {tuple(anchor.shape)}")
        if not anchor.is_floating_point():
            raise TypeError("bank features must be floating-point tensors")
        for index, feature in enumerate(bank):
            if not isinstance(feature, Tensor) or feature.shape != anchor.shape:
                raise ValueError(f"bank[{index}] must match the anchor shape {tuple(anchor.shape)}")
            if feature.dtype != anchor.dtype or feature.device != anchor.device:
                raise ValueError(f"bank[{index}] must match the anchor dtype and device")
        return bank

    def _require_triplet(self) -> None:
        if self.num_branches != 3:
            raise ValueError(f"{self.metadata['name']} requires exactly 3 branches [M,L,S]")


class MeanUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3) -> None:
        super().__init__("mean", channels, num_branches)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        return torch.stack(list(self._check(bank)), dim=1).mean(dim=1)


class ConcatUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3) -> None:
        super().__init__("concat", channels, num_branches)
        self.projection = nn.Conv2d(channels * num_branches, channels, 1, bias=False)
        self.metadata["formula"] = "P(concat(M,L,S)); P is a bias-free 1x1 convolution"

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        return self.projection(torch.cat(list(self._check(bank)), dim=1))


class DifferenceUtilizer(_Utilizer):
    """Anchor plus a bias-free learned map of differences, not a GT error."""

    def __init__(
        self, mode: str, channels: int, num_branches: int = 3,
        eps: float = 1e-6, activation: str = "identity",
    ) -> None:
        if mode not in {"signed", "absolute", "relative", "normalized", "adjacent"}:
            raise ValueError(f"unsupported difference mode: {mode}")
        super().__init__(mode, channels, num_branches)
        if num_branches < 2:
            raise ValueError("difference utilizers require at least one auxiliary branch")
        if mode == "adjacent":
            self._require_triplet()
        self.mode = mode
        self.eps = _positive_eps(eps)
        activations = {"identity": nn.Identity, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}, got {activation!r}")
        self.projection = nn.Conv2d(channels * (num_branches - 1), channels, 1, bias=False)
        self.activation = activations[activation]()
        self.metadata.update({
            "eps": self.eps,
            "activation": activation,
            "formula": "main + activation(bias_free_1x1(concat(differences)))",
            "normalization": {
                "normalized": "pixelwise channel-vector L2, dim=1, norm clamped below by eps",
                "relative": "(aux-main)/(abs(aux)+abs(main)+eps), stable scaled evaluation",
            }.get(mode, "none"),
            "adaptation": "Study ablation. Learned projection maps difference coordinates to anchor units; no approximation-error claim.",
        })

    def differences(self, bank: Sequence[Tensor]) -> list[Tensor]:
        bank = self._check(bank)
        main = bank[0]
        if self.mode == "adjacent":
            return [bank[1] - main, main - bank[2]]
        if self.mode == "normalized":
            # Evaluate denominators in float32 for half precision, preserving
            # the autograd graph and returning the original feature dtype.
            work = [x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in bank]
            normed = [F.normalize(x, p=2, dim=1, eps=self.eps) for x in work]
            return [(x - normed[0]).to(dtype=main.dtype) for x in normed[1:]]
        if self.mode == "relative":
            work = [x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in bank]
            ref = work[0]
            results = []
            for aux in work[1:]:
                # Algebraically the stated relative difference; scaling also
                # avoids overflow of abs(aux)+abs(main) for large finite values.
                scale = torch.maximum(aux.abs(), ref.abs()).clamp_min(self.eps)
                a, m = aux / scale, ref / scale
                delta = (a - m) / (a.abs() + m.abs() + self.eps / scale)
                results.append(delta.to(dtype=main.dtype))
            return results
        differences = [x - main for x in bank[1:]]
        return [x.abs() for x in differences] if self.mode == "absolute" else differences

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        delta = torch.cat(self.differences(bank), dim=1)
        return bank[0] + self.activation(self.projection(delta))


class _LegacyConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class _LegacyDSConvBNReLU(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )


class LegacyGatedUtilizer(_Utilizer):
    """Exact fused branch of existing ScaleResidualIntegration; no residual_proj."""

    def __init__(self, channels: int, num_branches: int = 3, gate_reduction: int = 2) -> None:
        super().__init__("legacy_gated", channels, num_branches)
        self._require_triplet()
        gate_reduction = _positive_int(gate_reduction, "gate_reduction")
        hidden = max(8, channels // gate_reduction)
        self.gate = nn.Sequential(
            _LegacyConvBNAct(3 * channels, hidden),
            nn.Conv2d(hidden, 2 * channels, 1, bias=True), nn.Sigmoid(),
        )
        self.out_proj = _LegacyDSConvBNReLU(channels)
        self.metadata.update({
            "gate_reduction": gate_reduction,
            "normalization": "original BatchNorm2d; train B*H*W must exceed 1",
            "adaptation": "Original integration fused branch only, including out_proj; residual_proj and full legacy decoder are excluded. Not legacy_full.",
            "formula": "out_proj(M + gL*(L-M) + gS*(S-M))",
            "weight_note": "Two independent sigmoid gates; effective main coefficient 1-gL-gS can be negative.",
        })

    def gates(self, bank: Sequence[Tensor]) -> Tensor:
        main, large, small = self._check(bank)
        return self.gate(torch.cat([main, large - main, small - main], dim=1)).reshape(
            main.shape[0], 2, self.channels, *main.shape[-2:]
        )

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        gates = self.gates(bank)
        return torch.stack([1 - gates[:, 0] - gates[:, 1], gates[:, 0], gates[:, 1]], dim=1)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        main, large, small = self._check(bank)
        gates = self.gates(bank)
        y = main + gates[:, 0] * (large - main) + gates[:, 1] * (small - main)
        return self.out_proj(y)


class BiFPNUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3, eps: float = 1e-4) -> None:
        super().__init__("bifpn", channels, num_branches)
        self.eps = _positive_eps(eps)
        self.branch_weights = nn.Parameter(torch.ones(num_branches))
        self.metadata.update({
            "eps": self.eps,
            "normalization": "ReLU(theta)/(sum(ReLU(theta))+eps); sum is slightly less than 1",
            "adaptation": "EfficientDet fast normalized fusion operator only; no repeated bidirectional BiFPN graph or separable output convolution.",
        })

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        main = self._check(bank)[0]
        positive = F.relu(self.branch_weights)
        weights = positive / (positive.sum() + self.eps)
        return weights.to(dtype=main.dtype).view(1, self.num_branches, 1, 1, 1).expand(
            main.shape[0], -1, -1, -1, -1
        )

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        weights = self.weights(bank)
        return (torch.stack(list(bank), dim=1) * weights).sum(dim=1)


class SKUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3, reduction: int = 4, min_hidden: int = 8) -> None:
        super().__init__("sk", channels, num_branches)
        reduction = _positive_int(reduction, "reduction")
        hidden = max(_positive_int(min_hidden, "min_hidden"), channels // reduction)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU())
        self.expand = nn.Conv2d(hidden, num_branches * channels, 1)
        self.metadata.update({
            "reduction": reduction, "hidden_channels": hidden,
            "normalization": "softmax over branches per channel; no normalization in pooled MLP",
            "adaptation": "SK Fuse/Select on externally generated branches. Pooled BN omitted for batch-one support; hidden size configurable. No original SK split kernels.",
        })

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        bank = self._check(bank)
        pooled = torch.stack(list(bank), dim=1).sum(dim=1).mean(dim=(2, 3), keepdim=True)
        logits = self.expand(self.reduce(pooled)).reshape(bank[0].shape[0], self.num_branches, self.channels, 1, 1)
        return logits.softmax(dim=1)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        weights = self.weights(bank)
        return (torch.stack(list(bank), dim=1) * weights).sum(dim=1)


class ASFFUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3, hidden_channels: int = 16) -> None:
        super().__init__("asff", channels, num_branches)
        hidden = _positive_int(hidden_channels, "hidden_channels")
        self.compress = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), _local_norm(hidden), nn.ReLU())
            for _ in range(num_branches)
        ])
        self.attention = nn.Conv2d(hidden * num_branches, num_branches, 1)
        self.metadata.update({
            "hidden_channels": hidden,
            "normalization": "batch-one-safe local GN; softmax over branches at each pixel, shared across channels",
            "adaptation": "Single aligned pseudo-scale level; ASFF-style compressed-branch attention. GN replaces BN; no detector-level resize or output expansion.",
        })

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        bank = self._check(bank)
        context = torch.cat([projection(x) for projection, x in zip(self.compress, bank)], dim=1)
        return self.attention(context).softmax(dim=1).unsqueeze(2)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        weights = self.weights(bank)
        return (torch.stack(list(bank), dim=1) * weights).sum(dim=1)


class _AFFGate(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), _local_norm(hidden), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False), _local_norm(channels),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return (self.local(x) + self.global_context(x)).sigmoid()


class AFFUtilizer(_Utilizer):
    def __init__(self, channels: int, num_branches: int = 3, reduction: int = 4) -> None:
        super().__init__("aff", channels, num_branches)
        self._require_triplet()
        reduction = _positive_int(reduction, "reduction")
        hidden = max(1, channels // reduction)
        self.aux_gate = _AFFGate(channels, hidden)
        self.main_gate = _AFFGate(channels, hidden)
        self.metadata.update({
            "reduction": reduction, "hidden_channels": hidden,
            "normalization": "local GN (Identity at C=1); no norm after global pooling; sigmoid pairwise selection",
            "adaptation": "Paper Eq.(4) convex blend, no code x2 gain. New three-way order: AFF(M, AFF(L,S)); separate gates, not iAFF.",
            "formula": "aux=aL*L+(1-aL)*S; fused=aM*M+(1-aM)*aux",
        })

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        main, large, small = self._check(bank)
        aux_weight = self.aux_gate(large + small)
        aux = aux_weight * large + (1 - aux_weight) * small
        main_weight = self.main_gate(main + aux)
        return torch.stack([
            main_weight,
            (1 - main_weight) * aux_weight,
            (1 - main_weight) * (1 - aux_weight),
        ], dim=1)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        weights = self.weights(bank)
        return (torch.stack(list(bank), dim=1) * weights).sum(dim=1)


class MHSIUUtilizer(_Utilizer):
    def __init__(
        self, channels: int, num_branches: int = 3, heads: int = 4,
        reduction: int = 4,
    ) -> None:
        super().__init__("mhsiu", channels, num_branches)
        requested_heads = _positive_int(heads, "heads")
        reduction = _positive_int(reduction, "reduction")
        self.heads = math.gcd(channels, requested_heads)
        self.head_channels = channels // self.heads
        hidden = max(4, self.head_channels // reduction)
        total = channels * num_branches
        self.value_transform = nn.Conv2d(total, total, 1, groups=self.heads, bias=False)
        self.attention = nn.Sequential(
            nn.Conv2d(total, self.heads * hidden, 1, groups=self.heads, bias=False),
            nn.GroupNorm(self.heads, self.heads * hidden), nn.ReLU(),
            nn.Conv2d(self.heads * hidden, self.heads * num_branches, 1, groups=self.heads),
        )
        self.metadata.update({
            "requested_heads": requested_heads, "heads": self.heads,
            "head_channels": self.head_channels, "hidden_per_head": hidden,
            "reduction": reduction,
            "normalization": "GN within heads; branch softmax per head and pixel",
            "adaptation": "MHSIU-inspired grouped value and spatial-attention transforms on already aligned pseudo features; not original image-pyramid encoder or MHSIU layer reproduction. heads=gcd(C,requested_heads).",
        })

    def _pack(self, bank: Sequence[Tensor]) -> Tensor:
        main = bank[0]
        b, _, h, w = main.shape
        # Group channels by head first, then branch. Plain cat(bank) would mix
        # branch groups incorrectly when used with grouped convolutions.
        stacked = torch.stack(list(bank), dim=1).reshape(
            b, self.num_branches, self.heads, self.head_channels, h, w
        )
        return stacked.permute(0, 2, 1, 3, 4, 5).reshape(b, self.num_branches * self.channels, h, w)

    def _weights_from_packed(self, packed: Tensor) -> Tensor:
        b, _, h, w = packed.shape
        logits = self.attention(packed).reshape(b, self.heads, self.num_branches, h, w)
        return logits.softmax(dim=2).permute(0, 2, 1, 3, 4)

    def weights(self, bank: Sequence[Tensor]) -> Tensor:
        """B,branches,heads,H,W weights; dim=1 sums to one."""
        return self._weights_from_packed(self._pack(self._check(bank)))

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        bank = self._check(bank)
        packed = self._pack(bank)
        b, _, h, w = packed.shape
        weights = self._weights_from_packed(packed).permute(0, 2, 1, 3, 4).unsqueeze(3)
        values = self.value_transform(packed).reshape(
            b, self.heads, self.num_branches, self.head_channels, h, w
        )
        return (values * weights).sum(dim=2).reshape(b, self.channels, h, w)


class AnchorGatedUtilizer(_Utilizer):
    """Preserve the main feature and learn a small, channel-gated supplement."""
    def __init__(self, channels, num_branches=3, gate_init=0.01):
        super().__init__("anchor_gated", channels, num_branches)
        self._require_triplet()
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("gate_init must be finite and strictly between zero and one")
        self.projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.gate_logits = nn.Parameter(torch.full((1, channels, 1, 1), math.log(gate_init / (1 - gate_init))))
        self.metadata.update({"formula": "M + sigmoid(a_c) * P(concat(L,S))",
                              "gate_init": gate_init, "gate": "learned per-channel, shared across images and spatial locations",
                              "adaptation": "Study proposal; no branch subtraction or performance guarantee"})

    def forward(self, bank):
        main, large, small = self._check(bank)
        return main + self.gate_logits.sigmoid() * self.projection(torch.cat([large, small], dim=1))


def build_utilizer(name: str, channels: int, num_branches: int = 3, **kwargs: Any) -> nn.Module:
    """Construct a utilizer; unknown arguments raise instead of being ignored."""
    if name == "anchor_gated":
        return AnchorGatedUtilizer(channels, num_branches, **kwargs)
    if name not in UTILIZER_NAMES:
        raise ValueError(f"unknown utilizer {name!r}; choose from {UTILIZER_NAMES}")
    if name in {"signed", "absolute", "relative", "normalized", "adjacent"}:
        return DifferenceUtilizer(name, channels, num_branches, **kwargs)
    factories = {
        "mean": MeanUtilizer, "concat": ConcatUtilizer,
        "legacy_gated": LegacyGatedUtilizer, "bifpn": BiFPNUtilizer,
        "sk": SKUtilizer, "asff": ASFFUtilizer, "aff": AFFUtilizer,
        "mhsiu": MHSIUUtilizer,
    }
    return factories[name](channels, num_branches, **kwargs)
