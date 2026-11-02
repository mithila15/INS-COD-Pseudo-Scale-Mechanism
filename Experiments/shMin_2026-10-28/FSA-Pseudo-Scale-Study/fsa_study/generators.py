"""Controlled three-branch feature generators; see docs/GENERATION.md.

These are independently written adaptations of published mathematical ideas,
not copies of the cited repositories or reproductions of their full networks.
Every generator returns [main, local/large, context/small] at the input shape.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


GENERATOR_NAMES = (
    "identity", "projection", "legacy", "resample", "native_conv",
    "shared_native", "ffp_gain", "sfp", "ppm", "aspp", "carafe", "dysample",
)


def _resize(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _scaled_size(x, scale):
    return tuple(max(1, int(round(n * scale))) for n in x.shape[-2:])


def _half_pool(x):
    return F.adaptive_avg_pool2d(x, tuple(max(1, n // 2) for n in x.shape[-2:]))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels, including batch-one/single-pixel inputs."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _projection(channels, dilation=1):
    # The same architecture makes projection/native_conv parameter matched.
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=dilation,
                  dilation=dilation, groups=channels, bias=False),
        ChannelLayerNorm(channels), nn.GELU(),
        nn.Conv2d(channels, channels, 1, bias=False),
        ChannelLayerNorm(channels), nn.GELU(),
    )


class _SingletonSafeBatchNorm(nn.BatchNorm2d):
    """Legacy BN except for the otherwise undefined one-value training case."""

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


def _legacy_projection(channels):
    # Exact old depthwise + BN + ReLU, pointwise + BN + ReLU,
    # then separate pointwise + BN + ReLU, preserving operation order.
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        _SingletonSafeBatchNorm(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        _SingletonSafeBatchNorm(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        _SingletonSafeBatchNorm(channels), nn.ReLU(inplace=True),
    )


class _Generator(nn.Module):
    num_branches = 3

    def __init__(self, channels, name, family, description, papers=()):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.metadata = {
            "name": name, "channels": channels, "num_branches": 3,
            "branch_order": ["main", "large_or_local", "small_or_context"],
            "output_alignment": "input B,C,H,W; bilinear align_corners=False",
            "family": family, "description": description,
            "papers": list(papers), "normalization": "none",
            "checked_date": "2026-10-28", "full_paper_reproduction": False,
        }

    def _check(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("expected B,C,H,W with configured channel count")
        if min(x.shape) < 1:
            raise ValueError("all input dimensions must be positive")


class IdentityGenerator(_Generator):
    def __init__(self, channels):
        super().__init__(channels, "identity", "control", "Three identical feature references")

    def forward(self, x):
        self._check(x)
        return [x, x, x]


class ProjectionGenerator(_Generator):
    def __init__(self, channels):
        super().__init__(channels, "projection", "capacity_control",
                         "Independent learned projections at unchanged resolution")
        self.branches = nn.ModuleList([_projection(channels), _projection(channels)])
        self.metadata["normalization"] = "channel LayerNorm"
        self.metadata["parameter_match"] = "native_conv; same projection operations"

    def forward(self, x):
        self._check(x)
        return [x, self.branches[0](x), self.branches[1](x)]


class LegacyGenerator(_Generator):
    def __init__(self, channels):
        super().__init__(channels, "legacy", "existing_fsa",
                         "Original FSA PS; small pool then align then project; large projects input")
        self.large_proj = _legacy_projection(channels)
        self.small_proj = _legacy_projection(channels)
        self.metadata["normalization"] = "BatchNorm2d; singleton training uses running statistics"
        self.metadata["exception"] = (
            "Only B*H*W=1 training BN differs from old code, which raises an error. "
            "Branch return order is main/large/small instead of large/main/small."
        )

    def forward(self, x):
        self._check(x)
        small = _resize(_half_pool(x), x.shape[-2:])
        return [x, self.large_proj(x), self.small_proj(small)]


class ResampleGenerator(_Generator):
    def __init__(self, channels, scale_factors=(1.5, 0.5)):
        super().__init__(channels, "resample", "resampling_control",
                         "Bilinear native resampling followed by alignment, no learned operations")
        if len(scale_factors) != 2 or not all(math.isfinite(float(s)) and float(s) > 0 for s in scale_factors):
            raise ValueError("scale_factors must contain two finite positive numbers")
        self.scales = tuple(float(s) for s in scale_factors)
        self.metadata["scale_factors"] = list(self.scales)
        self.metadata["rounding"] = "max(1, round(size*scale)); Python round"

    def _native(self, x, scale):
        return _resize(x, _scaled_size(x, scale))

    def forward(self, x):
        self._check(x)
        return [x] + [_resize(self._native(x, s), x.shape[-2:]) for s in self.scales]


class NativeConvGenerator(ResampleGenerator):
    def __init__(self, channels, scale_factors=(1.5, 0.5), shared=False):
        super().__init__(channels, scale_factors)
        self.shared = shared
        self.branches = nn.ModuleList([_projection(channels) for _ in range(1 if shared else 2)])
        self.metadata.update({
            "name": "shared_native" if shared else "native_conv",
            "family": "native_scale_projection",
            "description": "Resample, project at native scale, then align",
            "normalization": "channel LayerNorm", "shared_projection": shared,
        })

    def forward(self, x):
        self._check(x)
        out = [x]
        for i, scale in enumerate(self.scales):
            branch = self.branches[0 if self.shared else i](self._native(x, scale))
            out.append(_resize(branch, x.shape[-2:]))
        return out


class FFPGainGenerator(ResampleGenerator):
    def __init__(self, channels, scale_factors=(1.5, 0.5)):
        super().__init__(channels, scale_factors)
        # A common per-channel exponent models a power law across both scales.
        self.scale_exponent = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.metadata.update({
            "name": "ffp_gain", "family": "scale_approximation_hypothesis",
            "description": "Resampled deep feature times exp(-lambda_c*log(scale)); lambda starts at zero",
            "papers": ["Dollar et al., Fast Feature Pyramids for Object Detection, TPAMI 2014"],
            "adaptation": "Learned deep-feature channel exponents, not calibrated low-level FFP statistics",
        })

    def forward(self, x):
        self._check(x)
        out = [x]
        # Keep exponentiation stable in mixed precision without promoting the
        # resulting feature bank away from the anchor dtype. Double inputs retain
        # double arithmetic. Casting the parameter preserves its autograd path.
        work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        for scale in self.scales:
            log_scale = torch.tensor(scale, device=x.device, dtype=work_dtype).log()
            gain = torch.exp(-self.scale_exponent.to(dtype=work_dtype) * log_scale).to(dtype=x.dtype)
            out.append(_resize(self._native(x, scale) * gain, x.shape[-2:]))
        return out


class SimplePyramidGenerator(_Generator):
    def __init__(self, channels):
        super().__init__(channels, "sfp", "learned_resolution_pyramid",
                         "Twofold transposed convolution / half max-pool, necks before alignment",
                         ["Li et al., Exploring Plain Vision Transformer Backbones for Object Detection, ECCV 2022"])
        hidden = max(1, channels // 2)
        self.up = nn.ConvTranspose2d(channels, hidden, 2, stride=2)
        self.large_neck = self._neck(hidden, channels)
        self.small_neck = self._neck(channels, channels)
        self.metadata.update({"native_scale_factors": [2.0, 0.5],
                              "normalization": "channel LayerNorm",
                              "adaptation": "Two non-main branches, then align; no ViT or detector reproduction"})

    @staticmethod
    def _neck(in_channels, channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False), ChannelLayerNorm(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), ChannelLayerNorm(channels),
        )

    def forward(self, x):
        self._check(x)
        # Floor half matches MaxPool2d(2,2); the clamp makes singleton axes valid.
        # Adaptive max-pool differs on odd sizes (explicitly recorded below).
        target = tuple(max(1, n // 2) for n in x.shape[-2:])
        small = F.adaptive_max_pool2d(x, target)
        return [x, _resize(self.large_neck(self.up(x)), x.shape[-2:]),
                _resize(self.small_neck(small), x.shape[-2:])]


class PoolingPyramidGenerator(_Generator):
    def __init__(self, channels, ppm_bins=(2, 4)):
        super().__init__(channels, "ppm", "context_pyramid",
                         "Adaptive-average context bins, pointwise projections, then alignment",
                         ["Zhao et al., Pyramid Scene Parsing Network, CVPR 2017"])
        if len(ppm_bins) != 2 or any(int(b) != b or b < 1 for b in ppm_bins):
            raise ValueError("ppm_bins must contain two positive integers")
        self.bins = tuple(int(b) for b in ppm_bins)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                          ChannelLayerNorm(channels), nn.GELU()) for _ in range(2)
        ])
        self.metadata.update({
            "ppm_bins": list(self.bins), "normalization": "channel LayerNorm",
            "branch_order": ["main", "finer_pooling_context", "coarser_pooling_context"],
            "adaptation": "Two C-channel branches vs original four reduced-channel bins; no global bin",
            "bin_order": "descending bin count = finer/local first; each axis capped to input size",
        })

    def forward(self, x):
        self._check(x)
        out = [x]
        for branch, bin_size in zip(self.branches, sorted(self.bins, reverse=True)):
            size = tuple(min(bin_size, n) for n in x.shape[-2:])
            out.append(_resize(branch(F.adaptive_avg_pool2d(x, size)), x.shape[-2:]))
        return out


class AtrousPyramidGenerator(_Generator):
    def __init__(self, channels, aspp_rates=(1, 3)):
        super().__init__(channels, "aspp", "receptive_field_pyramid",
                         "Parallel dense atrous convolutions with different receptive fields",
                         ["Chen et al., Rethinking Atrous Convolution for Semantic Image Segmentation, arXiv 2017"])
        if len(aspp_rates) != 2 or any(int(r) != r or r < 1 for r in aspp_rates):
            raise ValueError("aspp_rates must contain two positive integers")
        self.rates = tuple(int(r) for r in aspp_rates)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False),
                          ChannelLayerNorm(channels), nn.GELU()) for r in self.rates
        ])
        self.metadata.update({
            "aspp_rates": list(self.rates), "normalization": "channel LayerNorm",
            "adaptation": "Two dilated branches; omits original 1x1/global branch and fixed fusion",
            "limitation": "Large dilation on small features mostly samples padding; no rate clamping",
        })

    def forward(self, x):
        self._check(x)
        return [x, self.branches[0](x), self.branches[1](x)]


class ContentAwareUpsampler(nn.Module):
    """CARAFE mathematical operation using low-resolution unfold + contraction.

    Kernel channels are indexed by (subpixel, neighborhood). Reassembly uses
    zero padding, so kernels straddling the image boundary include zeros.
    """

    def __init__(self, channels, kernel_size=5, compressed=64, scale=2):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("carafe_kernel must be a positive odd integer")
        if not isinstance(compressed, int) or compressed < 1:
            raise ValueError("carafe_compressed must be a positive integer")
        self.kernel_size = kernel_size
        self.scale = scale
        self.compress = nn.Conv2d(channels, compressed, 1)
        self.encode = nn.Conv2d(compressed, scale * scale * kernel_size * kernel_size, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        k, s = self.kernel_size, self.scale
        weights = self.encode(self.compress(x)).reshape(b, s * s, k * k, h, w)
        weights = weights.softmax(dim=2)
        patches = F.unfold(x, kernel_size=k, padding=k // 2).reshape(b, c, k * k, h, w)
        reassembled = torch.einsum("bckhw,bskhw->bcshw", patches, weights)
        return F.pixel_shuffle(reassembled.reshape(b, c * s * s, h, w), s)


class CARAFEGenerator(_Generator):
    def __init__(self, channels, carafe_kernel=5, carafe_compressed=64):
        super().__init__(channels, "carafe", "content_aware_upsampling",
                         "Content-predicted normalized local kernels upsample by two; small branch average-pools",
                         ["Wang et al., CARAFE: Content-Aware ReAssembly of FEatures, ICCV 2019"])
        self.upsampler = ContentAwareUpsampler(channels, carafe_kernel, carafe_compressed)
        self.metadata.update({
            "carafe_kernel": carafe_kernel, "carafe_compressed": carafe_compressed,
            "native_scale_factors": [2.0, 0.5], "padding": "zero for CARAFE local patches",
            "adaptation": "Torch unfold/einsum implementation; pooled small branch; output realignment",
            "memory": "unfold workspace B*C*k*k*H*W; no official fused CUDA kernel",
            "official_checkpoint_compatible": False,
            "checkpoint_note": "Subpixel/kernel channel order differs; no direct official-weight import or mapping provided",
        })

    def forward(self, x):
        self._check(x)
        return [x, _resize(self.upsampler(x), x.shape[-2:]),
                _resize(_half_pool(x), x.shape[-2:])]


class DynamicSampleUpsampler(nn.Module):
    """Independent LP-style DySample formulation on an explicit bilinear grid.

    This static-scope version predicts grouped offsets at low resolution. The
    regular bilinear target grid incorporates the subpixel initial positions.
    """

    def __init__(self, channels, groups=4, scale=2):
        super().__init__()
        if not isinstance(groups, int) or groups < 1:
            raise ValueError("dysample_groups must be a positive integer")
        self.groups = max(g for g in range(1, min(channels, groups) + 1) if channels % g == 0)
        self.scale = scale
        self.offset = nn.Conv2d(channels, 2 * self.groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        s, g = self.scale, self.groups
        offsets = F.pixel_shuffle(self.offset(x) * 0.25, s)
        offsets = offsets.reshape(b, g, 2, s * h, s * w).permute(0, 1, 3, 4, 2)
        # Target positions expressed in low-resolution pixel-center coordinates.
        yy = (torch.arange(s * h, device=x.device, dtype=x.dtype) + 0.5) / s
        xx = (torch.arange(s * w, device=x.device, dtype=x.dtype) + 0.5) / s
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((gx, gy), dim=-1)[None, None] + offsets
        normalizer = x.new_tensor([w, h])
        grid = (coords * (2.0 / normalizer) - 1.0).reshape(b * g, s * h, s * w, 2)
        sampled = F.grid_sample(x.reshape(b * g, c // g, h, w), grid,
                                mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(b, c, s * h, s * w)


class DySampleGenerator(_Generator):
    def __init__(self, channels, dysample_groups=4):
        super().__init__(channels, "dysample", "learned_sampling_upsampling",
                         "LP-style grouped offsets with static scope; small branch average-pools",
                         ["Liu et al., Learning to Upsample by Learning to Sample, ICCV 2023"])
        self.upsampler = DynamicSampleUpsampler(channels, dysample_groups)
        self.metadata.update({
            "dysample_groups_requested": dysample_groups,
            "dysample_groups_effective": self.upsampler.groups,
            "native_scale_factors": [2.0, 0.5], "offset_initialization_std": 0.001,
            "offset_multiplier": 0.25, "padding": "border",
            "adaptation": "LP/static-scope idea; divisor-safe grouping; pooled small branch; output realignment",
            "official_checkpoint_compatible": False,
            "checkpoint_note": "Group/coordinate channel order differs; no direct official-weight import or mapping provided",
        })

    def forward(self, x):
        self._check(x)
        return [x, _resize(self.upsampler(x), x.shape[-2:]),
                _resize(_half_pool(x), x.shape[-2:])]


def build_generator(name, channels, **kwargs):
    """Build one generator, rejecting irrelevant or misspelled options.

    Tunable keyword arguments are constructor-specific: scale_factors for
    resample/native_conv/shared_native/ffp_gain; ppm_bins; aspp_rates;
    carafe_kernel/carafe_compressed; or dysample_groups.
    """
    constructors = {
        "identity": IdentityGenerator, "projection": ProjectionGenerator,
        "legacy": LegacyGenerator, "resample": ResampleGenerator,
        "native_conv": NativeConvGenerator, "ffp_gain": FFPGainGenerator,
        "sfp": SimplePyramidGenerator, "ppm": PoolingPyramidGenerator,
        "aspp": AtrousPyramidGenerator, "carafe": CARAFEGenerator,
        "dysample": DySampleGenerator,
    }
    if name == "shared_native":
        if "shared" in kwargs:
            raise TypeError("shared is fixed by generator name")
        return NativeConvGenerator(channels, shared=True, **kwargs)
    if name not in constructors:
        raise ValueError("unknown generator {!r}; choose {}".format(name, GENERATOR_NAMES))
    if name == "native_conv" and "shared" in kwargs:
        raise TypeError("use shared_native to share projections")
    result = constructors[name](channels, **kwargs)
    if name == "sfp":
        result.metadata["odd_size_exception"] = "adaptive max-pool to floor-half shape; differs from fixed pool on odd axes"
    return result
