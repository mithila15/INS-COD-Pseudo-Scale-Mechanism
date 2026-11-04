"""Single-pass encoder study with multistage and final-stage-only decoders."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .generators import build_generator
from .utilizers import build_utilizer
from .backbones import (BACKBONE_CHANNELS, BACKBONE_STAGES, build_cnn_backbone,
                        load_backbone, validate_backbone_options)

ARCHIVE = Path(__file__).resolve().parents[3] / "FSA-ZoomNeXt"
if "methods" in sys.modules:
    origin = Path(sys.modules["methods"].__file__).resolve()
    if not origin.is_relative_to(ARCHIVE):
        raise ImportError(f"Conflicting methods package already imported: {origin}")
sys.path.insert(0, str(ARCHIVE))
from methods.backbone.pvt_v2_eff import pvt_v2_eff_b2
from methods.zoomnext.fsa_layers import DSConvBNReLU, ConvBNAct, SafeSimpleASPP
from methods.zoomnext.fsa_zoomnext import PvtV2B2_FSA_ZoomNeXt
from methods.zoomnext.ops import ConvBNReLU, PixelNormalizer


def resize(x, like):
    return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)


class CommonPrior(nn.Module):
    """Both auxiliary targets use fused features, including the no-pseudo control."""
    def __init__(self, channels, levels=4):
        super().__init__()
        def head():
            return nn.Sequential(ConvBNAct(levels * channels, channels),
                                 DSConvBNReLU(channels, channels),
                                 DSConvBNReLU(channels, channels), nn.Conv2d(channels, 1, 1))
        self.region = head()
        self.edge = head()

    def forward(self, features):
        joined = torch.cat([resize(x, features[0]) for x in features], dim=1)
        return self.region(joined), self.edge(joined)


class CommonRefinement(nn.Module):
    """Original region-guided top-down design without residual-specific parameters."""
    def __init__(self, channels):
        super().__init__()
        self.pre = ConvBNAct(2 * channels, channels)
        self.branches = nn.ModuleList([DSConvBNReLU(channels, channels, 3),
                                      DSConvBNReLU(channels, channels, 5),
                                      ConvBNAct(channels, channels)])
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.out = ConvBNAct(channels, channels)

    def forward(self, x, deeper, po):
        deeper = torch.zeros_like(x) if deeper is None else resize(deeper, x)
        y = self.pre(torch.cat([x, deeper], dim=1)) * resize(po, x).sigmoid()
        weights = self.branch_logits.softmax(0).to(y.dtype) * 3
        return self.out(sum(w * branch(y) for w, branch in zip(weights, self.branches)))


class LastStageDecoder(nn.Module):
    """Decode a single stride-32 feature without earlier-stage skip connections.

    The only spatial convolution kernel is 3x3. Five interpolation/convolution
    steps recover image resolution; this is decoding, not a feature bank for
    cross-stage fusion. The region gate and auxiliary losses remain supervised
    by the same GT targets as the historical study.
    """
    def __init__(self, channels):
        super().__init__()
        self.refine = DSConvBNReLU(channels, channels, 3)
        self.deep_head = nn.Conv2d(channels, 1, 1)
        self.upsampling = nn.ModuleList([DSConvBNReLU(channels, channels, 3) for _ in range(5)])
        self.predictor = nn.Conv2d(channels, 1, 1)

    def forward(self, feature, region, size):
        y = self.refine(feature * region.sigmoid())
        deep_logits = self.deep_head(y)
        for stride, block in zip((16, 8, 4, 2, 1), self.upsampling):
            spatial = tuple((dimension + stride - 1) // stride for dimension in size)
            y = block(F.interpolate(y, size=spatial, mode="bilinear", align_corners=False))
        return self.predictor(y), [deep_logits]


class StudyModel(nn.Module):
    def __init__(self, generator="legacy", utilizer="mean", channels=48, seed=0,
                 backbone_weights=None, baseline=False, legacy_reference=False,
                 generator_options=None, utilizer_options=None,
                 backbone="pvt_v2_b2", feature_mode="multistage"):
        super().__init__()
        validate_backbone_options(backbone, feature_mode)
        if channels < 4:
            raise ValueError("channels must be >=4")
        if baseline and legacy_reference:
            raise ValueError("baseline and legacy_reference are mutually exclusive")
        if legacy_reference and (backbone != "pvt_v2_b2" or feature_mode != "multistage"):
            raise ValueError("legacy_reference requires backbone=pvt_v2_b2 and feature_mode=multistage")
        self.baseline, self.legacy_reference = baseline, legacy_reference
        self.backbone_name, self.feature_mode = backbone, feature_mode
        self.feature_levels = (5,) if feature_mode == "last" else (2, 3, 4, 5)
        self.endpoint_keys = tuple(f"reduction_{i}" for i in self.feature_levels)
        self.config = dict(generator=generator, utilizer=utilizer, channels=channels, seed=seed,
                           baseline=baseline, legacy_reference=legacy_reference,
                           generator_options=generator_options or {}, utilizer_options=utilizer_options or {},
                           backbone=backbone, feature_mode=feature_mode)
        # All common modules are created first, under the same isolated seed. Changing
        # variant parameter counts therefore cannot perturb encoder/decoder initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if legacy_reference:
                self.reference = PvtV2B2_FSA_ZoomNeXt(pretrained=False, mid_dim=channels)
            else:
                self.backbone = (pvt_v2_eff_b2(pretrained=False) if backbone == "pvt_v2_b2"
                                 else build_cnn_backbone(backbone, feature_mode))
                self.normalizer = PixelNormalizer()
                stage_channels = BACKBONE_CHANNELS[backbone]
                if feature_mode == "multistage":
                    # Preserve module creation order, names, shapes and initialization
                    # of all historical PVT common-decoder checkpoints.
                    self.projections = nn.ModuleList([
                        ConvBNReLU(c, channels, 3, 1, 1) for c in stage_channels[:-1]
                    ] + [SafeSimpleASPP(stage_channels[-1], channels)])
                    self.prior = CommonPrior(channels)
                    self.refinement = nn.ModuleList([CommonRefinement(channels) for _ in range(4)])
                    self.deep_heads = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in range(4)])
                    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                            DSConvBNReLU(channels, channels))
                    self.predictor = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                                   DSConvBNReLU(channels, 32), nn.Conv2d(32, 1, 1))
                else:
                    self.projections = nn.ModuleList([ConvBNAct(stage_channels[-1], channels)])
                    self.prior = CommonPrior(channels, levels=1)
                    self.last_decoder = LastStageDecoder(channels)
        if backbone_weights:
            load_backbone(self.encoder, backbone_weights, backbone)
        if not baseline and not legacy_reference:
            generators, utilizers = [], []
            for level in range(len(self.endpoint_keys)):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed + 10000 + level)
                    generators.append(build_generator(generator, channels, **(generator_options or {})))
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed + 20000 + level)
                    utilizers.append(build_utilizer(utilizer, channels, num_branches=3, **(utilizer_options or {})))
            self.generators = nn.ModuleList(generators)
            self.utilizers = nn.ModuleList(utilizers)

    @property
    def encoder(self):
        return self.reference.encoder if self.legacy_reference else self.backbone

    @property
    def frozen_stem_modules(self):
        """Same first-stem freeze policy expressed without PVT-specific names."""
        if self.backbone_name == "convnext_tiny":
            return (self.encoder.features[0],)
        if self.backbone_name == "resnet50":
            return (self.encoder.conv1, self.encoder.bn1)
        return (self.encoder.patch_embed1,)

    @property
    def metadata(self):
        stages = BACKBONE_STAGES[self.backbone_name]
        return {"config": self.config,
                "architecture": ("original_full_FSA_reference" if self.legacy_reference else
                                 "last_stage_progressive_decoder" if self.feature_mode == "last" else
                                 "controlled_common_decoder"),
                "backbone": self.backbone_name, "feature_mode": self.feature_mode,
                "encoder_calls_per_image": 1,
                "encoder_stage_sources": list(stages[-1:] if self.feature_mode == "last" else stages),
                "encoder_output_strides": [32] if self.feature_mode == "last" else [4, 8, 16, 32],
                "encoder_stage_fusion": self.feature_mode == "multistage",
                "deep_supervision_heads": len(self.endpoint_keys),
                "frozen_stem_policy": "first encoder stem; normalization statistics follow training policy",
                "generator": None if self.baseline or self.legacy_reference else self.generators[0].metadata,
                "utilizer": None if self.baseline or self.legacy_reference else self.utilizers[0].metadata}

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 64:
            raise ValueError("Input must have shape B,3,H,W with H,W>=64 and RGB values in [0,1]")
        if self.legacy_reference:
            logits, auxiliary = self.reference.body({"image_m": x})
            return {"logits": resize(logits, x), **auxiliary}
        endpoints = self.encoder(self.normalizer(x))  # Exactly one encoder call.
        features = [projection(endpoints[key]) for key, projection in zip(self.endpoint_keys, self.projections)]
        if not self.baseline:
            features = [u(g(f)) for f, g, u in zip(features, self.generators, self.utilizers)]
        po, pe = self.prior(features)
        if self.feature_mode == "last":
            logits, deep_logits = self.last_decoder(features[0], po, x.shape[-2:])
            return {"logits": logits, "po": po, "pe": pe, "deep_logits": deep_logits}
        refined = [None] * 4
        deeper = None
        for level in reversed(range(4)):
            deeper = self.refinement[level](features[level], deeper, po)
            refined[level] = deeper
        logits = resize(self.predictor(self.up(refined[0])), x)
        return {"logits": logits, "po": po, "pe": pe,
                "deep_logits": [head(f) for head, f in zip(self.deep_heads, refined)]}
