"""Common PVTv2-B2 encoder/decoder with independently seeded experimental modules."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .generators import build_generator
from .utilizers import build_utilizer

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


def load_backbone(encoder, path):
    """Strict official PVT checkpoint import; only classification head is omitted."""
    state = torch.load(Path(path).expanduser().resolve(strict=True), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Backbone checkpoint must be a flat state dict or {'state_dict': ...}")
    state = dict(state)
    for name in ("head.weight", "head.bias"):
        state.pop(name, None)
    encoder.load_state_dict(state, strict=True)


class CommonPrior(nn.Module):
    """Both auxiliary targets use fused features, including the no-pseudo control."""
    def __init__(self, channels):
        super().__init__()
        def head():
            return nn.Sequential(ConvBNAct(4 * channels, channels),
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


class StudyModel(nn.Module):
    def __init__(self, generator="legacy", utilizer="mean", channels=48, seed=0,
                 backbone_weights=None, baseline=False, legacy_reference=False,
                 generator_options=None, utilizer_options=None):
        super().__init__()
        if channels < 4:
            raise ValueError("channels must be >=4")
        if baseline and legacy_reference:
            raise ValueError("baseline and legacy_reference are mutually exclusive")
        self.baseline, self.legacy_reference = baseline, legacy_reference
        self.config = dict(generator=generator, utilizer=utilizer, channels=channels, seed=seed,
                           baseline=baseline, legacy_reference=legacy_reference,
                           generator_options=generator_options or {}, utilizer_options=utilizer_options or {})
        # All common modules are created first, under the same isolated seed. Changing
        # variant parameter counts therefore cannot perturb encoder/decoder initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if legacy_reference:
                self.reference = PvtV2B2_FSA_ZoomNeXt(pretrained=False, mid_dim=channels)
            else:
                self.backbone = pvt_v2_eff_b2(pretrained=False)
                self.normalizer = PixelNormalizer()
                self.projections = nn.ModuleList([
                    ConvBNReLU(c, channels, 3, 1, 1) for c in (64, 128, 320)
                ] + [SafeSimpleASPP(512, channels)])
                self.prior = CommonPrior(channels)
                self.refinement = nn.ModuleList([CommonRefinement(channels) for _ in range(4)])
                self.deep_heads = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in range(4)])
                self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                        DSConvBNReLU(channels, channels))
                self.predictor = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                               DSConvBNReLU(channels, 32), nn.Conv2d(32, 1, 1))
        if backbone_weights:
            load_backbone(self.encoder, backbone_weights)
        if not baseline and not legacy_reference:
            generators, utilizers = [], []
            for level in range(4):
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
    def metadata(self):
        return {"config": self.config,
                "architecture": "original_full_FSA_reference" if self.legacy_reference else "controlled_common_decoder",
                "generator": None if self.baseline or self.legacy_reference else self.generators[0].metadata,
                "utilizer": None if self.baseline or self.legacy_reference else self.utilizers[0].metadata}

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 64:
            raise ValueError("Input must have shape B,3,H,W with H,W>=64 and RGB values in [0,1]")
        if self.legacy_reference:
            logits, auxiliary = self.reference.body({"image_m": x})
            return {"logits": resize(logits, x), **auxiliary}
        endpoints = self.encoder(self.normalizer(x))  # Exactly one encoder call.
        features = [projection(endpoints[f"reduction_{level + 2}"])
                    for level, projection in enumerate(self.projections)]
        if not self.baseline:
            features = [u(g(f)) for f, g, u in zip(features, self.generators, self.utilizers)]
        po, pe = self.prior(features)
        refined = [None] * 4
        deeper = None
        for level in reversed(range(4)):
            deeper = self.refinement[level](features[level], deeper, po)
            refined[level] = deeper
        logits = resize(self.predictor(self.up(refined[0])), x)
        return {"logits": logits, "po": po, "pe": pe,
                "deep_logits": [head(f) for head, f in zip(self.deep_heads, refined)]}
