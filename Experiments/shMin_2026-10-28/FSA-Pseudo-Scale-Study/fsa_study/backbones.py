"""Single-pass CNN feature adapters and strict, local-only checkpoint loading.

CNN implementations are torchvision's ConvNeXt-Tiny and ResNet-50 (V1.5).
Classification pooling/heads are not part of these segmentation encoders.
No constructor requests pretrained weights or accesses the network.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


BACKBONE_NAMES = ("pvt_v2_b2", "convnext_tiny", "resnet50")
FEATURE_MODES = ("multistage", "last")
BACKBONE_CHANNELS = {
    "pvt_v2_b2": (64, 128, 320, 512),
    "convnext_tiny": (96, 192, 384, 768),
    "resnet50": (256, 512, 1024, 2048),
}
BACKBONE_STAGES = {
    "pvt_v2_b2": ("reduction_2", "reduction_3", "reduction_4", "reduction_5"),
    "convnext_tiny": ("features.1", "features.3", "features.5", "features.7"),
    "resnet50": ("layer1", "layer2", "layer3", "layer4"),
}
CLASSIFIER_KEYS = {
    "pvt_v2_b2": ("head.weight", "head.bias"),
    "convnext_tiny": ("classifier.0.weight", "classifier.0.bias",
                      "classifier.2.weight", "classifier.2.bias"),
    "resnet50": ("fc.weight", "fc.bias"),
}


def validate_backbone_options(backbone, feature_mode):
    if backbone not in BACKBONE_NAMES:
        raise ValueError(f"Unknown backbone {backbone!r}; choose {BACKBONE_NAMES}")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode {feature_mode!r}; choose {FEATURE_MODES}")


class ConvNeXtFeatures(nn.Module):
    """Expose stage endpoints while executing the CNN exactly once per image."""
    backbone_name = "convnext_tiny"

    def __init__(self, feature_mode="multistage"):
        super().__init__()
        validate_backbone_options(self.backbone_name, feature_mode)
        from torchvision.models import convnext_tiny
        self.features = convnext_tiny(weights=None).features
        self.feature_mode = feature_mode

    def forward(self, x):
        endpoints = {}
        for index, block in enumerate(self.features):
            x = block(x)
            if index in (1, 3, 5, 7) and (self.feature_mode == "multistage" or index == 7):
                endpoints[f"reduction_{(index + 3) // 2}"] = x
        return endpoints


class ResNetFeatures(nn.Module):
    """Use torchvision's ResNet-50 V1.5; last mode exports only Layer4."""
    backbone_name = "resnet50"

    def __init__(self, feature_mode="multistage"):
        super().__init__()
        validate_backbone_options(self.backbone_name, feature_mode)
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            self.add_module(name, getattr(model, name))
        self.feature_mode = feature_mode

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        endpoints = {}
        for index in range(1, 5):
            x = getattr(self, f"layer{index}")(x)
            if self.feature_mode == "multistage" or index == 4:
                endpoints[f"reduction_{index + 1}"] = x
        return endpoints


def build_cnn_backbone(backbone, feature_mode):
    validate_backbone_options(backbone, feature_mode)
    if backbone == "convnext_tiny":
        return ConvNeXtFeatures(feature_mode)
    if backbone == "resnet50":
        return ResNetFeatures(feature_mode)
    raise ValueError("The archived PVT encoder must be constructed by StudyModel")


def load_backbone(encoder, path, backbone=None):
    """Import only a matching local encoder, never silently filter feature keys.

    Accept official torchvision CNN or archived PVT flat state dictionaries,
    optionally wrapped in ``state_dict``. Only explicitly named classification
    head tensors are discarded. Other keys and tensor shapes must match exactly.
    """
    backbone = backbone or getattr(encoder, "backbone_name", "pvt_v2_b2")
    if backbone not in BACKBONE_NAMES:
        raise ValueError(f"Unknown backbone: {backbone}")
    state = torch.load(Path(path).expanduser().resolve(strict=True), map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k, v in state.items()):
        raise ValueError("Backbone checkpoint must be a flat tensor state dict or {'state_dict': ...}; "
                         "CNN checkpoints must use torchvision keys")
    state = dict(state)
    for name in CLASSIFIER_KEYS[backbone]:
        state.pop(name, None)
    encoder.load_state_dict(state, strict=True)
