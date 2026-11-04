"""Architecture/experiment identities, importable without torch."""
import math


def variant_name(config):
    if config.get("baseline", False):
        return "single"
    if config.get("legacy_reference", False):
        return "legacy_full"
    if config.get("input_mode", "single") == "image_pyramid":
        return f"image_pyramid__{config['utilizer']}"
    return f"{config['generator']}__{config['utilizer']}"


def validate_scale_options(backbone, feature_mode, output_stride=32, input_mode="single",
                           baseline=False, legacy_reference=False):
    if output_stride not in (16, 32):
        raise ValueError("output_stride must be 16 or 32")
    if output_stride == 16 and (backbone != "convnext_tiny" or feature_mode != "last"):
        raise ValueError("Stride16 adaptation requires ConvNeXt-Tiny and feature_mode=last")
    if input_mode not in ("single", "image_pyramid"):
        raise ValueError("input_mode must be single or image_pyramid")
    if input_mode == "image_pyramid" and (feature_mode != "last" or baseline or legacy_reference):
        raise ValueError("Image-pyramid reference requires last mode without baseline/legacy flags")


def validate_distillation_options(teacher, weight, model):
    if not math.isfinite(weight) or weight < 0 or bool(teacher) != (weight > 0):
        raise ValueError("Supply both --distill-teacher and positive --distill-weight, or neither")
    if teacher and (model.get("feature_mode") != "last" or model.get("baseline")
                    or model.get("legacy_reference") or model.get("input_mode", "single") != "single"
                    or model.get("generator") not in {"native_conv", "projection"}
                    or model.get("generator_options")):
        raise ValueError("Feature distillation supports last-mode native_conv/projection with default scale options")
