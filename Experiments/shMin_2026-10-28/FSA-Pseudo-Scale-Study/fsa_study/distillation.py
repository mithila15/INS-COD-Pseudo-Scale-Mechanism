"""Optional frozen real-image-scale feature targets, used during training only."""
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import sha256_file


def scale_feature_loss(student_banks, teacher_banks):
    """Mean channel-vector cosine distance for large/small branches, not masks."""
    if len(student_banks) != 1 or len(teacher_banks) != 1:
        raise ValueError("Feature distillation requires one final-stage feature bank")
    student, teacher = student_banks[0], teacher_banks[0]
    if len(student) != 3 or len(teacher) != 3:
        raise ValueError("Feature banks must be ordered main, large, small")
    terms = []
    for prediction, target in zip(student[1:], teacher[1:]):
        if prediction.shape[:2] != target.shape[:2]:
            raise ValueError("Student/teacher batch and channel dimensions must match")
        target = F.interpolate(target.detach().float(), size=prediction.shape[-2:],
                               mode="bilinear", align_corners=False)
        terms.append((1 - F.cosine_similarity(prediction.float(), target, dim=1, eps=1e-6)).mean())
    return torch.stack(terms).mean()


class FrozenScaleTeacher:
    """Kept outside StudyModel, so student state_dict/inference never includes it."""
    def __init__(self, model, metadata):
        self.model = model.eval().requires_grad_(False)
        self.metadata = metadata

    def loss(self, student_banks, image):
        self.model.eval()
        with torch.no_grad():
            target_banks = self.model.image_pyramid_banks(image)
        return scale_feature_loss(student_banks, target_banks)


def load_feature_teacher(path, manifest, student_config, training_protocol, size, device):
    """Reject untrained, unrelated or differently split teacher checkpoints."""
    from .engine import validate_checkpoint, checkpoint_model_kwargs
    from .model import StudyModel
    path = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, manifest)
    model_config, run = checkpoint["model_config"], checkpoint.get("run_config", {})
    if run.get("model_config") != model_config or run.get("manifest_sha256") != manifest["content_sha256"]:
        raise ValueError("Teacher run/model metadata is inconsistent")
    if (model_config.get("input_mode", "single") != "image_pyramid"
            or model_config.get("utilizer") != "mean" or model_config.get("baseline")
            or model_config.get("legacy_reference")):
        raise ValueError("Teacher must be the trained image_pyramid__mean reference")
    if (training_protocol != "full_train" or run.get("training_protocol") != "full_train"
            or run.get("checkpoint_selection") != "fixed final epoch; no validation or test selection"
            or checkpoint.get("epoch", 0) < 1 or checkpoint.get("epoch") != run.get("epochs")
            or checkpoint.get("step", 0) < 1):
        raise ValueError("Teacher must be a completed full_train fixed-final-epoch checkpoint")
    for key, default in (("backbone", "pvt_v2_b2"), ("feature_mode", "multistage"),
                         ("output_stride", 32), ("channels", 48)):
        if model_config.get(key, default) != student_config.get(key, default):
            raise ValueError(f"Teacher/student {key} mismatch")
    if model_config.get("feature_mode") != "last" or run.get("size") != size:
        raise ValueError("Teacher must use last mode and the same input size")
    model = StudyModel(**checkpoint_model_kwargs(model_config, StudyModel))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    metadata = {"checkpoint_sha256": sha256_file(path), "model_config": model_config,
                "manifest_sha256": manifest["content_sha256"], "epoch": checkpoint["epoch"],
                "source_sha256": run.get("source_sha256"), "input_scales": [1., 1.5, .5],
                "targets": "aligned projected large/small features; no mask pseudo-labels",
                "loss": "mean channel cosine distance over two auxiliary branches; eps=1e-6",
                "frozen": True, "inference_teacher": False}
    return FrozenScaleTeacher(model, metadata)
