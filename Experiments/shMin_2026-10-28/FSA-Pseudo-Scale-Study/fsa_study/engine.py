"""Opt-in training and checkpoint evaluation with training-only model selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .data import CODDataset, load_manifest, sha256_file


def seed_everything(seed, determinism="warn"):
    if not 0 <= seed < 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # This is recorded, not a promise of bitwise equality across hardware/releases.
    torch.use_deterministic_algorithms(True, warn_only=determinism == "warn")


def weighted_bce_iou(logits, target):
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    weights = 1 + 5 * (F.avg_pool2d(target, 31, 1, 15) - target).abs()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weights * bce).sum((2, 3)) / (weights.sum((2, 3)) + 1e-7)
    probability = logits.sigmoid()
    intersection = (probability * target * weights).sum((2, 3))
    union = ((probability + target) * weights).sum((2, 3))
    iou = 1 - (intersection + 1) / (union - intersection + 1 + 1e-7)
    return (bce + iou).mean()


def dice_loss(logits, target):
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((2, 3))
    denominator = probability.sum((2, 3)) + target.sum((2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1 + 1e-7)).mean()


def study_loss(outputs, mask, edge, progress):
    terms = {"seg": weighted_bce_iou(outputs["logits"], mask)}
    deep = outputs["deep_logits"]
    if deep:
        terms["deep"] = 0.25 * sum(weighted_bce_iou(logits, mask) for logits in deep) / len(deep)
    terms["region"] = 0.5 * weighted_bce_iou(outputs["po"], mask)
    terms["edge"] = dice_loss(outputs["pe"], edge)
    coefficient = 0.5 * (1 - math.cos(math.pi * min(1.0, max(0.0, progress))))
    probability = outputs["logits"].sigmoid()
    terms["ual"] = coefficient * (1 - (2 * probability - 1).abs().square()).mean()
    return sum(terms.values()), terms


def _json_write(path, payload):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "timm", "einops", "pysodmetrics"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    repo = Path(__file__).resolve().parents[4]
    git = {}
    for key, command in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        try:
            result = subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True, check=True)
            git[key] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git[key] = "unavailable"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "git": git,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _source_hashes():
    directory = Path(__file__).resolve().parent.parent
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*.py")) if "results" not in path.relative_to(directory).parts}


def _loader(dataset, batch_size, workers, seed, shuffle=False, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                      drop_last=drop_last, generator=generator, persistent_workers=False,
                      pin_memory=torch.cuda.is_available())


def _official_metrics():
    # Kept as a dependency-check helper for existing callers.
    from .metrics import official_metrics
    return official_metrics()


@torch.inference_mode()
def evaluate_model(model, loader, device, official=False, output=None, save_predictions=True):
    model.eval()
    totals, counts = defaultdict(float), defaultdict(int)
    recorder = None
    if official:
        from .metrics import METRIC_PROTOCOL, MetricRecorder, protocol_details
        recorder = MetricRecorder(output, save_predictions)
    for batch in loader:
        outputs = model(batch["image"].to(device, non_blocking=True))
        if not torch.isfinite(outputs["logits"]).all():
            raise FloatingPointError("Nonfinite evaluation logits; refusing to produce metrics")
        probabilities = outputs["logits"].sigmoid()
        for index, (mask_path, dataset) in enumerate(zip(batch["mask_path"], batch["dataset"])):
            with Image.open(mask_path) as handle:
                mask = np.array(handle.convert("L")) > 0
            prediction = probabilities[index:index + 1].float()
            # Match legacy COD convention: min-max at model resolution, then resize.
            prediction = prediction - prediction.amin(dim=(-2, -1), keepdim=True)
            prediction = prediction / (prediction.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            prediction = F.interpolate(prediction, size=mask.shape, mode="bilinear", align_corners=False)
            prediction = prediction[0, 0].cpu().numpy().clip(0, 1)
            counts[dataset] += 1
            if official:
                prediction_uint8 = (prediction * 255).astype(np.uint8)
                mask_uint8 = mask.astype(np.uint8) * 255
                sample_id = batch["id"][index] if "id" in batch else str(mask_path)
                recorder.step(prediction_uint8, mask_uint8, dataset, sample_id, mask_path)
            else:
                totals[dataset] += float(np.abs(prediction - mask.astype(np.float32)).mean())
    if not counts:
        raise ValueError("Evaluation loader is empty")
    if official:
        results, artifacts = recorder.finish()
    else:
        results = {dataset: {"count": count, "MAE": totals[dataset] / count}
                   for dataset, count in sorted(counts.items())}
    # Validation selection is a sample-weighted mean, not test-set selection.
    mae = sum(values["MAE"] * values["count"] for values in results.values()) / sum(counts.values())
    report = {"datasets": results, "sample_weighted_MAE": mae,
              "metric_protocol": METRIC_PROTOCOL if official else "float_MAE",
              "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0"}
    if official:
        report.update({"metric_protocol_details": protocol_details(), "artifacts": artifacts})
    return report


def _validate_args(args):
    if args.size < 64 or args.size % 32:
        raise ValueError("--size must be >=64 and divisible by 32 for the shared PVT decoder")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu explicitly")
    if Path(args.output).expanduser().exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {args.output}")


def _freeze_encoder_bn(model):
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def _optimizer(model, learning_rate):
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    fixed_ids = {id(parameter) for parameter in model.encoder.patch_embed1.parameters()}
    groups = {"encoder": [], "other": []}
    for parameter in model.parameters():
        if id(parameter) in fixed_ids:
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            groups["encoder" if id(parameter) in encoder_ids else "other"].append(parameter)
    return torch.optim.Adam([{"params": groups["encoder"], "lr": learning_rate * 0.1},
                             {"params": groups["other"], "lr": learning_rate}], weight_decay=0)


def _model_config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_checkpoint(checkpoint, manifest):
    """Validate exact variant metadata before constructing or loading any model."""
    if checkpoint.get("schema_version") != 1 or "model_config" not in checkpoint:
        raise ValueError("Expected a study best.pth/last.pth checkpoint containing exact variant metadata")
    if checkpoint.get("model_config_sha256") != _model_config_digest(checkpoint["model_config"]):
        raise ValueError("Checkpoint variant metadata checksum mismatch")
    if checkpoint.get("manifest_sha256") != manifest["content_sha256"]:
        raise ValueError("Checkpoint was created from a different dataset manifest")


def train(args):
    """Only called by explicit `train`; no other command performs optimizer steps."""
    _validate_args(args)
    if args.epochs < 1 or args.lr <= 0 or args.eval_interval < 1:
        raise ValueError("--epochs, --lr and --eval-interval must be positive")
    manifest, root = load_manifest(args.manifest, verify_files=True, data_root=args.data_root)
    train_data = CODDataset(manifest, root, "train", args.size, args.seed)
    val_data = CODDataset(manifest, root, "val", args.size, args.seed)
    if len(train_data) < args.batch_size:
        raise ValueError("Training split is smaller than --batch-size")
    # Decode one complete pass before allocating a model: catches corrupt/geometry inputs.
    from PIL import Image
    for record in manifest["records"]:
        if record["split"] not in {"train", "val"}:
            continue
        with Image.open(root / record["image"]) as image, Image.open(root / record["mask"]) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask geometry mismatch: {record['id']}")
            image.verify()
            mask.verify()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed, args.determinism)
    weights = None
    if args.backbone_weights:
        weights = Path(args.backbone_weights).expanduser().resolve(strict=True)
        weight_hash = sha256_file(weights)
    else:
        weight_hash = None
    from .model import StudyModel
    model_config = {"generator": args.generator, "utilizer": args.utilizer, "channels": args.channels,
                    "seed": args.seed, "baseline": args.baseline, "legacy_reference": args.legacy_reference,
                    "generator_options": args.generator_options, "utilizer_options": args.utilizer_options}
    model = StudyModel(**model_config, backbone_weights=str(weights) if weights else None).to(args.device)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if key != "handler"}
    config.update({"model_config": model_config, "manifest_sha256": manifest["content_sha256"],
                   "backbone_sha256": weight_hash, "backbone_initialization": "local_checkpoint" if weights else "random_explicit",
                   "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
                   "loss_weights": {"seg": 1, "deep": .25, "region": .5, "edge": 1, "ual": 1},
                   "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
                   "edge_protocol": "5x5 morphological gradient after nearest mask resize",
                   "model_metadata": model.metadata, "source_sha256": _source_hashes(), "environment": _environment()})
    _json_write(output / "config.json", config)
    _json_write(output / "manifest.json", manifest)
    training_loader = _loader(train_data, args.batch_size, args.workers, args.seed, shuffle=True, drop_last=True)
    validation_loader = _loader(val_data, args.batch_size, args.workers, args.seed)
    _freeze_encoder_bn(model)
    optimizer = _optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, int(args.epochs * 2 / 3))], gamma=.1)
    best_mae = float("inf")
    total_steps = args.epochs * len(training_loader)
    step = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_data.epoch = epoch
        model.train()
        _freeze_encoder_bn(model)
        loss_totals = defaultdict(float)
        for batch in training_loader:
            image = batch["image"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)
            edge = batch["edge"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=False)
            outputs = model(image)
            loss, terms = study_loss(outputs, mask, edge, step / max(1, total_steps - 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, step={step}")
            loss.backward()
            optimizer.step()
            for key, value in terms.items():
                loss_totals[key] += float(value.detach())
            step += 1
        scheduler.step()
        validation = None
        improved = False
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            validation = evaluate_model(model, validation_loader, args.device, official=False)
            value = validation["sample_weighted_MAE"]
            if not math.isfinite(value):
                raise FloatingPointError("Validation MAE is not finite")
            improved = value < best_mae
            if improved:
                best_mae = value
        checkpoint = {"schema_version": 1, "model": model.state_dict(), "model_config": model_config,
                      "run_config": config, "model_config_sha256": _model_config_digest(model_config),
                      "manifest_sha256": manifest["content_sha256"],
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch, "step": step, "best_val_mae": best_mae,
                      "validation": validation, "torch_rng_state": torch.get_rng_state()}
        temporary = output / "last.tmp.pth"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pth")
        if improved:
            torch.save(checkpoint, output / "best.pth")
        row = {"epoch": epoch, "step": step, "seconds_elapsed": time.perf_counter() - start,
               "train_losses": {key: value / len(training_loader) for key, value in loss_totals.items()},
               "validation": validation, "best_val_mae": best_mae if math.isfinite(best_mae) else None}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    _json_write(output / "status.json", {"status": "training_complete", "epochs": args.epochs,
                                        "optimizer_steps": step, "best_val_mae": best_mae,
                                        "test_evaluation": "not_run"})
    return output


def evaluate(args):
    if Path(args.output).expanduser().exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {args.output}")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    # Full input checks precede model allocation; test metric dependency is mandatory.
    manifest, root = load_manifest(args.manifest, verify_files=True, data_root=args.data_root)
    if not any(record["split"] == args.split for record in manifest["records"]):
        raise ValueError(f"Manifest has no samples for requested split {args.split}")
    if args.split == "test":
        _official_metrics()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, manifest)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(checkpoint["model_config"]["seed"], checkpoint["run_config"].get("determinism", "warn"))
    from .model import StudyModel
    model = StudyModel(**checkpoint["model_config"], backbone_weights=None)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    size = checkpoint["run_config"]["size"]
    dataset = CODDataset(manifest, root, args.split, size, checkpoint["model_config"]["seed"])
    loader = _loader(dataset, args.batch_size, args.workers, checkpoint["model_config"]["seed"])
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = evaluate_model(model, loader, args.device, official=args.split == "test", output=output,
                             save_predictions=not getattr(args, "no_save_predictions", False))
    results.update({"split": args.split, "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path), "epoch": checkpoint["epoch"],
                    "model_config": checkpoint["model_config"], "manifest_sha256": manifest["content_sha256"],
                    "environment": _environment(), "source_sha256": _source_hashes()})
    _json_write(output / "metrics.json", results)
    print(json.dumps(results, indent=2, allow_nan=False))
    return output
