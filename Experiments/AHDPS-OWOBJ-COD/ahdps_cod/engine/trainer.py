from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from ahdps_cod.engine.evaluator import evaluate
from ahdps_cod.utils.common import AverageMeter, ensure_dir, save_json


def build_optimizer(model: torch.nn.Module, cfg_train: Dict) -> torch.optim.Optimizer:
    opt_cfg = cfg_train["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 2e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise KeyError(f"Unsupported optimizer: {name}")


class CosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int = 0, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = max(warmup_epochs, 0)
        self.min_lr = float(min_lr)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            factor = float(epoch + 1) / float(self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = self.min_lr + (base_lr - self.min_lr) * factor


def _get_selection_metric(dev_metrics: Dict[str, float], cfg_train: Dict) -> float:
    metric_name = str(cfg_train.get("selection_metric", "Smeasure"))
    if metric_name not in dev_metrics:
        raise KeyError(f"selection_metric '{metric_name}' not found in dev metrics: {list(dev_metrics.keys())}")
    value = float(dev_metrics[metric_name])
    if metric_name.upper() == "MAE":
        return -value
    return value


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader,
    dev_loader,
    eval_loaders,
    cfg: Dict,
) -> Dict[str, object]:
    cfg_train = cfg["train"]
    device = torch.device(cfg_train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(cfg_train.get("amp", True)) and device.type == "cuda"
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = build_optimizer(model, cfg_train)
    scheduler = CosineScheduler(
        optimizer=optimizer,
        total_epochs=int(cfg_train["epochs"]),
        warmup_epochs=int(cfg_train.get("scheduler", {}).get("warmup_epochs", 0)),
        min_lr=float(cfg_train.get("scheduler", {}).get("min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)

    ckpt_dir = ensure_dir(cfg["paths"]["results"]["checkpoints"])
    metric_dir = ensure_dir(cfg["paths"]["results"]["metrics"])

    best_dev = float("-inf")
    best_state = None
    history = []

    for epoch in range(int(cfg_train["epochs"])):
        scheduler.step(epoch)
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter("loss")
        progress = tqdm(train_loader, desc=f"train {epoch+1}/{cfg_train['epochs']}", leave=False)

        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with (torch.amp.autocast("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=amp_enabled)):
                outputs = model(images, masks=masks)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict["loss_total"]

            scaler.scale(loss).backward()
            clip_norm = float(cfg_train.get("clip_grad_norm", 0.0))
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=images.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        dev_metrics = evaluate(model, dev_loader, device=device)
        summary = {
            "epoch": epoch + 1,
            "train_loss": loss_meter.avg,
            "dev": dev_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(summary)

        ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg}, ckpt_path)

        dev_score = _get_selection_metric(dev_metrics, cfg_train)
        if dev_score > best_dev:
            best_dev = dev_score
            best_state = {"model": model.state_dict(), "epoch": epoch + 1, "config": cfg}
            torch.save(best_state, ckpt_dir / "best.pt")

        save_json(history, metric_dir / "train_history.json")

    final_results = {"history": history}
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    full_eval = {}
    for split_alias, loader in eval_loaders.items():
        full_eval[split_alias] = evaluate(model, loader, device=device)
    final_results["eval"] = full_eval
    save_json(final_results, metric_dir / "final_eval.json")
    return final_results
