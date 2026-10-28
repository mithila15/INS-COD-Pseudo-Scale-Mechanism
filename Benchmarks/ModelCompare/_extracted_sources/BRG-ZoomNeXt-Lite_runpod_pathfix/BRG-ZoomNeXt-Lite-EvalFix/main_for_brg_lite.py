# -*- coding: utf-8 -*-
"""Train/evaluate BRG-ZoomNeXt-Lite on COD datasets.

This script is intentionally close to the original ZoomNeXt image trainer, but
uses only the main-scale image input and supports edge supervision.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import albumentations as A
import colorlog
import cv2
import numpy as np
import torch
import yaml
from mmengine import Config
from torch.utils import data
from tqdm import tqdm

import methods as model_zoo
from utils import io, ops, pipeline, pt_utils, py_utils, recorder

LOGGER = logging.getLogger("main")
LOGGER.propagate = False
LOGGER.setLevel(level=logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s[%(filename)s] %(reset)s%(message)s"))
LOGGER.addHandler(stream_handler)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _normalize_windows_path(path: str) -> str:
    """Accept Windows/Linux paths and environment variables.

    Examples:
        E:\\COD\\datasets -> E:/COD/datasets on Windows-compatible parsing
        ${DATASETS_ROOT}/COD10K-v3/Train -> /workspace/COD/datasets/COD10K-v3/Train on RunPod
    """
    path = os.path.expanduser(os.path.expandvars(str(path)))
    return path.replace("\\", os.sep)



def _dataset_root_score(root: str) -> int:
    """Return how many expected COD folders exist under a candidate root."""
    if not root:
        return 0
    root = _normalize_windows_path(root)
    expected_rel_paths = [
        "COD10K-v3/Train/Image",
        "COD10K-v3/Train/GT_Object",
        "COD10K-v3/Test/Image",
        "COD10K-v3/Test/GT_Object",
        "CAMO-V.1.0-CVIU2019/Images/Train",
        "CAMO-V.1.0-CVIU2019/Images/Test",
        "CAMO-V.1.0-CVIU2019/GT",
        "CHAMELEON/Imgs",
        "CHAMELEON/GT",
        "NC4K/Imgs",
        "NC4K/GT",
    ]
    score = 0
    for rel in expected_rel_paths:
        if os.path.isdir(os.path.join(root, *rel.split("/"))):
            score += 1
    return score


def _scan_for_dataset_roots(max_depth: int = 6) -> List[str]:
    """Search RunPod workspace for a folder that contains COD dataset folders."""
    bases = [Path("/workspace"), Path.cwd(), Path.cwd().parent]
    known_dataset_dirs = {"COD10K-v3", "CAMO-V.1.0-CVIU2019", "CHAMELEON", "NC4K"}
    roots: List[str] = []
    seen = set()

    for base in bases:
        if not base.exists() or not base.is_dir():
            continue
        try:
            base = base.resolve()
        except Exception:
            pass

        for dirpath, dirnames, _filenames in os.walk(str(base)):
            try:
                rel_depth = len(Path(dirpath).resolve().relative_to(base).parts)
            except Exception:
                rel_depth = 0
            if rel_depth > max_depth:
                dirnames[:] = []
                continue

            if known_dataset_dirs.intersection(set(dirnames)):
                candidate = str(Path(dirpath).resolve())
                if candidate not in seen:
                    seen.add(candidate)
                    roots.append(candidate)
                # Do not descend into large dataset image/mask folders once a root is found.
                dirnames[:] = [d for d in dirnames if d not in known_dataset_dirs]
    return roots


def _select_datasets_root(preferred: Optional[str] = None) -> str:
    """Select a valid DATASETS_ROOT.

    Priority:
      1. Explicit DATASETS_ROOT if it points to the expected folders.
      2. Common RunPod paths.
      3. Auto-discovered parent of COD10K-v3/CAMO/CHAMELEON/NC4K.
    """
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)

    candidates.extend(
        [
            "/workspace/datasets",
            "/workspace/COD/datasets",
            "/workspace/cod/datasets",
            "/workspace/data/datasets",
            "/workspace/data",
            str(Path.cwd() / "datasets"),
            str(Path.cwd().parent / "datasets"),
        ]
    )
    candidates.extend(_scan_for_dataset_roots())

    best_root = _normalize_windows_path(preferred or "/workspace/COD/datasets")
    best_score = -1
    seen = set()
    for cand in candidates:
        cand = _normalize_windows_path(cand)
        if cand in seen:
            continue
        seen.add(cand)
        score = _dataset_root_score(cand)
        if score > best_score:
            best_root = cand
            best_score = score

    if preferred and _dataset_root_score(preferred) == 0 and best_score > 0:
        LOGGER.warning(
            f"DATASETS_ROOT={_normalize_windows_path(preferred)} does not contain the expected COD folders. "
            f"Auto-detected DATASETS_ROOT={best_root}"
        )
    return best_root


def _format_dataset_root_hint(missing_folder: str) -> str:
    roots = _scan_for_dataset_roots()
    root_scores = [(r, _dataset_root_score(r)) for r in roots]
    root_scores = sorted(root_scores, key=lambda x: x[1], reverse=True)
    lines = [
        f"Current DATASETS_ROOT={os.environ.get('DATASETS_ROOT', '<unset>')}",
        "Expected layout example:",
        "  ${DATASETS_ROOT}/COD10K-v3/Train/Image",
        "  ${DATASETS_ROOT}/COD10K-v3/Train/GT_Object",
        "  ${DATASETS_ROOT}/CAMO-V.1.0-CVIU2019/Images/Train",
        "Quick check:",
        "  find /workspace -maxdepth 6 -type d -name COD10K-v3",
    ]
    if root_scores:
        lines.append("Auto-detected candidate roots:")
        for root, score in root_scores[:10]:
            lines.append(f"  score={score:02d}  {root}")
    else:
        lines.append("No candidate dataset root was found under /workspace.")
    lines.append(f"Missing folder: {missing_folder}")
    return "\n".join(lines)


def _expand_dataset_info_roots(dataset_infos: dict) -> dict:
    """Expand ${DATASETS_ROOT} in dataset yaml after auto-detection."""
    for _name, info in dataset_infos.items():
        if isinstance(info, dict) and "root" in info:
            info["root"] = _normalize_windows_path(info["root"])
    return dataset_infos


def _collect_by_stem(folder: str, suffixes: Iterable[str]) -> Dict[str, str]:
    folder = _normalize_windows_path(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}\n{_format_dataset_root_hint(folder)}")
    files: Dict[str, str] = {}
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith(tuple(suffixes)):
            stem = os.path.splitext(name)[0]
            files.setdefault(stem, os.path.join(folder, name))
    return files


def _resolve_path(root: str, child: Optional[str]) -> str:
    root = _normalize_windows_path(root)
    if child is None or child == "":
        return root
    child = _normalize_windows_path(child)
    return child if os.path.isabs(child) else os.path.join(root, child)


def _pairs_from_dataset_info(dataset_info: dict, require_mask: bool = True) -> List[Tuple[str, str, Optional[str]]]:
    root = dataset_info["root"]
    image_path = _resolve_path(root, dataset_info["image"]["path"])
    mask_path = _resolve_path(root, dataset_info["mask"]["path"])
    edge_info = dataset_info.get("edge", None)
    edge_path = _resolve_path(root, edge_info["path"]) if edge_info and edge_info.get("path") else None

    image_exts = tuple(dataset_info.get("image", {}).get("exts", IMAGE_EXTS))
    mask_exts = tuple(dataset_info.get("mask", {}).get("exts", MASK_EXTS))
    edge_exts = tuple(edge_info.get("exts", MASK_EXTS)) if edge_info else MASK_EXTS

    images = _collect_by_stem(image_path, image_exts)
    masks = _collect_by_stem(mask_path, mask_exts) if require_mask else {}
    edges = _collect_by_stem(edge_path, edge_exts) if edge_path and os.path.isdir(edge_path) else {}

    valid_names = sorted(set(images).intersection(masks)) if require_mask else sorted(images)
    pairs = []
    for stem in valid_names:
        pairs.append((images[stem], masks.get(stem, ""), edges.get(stem)))
    return pairs


def mask_to_edge(mask01: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Generate a binary edge map from a binary mask when GT_Edge is absent."""
    m = (mask01 > 0).astype(np.uint8) * 255
    kernel = np.ones((ksize, ksize), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    edge = ((dil - ero) > 0).astype(np.float32)
    return edge


class BRGImageTrainDataset(data.Dataset):
    def __init__(self, dataset_infos: dict, shape: dict):
        super().__init__()
        self.shape = shape
        self.total_data_paths: List[Tuple[str, str, Optional[str]]] = []
        for dataset_name, dataset_info in dataset_infos.items():
            data_paths = _pairs_from_dataset_info(dataset_info, require_mask=True)
            LOGGER.info(f"Length of {dataset_name}: {len(data_paths)}")
            self.total_data_paths.extend(data_paths)
        if not self.total_data_paths:
            raise RuntimeError("No training images were found. Check configs/dataset_cod_runpod.yaml and DATASETS_ROOT.")

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ],
            additional_targets={"edge": "mask"},
        )

    def __getitem__(self, index):
        image_path, mask_path, edge_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w, interpolation=cv2.INTER_LINEAR)

        if edge_path and os.path.exists(edge_path):
            edge = io.read_gray_array(edge_path, thr=0)
            if edge.shape != mask.shape:
                edge = ops.resize(edge, height=mask.shape[0], width=mask.shape[1], interpolation=cv2.INTER_NEAREST)
        else:
            edge = mask_to_edge(mask)

        transformed = self.trains(image=image, mask=mask.astype(np.float32), edge=edge.astype(np.float32))
        image = transformed["image"]
        mask = transformed["mask"]
        edge = transformed["edge"]

        base_h = self.shape["h"]
        base_w = self.shape["w"]
        image_m = ops.resize(image, height=base_h, width=base_w, interpolation=cv2.INTER_LINEAR)
        mask = ops.resize(mask, height=base_h, width=base_w, interpolation=cv2.INTER_NEAREST)
        edge = ops.resize(edge, height=base_h, width=base_w, interpolation=cv2.INTER_NEAREST)

        image_m = torch.from_numpy(image_m).div(255).permute(2, 0, 1).float()
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        edge = torch.from_numpy((edge > 0).astype(np.float32)).unsqueeze(0)

        return dict(data={"image_m": image_m, "mask": mask, "edge": edge})

    def __len__(self):
        return len(self.total_data_paths)


class BRGImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: dict, shape: dict, group_name: str):
        super().__init__()
        self.shape = shape
        self.group_name = group_name
        self.total_data_paths = _pairs_from_dataset_info(dataset_info, require_mask=True)
        if not self.total_data_paths:
            raise RuntimeError(f"No test images were found for {group_name}")

    def __getitem__(self, index):
        image_path, mask_path, _edge_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        base_h = self.shape["h"]
        base_w = self.shape["w"]
        image_m = ops.resize(image, height=base_h, width=base_w, interpolation=cv2.INTER_LINEAR)
        image_m = torch.from_numpy(image_m).div(255).permute(2, 0, 1).float()
        return dict(data={"image_m": image_m}, info=dict(mask_path=mask_path, group_name=self.group_name))

    def __len__(self):
        return len(self.total_data_paths)


class Evaluator:
    def __init__(self, device: str, metric_names: List[str], clip_range=None):
        self.device = device
        self.clip_range = clip_range
        self.metric_names = metric_names

    @torch.no_grad()
    def eval(self, model, data_loader, save_path: str = ""):
        model.eval()
        all_metrics = recorder.GroupedCODMetricRecorder(metric_names=self.metric_names)

        for batch in tqdm(data_loader, total=len(data_loader), ncols=79, desc="[EVAL]"):
            batch_images = pt_utils.to_device(batch["data"], device=self.device)
            logits = model(data=batch_images)
            probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()
            # Per-image min-max for conventional COD metric evaluation.
            probs = probs - probs.reshape(probs.shape[0], -1).min(axis=1)[:, None, None]
            probs = probs / (probs.reshape(probs.shape[0], -1).max(axis=1)[:, None, None] + 1e-8)

            mask_paths = batch["info"]["mask_path"]
            group_names = batch["info"]["group_name"]
            for pred_idx, pred in enumerate(probs):
                mask_path = mask_paths[pred_idx]
                mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_array[mask_array > 0] = 255
                mask_h, mask_w = mask_array.shape
                pred = ops.resize(pred, height=mask_h, width=mask_w, interpolation=cv2.INTER_LINEAR)

                if self.clip_range is not None:
                    pred = ops.clip_to_normalize(pred, clip_range=self.clip_range)

                group_name = group_names[pred_idx]
                if save_path:
                    ops.save_array_as_image(
                        data_array=pred,
                        save_name=os.path.basename(mask_path),
                        save_dir=os.path.join(save_path, group_name),
                    )

                pred_uint8 = (pred * 255).astype(np.uint8)
                all_metrics.step(group_name=group_name, pre=pred_uint8, gt=mask_array, gt_path=mask_path)
        return all_metrics.show(digits=4, return_group=True)


def _format_metrics(metrics: dict) -> str:
    return ", ".join([f"{k}: {float(v):.4f}" for k, v in metrics.items()])


def _save_eval_summary(results: dict, save_dir: str, filename_prefix: str = "metrics") -> None:
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{filename_prefix}.json")
    csv_path = os.path.join(save_dir, f"{filename_prefix}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    metric_names = []
    for metrics in results.values():
        for key in metrics.keys():
            if key not in metric_names:
                metric_names.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + metric_names)
        for dataset_name, metrics in results.items():
            writer.writerow([dataset_name] + [metrics.get(k, "") for k in metric_names])


def _monitor_score(results: dict, monitor: str):
    if not results:
        return None, None
    monitor = str(monitor)
    if "." in monitor:
        dataset_name, metric_name = monitor.split(".", 1)
        if dataset_name not in results:
            return None, None
        metric_name = recorder.cod_metrics.canonical_metric_name(metric_name)
        value = results[dataset_name].get(metric_name, None)
        return value, f"{dataset_name}.{metric_name}"
    metric_name = recorder.cod_metrics.canonical_metric_name(monitor)
    values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
    if not values:
        return None, None
    return float(np.mean(values)), f"avg.{metric_name}"


def _state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _save_checkpoint(model, optimizer, epoch: int, iteration: int, metrics: dict, best: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "model": _state_dict(model),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "best": best,
        },
        save_path,
    )


def test(model, cfg):
    test_wrapper = Evaluator(device=cfg.device, metric_names=cfg.metric_names, clip_range=cfg.test.clip_range)

    all_results = {}
    for te_name in cfg.test.data.names:
        te_info = cfg.dataset_infos[te_name]
        te_dataset = BRGImageTestDataset(dataset_info=te_info, shape=cfg.test.data.shape, group_name=te_name)
        te_loader = data.DataLoader(
            dataset=te_dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True
        )
        LOGGER.info(f"Testing with testset: {te_name}: {len(te_dataset)}")

        save_path = cfg.path.save if cfg.save_results else ""
        if save_path:
            LOGGER.info(f"Results will be saved into {os.path.join(save_path, te_name)}")

        avg_results, group_results = test_wrapper.eval(model=model, data_loader=te_loader, save_path=save_path)
        seg_results = group_results.get(te_name, avg_results)
        all_results[te_name] = dict(seg_results)
        LOGGER.info(f"({te_name}): {py_utils.mapping_to_str(te_info)}\n{_format_metrics(seg_results)}")

    _save_eval_summary(all_results, save_dir=cfg.path.pth_log, filename_prefix="metrics_latest")
    return all_results

def train(model, cfg):
    tr_dataset = BRGImageTrainDataset(
        dataset_infos={data_name: cfg.dataset_infos[data_name] for data_name in cfg.train.data.names},
        shape=cfg.train.data.shape,
    )
    LOGGER.info(f"Total Length of Image Trainset: {len(tr_dataset)}")

    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )

    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    scaler = pipeline.Scaler(optimizer, cfg.train.use_amp, set_to_none=cfg.train.optimizer.set_to_none)

    LOGGER.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")
    LOGGER.info(f"Image Mean: {model.normalizer.mean.flatten()}, Image Std: {model.normalizer.std.flatten()}")
    if cfg.train.bn.freeze_encoder:
        LOGGER.info(" >>> Freeze Backbone !!! <<< ")
        model.encoder.requires_grad_(False)

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()
    train_start_time = time.perf_counter()
    best_record = {"score": None, "monitor": getattr(cfg, "monitor", "cod10k_te.wF"), "epoch": None}

    for _ in range(counter.num_epochs):
        LOGGER.info(f"Exp_Name: {cfg.exp_name}")
        model.train()
        if cfg.train.bn.freeze_status:
            pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)

        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)

            data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp and cfg.device.startswith("cuda")):
                outputs = model(data=data_batch, iter_percentage=counter.curr_percent)

            loss = outputs["loss"] / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            if counter.every_n_iters(cfg.train.grad_acc_step):
                scaler.update_grad()

            item_loss = loss.item()
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(value=item_loss, num=data_shape[0])

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                eta_seconds = iter_time_recorder.avg * (counter.num_total_iters - counter.curr_iter - 1)
                eta_string = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                progress = (
                    f"{counter.curr_iter}:{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )
                loss_info = f"{outputs['loss_str']} (M:{loss_recorder.global_avg:.5f}/C:{item_loss:.5f})"
                lr_info = f"LR: {optimizer.lr_string()}"
                LOGGER.info(f"{eta_string}({gpu_mem}) | {progress} | {lr_info} | {loss_info} | {data_shape}")
                cfg.tb_logger.write_to_tb("lr", optimizer.lr_groups(), counter.curr_iter)
                cfg.tb_logger.write_to_tb("iter_loss", item_loss, counter.curr_iter)
                cfg.tb_logger.write_to_tb("avg_loss", loss_recorder.global_avg, counter.curr_iter)

            if counter.curr_iter < 3:
                recorder.plot_results(
                    dict(img=data_batch["image_m"], msk=data_batch["mask"], edge=data_batch["edge"], **outputs["vis"]),
                    save_path=os.path.join(cfg.path.pth_log, "img", f"iter_{counter.curr_iter}.png"),
                )

            iter_time_recorder.update(value=time.perf_counter() - iter_start_time)
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        recorder.plot_results(
            dict(img=data_batch["image_m"], msk=data_batch["mask"], edge=data_batch["edge"], **outputs["vis"]),
            save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{counter.curr_epoch}.png"),
        )
        # Always keep the latest state for safe recovery.
        io.save_weight(model=model, save_path=cfg.path.latest_state_net)
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=counter.curr_epoch + 1,
            iteration=counter.curr_iter + 1,
            metrics={},
            best=best_record,
            save_path=cfg.path.latest_full_net,
        )

        if getattr(cfg, "save_epoch_interval", 0) and counter.every_n_epochs(cfg.save_epoch_interval):
            io.save_weight(model=model, save_path=os.path.join(cfg.path.pth, f"state_epoch_{counter.curr_epoch + 1:03d}.pth"))

        if getattr(cfg, "has_test", False) and getattr(cfg, "eval_interval", 0) and counter.every_n_epochs(cfg.eval_interval):
            eval_results = test(model=model, cfg=cfg)
            score, monitor_key = _monitor_score(eval_results, getattr(cfg, "monitor", "cod10k_te.wF"))
            if score is not None:
                LOGGER.info(f"Monitor {monitor_key}: {score:.4f} | best: {best_record['score']}")
                if getattr(cfg, "save_best", True) and recorder.cod_metrics.is_better_metric(
                    monitor_key, score, best_record["score"], getattr(cfg, "monitor_mode", "auto")
                ):
                    best_record = {"score": float(score), "monitor": monitor_key, "epoch": counter.curr_epoch + 1}
                    io.save_weight(model=model, save_path=cfg.path.best_state_net)
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=counter.curr_epoch + 1,
                        iteration=counter.curr_iter + 1,
                        metrics=eval_results,
                        best=best_record,
                        save_path=cfg.path.best_full_net,
                    )
                    _save_eval_summary(eval_results, save_dir=cfg.path.pth_log, filename_prefix="metrics_best")
                    LOGGER.info(f"Saved new best checkpoint: {best_record}")

        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()
    io.save_weight(model=model, save_path=cfg.path.final_state_net)
    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=counter.curr_epoch,
        iteration=counter.curr_iter,
        metrics={},
        best=best_record,
        save_path=cfg.path.final_full_net,
    )
    total_train_time = time.perf_counter() - train_start_time
    total_other_time = datetime.timedelta(seconds=int(total_train_time - iter_time_recorder.global_sum))
    LOGGER.info(f"Total Training Time: {datetime.timedelta(seconds=int(total_train_time))} ({total_other_time} on others)")


def parse_cfg():
    parser = argparse.ArgumentParser("BRG-ZoomNeXt-Lite trainer")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", type=str, default="configs/dataset_cod_runpod.yaml")
    parser.add_argument("--model-name", type=str, choices=model_zoo.__dict__.keys())
    parser.add_argument("--output-dir", type=str, default="outputs_brg_lite")
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--metric-names",
        nargs="+",
        type=str,
        default=["S", "wF", "MAE", "adpE", "meanE", "maxE"],
        choices=recorder.GroupedCODMetricRecorder.supported_metrics,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    # RunPod convention: datasets may be mounted in different locations.
    # If DATASETS_ROOT is missing or invalid, auto-detect the parent folder that
    # contains COD10K-v3, CAMO-V.1.0-CVIU2019, CHAMELEON, and/or NC4K.
    preferred_root = os.environ.get("DATASETS_ROOT", "/workspace/COD/datasets")
    os.environ["DATASETS_ROOT"] = _select_datasets_root(preferred_root)
    LOGGER.info(f"DATASETS_ROOT = {os.environ['DATASETS_ROOT']}")

    with open(cfg.data_cfg, mode="r", encoding="utf-8") as f:
        cfg.dataset_infos = _expand_dataset_info_roots(yaml.safe_load(f))

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, cfg.output_dir)
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.path.latest_state_net = os.path.join(cfg.path.pth, "state_latest.pth")
    cfg.path.best_state_net = os.path.join(cfg.path.pth, "state_best.pth")
    cfg.path.latest_full_net = os.path.join(cfg.path.pth, "checkpoint_latest.pth")
    cfg.path.best_full_net = os.path.join(cfg.path.pth, "checkpoint_best.pth")
    cfg.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    file_handler = logging.FileHandler(cfg.path.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(filename)s] %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(cfg.pretty_text)
    cfg.tb_logger = recorder.TBLogger(tb_root=cfg.path.tb)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)

    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model_kwargs = dict(getattr(cfg, "model", {}))
    model_kwargs.update(dict(pretrained=cfg.pretrained, use_checkpoint=cfg.use_checkpoint, num_frames=1))
    model = model_class(**model_kwargs)
    LOGGER.info(inspect.getsource(model_class))
    model.to(cfg.device)

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=False)

    LOGGER.info(f"Number of Parameters: {sum(v.numel() for v in model.parameters(recurse=True))}")
    if not cfg.evaluate:
        train(model=model, cfg=cfg)

    if cfg.evaluate or cfg.has_test:
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        test(model=model, cfg=cfg)

    LOGGER.info("End training...")


if __name__ == "__main__":
    main()
