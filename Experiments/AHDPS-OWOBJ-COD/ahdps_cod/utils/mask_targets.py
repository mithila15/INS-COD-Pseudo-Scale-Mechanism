from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def dilate(mask: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def erode(mask: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)


def resize_mask(mask: Tensor, size: Tuple[int, int], mode: str = "nearest") -> Tensor:
    return F.interpolate(mask, size=size, mode=mode)


def build_hdps_targets(
    masks: Tensor,
    core_kernel: int,
    halo_kernel: int,
    halo_target: float,
) -> Dict[str, Tensor]:
    masks = masks.float().clamp(0.0, 1.0)
    body = masks
    core = erode(body, core_kernel)
    halo_outer = dilate(body, halo_kernel)
    ambiguous_halo = (halo_outer - body).clamp(0.0, 1.0)
    boundary_band = (halo_outer - core).clamp(0.0, 1.0)
    p3_soft = (body + halo_target * ambiguous_halo).clamp(0.0, 1.0)
    return {
        "body": body,
        "core": core,
        "halo": ambiguous_halo,
        "boundary": boundary_band,
        "p3_soft": p3_soft,
    }


def _region_pool(mask: Tensor, grid_size: Tuple[int, int], mode: str = "avg") -> Tensor:
    if mode == "avg":
        return F.adaptive_avg_pool2d(mask, output_size=grid_size)
    if mode == "max":
        return F.adaptive_max_pool2d(mask, output_size=grid_size)
    raise KeyError(f"Unsupported region pool mode: {mode}")


def build_region_objectness_targets(
    masks: Tensor,
    grid_size: Tuple[int, int],
    core_kernel: int,
    halo_kernel: int,
    halo_target: float,
    pos_body_thresh: float,
    pos_core_thresh: float,
    neg_body_thresh: float,
    neg_halo_thresh: float,
    amb_halo_thresh: float,
    amb_boundary_thresh: float,
    amb_guidance_mix: float,
    guidance_map: Tensor | None = None,
) -> Dict[str, Tensor]:
    hdps = build_hdps_targets(
        masks=masks,
        core_kernel=core_kernel,
        halo_kernel=halo_kernel,
        halo_target=halo_target,
    )
    body = hdps["body"]
    core = hdps["core"]
    halo = hdps["halo"]
    boundary = hdps["boundary"]

    body_r = _region_pool(body, grid_size)
    core_r = _region_pool(core, grid_size)
    halo_r = _region_pool(halo, grid_size)
    boundary_r = _region_pool(boundary, grid_size)

    positive = (body_r >= pos_body_thresh) | (core_r >= pos_core_thresh)
    ambiguous = (~positive) & ((halo_r >= amb_halo_thresh) | (boundary_r >= amb_boundary_thresh))
    negative = (~positive) & (~ambiguous) & (body_r <= neg_body_thresh) & (halo_r <= neg_halo_thresh)
    valid = positive | ambiguous | negative

    if guidance_map is None:
        guidance_r = body_r
    else:
        guidance_r = _region_pool(guidance_map.float().clamp(0.0, 1.0), grid_size)

    amb_prior = 0.5 * guidance_r + 0.25 * halo_r + 0.25 * boundary_r
    amb_target = (
        amb_guidance_mix * amb_prior + (1.0 - amb_guidance_mix) * (0.5 * body_r + 0.5 * halo_r)
    ).clamp(0.05, 0.95)

    targets = torch.full_like(body_r, fill_value=-1.0)
    targets = torch.where(positive, torch.ones_like(targets), targets)
    targets = torch.where(negative, torch.zeros_like(targets), targets)
    targets = torch.where(ambiguous, amb_target, targets)

    return {
        "targets": targets.flatten(1),
        "positive": positive.flatten(1),
        "ambiguous": ambiguous.flatten(1),
        "negative": negative.flatten(1),
        "valid": valid.flatten(1),
        "body": body_r.flatten(1),
        "core": core_r.flatten(1),
        "halo": halo_r.flatten(1),
        "boundary": boundary_r.flatten(1),
        "guidance": guidance_r.flatten(1),
    }
