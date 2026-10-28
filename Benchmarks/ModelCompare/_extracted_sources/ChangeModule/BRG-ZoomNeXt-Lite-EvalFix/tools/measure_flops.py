#!/usr/bin/env python3
"""Measure model complexity for original ZoomNeXt and BRG-ZoomNeXt-Lite.

This script intentionally uses the same input protocol for both models:
- Original ZoomNeXt receives image_s/image_m/image_l with scale 0.5/1.0/1.5.
- BRG-ZoomNeXt-Lite receives image_m only.

Use the same tool value for all models when reporting numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import methods as model_zoo  # noqa: E402


ORIGINAL_MODEL_NAMES = {
    "EffB1_ZoomNeXt",
    "EffB4_ZoomNeXt",
    "RN50_ZoomNeXt",
    "PvtV2B2_ZoomNeXt",
    "PvtV2B3_ZoomNeXt",
    "PvtV2B4_ZoomNeXt",
    "PvtV2B5_ZoomNeXt",
}


class OriginalZoomNeXtWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_m: torch.Tensor, image_s: torch.Tensor, image_l: torch.Tensor) -> torch.Tensor:
        return self.model({"image_m": image_m, "image_s": image_s, "image_l": image_l})


class BRGLiteWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_m: torch.Tensor) -> torch.Tensor:
        return self.model({"image_m": image_m})


def parse_bool_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y", "on"}


def build_model(args: argparse.Namespace) -> nn.Module:
    model_cls = getattr(model_zoo, args.model_name, None)
    if model_cls is None:
        raise ValueError(f"Unknown model name: {args.model_name}")

    kwargs: Dict[str, Any] = {
        "pretrained": args.pretrained,
        "num_frames": 1,
        "use_checkpoint": False,
    }
    if args.model_name.endswith("BRG_ZoomNeXt_Lite"):
        kwargs.update(
            mid_dim=args.mid_dim,
            hmu_groups=args.hmu_groups,
            use_rfrm=not args.use_rgpu_g2,
            use_gmffm=not args.no_gmffm,
            ual_weight=1.0,
            edge_weight=1.0,
            region_weight=0.5,
            deep_weight=0.25,
            kd_weight=0.0,
        )
    elif args.model_name.endswith("SRG_ZoomNeXt_Lite"):
        kwargs.update(
            mid_dim=args.mid_dim,
            use_compensation=not args.no_compensation,
            residual_refine_weight=args.residual_refine_weight,
            ual_weight=1.0,
            edge_weight=1.0,
            region_weight=0.5,
            deep_weight=0.25,
            kd_weight=0.0,
        )
    else:
        # Original ZoomNeXt default is mid_dim=64, siu_groups=4, hmu_groups=6.
        kwargs.update(mid_dim=args.original_mid_dim, siu_groups=args.siu_groups, hmu_groups=args.original_hmu_groups)

    model = model_cls(**kwargs).eval()
    return model


def make_inputs(model_name: str, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    image_m = torch.randn(1, 3, h, w, device=device)
    if model_name in ORIGINAL_MODEL_NAMES:
        hs, ws = max(1, h // 2), max(1, w // 2)
        hl, wl = int(round(h * 1.5)), int(round(w * 1.5))
        image_s = torch.randn(1, 3, hs, ws, device=device)
        image_l = torch.randn(1, 3, hl, wl, device=device)
        return image_m, image_s, image_l
    return (image_m,)


def wrap_model(model_name: str, model: nn.Module) -> nn.Module:
    if model_name in ORIGINAL_MODEL_NAMES:
        return OriginalZoomNeXtWrapper(model).eval()
    return BRGLiteWrapper(model).eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_with_fvcore(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[float, str]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("fvcore is not installed. Try `pip install fvcore` or use --tool thop/torchprofiler.") from exc
    with torch.no_grad():
        analysis = FlopCountAnalysis(wrapper, inputs)
        flops = float(analysis.total())
    return flops, "fvcore_total_ops"


def measure_with_thop(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[float, str]:
    try:
        from thop import profile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("thop is not installed. Try `pip install thop` or use --tool fvcore/torchprofiler.") from exc
    with torch.no_grad():
        macs, _ = profile(wrapper, inputs=inputs, verbose=False)
    return float(macs), "thop_macs"


def measure_with_torchprofiler(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[float, str]:
    # This measures only PyTorch profiler-supported FLOPs, mainly matmul/conv.
    # Use this only for same-tool comparisons, not for paper tables unless you document it.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if inputs[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        # Warm-up once.
        _ = wrapper(*inputs)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            _ = wrapper(*inputs)
    total = 0.0
    for evt in prof.key_averages():
        total += float(getattr(evt, "flops", 0) or 0)
    return total, "torchprofiler_flops"


def measure(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    model = build_model(args).to(device).eval()
    wrapper = wrap_model(args.model_name, model).to(device).eval()
    inputs = make_inputs(args.model_name, args.height, args.width, device)

    if args.tool == "fvcore":
        ops, unit_name = measure_with_fvcore(wrapper, inputs)
    elif args.tool == "thop":
        ops, unit_name = measure_with_thop(wrapper, inputs)
    elif args.tool == "torchprofiler":
        ops, unit_name = measure_with_torchprofiler(wrapper, inputs)
    else:
        raise ValueError(args.tool)

    out = {
        "model_name": args.model_name,
        "height": args.height,
        "width": args.width,
        "params": count_params(model),
        "params_M": count_params(model) / 1e6,
        "tool": args.tool,
        "unit_name": unit_name,
        "ops": ops,
        "ops_G": ops / 1e9,
        "note": "Use identical tool/input protocol when comparing. Original ZoomNeXt uses 0.5/1.0/1.5 input branches; BRG-Lite uses image_m only.",
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tool", choices=["fvcore", "thop", "torchprofiler"], default="fvcore")
    parser.add_argument("--pretrained", action="store_true", help="Usually keep off for FLOPs; pretrained weights do not change FLOPs.")

    # Original ZoomNeXt kwargs.
    parser.add_argument("--original-mid-dim", type=int, default=64)
    parser.add_argument("--siu-groups", type=int, default=4)
    parser.add_argument("--original-hmu-groups", type=int, default=6)

    # BRG-Lite kwargs.
    parser.add_argument("--mid-dim", type=int, default=48)
    parser.add_argument("--hmu-groups", type=int, default=2)
    parser.add_argument("--use-rgpu-g2", action="store_true", help="Use RGPU-g2 instead of RFRM for BRG-Lite.")
    parser.add_argument("--no-gmffm", action="store_true")
    # SRG-Lite kwargs.
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--residual-refine-weight", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = measure(args)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model: {out['model_name']}")
        print(f"input: {out['height']}x{out['width']}")
        print(f"params: {out['params_M']:.3f} M")
        print(f"{out['unit_name']}: {out['ops_G']:.3f} G")
        print("note: multiply-add convention depends on tool. Use the same tool for original/lite.")


if __name__ == "__main__":
    main()
