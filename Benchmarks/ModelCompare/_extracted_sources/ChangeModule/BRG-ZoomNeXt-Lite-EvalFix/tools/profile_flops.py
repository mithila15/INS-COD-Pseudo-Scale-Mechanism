# -*- coding: utf-8 -*-
"""Profile Params/MACs/FLOPs for original ZoomNeXt and BRG-ZoomNeXt-Lite.

This script intentionally uses the same dummy-input protocol for both models:
- original ZoomNeXt: image_s=0.5x, image_m=1.0x, image_l=1.5x
- BRG-Lite: image_m=1.0x only, because that is the proposed architecture

The counter reports MACs. FLOPs are also printed as 2 * MACs because papers and
profilers differ on whether one multiply-add is counted as one or two FLOPs.
Use the same column consistently when comparing models.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_model_class(name: str):
    methods = importlib.import_module("methods")
    if not hasattr(methods, name):
        available = sorted([x for x in dir(methods) if "ZoomNeXt" in x])
        raise KeyError(f"Unknown model class: {name}. Available ZoomNeXt classes: {available}")
    return getattr(methods, name)


def _num_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _conv2d_macs(module: nn.Conv2d, inputs: Tuple[Any, ...], output: torch.Tensor) -> int:
    x = inputs[0]
    if not isinstance(output, torch.Tensor):
        return 0
    batch = output.shape[0]
    out_channels = output.shape[1]
    out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
    k_h, k_w = module.kernel_size
    in_channels = module.in_channels
    groups = module.groups
    return int(batch * out_channels * out_h * out_w * (in_channels // groups) * k_h * k_w)


def _linear_macs(module: nn.Linear, inputs: Tuple[Any, ...], output: torch.Tensor) -> int:
    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        return 0
    # Linear applies to the last dimension. Count all leading positions.
    positions = 1
    for s in x.shape[:-1]:
        positions *= int(s)
    return int(positions * module.in_features * module.out_features)


def _attention_sdp_macs(module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor) -> int:
    """Count only q@k^T and attn@v inside PVTv2 Attention.

    q/kv/proj Linear layers and spatial-reduction Conv are counted separately
    by their own hooks. This hook adds the missing scaled-dot-product attention
    matmuls that common hook-based profilers often miss.
    """
    if len(inputs) < 3 or not isinstance(inputs[0], torch.Tensor):
        return 0
    x = inputs[0]
    H, W = int(inputs[1]), int(inputs[2])
    batch, n_tokens, channels = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    sr_ratio = int(getattr(module, "sr_ratio", 1))
    linear = bool(getattr(module, "linear", False))
    if linear:
        n_kv = 49  # PVTv2 linear attention uses AdaptiveAvgPool2d(7)
    elif sr_ratio > 1:
        n_kv = max(1, H // sr_ratio) * max(1, W // sr_ratio)
    else:
        n_kv = n_tokens
    # q @ k^T and attention @ v. num_heads * head_dim = channels.
    return int(2 * batch * n_tokens * n_kv * channels)


class HookBasedFlopCounter:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: List[Any] = []
        self.by_type: Dict[str, int] = defaultdict(int)
        self.by_module: Dict[str, int] = defaultdict(int)
        self._module_names = {m: n for n, m in model.named_modules()}
        self._attention_cls = None
        try:
            from methods.backbone.pvt_v2_eff import Attention
            self._attention_cls = Attention
        except Exception:
            self._attention_cls = None

    def _make_hook(self, kind: str, fn):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            try:
                macs = int(fn(module, inputs, output))
            except Exception:
                macs = 0
            if macs:
                self.by_type[kind] += macs
                name = self._module_names.get(module, module.__class__.__name__)
                self.by_module[f"{name} ({kind})"] += macs
        return hook

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._make_hook("Conv2d", _conv2d_macs)))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._make_hook("Linear", _linear_macs)))
            elif self._attention_cls is not None and isinstance(module, self._attention_cls):
                self.handles.append(module.register_forward_hook(self._make_hook("SDPA", _attention_sdp_macs)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @property
    def total_macs(self) -> int:
        return int(sum(self.by_type.values()))


def _dummy_inputs(model_name: str, batch_size: int, height: int, width: int, device: torch.device) -> Dict[str, torch.Tensor]:
    image_m = torch.randn(batch_size, 3, height, width, device=device)
    data: Dict[str, torch.Tensor] = {"image_m": image_m}
    if "BRG_ZoomNeXt_Lite" not in model_name:
        # Original ZoomNeXt uses three actual input scales: 0.5, 1.0, 1.5.
        data["image_s"] = torch.randn(batch_size, 3, max(1, height // 2), max(1, width // 2), device=device)
        data["image_l"] = torch.randn(batch_size, 3, int(round(height * 1.5)), int(round(width * 1.5)), device=device)
    return data


def profile_one(
    model_name: str,
    height: int,
    width: int,
    batch_size: int = 1,
    device: str = "cuda",
    pretrained: bool = False,
    mid_dim: int | None = None,
    hmu_groups: int | None = None,
    use_rfrm: bool | None = None,
    use_gmffm: bool | None = None,
    use_checkpoint: bool = False,
    warmup: int = 0,
    benchmark_iters: int = 0,
) -> Dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    cls = _import_model_class(model_name)
    kwargs: Dict[str, Any] = dict(pretrained=pretrained, num_frames=1, use_checkpoint=use_checkpoint)
    if mid_dim is not None:
        kwargs["mid_dim"] = mid_dim
    if hmu_groups is not None:
        kwargs["hmu_groups"] = hmu_groups
    if use_rfrm is not None and "BRG_ZoomNeXt_Lite" in model_name:
        kwargs["use_rfrm"] = use_rfrm
    if use_gmffm is not None and "BRG_ZoomNeXt_Lite" in model_name:
        kwargs["use_gmffm"] = use_gmffm

    model = cls(**kwargs).to(dev).eval()
    data = _dummy_inputs(model_name, batch_size=batch_size, height=height, width=width, device=dev)
    total_params, trainable_params = _num_params(model)

    with torch.no_grad():
        # Hooks measure one clean forward pass.
        with HookBasedFlopCounter(model) as counter:
            _ = model(data)
        macs = counter.total_macs

        latency_ms = None
        if benchmark_iters > 0:
            for _ in range(max(0, warmup)):
                _ = model(data)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(benchmark_iters):
                    _ = model(data)
                end.record()
                torch.cuda.synchronize()
                latency_ms = float(start.elapsed_time(end) / benchmark_iters)
            else:
                import time
                start_t = time.perf_counter()
                for _ in range(benchmark_iters):
                    _ = model(data)
                latency_ms = float((time.perf_counter() - start_t) * 1000 / benchmark_iters)

    result = {
        "model": model_name,
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "params": total_params,
        "params_M": total_params / 1e6,
        "trainable_params": trainable_params,
        "macs": macs,
        "macs_G": macs / 1e9,
        "flops_2mac_G": 2 * macs / 1e9,
        "latency_ms": latency_ms,
        "fps": (1000.0 / latency_ms) if latency_ms and latency_ms > 0 else None,
        "by_type_GMAC": {k: v / 1e9 for k, v in sorted(counter.by_type.items())},
        "top_modules_GMAC": dict(sorted(((k, v / 1e9) for k, v in counter.by_module.items()), key=lambda kv: kv[1], reverse=True)[:30]),
        "kwargs": kwargs,
    }
    return result


def _print_result(r: Dict[str, Any]) -> None:
    latency = "-" if r["latency_ms"] is None else f'{r["latency_ms"]:.2f} ms / {r["fps"]:.2f} FPS'
    print(
        f'{r["model"]:36s} | Params {r["params_M"]:8.3f} M | '
        f'MACs {r["macs_G"]:9.3f} G | FLOPs(2*MAC) {r["flops_2mac_G"]:9.3f} G | {latency}'
    )
    print("  by_type_GMAC:", ", ".join(f"{k}={v:.3f}" for k, v in r["by_type_GMAC"].items()))


def _write_outputs(results: List[Dict[str, Any]], output: str | None) -> None:
    if not output:
        return
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "height", "width", "batch_size", "params_M", "macs_G", "flops_2mac_G", "latency_ms", "fps"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(f"Saved: {out}")
    print(f"Saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile original ZoomNeXt and BRG-ZoomNeXt-Lite under the same protocol.")
    p.add_argument("--model-name", default=None, help="Single model to profile. If omitted with --compare, profiles original+lite.")
    p.add_argument("--compare", action="store_true", help="Profile original ZoomNeXt-B2 and BRG-ZoomNeXt-Lite-B2 together.")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--pretrained", action="store_true", help="Use pretrained weights. Usually disable this for FLOPs profiling.")
    p.add_argument("--mid-dim", type=int, default=None, help="mid_dim for the selected model. Original default is 64; lite default is 48.")
    p.add_argument("--hmu-groups", type=int, default=None)
    p.add_argument("--use-rfrm", type=int, default=None, choices=[0, 1], help="BRG-Lite only. 1 uses RFRM, 0 uses reduced RGPU.")
    p.add_argument("--use-gmffm", type=int, default=None, choices=[0, 1], help="BRG-Lite only.")
    p.add_argument("--sweep-lite-mid-dims", default="", help="Comma-separated mid_dim list for BRG-Lite FLOPs-budget sweep, e.g. 32,48,64,96,128")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--benchmark-iters", type=int, default=0, help="If >0, also measure latency/FPS.")
    p.add_argument("--output", default="/workspace/outputs_brg_lite/flops_profile.json")
    return p.parse_args()


def main() -> None:
    os.chdir(_repo_root())
    args = parse_args()
    results: List[Dict[str, Any]] = []

    if args.compare or args.model_name is None:
        # Defaults used in the original PVT-B2 ZoomNeXt and the recommended BRG-Lite config.
        jobs = [
            dict(model_name="PvtV2B2_ZoomNeXt", mid_dim=64, hmu_groups=6, use_rfrm=None, use_gmffm=None),
            dict(model_name="PvtV2B2_BRG_ZoomNeXt_Lite", mid_dim=48, hmu_groups=2, use_rfrm=True, use_gmffm=True),
        ]
    else:
        jobs = [
            dict(
                model_name=args.model_name,
                mid_dim=args.mid_dim,
                hmu_groups=args.hmu_groups,
                use_rfrm=None if args.use_rfrm is None else bool(args.use_rfrm),
                use_gmffm=None if args.use_gmffm is None else bool(args.use_gmffm),
            )
        ]

    for job in jobs:
        r = profile_one(
            model_name=job["model_name"],
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            device=args.device,
            pretrained=args.pretrained,
            mid_dim=job["mid_dim"],
            hmu_groups=job["hmu_groups"],
            use_rfrm=job["use_rfrm"],
            use_gmffm=job["use_gmffm"],
            warmup=args.warmup,
            benchmark_iters=args.benchmark_iters,
        )
        _print_result(r)
        results.append(r)

    if args.sweep_lite_mid_dims:
        dims = [int(x.strip()) for x in args.sweep_lite_mid_dims.split(",") if x.strip()]
        print("\nBRG-Lite mid_dim FLOPs sweep")
        for d in dims:
            r = profile_one(
                model_name="PvtV2B2_BRG_ZoomNeXt_Lite",
                height=args.height,
                width=args.width,
                batch_size=args.batch_size,
                device=args.device,
                pretrained=args.pretrained,
                mid_dim=d,
                hmu_groups=2 if args.hmu_groups is None else args.hmu_groups,
                use_rfrm=True if args.use_rfrm is None else bool(args.use_rfrm),
                use_gmffm=True if args.use_gmffm is None else bool(args.use_gmffm),
                warmup=0,
                benchmark_iters=0,
            )
            _print_result(r)
            results.append(r)

    if len(results) >= 2:
        base = results[0]
        print("\nRelative to first result:")
        for r in results[1:]:
            print(
                f'  {r["model"]} | MAC ratio={r["macs_G"] / base["macs_G"]:.4f}, '
                f'Param ratio={r["params_M"] / base["params_M"]:.4f}'
            )

    _write_outputs(results, args.output)


if __name__ == "__main__":
    main()
