#!/usr/bin/env python3
"""Profile Params/FLOPs for case02_av_stage5 only.

case02 setting:
  model_name      = PvtV2B2_FSA_ZoomNeXt
  mid_dim         = 48
  av_gate_mode    = stage5
  use_compensation= True
  input           = image_m only, default 384x384

The script intentionally does not load pretrained weights because pretrained
weights do not change Params/FLOPs and can slow down or fail offline profiling.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import methods as model_zoo  # noqa: E402


class FSAWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_m: torch.Tensor) -> torch.Tensor:
        return self.model({"image_m": image_m})


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith("encoder."))
    non_encoder = total - encoder
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_encoder": encoder,
        "params_non_encoder": non_encoder,
    }


def build_case02_model(args: argparse.Namespace) -> nn.Module:
    model_cls = getattr(model_zoo, args.model_name, None)
    if model_cls is None:
        raise ValueError(f"Unknown model class: {args.model_name}")

    model = model_cls(
        pretrained=False,
        num_frames=1,
        input_norm=True,
        mid_dim=args.mid_dim,
        hmu_groups=2,
        use_checkpoint=False,
        use_compensation=True,
        residual_refine_weight=args.residual_refine_weight,
        av_gate_mode="stage5",
        ual_weight=1.0,
        edge_weight=1.0,
        region_weight=0.5,
        deep_weight=0.25,
        kd_weight=0.0,
    )
    return model.eval()


def make_input(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor]:
    image_m = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    return (image_m,)


def profile_fvcore(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    with torch.no_grad():
        analysis = FlopCountAnalysis(wrapper, inputs)
        total_ops = float(analysis.total())
        by_operator = {str(k): float(v) for k, v in analysis.by_operator().items()}
        unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    return {
        "tool": "fvcore",
        "unit_name": "fvcore_total_ops",
        "ops": total_ops,
        "ops_G": total_ops / 1e9,
        "by_operator": by_operator,
        "unsupported_ops": unsupported,
    }


def profile_thop(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from thop import profile

    with torch.no_grad():
        macs, params_seen = profile(wrapper, inputs=inputs, verbose=False)
    return {
        "tool": "thop",
        "unit_name": "thop_macs",
        "ops": float(macs),
        "ops_G": float(macs) / 1e9,
        "thop_params": int(params_seen),
        "note": "THOP reports MACs. If your paper defines FLOPs as multiply+add, report approximately 2x MACs.",
    }


def profile_torchprofiler(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if inputs[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        _ = wrapper(*inputs)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            _ = wrapper(*inputs)
    total = 0.0
    for evt in prof.key_averages():
        total += float(getattr(evt, "flops", 0) or 0)
    return {
        "tool": "torchprofiler",
        "unit_name": "torchprofiler_flops",
        "ops": total,
        "ops_G": total / 1e9,
        "note": "torch.profiler counts only profiler-supported FLOPs; use only for same-tool comparison.",
    }


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    model = build_case02_model(args).to(device).eval()
    wrapper = FSAWrapper(model).to(device).eval()
    inputs = make_input(args, device=device)

    # quick forward sanity check
    with torch.no_grad():
        logits = wrapper(*inputs)

    result: Dict[str, Any] = {
        "case": "case02_av_stage5",
        "model_name": args.model_name,
        "input_protocol": "image_m only",
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "mid_dim_d": args.mid_dim,
        "av_gate_mode": "stage5",
        "use_compensation": True,
        "residual_refine_weight": args.residual_refine_weight,
        "pretrained_loaded": False,
        "output_shape": list(logits.shape),
        **count_params(model),
    }
    result["params_total_M"] = result["params_total"] / 1e6
    result["params_trainable_M"] = result["params_trainable"] / 1e6
    result["params_encoder_M"] = result["params_encoder"] / 1e6
    result["params_non_encoder_M"] = result["params_non_encoder"] / 1e6

    tools: List[str]
    if args.tool == "all":
        tools = ["fvcore", "thop"]
        if args.include_torchprofiler:
            tools.append("torchprofiler")
    else:
        tools = [args.tool]

    profiles = []
    for tool in tools:
        try:
            if tool == "fvcore":
                profiles.append(profile_fvcore(wrapper, inputs))
            elif tool == "thop":
                profiles.append(profile_thop(wrapper, inputs))
            elif tool == "torchprofiler":
                profiles.append(profile_torchprofiler(wrapper, inputs))
            else:
                raise ValueError(tool)
        except Exception as exc:
            profiles.append({"tool": tool, "error": repr(exc)})
    result["profiles"] = profiles
    return result


def save_outputs(result: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "case02_params_flops.json"
    csv_path = out_dir / "case02_params_flops.csv"
    txt_path = out_dir / "case02_params_flops.txt"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    rows = []
    base = {
        "case": result["case"],
        "model_name": result["model_name"],
        "input_shape": "x".join(map(str, result["input_shape"])),
        "mid_dim_d": result["mid_dim_d"],
        "av_gate_mode": result["av_gate_mode"],
        "params_total_M": result["params_total_M"],
        "params_encoder_M": result["params_encoder_M"],
        "params_non_encoder_M": result["params_non_encoder_M"],
    }
    for prof in result["profiles"]:
        row = dict(base)
        row.update({
            "tool": prof.get("tool"),
            "unit_name": prof.get("unit_name", ""),
            "ops_G": prof.get("ops_G", ""),
            "error": prof.get("error", ""),
        })
        rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "model_name",
                "input_shape",
                "mid_dim_d",
                "av_gate_mode",
                "params_total_M",
                "params_encoder_M",
                "params_non_encoder_M",
                "tool",
                "unit_name",
                "ops_G",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def fmt_num(x: Any, digits: int = 3) -> str:
        if isinstance(x, (int, float)):
            return f"{x:.{digits}f}"
        return str(x)

    lines = []
    lines.append("# case02 Params/FLOPs profile")
    lines.append("")
    lines.append(f"case: {result['case']}")
    lines.append(f"model: {result['model_name']}")
    lines.append(f"input: {result['input_shape']}")
    lines.append(f"d / mid_dim: {result['mid_dim_d']}")
    lines.append(f"A-V gate: {result['av_gate_mode']}")
    lines.append(f"params total: {result['params_total_M']:.3f} M")
    lines.append(f"params encoder: {result['params_encoder_M']:.3f} M")
    lines.append(f"params non-encoder: {result['params_non_encoder_M']:.3f} M")
    lines.append("")
    lines.append("| tool | unit | ops_G | note/error |")
    lines.append("|---|---:|---:|---|")
    for prof in result["profiles"]:
        note = prof.get("error") or prof.get("note") or ""
        lines.append(f"| {prof.get('tool')} | {prof.get('unit_name','')} | {fmt_num(prof.get('ops_G',''))} | {note} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- fvcore_total_ops and thop_macs use different counting conventions; do not mix them in one comparison table.")
    lines.append("- THOP reports MACs. If needed, multiply by 2 for multiply+add FLOPs convention.")
    txt_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved TXT : {txt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Params/FLOPs for FSA-ZoomNeXt case02_av_stage5 only.")
    p.add_argument("--model-name", default="PvtV2B2_FSA_ZoomNeXt")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mid-dim", type=int, default=48)
    p.add_argument("--residual-refine-weight", type=float, default=0.5)
    p.add_argument("--device", default="cpu", help="Use cpu for the most stable profiling; cuda:0 also works if your CUDA env is OK.")
    p.add_argument("--tool", choices=["fvcore", "thop", "torchprofiler", "all"], default="fvcore")
    p.add_argument("--include-torchprofiler", action="store_true", help="Only used with --tool all.")
    p.add_argument("--out-dir", default="outputs_profile_case02")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_profile(args)
    save_outputs(result, Path(args.out_dir))


if __name__ == "__main__":
    main()
