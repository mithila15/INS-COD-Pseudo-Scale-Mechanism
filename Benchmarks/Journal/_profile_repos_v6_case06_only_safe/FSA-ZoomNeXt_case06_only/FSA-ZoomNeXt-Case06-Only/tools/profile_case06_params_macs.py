#!/usr/bin/env python3
"""Profile Params / MACs for FSA-ZoomNeXt case06 only.

case06_av_residual_spatial_d64:
  model_name       = PvtV2B2_FSA_ZoomNeXt
  backbone         = PVTv2-B2
  input            = image_m only, default 384x384
  mid_dim / d      = 64
  av_gate_mode     = residual_spatial
  use_compensation = True
  use_pseudo_scale = True, if the model implementation exposes this option

The script intentionally does not load pretrained weights because pretrained
weights do not change Params/MACs and can slow down or fail offline profiling.

Outputs:
  - Params
  - total MACs / ops
  - encoder MACs / ops
  - non-encoder MACs / ops = total - encoder

Notes on counting conventions:
  - fvcore reports operator counts under its own FLOP convention. In many vision
    papers this is used as MAC-equivalent cost for same-tool comparison.
  - thop reports MACs. Transformer attention support can differ by implementation.
  - For the final table, use one tool consistently across all models.
"""
from __future__ import annotations

import argparse
import csv
import inspect
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


class FSATotalWrapper(nn.Module):
    """Expose a single-tensor input interface for total model profiling."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_m: torch.Tensor) -> torch.Tensor:
        out = self.model({"image_m": image_m})
        if isinstance(out, dict):
            # Defensive fallback for trainer-style outputs.
            for key in ("logits", "pred", "prediction"):
                if key in out:
                    return out[key]
            raise RuntimeError(f"Model returned a dict without a known logit key: {list(out.keys())}")
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class FSAEncoderWrapper(nn.Module):
    """Profile only normalizer + PVT encoder used by FSA-ZoomNeXt."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_m: torch.Tensor):
        if hasattr(self.model, "normalize_encoder"):
            return self.model.normalize_encoder(image_m)
        x = self.model.normalizer(image_m) if hasattr(self.model, "normalizer") else image_m
        return self.model.encoder(x)


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith("encoder."))
    non_encoder = total - encoder
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_encoder": int(encoder),
        "params_non_encoder": int(non_encoder),
    }


def _filtered_model_kwargs(model_cls: type, raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in raw_kwargs.items() if k in allowed}


def build_case06_model(args: argparse.Namespace) -> nn.Module:
    model_cls = getattr(model_zoo, args.model_name, None)
    if model_cls is None:
        raise ValueError(f"Unknown model class: {args.model_name}")

    raw_kwargs: Dict[str, Any] = dict(
        pretrained=False,
        num_frames=1,
        input_norm=True,
        mid_dim=args.mid_dim,
        hmu_groups=2,
        use_checkpoint=False,
        use_compensation=True,
        use_pseudo_scale=True,
        residual_refine_weight=args.residual_refine_weight,
        av_gate_mode="residual_spatial",
        ual_weight=1.0,
        edge_weight=1.0,
        region_weight=0.5,
        deep_weight=0.25,
        kd_weight=0.0,
    )
    kwargs = _filtered_model_kwargs(model_cls, raw_kwargs)
    model = model_cls(**kwargs)
    return model.eval()


def make_input(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor]:
    image_m = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    return (image_m,)


def profile_fvcore(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    with torch.no_grad():
        analysis = FlopCountAnalysis(module, inputs)
        total_ops = float(analysis.total())
        by_operator = {str(k): float(v) for k, v in analysis.by_operator().items()}
        unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    return {
        "ops": total_ops,
        "ops_G": total_ops / 1e9,
        "by_operator": by_operator,
        "unsupported_ops": unsupported,
    }


def profile_thop(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from thop import profile

    with torch.no_grad():
        macs, params_seen = profile(module, inputs=inputs, verbose=False)
    return {
        "ops": float(macs),
        "ops_G": float(macs) / 1e9,
        "thop_params": int(params_seen),
    }


def profile_one_tool(tool: str, total_wrapper: nn.Module, encoder_wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    if tool == "fvcore":
        total = profile_fvcore(total_wrapper, inputs)
        enc = profile_fvcore(encoder_wrapper, inputs)
        unit = "fvcore_ops"
        note = "fvcore operator counts; use as same-tool MAC/FLOP-equivalent comparison."
    elif tool == "thop":
        total = profile_thop(total_wrapper, inputs)
        enc = profile_thop(encoder_wrapper, inputs)
        unit = "thop_macs"
        note = "THOP reports MACs. If defining FLOPs as multiply+add, use approximately 2x MACs."
    else:
        raise ValueError(tool)

    total_ops = float(total["ops"])
    enc_ops = float(enc["ops"])
    non_enc_ops = max(total_ops - enc_ops, 0.0)

    out = {
        "tool": tool,
        "unit_name": unit,
        "macs_total": total_ops,
        "macs_encoder": enc_ops,
        "macs_non_encoder": non_enc_ops,
        "macs_total_G": total_ops / 1e9,
        "macs_encoder_G": enc_ops / 1e9,
        "macs_non_encoder_G": non_enc_ops / 1e9,
        "encoder_macs_ratio": enc_ops / total_ops if total_ops > 0 else None,
        "non_encoder_macs_ratio": non_enc_ops / total_ops if total_ops > 0 else None,
        "note": note,
    }
    if tool == "fvcore":
        out["total_unsupported_ops"] = total.get("unsupported_ops", {})
        out["encoder_unsupported_ops"] = enc.get("unsupported_ops", {})
    if tool == "thop":
        out["thop_total_params_seen"] = total.get("thop_params")
        out["thop_encoder_params_seen"] = enc.get("thop_params")
    return out


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    model = build_case06_model(args).to(device).eval()
    total_wrapper = FSATotalWrapper(model).to(device).eval()
    encoder_wrapper = FSAEncoderWrapper(model).to(device).eval()
    inputs = make_input(args, device=device)

    # Sanity checks.
    with torch.no_grad():
        logits = total_wrapper(*inputs)
        enc_out = encoder_wrapper(*inputs)

    result: Dict[str, Any] = {
        "case": "case06_av_residual_spatial_d64",
        "model_name": args.model_name,
        "input_protocol": "image_m only",
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "mid_dim_d": args.mid_dim,
        "av_gate_mode": "residual_spatial",
        "use_compensation": True,
        "use_pseudo_scale": True,
        "residual_refine_weight": args.residual_refine_weight,
        "pretrained_loaded": False,
        "output_shape": list(logits.shape),
        "encoder_output_shapes": [list(t.shape) for t in enc_out] if isinstance(enc_out, (tuple, list)) else str(type(enc_out)),
        **count_params(model),
    }
    result["params_total_M"] = result["params_total"] / 1e6
    result["params_trainable_M"] = result["params_trainable"] / 1e6
    result["params_encoder_M"] = result["params_encoder"] / 1e6
    result["params_non_encoder_M"] = result["params_non_encoder"] / 1e6

    tools: List[str] = ["fvcore", "thop"] if args.tool == "all" else [args.tool]
    profiles: List[Dict[str, Any]] = []
    for tool in tools:
        try:
            profiles.append(profile_one_tool(tool, total_wrapper, encoder_wrapper, inputs))
        except Exception as exc:
            profiles.append({"tool": tool, "error": repr(exc)})
    result["profiles"] = profiles
    return result


def _fmt(x: Any, digits: int = 3) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def save_outputs(result: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "case06_params_macs.json"
    csv_path = out_dir / "case06_params_macs.csv"
    txt_path = out_dir / "case06_params_macs.txt"
    md_path = out_dir / "case06_params_macs.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    rows = []
    base = {
        "case": result["case"],
        "model_name": result["model_name"],
        "input_shape": "x".join(map(str, result["input_shape"])),
        "d": result["mid_dim_d"],
        "params_M": result["params_total_M"],
        "enc_params_M": result["params_encoder_M"],
        "non_enc_params_M": result["params_non_encoder_M"],
    }
    for prof in result["profiles"]:
        row = dict(base)
        row.update({
            "tool": prof.get("tool"),
            "unit": prof.get("unit_name", ""),
            "MACs_G": prof.get("macs_total_G", ""),
            "Enc_MACs_G": prof.get("macs_encoder_G", ""),
            "NonEnc_MACs_G": prof.get("macs_non_encoder_G", ""),
            "Enc_ratio": prof.get("encoder_macs_ratio", ""),
            "NonEnc_ratio": prof.get("non_encoder_macs_ratio", ""),
            "error": prof.get("error", ""),
        })
        rows.append(row)

    fieldnames = [
        "case", "model_name", "input_shape", "d",
        "params_M", "enc_params_M", "non_enc_params_M",
        "tool", "unit", "MACs_G", "Enc_MACs_G", "NonEnc_MACs_G",
        "Enc_ratio", "NonEnc_ratio", "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("# case06 Params / MACs profile")
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
    lines.append("| tool | Params(M) | MACs(G) | Enc. MACs(G) | Non-Enc. MACs(G) | Enc. ratio | note/error |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for prof in result["profiles"]:
        note = prof.get("error") or prof.get("note") or ""
        lines.append(
            f"| {prof.get('tool')} "
            f"| {_fmt(result['params_total_M'])} "
            f"| {_fmt(prof.get('macs_total_G',''))} "
            f"| {_fmt(prof.get('macs_encoder_G',''))} "
            f"| {_fmt(prof.get('macs_non_encoder_G',''))} "
            f"| {_fmt(prof.get('encoder_macs_ratio',''), 4)} "
            f"| {note} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Enc. MACs are measured by profiling normalizer + encoder only.")
    lines.append("- Non-Enc. MACs = total MACs - Enc. MACs using the same profiling tool.")
    lines.append("- Use one tool consistently across all models; fvcore and thop can differ.")
    text = "\n".join(lines)
    txt_path.write_text(text)
    md_path.write_text(text)

    print(text)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved TXT : {txt_path}")
    print(f"Saved MD  : {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Params/MACs for FSA-ZoomNeXt case06 only.")
    p.add_argument("--model-name", default="PvtV2B2_FSA_ZoomNeXt")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mid-dim", type=int, default=64)
    p.add_argument("--residual-refine-weight", type=float, default=0.5)
    p.add_argument("--device", default="cpu", help="cpu is most stable for profiling; cuda:0 also works if CUDA is OK.")
    p.add_argument("--tool", choices=["fvcore", "thop", "all"], default="fvcore")
    p.add_argument("--out-dir", default="outputs_profile_case06")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_profile(args)
    save_outputs(result, Path(args.out_dir))


if __name__ == "__main__":
    main()
