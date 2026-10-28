from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_model_class(model_name: str):
    methods = importlib.import_module("methods")
    if not hasattr(methods, model_name):
        available = [n for n in dir(methods) if ("ZoomNeXt" in n or "BRG" in n)]
        raise ValueError(f"Unknown model name: {model_name}\nAvailable: {available}")
    return getattr(methods, model_name)


def _looks_like_brg_lite(model_name: str) -> bool:
    return "BRG" in model_name or "Lite" in model_name


class DictInputWrapper(nn.Module):
    """Wrap ZoomNeXt-style models so FLOPs tools can trace tensor-only inputs."""

    def __init__(self, model: nn.Module, input_mode: str):
        super().__init__()
        self.model = model
        self.input_mode = input_mode

    def forward(self, *inputs):  # noqa: D401
        if self.input_mode == "original":
            image_l, image_m, image_s = inputs
            data = {"image_l": image_l, "image_m": image_m, "image_s": image_s}
        elif self.input_mode == "lite":
            (image_m,) = inputs
            data = {"image_m": image_m}
        else:
            raise ValueError(f"Unsupported input_mode={self.input_mode}")
        out = self.model(data)
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            # Defensive fallback; normal eval forward returns logits.
            for key in ("logits", "sal", "pred", "out"):
                if key in out:
                    return out[key]
            return next(iter(out.values()))
        return out


def build_model(
    model_name: str,
    *,
    pretrained: bool = False,
    use_checkpoint: bool = False,
    mid_dim: Optional[int] = None,
    hmu_groups: Optional[int] = None,
    use_rfrm: Optional[bool] = None,
    use_gmffm: Optional[bool] = None,
    device: str = "cpu",
) -> nn.Module:
    cls = _import_model_class(model_name)
    kwargs = {"pretrained": pretrained, "use_checkpoint": use_checkpoint}
    if _looks_like_brg_lite(model_name):
        if mid_dim is not None:
            kwargs["mid_dim"] = mid_dim
        if hmu_groups is not None:
            kwargs["hmu_groups"] = hmu_groups
        if use_rfrm is not None:
            kwargs["use_rfrm"] = use_rfrm
        if use_gmffm is not None:
            kwargs["use_gmffm"] = use_gmffm
    else:
        # Original ZoomNeXt also supports some of these kwargs, but keep the
        # official default unless explicitly supported by the constructor.
        if mid_dim is not None:
            kwargs["mid_dim"] = mid_dim
        if hmu_groups is not None:
            kwargs["hmu_groups"] = hmu_groups

    model = cls(**kwargs)
    model.eval().to(device)
    return model


def make_inputs(input_mode: str, image_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, ...]:
    h = w = int(image_size)
    if input_mode == "original":
        hl = int(round(h * 1.5))
        wl = int(round(w * 1.5))
        hs = max(1, int(round(h * 0.5)))
        ws = max(1, int(round(w * 0.5)))
        image_l = torch.randn(batch_size, 3, hl, wl, device=device)
        image_m = torch.randn(batch_size, 3, h, w, device=device)
        image_s = torch.randn(batch_size, 3, hs, ws, device=device)
        return image_l, image_m, image_s
    if input_mode == "lite":
        image_m = torch.randn(batch_size, 3, h, w, device=device)
        return (image_m,)
    raise ValueError(f"Unsupported input_mode={input_mode}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _flops_with_fvcore(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[int, Dict[str, int]]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "fvcore is required for FLOPs measurement. Install it with: pip install fvcore"
        ) from exc

    with torch.no_grad():
        analysis = FlopCountAnalysis(wrapper, inputs)
        # Avoid excessive unsupported op warnings on RunPod logs.
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = int(analysis.total())
        unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    return total, unsupported


def measure_model(
    model_name: str,
    *,
    input_mode: str,
    image_size: int,
    batch_size: int = 1,
    device: str = "cpu",
    pretrained: bool = False,
    use_checkpoint: bool = False,
    mid_dim: Optional[int] = None,
    hmu_groups: Optional[int] = None,
    use_rfrm: Optional[bool] = None,
    use_gmffm: Optional[bool] = None,
) -> Dict[str, object]:
    if input_mode == "auto":
        input_mode = "lite" if _looks_like_brg_lite(model_name) else "original"
    model = build_model(
        model_name,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
        mid_dim=mid_dim,
        hmu_groups=hmu_groups,
        use_rfrm=use_rfrm,
        use_gmffm=use_gmffm,
        device=device,
    )
    wrapper = DictInputWrapper(model, input_mode=input_mode).eval().to(device)
    inputs = make_inputs(input_mode, image_size=image_size, batch_size=batch_size, device=device)
    flops, unsupported = _flops_with_fvcore(wrapper, inputs)
    params = count_params(model)
    return {
        "model_name": model_name,
        "input_mode": input_mode,
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "params": int(params),
        "params_M": params / 1e6,
        "fvcore_flops": int(flops),
        "fvcore_GFLOPs": flops / 1e9,
        "unsupported_ops": unsupported,
    }


def format_result(result: Dict[str, object]) -> str:
    unsupported = result.get("unsupported_ops") or {}
    unsupported_str = ", ".join(f"{k}:{v}" for k, v in unsupported.items()) if unsupported else "none"
    return (
        f"{result['model_name']} | mode={result['input_mode']} | "
        f"size={result['image_size']} | params={result['params_M']:.3f}M | "
        f"fvcore_GFLOPs={result['fvcore_GFLOPs']:.3f} | unsupported_ops={unsupported_str}"
    )
