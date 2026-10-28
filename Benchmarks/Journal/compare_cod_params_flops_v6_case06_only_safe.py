#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Params and FLOPs/MACs for FSA-ZoomNeXt case02/case06, ZoomNeXt, and ZoomNet.

v5 changes:
  - Handles FSA_case06_profile_tools.zip as a profile-tool patch, not a full repo.
  - Adds FSA-case06 as a fourth model using case06 config: mid_dim=64, av_gate_mode=residual_spatial.
  - Keeps CPU-safe profiling by default even when --device cuda is requested.

v3 cuda-safe changes:
  - CUDA execution is opt-in. If --device cuda is passed, the script still profiles on CPU
    unless --unsafe-use-cuda is also passed. This avoids Windows fatal access violations
    at model.to("cuda"), which Python cannot catch.
  - Adds --avoid-cuda-init/--allow-cuda-init. CPU profiling can fake CUDA device
    properties during repository model build, because Params/FLOPs do not depend on
    the selected CUDA attention backend.

v2 changes:
  - Adds a hook-based MAC profiler that counts actual module calls.
    This is important because ZoomNeXt/ZoomNet reuse the same encoder for 3 input scales;
    Params are counted once, but MACs/FLOPs must count every forward call.
  - Prints encoder forward call counts and encoder input shapes.
  - Runs the hook profiler before optional fvcore/thop, so you can diagnose hangs/crashes.
  - Keeps fvcore/thop as optional comparison tools.

Recommended first run:
  python compare_cod_params_flops_v3_cuda_safe.py --tool hook --device cuda ...

Notes:
  - Params do not depend on input image count, input size, pixel values, or pretrained weights.
  - MACs/FLOPs depend on the actual forward path and tensor shapes.
  - THOP reports MACs. Some papers report FLOPs as 2 x MACs.
"""
from __future__ import annotations

import argparse
import faulthandler
import contextlib
import csv
import importlib
import importlib.machinery
import json
import os
import platform
import re
import inspect
import sys
import time
import traceback
import types
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FSA_ZIP = r"E:\COD\Journal\FSA-ZoomNeXt_case02.zip"
DEFAULT_FSA_CASE06_ZIP = r"E:\COD\Journal\FSA-ZoomNeXt_case06_only.zip"
DEFAULT_ZOOMNET_ZIP = r"E:\COD\Journal\ZoomNet-main.zip"
DEFAULT_ZOOMNEXT_ZIP = r"E:\COD\Journal\ZoomNeXt-main.zip"


@dataclass
class Job:
    name: str
    root_key: str
    module: str
    class_name: str
    family: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[str] = None


def default_jobs(load_pretrained: bool) -> List[Job]:
    return [
        Job(
            name="FSA-ZoomNeXt-case02",
            root_key="fsa",
            module="methods.zoomnext.fsa_zoomnext",
            class_name="PvtV2B2_FSA_ZoomNeXt",
            family="fsa_zoomnext",
            kwargs=dict(
                pretrained=load_pretrained,
                num_frames=1,
                input_norm=True,
                mid_dim=48,
                hmu_groups=2,
                use_checkpoint=False,
                use_compensation=True,
                residual_refine_weight=0.5,
                av_gate_mode="stage5",
                ual_weight=1.0,
                edge_weight=1.0,
                region_weight=0.5,
                deep_weight=0.25,
                kd_weight=0.0,
            ),
        ),
        Job(
            name="ZoomNeXt-PvtV2B2",
            root_key="zoomnext",
            module="methods.zoomnext.zoomnext",
            class_name="PvtV2B2_ZoomNeXt",
            family="zoomnext",
            kwargs=dict(
                pretrained=load_pretrained,
                num_frames=1,
                input_norm=True,
                mid_dim=64,
                siu_groups=4,
                hmu_groups=6,
                use_checkpoint=False,
            ),
        ),
        Job(
            name="ZoomNet",
            root_key="zoomnet",
            module="methods.zoomnet.zoomnet",
            class_name="ZoomNet",
            family="zoomnet",
            kwargs=dict(),
        ),
    ]


PROJECT_MODULE_PREFIXES = (
    "methods",
    "utils",
    "configs",
    "dataset",
    "main",
    "main_for_image",
    "main_for_video",
    "main_for_fsa",
    "test",
)

_TORCHVISION_STUBS: List[Any] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def define_torchvision_nms_stub() -> None:
    try:
        from torch.library import Library

        lib = Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_STUBS.append(lib)
    except Exception:
        pass


def install_timm_fallback_stubs() -> None:
    """Install a minimal timm stub when timm is absent.

    The PVTv2 code used by FSA/ZoomNeXt only needs DropPath, to_2tuple,
    trunc_normal_, and _cfg. A real timm install is still preferred.
    """
    try:
        import timm  # noqa: F401
        return
    except Exception:
        pass

    if "timm" in sys.modules:
        return

    timm_mod = types.ModuleType("timm")
    models_mod = types.ModuleType("timm.models")
    layers_mod = types.ModuleType("timm.models.layers")
    tlayers_mod = types.ModuleType("timm.layers")
    vt_mod = types.ModuleType("timm.models.vision_transformer")

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean, std)

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def _cfg(**kwargs):
        return kwargs

    def create_model(*args, **kwargs):
        raise RuntimeError("timm is not installed; timm.create_model is unavailable in fallback stub")

    for mod in (layers_mod, tlayers_mod):
        mod.DropPath = DropPath
        mod.to_2tuple = to_2tuple
        mod.trunc_normal_ = trunc_normal_
    vt_mod._cfg = _cfg
    timm_mod.create_model = create_model
    timm_mod.models = models_mod
    models_mod.layers = layers_mod
    models_mod.vision_transformer = vt_mod

    sys.modules["timm"] = timm_mod
    sys.modules["timm.models"] = models_mod
    sys.modules["timm.models.layers"] = layers_mod
    sys.modules["timm.layers"] = tlayers_mod
    sys.modules["timm.models.vision_transformer"] = vt_mod
    log("[env] timm is not installed; using a minimal fallback stub for PVT-based models")


@contextlib.contextmanager
def patch_timm_fast_init():
    """Patch timm trunc_normal_ to speed up construction.

    Params/FLOPs are independent of random initialization values.
    """
    patched: List[Tuple[Any, str, Any]] = []

    def fast_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.zero_()

    for mod_name in ("timm.models.layers", "timm.layers"):
        try:
            mod = importlib.import_module(mod_name)
            old = getattr(mod, "trunc_normal_", None)
            if old is not None:
                patched.append((mod, "trunc_normal_", old))
                setattr(mod, "trunc_normal_", fast_trunc_normal_)
        except Exception:
            pass

    try:
        yield
    finally:
        for mod, attr, old in reversed(patched):
            try:
                setattr(mod, attr, old)
            except Exception:
                pass


@contextlib.contextmanager
def patch_timm_no_pretrained():
    """Force timm.create_model(..., pretrained=False).

    ZoomNet hard-codes timm.create_model(pretrained=True). For Params/FLOPs this is unnecessary
    and may trigger a network download. This patch is disabled when --load-pretrained is used.
    """
    old_create_model = None
    try:
        import timm

        old_create_model = timm.create_model

        def create_model_no_pretrained(*args, **kwargs):
            kwargs["pretrained"] = False
            return old_create_model(*args, **kwargs)

        timm.create_model = create_model_no_pretrained
    except Exception:
        pass

    try:
        yield
    finally:
        if old_create_model is not None:
            try:
                import timm

                timm.create_model = old_create_model
            except Exception:
                pass


@contextlib.contextmanager
def patch_cuda_props_when_cpu(force: bool = False):
    """Let PVT code instantiate without touching the CUDA runtime when profiling on CPU.

    The uploaded ZoomNeXt/PVT code calls torch.cuda.get_device_properties('cuda')
    during Attention init. For CPU complexity profiling, this CUDA query is unnecessary
    and can make Windows CUDA environments fragile.
    """
    if torch.cuda.is_available() and not force:
        yield
        return

    old_get_device_properties = torch.cuda.get_device_properties

    class _FakeProps:
        major = 0
        minor = 0
        name = "cpu-profile-fake-cuda-props"

    def fake_get_device_properties(device=None):
        return _FakeProps()

    torch.cuda.get_device_properties = fake_get_device_properties
    try:
        yield
    finally:
        torch.cuda.get_device_properties = old_get_device_properties


def normalize_attention_cuda_config(model: nn.Module) -> None:
    """Use a safe attention config for profiling.

    The repository sets SDP kernel preferences during Attention.__init__.
    For profiling, math attention is safer and easier to count than memory-efficient/flash kernels.
    """
    for module in model.modules():
        if hasattr(module, "cuda_config"):
            try:
                module.cuda_config = {"enable_flash": False, "enable_math": True, "enable_mem_efficient": False}
            except Exception:
                pass


def clear_project_modules() -> None:
    prefixes = tuple(p + "." for p in PROJECT_MODULE_PREFIXES)
    for key in list(sys.modules.keys()):
        if key in PROJECT_MODULE_PREFIXES or key.startswith(prefixes):
            del sys.modules[key]


def install_repo_package_stubs(repo_root: Path) -> None:
    """Avoid heavy package __init__.py side effects."""
    for pkg_name in ("methods", "utils", "configs", "dataset"):
        pkg_path = repo_root / pkg_name
        if pkg_path.is_dir() and pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            pkg.__package__ = pkg_name
            pkg.__file__ = str(pkg_path / "__init__.py")
            spec = importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(pkg_path)]
            pkg.__spec__ = spec
            sys.modules[pkg_name] = pkg


@contextlib.contextmanager
def repo_context(repo_root: Path, *, load_pretrained: bool, fast_init: bool, force_fake_cuda_props: bool = False):
    repo_root = repo_root.resolve()
    clear_project_modules()
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(repo_root))

    define_torchvision_nms_stub()
    install_timm_fallback_stubs()
    install_repo_package_stubs(repo_root)

    managers: List[contextlib.AbstractContextManager[Any]] = [patch_cuda_props_when_cpu(force=force_fake_cuda_props)]
    if fast_init:
        managers.append(patch_timm_fast_init())
    if not load_pretrained:
        managers.append(patch_timm_no_pretrained())

    entered: List[contextlib.AbstractContextManager[Any]] = []
    try:
        for m in managers:
            m.__enter__()
            entered.append(m)
        yield
    finally:
        for m in reversed(entered):
            try:
                m.__exit__(None, None, None)
            except Exception:
                pass
        sys.path[:] = old_sys_path
        clear_project_modules()


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / zip_path.stem.replace(" ", "_")
    marker = target / ".extracted_ok"
    if not marker.exists():
        if target.exists():
            import shutil

            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)
        marker.write_text("ok", encoding="utf-8")

    children = [p for p in target.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "methods").exists():
        return children[0]
    if (target / "methods").exists():
        return target
    for p in children:
        if (p / "methods").exists():
            return p
    raise RuntimeError(f"Could not find repo root containing methods/ under {target}")



def extract_zip_raw(zip_path: Path, out_dir: Path) -> Path:
    """Extract a ZIP even when it is not a full repository."""
    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / zip_path.stem.replace(" ", "_")
    marker = target / ".extracted_ok"
    if not marker.exists():
        if target.exists():
            import shutil

            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)
        marker.write_text("ok", encoding="utf-8")
    return target


def zip_has_methods_repo(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path.expanduser(), "r") as zf:
            names = [n.replace("\\", "/").rstrip("/") for n in zf.namelist()]
    except Exception:
        return False
    return any(n == "methods" or n.endswith("/methods") for n in names)


def case06_default_kwargs(load_pretrained: bool) -> Dict[str, Any]:
    """Constructor kwargs for case06_av_residual_spatial_d64."""
    return dict(
        pretrained=load_pretrained,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        hmu_groups=2,
        use_checkpoint=False,
        use_compensation=True,
        residual_refine_weight=0.5,
        av_gate_mode="residual_spatial",
        ual_weight=1.0,
        edge_weight=1.0,
        region_weight=0.5,
        deep_weight=0.25,
        kd_weight=0.0,
    )


def module_name_from_file(repo_root: Path, py_file: Path) -> str:
    rel = py_file.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def scan_model_candidates(repo_root: Path, family: str) -> List[Tuple[int, str, str]]:
    methods_dir = repo_root / "methods"
    if not methods_dir.exists():
        return []
    class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:", re.MULTILINE)
    candidates: List[Tuple[int, str, str]] = []
    for py_file in methods_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        module_name = module_name_from_file(repo_root, py_file)
        lower_path = str(py_file.relative_to(repo_root)).lower().replace("\\", "/")
        for match in class_re.finditer(source):
            cls_name, bases = match.group(1), match.group(2)
            lower_cls = cls_name.lower()
            lower_bases = bases.lower()
            score = 0
            if "nn.module" in lower_bases or "module" in lower_bases:
                score += 5
            if "pvtv2" in lower_cls or "pvt" in lower_cls:
                score += 10
            if "zoomnext" in lower_cls:
                score += 20
            if "zoomnet" in lower_cls:
                score += 20
            if "fsa" in lower_cls or "fsa" in lower_path:
                score += 30
            if py_file.name in {"fsa_zoomnext.py", "zoomnext.py", "zoomnet.py"}:
                score += 15
            if family == "fsa_zoomnext":
                if "fsa" in lower_cls:
                    score += 60
                if "fsa_zoomnext" in lower_path:
                    score += 50
                if "zoomnext" in lower_cls:
                    score += 15
            elif family == "zoomnext":
                if "zoomnext" in lower_cls or "zoomnext" in lower_path:
                    score += 50
            elif family == "zoomnet":
                if "zoomnet" in lower_cls or "zoomnet" in lower_path:
                    score += 50
            if any(tok in lower_cls for tok in ("block", "layer", "head", "decoder", "encoder", "attention", "norm", "conv")):
                score -= 25
            if score > 0:
                candidates.append((score, module_name, cls_name))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return candidates


def detect_model_target(repo_root: Path, family: str) -> Tuple[str, str]:
    candidates = scan_model_candidates(repo_root, family)
    if not candidates:
        raise RuntimeError(
            f"Could not auto-detect a model class under {repo_root / 'methods'}. "
            "Pass --fsa-case06-module and --fsa-case06-class explicitly."
        )
    score, module_name, cls_name = candidates[0]
    log(f"[auto-detect] selected module={module_name}, class={cls_name}, score={score}")
    if len(candidates) > 1:
        preview = ", ".join(f"{m}:{c}({s})" for s, m, c in candidates[:5])
        log(f"[auto-detect] top candidates: {preview}")
    return module_name, cls_name


def filter_kwargs_for_class(model_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(model_cls.__init__)
    except Exception:
        return kwargs
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs
    allowed = {
        p.name
        for p in params
        if p.name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        log(f"[kwargs] dropped unsupported constructor args for {model_cls.__name__}: {dropped}")
    return filtered

def resolve_roots(args: argparse.Namespace) -> Dict[str, Path]:
    base = Path(args.extract_dir)
    roots: Dict[str, Path] = {}
    roots["fsa"] = Path(args.fsa_root).expanduser().resolve() if args.fsa_root else extract_zip(Path(args.fsa_zip), base)

    args.fsa_case06_source = ""
    if not getattr(args, "skip_fsa_case06", False):
        if args.fsa_case06_root:
            case06_root = Path(args.fsa_case06_root).expanduser().resolve()
            roots["fsa_case06"] = case06_root
            args.fsa_case06_source = f"root:{case06_root}"
        else:
            case06_zip = Path(args.fsa_case06_zip).expanduser() if args.fsa_case06_zip else None
            if case06_zip is not None and case06_zip.exists():
                if zip_has_methods_repo(case06_zip):
                    roots["fsa_case06"] = extract_zip(case06_zip, base)
                    args.fsa_case06_source = f"case06_repo_zip:{case06_zip}"
                    log(f"[case06] Detected full case06 repo ZIP with methods/: {case06_zip}")
                else:
                    patch_root = extract_zip_raw(case06_zip, base)
                    roots["fsa_case06"] = roots["fsa"]
                    args.fsa_case06_source = f"profile_patch_zip:{case06_zip}; extracted={patch_root}; model_root=fsa"
                    log(
                        "[case06] Detected a profile-tools patch ZIP, not a full repo. "
                        "Using the FSA repo as model_root and applying case06 kwargs "
                        "(mid_dim=64, av_gate_mode=residual_spatial)."
                    )
            elif getattr(args, "require_fsa_case06", False):
                raise FileNotFoundError(f"FSA case06 ZIP not found: {args.fsa_case06_zip}")
            else:
                log(f"[skip] FSA-case06 ZIP not found, skipping optional model: {args.fsa_case06_zip}")

    if not getattr(args, "skip_zoomnet", False):
        roots["zoomnet"] = (
            Path(args.zoomnet_root).expanduser().resolve() if args.zoomnet_root else extract_zip(Path(args.zoomnet_zip), base)
        )
    if not getattr(args, "skip_zoomnext", False):
        roots["zoomnext"] = (
            Path(args.zoomnext_root).expanduser().resolve() if args.zoomnext_root else extract_zip(Path(args.zoomnext_zip), base)
        )
    for key, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(f"{key} root not found: {root}")
        if not (root / "methods").exists():
            raise FileNotFoundError(f"{key} root has no methods/ directory: {root}")
    return roots

def load_checkpoint(model: nn.Module, ckpt_path: Path, *, strict: bool = False) -> None:
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu")
    state = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state", "model_state_dict", "net", "params"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")

    cleaned = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = str(k)
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    log(f"[checkpoint] {ckpt_path.name}: missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}")


def instantiate_model(job: Job, repo_root: Path, args: argparse.Namespace) -> nn.Module:
    with repo_context(
        repo_root,
        load_pretrained=args.load_pretrained,
        fast_init=args.fast_init,
        force_fake_cuda_props=(args.device == "cpu" and args.avoid_cuda_init),
    ):
        module_name = job.module
        class_name = job.class_name
        if module_name == "auto" or class_name == "auto":
            detected_module, detected_class = detect_model_target(repo_root, job.family)
            if module_name == "auto":
                module_name = detected_module
            if class_name == "auto":
                class_name = detected_class
            job.module = module_name
            job.class_name = class_name

        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
        kwargs = filter_kwargs_for_class(model_cls, job.kwargs)
        model = model_cls(**kwargs)
        normalize_attention_cuda_config(model)

    if job.checkpoint:
        load_checkpoint(model, Path(job.checkpoint), strict=args.strict_checkpoint)
    model.eval()
    return model


class ProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module, family: str) -> None:
        super().__init__()
        self.model = model
        self.family = family

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.family == "fsa_zoomnext":
            output = self.model({"image_m": inputs[0]})
        elif self.family == "zoomnext":
            image_l, image_m, image_s = inputs
            output = self.model({"image_l": image_l, "image_m": image_m, "image_s": image_s})
        elif self.family == "zoomnet":
            image_l, image_m, image_s = inputs
            output = self.model(data={"image1.5": image_l, "image1.0": image_m, "image0.5": image_s})
        else:
            raise ValueError(f"Unknown family: {self.family}")
        return unwrap_tensor(output)


def unwrap_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("seg", "logits", "sal", "pred", "out"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value
    if isinstance(output, (list, tuple)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Could not unwrap tensor from output type: {type(output)}")


def load_image_tensor(image_path: Path, height: int, width: int, device: torch.device) -> torch.Tensor:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:
        raise RuntimeError("--input-mode image requires pillow and numpy") from exc

    img = Image.open(str(image_path)).convert("RGB").resize((width, height))
    arr = np.asarray(img).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device=device)


def resize_tensor(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


def make_base_tensor(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    b, h, w = args.batch_size, args.height, args.width
    if args.input_mode == "ones":
        return torch.ones(b, 3, h, w, device=device)
    if args.input_mode == "zeros":
        return torch.zeros(b, 3, h, w, device=device)
    if args.input_mode == "randn":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        return torch.randn(b, 3, h, w, generator=generator).to(device=device)
    if args.input_mode == "image":
        if not args.image:
            raise ValueError("--input-mode image requires --image path")
        img = load_image_tensor(Path(args.image), h, w, device=device)
        if b != 1:
            img = img.repeat(b, 1, 1, 1)
        return img
    raise ValueError(f"Unknown input mode: {args.input_mode}")


def make_inputs(family: str, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, ...]:
    image_m = make_base_tensor(args, device=device)
    h, w = args.height, args.width
    if family == "fsa_zoomnext":
        return (image_m,)
    image_l = resize_tensor(image_m, int(round(h * args.large_scale)), int(round(w * args.large_scale)))
    image_s = resize_tensor(image_m, int(round(h * args.small_scale)), int(round(w * args.small_scale)))
    return (image_l.contiguous(), image_m.contiguous(), image_s.contiguous())


def input_protocol(family: str, args: argparse.Namespace) -> str:
    if family == "fsa_zoomnext":
        return f"image_m only: {args.height}x{args.width}"
    return (
        f"3-scale: image_l={int(round(args.height * args.large_scale))}x{int(round(args.width * args.large_scale))}, "
        f"image_m={args.height}x{args.width}, "
        f"image_s={int(round(args.height * args.small_scale))}x{int(round(args.width * args.small_scale))}"
    )


def count_params(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("encoder.", "shared_encoder.")))
    non_encoder = total - encoder
    return dict(
        params=int(total),
        trainable_params=int(trainable),
        encoder_params=int(encoder),
        non_encoder_params=int(non_encoder),
        params_M=total / 1e6,
        trainable_params_M=trainable / 1e6,
        encoder_params_M=encoder / 1e6,
        non_encoder_params_M=non_encoder / 1e6,
    )


def shape_of(x: Any) -> Any:
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    return str(type(x))


class EncoderCallCounter:
    def __init__(self, model: nn.Module) -> None:
        self.calls = 0
        self.input_shapes: List[Any] = []
        self.handles: List[Any] = []
        for attr in ("encoder", "shared_encoder"):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                self.handles.append(mod.register_forward_pre_hook(self._hook))

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        self.calls += 1
        if inputs:
            self.input_shapes.append(shape_of(inputs[0]))
        else:
            self.input_shapes.append([])

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


def strip_wrapper_prefix(name: str) -> str:
    return name[6:] if name.startswith("model.") else name


def group_from_module_name(name: str) -> str:
    raw = strip_wrapper_prefix(name)
    if raw.startswith("encoder.") or raw == "encoder":
        return "encoder"
    if raw.startswith("shared_encoder.") or raw == "shared_encoder":
        return "encoder"
    return "non_encoder"


class HookMacProfiler:
    """Approximate MAC profiler using PyTorch forward hooks.

    Counts Conv/Linear calls every time they run, so shared modules called 3 times are counted 3 times.
    Also counts SDPA matmul MACs through patch_sdpa_counter().
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.total_macs: float = 0.0
        self.by_group: DefaultDict[str, float] = defaultdict(float)
        self.by_type: DefaultDict[str, float] = defaultdict(float)
        self.by_module: DefaultDict[str, float] = defaultdict(float)
        self.call_counts: DefaultDict[str, int] = defaultdict(int)
        self.handles: List[Any] = []
        self.name_of: Dict[nn.Module, str] = {m: n for n, m in module.named_modules()}

    def add(self, name: str, macs: float, op_type: str, group: Optional[str] = None) -> None:
        macs = float(macs)
        if macs <= 0:
            return
        if group is None:
            group = group_from_module_name(name)
        self.total_macs += macs
        self.by_group[group] += macs
        self.by_type[op_type] += macs
        self.by_module[name] += macs
        self.call_counts[name] += 1

    def _conv2d_hook(self, module: nn.Conv2d, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        out = output
        batch = int(out.shape[0])
        out_c = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        k_h, k_w = module.kernel_size
        in_per_group = int(module.in_channels // module.groups)
        macs = batch * out_h * out_w * out_c * in_per_group * k_h * k_w
        name = self.name_of.get(module, module.__class__.__name__)
        self.add(name, macs, "Conv2d")

    def _convtranspose2d_hook(self, module: nn.ConvTranspose2d, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        out = output
        batch = int(out.shape[0])
        out_c = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        k_h, k_w = module.kernel_size
        in_per_group = int(module.in_channels // module.groups)
        macs = batch * out_h * out_w * out_c * in_per_group * k_h * k_w
        name = self.name_of.get(module, module.__class__.__name__)
        self.add(name, macs, "ConvTranspose2d")

    def _linear_hook(self, module: nn.Linear, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        # One dot product of length in_features per output element.
        macs = int(output.numel()) * int(module.in_features)
        name = self.name_of.get(module, module.__class__.__name__)
        self.add(name, macs, "Linear")

    def _batchnorm_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        # Not MAC-heavy, but include a small elementwise estimate for transparency.
        name = self.name_of.get(module, module.__class__.__name__)
        self.add(name, int(output.numel()) * 2, module.__class__.__name__)

    def _layernorm_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        name = self.name_of.get(module, module.__class__.__name__)
        self.add(name, int(output.numel()) * 5, module.__class__.__name__)

    def __enter__(self):
        for m in self.module.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv2d_hook))
            elif isinstance(m, nn.ConvTranspose2d):
                self.handles.append(m.register_forward_hook(self._convtranspose2d_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.handles.append(m.register_forward_hook(self._batchnorm_hook))
            elif isinstance(m, nn.LayerNorm):
                self.handles.append(m.register_forward_hook(self._layernorm_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        return False

    def top_modules(self, n: int = 20) -> List[Dict[str, Any]]:
        items = sorted(self.by_module.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [
            dict(name=name, macs=macs, macs_G=macs / 1e9, calls=int(self.call_counts.get(name, 0)))
            for name, macs in items
        ]


@contextlib.contextmanager
def patch_sdpa_counter(profiler: Optional[HookMacProfiler], mode: str = "explicit"):
    """Patch F.scaled_dot_product_attention for counting and optional explicit math.

    mode:
      - explicit: count SDPA MACs and compute qk/softmax/av explicitly.
      - count_original: count SDPA MACs, then call the original PyTorch kernel.
      - none: no patch.
    """
    if mode == "none":
        yield
        return

    old_sdpa = F.scaled_dot_product_attention

    def counted_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        try:
            b = int(q.shape[0])
            heads = int(q.shape[1])
            q_len = int(q.shape[-2])
            k_len = int(k.shape[-2])
            head_dim = int(q.shape[-1])
            macs = 2.0 * b * heads * q_len * k_len * head_dim
            if profiler is not None:
                profiler.add("functional.scaled_dot_product_attention", macs, "SDPA", group="encoder")
        except Exception:
            pass

        if mode == "count_original":
            return old_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        scale_factor = (q.shape[-1] ** -0.5) if scale is None else scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if is_causal:
            ql, kl = scores.shape[-2], scores.shape[-1]
            causal = torch.ones(ql, kl, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        if dropout_p and torch.is_grad_enabled():
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)

    F.scaled_dot_product_attention = counted_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = old_sdpa


def profile_hook(wrapper: nn.Module, model: nn.Module, inputs: Tuple[torch.Tensor, ...], args: argparse.Namespace) -> Dict[str, Any]:
    encoder_counter = EncoderCallCounter(model)
    start = time.perf_counter()
    profiler = HookMacProfiler(wrapper)
    try:
        log("[forward:hook] start")
        with torch.no_grad(), profiler, patch_sdpa_counter(profiler, mode=args.attention_mode):
            output = wrapper(*inputs)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        log(f"[forward:hook] done output_shape={list(output.shape)} elapsed={elapsed:.3f}s")
        result = dict(
            output_shape=list(output.shape),
            hook_MACs=float(profiler.total_macs),
            hook_MACs_G=float(profiler.total_macs / 1e9),
            hook_FLOPs_2x_G=float(2.0 * profiler.total_macs / 1e9),
            hook_encoder_MACs_G=float(profiler.by_group.get("encoder", 0.0) / 1e9),
            hook_non_encoder_MACs_G=float(profiler.by_group.get("non_encoder", 0.0) / 1e9),
            hook_by_type={k: float(v) for k, v in sorted(profiler.by_type.items())},
            hook_top_modules=profiler.top_modules(args.topk),
            encoder_forward_calls=int(encoder_counter.calls),
            encoder_input_shapes=encoder_counter.input_shapes,
            hook_elapsed_sec=elapsed,
        )
        return result
    finally:
        encoder_counter.close()


def profile_fvcore(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    log("[profile:fvcore] start")
    with torch.no_grad():
        analysis = FlopCountAnalysis(wrapper, inputs)
        total = float(analysis.total())
        by_operator = {str(k): float(v) for k, v in analysis.by_operator().items()}
        unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    log("[profile:fvcore] done")
    return dict(
        fvcore_FLOPs=total,
        fvcore_FLOPs_G=total / 1e9,
        fvcore_by_operator=by_operator,
        fvcore_unsupported_ops=unsupported,
    )


def profile_thop(wrapper: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    from thop import profile

    log("[profile:thop] start")
    with torch.no_grad():
        macs, params_seen = profile(wrapper, inputs=inputs, verbose=False)
    log("[profile:thop] done")
    macs = float(macs)
    return dict(
        thop_MACs=macs,
        thop_MACs_G=macs / 1e9,
        thop_FLOPs_2x_G=(2.0 * macs) / 1e9,
        thop_params=int(params_seen),
    )


def requested_tools(args: argparse.Namespace) -> List[str]:
    if args.tool == "params":
        return []
    if args.tool == "hook":
        return ["hook"]
    if args.tool == "fvcore":
        return ["hook", "fvcore"] if args.hook_first else ["fvcore"]
    if args.tool == "thop":
        return ["hook", "thop"] if args.hook_first else ["thop"]
    if args.tool == "both":
        return ["hook", "fvcore", "thop"] if args.hook_first else ["fvcore", "thop"]
    if args.tool == "all":
        return ["hook", "fvcore", "thop"]
    raise ValueError(args.tool)


def input_area_factor(inputs: Tuple[torch.Tensor, ...], args: argparse.Namespace) -> float:
    base_area = float(args.height * args.width)
    if base_area <= 0:
        return 0.0
    return float(sum(int(x.shape[-2]) * int(x.shape[-1]) for x in inputs) / base_area)


def add_called_weighted_params(result: Dict[str, Any]) -> None:
    """Diagnostic only: multiply shared encoder params by actual encoder calls.

    Standard Params count each stored tensor once. This field is useful only to explain
    why a multi-scale model can have similar standard Params but much larger compute.
    """
    try:
        enc = int(result.get("encoder_params", 0))
        non = int(result.get("non_encoder_params", 0))
        calls = max(int(result.get("encoder_forward_calls", 1)), 1)
    except Exception:
        return
    weighted = non + enc * calls
    result["called_weighted_params"] = int(weighted)
    result["called_weighted_params_M"] = float(weighted / 1e6)


def profile_one(job: Job, repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    log(f"[build] start {job.name} ({job.class_name})")
    build_start = time.perf_counter()
    model = instantiate_model(job, repo_root, args)
    log(f"[build] done {job.name}")

    log(f"[device] move start {job.name} -> {device}")
    model = model.to(device=device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    log(f"[device] move done {job.name} -> {device}")
    build_elapsed = time.perf_counter() - build_start

    params = count_params(model)
    log(
        f"[params] {job.name}: total={params['params_M']:.3f}M, "
        f"encoder={params['encoder_params_M']:.3f}M, non_encoder={params['non_encoder_params_M']:.3f}M, "
        f"build={build_elapsed:.3f}s"
    )

    wrapper = ProfileWrapper(model, family=job.family).to(device=device).eval()
    inputs = make_inputs(job.family, args, device=device)
    log(f"[input] {job.name}: {input_protocol(job.family, args)} shapes={[list(x.shape) for x in inputs]}")

    result: Dict[str, Any] = dict(
        name=job.name,
        class_name=job.class_name,
        family=job.family,
        repo_root=str(repo_root),
        input_protocol=input_protocol(job.family, args),
        input_shapes=[list(x.shape) for x in inputs],
        input_area_factor_vs_base=input_area_factor(inputs, args),
        input_mode=args.input_mode,
        pretrained_requested=bool(args.load_pretrained),
        requested_device=getattr(args, "requested_device", args.device),
        actual_device=args.device,
        execution_device=args.device,
        checkpoint=job.checkpoint or "",
        case06_source=getattr(args, "fsa_case06_source", "") if job.root_key == "fsa_case06" else "",
        build_elapsed_sec=build_elapsed,
    )
    result.update(params)

    for tool in requested_tools(args):
        try:
            if tool == "hook":
                result.update(profile_hook(wrapper, model, inputs, args))
                add_called_weighted_params(result)
            elif tool == "fvcore":
                with patch_sdpa_counter(None, mode="explicit"):
                    result.update(profile_fvcore(wrapper, inputs))
            elif tool == "thop":
                with patch_sdpa_counter(None, mode="explicit"):
                    result.update(profile_thop(wrapper, inputs))
            else:
                raise ValueError(tool)
        except Exception as exc:
            result[f"{tool}_error"] = repr(exc)
            result[f"{tool}_traceback"] = traceback.format_exc(limit=8)
            log(f"[profile:{tool}:ERROR] {job.name}: {exc!r}")
            if args.stop_on_error:
                raise
    return result


def format_float(x: Any, digits: int = 3) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, (float, int)):
        return f"{float(x):.{digits}f}"
    return str(x)


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    cols = [
        ("name", "Model"),
        ("class_name", "Class"),
        ("params_M", "Params(M)"),
        ("encoder_params_M", "Encoder(M)"),
        ("non_encoder_params_M", "NonEnc(M)"),
        ("called_weighted_params_M", "CalledWt(M)"),
        ("actual_device", "Device"),
        ("encoder_forward_calls", "EncCalls"),
        ("input_area_factor_vs_base", "AreaX"),
        ("hook_MACs_G", "Hook MACs(G)"),
        ("hook_encoder_MACs_G", "Hook Enc(G)"),
        ("hook_non_encoder_MACs_G", "Hook NonEnc(G)"),
        ("fvcore_FLOPs_G", "fvcore(G)"),
        ("thop_MACs_G", "THOP MACs(G)"),
        ("output_shape", "Output"),
        ("input_protocol", "Input protocol"),
    ]
    table = []
    for r in rows:
        line = []
        for key, _title in cols:
            value = r.get(key, "")
            if key.endswith("_M") or key.endswith("_G") or key == "input_area_factor_vs_base":
                line.append(format_float(value, 3))
            elif key == "output_shape":
                line.append("x".join(map(str, value)) if value else "")
            else:
                line.append(str(value))
        table.append(line)

    widths = [len(title) for _key, title in cols]
    for line in table:
        widths = [max(w, len(cell)) for w, cell in zip(widths, line)]

    header = " | ".join(title.ljust(w) for (_key, title), w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    print("\n" + header)
    print(sep)
    for line in table:
        print(" | ".join(cell.ljust(w) for cell, w in zip(line, widths)))
    print()


def json_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out


def save_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "params_flops_compare_v6_case06_only_safe.json"
    csv_path = out_dir / "params_flops_compare_v6_case06_only_safe.csv"
    txt_path = out_dir / "params_flops_compare_v6_case06_only_safe.txt"

    safe_rows = [json_safe_row(dict(r)) for r in rows]
    json_path.write_text(json.dumps(safe_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "name",
        "class_name",
        "family",
        "params_M",
        "trainable_params_M",
        "encoder_params_M",
        "non_encoder_params_M",
        "called_weighted_params_M",
        "encoder_forward_calls",
        "encoder_input_shapes",
        "input_area_factor_vs_base",
        "hook_MACs_G",
        "hook_FLOPs_2x_G",
        "hook_encoder_MACs_G",
        "hook_non_encoder_MACs_G",
        "fvcore_FLOPs_G",
        "thop_MACs_G",
        "thop_FLOPs_2x_G",
        "input_protocol",
        "input_shapes",
        "output_shape",
        "input_mode",
        "pretrained_requested",
        "requested_device",
        "actual_device",
        "checkpoint",
        "case06_source",
        "build_elapsed_sec",
        "hook_elapsed_sec",
        "hook_error",
        "fvcore_error",
        "thop_error",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in safe_rows:
            flat = dict(row)
            for key in ("input_shapes", "output_shape", "encoder_input_shapes"):
                flat[key] = json.dumps(row.get(key, []), ensure_ascii=False)
            writer.writerow(flat)

    lines = []
    lines.append("Params/FLOPs comparison v6 case06-only safe")
    lines.append("")
    for r in safe_rows:
        lines.append(f"[{r.get('name', '')}]")
        if r.get("error"):
            lines.append(f"  error: {r.get('error')}")
        lines.append(f"  class: {r.get('class_name', '')}")
        lines.append(f"  input: {r.get('input_protocol', '')}")
        lines.append(f"  input shapes: {r.get('input_shapes', '')}")
        if r.get("requested_device") or r.get("actual_device"):
            lines.append(f"  device: requested={r.get('requested_device', '')}, actual={r.get('actual_device', '')}")
        if r.get("case06_source"):
            lines.append(f"  case06 source: {r.get('case06_source')}")
        lines.append(f"  params: {format_float(r.get('params_M'), 3)} M")
        lines.append(f"  encoder params: {format_float(r.get('encoder_params_M'), 3)} M")
        lines.append(f"  non-encoder params: {format_float(r.get('non_encoder_params_M'), 3)} M")
        if "called_weighted_params_M" in r:
            lines.append(f"  called-weighted params*: {format_float(r.get('called_weighted_params_M'), 3)} M")
        if "input_area_factor_vs_base" in r:
            lines.append(f"  input area factor vs base: {format_float(r.get('input_area_factor_vs_base'), 3)} x")
        if "encoder_forward_calls" in r:
            lines.append(f"  encoder forward calls: {r.get('encoder_forward_calls')}")
            lines.append(f"  encoder input shapes: {r.get('encoder_input_shapes')}")
        if "hook_MACs_G" in r:
            lines.append(f"  hook MACs: {format_float(r.get('hook_MACs_G'), 3)} G")
            lines.append(f"  hook encoder MACs: {format_float(r.get('hook_encoder_MACs_G'), 3)} G")
            lines.append(f"  hook non-encoder MACs: {format_float(r.get('hook_non_encoder_MACs_G'), 3)} G")
            lines.append(f"  hook FLOPs if multiply+add=2 ops: {format_float(r.get('hook_FLOPs_2x_G'), 3)} G")
        if "fvcore_FLOPs_G" in r:
            lines.append(f"  fvcore FLOPs: {format_float(r.get('fvcore_FLOPs_G'), 3)} G")
        if "thop_MACs_G" in r:
            lines.append(f"  THOP MACs: {format_float(r.get('thop_MACs_G'), 3)} G")
        for key in ("hook_error", "fvcore_error", "thop_error"):
            if r.get(key):
                lines.append(f"  {key}: {r[key]}")
        lines.append("")
    lines.append("Notes:")
    lines.append("- Params are counted once per stored weight tensor; shared encoders are not multiplied by input scale count.")
    lines.append("- Hook MACs count actual Conv/Linear/SDPA calls in forward, so repeated encoder calls are reflected.")
    lines.append("- called-weighted params are diagnostic only: non_encoder_params + encoder_params * encoder_forward_calls.")
    lines.append("- For 1.5/1.0/0.5 multi-scale inputs, AreaX should be 3.5. For FSA image_m-only, AreaX should be 1.0.")
    lines.append("- fvcore/thop are optional and may differ because unsupported ops and FLOP/MAC conventions differ.")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"[saved] {json_path}")
    log(f"[saved] {csv_path}")
    log(f"[saved] {txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fsa-zip", default=DEFAULT_FSA_ZIP)
    parser.add_argument("--fsa-case06-zip", default=DEFAULT_FSA_CASE06_ZIP, help="FSA case06 full repo ZIP, e.g. FSA-ZoomNeXt_case06_only.zip; profile-tools patch ZIP is still accepted as fallback")
    parser.add_argument("--zoomnet-zip", default=DEFAULT_ZOOMNET_ZIP)
    parser.add_argument("--zoomnext-zip", default=DEFAULT_ZOOMNEXT_ZIP)
    parser.add_argument("--fsa-root", default="", help="Use extracted repo root instead of ZIP")
    parser.add_argument("--fsa-case06-root", default="", help="Use full extracted case06 repo root instead of ZIP")
    parser.add_argument("--fsa-case06-module", default="methods.zoomnext.fsa_zoomnext", help="FSA-case06 model module; use auto only for a full case06 repo with different names")
    parser.add_argument("--fsa-case06-class", default="PvtV2B2_FSA_ZoomNeXt", help="FSA-case06 model class; use auto only for a full case06 repo with different names")
    parser.add_argument("--fsa-case06-family", choices=["fsa_zoomnext", "zoomnext", "zoomnet"], default="fsa_zoomnext", help="Input protocol/wrapper family for FSA-case06")
    parser.add_argument("--require-fsa-case06", action="store_true", help="Fail instead of skipping when FSA-case06 ZIP/root is missing")
    parser.add_argument("--zoomnet-root", default="", help="Use extracted repo root instead of ZIP")
    parser.add_argument("--zoomnext-root", default="", help="Use extracted repo root instead of ZIP")
    parser.add_argument("--extract-dir", default="_profile_repos_v6_case06_only_safe")
    parser.add_argument("--out-dir", default="outputs_params_flops_compare_v6_case06_only_safe")

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--small-scale", type=float, default=0.5)
    parser.add_argument("--large-scale", type=float, default=1.5)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")

    parser.add_argument("--tool", choices=["params", "hook", "fvcore", "thop", "both", "all"], default="hook")
    parser.add_argument(
        "--hook-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --tool fvcore/thop/both, run the hook profiler first for diagnostics.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["explicit", "count_original", "none"],
        default="explicit",
        help="How to handle torch.nn.functional.scaled_dot_product_attention during hook profiling.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Number of top hook modules to store in JSON.")

    parser.add_argument("--input-mode", choices=["ones", "zeros", "randn", "image"], default="ones")
    parser.add_argument("--image", default="", help="Only used when --input-mode image")
    parser.add_argument("--seed", type=int, default=112358)
    parser.add_argument("--num-threads", type=int, default=0, help="Set torch CPU threads; 0 keeps PyTorch default")
    parser.add_argument(
        "--unsafe-use-cuda",
        action="store_true",
        help=(
            "Actually move models/tensors to CUDA. By default, CUDA execution is disabled "
            "for complexity profiling because a Windows fatal access violation inside "
            "model.to('cuda') cannot be caught by Python. Params/FLOPs are unchanged on CPU."
        ),
    )
    parser.add_argument(
        "--avoid-cuda-init",
        action="store_true",
        default=True,
        help="When profiling on CPU, fake CUDA device properties during repository model build.",
    )
    parser.add_argument(
        "--allow-cuda-init",
        dest="avoid_cuda_init",
        action="store_false",
        help="Allow repository code to query real CUDA device properties during CPU profiling.",
    )

    parser.add_argument(
        "--load-pretrained",
        action="store_true",
        help="Actually request repository pretrained backbone weights. Not needed for Params/FLOPs; may download weights.",
    )
    parser.add_argument(
        "--no-fast-init",
        dest="fast_init",
        action="store_false",
        help="Disable fast zero initialization patch. Slower; weight values still do not affect Params/FLOPs.",
    )
    parser.set_defaults(fast_init=True)

    parser.add_argument("--fsa-checkpoint", default="")
    parser.add_argument("--fsa-case06-checkpoint", default="")
    parser.add_argument("--zoomnext-checkpoint", default="")
    parser.add_argument("--zoomnet-checkpoint", default="")
    parser.add_argument("--strict-checkpoint", action="store_true")

    parser.add_argument("--skip-fsa", action="store_true")
    parser.add_argument("--skip-fsa-case06", action="store_true")
    parser.add_argument("--skip-zoomnext", action="store_true")
    parser.add_argument("--skip-zoomnet", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    torch.set_grad_enabled(False)

    args.requested_device = args.device
    if args.device.startswith("cuda") and not args.unsafe_use_cuda:
        log(
            "[device] CUDA was requested, but CPU-safe profiling is active. "
            "Using CPU to avoid fatal Windows/PyTorch access violations at model.to('cuda'). "
            "Add --unsafe-use-cuda only if CUDA execution is intentional."
        )
        args.device = "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    log(
        f"[env] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
        f"requested_device={args.requested_device}, actual_device={args.device}, platform={platform.system()}"
    )
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        try:
            log(f"[env] cuda_device={torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    roots = resolve_roots(args)
    jobs = default_jobs(load_pretrained=args.load_pretrained)

    if "fsa_case06" in roots and not args.skip_fsa_case06:
        case06_job = Job(
            name="FSA-ZoomNeXt-case06",
            root_key="fsa_case06",
            module=args.fsa_case06_module,
            class_name=args.fsa_case06_class,
            family=args.fsa_case06_family,
            kwargs=case06_default_kwargs(args.load_pretrained),
        )
        jobs.insert(1, case06_job)

    for job in jobs:
        if job.root_key == "fsa":
            job.checkpoint = args.fsa_checkpoint or None
        elif job.root_key == "fsa_case06":
            job.checkpoint = args.fsa_case06_checkpoint or None
        elif job.root_key == "zoomnext":
            job.checkpoint = args.zoomnext_checkpoint or None
        elif job.root_key == "zoomnet":
            job.checkpoint = args.zoomnet_checkpoint or None

    filtered_jobs = []
    for job in jobs:
        if args.skip_fsa and job.root_key == "fsa":
            continue
        if args.skip_fsa_case06 and job.root_key == "fsa_case06":
            continue
        if job.root_key == "fsa_case06" and job.root_key not in roots:
            continue
        if args.skip_zoomnext and job.root_key == "zoomnext":
            continue
        if args.skip_zoomnet and job.root_key == "zoomnet":
            continue
        filtered_jobs.append(job)

    rows: List[Dict[str, Any]] = []
    for job in filtered_jobs:
        try:
            rows.append(profile_one(job, roots[job.root_key], args))
        except Exception as exc:
            row = dict(
                name=job.name,
                class_name=job.class_name,
                family=job.family,
                error=repr(exc),
                traceback=traceback.format_exc(limit=12),
            )
            rows.append(row)
            log(f"[ERROR] {job.name}: {exc!r}")
            if args.stop_on_error:
                raise

    print_table(rows)
    save_outputs(rows, Path(args.out_dir))

    log(
        "\nInterpretation hints:\n"
        "- If ZoomNeXt encoder_forward_calls is 3 and each FSA variant is 1, the forward path is being compared correctly.\n"
        "- Params can remain close because the 3-scale models reuse one shared encoder; they do not store three encoders.\n"
        "- CalledWt(M) is not standard Params; it is only a diagnostic for shared-encoder reuse.\n"
        "- FSA-case06 uses the case06-only repo when --fsa-case06-zip points to FSA-ZoomNeXt_case06_only.zip; config: mid_dim=64, av_gate_mode=residual_spatial.\n- For 384x384 with 1.5/1.0/0.5 scales, AreaX should be 3.5 for ZoomNeXt/ZoomNet and 1.0 for FSA variants.\n"
        "- If fvcore/thop differ from hook MACs, prefer one tool consistently and report the tool/convention in the table caption.\n"
        "- For the first diagnostic run, use --tool hook. Use --tool all only after hook results look sane."
    )


if __name__ == "__main__":
    main()
