import argparse
import csv
import gc
import importlib
import inspect
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

SCRIPT_VERSION = "measure_compare_sinet_v8_2026_05_28"


# -----------------------------------------------------------------------------
# Default Windows paths from the user
# -----------------------------------------------------------------------------
DEFAULT_OFFICIAL_ROOT = r"C:\Users\User\Downloads\ZoomNeXt-main.zip"
DEFAULT_LITE_ROOT = r"E:\COD\BRG-ZoomNeXt-Lite-EvalFix"
DEFAULT_CHANGE_ROOT = r"E:\COD\ModelCompare\ChangeModule.zip"
DEFAULT_FSA_ROOT = r"E:\COD\ModelCompare\FSA-ZoomNeXt.zip"
DEFAULT_FSA_MOBILE_ROOT = r"E:\COD\ModelCompare\FSA-ZoomNeXt_mobilenet_metrics.zip"
DEFAULT_FSA_CASE02_ROOT = r"E:\COD\ModelCompare\FSA-ZoomNeXt_case02_only.zip"
DEFAULT_SINET_ROOT = r"E:\COD\ModelCompare\SINet-V2-main.zip"

ADDED_SYS_PATHS: List[str] = []


# -----------------------------------------------------------------------------
# Path / import utilities
# -----------------------------------------------------------------------------
def find_project_root(root: Path) -> Path:
    root = root.resolve()

    # ZoomNeXt/FSA-style projects usually expose a methods/ package.
    if (root / "methods").is_dir():
        return root

    # SINet/SINet-V2 repositories often use Src/, lib/, model(s)/, or network(s)/.
    package_markers = ["methods", "Src", "src", "lib", "libs", "model", "models", "network", "networks"]
    if any((root / marker).is_dir() for marker in package_markers):
        return root

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any((child / marker).is_dir() for marker in package_markers):
            return child.resolve()

    for marker in package_markers:
        for p in root.rglob(marker):
            if p.is_dir():
                return p.parent.resolve()

    # Common zip layout: extracted_dir/project-main/*.py without methods/.
    child_dirs = [c for c in root.iterdir() if c.is_dir() and not c.name.startswith("__")]
    if len(child_dirs) == 1 and any(child_dirs[0].rglob("*.py")):
        return child_dirs[0].resolve()

    return root


def resolve_root(path_str: str, fresh_extract: bool = False) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    if p.is_dir():
        return find_project_root(p)

    if p.is_file() and p.suffix.lower() == ".zip":
        # Use mtime to avoid stale extraction when zip file is replaced.
        mtime = int(p.stat().st_mtime)
        out_dir = Path(tempfile.gettempdir()) / f"{p.stem}_{mtime}_extracted_for_model_compare"

        if fresh_extract and out_dir.exists():
            shutil.rmtree(out_dir)

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(out_dir)

        return find_project_root(out_dir)

    raise FileNotFoundError(f"Invalid root path: {path_str}")


def purge_project_modules():
    # All projects reuse methods.xxx and utils.xxx, so import cache must be cleared.
    prefixes = ("methods", "utils", "configs", "config", "datasets", "Src", "src", "lib", "libs", "model", "models", "network", "networks")
    for name in list(sys.modules.keys()):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]
    importlib.invalidate_caches()


def set_project_path(root: Path):
    global ADDED_SYS_PATHS

    for p in ADDED_SYS_PATHS:
        while p in sys.path:
            sys.path.remove(p)
    ADDED_SYS_PATHS = []

    root_str = str(root.resolve())
    while root_str in sys.path:
        sys.path.remove(root_str)

    sys.path.insert(0, root_str)
    ADDED_SYS_PATHS.append(root_str)
    os.chdir(root)


def import_class_from_project(root: Path, module_name: str, class_name: str):
    purge_project_modules()
    set_project_path(root)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def discover_pvt_b2_classes(root: Path) -> List[Tuple[str, str]]:
    """Return candidate (module_name, class_name) pairs containing PvtV2B2."""
    candidates: List[Tuple[str, str]] = []
    methods_dir = root / "methods"
    if not methods_dir.is_dir():
        return candidates

    class_pattern = re.compile(
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*PvtV2B2[A-Za-z0-9_]*|PvtV2B2[A-Za-z0-9_]*)\s*\(",
        re.M,
    )

    for py_file in sorted(methods_dir.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        class_names = class_pattern.findall(text)
        if not class_names:
            continue

        rel = py_file.relative_to(root).with_suffix("")
        module_name = ".".join(rel.parts)
        for cls_name in class_names:
            candidates.append((module_name, cls_name))

    def score(item: Tuple[str, str]) -> Tuple[int, str, str]:
        module_name, cls_name = item
        s = 0
        low = (module_name + "." + cls_name).lower()
        if "fsa" in low:
            s -= 30
        if "zoomnext" in low:
            s -= 10
        if cls_name == "PvtV2B2_ZoomNeXt":
            s -= 5
        if "BRG" in cls_name or "Lite" in cls_name:
            s -= 3
        return (s, module_name, cls_name)

    return sorted(set(candidates), key=score)


def auto_import_pvt_b2_class(root: Path, label: str):
    candidates = discover_pvt_b2_classes(root)
    if not candidates:
        raise RuntimeError(
            f"No PvtV2B2 class found under {root}. "
            f"Use --{label.lower()}-module and --{label.lower()}-class manually."
        )

    last_error = None
    for module_name, class_name in candidates:
        try:
            cls = import_class_from_project(root, module_name, class_name)
            if inspect.isclass(cls):
                print(f"[INFO] Auto-selected {label} class: {module_name}.{class_name}")
                return cls, module_name, class_name, candidates
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not import any discovered PvtV2B2 class under {root}. "
        f"Last error: {last_error}"
    )




def discover_model_classes(root: Path, preferred_terms: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Generic model-class discovery for variants whose class name may not contain PvtV2B2,
    e.g. MobileNet-based FSA-ZoomNeXt.
    """
    preferred_terms = [t.lower() for t in (preferred_terms or [])]
    candidates: List[Tuple[str, str]] = []
    class_pattern = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)

    for py_file in sorted(root.rglob("*.py")):
        parts_low = [part.lower() for part in py_file.parts]
        if "__pycache__" in parts_low:
            continue
        if any(part in {"site-packages", "dist-packages", ".git", "build"} for part in parts_low):
            continue

        text = py_file.read_text(encoding="utf-8", errors="ignore")
        class_names = class_pattern.findall(text)
        if not class_names:
            continue

        try:
            rel = py_file.relative_to(root).with_suffix("")
        except ValueError:
            continue
        module_name = ".".join(rel.parts)

        for cls_name in class_names:
            low = f"{module_name}.{cls_name}".lower()
            base_terms = ["zoomnext", "fsa", "mobile", "mobilenet", "pvt", "sinet", "resnet", "res2net", "model", "network", "net"]
            if any(t in low for t in base_terms + preferred_terms):
                candidates.append((module_name, cls_name))

    bad_terms = [
        "conv", "bn", "relu", "norm", "loss", "dataset", "loader", "sampler",
        "metric", "recorder", "logger", "optimizer", "scheduler", "wrapper",
        "aspp", "attention", "block", "layer", "modulelist", "sequential",
    ]

    def score(item: Tuple[str, str]) -> Tuple[int, str, str]:
        module_name, cls_name = item
        low = f"{module_name}.{cls_name}".lower()
        s = 0
        # Prefer full network classes over individual blocks.
        if "zoomnext" in low:
            s -= 80
        if "fsa" in low:
            s -= 60
        if "mobile" in low or "mobilenet" in low:
            s -= 50
        if "sinet" in low:
            s -= 70
        if "res2net" in low or "resnet" in low:
            s -= 20
        if cls_name.lower() in {"network", "sinet", "sinetv2", "sinet_v2", "sinet_resnet50", "sinet_res2net50"}:
            s -= 40
        if "pvtv2b2" in low or "pvt_v2_b2" in low:
            s -= 20
        if cls_name.lower().startswith(("pvt", "fsa")):
            s -= 10
        for term in preferred_terms:
            if term in low:
                s -= 30
        if any(term in cls_name.lower() for term in bad_terms):
            s += 150
        if any(term in module_name.lower() for term in ["layers", "ops", "loss", "metric", "dataset", "utils"]):
            s += 100
        return (s, module_name, cls_name)

    return sorted(set(candidates), key=score)


def auto_import_model_class(root: Path, label: str, preferred_terms: Optional[List[str]] = None):
    candidates = discover_model_classes(root, preferred_terms=preferred_terms)
    if not candidates:
        raise RuntimeError(
            f"No model-like class found under {root}. Use explicit --<model>-module and --<model>-class."
        )

    last_error = None
    for module_name, class_name in candidates:
        try:
            cls = import_class_from_project(root, module_name, class_name)
            if inspect.isclass(cls):
                print(f"[INFO] Auto-selected {label} class: {module_name}.{class_name}")
                return cls, module_name, class_name, candidates
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not import any discovered model-like class under {root}. Last error: {last_error}"
    )

# -----------------------------------------------------------------------------
# Input wrappers
# -----------------------------------------------------------------------------
def make_multiscale_input(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    _, _, h, w = x.shape
    image_s = F.interpolate(
        x,
        size=(max(1, h // 2), max(1, w // 2)),
        mode="bilinear",
        align_corners=False,
    )
    image_m = x
    image_l = F.interpolate(
        x,
        size=(int(h * 1.5), int(w * 1.5)),
        mode="bilinear",
        align_corners=False,
    )
    return {
        "image": image_m,
        "image_s": image_s,
        "image_m": image_m,
        "image_l": image_l,
    }


def make_single_input(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"image": x, "image_m": x}


def make_image_m_only_input(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"image_m": x}


def first_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                t = first_tensor(item)
                if torch.is_tensor(t):
                    return t
            except RuntimeError:
                pass

    if isinstance(output, dict):
        for value in output.values():
            try:
                t = first_tensor(value)
                if torch.is_tensor(t):
                    return t
            except RuntimeError:
                pass

    raise RuntimeError(f"Cannot find tensor output from type: {type(output)}")


class FullWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_mode: str):
        super().__init__()
        self.model = model
        self.input_mode = input_mode

    def forward(self, x):
        if self.input_mode == "multiscale":
            data = make_multiscale_input(x)
            out = self.model(data)
        elif self.input_mode == "single":
            data = make_single_input(x)
            out = self.model(data)
        elif self.input_mode == "image_m_only":
            data = make_image_m_only_input(x)
            out = self.model(data)
        elif self.input_mode == "tensor":
            out = self.model(x)
        else:
            raise ValueError(f"Unknown input mode: {self.input_mode}")
        return first_tensor(out)


def get_encoder_like_module(model: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
    """
    Return the main feature extractor for encoder/backbone MAC breakdown.
    PVT models usually expose .encoder; MobileNet variants may expose .backbone or .features.
    """
    for name in ["encoder", "backbone", "feature_extractor", "features", "resnet", "res2net", "resnet50", "res2net50", "base", "base_model"]:
        if not hasattr(model, name):
            continue
        module = getattr(model, name)
        if isinstance(module, nn.Module) and not isinstance(module, nn.ModuleList):
            return name, module
    return None, None


class EncoderWrapper(nn.Module):
    """Profile only normalizer + encoder/backbone, not the full decoder."""
    def __init__(self, model: nn.Module):
        super().__init__()
        encoder_name, encoder = get_encoder_like_module(model)
        if encoder is None:
            raise AttributeError("model has no encoder/backbone-like module")
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.normalizer = getattr(model, "normalizer", nn.Identity())

    def forward(self, x):
        x = self.normalizer(x)
        feats = self.encoder(x)

        if isinstance(feats, dict):
            if "reduction_5" in feats:
                return feats["reduction_5"]
            return first_tensor(feats)

        if isinstance(feats, (tuple, list)):
            return feats[-1]

        return feats


# -----------------------------------------------------------------------------
# Measurement utilities
# -----------------------------------------------------------------------------
def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt_num(x) -> str:
    return clever_format([x], "%.3f")[0]


def profile_macs(wrapper: nn.Module, x: torch.Tensor, label: str) -> float:
    wrapper.eval()
    with torch.no_grad():
        y = first_tensor(wrapper(x))
        print(f"[INFO] {label} output shape: {tuple(y.shape)}")

    if x.is_cuda:
        torch.cuda.synchronize()

    with torch.no_grad():
        macs, _ = profile(wrapper, inputs=(x,), verbose=False)

    if x.is_cuda:
        torch.cuda.synchronize()

    return float(macs)


def print_top_level_breakdown(model: nn.Module, title: str):
    total = sum(p.numel() for p in model.parameters())
    print("-" * 120)
    print(f"[Top-level parameter breakdown] {title}")
    print("-" * 120)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        ratio = n / total * 100 if total > 0 else 0.0
        print(f"{name:45s} {n:15,}  {ratio:6.2f}%")
    print("-" * 120)


def get_init_signature_text(cls) -> str:
    try:
        return str(inspect.signature(cls.__init__))
    except Exception:
        return "<signature unavailable>"


def has_var_kwargs(cls) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return False
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def accepted_param_names(cls) -> set:
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return set()
    return {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def build_constructor_attempts(
    cls,
    *,
    pretrained: bool,
    mid_dim: Optional[int],
    role: str,
    lite_use_rfrm: bool = True,
    lite_use_gmffm: bool = True,
    fsa_d: Optional[int] = None,
    fsa_d_param: str = "mid_dim",
    extra_constructor_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build robust constructor kwargs. For FSA-ZoomNeXt, the source code uses mid_dim as the lightweight dimension d.
    """
    names = accepted_param_names(cls)
    var_kw = has_var_kwargs(cls)

    def accepts(k: str) -> bool:
        return var_kw or k in names

    base: Dict[str, Any] = {}
    if accepts("pretrained"):
        base["pretrained"] = pretrained
    if mid_dim is not None and accepts("mid_dim"):
        base["mid_dim"] = mid_dim

    if role == "lite":
        if accepts("use_rfrm"):
            base["use_rfrm"] = lite_use_rfrm
        if accepts("use_gmffm"):
            base["use_gmffm"] = lite_use_gmffm

    if role == "fsa" and fsa_d is not None:
        # In the provided FSA-ZoomNeXt zip, d is exposed as mid_dim. Keep user-selected alias first.
        if fsa_d_param and accepts(fsa_d_param):
            base[fsa_d_param] = fsa_d

        # Additional likely aliases. Do not use generic 'channels' to avoid breaking models.
        for k in ["d", "D", "dim", "fsa_d", "fsa_dim", "fsa_channels", "fsa_ch", "hidden_dim"]:
            if k not in base and accepts(k):
                base[k] = fsa_d

    # Optional model-specific constructor arguments, e.g., av_gate_mode for FSA case02.
    # Only pass keys accepted by the target class so the script remains reusable for other variants.
    if extra_constructor_kwargs:
        for k, v in extra_constructor_kwargs.items():
            if v is None:
                continue
            if accepts(k):
                base[k] = v
            else:
                print(f"[WARN] Constructor for {role} does not accept extra arg {k}={v}; it will be ignored.")

    attempts: List[Dict[str, Any]] = []
    attempts.append(dict(base))

    # Fallbacks while keeping FSA d if possible.
    no_pretrained = {k: v for k, v in base.items() if k != "pretrained"}
    attempts.append(no_pretrained)

    no_mid = {k: v for k, v in base.items() if k != "mid_dim"}
    attempts.append(no_mid)

    if role == "fsa" and fsa_d is not None:
        d_only: Dict[str, Any] = {}
        if fsa_d_param and accepts(fsa_d_param):
            d_only[fsa_d_param] = fsa_d
        elif accepts("d"):
            d_only["d"] = fsa_d
        if d_only:
            attempts.append(d_only)

    if accepts("pretrained"):
        attempts.append({"pretrained": pretrained})
    if mid_dim is not None and accepts("mid_dim"):
        attempts.append({"mid_dim": mid_dim})

    attempts.append({})

    # Deduplicate while preserving order.
    unique: List[Dict[str, Any]] = []
    seen = set()
    for a in attempts:
        key = tuple(sorted(a.items()))
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def instantiate_model(cls, attempts: List[Dict], label: str) -> Tuple[nn.Module, Dict]:
    last_error = None
    for kwargs in attempts:
        try:
            print(f"[INFO] Instantiate {label} with args: {kwargs}")
            model = cls(**kwargs)
            return model, kwargs
        except TypeError as e:
            print(f"[WARN] Instantiate failed for {label} with args {kwargs}: {e}")
            last_error = e
            continue
    raise RuntimeError(f"Failed to instantiate {label}. Last TypeError: {last_error}")


def select_input_mode(model: nn.Module, x: torch.Tensor, requested: str, label: str) -> str:
    if requested != "auto":
        wrapper = FullWrapper(model, requested).to(x.device)
        with torch.no_grad():
            _ = first_tensor(wrapper(x))
        print(f"[INFO] {label} input mode: {requested}")
        return requested

    # Prefer single first. If a model only uses image_m, multiscale would also work but would mislead the encoder breakdown.
    # A real multi-scale ZoomNeXt usually fails on single because image_s/image_l are missing.
    for mode in ["single", "image_m_only", "multiscale", "tensor"]:
        try:
            wrapper = FullWrapper(model, mode).to(x.device)
            with torch.no_grad():
                y = first_tensor(wrapper(x))
            print(f"[INFO] {label} auto input mode selected: {mode}, output shape={tuple(y.shape)}")
            return mode
        except Exception as e:
            print(f"[WARN] {label} input mode failed: {mode} ({type(e).__name__}: {e})")
            continue

    raise RuntimeError(f"Could not find a valid input mode for {label}")


def encoder_scales_for_input_mode(input_mode: str, h: int, w: int) -> List[Tuple[str, int, int]]:
    if input_mode == "multiscale":
        return [
            ("image_s", max(1, h // 2), max(1, w // 2)),
            ("image_m", h, w),
            ("image_l", int(h * 1.5), int(w * 1.5)),
        ]
    return [("image_m", h, w)]


def scan_d_like_attributes(model: nn.Module, max_rows: int = 80) -> List[Tuple[str, str, str, Any]]:
    """Find attributes that may indicate d/mid_dim/dim, especially inside FSA modules."""
    attr_names = {
        "d", "D", "dim", "dims", "mid_dim", "hidden_dim", "fsa_dim", "fsa_d", "fsa_channels",
        "channels", "reduction", "groups", "av_gate_mode",
    }
    rows: List[Tuple[str, str, str, Any]] = []

    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        low = f"{module_name}.{cls_name}".lower()
        interesting_module = any(k in low for k in ["fsa", "fs", "attention", "fusion", "siu", "rfrm", "hmu", "rgpu"])

        for attr in attr_names:
            if not hasattr(module, attr):
                continue
            try:
                value = getattr(module, attr)
            except Exception:
                continue
            if isinstance(value, (int, float, str, bool, tuple, list)):
                # Avoid huge lists.
                if isinstance(value, list) and len(value) > 8:
                    value = value[:8] + ["..."]
                if interesting_module or attr in {"d", "D", "dim", "mid_dim", "fsa_dim", "fsa_d"}:
                    rows.append((module_name or "<root>", cls_name, attr, value))
                    if len(rows) >= max_rows:
                        return rows
    return rows


def verify_fsa_d(label: str, model: nn.Module, used_kwargs: Dict, requested_d: int):
    print("\n" + "!" * 120)
    print(f"[FSA d-check] {label}")
    print("!" * 120)

    d_keys = [k for k in used_kwargs.keys() if k.lower() in {"d", "dim", "mid_dim", "fsa_d", "fsa_dim", "fsa_channels", "fsa_ch", "hidden_dim"}]
    if d_keys:
        for k in d_keys:
            val = used_kwargs[k]
            status = "OK" if val == requested_d else "MISMATCH"
            print(f"[FSA d-check] constructor arg {k}={val}  -> {status}")
    else:
        print("[WARN] No d-like constructor argument was accepted/used.")
        print("       In the provided FSA-ZoomNeXt zip, the expected d-like parameter is mid_dim.")
        print("       Check whether the root model has mid_dim equal to the requested d below.")

    rows = scan_d_like_attributes(model)
    if rows:
        print("[FSA d-check] d/mid_dim/dim-like attributes found in instantiated model:")
        for module_name, cls_name, attr, value in rows:
            marker = ""
            if isinstance(value, int):
                if value == requested_d:
                    marker = "  <== matches requested d"
                elif module_name.startswith("encoder.") and attr == "dim":
                    marker = "  <== PVT backbone channel, not FSA d"
                elif value in (32, 64, 96, 128):
                    marker = "  <== check this value"
            print(f"  {module_name:45s} | {cls_name:30s} | {attr:12s} = {value}{marker}")
    else:
        print("[FSA d-check] No obvious d-like attributes found.")

    print("!" * 120 + "\n")


def measure_model(
    label: str,
    root: Path,
    cls,
    device: str,
    h: int,
    w: int,
    input_mode: str,
    pretrained: bool,
    mid_dim: Optional[int],
    role: str,
    lite_use_rfrm: bool = True,
    lite_use_gmffm: bool = True,
    fsa_d: Optional[int] = None,
    fsa_d_param: str = "mid_dim",
    extra_constructor_kwargs: Optional[Dict[str, Any]] = None,
    print_breakdown: bool = True,
) -> Dict:
    print("\n" + "=" * 120)
    print(f"[MODEL] {label}")
    print(f"[MODEL] root : {root}")
    print(f"[MODEL] class: {cls.__module__}.{cls.__name__}")
    print(f"[MODEL] __init__ signature: {get_init_signature_text(cls)}")
    print("=" * 120)

    attempts = build_constructor_attempts(
        cls,
        pretrained=pretrained,
        mid_dim=mid_dim,
        role=role,
        lite_use_rfrm=lite_use_rfrm,
        lite_use_gmffm=lite_use_gmffm,
        fsa_d=fsa_d,
        fsa_d_param=fsa_d_param,
        extra_constructor_kwargs=extra_constructor_kwargs,
    )

    model, used_kwargs = instantiate_model(cls, attempts, label)
    model = model.to(device)
    model.eval()

    if role in {"fsa", "fsa_mobile"} and fsa_d is not None:
        verify_fsa_d(label, model, used_kwargs, fsa_d)

    full_params, full_trainable = count_params(model)
    encoder_attr_name, encoder_module = get_encoder_like_module(model)
    has_encoder = encoder_module is not None
    encoder_params = None
    encoder_trainable = None
    non_encoder_params = None
    if has_encoder:
        encoder_params, encoder_trainable = count_params(encoder_module)
        non_encoder_params = full_params - encoder_params

    if print_breakdown:
        print_top_level_breakdown(model, label)

    x = torch.randn(1, 3, h, w, device=device)
    selected_mode = select_input_mode(model, x, input_mode, label)
    full_wrapper = FullWrapper(model, selected_mode).to(device)
    full_macs = profile_macs(full_wrapper, x, f"{label} full model ({selected_mode})")

    encoder_runtime_macs = None
    encoder_details: List[Tuple[str, float]] = []

    if has_encoder:
        encoder_wrapper = EncoderWrapper(model).to(device)
        for scale_name, hh, ww in encoder_scales_for_input_mode(selected_mode, h, w):
            xx = torch.randn(1, 3, hh, ww, device=device)
            macs = profile_macs(encoder_wrapper, xx, f"{label} encoder {scale_name} {hh}x{ww}")
            encoder_details.append((f"{scale_name}_{hh}x{ww}", macs))
            del xx
        encoder_runtime_macs = sum(m for _, m in encoder_details)

    non_encoder_macs = None
    if encoder_runtime_macs is not None:
        non_encoder_macs = full_macs - encoder_runtime_macs
        if non_encoder_macs < 0:
            print("[WARN] non_encoder_macs is negative.")
            print("       This usually means the selected input mode was treated as multi-scale, but the model did not actually run all encoder scales,")
            print("       or THOP counted the encoder differently. Re-run with --<model>-input-mode single if this model uses image_m only.")

    print("\n" + "#" * 120)
    print(f"{label} summary")
    print("#" * 120)
    print(f"{'Part':48s} | {'Params':>14s} | {'MACs':>14s} | {'FLOPs(2x)':>14s}")
    print("-" * 120)
    if has_encoder:
        print(
            f"{('Encoder/backbone params (' + str(encoder_attr_name) + ')'):48s} | "
            f"{encoder_params / 1e6:12.3f}M | "
            f"{encoder_runtime_macs / 1e9:12.3f}G | "
            f"{encoder_runtime_macs * 2 / 1e9:12.3f}G"
        )
        for detail_name, detail_macs in encoder_details:
            print(
                f"{'  - ' + detail_name:48s} | "
                f"{'-':>14s} | "
                f"{detail_macs / 1e9:12.3f}G | "
                f"{detail_macs * 2 / 1e9:12.3f}G"
            )
        print(
            f"{'Non-encoder / decoder / fusion':48s} | "
            f"{non_encoder_params / 1e6:12.3f}M | "
            f"{non_encoder_macs / 1e9:12.3f}G | "
            f"{non_encoder_macs * 2 / 1e9:12.3f}G"
        )
    else:
        print(f"{'Encoder':48s} | {'N/A':>14s} | {'N/A':>14s} | {'N/A':>14s}")

    print(
        f"{'Total':48s} | "
        f"{full_params / 1e6:12.3f}M | "
        f"{full_macs / 1e9:12.3f}G | "
        f"{full_macs * 2 / 1e9:12.3f}G"
    )
    print("#" * 120)

    result = {
        "label": label,
        "root": str(root),
        "class": f"{cls.__module__}.{cls.__name__}",
        "init_signature": get_init_signature_text(cls),
        "used_kwargs": repr(used_kwargs),
        "input_mode": selected_mode,
        "encoder_attr": encoder_attr_name,
        "full_params": full_params,
        "full_trainable": full_trainable,
        "encoder_params": encoder_params,
        "encoder_trainable": encoder_trainable,
        "non_encoder_params": non_encoder_params,
        "full_macs": full_macs,
        "full_flops_2xmac": full_macs * 2,
        "encoder_runtime_macs": encoder_runtime_macs,
        "encoder_runtime_flops_2xmac": encoder_runtime_macs * 2 if encoder_runtime_macs is not None else None,
        "non_encoder_macs": non_encoder_macs,
        "non_encoder_flops_2xmac": non_encoder_macs * 2 if non_encoder_macs is not None else None,
    }

    del model, full_wrapper, x
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def reduction(old, new) -> Optional[float]:
    if old is None or new is None or old == 0:
        return None
    return (1.0 - new / old) * 100.0


def fmt_metric(value, unit: str) -> str:
    if value is None:
        return "N/A"
    if unit == "M":
        return f"{value / 1e6:.3f}M"
    if unit == "G":
        return f"{value / 1e9:.3f}G"
    return str(value)


def print_comparison(results: List[Dict], baseline_index: int = 0):
    baseline = results[baseline_index]
    print("\n" + "=" * 140)
    print(f"FINAL COMPARISON  |  baseline = {baseline['label']}")
    print("=" * 140)
    print(
        f"{'Model':32s} | {'Input':>10s} | {'Params':>12s} | {'Total MACs':>12s} | "
        f"{'Enc MACs':>12s} | {'Non-enc MACs':>14s} | {'Total MAC red.':>14s} | {'Param red.':>11s} | {'Used kwargs':>24s}"
    )
    print("-" * 140)

    for r in results:
        total_mac_red = reduction(baseline["full_macs"], r["full_macs"])
        param_red = reduction(baseline["full_params"], r["full_params"])
        kwargs_short = r.get("used_kwargs", "")
        if len(kwargs_short) > 24:
            kwargs_short = kwargs_short[:21] + "..."
        print(
            f"{r['label']:32s} | "
            f"{r['input_mode']:>10s} | "
            f"{fmt_metric(r['full_params'], 'M'):>12s} | "
            f"{fmt_metric(r['full_macs'], 'G'):>12s} | "
            f"{fmt_metric(r['encoder_runtime_macs'], 'G'):>12s} | "
            f"{fmt_metric(r['non_encoder_macs'], 'G'):>14s} | "
            f"{(f'{total_mac_red:.2f}%' if total_mac_red is not None else 'N/A'):>14s} | "
            f"{(f'{param_red:.2f}%' if param_red is not None else 'N/A'):>11s} | "
            f"{kwargs_short:>24s}"
        )

    print("=" * 140)
    print("[NOTE] MACs are reported by THOP. FLOPs(2xMAC) is printed in each model summary.")
    print("[NOTE] For multi-scale ZoomNeXt-style models, encoder runtime MACs = PVT(0.5x) + PVT(1.0x) + PVT(1.5x).")
    print("[NOTE] For FSA/FSA-MobileNet, check the '[FSA d-check]' section. In the provided FSA zips, d is usually implemented as mid_dim.")
    print("[NOTE] For FSA-case02, check Used kwargs or the model summary to confirm av_gate_mode is not 'none'.")
    print("[NOTE] SINet-V2 is measured as a generic tensor-input model; encoder/backbone breakdown is shown only if a backbone-like submodule is detected.")


def write_csv(results: List[Dict], csv_path: str):
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "label", "class", "input_mode", "encoder_attr", "used_kwargs", "init_signature",
        "full_params", "encoder_params", "non_encoder_params",
        "full_macs", "encoder_runtime_macs", "non_encoder_macs",
        "full_flops_2xmac", "encoder_runtime_flops_2xmac", "non_encoder_flops_2xmac",
        "root",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})

    print(f"[INFO] CSV saved: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)

    parser.add_argument("--official-root", type=str, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--lite-root", type=str, default=DEFAULT_LITE_ROOT)
    parser.add_argument("--change-root", type=str, default=DEFAULT_CHANGE_ROOT)
    parser.add_argument("--fsa-root", type=str, default=DEFAULT_FSA_ROOT)
    parser.add_argument("--fsa-mobile-root", type=str, default=DEFAULT_FSA_MOBILE_ROOT)
    parser.add_argument("--fsa-case02-root", type=str, default=DEFAULT_FSA_CASE02_ROOT)
    parser.add_argument("--sinet-root", type=str, default=DEFAULT_SINET_ROOT)

    parser.add_argument("--official-module", type=str, default="methods.zoomnext.zoomnext")
    parser.add_argument("--official-class", type=str, default="PvtV2B2_ZoomNeXt")

    parser.add_argument("--lite-module", type=str, default="methods.zoomnext.brg_lite")
    parser.add_argument("--lite-class", type=str, default="PvtV2B2_BRG_ZoomNeXt_Lite")

    parser.add_argument("--change-module", type=str, default=None)
    parser.add_argument("--change-class", type=str, default=None)
    parser.add_argument("--fsa-module", type=str, default=None)
    parser.add_argument("--fsa-class", type=str, default=None)
    parser.add_argument("--fsa-mobile-module", type=str, default=None)
    parser.add_argument("--fsa-mobile-class", type=str, default=None)
    parser.add_argument("--fsa-case02-module", type=str, default=None)
    parser.add_argument("--fsa-case02-class", type=str, default=None)
    parser.add_argument("--sinet-module", type=str, default=None)
    parser.add_argument("--sinet-class", type=str, default=None)

    parser.add_argument("--list-change-classes", action="store_true")
    parser.add_argument("--list-fsa-classes", action="store_true")
    parser.add_argument("--list-fsa-mobile-classes", action="store_true")
    parser.add_argument("--list-fsa-case02-classes", action="store_true")
    parser.add_argument("--list-sinet-classes", action="store_true")

    parser.add_argument("--h", type=int, default=384)
    parser.add_argument("--w", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fresh-extract", action="store_true")

    parser.add_argument("--official-mid-dim", type=int, default=64)
    parser.add_argument("--lite-mid-dim", type=int, default=48)
    parser.add_argument("--change-mid-dim", type=int, default=64)
    parser.add_argument("--fsa-mid-dim", type=int, default=48)
    parser.add_argument("--fsa-mobile-mid-dim", type=int, default=48)
    parser.add_argument("--fsa-case02-mid-dim", type=int, default=48)
    parser.add_argument("--sinet-channel", type=int, default=32, help="Constructor channel width for SINet/SINet-V2 if supported. Default is 32.")
    parser.add_argument("--sinet-pretrained-param", type=str, default="imagenet_pretrained", help="Optional SINet constructor flag name for pretrained backbone. Passed as the same value as --pretrained if supported.")

    # IMPORTANT: In the provided FSA-ZoomNeXt zip, the paper/feature dimension d is implemented as mid_dim.
    parser.add_argument("--fsa-d", type=int, default=48)
    parser.add_argument("--fsa-d-param", type=str, default="mid_dim", help="Constructor parameter name for FSA d. Default: mid_dim for the provided FSA zip")
    parser.add_argument("--fsa-mobile-d", type=int, default=48)
    parser.add_argument("--fsa-mobile-d-param", type=str, default="mid_dim", help="Constructor parameter name for FSA-MobileNet d. Default: mid_dim")
    parser.add_argument("--fsa-case02-d", type=int, default=48)
    parser.add_argument("--fsa-case02-d-param", type=str, default="mid_dim", help="Constructor parameter name for FSA case02 d. Default: mid_dim")
    parser.add_argument(
        "--fsa-case02-av-gate-mode",
        type=str,
        default="all",
        choices=["all", "none", "global", "spatial", "residual_spatial", "stage5"],
        help=(
            "Value passed to FSA case02 constructor argument av_gate_mode. "
            "Valid modes discovered from the case02 code are: none, global, spatial, "
            "residual_spatial, stage5. Use 'all' to measure all valid modes. Default: all"
        ),
    )

    parser.add_argument("--official-input-mode", type=str, default="multiscale", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--lite-input-mode", type=str, default="single", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--change-input-mode", type=str, default="auto", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--fsa-input-mode", type=str, default="auto", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--fsa-mobile-input-mode", type=str, default="auto", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--fsa-case02-input-mode", type=str, default="auto", choices=["auto", "multiscale", "single", "image_m_only", "tensor"])
    parser.add_argument("--sinet-input-mode", type=str, default="auto", choices=["auto", "multiscale", "single", "image_m_only", "tensor"], help="SINet-V2 usually accepts a raw tensor; auto will try tensor after dict modes.")

    parser.add_argument("--lite-no-gmffm", action="store_true")
    parser.add_argument("--lite-use-rgpu", action="store_true")

    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--no-breakdown", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-lite", action="store_true")
    parser.add_argument("--skip-change", action="store_true")
    parser.add_argument("--skip-fsa", action="store_true")
    parser.add_argument("--skip-fsa-mobile", action="store_true")
    parser.add_argument("--skip-fsa-case02", action="store_true")
    parser.add_argument("--skip-sinet", action="store_true")
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail immediately when an optional model path is missing. Default behavior is to skip missing optional models.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Use CPU instead.")
        device = "cpu"

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Base input: 1x3x{args.h}x{args.w}")
    print(f"[INFO] Official root: {args.official_root}")
    print(f"[INFO] Lite root    : {args.lite_root}")
    print(f"[INFO] Change root  : {args.change_root}")
    print(f"[INFO] FSA root     : {args.fsa_root}")
    print(f"[INFO] FSA-Mobile root: {args.fsa_mobile_root}")
    print(f"[INFO] FSA case02 root: {args.fsa_case02_root}")
    print(f"[INFO] SINet-V2 root: {args.sinet_root}")
    print(f"[INFO] Requested FSA d: {args.fsa_d} via parameter name '{args.fsa_d_param}' (provided FSA zip uses mid_dim)")
    print(f"[INFO] Requested FSA-MobileNet d: {args.fsa_mobile_d} via parameter name '{args.fsa_mobile_d_param}'")
    print(f"[INFO] Requested FSA-case02 d: {args.fsa_case02_d} via parameter name '{args.fsa_case02_d_param}'")
    print(f"[INFO] Requested FSA-case02 av_gate_mode: {args.fsa_case02_av_gate_mode}")
    print(f"[INFO] Requested SINet channel: {args.sinet_channel}, pretrained flag param: {args.sinet_pretrained_param}")

    def safe_resolve(label: str, path_str: str, required: bool = False):
        try:
            return resolve_root(path_str, fresh_extract=args.fresh_extract)
        except FileNotFoundError as e:
            if required or args.strict_paths:
                raise
            print(f"[WARN] {label} root not found, so this model will be skipped: {path_str}")
            print(f"       {e}")
            return None

    if args.list_change_classes:
        change_root = resolve_root(args.change_root, fresh_extract=args.fresh_extract)
        cands = discover_pvt_b2_classes(change_root)
        print(f"[INFO] ChangeModule project root: {change_root}")
        print("[INFO] Discovered ChangeModule PvtV2B2 classes:")
        for module_name, class_name in cands:
            print(f"  {module_name}.{class_name}")
        return

    if args.list_fsa_classes:
        fsa_root = resolve_root(args.fsa_root, fresh_extract=args.fresh_extract)
        cands = discover_pvt_b2_classes(fsa_root)
        print(f"[INFO] FSA project root: {fsa_root}")
        print("[INFO] Discovered FSA PvtV2B2 classes:")
        for module_name, class_name in cands:
            print(f"  {module_name}.{class_name}")
        return

    if args.list_fsa_mobile_classes:
        mobile_root = resolve_root(args.fsa_mobile_root, fresh_extract=args.fresh_extract)
        cands = discover_model_classes(mobile_root, preferred_terms=["fsa", "zoomnext", "mobile", "mobilenet"])
        print(f"[INFO] FSA-MobileNet project root: {mobile_root}")
        print("[INFO] Discovered FSA-MobileNet model-like classes:")
        for module_name, class_name in cands:
            print(f"  {module_name}.{class_name}")
        return

    if args.list_fsa_case02_classes:
        case02_root = resolve_root(args.fsa_case02_root, fresh_extract=args.fresh_extract)
        cands = discover_pvt_b2_classes(case02_root)
        if not cands:
            cands = discover_model_classes(case02_root, preferred_terms=["fsa", "zoomnext", "case02", "pvt"])
        print(f"[INFO] FSA-case02 project root: {case02_root}")
        print("[INFO] Discovered FSA-case02 model classes:")
        for module_name, class_name in cands:
            print(f"  {module_name}.{class_name}")
        return
    if args.list_sinet_classes:
        sinet_root = resolve_root(args.sinet_root, fresh_extract=args.fresh_extract)
        cands = discover_model_classes(sinet_root, preferred_terms=["sinet", "resnet", "res2net", "network", "model"])
        print(f"[INFO] SINet-V2 project root: {sinet_root}")
        print("[INFO] Discovered SINet-V2 model-like classes:")
        for module_name, class_name in cands:
            print(f"  {module_name}.{class_name}")
        return


    results: List[Dict] = []

    if not args.skip_official:
        official_root = safe_resolve("Official", args.official_root, required=True)
        OfficialCls = import_class_from_project(official_root, args.official_module, args.official_class)
        results.append(
            measure_model(
                label="Official ZoomNeXt-B2",
                root=official_root,
                cls=OfficialCls,
                device=device,
                h=args.h,
                w=args.w,
                input_mode=args.official_input_mode,
                pretrained=args.pretrained,
                mid_dim=args.official_mid_dim,
                role="official",
                print_breakdown=not args.no_breakdown,
            )
        )

    if not args.skip_lite:
        lite_root = safe_resolve("Lite", args.lite_root, required=True)
        LiteCls = import_class_from_project(lite_root, args.lite_module, args.lite_class)
        results.append(
            measure_model(
                label="BRG-ZoomNeXt-Lite",
                root=lite_root,
                cls=LiteCls,
                device=device,
                h=args.h,
                w=args.w,
                input_mode=args.lite_input_mode,
                pretrained=args.pretrained,
                mid_dim=args.lite_mid_dim,
                role="lite",
                lite_use_rfrm=not args.lite_use_rgpu,
                lite_use_gmffm=not args.lite_no_gmffm,
                print_breakdown=not args.no_breakdown,
            )
        )

    if not args.skip_change:
        change_root = safe_resolve("ChangeModule", args.change_root, required=False)
        if change_root is not None:
            if args.change_module and args.change_class:
                ChangeCls = import_class_from_project(change_root, args.change_module, args.change_class)
            else:
                ChangeCls, _, _, candidates = auto_import_pvt_b2_class(change_root, "Change")
                if len(candidates) > 1:
                    print("[INFO] Other discovered ChangeModule candidates:")
                    for module_name, class_name in candidates:
                        print(f"  {module_name}.{class_name}")
            results.append(
                measure_model(
                    label="ChangeModule-B2",
                    root=change_root,
                    cls=ChangeCls,
                    device=device,
                    h=args.h,
                    w=args.w,
                    input_mode=args.change_input_mode,
                    pretrained=args.pretrained,
                    mid_dim=args.change_mid_dim,
                    role="change",
                    print_breakdown=not args.no_breakdown,
                )
            )

    if not args.skip_fsa:
        fsa_root = safe_resolve("FSA", args.fsa_root, required=False)
        if fsa_root is not None:
            if args.fsa_module and args.fsa_class:
                FsaCls = import_class_from_project(fsa_root, args.fsa_module, args.fsa_class)
            else:
                FsaCls, _, _, candidates = auto_import_pvt_b2_class(fsa_root, "FSA")
                if len(candidates) > 1:
                    print("[INFO] Other discovered FSA candidates:")
                    for module_name, class_name in candidates:
                        print(f"  {module_name}.{class_name}")
            results.append(
                measure_model(
                label="FSA-ZoomNeXt-B2",
                root=fsa_root,
                cls=FsaCls,
                device=device,
                h=args.h,
                w=args.w,
                input_mode=args.fsa_input_mode,
                pretrained=args.pretrained,
                mid_dim=args.fsa_mid_dim,
                role="fsa",
                fsa_d=args.fsa_d,
                fsa_d_param=args.fsa_d_param,
                print_breakdown=not args.no_breakdown,
            )
        )

    if not args.skip_fsa_case02:
        case02_root = safe_resolve("FSA-case02", args.fsa_case02_root, required=False)
        if case02_root is not None:
            if args.fsa_case02_module and args.fsa_case02_class:
                Case02Cls = import_class_from_project(case02_root, args.fsa_case02_module, args.fsa_case02_class)
            else:
                try:
                    Case02Cls, _, _, candidates = auto_import_pvt_b2_class(case02_root, "FSA-case02")
                except RuntimeError:
                    Case02Cls, _, _, candidates = auto_import_model_class(
                        case02_root,
                        "FSA-case02",
                        preferred_terms=["fsa", "zoomnext", "case02", "pvt"],
                    )
                if len(candidates) > 1:
                    print("[INFO] Other discovered FSA-case02 candidates:")
                    for module_name, class_name in candidates:
                        print(f"  {module_name}.{class_name}")
            valid_case02_modes = ["none", "global", "spatial", "residual_spatial", "stage5"]
            if args.fsa_case02_av_gate_mode == "all":
                modes_to_measure = valid_case02_modes
            else:
                modes_to_measure = [args.fsa_case02_av_gate_mode]

            for gate_mode in modes_to_measure:
                results.append(
                    measure_model(
                        label=f"FSA-ZoomNeXt-case02-{gate_mode}",
                        root=case02_root,
                        cls=Case02Cls,
                        device=device,
                        h=args.h,
                        w=args.w,
                        input_mode=args.fsa_case02_input_mode,
                        pretrained=args.pretrained,
                        mid_dim=args.fsa_case02_mid_dim,
                        role="fsa",
                        fsa_d=args.fsa_case02_d,
                        fsa_d_param=args.fsa_case02_d_param,
                        extra_constructor_kwargs={"av_gate_mode": gate_mode},
                        print_breakdown=not args.no_breakdown,
                    )
                )

    if not args.skip_fsa_mobile:
        mobile_root = safe_resolve("FSA-MobileNet", args.fsa_mobile_root, required=False)
        if mobile_root is not None:
            if args.fsa_mobile_module and args.fsa_mobile_class:
                MobileCls = import_class_from_project(mobile_root, args.fsa_mobile_module, args.fsa_mobile_class)
            else:
                MobileCls, _, _, candidates = auto_import_model_class(
                    mobile_root,
                    "FSA-MobileNet",
                    preferred_terms=["fsa", "zoomnext", "mobile", "mobilenet"],
                )
                if len(candidates) > 1:
                    print("[INFO] Other discovered FSA-MobileNet candidates:")
                    for module_name, class_name in candidates:
                        print(f"  {module_name}.{class_name}")
            results.append(
                measure_model(
                    label="FSA-ZoomNeXt-MobileNet",
                    root=mobile_root,
                    cls=MobileCls,
                    device=device,
                    h=args.h,
                    w=args.w,
                    input_mode=args.fsa_mobile_input_mode,
                    pretrained=args.pretrained,
                    mid_dim=args.fsa_mobile_mid_dim,
                    role="fsa_mobile",
                    fsa_d=args.fsa_mobile_d,
                    fsa_d_param=args.fsa_mobile_d_param,
                    print_breakdown=not args.no_breakdown,
                )
            )

    if not args.skip_sinet:
        sinet_root = safe_resolve("SINet-V2", args.sinet_root, required=False)
        if sinet_root is not None:
            if args.sinet_module and args.sinet_class:
                SinetCls = import_class_from_project(sinet_root, args.sinet_module, args.sinet_class)
            else:
                SinetCls, _, _, candidates = auto_import_model_class(
                    sinet_root,
                    "SINet-V2",
                    preferred_terms=["sinet", "resnet", "res2net", "network", "model"],
                )
                if len(candidates) > 1:
                    print("[INFO] Other discovered SINet-V2 candidates:")
                    for module_name, class_name in candidates:
                        print(f"  {module_name}.{class_name}")
            results.append(
                measure_model(
                    label="SINet-V2",
                    root=sinet_root,
                    cls=SinetCls,
                    device=device,
                    h=args.h,
                    w=args.w,
                    input_mode=args.sinet_input_mode,
                    pretrained=args.pretrained,
                    mid_dim=None,
                    role="sinet",
                    extra_constructor_kwargs={
                        "channel": args.sinet_channel,
                        "channels": args.sinet_channel,
                        "ch": args.sinet_channel,
                        args.sinet_pretrained_param: args.pretrained,
                    },
                    print_breakdown=not args.no_breakdown,
                )
            )

    if results:
        print_comparison(results, baseline_index=0)
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
