#!/usr/bin/env python3
"""RunPod preflight and auto-fix helper for FSA-ZoomNeXt experiments.

This script is deliberately stdlib-only. It checks the exact issues that have
appeared during RunPod training:
  1) PyTorch CUDA build missing the current GPU architecture, e.g. sm_120.
  2) NumPy 2.x being installed after torch reinstall, breaking SciPy,
     albumentations, matplotlib, and pysodmetrics.
  3) The BN-safe ASPP patch not being present.
  4) Broken/corrupted project files or stale pycache.

The train/debug/eval shell scripts call this before importing the training
module, so import-time failures can be repaired before training starts.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable

TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128")
TORCH_NIGHTLY_INDEX_URL = os.environ.get("TORCH_NIGHTLY_INDEX_URL", "https://download.pytorch.org/whl/nightly/cu128")

PROJECT_DEPS = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "matplotlib==3.8.2",
    "albumentations==1.3.1",
    "opencv-python-headless==4.9.0.80",
    "pysodmetrics==1.4.2",
    "timm>=0.6.13",
    "mmengine",
    "colorlog",
    "PyYAML",
    "tqdm",
    "einops",
    "tensorboard",
    "openpyxl",
    "fvcore",
    "thop",
]


def _run(cmd: list[str], *, check: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    print("[preflight] $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, env=env)


def _capture_python(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "0")
    return subprocess.run([PYTHON, "-c", code], text=True, capture_output=True, env=env)


def _pip_install(args: list[str]) -> None:
    _run([PYTHON, "-m", "pip", "install", "--no-cache-dir", *args], check=True)


def _pip_uninstall(pkgs: list[str]) -> None:
    _run([PYTHON, "-m", "pip", "uninstall", "-y", *pkgs], check=False)


def check_torch_cuda(verbose: bool = True) -> tuple[bool, str]:
    code = r'''
import sys
try:
    import torch
except Exception as e:
    print("IMPORT_TORCH_FAIL", repr(e))
    raise SystemExit(2)
print("python:", sys.executable)
print("torch file:", torch.__file__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    expected = f"sm_{cap[0]}{cap[1]}"
    print("device:", name)
    print("capability:", cap)
    print("arch list:", archs)
    if expected not in archs:
        print(f"ARCH_MISSING expected={expected}")
        raise SystemExit(3)
    x = torch.rand(4, 3, 384, 384, device="cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    y = x.sub(mean)
    torch.cuda.synchronize()
    print("CUDA_OP_OK", tuple(y.shape))
'''
    p = _capture_python(code)
    out = (p.stdout or "") + (p.stderr or "")
    if verbose:
        print(out, end="")
    ok = p.returncode == 0 and "CUDA_OP_OK" in out
    return ok, out


def install_torch_cuda128(use_nightly: bool = False) -> None:
    index_url = TORCH_NIGHTLY_INDEX_URL if use_nightly else TORCH_INDEX_URL
    _pip_uninstall(["torch", "torchvision", "torchaudio", "xformers", "triton"])
    _run([PYTHON, "-m", "pip", "cache", "purge"], check=False)
    _run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
        ],
        check=True,
    )


def check_numpy_stack(verbose: bool = True) -> tuple[bool, str]:
    code = r'''
import sys
mods = []
try:
    import numpy as np
    print("numpy:", np.__version__)
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print("NUMPY_TOO_NEW")
        raise SystemExit(4)
except Exception as e:
    print("NUMPY_FAIL", repr(e))
    raise SystemExit(5)
for name in ["scipy", "albumentations", "cv2", "matplotlib", "py_sod_metrics", "yaml", "mmengine", "timm", "colorlog"]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "<no __version__>")
        print(f"{name}: {ver}")
    except Exception as e:
        print(f"IMPORT_FAIL {name}: {repr(e)}")
        raise SystemExit(6)
print("NUMPY_STACK_OK")
'''
    p = _capture_python(code)
    out = (p.stdout or "") + (p.stderr or "")
    if verbose:
        print(out, end="")
    ok = p.returncode == 0 and "NUMPY_STACK_OK" in out
    return ok, out


def install_project_deps() -> None:
    # Keep PyTorch untouched. We pin NumPy<2 after PyTorch installation because
    # the cu128 torch wheel may pull NumPy 2.x, which breaks SciPy/albumentations
    # ABI-compatible imports in this project.
    _pip_install(["--force-reinstall", "--no-deps", "numpy==1.26.4"])
    _pip_install(PROJECT_DEPS)
    _pip_install(["--force-reinstall", "--no-deps", "numpy==1.26.4"])


def verify_source_patch() -> tuple[bool, str]:
    files = [
        ROOT / "methods" / "zoomnext" / "fsa_layers.py",
        ROOT / "methods" / "zoomnext" / "fsa_zoomnext.py",
        ROOT / "utils" / "pipeline" / "scheduler.py",
    ]
    messages = []
    ok = True
    for path in files:
        try:
            txt = path.read_text(encoding="utf-8")
            messages.append(f"read OK: {path.relative_to(ROOT)}")
        except Exception as e:
            ok = False
            messages.append(f"read FAIL: {path} :: {e!r}")
            continue
        if path.name == "fsa_zoomnext.py":
            if "SafeSimpleASPP" not in txt or "self.tra_5 = SafeSimpleASPP" not in txt:
                ok = False
                messages.append(f"SafeSimpleASPP patch missing in {path.relative_to(ROOT)}")
    return ok, "\n".join(messages)


def clean_pycache() -> None:
    for p in ROOT.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    for p in ROOT.rglob("*.pyc"):
        try:
            p.unlink()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-fix", action="store_true", help="Install/fix packages if checks fail.")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--clean-pycache", action="store_true")
    args = parser.parse_args()

    if args.clean_pycache:
        clean_pycache()

    ok_patch, patch_msg = verify_source_patch()
    print("[preflight] source patch check:")
    print(patch_msg)
    if not ok_patch:
        print("[preflight] Source files are not in the expected patched state. Re-extract the latest zip into a clean folder.")
        return 20

    if not args.skip_torch:
        torch_ok, torch_out = check_torch_cuda(verbose=True)
        if not torch_ok:
            print("[preflight] PyTorch/CUDA check failed.")
            if not args.auto_fix:
                return 30
            print("[preflight] Trying stable cu128 PyTorch reinstall...")
            install_torch_cuda128(use_nightly=False)
            torch_ok, _ = check_torch_cuda(verbose=True)
            if not torch_ok:
                print("[preflight] Stable cu128 still failed. Trying nightly cu128...")
                install_torch_cuda128(use_nightly=True)
                torch_ok, _ = check_torch_cuda(verbose=True)
            if not torch_ok:
                print("[preflight] PyTorch CUDA still failed. Use a RunPod PyTorch image that supports your GPU architecture.")
                return 31

    if not args.skip_deps:
        deps_ok, deps_out = check_numpy_stack(verbose=True)
        if not deps_ok:
            print("[preflight] Python dependency check failed.")
            if not args.auto_fix:
                return 40
            print("[preflight] Reinstalling/pinning project dependencies and numpy<2...")
            install_project_deps()
            deps_ok, _ = check_numpy_stack(verbose=True)
            if not deps_ok:
                print("[preflight] Dependency check still failed.")
                return 41

    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
