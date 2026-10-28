import argparse
import gc
import importlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format


# ============================================================
# 기본 경로: 사용자가 준 경로
# ============================================================
DEFAULT_LITE_ROOT = r"E:\COD\BRG-ZoomNeXt-Lite-EvalFix"
DEFAULT_OFFICIAL_ROOT = r"C:\Users\User\Downloads\ZoomNeXt-main.zip"

ADDED_SYS_PATHS = []


# ============================================================
# Path / import utilities
# ============================================================
def resolve_root(path_str: str) -> Path:
    p = Path(path_str).expanduser()

    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    if p.is_dir():
        return find_project_root(p)

    if p.is_file() and p.suffix.lower() == ".zip":
        out_dir = Path(tempfile.gettempdir()) / f"{p.stem}_extracted_for_flops"

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(out_dir)

        return find_project_root(out_dir)

    raise FileNotFoundError(f"Invalid root path: {path_str}")


def find_project_root(root: Path) -> Path:
    """
    root 또는 하위 폴더 중 methods 폴더가 있는 실제 프로젝트 루트를 찾는다.
    zip 압축 해제 시 ZoomNeXt-main/ZoomNeXt-main 같은 구조가 생기는 경우 대응.
    """
    root = root.resolve()

    if (root / "methods").is_dir():
        return root

    for child in root.iterdir():
        if child.is_dir() and (child / "methods").is_dir():
            return child.resolve()

    for p in root.rglob("methods"):
        if p.is_dir():
            return p.parent.resolve()

    return root


def purge_project_modules():
    """
    Official ZoomNeXt와 Lite가 둘 다 methods.xxx 이름을 쓰므로,
    프로젝트를 바꿔 import할 때 기존 모듈 캐시를 제거해야 한다.
    """
    prefixes = ("methods", "utils", "configs", "config")

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


def import_project_class(root: Path, module_name: str, class_name: str):
    purge_project_modules()
    set_project_path(root)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


# ============================================================
# Wrappers
# ============================================================
def make_official_multiscale_input(x: torch.Tensor):
    """
    Official ZoomNeXt 입력 구조:
    image_s = 0.5x
    image_m = 1.0x
    image_l = 1.5x
    """
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


class OfficialFullWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        data = make_official_multiscale_input(x)
        out = self.model(data)
        return first_tensor(out)


class LiteFullWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        data = {
            "image": x,
            "image_m": x,
        }
        out = self.model(data)
        return first_tensor(out)


class NormalizedEncoderWrapper(nn.Module):
    """
    model.normalize_encoder(x)가 있으면 그 경로를 사용한다.
    없으면 normalizer + encoder 또는 encoder만 사용한다.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "normalize_encoder"):
            feats = self.model.normalize_encoder(x)
        elif hasattr(self.model, "normalizer") and hasattr(self.model, "encoder"):
            feats = self.model.encoder(self.model.normalizer(x))
        else:
            feats = self.model.encoder(x)

        if isinstance(feats, dict):
            if "reduction_5" in feats:
                return feats["reduction_5"]
            return first_tensor(feats)

        if isinstance(feats, (tuple, list)):
            return feats[-1]

        return feats


# ============================================================
# Measurement utilities
# ============================================================
def count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt_num(x):
    return clever_format([x], "%.3f")[0]


def profile_macs(wrapper: nn.Module, x: torch.Tensor, label: str):
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


def print_param_breakdown(model: nn.Module, title: str):
    total = sum(p.numel() for p in model.parameters())

    print("-" * 100)
    print(f"[Top-level parameter breakdown] {title}")
    print("-" * 100)

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        ratio = n / total * 100 if total > 0 else 0.0
        print(f"{name:40s} {n:15,}  {ratio:6.2f}%")

    print("-" * 100)


def print_block(title: str, params: int, macs: float):
    flops = macs * 2

    print("=" * 100)
    print(title)
    print("-" * 100)
    print(f"Params       : {params:,} ({fmt_num(params)})")
    print(f"MACs         : {macs:,.0f} ({fmt_num(macs)})")
    print(f"FLOPs(2xMAC) : {flops:,.0f} ({fmt_num(flops)})")
    print("=" * 100)


def reduction(old, new):
    if old == 0:
        return 0.0
    return (1.0 - new / old) * 100.0


def instantiate_model(cls, attempts, name: str):
    last_error = None

    for kwargs in attempts:
        try:
            print(f"[INFO] Instantiate {name} with args: {kwargs}")
            return cls(**kwargs)
        except TypeError as e:
            last_error = e

    raise RuntimeError(f"Failed to instantiate {name}. Last error: {last_error}")


# ============================================================
# Official ZoomNeXt
# ============================================================
def measure_official(args, device):
    official_root = resolve_root(args.official_root)
    print(f"\n[INFO] Official ZoomNeXt root: {official_root}")

    OfficialCls = import_project_class(
        official_root,
        "methods.zoomnext.zoomnext",
        "PvtV2B2_ZoomNeXt",
    )

    model = instantiate_model(
        OfficialCls,
        attempts=[
            {
                "pretrained": args.pretrained,
                "mid_dim": args.official_mid_dim,
            },
            {
                "pretrained": args.pretrained,
            },
            {},
        ],
        name="Official ZoomNeXt",
    ).to(device)

    model.eval()

    full_params, _ = count_params(model)
    encoder_params, _ = count_params(model.encoder)
    non_encoder_params = full_params - encoder_params

    print_param_breakdown(model, "Official ZoomNeXt")

    x_m = torch.randn(1, 3, args.h, args.w, device=device)

    full_wrapper = OfficialFullWrapper(model).to(device)
    encoder_wrapper = NormalizedEncoderWrapper(model).to(device)

    # Official full forward: image_s, image_m, image_l 모두 사용
    official_full_macs = profile_macs(
        full_wrapper,
        x_m,
        "Official full model, multi-scale",
    )

    # Correct encoder runtime MACs:
    # PVT(192) + PVT(384) + PVT(576)
    h_s, w_s = max(1, args.h // 2), max(1, args.w // 2)
    h_m, w_m = args.h, args.w
    h_l, w_l = int(args.h * 1.5), int(args.w * 1.5)

    x_s = torch.randn(1, 3, h_s, w_s, device=device)
    x_l = torch.randn(1, 3, h_l, w_l, device=device)

    enc_s_macs = profile_macs(
        encoder_wrapper,
        x_s,
        f"Official encoder image_s {h_s}x{w_s}",
    )
    enc_m_macs = profile_macs(
        encoder_wrapper,
        x_m,
        f"Official encoder image_m {h_m}x{w_m}",
    )
    enc_l_macs = profile_macs(
        encoder_wrapper,
        x_l,
        f"Official encoder image_l {h_l}x{w_l}",
    )

    encoder_runtime_macs = enc_s_macs + enc_m_macs + enc_l_macs
    non_encoder_macs = official_full_macs - encoder_runtime_macs

    print("\n" + "#" * 100)
    print("Official ZoomNeXt corrected breakdown")
    print("#" * 100)
    print(f"{'Part':42s} | {'Params':>12s} | {'MACs':>12s} | {'FLOPs':>12s}")
    print("-" * 100)
    print(
        f"{'Encoder params, stored once':42s} | "
        f"{encoder_params / 1e6:10.3f}M | "
        f"{encoder_runtime_macs / 1e9:10.3f}G | "
        f"{encoder_runtime_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'  - encoder image_s only':42s} | "
        f"{'-':>12s} | "
        f"{enc_s_macs / 1e9:10.3f}G | "
        f"{enc_s_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'  - encoder image_m only':42s} | "
        f"{'-':>12s} | "
        f"{enc_m_macs / 1e9:10.3f}G | "
        f"{enc_m_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'  - encoder image_l only':42s} | "
        f"{'-':>12s} | "
        f"{enc_l_macs / 1e9:10.3f}G | "
        f"{enc_l_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'Decoder/Fusion/etc., corrected':42s} | "
        f"{non_encoder_params / 1e6:10.3f}M | "
        f"{non_encoder_macs / 1e9:10.3f}G | "
        f"{non_encoder_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'Total':42s} | "
        f"{full_params / 1e6:10.3f}M | "
        f"{official_full_macs / 1e9:10.3f}G | "
        f"{official_full_macs * 2 / 1e9:10.3f}G"
    )
    print("#" * 100)

    result = {
        "full_params": full_params,
        "encoder_params": encoder_params,
        "non_encoder_params": non_encoder_params,
        "full_macs": official_full_macs,
        "encoder_runtime_macs": encoder_runtime_macs,
        "encoder_m_macs": enc_m_macs,
        "non_encoder_macs": non_encoder_macs,
    }

    del model, full_wrapper, encoder_wrapper, x_s, x_m, x_l
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# BRG-ZoomNeXt-Lite
# ============================================================
def measure_lite(args, device):
    lite_root = resolve_root(args.lite_root)
    print(f"\n[INFO] BRG-ZoomNeXt-Lite root: {lite_root}")

    LiteCls = import_project_class(
        lite_root,
        "methods.zoomnext.brg_lite",
        "PvtV2B2_BRG_ZoomNeXt_Lite",
    )

    model = instantiate_model(
        LiteCls,
        attempts=[
            {
                "pretrained": args.pretrained,
                "mid_dim": args.lite_mid_dim,
                "use_rfrm": not args.lite_use_rgpu,
                "use_gmffm": not args.lite_no_gmffm,
            },
            {
                "pretrained": args.pretrained,
                "mid_dim": args.lite_mid_dim,
            },
            {
                "pretrained": args.pretrained,
            },
            {},
        ],
        name="BRG-ZoomNeXt-Lite",
    ).to(device)

    model.eval()

    full_params, _ = count_params(model)
    encoder_params, _ = count_params(model.encoder)
    non_encoder_params = full_params - encoder_params

    print_param_breakdown(model, "BRG-ZoomNeXt-Lite")

    x_m = torch.randn(1, 3, args.h, args.w, device=device)

    full_wrapper = LiteFullWrapper(model).to(device)
    encoder_wrapper = NormalizedEncoderWrapper(model).to(device)

    lite_full_macs = profile_macs(
        full_wrapper,
        x_m,
        "Lite full model, image_m only",
    )

    lite_encoder_macs = profile_macs(
        encoder_wrapper,
        x_m,
        f"Lite encoder image_m {args.h}x{args.w}",
    )

    lite_non_encoder_macs = lite_full_macs - lite_encoder_macs

    print("\n" + "#" * 100)
    print("BRG-ZoomNeXt-Lite breakdown")
    print("#" * 100)
    print(f"{'Part':42s} | {'Params':>12s} | {'MACs':>12s} | {'FLOPs':>12s}")
    print("-" * 100)
    print(
        f"{'Encoder params, stored once':42s} | "
        f"{encoder_params / 1e6:10.3f}M | "
        f"{lite_encoder_macs / 1e9:10.3f}G | "
        f"{lite_encoder_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'Decoder/Fusion/etc.':42s} | "
        f"{non_encoder_params / 1e6:10.3f}M | "
        f"{lite_non_encoder_macs / 1e9:10.3f}G | "
        f"{lite_non_encoder_macs * 2 / 1e9:10.3f}G"
    )
    print(
        f"{'Total':42s} | "
        f"{full_params / 1e6:10.3f}M | "
        f"{lite_full_macs / 1e9:10.3f}G | "
        f"{lite_full_macs * 2 / 1e9:10.3f}G"
    )
    print("#" * 100)

    result = {
        "full_params": full_params,
        "encoder_params": encoder_params,
        "non_encoder_params": non_encoder_params,
        "full_macs": lite_full_macs,
        "encoder_runtime_macs": lite_encoder_macs,
        "encoder_m_macs": lite_encoder_macs,
        "non_encoder_macs": lite_non_encoder_macs,
    }

    del model, full_wrapper, encoder_wrapper, x_m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# Final comparison
# ============================================================
def print_final_comparison(official, lite):
    print("\n" + "=" * 110)
    print("FINAL COMPARISON: Official ZoomNeXt vs BRG-ZoomNeXt-Lite")
    print("=" * 110)
    print(
        f"{'Metric':40s} | "
        f"{'Official':>16s} | "
        f"{'Lite':>16s} | "
        f"{'Reduction':>12s}"
    )
    print("-" * 110)

    rows = [
        ("Total params", official["full_params"], lite["full_params"], "params"),
        ("Encoder params", official["encoder_params"], lite["encoder_params"], "params"),
        ("Non-encoder params", official["non_encoder_params"], lite["non_encoder_params"], "params"),
        ("Total MACs", official["full_macs"], lite["full_macs"], "macs"),
        ("Encoder runtime MACs", official["encoder_runtime_macs"], lite["encoder_runtime_macs"], "macs"),
        ("Non-encoder MACs", official["non_encoder_macs"], lite["non_encoder_macs"], "macs"),
        ("Total FLOPs, 2xMAC", official["full_macs"] * 2, lite["full_macs"] * 2, "flops"),
    ]

    for name, old, new, kind in rows:
        if kind == "params":
            old_s = f"{old / 1e6:.3f}M"
            new_s = f"{new / 1e6:.3f}M"
        else:
            old_s = f"{old / 1e9:.3f}G"
            new_s = f"{new / 1e9:.3f}G"

        print(
            f"{name:40s} | "
            f"{old_s:>16s} | "
            f"{new_s:>16s} | "
            f"{reduction(old, new):10.2f}%"
        )

    print("=" * 110)

    print("\n[해석 기준]")
    print("- Encoder params 감소가 거의 0%이면 정상입니다. Lite도 PVT-B2 weight 자체는 그대로 사용합니다.")
    print("- Encoder runtime MACs 감소가 크면 정상입니다. Official은 192/384/576 세 해상도 encoder를 모두 돌립니다.")
    print("- Non-encoder MACs 감소가 크면 Lite의 decoder/fusion 경량화도 실제로 의미가 있다는 뜻입니다.")
    print("- 논문/보고서에서 FLOPs를 쓸 때는 MACs를 FLOPs로 부르는 경우가 많습니다. 이 코드는 네 방식대로 FLOPs=2×MAC도 같이 출력합니다.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--official-root",
        type=str,
        default=DEFAULT_OFFICIAL_ROOT,
        help="Official ZoomNeXt-main folder or zip path",
    )
    parser.add_argument(
        "--lite-root",
        type=str,
        default=DEFAULT_LITE_ROOT,
        help="BRG-ZoomNeXt-Lite folder or zip path",
    )

    parser.add_argument("--h", type=int, default=384)
    parser.add_argument("--w", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--official-mid-dim", type=int, default=64)

    # 실제 Lite config가 mid_dim=48이면 48 유지.
    # 네가 같은 mid_dim 조건에서 비교하고 싶으면 실행 시 --lite-mid-dim 64를 넣으면 됨.
    parser.add_argument("--lite-mid-dim", type=int, default=48)

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument(
        "--lite-use-rgpu",
        action="store_true",
        help="Lite에서 RegionGuidedRefinement 대신 RGPU 계열을 쓰는 옵션이 있을 때 사용",
    )
    parser.add_argument(
        "--lite-no-gmffm",
        action="store_true",
        help="Lite에서 GMFFM을 끄고 싶을 때 사용",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Use CPU instead.")
        device = "cpu"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Official root: {args.official_root}")
    print(f"[INFO] Lite root    : {args.lite_root}")
    print(f"[INFO] Base image_m : 1x3x{args.h}x{args.w}")
    print(f"[INFO] Official multi-scale: image_s=0.5x, image_m=1.0x, image_l=1.5x")
    print(f"[INFO] Official mid_dim: {args.official_mid_dim}")
    print(f"[INFO] Lite mid_dim    : {args.lite_mid_dim}")
    print(f"[INFO] Pretrained      : {args.pretrained}")

    official_result = measure_official(args, device)
    lite_result = measure_lite(args, device)

    print_final_comparison(official_result, lite_result)


if __name__ == "__main__":
    main()