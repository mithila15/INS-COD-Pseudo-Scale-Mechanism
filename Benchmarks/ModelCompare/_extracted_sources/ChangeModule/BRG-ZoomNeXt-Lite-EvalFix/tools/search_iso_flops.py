#!/usr/bin/env python3
"""Search a BRG-Lite input size whose measured compute is close to original ZoomNeXt.

Example:
  python tools/search_iso_flops.py \
    --target-model PvtV2B2_ZoomNeXt \
    --student-model PvtV2B2_BRG_ZoomNeXt_Lite \
    --target-size 384 \
    --candidate-sizes 384 448 512 576 608 640 672 704
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.measure_flops import measure  # noqa: E402


def ns_from_args(args: argparse.Namespace, model_name: str, size: int, is_student: bool) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.model_name = model_name
    ns.height = size
    ns.width = size
    ns.device = args.device
    ns.tool = args.tool
    ns.pretrained = False
    ns.original_mid_dim = args.original_mid_dim
    ns.siu_groups = args.siu_groups
    ns.original_hmu_groups = args.original_hmu_groups
    ns.mid_dim = args.mid_dim
    ns.hmu_groups = args.hmu_groups
    ns.use_rgpu_g2 = args.use_rgpu_g2
    ns.no_gmffm = args.no_gmffm
    ns.no_compensation = args.no_compensation
    ns.residual_refine_weight = args.residual_refine_weight
    ns.json = True
    return ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="PvtV2B2_ZoomNeXt")
    parser.add_argument("--student-model", default="PvtV2B2_BRG_ZoomNeXt_Lite")
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--candidate-sizes", nargs="+", type=int, default=[384, 448, 512, 576, 608, 640, 672, 704])
    parser.add_argument("--tool", choices=["fvcore", "thop", "torchprofiler"], default="fvcore")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-mid-dim", type=int, default=64)
    parser.add_argument("--siu-groups", type=int, default=4)
    parser.add_argument("--original-hmu-groups", type=int, default=6)
    parser.add_argument("--mid-dim", type=int, default=48)
    parser.add_argument("--hmu-groups", type=int, default=2)
    parser.add_argument("--use-rgpu-g2", action="store_true")
    parser.add_argument("--no-gmffm", action="store_true")
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--residual-refine-weight", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    target = measure(ns_from_args(args, args.target_model, args.target_size, is_student=False))
    rows = []
    for size in args.candidate_sizes:
        student = measure(ns_from_args(args, args.student_model, size, is_student=True))
        ratio = student["ops"] / target["ops"] if target["ops"] else 0.0
        rows.append({"size": size, "student_ops_G": student["ops_G"], "ratio_to_target": ratio, "abs_gap": abs(ratio - 1.0), "params_M": student["params_M"]})
    rows.sort(key=lambda r: r["abs_gap"])
    out = {"target": target, "best": rows[0], "candidates_sorted": rows}

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Target: {args.target_model} @ {args.target_size}x{args.target_size}")
        print(f"  {target['unit_name']}: {target['ops_G']:.3f} G, params: {target['params_M']:.3f} M")
        print("\nStudent candidates:")
        for r in rows:
            print(f"  {args.student_model} @ {r['size']}x{r['size']}: {r['student_ops_G']:.3f} G, ratio={r['ratio_to_target']:.3f}, params={r['params_M']:.3f} M")
        print("\nBest iso-FLOPs candidate:")
        print(f"  input_size={rows[0]['size']}  ratio={rows[0]['ratio_to_target']:.3f}")
        print("\nUse the matching config or create one with train/test shape set to this input_size.")


if __name__ == "__main__":
    main()
