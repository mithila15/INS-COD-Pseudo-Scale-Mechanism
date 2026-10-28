from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from flops_utils import format_result, measure_model


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Find a BRG-Lite setting whose measured FLOPs are close to original ZoomNeXt.")
    p.add_argument("--target-model", default="PvtV2B2_ZoomNeXt")
    p.add_argument("--target-input-mode", default="original", choices=["original", "lite", "auto"])
    p.add_argument("--target-size", type=int, default=384)
    p.add_argument("--candidate-model", default="PvtV2B2_BRG_ZoomNeXt_Lite")
    p.add_argument("--candidate-input-mode", default="lite", choices=["original", "lite", "auto"])
    p.add_argument("--candidate-sizes", default="384,416,448,480,512,544,576,608,640,672,704,736,768")
    p.add_argument("--mid-dims", default="48", help="comma separated, e.g. 48,64")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hmu-groups", type=int, default=2)
    p.add_argument("--use-rfrm", type=int, choices=[0, 1], default=1)
    p.add_argument("--use-gmffm", type=int, choices=[0, 1], default=1)
    p.add_argument("--save-csv", default="outputs_flops/iso_flops_scan.csv")
    p.add_argument("--save-json", default="outputs_flops/iso_flops_best.json")
    return p.parse_args()


def main():
    args = parse_args()
    print("[1/2] Measuring target/original model...")
    target = measure_model(
        args.target_model,
        input_mode=args.target_input_mode,
        image_size=args.target_size,
        batch_size=args.batch_size,
        device=args.device,
        pretrained=False,
    )
    print(format_result(target))
    target_g = float(target["fvcore_GFLOPs"])

    print("\n[2/2] Scanning candidate settings...")
    rows = []
    best = None
    for mid_dim in parse_int_list(args.mid_dims):
        for size in parse_int_list(args.candidate_sizes):
            try:
                res = measure_model(
                    args.candidate_model,
                    input_mode=args.candidate_input_mode,
                    image_size=size,
                    batch_size=args.batch_size,
                    device=args.device,
                    pretrained=False,
                    mid_dim=mid_dim,
                    hmu_groups=args.hmu_groups,
                    use_rfrm=bool(args.use_rfrm),
                    use_gmffm=bool(args.use_gmffm),
                )
            except RuntimeError as exc:
                # Usually OOM on GPU. Keep the scan robust.
                print(f"skip size={size}, mid_dim={mid_dim}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            cand_g = float(res["fvcore_GFLOPs"])
            diff = abs(cand_g - target_g)
            ratio = cand_g / target_g if target_g > 0 else 0.0
            row = {
                "target_model": args.target_model,
                "target_size": args.target_size,
                "target_GFLOPs": target_g,
                "candidate_model": args.candidate_model,
                "candidate_size": size,
                "candidate_mid_dim": mid_dim,
                "candidate_GFLOPs": cand_g,
                "ratio_to_target": ratio,
                "abs_diff_GFLOPs": diff,
                "candidate_params_M": float(res["params_M"]),
            }
            rows.append(row)
            print(f"size={size:4d}, mid_dim={mid_dim:3d} -> {cand_g:8.3f}G, ratio={ratio:6.3f}, diff={diff:8.3f}G")
            if best is None or diff < best["abs_diff_GFLOPs"]:
                best = row

    if not rows:
        raise RuntimeError("No candidate setting was successfully measured.")

    print("\nBest iso-FLOPs candidate:")
    print(json.dumps(best, indent=2))

    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps({"target": target, "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
