#!/usr/bin/env python3
"""Inference-only latency, memory, MACs/FLOPs, parameters and feature diagnostics."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import time

from run_study import BACKBONE_NAMES, FEATURE_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="legacy")
    parser.add_argument("--utilizer", default="mean")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="pvt_v2_b2",
                        help="Random-initialization profiling only; checkpoints retain their saved backbone")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="multistage",
                        help="Random-initialization profiling only; checkpoints retain their saved endpoint mode")
    parser.add_argument("--output-stride", type=int, choices=[16, 32], default=32)
    parser.add_argument("--input-mode", choices=["single", "image_pyramid"], default="single")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--baseline", action="store_true")
    controls.add_argument("--legacy-reference", action="store_true")
    parser.add_argument("--checkpoint", help="Strict study checkpoint; architecture comes from checkpoint")
    parser.add_argument("--image", help="Optional local RGB image; otherwise synthetic input, explicitly labeled")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or min(args.batch_size, args.threads, args.repeats) < 1 or args.warmup < 0:
        parser.error("invalid size, batch, threads or repeats")
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        parser.error("output already exists")
    import torch
    from torch.nn import functional as F
    from fsa_study.model import StudyModel
    from fsa_study.data import sha256_file
    from fsa_study.complexity import COMPLEXITY_PROTOCOL, measure_complexity
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        from fsa_study.engine import validate_checkpoint, checkpoint_model_kwargs
        validate_checkpoint(checkpoint, {"content_sha256": checkpoint.get("manifest_sha256")})
        model = StudyModel(**checkpoint_model_kwargs(checkpoint["model_config"], StudyModel))
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model = StudyModel(generator=args.generator, utilizer=args.utilizer, channels=args.channels,
                           seed=args.seed, baseline=args.baseline, legacy_reference=args.legacy_reference,
                           backbone=args.backbone, feature_mode=args.feature_mode,
                           output_stride=args.output_stride, input_mode=args.input_mode)
    model.eval().to(args.device)
    if args.image:
        from PIL import Image
        import numpy as np
        with Image.open(args.image) as handle:
            array = np.array(handle.convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR))
        x = torch.from_numpy(array.copy()).permute(2, 0, 1).float()[None] / 255
        x = x.repeat(args.batch_size, 1, 1, 1).to(args.device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        x = torch.rand(args.batch_size, 3, args.size, args.size, generator=generator).to(args.device)
    cuda = torch.device(args.device).type == "cuda"
    def sync():
        if cuda:
            torch.cuda.synchronize(args.device)
    durations = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
            memory_before = torch.cuda.memory_allocated(args.device)
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                  "peak_increment_bytes": torch.cuda.max_memory_allocated(args.device) - memory_before} if cuda else None
        # Collect diagnostics in a separate forward so hooks do not contaminate latency.
        diagnostics, handles = [], []
        def branch_hook(level):
            def hook(module, inputs, bank):
                main = bank[0].float()
                row = {"level": level, "mean_rms": float(main.square().mean().sqrt()), "branches": []}
                for branch in bank[1:]:
                    b = branch.float()
                    row["branches"].append({"rms": float(b.square().mean().sqrt()),
                        "relative_difference_rms": float((b-main).square().mean().sqrt() / main.square().mean().sqrt().clamp_min(1e-8)),
                        "cosine_channel_mean": float(F.cosine_similarity(main, b, dim=1).mean())})
                diagnostics.append(row)
            return hook
        if hasattr(model, "generators"):
            handles += [m.register_forward_hook(branch_hook(level))
                        for level, m in zip(model.feature_levels, model.generators)]
        elif model.input_mode == "image_pyramid":
            for level, module in zip(model.feature_levels, model.utilizers):
                record = branch_hook(level)
                handles.append(module.register_forward_pre_hook(
                    lambda module, inputs, record=record: record(module, inputs, inputs[0])))
        def utilization_hook(level):
            def hook(module, inputs, fused):
                bank = inputs[0]
                row = next(row for row in diagnostics if row["level"] == level)
                row["fused_rms"] = float(fused.float().square().mean().sqrt())
                row["fused_main_cosine"] = float(F.cosine_similarity(fused.float(), bank[0].float(), dim=1).mean())
                if hasattr(module, "weights"):
                    weights = module.weights(bank).float()
                    row["mean_branch_weights"] = weights.mean(dim=(0, 2, 3, 4)).tolist()
                    row["weight_min"] = float(weights.min())
            return hook
        if hasattr(model, "utilizers"):
            handles += [m.register_forward_hook(utilization_hook(level))
                        for level, m in zip(model.feature_levels, model.utilizers)]
        prediction = model(x)["logits"]
        for handle in handles:
            handle.remove()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite model output")
    # Counting adds dispatch overhead and is deliberately outside the timing,
    # peak-memory measurement and feature-diagnostic hooks above.
    complexity = measure_complexity(model, x)
    result = {"schema_version": 1, "metadata": model.metadata, "parameters": sum(p.numel() for p in model.parameters()),
              "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
              "latency_ms": {"median": statistics.median(durations), "mean": statistics.mean(durations),
                             "stdev": statistics.stdev(durations) if len(durations)>1 else 0, "samples": durations},
              "images_per_second": args.batch_size * 1000 / statistics.mean(durations),
              "cuda_memory": memory, "diagnostics": diagnostics, "complexity": complexity,
              "protocol": {**vars(args), "dtype": "float32", "inference_mode": True,
                           "backbone": model.config.get("backbone", "pvt_v2_b2"),
                           "feature_mode": model.config.get("feature_mode", "multistage"),
                           "output_stride": model.config.get("output_stride", 32),
                           "input_mode": model.config.get("input_mode", "single"),
                           "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
                           "input": "local_image" if args.image else "synthetic_uniform_RGB",
                           "weights": "study_checkpoint" if args.checkpoint else "random_initialization",
                           "complexity_protocol": COMPLEXITY_PROTOCOL,
                           "interpretation": "Execution diagnostics only; random input/weights cannot establish COD quality or learned branch utility"},
              "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
              "image_sha256": sha256_file(args.image) if args.image else None,
              "environment": {"torch": str(torch.__version__), "python": platform.python_version(),
                              "device": torch.cuda.get_device_name(args.device) if cuda else platform.processor()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target), "parameters": result["parameters"],
                      "median_ms": result["latency_ms"]["median"],
                      "macs_per_image": complexity["macs_per_image"],
                      "flops_per_image": complexity["flops_per_image"],
                      "complexity_unsupported_ops": complexity["coverage"]["unsupported_operators"],
                      "training": "not_run"}))


if __name__ == "__main__":
    main()
