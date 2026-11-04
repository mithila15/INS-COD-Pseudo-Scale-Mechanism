#!/usr/bin/env python3
"""Write reviewable experiment commands. This program never starts training."""
import argparse
import ast
import itertools
import json
from pathlib import Path
import shlex
import sys

from run_study import BACKBONE_NAMES, FEATURE_MODES, build_parser
from fsa_study.conditions import variant_name, validate_scale_options, validate_distillation_options

HERE = Path(__file__).resolve().parent


def names(module, constant):
    tree = ast.parse((HERE / "fsa_study" / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(constant)


def variants(phase, generator, utilizer, baseline=False):
    generators = names("generators", "GENERATOR_NAMES")
    utilizers = names("utilizers", "UTILIZER_NAMES")
    if generator not in generators or utilizer not in utilizers + names("utilizers", "EXPERIMENTAL_UTILIZER_NAMES"):
        raise ValueError(f"Unknown generator/utilizer; choose {generators} / {utilizers}")
    if baseline and phase != "selected":
        raise ValueError("--baseline is only valid with --phase selected")
    if phase == "selected":
        return [("single", ["--baseline"])] if baseline else [
            (f"{generator}__{utilizer}", ["--generator", generator, "--utilizer", utilizer])]
    if phase in {"mechanism4", "fusion2", "distill2"}:
        use = "anchor_gated" if phase == "fusion2" else "mean"
        pairs = [(f"{g}__{use}", ["--generator", g, "--utilizer", use]) for g in ("projection", "native_conv")]
        if phase == "mechanism4":
            return [("single", ["--baseline"])] + pairs + [
                ("image_pyramid__mean", ["--input-mode", "image_pyramid", "--generator", "identity", "--utilizer", "mean"])]
        return pairs
    if phase == "priority8":
        pairs = [("native_conv", "asff"), ("native_conv", "sk"), ("native_conv", "mean"),
                 ("projection", "mean"), ("dysample", "asff"), ("dysample", "sk"), ("dysample", "mean")]
        return [("single", ["--baseline"])] + [
            (f"{g}__{u}", ["--generator", g, "--utilizer", u]) for g, u in pairs]
    pairs = itertools.product(generators, utilizers) if phase == "cross" else (
        ((g, utilizer) for g in generators) if phase == "generation" else ((generator, u) for u in utilizers))
    return [("single", ["--baseline"]), ("legacy_full", ["--legacy-reference"])] + [
        (f"{g}__{u}", ["--generator", g, "--utilizer", u]) for g, u in pairs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["selected", "generation", "utilization", "cross", "priority8", "mechanism4", "fusion2", "distill2"], default="priority8",
                        help="Default priority8: the previous eight-case shortlist; selected creates only one requested case")
    parser.add_argument("--training-protocol", choices=["holdout", "full_train"], default="holdout")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="pvt_v2_b2")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="multistage")
    parser.add_argument("--output-stride", type=int, choices=[16, 32], default=32)
    parser.add_argument("--input-mode", choices=["single", "image_pyramid"], default="single")
    parser.add_argument("--distill-teacher")
    parser.add_argument("--distill-weight", type=float, default=0.)
    parser.add_argument("--baseline", action="store_true", help="Selected phase only: run the no-pseudo control instead of the requested pair")
    parser.add_argument("--generator", default="legacy", help="Fixed generator for utilization phase")
    parser.add_argument("--utilizer", default="mean", help="Fixed utilizer for generation phase")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407],
                        help="Default: one seed3407. Multiple seeds must be requested explicitly")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New plan directory; command scripts and plan.json only")
    parser.add_argument("--results", required=True, help="Absolute NEW root for future run directories")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evaluation-subdir", default="test", help="New relative evaluation folder inside each run")
    parser.add_argument("--test-geometry", choices=["strict", "resize-prediction-to-gt"], default="strict")
    parser.add_argument("--profile-name", default="profile.json", help="New relative profile filename inside each run")
    init = parser.add_mutually_exclusive_group(required=True)
    init.add_argument("--backbone-weights")
    init.add_argument("--random-init", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    for field in ("evaluation_subdir", "profile_name"):
        value = getattr(args, field)
        if Path(value).is_absolute() or ".." in Path(value).parts or value in {"", "."}:
            parser.error(f"{field} must be a nonempty relative path")
    for field in ("manifest", "output", "results", "python", "backbone_weights", "distill_teacher"):
        value = getattr(args, field)
        if value and not Path(value).is_absolute():
            parser.error(f"{field} must be an absolute path")
    if args.epochs < 1 or args.batch_size < 1 or args.size < 64 or args.size % 32 or args.channels < 4:
        parser.error("invalid epochs/batch-size/size/channels")
    try:
        variants_list = variants(args.phase, args.generator, args.utilizer, args.baseline)
    except ValueError as error:
        parser.error(str(error))
    if args.input_mode == "image_pyramid":
        if args.phase != "selected" or args.baseline:
            parser.error("Explicit --input-mode image_pyramid requires --phase selected without --baseline")
        variants_list = [(f"image_pyramid__{args.utilizer}", ["--utilizer", args.utilizer])]
    if args.phase == "distill2" and not args.distill_teacher:
        parser.error("distill2 requires a completed teacher checkpoint and positive --distill-weight")
    if args.distill_teacher and args.training_protocol != "full_train":
        parser.error("Feature-distillation study requires full_train")
    if (args.backbone != "pvt_v2_b2" or args.feature_mode != "multistage") and any(
            "--legacy-reference" in flags for _, flags in variants_list):
        parser.error("This phase includes legacy_full, which requires PVTv2-B2 multistage. Use --phase selected or priority8 for this backbone/feature mode.")
    output = Path(args.output)
    jobs = []
    for name, flags in variants_list:
        for seed in args.seeds:
            run = Path(args.results) / f"{name}__seed{seed}"
            base = [args.python, str(HERE / "run_study.py")]
            train = base + ["train", "--manifest", args.manifest, "--output", str(run), "--seed", str(seed),
                            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                            "--size", str(args.size), "--channels", str(args.channels),
                            "--workers", str(args.workers), "--device", args.device,
                            "--backbone", args.backbone, "--feature-mode", args.feature_mode,
                            "--output-stride", str(args.output_stride), "--input-mode", args.input_mode,
                            "--training-protocol", args.training_protocol] + flags
            train += ["--backbone-weights", args.backbone_weights] if args.backbone_weights else ["--random-init"]
            if args.distill_teacher:
                train += ["--distill-teacher", args.distill_teacher]
            train += ["--distill-weight", str(args.distill_weight)]
            parsed = vars(build_parser().parse_args(train[2:]))
            try:
                validate_scale_options(**{key: parsed[key] for key in ("backbone", "feature_mode", "output_stride", "input_mode", "baseline", "legacy_reference")})
                validate_distillation_options(args.distill_teacher, args.distill_weight, parsed)
            except ValueError as error:
                parser.error(str(error))
            if name != variant_name(parsed):
                parser.error("Internal plan variant does not match its command")
            evaluate = base + ["evaluate", "--manifest", args.manifest, "--checkpoint", str(run / "best.pth"),
                               "--split", "test", "--output", str(run / args.evaluation_subdir), "--device", args.device,
                               "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                               "--test-geometry", args.test_geometry]
            profile = [args.python, str(HERE / "profile_study.py"), "--checkpoint", str(run / "best.pth"),
                       "--output", str(run / args.profile_name), "--device", args.device,
                       "--size", str(args.size), "--batch-size", "1"]
            jobs.append({"variant": name, "seed": seed, "run": str(run),
                         "train": train, "evaluate": evaluate, "profile": profile})
    payload = {"schema_version": 2, "phase": args.phase, "variant_count": len(variants_list),
               "training_protocol": args.training_protocol,
               "backbone": args.backbone, "feature_mode": args.feature_mode, "output_stride": args.output_stride,
               "results": args.results, "evaluation_subdir": args.evaluation_subdir, "profile_name": args.profile_name,
               "run_count": len(jobs), "seeds": args.seeds, "status": "commands_only_not_executed", "jobs": jobs}
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    for action in ("train", "evaluate", "profile", "run_all"):
        actions = [action] if action != "run_all" else ["train", "evaluate", "profile"]
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "# Explicit execution only. Completed stages skip; incomplete outputs are preserved and rejected."]
        lines += [shlex.join([args.python, str(HERE / "run_plan.py"), "--plan", str(output / "plan.json"),
                             "--actions", *actions, "--skip-completed"])]
        (output / f"{action}.sh").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
