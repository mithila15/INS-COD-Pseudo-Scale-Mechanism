#!/usr/bin/env python3
"""Write reviewable experiment commands. This program never starts training."""
import argparse
import ast
import itertools
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent


def names(module, constant):
    tree = ast.parse((HERE / "fsa_study" / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(constant)


def variants(phase, generator, utilizer):
    generators = names("generators", "GENERATOR_NAMES")
    utilizers = names("utilizers", "UTILIZER_NAMES")
    if generator not in generators or utilizer not in utilizers:
        raise ValueError(f"Unknown generator/utilizer; choose {generators} / {utilizers}")
    pairs = itertools.product(generators, utilizers) if phase == "cross" else (
        ((g, utilizer) for g in generators) if phase == "generation" else ((generator, u) for u in utilizers))
    return [("single", ["--baseline"]), ("legacy_full", ["--legacy-reference"])] + [
        (f"{g}__{u}", ["--generator", g, "--utilizer", u]) for g, u in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["generation", "utilization", "cross"], default="cross")
    parser.add_argument("--generator", default="legacy", help="Fixed generator for utilization phase")
    parser.add_argument("--utilizer", default="mean", help="Fixed utilizer for generation phase")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
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
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    for field in ("evaluation_subdir", "profile_name"):
        value = getattr(args, field)
        if Path(value).is_absolute() or ".." in Path(value).parts or value in {"", "."}:
            parser.error(f"{field} must be a nonempty relative path")
    for field in ("manifest", "output", "results", "python", "backbone_weights"):
        value = getattr(args, field)
        if value and not Path(value).is_absolute():
            parser.error(f"{field} must be an absolute path")
    if args.epochs < 1 or args.batch_size < 1 or args.size < 64 or args.size % 32 or args.channels < 4:
        parser.error("invalid epochs/batch-size/size/channels")
    variants_list = variants(args.phase, args.generator, args.utilizer)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for name, flags in variants_list:
        for seed in args.seeds:
            run = Path(args.results) / f"{name}__seed{seed}"
            base = [args.python, str(HERE / "run_study.py")]
            train = base + ["train", "--manifest", args.manifest, "--output", str(run), "--seed", str(seed),
                            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                            "--size", str(args.size), "--channels", str(args.channels),
                            "--workers", str(args.workers), "--device", args.device] + flags
            train += ["--backbone-weights", args.backbone_weights] if args.backbone_weights else ["--random-init"]
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
               "results": args.results, "evaluation_subdir": args.evaluation_subdir, "profile_name": args.profile_name,
               "run_count": len(jobs), "seeds": args.seeds, "status": "commands_only_not_executed", "jobs": jobs}
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
