#!/usr/bin/env python3
"""Explicitly execute a saved plan and refresh its descriptive leaderboards.

Importing this module and creating plans never starts a process. Only invoking
this CLI executes the selected actions; the default includes actual training.
Completed stages can be skipped, but partially written outputs are preserved
and rejected rather than retrained or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys

from fsa_study.data import sha256_file
from run_study import build_parser

HERE = Path(__file__).resolve().parent
ACTIONS = ("train", "evaluate", "profile")


def read_plan(path, *, rebind_entrypoints=False):
    plan = json.loads(Path(path).read_text())
    jobs = plan.get("jobs", [])
    if not jobs or len(jobs) != plan.get("run_count"):
        raise ValueError("Plan has no jobs or an inconsistent run_count")
    runs = [Path(job["run"]) for job in jobs]
    if any(not run.is_absolute() for run in runs) or len(set(runs)) != len(runs):
        raise ValueError("Plan must contain distinct absolute run paths")
    for job in jobs:
        for action in ACTIONS:
            command = job.get(action)
            if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
                raise ValueError(f"Invalid {action} command in plan")
    if rebind_entrypoints:
        parents = set()
        for job in jobs:
            for action in ACTIONS:
                command = job[action]
                expected_name = "profile_study.py" if action == "profile" else "run_study.py"
                if (len(command) < 3 or not Path(command[1]).is_absolute()
                        or Path(command[1]).name != expected_name):
                    raise ValueError("Entrypoint rebinding requires absolute original scripts with the expected study filenames")
                parents.add(Path(command[1]).parent.resolve())
        if len(parents) != 1:
            raise ValueError("Entrypoint rebinding requires one shared original study directory")
        plan = copy.deepcopy(plan)
        jobs = plan["jobs"]
        for job in jobs:
            for action in ACTIONS:
                job[action][1] = str(HERE / ("profile_study.py" if action == "profile" else "run_study.py"))
    results = Path(plan.get("results", runs[0].parent))
    if not results.is_absolute() or any(run.parent != results for run in runs):
        raise ValueError("Plan runs must share one absolute results directory")
    for job in jobs:
        run = Path(job["run"]).resolve()
        parsed = {}
        for action in ACTIONS:
            command = job[action]
            expected_script = HERE / ("profile_study.py" if action == "profile" else "run_study.py")
            if len(command) < 3 or not Path(command[0]).is_absolute() or Path(command[1]).resolve() != expected_script:
                raise ValueError(f"Plan {action} must invoke the intended study entrypoint")
            if action != "profile":
                parsed[action] = build_parser().parse_args(command[2:])
                if parsed[action].command != action:
                    raise ValueError(f"Plan {action} command performs a different action")
            try:
                target = Path(command[command.index("--output") + 1]).resolve()
            except (ValueError, IndexError):
                raise ValueError(f"Plan {action} command has no output") from None
            relative = plan.get("evaluation_subdir", "test") if action == "evaluate" else plan.get("profile_name", "profile.json")
            expected_target = run if action == "train" else run / relative
            if target != expected_target or not target.is_relative_to(run):
                raise ValueError(f"Plan {action} output does not match the declared run")
            if action != "train":
                try:
                    checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
                except (ValueError, IndexError):
                    raise ValueError(f"Plan {action} command has no checkpoint") from None
                if checkpoint != run / "best.pth":
                    raise ValueError(f"Plan {action} must use the declared run's best checkpoint")
        train, evaluate = parsed["train"], parsed["evaluate"]
        if train.legacy_reference and (train.backbone != "pvt_v2_b2" or train.feature_mode != "multistage"):
            raise ValueError("Legacy reference requires PVTv2-B2 multistage")
        for field in ("backbone", "feature_mode"):
            if field in plan and plan[field] != getattr(train, field):
                raise ValueError(f"Plan {field} disagrees with training command")
        variant = "single" if train.baseline else "legacy_full" if train.legacy_reference else f"{train.generator}__{train.utilizer}"
        if variant != job["variant"] or train.seed != job["seed"]:
            raise ValueError("Plan variant/seed disagrees with training command")
        if evaluate.split != "test":
            raise ValueError("Plan evaluation must use test split")
        training_manifest = getattr(evaluate, "training_manifest", None)
        if training_manifest:
            if Path(training_manifest).resolve() != Path(train.manifest).resolve():
                raise ValueError("Plan evaluation extension must reference the original training manifest")
            training_root = getattr(evaluate, "training_data_root", None)
            if train.data_root and (not training_root or Path(training_root).resolve() != Path(train.data_root).resolve()):
                raise ValueError("Plan evaluation extension must retain the original training data root")
        elif (Path(evaluate.manifest).resolve() != Path(train.manifest).resolve()
              or getattr(evaluate, "training_data_root", None)):
            raise ValueError("Plan evaluation must use the training manifest or an explicit audited test extension")
    plan["results"] = str(results)
    return plan


def completed(job, action):
    """Check the selected stage's identity before allowing a completed skip."""
    run = Path(job["run"])
    status_path, config_path = run / "status.json", run / "config.json"
    if not status_path.exists() or not config_path.exists() or not (run / "best.pth").is_file():
        return False
    status, config = json.loads(status_path.read_text()), json.loads(config_path.read_text())
    if status.get("status") != "training_complete":
        return False
    arguments = vars(build_parser().parse_args(job["train"][2:]))
    # Ignore only filesystem/loader choices that do not alter the trained model.
    ignore = {"output", "manifest", "data_root", "workers", "device", "backbone_weights"}
    def recorded_argument(key):
        # Historical checkpoints predate these explicit options. Their original
        # implementation was always PVTv2-B2 with four-stage fusion.
        defaults = {"training_protocol": "holdout", "backbone": "pvt_v2_b2", "feature_mode": "multistage"}
        return config.get(key, defaults.get(key))
    if any(recorded_argument(k) != v for k, v in arguments.items() if k not in ignore):
        raise ValueError(f"Existing training configuration disagrees with plan: {run}")
    manifest = json.loads(Path(arguments["manifest"]).read_text())
    if config.get("manifest_sha256") != manifest.get("content_sha256"):
        raise ValueError(f"Existing training manifest disagrees with plan: {run}")
    if arguments["backbone_weights"] and config.get("backbone_sha256") != sha256_file(arguments["backbone_weights"]):
        raise ValueError(f"Existing backbone initialization disagrees with plan: {run}")
    if action == "train":
        return True
    command = job[action]
    target = Path(command[command.index("--output") + 1])
    path = target / "metrics.json" if action == "evaluate" else target
    if not path.is_file():
        return False
    artifact = json.loads(path.read_text())
    if artifact.get("checkpoint_sha256") != sha256_file(run / "best.pth"):
        raise ValueError(f"Existing {action} artifact belongs to a different checkpoint: {path}")
    if action == "evaluate":
        requested = build_parser().parse_args(command[2:])
        requested_manifest = json.loads(Path(requested.manifest).read_text())
        if (artifact.get("split") != "test"
                or artifact.get("manifest_sha256") != requested_manifest.get("content_sha256")
                or artifact.get("model_config") != config.get("model_config")):
            raise ValueError(f"Existing evaluation metadata disagrees with plan: {path}")
        from fsa_study.overlap import overlap_metadata
        recorded_overlap = artifact.get("overlap_audit")
        if recorded_overlap is not None or requested_manifest.get("overlap_policy", "strict") != "strict":
            if recorded_overlap != overlap_metadata(requested_manifest):
                raise ValueError(f"Existing evaluation overlap audit disagrees with its manifest: {path}")
        if getattr(requested, "training_manifest", None):
            from fsa_study.benchmark import benchmark_coverage, validate_test_extension
            if Path(requested.training_manifest).resolve() != Path(arguments["manifest"]).resolve():
                raise ValueError("Evaluation extension references a different training manifest")
            proof = validate_test_extension(
                manifest, requested.training_data_root or manifest["data_root"],
                requested_manifest, requested.data_root or requested_manifest["data_root"], verify_files=False)
            if (artifact.get("training_manifest_sha256") != config.get("manifest_sha256")
                    or artifact.get("manifest_extension") != proof):
                raise ValueError(f"Existing evaluation has invalid test extension provenance: {path}")
            for name, expected in (("manifest.json", requested_manifest), ("training_manifest.json", manifest)):
                snapshot = target / name
                if (not snapshot.is_file()
                        or artifact.get("artifacts", {}).get("sha256", {}).get(name) != sha256_file(snapshot)
                        or json.loads(snapshot.read_text()) != expected):
                    raise ValueError(f"Existing evaluation is missing or changed its manifest snapshot: {name}")
            coverage = benchmark_coverage({name: values["count"] for name, values in artifact.get("datasets", {}).items()})
            if coverage != proof["benchmark_coverage"] or artifact.get("benchmark_coverage") != coverage:
                raise ValueError(f"Existing evaluation does not contain the complete audited benchmark: {path}")
        elif artifact.get("manifest_sha256") != config.get("manifest_sha256"):
            raise ValueError(f"Existing evaluation metadata disagrees with training manifest: {path}")
        from fsa_study.data import geometry_protocol
        requested_geometry = requested.test_geometry
        recorded_geometry = artifact.get("geometry_protocol", geometry_protocol("strict"))
        if recorded_geometry != geometry_protocol(requested_geometry):
            raise ValueError(f"Existing evaluation uses a different geometry policy; use a new evaluation subdirectory: {path}")
        if requested_geometry != "strict" and (
                artifact.get("geometry_audit", {}).get("policy") != requested_geometry
                or artifact.get("artifacts", {}).get("geometry_audit") != "geometry_audit.json"
                or "geometry_audit.json" not in artifact.get("artifacts", {}).get("sha256", {})):
            raise ValueError(f"Existing evaluation is missing its geometry audit; use a new evaluation subdirectory: {path}")
        from fsa_study.metrics import METRIC_PROTOCOL, METRIC_KEYS, protocol_details
        if (artifact.get("metric_protocol") != METRIC_PROTOCOL
                or artifact.get("metric_protocol_details") != protocol_details()
                or not artifact.get("datasets")
                or any(not set(METRIC_KEYS).issubset(values) for values in artifact["datasets"].values())):
            raise ValueError(f"Existing evaluation uses an older/different metric protocol; preserve it and use --evaluation-subdir test-v2 in a new plan: {path}")
        artifacts = artifact.get("artifacts", {})
        if not artifacts.get("per_image_metrics") or not artifacts.get("curves"):
            raise ValueError(f"Existing evaluation is missing exported metric artifacts: {path}")
        if "--no-save-predictions" not in command and not artifacts.get("predictions_saved"):
            raise ValueError(f"Existing evaluation did not save requested predictions; use a new evaluation subdirectory: {path}")
        hashes = artifacts.get("sha256", {})
        if not hashes:
            raise ValueError(f"Existing evaluation has no artifact checksums: {path}")
        for relative, digest in hashes.items():
            saved = (target / relative).resolve()
            if not saved.is_relative_to(target.resolve()) or not saved.is_file() or sha256_file(saved) != digest:
                raise ValueError(f"Existing evaluation artifact is missing or changed: {relative}")
    else:
        from fsa_study.complexity import COMPLEXITY_PROTOCOL
        if any(artifact.get("complexity", {}).get(key) != value for key, value in COMPLEXITY_PROTOCOL.items()):
            raise ValueError(f"Existing profile uses an older/different complexity protocol; preserve it and use --profile-name profile-v2.json in a new plan: {path}")
        defaults = {"size": 384, "batch_size": 1, "threads": 2, "warmup": 10,
                    "repeats": 50, "seed": 3407, "device": "cpu", "image": None}
        for key, default in defaults.items():
            flag = "--" + key.replace("_", "-")
            value = command[command.index(flag) + 1] if flag in command else default
            value = int(value) if isinstance(default, int) else value
            if artifact.get("protocol", {}).get(key) != value:
                raise ValueError(f"Existing profile disagrees with requested {flag}; use a new --profile-name: {path}")
        if "--image" in command and artifact.get("image_sha256") != sha256_file(command[command.index("--image") + 1]):
            raise ValueError(f"Existing profile input image changed: {path}")
    return True


def execute_plan(plan_path, actions=ACTIONS, report_dir=None, skip_completed=False, dry_run=False,
                 run_command=None):
    """Run reviewable command arrays (never through a shell), reporting each stage."""
    run_command = subprocess.run if run_command is None else run_command
    plan_path = Path(plan_path).expanduser().resolve(strict=True)
    plan = read_plan(plan_path)
    if plan.get("evaluation_only") and tuple(actions) != ("evaluate",):
        raise ValueError("This is an evaluation-only recovery plan; use --actions evaluate. Training and profiling are disabled.")
    report_dir = Path(report_dir).expanduser().resolve() if report_dir else plan_path.parent / "leaderboard"
    summary = [plan["jobs"][0]["train"][0], str(HERE / "summarize_study.py"),
               "--results", plan["results"], "--output", str(report_dir),
               "--plan", str(plan_path), "--evaluation-subdir", plan.get("evaluation_subdir", "test"),
               "--profile-name", plan.get("profile_name", "profile.json"),
               "--overwrite"]
    if dry_run:
        for job in plan["jobs"]:
            for action in actions:
                print(shlex.join(job[action]))
        print(shlex.join(summary))
        return 0

    def refresh():
        result = run_command(summary, check=False)
        if result.returncode:
            print("Leaderboard refresh failed; experiment artifacts are preserved.", file=sys.stderr)
        return result.returncode

    if refresh():
        return 2
    total = len(plan["jobs"])
    for index, job in enumerate(plan["jobs"], 1):
        for action in actions:
            if skip_completed and completed(job, action):
                print(f"[{index}/{total}] {job['variant']} seed={job['seed']} {action}: completed, skipped", flush=True)
                continue
            print(f"[{index}/{total}] {job['variant']} seed={job['seed']} {action}", flush=True)
            print(shlex.join(job[action]), flush=True)
            try:
                result = run_command(job[action], check=False)
            except BaseException:
                refresh()
                raise
            report_code = refresh()
            if result.returncode:
                print(f"Stage failed ({result.returncode}); partial results preserved at {job['run']}", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
            if report_code:
                return report_code
    return refresh()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Reviewed plan.json from plan_study.py")
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=list(ACTIONS))
    parser.add_argument("--report-dir", help="Derived leaderboard files refreshed here; raw outputs never overwritten")
    parser.add_argument("--skip-completed", action="store_true", help="Skip verified completed stages only; no partial-training resume")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; no training, inference or output writes")
    args = parser.parse_args(argv)
    if len(set(args.actions)) != len(args.actions):
        parser.error("actions must be unique")
    try:
        return execute_plan(args.plan, args.actions, args.report_dir, args.skip_completed, args.dry_run)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
