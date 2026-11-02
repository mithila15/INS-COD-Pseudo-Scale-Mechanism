#!/usr/bin/env python3
"""Prepare a new evaluation plan while preserving checkpoints and partial outputs.

This command audits local data and writes commands only. It never starts training,
inference, profiling, or subprocesses. Invoke the generated run_all.sh explicitly
to skip verified completed training, evaluate, and continue the remaining jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import tempfile

from fsa_study.data import audit_geometry, load_manifest, sha256_file
from run_plan import HERE, completed, read_plan
from run_study import build_parser

POLICY = "resize-prediction-to-gt"
TRAINING_FILES = {
    "best.pth", "last.pth", "last.tmp.pth", "config.json", "manifest.json",
    "status.json", "history.jsonl",
}


def _put(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def _absolute(path, name):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    return path.resolve()


def recover_plan(plan_path, output, evaluation_subdir="test-geometry-v1"):
    """Validate data and preserve original job identities in a fresh plan folder."""
    source = _absolute(plan_path, "--plan")
    destination = _absolute(output, "--output")
    relative = Path(evaluation_subdir)
    if (not evaluation_subdir.strip() or relative.is_absolute()
            or not relative.parts or any(part == ".." for part in relative.parts)
            or relative.parts[0] in TRAINING_FILES):
        raise ValueError("--evaluation-subdir must be a safe nonempty relative directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Recovery plan output already exists; choose a new folder: {destination}")

    original = read_plan(source)
    old_relative = Path(original.get("evaluation_subdir", "test"))
    if relative == old_relative or relative in old_relative.parents or old_relative in relative.parents:
        raise ValueError("Choose a separate --evaluation-subdir to preserve the original evaluation")
    profile = Path(original.get("profile_name", "profile.json"))
    if relative == profile or relative in profile.parents or profile in relative.parents:
        raise ValueError("Evaluation directory must not overlap the profile output")
    plan = copy.deepcopy(original)
    plan["evaluation_subdir"] = relative.as_posix()
    plan["status"] = "commands_only_not_executed"
    audits = {}
    states = []
    parser = build_parser()

    for job in plan["jobs"]:
        run = Path(job["run"]).resolve()
        if destination.is_relative_to(run):
            raise ValueError(f"Recovery plan output must be outside every training run: {destination}")
        target = run / relative
        if not target.resolve().is_relative_to(run):
            raise ValueError(f"Evaluation directory escapes the run: {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"New evaluation output already exists; choose another subdirectory: {target}")
        if run.exists() or run.is_symlink():
            if not completed(job, "train"):
                raise ValueError(f"Incomplete training output is preserved; recovery cannot resume it: {run}")
            state = "completed_training_reused"
        else:
            state = "training_not_started"
        states.append({"variant": job["variant"], "seed": job["seed"], "run": str(run), "state": state})

        command = job["evaluate"]
        _put(command, "--output", target)
        _put(command, "--test-geometry", POLICY)
        args = parser.parse_args(command[2:])
        manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
        data_root = str(Path(args.data_root).expanduser().resolve()) if args.data_root else None
        key = (str(manifest_path), data_root)
        if key not in audits:
            manifest, root = load_manifest(manifest_path, verify_files=True, data_root=data_root)
            audit = audit_geometry(manifest, root, split="test", policy=POLICY)
            audits[key] = {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest["content_sha256"],
                "data_root": str(root),
                "test_datasets_selected": sorted({
                    record["dataset"] for record in manifest["records"] if record["split"] == "test"
                }),
                "geometry_audit": audit,
            }

    plan["recovery"] = {
        "schema_version": 1,
        "source_plan": str(source),
        "source_plan_sha256": sha256_file(source),
        "test_geometry": POLICY,
        "original_evaluation_subdir": original.get("evaluation_subdir", "test"),
        "training_states": states,
        "data_audits": list(audits.values()),
        "execution": "not_started",
    }
    contents = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    # Validate the exact new commands before creating the requested output tree.
    with tempfile.TemporaryDirectory(prefix="fsa-recovery-preflight-") as temporary:
        candidate = Path(temporary) / "plan.json"
        candidate.write_text(contents, encoding="utf-8")
        read_plan(candidate)

    plan_file = destination / "plan.json"
    runner = [plan["jobs"][0]["train"][0], str(HERE / "run_plan.py"),
              "--plan", str(plan_file), "--actions", "train", "evaluate", "profile", "--skip-completed"]
    script = "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(runner) + "\n"
    destination.mkdir(parents=True, exist_ok=False)
    plan_file.write_text(contents, encoding="utf-8")
    (destination / "run_all.sh").write_text(script, encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Absolute path to the original plan.json")
    parser.add_argument("--output", required=True, help="New absolute plan folder; existing folders are rejected")
    parser.add_argument("--evaluation-subdir", default="test-geometry-v1",
                        help="New relative directory inside each existing/planned run")
    args = parser.parse_args(argv)
    try:
        plan = recover_plan(args.plan, args.output, args.evaluation_subdir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({
        "plan": str(Path(args.output).expanduser().resolve() / "plan.json"),
        "status": "commands_only_not_executed",
        "evaluation_subdir": plan["evaluation_subdir"],
        "training_states": plan["recovery"]["training_states"],
        "data_audits": [{
            "manifest": audit["manifest"],
            "test_datasets_selected": audit["test_datasets_selected"],
            "checked_pairs": audit["geometry_audit"]["checked_pairs"],
            "mismatch_count": audit["geometry_audit"]["mismatch_count"],
        } for audit in plan["recovery"]["data_audits"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
