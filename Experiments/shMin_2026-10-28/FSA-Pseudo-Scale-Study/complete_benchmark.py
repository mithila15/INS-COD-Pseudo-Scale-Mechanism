#!/usr/bin/env python3
"""Prepare evaluation of all four COD benchmarks using completed checkpoints.

Preparation reads local data/checkpoints and writes a fresh plan only. It never
downloads data, starts subprocesses, trains, profiles, or runs inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import tempfile

from fsa_study.benchmark import extend_test_manifest, validate_test_extension
from fsa_study.data import audit_geometry, load_manifest, sha256_file
from recover_evaluation import POLICY, TRAINING_FILES, _absolute, _put
from run_plan import HERE, completed, read_plan
from run_study import build_parser


def _remove(command, flag):
    while flag in command:
        index = command.index(flag)
        del command[index:index + 2]


def complete_benchmark_plan(plan_path, output, evaluation_subdir="test-benchmark-v1",
                            cod10k_images=None, cod10k_masks=None, rebind_entrypoints=False,
                            acknowledge_cod10k=False):
    """Generate an evaluation-only plan; every original training run must exist."""
    source = _absolute(plan_path, "--plan")
    if Path(output).expanduser().is_symlink():
        raise FileExistsError(f"Benchmark plan output is an existing symlink: {output}")
    destination = _absolute(output, "--output")
    relative = Path(evaluation_subdir)
    if (not evaluation_subdir.strip() or relative.is_absolute() or not relative.parts
            or ".." in relative.parts or relative.parts[0] in TRAINING_FILES):
        raise ValueError("--evaluation-subdir must be a safe nonempty relative directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Benchmark plan output already exists; choose a new folder: {destination}")
    source_hash = sha256_file(source)
    original = read_plan(source, rebind_entrypoints=rebind_entrypoints)
    old_relative = Path(original.get("evaluation_subdir", "test"))
    profile = Path(original.get("profile_name", "profile.json"))
    for existing, label in ((old_relative, "original evaluation"), (profile, "profile output")):
        if relative == existing or relative in existing.parents or existing in relative.parents:
            raise ValueError(f"New evaluation directory must not overlap the {label}")

    parser = build_parser()
    manifests = {}
    states = []
    first_manifest = first_root = first_path = None
    for job in original["jobs"]:
        run = Path(job["run"]).resolve()
        if destination.is_relative_to(run) or run.is_relative_to(destination):
            raise ValueError("Benchmark plan output must be outside every training run")
        target = run / relative
        if not target.resolve().is_relative_to(run):
            raise ValueError(f"Evaluation directory escapes the run: {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"New evaluation output already exists; choose another subdirectory: {target}")
        if not completed(job, "train"):
            raise ValueError(f"Every run must have completed training; unfinished run is preserved: {run}")
        args = parser.parse_args(job["train"][2:])
        manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
        root_override = str(Path(args.data_root).expanduser().resolve()) if args.data_root else None
        key = (str(manifest_path), root_override)
        if key not in manifests:
            manifests[key] = load_manifest(manifest_path, data_root=root_override)
        manifest, root = manifests[key]
        if first_manifest is None:
            first_manifest, first_root, first_path = manifest, root, manifest_path
        elif manifest["content_sha256"] != first_manifest["content_sha256"] or root != first_root:
            raise ValueError("All runs must share the same original training manifest content and resolved data root")
        states.append({"variant": job["variant"], "seed": job["seed"], "run": str(run),
                       "state": "completed_training_reused", "checkpoint_sha256": sha256_file(run / "best.pth")})

    plan = copy.deepcopy(original)
    plan["evaluation_subdir"] = relative.as_posix()
    plan["evaluation_only"] = True
    plan["status"] = "commands_only_not_executed"
    manifest_path = destination / "manifest.json"
    # All input audits and command checks finish before the destination is made.
    with tempfile.TemporaryDirectory(prefix="fsa-benchmark-preflight-") as temporary:
        temporary = Path(temporary)
        evaluation_manifest = extend_test_manifest(
            first_path, temporary / "manifest.json", cod10k_images=cod10k_images,
            cod10k_masks=cod10k_masks, data_root=first_root, acknowledge_cod10k=acknowledge_cod10k)
        evaluation_root = Path(evaluation_manifest["data_root"])
        proof = validate_test_extension(first_manifest, first_root, evaluation_manifest, evaluation_root, verify_files=False)
        geometry = audit_geometry(evaluation_manifest, evaluation_root, split="test", policy=POLICY)
        for job in plan["jobs"]:
            train = parser.parse_args(job["train"][2:])
            command = job["evaluate"]
            _put(command, "--manifest", manifest_path)
            _put(command, "--training-manifest", Path(train.manifest).expanduser().resolve())
            _remove(command, "--data-root")
            _remove(command, "--training-data-root")
            if train.data_root:
                _put(command, "--training-data-root", Path(train.data_root).expanduser().resolve())
            _put(command, "--output", Path(job["run"]) / relative)
            _put(command, "--test-geometry", POLICY)
        plan["benchmark_repair"] = {
            "schema_version": 1, "source_plan": str(source), "source_plan_sha256": source_hash,
            "original_evaluation_subdir": original.get("evaluation_subdir", "test"),
            "manifest_extension": proof, "geometry_audit": geometry,
            "training_states": states,
            "execution": "not_started; explicit evaluation only; no training or profiling",
            "entrypoint_rebinding": {"enabled": bool(rebind_entrypoints),
                                      "current_study_directory": str(HERE),
                                      "policy": "only script paths change; original Python, arguments, checkpoints and outputs are preserved"},
        }
        contents = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        candidate = temporary / "plan.json"
        candidate.write_text(contents, encoding="utf-8")
        read_plan(candidate)
        plan_file = destination / "plan.json"
        runner = [plan["jobs"][0]["train"][0], str(HERE / "run_plan.py"), "--plan", str(plan_file),
                  "--actions", "evaluate", "--skip-completed"]
        script = "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(runner) + "\n"
        if sha256_file(source) != source_hash:
            raise ValueError("Source plan changed during preparation; no output written")
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "manifest.json").write_bytes((temporary / "manifest.json").read_bytes())
        (destination / "source_plan.json").write_bytes(source.read_bytes())
        (destination / "geometry_audit.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
        plan_file.write_text(contents, encoding="utf-8")
        (destination / "run_evaluation.sh").write_text(script, encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Absolute path to the original completed plan.json")
    parser.add_argument("--output", required=True, help="New absolute folder for evaluation manifest/plan; never overwritten")
    parser.add_argument("--evaluation-subdir", default="test-benchmark-v1", help="New relative evaluation directory in each existing run")
    parser.add_argument("--cod10k-images", help="COD10K test image directory; omitted only for unique recognized layouts under the existing data root")
    parser.add_argument("--cod10k-masks", help="COD10K test object-GT directory, paired with --cod10k-images")
    parser.add_argument("--rebind-entrypoints", action="store_true",
                        help="Explicitly rebind recognized original study script paths to this isolated code directory; keep the original Python and all arguments")
    parser.add_argument("--acknowledge-cod10k-overlap", action="store_true",
                        help="Explicitly acknowledge only the seven observed COD10K duplicate ID/hash pairs; preserve every record and report overlap, never label the split clean")
    args = parser.parse_args(argv)
    try:
        plan = complete_benchmark_plan(args.plan, args.output, args.evaluation_subdir, args.cod10k_images, args.cod10k_masks,
                                       rebind_entrypoints=args.rebind_entrypoints,
                                       acknowledge_cod10k=args.acknowledge_cod10k_overlap)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    summary = {"plan": str(Path(args.output).expanduser().resolve() / "plan.json"),
               "status": "commands_only_not_executed", "evaluation_only": True,
               "evaluation_subdir": plan["evaluation_subdir"],
               "benchmark_coverage": plan["benchmark_repair"]["manifest_extension"]["benchmark_coverage"],
               "training_states": plan["benchmark_repair"]["training_states"],
               "geometry_audit": {key: plan["benchmark_repair"]["geometry_audit"][key]
                                  for key in ("checked_pairs", "dataset_counts", "mismatch_count")}}
    proof = plan["benchmark_repair"]["manifest_extension"]
    if "overlap_audit" in proof:
        summary["overlap_audit"] = proof["overlap_audit"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
