#!/usr/bin/env python3
"""Prepare fresh full-training runs from a saved plan; never execute experiments.

The complete four-test manifest is verified before promoting its original
validation records into training. Original commands, checkpoints and results are
preserved. Run the generated scripts explicitly to train the NEW models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shlex
import tempfile

from fsa_study.benchmark import benchmark_coverage, validate_test_extension
from fsa_study.data import audit_geometry, manifest_digest, sha256_file
from fsa_study.training_protocol import prepare_full_training_manifest, training_metadata
from run_plan import HERE, read_plan
from run_study import build_parser

PHASE = "priority8"
EVALUATION_SUBDIR = "test-four-benchmarks"
GEOMETRY_POLICY = "resize-prediction-to-gt"
CHANGED_TRAIN_SETTINGS = {"manifest", "output", "data_root", "training_protocol"}


def _absolute(value, name):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    return path.resolve()


def _remove(command, flag):
    """Remove all spellings of a value-taking flag before inserting its value."""
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == flag:
            if index + 1 >= len(command):
                raise ValueError(f"Command has no value for {flag}")
            index += 2
        elif token.startswith(flag + "="):
            index += 1
        else:
            result.append(token)
            index += 1
    command[:] = result


def _put(command, flag, value):
    _remove(command, flag)
    command.extend([flag, str(value)])


def _source_manifests(original, complete, complete_root):
    """Verify that the complete benchmark really extends each original input."""
    parser = build_parser()
    verified = {}
    for job in original["jobs"]:
        args = parser.parse_args(job["train"][2:])
        source = _absolute(args.manifest, "original training manifest")
        key = (str(source), args.data_root)
        if key in verified:
            continue
        payload = json.loads(source.read_text())
        if payload.get("content_sha256") != manifest_digest(payload):
            raise ValueError(f"Original training manifest checksum mismatch: {source}")
        if payload["content_sha256"] == complete["content_sha256"]:
            proof = None
        else:
            provenance = complete.get("test_extension", {})
            root = args.data_root or provenance.get("training_data_root") or payload["data_root"]
            proof = validate_test_extension(payload, root, complete, complete_root, verify_files=False)
        verified[key] = {"manifest": str(source), "manifest_sha256": payload["content_sha256"],
                         "manifest_file_sha256": sha256_file(source), "extension_proof": proof}
    return list(verified.values())


def prepare_full_training_plan(plan_path, manifest_path, output, data_root=None, *, rebind_entrypoints=False):
    """Audit and write a new manifest/plan tree without loading any model."""
    source = _absolute(plan_path, "--plan")
    source_manifest = _absolute(manifest_path, "--manifest")
    if Path(output).expanduser().is_symlink():
        raise FileExistsError(f"Full-training output is an existing symlink: {output}")
    destination = _absolute(output, "--output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Full-training output already exists; choose a new root: {destination}")
    original = read_plan(source, rebind_entrypoints=rebind_entrypoints)
    for job in original["jobs"]:
        old_run = Path(job["run"]).resolve()
        if destination.is_relative_to(old_run) or old_run.is_relative_to(destination):
            raise ValueError(f"New output must not overlap an original run: {old_run}")
    old_results = Path(original["results"]).resolve()
    if destination.is_relative_to(old_results) or old_results.is_relative_to(destination):
        raise ValueError("New output must be separate from the original results directory")
    profile = Path(original.get("profile_name", "profile.json"))
    if (profile.is_absolute() or ".." in profile.parts or profile == Path(".")
            or profile.parts[0] in {"best.pth", "last.pth", "config.json", "status.json", "manifest.json", "history.jsonl"}
            or profile == Path(EVALUATION_SUBDIR) or Path(EVALUATION_SUBDIR) in profile.parents):
        raise ValueError("Profile output must be a safe file outside training artifacts and the evaluation directory")

    source_hash = sha256_file(source)
    manifest_file_hash = sha256_file(source_manifest)
    complete = json.loads(source_manifest.read_text())
    new_manifest_path = destination / "manifest.json"
    new_results = destination / "runs" / PHASE
    plan_file = destination / "plans" / PHASE / "plan.json"
    parser = build_parser()
    plan = copy.deepcopy(original)
    # Historical recovery states belong to the source snapshot, not new models.
    for key in ("evaluation_only", "recovery", "benchmark_completion", "benchmark_repair", "relocation", "full_training"):
        plan.pop(key, None)
    plan.update({"phase": PHASE, "results": str(new_results), "training_protocol": "full_train",
                 "evaluation_subdir": EVALUATION_SUBDIR, "profile_name": profile.as_posix(),
                 "status": "commands_only_not_executed"})
    changes = []
    names = set()
    initializations = {}
    for job in plan["jobs"]:
        variant, seed = job["variant"], job["seed"]
        if not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", variant):
            raise ValueError(f"Unsafe experiment variant name: {variant!r}")
        name = f"{variant}__seed{seed}"
        if name in names:
            raise ValueError(f"Repeated variant/seed in original plan: {name}")
        names.add(name)
        old_run = job["run"]
        new_run = new_results / name
        before = vars(parser.parse_args(job["train"][2:]))
        if before.get("backbone_weights"):
            weights = _absolute(before["backbone_weights"], "original backbone weights")
            if not weights.is_file():
                raise FileNotFoundError(f"Original backbone weights are missing: {weights}")
            if str(weights) not in initializations:
                initializations[str(weights)] = sha256_file(weights)
        job["run"] = str(new_run)
        for action in ("train", "evaluate"):
            _put(job[action], "--manifest", new_manifest_path)
            _remove(job[action], "--data-root")
        _put(job["train"], "--output", new_run)
        _put(job["train"], "--training-protocol", "full_train")
        for flag in ("--training-manifest", "--training-data-root"):
            _remove(job["evaluate"], flag)
        _put(job["evaluate"], "--output", new_run / EVALUATION_SUBDIR)
        _put(job["evaluate"], "--test-geometry", GEOMETRY_POLICY)
        _put(job["profile"], "--output", new_run / profile)
        for action in ("evaluate", "profile"):
            _put(job[action], "--checkpoint", new_run / "best.pth")
        after = vars(parser.parse_args(job["train"][2:]))
        preserved = {key: value for key, value in before.items() if key not in CHANGED_TRAIN_SETTINGS}
        if preserved != {key: value for key, value in after.items() if key not in CHANGED_TRAIN_SETTINGS}:
            raise ValueError(f"Unexpected training setting change: {variant}, seed {seed}")
        changes.append({"variant": variant, "seed": seed, "original_run": old_run, "new_run": str(new_run),
                        "preserved_training_settings": preserved,
                        "changed_training_settings": {key: {"before": before.get(key), "after": after.get(key)}
                                                       for key in sorted(CHANGED_TRAIN_SETTINGS)}})

    with tempfile.TemporaryDirectory(prefix="fsa-full-training-preflight-") as temporary:
        staging = Path(temporary)
        manifest = prepare_full_training_manifest(source_manifest, staging / "manifest.json", data_root=data_root)
        complete_root = manifest["full_training_source"]["source_data_root"]
        source_inputs = _source_manifests(original, complete, complete_root)
        metadata = training_metadata(manifest, "full_train")
        coverage = benchmark_coverage(dict(Counter(record["dataset"] for record in manifest["records"]
                                                   if record["split"] == "test")))
        geometry = audit_geometry(manifest, manifest["data_root"], "train", "strict")
        test_geometry = audit_geometry(manifest, manifest["data_root"], "test", GEOMETRY_POLICY)
        plan["full_training"] = {
            "schema_version": 1, "source_plan": str(source), "source_plan_sha256": source_hash,
            "entrypoints_rebound": bool(rebind_entrypoints),
            "source_manifest": str(source_manifest), "source_manifest_file_sha256": manifest_file_hash,
            "source_manifest_sha256": complete["content_sha256"], "manifest_sha256": manifest["content_sha256"],
            "source_inputs": source_inputs, "training": metadata, "benchmark_coverage": coverage,
            "backbone_initializations_sha256": initializations,
            "settings_audit": changes, "training_geometry": geometry, "test_geometry": test_geometry,
            "checkpoint_reuse": "none; every new run starts from its original declared initialization",
            "execution": "not_started",
        }
        contents = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        candidate = staging / "plan.json"
        candidate.write_text(contents, encoding="utf-8")
        read_plan(candidate)
        if sha256_file(source) != source_hash or sha256_file(source_manifest) != manifest_file_hash:
            raise ValueError("Source plan or manifest changed during preparation; no output written")
        for item in source_inputs:
            if sha256_file(item["manifest"]) != item["manifest_file_sha256"]:
                raise ValueError("Original training manifest changed during preparation; no output written")
        if any(sha256_file(path) != digest for path, digest in initializations.items()):
            raise ValueError("Backbone initialization changed during preparation; no output written")
        destination.mkdir(parents=True, exist_ok=False)
        new_manifest_path.write_bytes((staging / "manifest.json").read_bytes())
        plan_file.parent.mkdir(parents=True)
        plan_file.write_text(contents, encoding="utf-8")
        (plan_file.parent / "source_plan.json").write_bytes(source.read_bytes())
        for script, actions in (("run_all.sh", ("train", "evaluate", "profile")), ("train.sh", ("train",))):
            command = [plan["jobs"][0]["train"][0], str(HERE / "run_plan.py"), "--plan", str(plan_file),
                       "--actions", *actions, "--skip-completed"]
            text = "#!/usr/bin/env bash\nset -euo pipefail\n# Explicit execution: starts fresh full-training runs.\n"
            (plan_file.parent / script).write_text(text + shlex.join(command) + "\n", encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Absolute path to the original experiment plan")
    parser.add_argument("--manifest", required=True, help="Complete four-test holdout manifest, e.g. complete_benchmark.py output")
    parser.add_argument("--output", required=True, help="NEW absolute result root; original results are never overwritten")
    parser.add_argument("--data-root", help="Optional explicit relocation of the complete manifest's data root")
    parser.add_argument("--rebind-entrypoints", action="store_true",
                        help="Explicitly bind a validated older plan to this isolated code release; Python environment is retained")
    args = parser.parse_args(argv)
    try:
        plan = prepare_full_training_plan(args.plan, args.manifest, args.output, args.data_root,
                                          rebind_entrypoints=args.rebind_entrypoints)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"plan": str(Path(args.output).expanduser().resolve() / "plans" / PHASE / "plan.json"),
                      "status": plan["status"], "run_count": plan["run_count"], "results": plan["results"],
                      "training": plan["full_training"]["training"],
                      "benchmark_coverage": plan["full_training"]["benchmark_coverage"],
                      "checkpoint_reuse": plan["full_training"]["checkpoint_reuse"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
