#!/usr/bin/env python3
"""Reproduce coverage for all 158 full models using inference only.

Run with the study's pinned Python environment and absolute study location:
  python verify_complexity_grid.py --study /absolute/FSA-Pseudo-Scale-Study \
      --output /absolute/new-coverage.json

CPU, two threads, batch one, RGB 64x64, eight decoder channels, random weights.
No datasets, downloads, optimizer, backward, or training-mode forward is used.
This reproduces operator counts and coverage in all158-fullmodel-coverage.json;
elapsed time is machine-dependent. Counts depend on executed shapes/operators,
not the synthetic input's numerical values. The original report did not fix the
input seed; this reusable verifier fixes it to zero for repeatability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="Absolute study directory")
    parser.add_argument("--output", required=True, help="New JSON output; existing files are rejected")
    args = parser.parse_args()
    if not Path(args.study).is_absolute():
        parser.error("--study must be an absolute path")
    study = Path(args.study).resolve(strict=True)
    if not (study / "fsa_study" / "complexity.py").is_file():
        parser.error("--study must contain fsa_study/complexity.py")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("--output already exists; choose a new path")

    sys.path.insert(0, str(study))
    import torch
    from fsa_study.complexity import COMPLEXITY_PROTOCOL, measure_complexity
    from fsa_study.generators import GENERATOR_NAMES
    from fsa_study.model import ARCHIVE, StudyModel
    from fsa_study.utilizers import UTILIZER_NAMES

    if (len(GENERATOR_NAMES), len(UTILIZER_NAMES)) != (12, 13):
        parser.error("this historical verifier expects 12 generators and 13 utilizers")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    input_rng = torch.Generator(device="cpu").manual_seed(0)
    configs = [{"generator": generator, "utilizer": utilizer}
               for generator in GENERATOR_NAMES for utilizer in UTILIZER_NAMES]
    configs += [{"baseline": True}, {"legacy_reference": True}]

    # Preserve source provenance for both the new suite and its archived model
    # dependency. Hash contents before executing, then check they did not change.
    sources = sorted((study / "fsa_study").glob("*.py"))
    sources += sorted(ARCHIVE.rglob("*.py"))
    sources = [path for path in sources if "__pycache__" not in path.parts]
    source_hashes = {str(path): sha256(path) for path in sources}
    rows = []
    start = time.monotonic()
    for config in configs:
        model = StudyModel(channels=8, seed=0, **config).eval().cpu()
        x = torch.rand(1, 3, 64, 64, generator=input_rng)
        report = measure_complexity(model, x)
        rows.append({"config": config, "macs": report["macs"],
                     "estimated_total_flops": report["estimated_total_flops"],
                     "coverage": report["coverage"]})
        del model, x
    elapsed = time.monotonic() - start
    changed = [path for path, expected in source_hashes.items() if sha256(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"Source changed during verification: {changed}")
    complete = all(row["coverage"]["all_observed_ops_classified"] for row in rows)
    result = {
        "scope": "inference only; synthetic random weights/input; channels8 size64 batch1 CPU",
        "elapsed_s": elapsed,
        "variants": rows,
        "all_observed_ops_classified": complete,
        "protocol": {"size": 64, "channels": 8, "batch_size": 1, "device": "cpu",
                     "threads": 2, "seed": 0, "weights": "random_initialization",
                     "training": "not_run", "complexity": COMPLEXITY_PROTOCOL},
        "environment": {"torch": str(torch.__version__), "python": platform.python_version()},
        "source_sha256": source_hashes,
        "verifier_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "variants": len(rows),
                      "all_observed_ops_classified": complete, "training": "not_run"}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
