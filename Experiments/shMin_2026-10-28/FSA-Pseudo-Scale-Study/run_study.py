#!/usr/bin/env python3
"""Explicit prepare/train/evaluate entrypoint; --help and prepare do not import torch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def json_object(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Expected a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def _common_input(parser):
    parser.add_argument("--manifest", required=True, help="Audited dataset manifest JSON")
    parser.add_argument("--data-root", help="Explicit relocation of dataset root; hashes must still match")
    parser.add_argument("--output", required=True, help="New result directory; existing directories are rejected")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="Explicit torch device, e.g. cuda:0 or cpu")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Hash local image/mask pairs and create fixed training holdout")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--split-seed", type=int, default=1729)
    prepare.add_argument("--val-fraction", type=float, default=.1)
    prepare.add_argument("--cod10k-prefix", default="COD10K-CAM-", help="Explicit COD10K foreground prefix; exclusions and hashes recorded. Empty string requires an audited foreground-only renamed copy")
    prepare.add_argument("--allow-nonstandard-counts", action="store_true", help="Explicit custom-data/fixture override; record actual counts")
    prepare.add_argument("--test-datasets", nargs="+", choices=["cod10k", "camo", "chameleon", "nc4k", "none"],
                         default=None, help="Default: all available test datasets, recording missing ones; explicit names must exist")

    train = subparsers.add_parser("train", help="Run explicitly requested optimization; select only on training holdout")
    _common_input(train)
    train.add_argument("--generator", default="legacy")
    train.add_argument("--utilizer", default="mean")
    train.add_argument("--generator-options", type=json_object, default={})
    train.add_argument("--utilizer-options", type=json_object, default={})
    controls = train.add_mutually_exclusive_group()
    controls.add_argument("--baseline", action="store_true", help="Single-encoder control without pseudo generation/use")
    controls.add_argument("--legacy-reference", action="store_true", help="Original full FSA architecture reference")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--size", type=int, default=384)
    train.add_argument("--channels", type=int, default=48)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--lr", type=float, default=.0001)
    train.add_argument("--eval-interval", type=int, default=1, help="Validation-only checkpoint selection interval in epochs")
    train.add_argument("--determinism", choices=["warn", "strict"], default="warn",
                       help="warn: record unsupported deterministic ops; strict: reject them (CUDA grid_sample backward may fail)")
    initialization = train.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--backbone-weights", help="Existing local PVT backbone checkpoint; never downloaded")
    initialization.add_argument("--random-init", action="store_true", help="Explicitly run all weights from random initialization")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one exact study checkpoint without optimizer updates")
    _common_input(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], required=True)
    evaluate.add_argument("--test-geometry", choices=["strict", "resize-prediction-to-gt"], default="strict",
                          help="Test only: opt into prediction resizing to untouched original GT; mismatches are audited")
    evaluate.add_argument("--no-save-predictions", action="store_true",
                          help="Skip test probability PNGs; scores, per-image CSV and CSV/SVG/PNG curves are still saved")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from fsa_study.data import prepare_manifest
            manifest = prepare_manifest(args.data_root, args.manifest, args.split_seed, args.val_fraction,
                                        args.allow_nonstandard_counts, args.test_datasets, args.cod10k_prefix)
            print(json.dumps({"manifest": str(Path(args.manifest).expanduser().resolve()),
                              "content_sha256": manifest["content_sha256"],
                              "source_counts": manifest["source_counts"], "split_counts": manifest["split_counts"],
                              "test_datasets_selected": manifest["test_datasets_selected"],
                              "test_datasets_missing": manifest["test_datasets_missing"]}, indent=2))
        else:
            # Must precede torch import / CUDA initialization.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from fsa_study import engine
            getattr(engine, args.command)(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
