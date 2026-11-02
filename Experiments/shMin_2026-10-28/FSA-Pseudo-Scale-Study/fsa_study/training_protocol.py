"""Explicit development holdout and full-training policies; no tensor imports."""
from __future__ import annotations

from collections import Counter

TRAINING_PROTOCOLS = ("holdout", "full_train")
HOLDOUT_SELECTION = "minimum sample-weighted validation float MAE; strict improvement; no test access"
FULL_TRAIN_SELECTION = "fixed final epoch; no validation or test selection"
OFFICIAL_TRAIN_COUNTS = {"camo": 1000, "cod10k": 3040}


def resolve_val_fraction(protocol="holdout", val_fraction=None):
    """Resolve an omitted CLI fraction without silently changing explicit values."""
    if protocol not in TRAINING_PROTOCOLS:
        raise ValueError(f"Unknown training protocol: {protocol}")
    if val_fraction is None:
        val_fraction = 0.1 if protocol == "holdout" else 0.0
    if protocol == "full_train":
        if val_fraction != 0:
            raise ValueError("full_train requires val_fraction=0; no validation samples may be reserved")
    elif not 0 < val_fraction < 1:
        raise ValueError("holdout val_fraction must be between 0 and 1")
    return float(val_fraction)


def validate_training_protocol(manifest, protocol="holdout"):
    """Reject ambiguous splits before a model or optimizer is constructed.

    Historical manifests have no protocol marker and retain holdout semantics.
    A full-training manifest must be explicitly marked and contain all official
    training pairs, except when its audited nonstandard-count override is set.
    Test-source records may never be reassigned to either training split.
    """
    if protocol not in TRAINING_PROTOCOLS:
        raise ValueError(f"Unknown training protocol: {protocol}")
    recorded = manifest.get("training_protocol", "holdout")
    if recorded != protocol:
        raise ValueError(f"Training protocol mismatch: requested {protocol}, manifest declares {recorded}")
    resolve_val_fraction(protocol, manifest.get("val_fraction"))
    counts = {split: Counter() for split in ("train", "val")}
    for record in manifest["records"]:
        split = record["split"]
        if split not in counts:
            continue
        dataset = record["dataset"]
        if dataset not in OFFICIAL_TRAIN_COUNTS or record["source"] != f"{dataset}_train":
            raise ValueError(f"Training split contains a non-training source: {record['id']}")
        counts[split][dataset] += 1
    if not counts["train"]:
        raise ValueError("Training protocol requires a nonempty train set")
    if protocol == "holdout":
        if not counts["val"]:
            raise ValueError("Manifest must retain distinct nonempty train and validation sets")
        return
    if counts["val"]:
        raise ValueError("full_train must not contain validation records")
    if set(counts["train"]) != set(OFFICIAL_TRAIN_COUNTS):
        raise ValueError("full_train requires both CAMO and COD10K training sources")
    if not manifest.get("allow_nonstandard_counts", False) and dict(counts["train"]) != OFFICIAL_TRAIN_COUNTS:
        raise ValueError("full_train requires 4040 training pairs (CAMO 1000 + COD10K 3040); "
                         "nonstandard counts need an explicitly audited manifest override")


def training_metadata(manifest, protocol="holdout"):
    """Count actual records and describe checkpoint selection, never test scores."""
    validate_training_protocol(manifest, protocol)
    counts = {split: dict(sorted(Counter(record["dataset"] for record in manifest["records"]
                                         if record["split"] == split).items()))
              for split in ("train", "val")}
    return {"training_protocol": protocol,
            "training_split_counts": counts,
            "training_sample_count": sum(counts["train"].values()),
            "validation_sample_count": sum(counts["val"].values()),
            "checkpoint_selection": HOLDOUT_SELECTION if protocol == "holdout" else FULL_TRAIN_SELECTION}


def prepare_full_training_manifest(source_manifest_path, output_path, data_root=None):
    """Create a new full-training manifest from verified, complete held-out data.

    Preserve custom layouts and every test record byte-for-byte at record level.
    Only original validation records become training records. No input is moved,
    downloaded or modified, and no model/checkpoint is constructed or changed.
    """
    import copy
    import json
    from pathlib import Path

    # Local imports avoid the data -> training_protocol dependency cycle.
    from .benchmark import benchmark_coverage
    from .data import load_manifest, manifest_digest, sha256_file, validate_manifest
    from .overlap import overlap_metadata, refresh_overlap_audit

    source_path = Path(source_manifest_path).expanduser().resolve(strict=True)
    requested_target = Path(output_path).expanduser()
    if requested_target.is_symlink():
        raise FileExistsError(f"Full-training manifest path is an existing symlink: {requested_target}")
    target = requested_target.resolve()
    if target.exists():
        raise FileExistsError(f"Full-training manifest already exists; choose a new path: {target}")
    source_file_hash = sha256_file(source_path)
    original, root = load_manifest(source_path, verify_files=True, data_root=data_root)
    validate_training_protocol(original, "holdout")
    counts = dict(sorted(Counter(record["dataset"] for record in original["records"]
                                 if record["split"] == "test").items()))
    coverage = benchmark_coverage(counts)
    if not coverage["complete"]:
        raise ValueError(f"Full training requires the complete four-dataset test benchmark: {coverage}")
    result = copy.deepcopy(original)
    result["data_root"] = str(root)
    result["training_protocol"] = "full_train"
    result["val_fraction"] = 0.0
    promoted = 0
    for record in result["records"]:
        if record["split"] == "val":
            record["split"] = "train"
            promoted += 1
    result["source_counts"] = dict(sorted(Counter(record["source"] for record in result["records"]).items()))
    result["split_counts"] = dict(sorted(Counter(f"{record['dataset']}/{record['split']}"
                                                for record in result["records"]).items()))
    # An evaluation extension's original holdout invariant no longer applies to
    # the NEW training protocol. Preserve that evidence as historical provenance.
    previous_extension = result.pop("test_extension", None)
    result["full_training_source"] = {
        "schema_version": 1, "source_manifest": str(source_path),
        "source_manifest_sha256": original["content_sha256"],
        "source_manifest_file_sha256": source_file_hash,
        "source_training_protocol": "holdout", "source_data_root": str(root),
        "promoted_validation_samples": promoted, "source_test_extension": previous_extension,
        "policy": "preserve original train and test records; promote original validation to train; fixed final epoch",
    }
    if "overlap_policy" in original:
        result["full_training_source"]["source_overlap_audit"] = overlap_metadata(original)
        refresh_overlap_audit(result)
    result["content_sha256"] = manifest_digest(result)
    validate_manifest(result, verify_files=False)
    validate_training_protocol(result, "full_train")
    if sha256_file(source_path) != source_file_hash:
        raise ValueError("Source manifest changed during full-training preparation; no output written")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result
