"""Complete test coverage without changing a checkpoint's training manifest.

All operations here inspect local files only. No dataset download, training,
checkpoint editing or image/ground-truth conversion is performed.
"""
from __future__ import annotations

import copy
import os
from collections import Counter
from numbers import Integral
from pathlib import Path

from .data import SOURCES, _scan, load_manifest, manifest_digest, sha256_file, validate_manifest


EXPECTED_TEST_COUNTS = {"camo": 250, "chameleon": 76, "cod10k": 2026, "nc4k": 4121}
COD10K_PREFIX = "COD10K-CAM-"


def benchmark_coverage(counts):
    """Describe official four-dataset coverage; zero counts mean missing data."""
    if any(isinstance(count, bool) or not isinstance(count, Integral) for count in counts.values()):
        raise ValueError("Dataset counts must be integers")
    observed = {str(name): int(count) for name, count in sorted(counts.items())}
    if any(count < 0 for count in observed.values()):
        raise ValueError("Dataset counts cannot be negative")
    expected = dict(EXPECTED_TEST_COUNTS)
    missing = sorted(name for name in expected if observed.get(name, 0) == 0)
    unexpected = sorted(set(observed) - set(expected))
    nonstandard = {name: {"expected": expected[name], "observed": observed[name]}
                   for name in sorted(set(expected) & set(observed))
                   if observed[name] > 0 and observed[name] != expected[name]}
    return {"expected_counts": expected, "observed_counts": observed,
            "missing_datasets": missing, "unexpected_datasets": unexpected,
            "nonstandard_counts": nonstandard,
            "complete": not (missing or unexpected or nonstandard)}


def _test_counts(manifest):
    return dict(sorted(Counter(record["dataset"] for record in manifest["records"]
                               if record["split"] == "test").items()))


def _require_complete(manifest):
    coverage = benchmark_coverage(_test_counts(manifest))
    if not coverage["complete"]:
        raise ValueError(f"Four-dataset test benchmark is incomplete or nonstandard: {coverage}")
    return coverage


def _rebase(path, old_root, new_root, strict=True):
    return (old_root / path).resolve(strict=strict).relative_to(new_root).as_posix()


def _rebased_original(manifest, old_root, new_root, strict=True):
    result = copy.deepcopy(manifest)
    result["data_root"] = str(new_root)
    for record in result["records"]:
        for kind in ("image", "mask"):
            record[kind] = _rebase(record[kind], old_root, new_root, strict=strict)
    for source in result.get("excluded_files", {}).values():
        for files in source.values():
            for item in files:
                item["path"] = _rebase(item["path"], old_root, new_root, strict=strict)
    result["unselected_shared_masks"] = {
        _rebase(folder, old_root, new_root, strict=strict): value
        for folder, value in result.get("unselected_shared_masks", {}).items()
    }
    return result


def _find_cod10k(root, images=None, masks=None):
    if (images is None) != (masks is None):
        raise ValueError("Provide both COD10K image and mask directories")
    if images is not None:
        def resolve(path):
            path = Path(path).expanduser()
            return (path if path.is_absolute() else root / path).resolve(strict=True)
        image_dir, mask_dir = resolve(images), resolve(masks)
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise NotADirectoryError("COD10K images and masks must both be directories")
        if image_dir == mask_dir:
            raise ValueError("COD10K image and mask directories must be different")
        return image_dir, mask_dir

    # Deliberately bounded: do not recursively search an entire server or select
    # the first of several copies. Explicit arguments handle other layouts.
    layouts = [("COD10K-v3/Test/Image", "COD10K-v3/Test/GT_Object")]
    for prefix in ("", "TestDataset/", "Dataset/TestDataset/", "COD10K-test + CAMO-test + CHAMELEON/"):
        for dataset in ("COD10K", "COD10K-test"):
            for image_name, mask_name in (("Image", "GT_Object"), ("Imgs", "GT"), ("Image", "GT")):
                base = f"{prefix}{dataset}/"
                layouts.append((base + image_name, base + mask_name))
    candidates = set()
    for image_rel, mask_rel in layouts:
        image_dir, mask_dir = root / image_rel, root / mask_rel
        if image_dir.is_dir() and mask_dir.is_dir():
            candidates.add((image_dir.resolve(), mask_dir.resolve()))
    if not candidates:
        raise FileNotFoundError("Cannot find COD10K-test beneath the original data root; provide explicit --cod10k-images and --cod10k-masks directories")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous COD10K-test directories; provide explicit paths: {sorted((str(a), str(b)) for a, b in candidates)}")
    return next(iter(candidates))


def _validate_snapshot(manifest, root):
    """Validate portable record metadata even when source disks are unmounted."""
    if manifest.get("schema_version") != 1 or manifest.get("content_sha256") != manifest_digest(manifest):
        raise ValueError("Invalid manifest schema/checksum")
    root = Path(root or manifest["data_root"]).expanduser().resolve()
    records = manifest["records"]
    ids, paths, hashes = set(), {}, {}
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {record['id']}")
        ids.add(record["id"])
        source = SOURCES.get(record["source"])
        if (source is None or record["split"] not in {"train", "val", "test"}
                or source[0] != record["dataset"] or (source[1] == "test") != (record["split"] == "test")):
            raise ValueError(f"Invalid source/split separation: {record['id']}")
        for kind in ("image", "mask"):
            path = (root / record[kind]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Path escapes dataset root: {record[kind]}")
            digest = record[f"{kind}_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid {kind} byte hash: {record['id']}")
            if kind == "image":
                paths.setdefault(str(path), set()).add(record["split"])
        hashes.setdefault(record["image_sha256"], set()).add(record["split"])
    if any(len(splits) > 1 for splits in [*paths.values(), *hashes.values()]):
        raise ValueError("Identical image paths or byte hashes cross train/val/test splits")
    from .training_protocol import validate_training_protocol
    validate_training_protocol(manifest, manifest.get("training_protocol", "holdout"))
    for exclusions in manifest.get("excluded_files", {}).values():
        for files in exclusions.values():
            for item in files:
                if not (root / item["path"]).resolve().is_relative_to(root):
                    raise ValueError("Excluded path escapes dataset root")
    if (manifest["source_counts"] != dict(Counter(record["source"] for record in records))
            or manifest["split_counts"] != dict(Counter(f"{record['dataset']}/{record['split']}" for record in records))):
        raise ValueError("Manifest count metadata disagrees with records")
    return root


def validate_test_extension(training_manifest, training_root, evaluation_manifest, evaluation_root, *, verify_files=True):
    """Verify an additive, complete test manifest against immutable training data.

    Existing records must refer to the exact same resolved files and hashes.
    The only additional samples allowed are canonical COD10K foreground tests.
    Original split parameters and selection policy must remain unchanged.
    Returns the verified coverage/provenance for evaluation reports.
    """
    training_root = _validate_snapshot(training_manifest, training_root)
    evaluation_root = _validate_snapshot(evaluation_manifest, evaluation_root)
    if verify_files:
        validate_manifest(training_manifest, data_root=training_root)
        validate_manifest(evaluation_manifest, data_root=evaluation_root)
    coverage = _require_complete(evaluation_manifest)
    original = _rebased_original(training_manifest, training_root, evaluation_root, strict=verify_files)
    mutable_keys = {"content_sha256", "data_root", "records", "source_counts", "split_counts",
                    "test_datasets_selected", "test_datasets_missing", "test_datasets_unselected",
                    "raw_source_counts", "excluded_files", "test_extension"}
    for key in (set(original) | set(evaluation_manifest)) - mutable_keys:
        if original.get(key) != evaluation_manifest.get(key):
            raise ValueError(f"Evaluation extension changed original manifest metadata: {key}")
    if original.get("selection_policy", {}).get("cod10k_prefix") != COD10K_PREFIX:
        raise ValueError("Canonical COD10K-test extension requires the original COD10K-CAM- selection policy")

    count = len(original["records"])
    if evaluation_manifest["records"][:count] != original["records"]:
        raise ValueError("Evaluation extension changed, removed or reordered original records")
    added = evaluation_manifest["records"][count:]
    old_test_cod10k = any(record["source"] == "cod10k_test" for record in original["records"])
    if old_test_cod10k and added:
        raise ValueError("Existing COD10K-test records cannot be supplemented or replaced")
    protected_paths = {str((training_root / record[kind]).resolve())
                       for record in training_manifest["records"] if record["split"] != "test"
                       for kind in ("image", "mask")}
    original_image_paths = {str((training_root / record["image"]).resolve())
                            for record in training_manifest["records"]}
    original_image_hashes = {record["image_sha256"] for record in training_manifest["records"]}
    for record in added:
        stem = Path(record["image"]).stem
        if (record["source"] != "cod10k_test" or record["dataset"] != "cod10k"
                or record["split"] != "test" or not stem.startswith(COD10K_PREFIX)
                or Path(record["mask"]).stem != stem or record["id"] != f"cod10k_test/{stem}"):
            raise ValueError("Only paired COD10K-CAM- foreground test records may be added")
        for kind in ("image", "mask"):
            if str((evaluation_root / record[kind]).resolve()) in protected_paths:
                raise ValueError("Added test path overlaps an original training/validation file")
        if (str((evaluation_root / record["image"]).resolve()) in original_image_paths
                or record["image_sha256"] in original_image_hashes):
            raise ValueError("Added test image duplicates an original image path or byte hash")

    for key in ("raw_source_counts", "excluded_files"):
        expected = copy.deepcopy(original.get(key, {}))
        actual = copy.deepcopy(evaluation_manifest.get(key, {}))
        if added:
            expected.pop("cod10k_test", None)
            actual.pop("cod10k_test", None)
        if expected != actual:
            raise ValueError(f"Evaluation extension changed original audit metadata: {key}")
    tests = sorted(_test_counts(evaluation_manifest))
    if evaluation_manifest.get("test_datasets_selected") != tests:
        raise ValueError("Evaluation test selection metadata disagrees with records")
    if evaluation_manifest.get("test_datasets_missing") or evaluation_manifest.get("test_datasets_unselected"):
        raise ValueError("Complete evaluation must not retain missing/unselected test metadata")
    provenance = evaluation_manifest.get("test_extension", {})
    if (provenance.get("schema_version") != 1
            or provenance.get("training_manifest_sha256") != training_manifest["content_sha256"]
            or provenance.get("original_record_count") != count
            or provenance.get("added_records") != len(added)):
        raise ValueError("Evaluation extension provenance does not match the training manifest")
    if (str(training_root) != provenance.get("training_data_root")
            or evaluation_root != Path(evaluation_manifest["data_root"]).expanduser().resolve()):
        raise ValueError("Evaluation extension roots disagree with audited provenance; regenerate the extension for relocated data")
    return {"schema_version": 1, "training_manifest_sha256": training_manifest["content_sha256"],
            "evaluation_manifest_sha256": evaluation_manifest["content_sha256"],
            "original_record_count": len(original["records"]),
            "added_records": len(added), "benchmark_coverage": coverage}


def extend_test_manifest(training_manifest_path, output_path, cod10k_images=None, cod10k_masks=None, data_root=None):
    """Write a new complete evaluation manifest while leaving the original intact."""
    import json

    source_path = Path(training_manifest_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Evaluation manifest already exists: {target}")
    manifest, root = load_manifest(source_path, data_root=data_root)
    counts = _test_counts(manifest)
    preflight = benchmark_coverage(counts)
    if preflight["unexpected_datasets"] or preflight["nonstandard_counts"]:
        raise ValueError(f"Original test datasets have nonstandard counts: {preflight}")
    if set(preflight["missing_datasets"]) - {"cod10k"}:
        raise ValueError(f"This extension adds COD10K-test only; other test datasets are missing: {preflight['missing_datasets']}")
    images, masks, exclusions = {}, {}, {}
    new_root = root
    if not counts.get("cod10k"):
        image_dir, mask_dir = _find_cod10k(root, cod10k_images, cod10k_masks)
        raw_images, raw_masks = _scan(image_dir), _scan(mask_dir)
        images = {stem: path for stem, path in raw_images.items() if stem.startswith(COD10K_PREFIX)}
        masks = {stem: path for stem, path in raw_masks.items() if stem.startswith(COD10K_PREFIX)}
        if set(images) != set(masks):
            raise ValueError(f"COD10K-test image/mask pairing mismatch: missing masks={sorted(set(images) - set(masks))[:8]}, orphan masks={sorted(set(masks) - set(images))[:8]}")
        if len(images) != EXPECTED_TEST_COUNTS["cod10k"]:
            raise ValueError(f"COD10K-test requires {EXPECTED_TEST_COUNTS['cod10k']} foreground pairs; found {len(images)}")
        new_root = Path(os.path.commonpath([str(root), str(image_dir), str(mask_dir)]))
        exclusions = {kind: [{"path": path.resolve().relative_to(new_root).as_posix(), "sha256": sha256_file(path)}
                              for stem, path in paths.items() if not stem.startswith(COD10K_PREFIX)]
                      for kind, paths in (("images", raw_images), ("masks", raw_masks))}
    elif cod10k_images is not None or cod10k_masks is not None:
        raise ValueError("COD10K-test is already present; explicit replacement directories are not allowed")
    extended = _rebased_original(manifest, root, new_root)
    for stem, image_path in sorted(images.items()):
        mask_path = masks[stem]
        extended["records"].append({"id": f"cod10k_test/{stem}", "dataset": "cod10k",
                                    "source": "cod10k_test", "split": "test",
                                    "image": image_path.resolve().relative_to(new_root).as_posix(),
                                    "mask": mask_path.resolve().relative_to(new_root).as_posix(),
                                    "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(mask_path)})
    if images:
        extended.setdefault("raw_source_counts", {})["cod10k_test"] = {"images": len(raw_images), "masks": len(raw_masks)}
        extended.setdefault("excluded_files", {})["cod10k_test"] = exclusions
    extended["source_counts"] = dict(sorted(Counter(record["source"] for record in extended["records"]).items()))
    extended["split_counts"] = dict(sorted(Counter(f"{record['dataset']}/{record['split']}" for record in extended["records"]).items()))
    extended["test_datasets_selected"] = sorted(_test_counts(extended))
    extended["test_datasets_missing"] = []
    extended["test_datasets_unselected"] = []
    extended["test_extension"] = {"schema_version": 1, "training_manifest": str(source_path),
                                  "training_manifest_sha256": manifest["content_sha256"],
                                  "training_manifest_file_sha256": sha256_file(source_path),
                                  "training_data_root": str(root), "original_record_count": len(manifest["records"]),
                                  "added_records": len(images), "policy": "preserve all original records; append canonical COD10K-test only"}
    extended["content_sha256"] = manifest_digest(extended)
    validate_test_extension(manifest, root, extended, new_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(extended, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return extended
