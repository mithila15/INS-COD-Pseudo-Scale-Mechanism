"""Auditable COD split manifests; no downloads and no torch import at module import."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .training_protocol import resolve_val_fraction, validate_training_protocol
from .overlap import ACKNOWLEDGED_POLICY, acknowledge_cod10k_overlap, overlap_metadata

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GEOMETRY_POLICIES = ("strict", "resize-prediction-to-gt")
SOURCES = {
    "cod10k_train": ("cod10k", "train", "COD10K-v3/Train/Image", "COD10K-v3/Train/GT_Object", 3040),
    "camo_train": ("camo", "train", "CAMO-V.1.0-CVIU2019/Images/Train", "CAMO-V.1.0-CVIU2019/GT", 1000),
    "cod10k_test": ("cod10k", "test", "COD10K-v3/Test/Image", "COD10K-v3/Test/GT_Object", 2026),
    "camo_test": ("camo", "test", "CAMO-V.1.0-CVIU2019/Images/Test", "CAMO-V.1.0-CVIU2019/GT", 250),
    "chameleon_test": ("chameleon", "test", "CHAMELEON/Imgs", "CHAMELEON/GT", 76),
    "nc4k_test": ("nc4k", "test", "NC4K/Imgs", "NC4K/GT", 4121),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _scan(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Required dataset directory is missing: {folder}")
    paths = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in paths:
                raise ValueError(f"Duplicate image/mask stem {path.stem!r} in {folder}")
            paths[path.stem] = path
    if not paths:
        raise ValueError(f"No image/mask files in {folder}")
    return paths


def prepare_manifest(data_root, manifest_path, split_seed=1729, val_fraction=0.1, allow_nonstandard_counts=False, test_datasets=None, cod10k_prefix="COD10K-CAM-", training_protocol="holdout", acknowledge_cod10k=False):
    root = Path(data_root).expanduser().resolve(strict=True)
    target = Path(manifest_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Manifest already exists; choose another path: {target}")
    val_fraction = resolve_val_fraction(training_protocol, val_fraction)
    test_names = {spec[0] for spec in SOURCES.values() if spec[1] == "test"}
    missing_tests = []
    available_tests = []
    for dataset in sorted(test_names):
        _, _, image_rel, mask_rel, _ = SOURCES[f"{dataset}_test"]
        if (root / image_rel).is_dir() and (root / mask_rel).is_dir():
            available_tests.append(dataset)
        else:
            missing_tests.append(dataset)
    selected_tests = set(available_tests if test_datasets is None else test_datasets)
    if "none" in selected_tests:
        if len(selected_tests) != 1:
            raise ValueError("--test-datasets none cannot be combined with dataset names")
        selected_tests = set()
    if not selected_tests.issubset(test_names):
        raise ValueError(f"Unknown test datasets: {selected_tests - test_names}")
    if selected_tests.intersection(missing_tests):
        raise FileNotFoundError(f"Explicitly selected test datasets are missing: {sorted(selected_tests.intersection(missing_tests))}")
    active_sources = {source: spec for source, spec in SOURCES.items()
                      if spec[1] == "train" or spec[0] in selected_tests}
    sources = {}
    raw_source_counts = {}
    excluded_files = {}
    mask_users = defaultdict(set)
    mask_files = {}
    for source, (dataset, source_split, image_rel, mask_rel, expected) in active_sources.items():
        images, masks = _scan(root / image_rel), _scan(root / mask_rel)
        raw_source_counts[source] = {"images": len(images), "masks": len(masks)}
        if dataset == "cod10k" and cod10k_prefix:
            excluded_files[source] = {kind: [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                                                for stem, path in paths.items() if not stem.startswith(cod10k_prefix)]
                                      for kind, paths in (("images", images), ("masks", masks))}
            images = {stem: path for stem, path in images.items() if stem.startswith(cod10k_prefix)}
            masks = {stem: path for stem, path in masks.items() if stem.startswith(cod10k_prefix)}
            if not images:
                raise ValueError(f"{source}: no images match explicit COD10K prefix {cod10k_prefix!r}; use --cod10k-prefix '' only for an audited renamed foreground-only copy")
        missing = sorted(set(images) - set(masks))
        if missing:
            raise ValueError(f"{source}: {len(missing)} images have no mask: {missing[:8]}")
        if not allow_nonstandard_counts and len(images) != expected:
            raise ValueError(f"{source}: expected {expected} pairs, found {len(images)}. Check the official foreground-only split; use --allow-nonstandard-counts only for an explicitly documented alternate dataset.")
        mask_users[mask_rel].update(images)
        mask_files[mask_rel] = masks
        sources[source] = (images, masks)
    # CAMO shares one GT directory between Train and Test; check its union.
    unselected_shared_masks = {}
    for mask_rel, used in mask_users.items():
        extra = sorted(set(mask_files[mask_rel]) - used)
        inactive_owners = [source for source, spec in SOURCES.items() if spec[3] == mask_rel and source not in active_sources]
        if extra and inactive_owners:
            unselected_shared_masks[mask_rel] = {"stems": extra, "unselected_source_splits": inactive_owners}
        elif extra:
            raise ValueError(f"{mask_rel}: {len(extra)} masks have no matching image in its declared source split(s): {extra[:8]}")
    records = []
    for source, (dataset, source_split, _image_rel, _mask_rel, expected) in active_sources.items():
        images, masks = sources[source]
        stems = sorted(images)
        validation = set()
        if source_split == "train" and training_protocol == "holdout":
            if len(stems) < 2:
                raise ValueError(f"{source}: at least two pairs required for train/validation split")
            # Dataset-specific seed isolates the split from iteration order and model RNG.
            rng = random.Random(f"{split_seed}:{source}")
            shuffled = list(stems)
            rng.shuffle(shuffled)
            count = max(1, min(len(stems) - 1, round(len(stems) * val_fraction)))
            validation = set(shuffled[:count])
        for stem in stems:
            image_path, mask_path = images[stem], masks[stem]
            records.append({
                "id": f"{source}/{stem}", "dataset": dataset, "source": source,
                "split": "val" if stem in validation else source_split,
                "image": image_path.relative_to(root).as_posix(),
                "mask": mask_path.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(mask_path),
            })
    manifest = {
        "schema_version": 1, "data_root": str(root), "split_seed": int(split_seed),
        "val_fraction": float(val_fraction), "allow_nonstandard_counts": bool(allow_nonstandard_counts),
        "test_datasets_selected": sorted(selected_tests), "test_datasets_missing": missing_tests,
        "test_datasets_unselected": sorted(test_names - selected_tests),
        "unselected_shared_masks": unselected_shared_masks,
        "selection_policy": {"cod10k_prefix": cod10k_prefix, "selection": "explicit filename prefix before strict image-mask pairing"},
        "raw_source_counts": raw_source_counts, "excluded_files": excluded_files,
        "source_counts": dict(sorted(Counter(record["source"] for record in records).items())),
        "split_counts": dict(sorted(Counter(f"{record['dataset']}/{record['split']}" for record in records).items())),
        "records": records,
    }
    # Preserve historical holdout serialization/digests. Full training is an
    # explicit new manifest, never an in-place relabeling of a completed run.
    if training_protocol == "full_train":
        manifest["training_protocol"] = training_protocol
    if acknowledge_cod10k:
        acknowledge_cod10k_overlap(manifest)
    manifest["content_sha256"] = manifest_digest(manifest)
    validate_manifest(manifest, verify_files=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def validate_manifest(manifest, verify_files=True, data_root=None):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    if manifest.get("content_sha256") != manifest_digest(manifest):
        raise ValueError("Manifest checksum mismatch; regenerate an audited manifest")
    root = Path(data_root or manifest["data_root"]).expanduser().resolve(strict=True)
    records = manifest["records"]
    if not records:
        raise ValueError("Manifest contains no samples")
    ids, image_paths = set(), {}
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {record['id']}")
        ids.add(record["id"])
        if record["source"] not in SOURCES or record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid source/split: {record['id']}")
        dataset, source_split, _, _, _ = SOURCES[record["source"]]
        if dataset != record["dataset"] or (source_split == "test") != (record["split"] == "test"):
            raise ValueError(f"Test/train source separation violation: {record['id']}")
        for kind in ("image", "mask"):
            path = (root / record[kind]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"Path escapes dataset root: {record[kind]}")
            if verify_files and sha256_file(path) != record[f"{kind}_sha256"]:
                raise ValueError(f"Changed {kind} content: {path}")
            if kind == "image":
                previous = image_paths.get(str(path))
                if previous is not None and (previous != record["split"] or manifest.get("overlap_policy") == ACKNOWLEDGED_POLICY):
                    raise ValueError(f"Same image path crosses splits: {path}")
                image_paths[str(path)] = record["split"]
    for exclusions in manifest.get("excluded_files", {}).values():
        for paths in exclusions.values():
            for excluded in paths:
                path = (root / excluded["path"]).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError(f"Excluded path escapes dataset root: {path}")
                if verify_files and sha256_file(path) != excluded["sha256"]:
                    raise ValueError(f"Changed explicitly excluded source file: {path}")
    overlap_metadata(manifest)
    source_counts = dict(sorted(Counter(record["source"] for record in records).items()))
    split_counts = dict(sorted(Counter(f"{record['dataset']}/{record['split']}" for record in records).items()))
    if manifest["source_counts"] != source_counts or manifest["split_counts"] != split_counts:
        raise ValueError("Manifest count metadata disagrees with records")
    validate_training_protocol(manifest, manifest.get("training_protocol", "holdout"))
    return root


def load_manifest(path, verify_files=True, data_root=None):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    root = validate_manifest(manifest, verify_files=verify_files, data_root=data_root)
    return manifest, root


def geometry_protocol(policy="strict"):
    if policy not in GEOMETRY_POLICIES:
        raise ValueError(f"Unknown geometry policy: {policy}")
    return {"version": 1, "policy": policy, "prediction_grid": "original GT mask height/width",
            "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
            "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
            "alignment": "full-image normalized coordinates; registration is not verified"}


def _validate_geometry_policy(split, policy):
    geometry_protocol(policy)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if policy != "strict" and split != "test":
        raise ValueError("resize-prediction-to-gt is allowed only for test; train/val remain strict")


def audit_geometry(manifest, root, split="test", policy="strict"):
    """Inspect every selected pair before inference; never modify input files.

    A size difference is recorded, not interpreted as a verified registration.
    The opt-in policy uses the evaluator's existing prediction-to-GT mapping.
    """
    from PIL import Image

    _validate_geometry_policy(split, policy)
    root = Path(root)
    counts, mismatches = Counter(), []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        sizes = {}
        for kind in ("image", "mask"):
            with Image.open(root / record[kind]) as handle:
                sizes[kind] = list(handle.size)
                handle.verify()
        counts[record["dataset"]] += 1
        if sizes["image"] != sizes["mask"]:
            mismatches.append({key: record[key] for key in ("id", "dataset", "image", "mask")}
                              | {"image_size": sizes["image"], "mask_size": sizes["mask"]})
    if not counts:
        raise ValueError(f"No samples for split {split}")
    report = {"policy": policy, "split": split, "checked_pairs": sum(counts.values()),
              "dataset_counts": dict(sorted(counts.items())), "mismatch_count": len(mismatches),
              "mismatches": mismatches, "manifest_sha256": manifest.get("content_sha256"),
              "protocol": geometry_protocol(policy)}
    if mismatches and policy == "strict":
        first = mismatches[0]
        hint = (" For test-only prediction-to-original-GT evaluation, explicitly select "
                "--test-geometry resize-prediction-to-gt and use a fresh output directory."
                if split == "test" else "")
        raise ValueError(f"Image/mask geometry mismatch for {first['id']}: "
                         f"{tuple(first['image_size'])} != {tuple(first['mask_size'])}; "
                         f"{len(mismatches)} mismatched {split} pairs.{hint}")
    return report


class CODDataset:
    """Lazy-import dataset: deterministic augmentation keyed by seed/epoch/sample ID."""

    def __init__(self, manifest, root, split, size=384, seed=0, geometry_policy="strict"):
        _validate_geometry_policy(split, geometry_policy)
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise ValueError(f"No samples for split {split}")
        self.root, self.split = Path(root), split
        self.size, self.seed, self.epoch = int(size), int(seed), 0
        self.geometry_policy = geometry_policy

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image, ImageOps

        record = self.records[index]
        with Image.open(self.root / record["image"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / record["mask"]) as handle:
            mask = handle.convert("L")
        if image.size != mask.size and self.geometry_policy == "strict":
            raise ValueError(f"Image/mask geometry mismatch for {record['id']}: {image.size} != {mask.size}")
        if self.split == "train":
            rng = random.Random(f"{self.seed}:{self.epoch}:{record['id']}")
            if rng.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if rng.random() < 0.2:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1).div_(255)
        if self.split != "train":
            # Evaluator loads original mask resolution, avoiding variable-size collation.
            return {"image": image_tensor, "mask_path": str(self.root / record["mask"]),
                    "dataset": record["dataset"], "id": record["id"]}
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        expanded = mask_tensor.unsqueeze(0)
        # 5x5 morphological gradient after mask resize; max-pool padding ignores outside.
        dilated = F.max_pool2d(expanded, 5, 1, 2)
        eroded = -F.max_pool2d(-expanded, 5, 1, 2)
        edge = (dilated - eroded > 0).float().squeeze(0)
        return {"image": image_tensor, "mask": mask_tensor, "edge": edge, "id": record["id"]}
