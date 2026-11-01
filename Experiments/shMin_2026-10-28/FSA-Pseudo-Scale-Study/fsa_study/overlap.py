"""Narrow, explicit acknowledgement of seven observed COD10K byte duplicates.

This is not a claim that the downloaded files form a clean official split.
Only the exact IDs and bytes observed in the 2026-11-01 server audit qualify.
The default remains strict and historical manifest serialization is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


ACKNOWLEDGED_POLICY = "acknowledged-cod10k-v1"
# (training stem, test stem, SHA-256 of both image files). Mask contents are
# independently hashed by the ordinary manifest validator, never exempted.
KNOWN_COD10K_OVERLAPS = (
    ("COD10K-CAM-1-Aquatic-3-Crab-32", "COD10K-CAM-1-Aquatic-3-Crab-32",
     "e60fced9bfb2eeb8df4b7b9dcce931d6537b7f4f8ff4e49296e4bd763897d27a"),
    ("COD10K-CAM-2-Terrestrial-23-Cat-1506", "COD10K-CAM-2-Terrestrial-23-Cat-1506",
     "311809538532626c3c07e95076c32130b7010d16cf5ae6e6d5624819ab1d70d0"),
    ("COD10K-CAM-2-Terrestrial-28-Deer-1796", "COD10K-CAM-2-Terrestrial-28-Deer-1762",
     "58a9a7b96f9f5643139d33d73331c6f1851b599df7fa60a916e451ef01ca3afe"),
    ("COD10K-CAM-2-Terrestrial-26-Chameleon-1694", "COD10K-CAM-2-Terrestrial-31-Gecko-1892",
     "1be584edf381ee65c4e802dedefb671742122d6db3f00af3b204d6b4754309d7"),
    ("COD10K-CAM-2-Terrestrial-31-Gecko-1928", "COD10K-CAM-2-Terrestrial-31-Gecko-1895",
     "e5f0ea8dec16c4b4ed178c58a9827cd946fa9e0e860e1d56ae769b50d59e6c4e"),
    ("COD10K-CAM-2-Terrestrial-32-Giraffe-1930", "COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
     "6988e9b908fb9a6c8157405f49dcd37586ef14f08d6374e33996c780dcbb67e0"),
    ("COD10K-CAM-3-Flying-53-Bird-3205", "COD10K-CAM-3-Flying-65-Owl-4633",
     "6368c9991b9cc8f2d45eb87bb172e2611154efaaa99113fc1fab0d1943276390"),
)


def _computed_metadata(manifest):
    policy = manifest.get("overlap_policy", "strict")
    if policy not in {"strict", ACKNOWLEDGED_POLICY}:
        raise ValueError(f"Unknown overlap policy: {policy!r}")
    known = {digest: (train, test) for train, test, digest in KNOWN_COD10K_OVERLAPS}
    hashes = defaultdict(list)
    paths = defaultdict(list)
    for record in manifest["records"]:
        hashes[record["image_sha256"]].append(record)
        path = (Path(manifest.get("data_root", "/")) / record["image"]).resolve()
        paths[str(path)].append(record)
    for path, records in paths.items():
        if len(records) > 1 and (policy == ACKNOWLEDGED_POLICY or len({r["split"] for r in records}) > 1):
            raise ValueError(f"Same image path crosses splits or aliases acknowledged records: {path}")
    groups = []
    for digest, records in sorted(hashes.items()):
        splits = {record["split"] for record in records}
        if len(splits) < 2 and not (policy == ACKNOWLEDGED_POLICY and len(records) > 1 and "test" in splits):
            # Retain historical same-split semantics. Added evaluation records
            # are additionally checked against every original image below.
            continue
        if policy == "strict":
            raise ValueError("Identical image byte hashes cross train/val/test splits; "
                             "strict policy refuses overlap. Audit the input files first.")
        expected = known.get(digest)
        if expected is None or len(records) != 2:
            raise ValueError("Unacknowledged image overlap: unknown hash or a group with more than two records")
        train_stem, test_stem = expected
        train_id, test_id = f"cod10k_train/{train_stem}", f"cod10k_test/{test_stem}"
        indexed = {record["id"]: record for record in records}
        if set(indexed) != {train_id, test_id}:
            raise ValueError("Unacknowledged image overlap: IDs differ from the exact observed COD10K pair")
        train, test = indexed[train_id], indexed[test_id]
        for record, source, stem in ((train, "cod10k_train", train_stem), (test, "cod10k_test", test_stem)):
            if (record["source"] != source or record["dataset"] != "cod10k"
                    or Path(record["image"]).stem != stem or Path(record["mask"]).stem != stem):
                raise ValueError("Unacknowledged image overlap: source, dataset or canonical file stem differs")
        if train["split"] not in {"train", "val"} or test["split"] != "test":
            raise ValueError("Unacknowledged image overlap: only observed train/val versus test pairs qualify")
        groups.append({"image_sha256": digest, "train_id": train_id, "train_split": train["split"],
                       "test_id": test_id, "test_split": "test"})
    return {"schema_version": 1, "policy": policy, "disjoint": not groups,
            "train_test_count": sum(group["train_split"] == "train" for group in groups),
            "val_test_count": sum(group["train_split"] == "val" for group in groups),
            "affected_test_count": len(groups), "groups": groups}


def overlap_metadata(manifest):
    """Recompute split integrity and reject stale/forged acknowledgement metadata.

    Historical strict manifests acquire no new fields. Callers can use this
    returned metadata in evaluation reports without modifying training inputs.
    """
    metadata = _computed_metadata(manifest)
    if metadata["policy"] == ACKNOWLEDGED_POLICY:
        if json.dumps(manifest.get("overlap_audit"), sort_keys=True) != json.dumps(metadata, sort_keys=True):
            raise ValueError("Overlap audit disagrees with manifest records; regenerate the acknowledged manifest")
    elif "overlap_audit" in manifest:
        raise ValueError("An overlap audit requires the explicit acknowledged COD10K policy")
    return metadata


def acknowledge_cod10k_overlap(manifest):
    """Preparation-only opt-in; validates exact pairs before recording an audit."""
    manifest["overlap_policy"] = ACKNOWLEDGED_POLICY
    manifest["overlap_audit"] = _computed_metadata(manifest)
    return manifest["overlap_audit"]


def refresh_overlap_audit(manifest):
    """Recompute already acknowledged evidence after deliberate val promotion."""
    if manifest.get("overlap_policy") == ACKNOWLEDGED_POLICY:
        manifest["overlap_audit"] = _computed_metadata(manifest)


def acknowledged_test_duplicate(manifest, record):
    """Whether an added test record belongs to an exactly validated pair."""
    metadata = overlap_metadata(manifest)
    return any(group["test_id"] == record["id"] and group["image_sha256"] == record["image_sha256"]
               for group in metadata["groups"])
