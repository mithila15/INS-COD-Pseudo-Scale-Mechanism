"""Exact observed-copy acknowledgement; tiny files only, never optimize/download."""
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from fsa_study import benchmark, overlap
from fsa_study.benchmark import extend_test_manifest, validate_test_extension
from fsa_study.data import SOURCES, load_manifest, manifest_digest, prepare_manifest, sha256_file, validate_manifest
from fsa_study.overlap import (ACKNOWLEDGED_POLICY, KNOWN_COD10K_OVERLAPS,
                               acknowledge_cod10k_overlap, overlap_metadata)
from fsa_study.training_protocol import prepare_full_training_manifest


def observed_records():
    records = []
    for index, (train, test, digest) in enumerate(KNOWN_COD10K_OVERLAPS):
        for source, stem, split in (("cod10k_train", train, "val" if index == 5 else "train"),
                                    ("cod10k_test", test, "test")):
            records.append({"id": f"{source}/{stem}", "source": source, "dataset": "cod10k",
                            "split": split, "image": f"{source}/Image/{stem}.jpg",
                            "mask": f"{source}/GT_Object/{stem}.png", "image_sha256": digest,
                            "mask_sha256": "0" * 64})
    return {"data_root": "/unmounted/cod10k-overlap-fixture", "records": records}


def test_exact_seven_observations_require_acknowledgement():
    manifest = observed_records()
    with pytest.raises(ValueError, match="strict policy refuses"):
        overlap_metadata(manifest)
    audit = acknowledge_cod10k_overlap(manifest)
    assert audit["policy"] == ACKNOWLEDGED_POLICY
    assert audit["disjoint"] is False
    assert (audit["train_test_count"], audit["val_test_count"], audit["affected_test_count"]) == (6, 1, 7)
    assert len({group["image_sha256"] for group in audit["groups"]}) == 7
    assert overlap_metadata(manifest) == audit


def test_policy_constants_match_recorded_pointer_evidence():
    path = Path(__file__).resolve().parents[1] / "results/verification-2026-11-01-overlap/upstream-evidence.json"
    evidence = json.loads(path.read_text())
    observed = {(pair["train"]["name"], pair["test"]["name"], pair["train"]["sha256"])
                for pair in evidence["pairs"]}
    assert observed == set(KNOWN_COD10K_OVERLAPS)
    assert all(pair["train"]["sha256"] == pair["test"]["sha256"] for pair in evidence["pairs"])
    assert evidence["mirror_is_official_cod10k_distribution"] is False


def test_strict_disjoint_metadata_is_computed_without_mutating_manifest():
    manifest = observed_records()
    manifest["records"] = [r for r in manifest["records"] if r["split"] != "test"]
    before = copy.deepcopy(manifest)
    audit = overlap_metadata(manifest)
    assert manifest == before
    assert audit == {"schema_version": 1, "policy": "strict", "disjoint": True,
                     "train_test_count": 0, "val_test_count": 0, "affected_test_count": 0, "groups": []}


@pytest.mark.parametrize("mutation", ["hash", "id", "source", "dataset", "image_stem", "mask_stem",
                                     "third_record", "train_val", "same_path"])
def test_acknowledgement_cannot_exempt_other_overlaps(mutation):
    manifest = observed_records()
    train, test = manifest["records"][:2]
    if mutation == "hash":
        train["image_sha256"] = test["image_sha256"] = "f" * 64
    elif mutation == "id":
        test["id"] += "-other"
    elif mutation == "source":
        test["source"] = "camo_test"
    elif mutation == "dataset":
        test["dataset"] = "camo"
    elif mutation == "image_stem":
        test["image"] = "other/Image/different.jpg"
    elif mutation == "mask_stem":
        test["mask"] = "other/GT/different.png"
    elif mutation == "third_record":
        third = dict(train, id="cod10k_train/third", image="third.jpg")
        manifest["records"].append(third)
    elif mutation == "train_val":
        test["split"] = "val"
    else:
        test["image"] = train["image"]
    with pytest.raises(ValueError, match="Unacknowledged|Same image path"):
        acknowledge_cod10k_overlap(manifest)


@pytest.mark.parametrize("mutation", ["missing", "count", "group", "policy", "float_count", "bool_count"])
def test_audit_must_be_present_exact_and_revalidated(mutation):
    manifest = observed_records()
    acknowledge_cod10k_overlap(manifest)
    if mutation == "missing":
        manifest.pop("overlap_audit")
    elif mutation == "count":
        manifest["overlap_audit"]["train_test_count"] = 0
    elif mutation == "group":
        manifest["overlap_audit"]["groups"].pop()
    elif mutation == "policy":
        manifest["overlap_policy"] = "allow-anything"
    elif mutation == "float_count":
        manifest["overlap_audit"]["train_test_count"] = 6.0
    else:
        manifest["overlap_audit"]["val_test_count"] = True
    with pytest.raises(ValueError, match="Overlap audit|Unknown overlap policy"):
        overlap_metadata(manifest)


def test_policy_cannot_be_removed_while_retaining_audit():
    manifest = observed_records()
    manifest["records"] = [r for r in manifest["records"] if r["split"] != "test"]
    acknowledge_cod10k_overlap(manifest)
    manifest.pop("overlap_policy")
    with pytest.raises(ValueError, match="requires the explicit"):
        overlap_metadata(manifest)


@pytest.fixture
def files(tmp_path, monkeypatch):
    root = tmp_path / "data"
    known = []
    for index, (train, test, _) in enumerate(KNOWN_COD10K_OVERLAPS):
        for source, stem in (("cod10k_train", train), ("cod10k_test", test)):
            _, _, images, masks, _ = SOURCES[source]
            for folder, suffix, contents in ((images, ".jpg", f"shared-pair-{index}"),
                                              (masks, ".png", f"mask-{source}-{index}")):
                path = root / folder / (stem + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
        known.append((train, test, sha256_file(root / SOURCES["cod10k_train"][2] / (train + ".jpg"))))
    # The unit tests above exercise the real hard-coded hashes. These tiny file
    # integration tests use only test-local hashes, never edit production policy.
    monkeypatch.setattr(overlap, "KNOWN_COD10K_OVERLAPS", tuple(known))
    monkeypatch.setattr(benchmark, "EXPECTED_TEST_COUNTS", {"camo": 1, "chameleon": 1, "cod10k": 7, "nc4k": 1})
    for source, (_, split, images, masks, _) in SOURCES.items():
        if source.startswith("cod10k"):
            continue
        for index in range(2 if split == "train" else 1):
            stem = f"{source}-{index}"
            for folder, kind in ((images, "image"), (masks, "mask")):
                path = root / folder / (stem + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{kind}-{source}-{index}")
    source = tmp_path / "original.json"
    original = prepare_manifest(root, source, allow_nonstandard_counts=True,
                                test_datasets=["camo", "chameleon", "nc4k"])
    for record in original["records"]:
        if record["source"] == "cod10k_train":
            record["split"] = "val" if record["id"].endswith(known[5][0]) else "train"
    original["split_counts"] = dict(sorted(Counter(f"{r['dataset']}/{r['split']}" for r in original["records"]).items()))
    original["content_sha256"] = manifest_digest(original)
    source.write_text(json.dumps(original))
    return root, source, original, tmp_path / "extended.json"


def _write_manifest(path, manifest):
    manifest["source_counts"] = dict(sorted(Counter(r["source"] for r in manifest["records"]).items()))
    manifest["split_counts"] = dict(sorted(Counter(f"{r['dataset']}/{r['split']}" for r in manifest["records"]).items()))
    manifest["content_sha256"] = manifest_digest(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("same_path", [False, True])
@pytest.mark.parametrize("count", [2, 3])
def test_same_split_test_duplicates_preserve_historical_metadata(acknowledged, same_path, count):
    records = []
    for index, dataset in enumerate(("camo", "chameleon", "nc4k")[:count]):
        records.append({"id": f"{dataset}_test/{index}", "source": f"{dataset}_test",
                        "dataset": dataset, "split": "test",
                        "image": "shared.jpg" if same_path else f"{dataset}/{index}.jpg",
                        "mask": f"{dataset}/{index}.png", "image_sha256": "f" * 64,
                        "mask_sha256": "0" * 64})
    manifest = {"data_root": "/unmounted/same-split-fixture", "records": records}
    if acknowledged:
        acknowledge_cod10k_overlap(manifest)
    before = copy.deepcopy(manifest)
    audit = overlap_metadata(manifest)
    assert manifest == before
    assert audit == {"schema_version": 1,
                     "policy": ACKNOWLEDGED_POLICY if acknowledged else "strict",
                     "disjoint": True, "train_test_count": 0, "val_test_count": 0,
                     "affected_test_count": 0, "groups": []}


@pytest.mark.parametrize("same_path", [False, True])
def test_extension_preserves_existing_same_split_duplicates(files, same_path):
    root, source, original, output = files
    camo = next(r for r in original["records"] if r["source"] == "camo_test")
    nc4k = next(r for r in original["records"] if r["source"] == "nc4k_test")
    if same_path:
        camo["image"] = nc4k["image"]
    else:
        (root / camo["image"]).write_bytes((root / nc4k["image"]).read_bytes())
    camo["image_sha256"] = nc4k["image_sha256"]
    _write_manifest(source, original)
    source_bytes = source.read_bytes()
    file_hashes = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}

    assert load_manifest(source)[0] == original
    benchmark._validate_snapshot(original, root)
    extended = extend_test_manifest(source, output, acknowledge_cod10k=True)
    assert load_manifest(output)[0] == extended
    benchmark._validate_snapshot(extended, root)
    proof = validate_test_extension(original, root, extended, root, verify_files=False)
    assert proof["overlap_audit"]["affected_test_count"] == 7
    assert (proof["overlap_audit"]["train_test_count"], proof["overlap_audit"]["val_test_count"]) == (6, 1)
    assert extended["records"][:len(original["records"])] == original["records"]
    full_path = output.with_name("full-with-test-duplicates.json")
    full = prepare_full_training_manifest(output, full_path)
    assert load_manifest(full_path)[0] == full
    benchmark._validate_snapshot(full, root)
    assert full["full_training_source"]["source_overlap_audit"] == extended["overlap_audit"]
    assert (full["overlap_audit"]["train_test_count"], full["overlap_audit"]["val_test_count"]) == (7, 0)
    assert [r for r in full["records"] if r["split"] == "test"] == [
        r for r in extended["records"] if r["split"] == "test"]
    assert source.read_bytes() == source_bytes
    assert {p: sha256_file(p) for p in file_hashes} == file_hashes


def test_extension_accepts_internal_new_cod10k_test_duplicates(files, monkeypatch):
    root, source, original, output = files
    _, _, images, masks, _ = SOURCES["cod10k_test"]
    for index in range(2):
        stem = f"COD10K-CAM-test-only-duplicate-{index}"
        (root / images / f"{stem}.jpg").write_text("new-test-only-shared-image")
        (root / masks / f"{stem}.png").write_text(f"new-test-only-mask-{index}")
    monkeypatch.setattr(benchmark, "EXPECTED_TEST_COUNTS", {"camo": 1, "chameleon": 1, "cod10k": 9, "nc4k": 1})
    source_bytes = source.read_bytes()
    extended = extend_test_manifest(source, output, acknowledge_cod10k=True)
    assert load_manifest(output)[0] == extended
    proof = validate_test_extension(original, root, extended, root, verify_files=False)
    assert proof["benchmark_coverage"]["complete"]
    assert proof["overlap_audit"]["affected_test_count"] == 7
    assert extended["source_counts"]["cod10k_test"] == 9
    assert source.read_bytes() == source_bytes


def test_train_source_relabelled_test_still_fails_source_split_validation(files):
    root, source, _, output = files
    extended = extend_test_manifest(source, output, acknowledge_cod10k=True)
    train = next(r for r in extended["records"] if r["source"] == "cod10k_train")
    train["split"] = "test"
    acknowledge_cod10k_overlap(extended)
    _write_manifest(output, extended)
    with pytest.raises(ValueError, match="source separation"):
        validate_manifest(extended)
    with pytest.raises(ValueError, match="source/split separation"):
        benchmark._validate_snapshot(extended, root)


def test_default_extension_refuses_known_copies_without_writing(files):
    root, source, original, output = files
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    source_hash = sha256_file(source)
    with pytest.raises(ValueError, match="strict policy refuses"):
        extend_test_manifest(source, output)
    assert not output.exists()
    assert sha256_file(source) == source_hash
    assert {p: sha256_file(p) for p in before} == before
    assert "overlap_policy" not in original and "overlap_audit" not in original


def test_extension_preserves_original_and_revalidates_snapshot(files):
    root, source, original, output = files
    source_bytes = source.read_bytes()
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    extended = extend_test_manifest(source, output, acknowledge_cod10k=True)
    assert source.read_bytes() == source_bytes
    assert {p: sha256_file(p) for p in before} == before
    assert extended["records"][:len(original["records"])] == original["records"]
    assert load_manifest(output)[0] == extended
    proof = validate_test_extension(original, root, extended, root, verify_files=False)
    assert proof["benchmark_coverage"]["complete"]
    assert proof["overlap_audit"]["affected_test_count"] == 7
    assert not proof["overlap_audit"]["disjoint"]
    tampered = copy.deepcopy(extended)
    tampered["overlap_audit"]["disjoint"] = True
    tampered["content_sha256"] = manifest_digest(tampered)
    with pytest.raises(ValueError, match="Overlap audit"):
        validate_test_extension(original, root, tampered, root, verify_files=False)


def test_full_training_promotes_val_and_preserves_overlap_source_evidence(files):
    root, source, original, output = files
    extended = extend_test_manifest(source, output, acknowledge_cod10k=True)
    old_bytes = output.read_bytes()
    full = prepare_full_training_manifest(output, output.with_name("full.json"))
    assert output.read_bytes() == old_bytes
    assert full["full_training_source"]["source_overlap_audit"] == extended["overlap_audit"]
    audit = overlap_metadata(full)
    assert (audit["train_test_count"], audit["val_test_count"], audit["affected_test_count"]) == (7, 0, 7)
    assert [r for r in full["records"] if r["split"] == "test"] == [r for r in extended["records"] if r["split"] == "test"]
    assert load_manifest(output.with_name("full.json"))[0] == full


def test_prepare_flag_preserves_all_pairs_and_strict_default(files):
    root, _, _, output = files
    with pytest.raises(ValueError, match="strict policy refuses"):
        prepare_manifest(root, output, allow_nonstandard_counts=True)
    manifest = prepare_manifest(root, output, allow_nonstandard_counts=True, acknowledge_cod10k=True)
    assert manifest["source_counts"]["cod10k_train"] == 7
    assert manifest["source_counts"]["cod10k_test"] == 7
    assert overlap_metadata(manifest)["affected_test_count"] == 7


def test_prepare_parser_exposes_explicit_flag_only_on_prepare():
    from run_study import build_parser
    parser = build_parser()
    args = parser.parse_args(["prepare", "--data-root", "/data", "--manifest", "/manifest.json",
                              "--acknowledge-cod10k-overlap"])
    assert args.acknowledge_cod10k_overlap is True
    strict = parser.parse_args(["prepare", "--data-root", "/data", "--manifest", "/manifest.json"])
    assert strict.acknowledge_cod10k_overlap is False
