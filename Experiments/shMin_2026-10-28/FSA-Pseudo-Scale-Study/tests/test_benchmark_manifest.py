"""Audit additive benchmark manifests with tiny local files; never train/download."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from fsa_study import benchmark
from fsa_study.benchmark import benchmark_coverage, extend_test_manifest, validate_test_extension
from fsa_study.data import SOURCES, load_manifest, manifest_digest, prepare_manifest, sha256_file


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "EXPECTED_TEST_COUNTS", {name: 2 for name in benchmark.EXPECTED_TEST_COUNTS})
    root = tmp_path / "data"
    for source, (name, split, images, masks, _) in SOURCES.items():
        for index in range(4 if split == "train" else 2):
            stem = ("COD10K-CAM-" if name == "cod10k" else "") + f"{source}-{index}"
            for kind, folder in (("image", images), ("mask", masks)):
                path = root / folder / f"{stem}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{kind}/{source}/{index}".encode())
    excluded = root / "COD10K-v3/Train/Image/COD10K-NonCAM-background.png"
    excluded.write_bytes(b"excluded-training-background")
    original_path = tmp_path / "training-manifest.json"
    original = prepare_manifest(root, original_path, allow_nonstandard_counts=True,
                                test_datasets=["camo", "chameleon", "nc4k"])
    return original_path, original, root, tmp_path / "evaluation-manifest.json"


def _rehash(manifest):
    manifest["content_sha256"] = manifest_digest(manifest)
    return manifest


def test_standard_benchmark_coverage():
    counts = {"camo": 250, "chameleon": 76, "cod10k": 2026, "nc4k": 4121}
    result = benchmark_coverage(counts)
    assert result["complete"]
    assert result["expected_counts"] == counts
    assert result["observed_counts"] == counts
    assert not result["missing_datasets"]
    assert not result["nonstandard_counts"]


def test_missing_cod10k_is_not_complete():
    result = benchmark_coverage({"camo": 250, "chameleon": 76, "nc4k": 4121})
    assert not result["complete"]
    assert result["missing_datasets"] == ["cod10k"]


def test_nonstandard_counts_unexpected_and_zero_are_explicit():
    result = benchmark_coverage({"camo": 100, "chameleon": 0, "cod10k": 2026, "nc4k": 4121, "extra": 1})
    assert not result["complete"]
    assert result["missing_datasets"] == ["chameleon"]
    assert result["unexpected_datasets"] == ["extra"]
    assert result["nonstandard_counts"] == {"camo": {"expected": 250, "observed": 100}}
    with pytest.raises(ValueError, match="negative"):
        benchmark_coverage({"camo": -1})


@pytest.mark.parametrize("count", [250.9, 250.0, "250", True, None])
def test_counts_cannot_be_coerced_to_standard_integers(count):
    with pytest.raises(ValueError, match="integers"):
        benchmark_coverage({"camo": count})


def test_addition_preserves_original_manifest_records_and_files(dataset):
    path, original, root, target = dataset
    before = {file: sha256_file(file) for file in root.rglob("*") if file.is_file()}
    original_hash = sha256_file(path)
    extended = extend_test_manifest(path, target)
    assert sha256_file(path) == original_hash
    assert before == {file: sha256_file(file) for file in root.rglob("*") if file.is_file()}
    assert extended["records"][:len(original["records"])] == original["records"]
    assert extended["split_seed"] == original["split_seed"]
    assert extended["selection_policy"] == original["selection_policy"]
    assert extended["split_counts"]["cod10k/test"] == 2
    assert extended["test_extension"]["training_manifest_file_sha256"] == original_hash
    proof = validate_test_extension(original, root, extended, root)
    assert proof["schema_version"] == 1
    assert proof["added_records"] == 2
    assert proof["original_record_count"] == len(original["records"])
    assert proof["benchmark_coverage"]["complete"]
    assert load_manifest(target)[0] == extended


def test_background_files_are_excluded_but_hashed(dataset):
    path, _, root, target = dataset
    folder = root / "COD10K-v3/Test/Image"
    background = folder / "COD10K-NonCAM-background.png"
    background.write_bytes(b"not-foreground")
    extended = extend_test_manifest(path, target)
    assert extended["raw_source_counts"]["cod10k_test"] == {"images": 3, "masks": 2}
    assert extended["excluded_files"]["cod10k_test"]["images"] == [
        {"path": background.relative_to(root).as_posix(), "sha256": sha256_file(background)}]
    assert not any("NonCAM" in record["id"] for record in extended["records"])
    background.write_bytes(b"changed-background")
    with pytest.raises(ValueError, match="Changed explicitly excluded"):
        load_manifest(target)


def test_external_directories_lift_root_and_rebase_original_exclusions(dataset):
    path, original, root, target = dataset
    external = root.parent / "separate-download"
    shutil.move(str(root / "COD10K-v3/Test"), external)
    extended = extend_test_manifest(path, target, external / "Image", external / "GT_Object")
    new_root = Path(extended["data_root"])
    assert new_root == root.parent
    assert extended["records"][0]["image"] == "data/" + original["records"][0]["image"]
    excluded = extended["excluded_files"]["cod10k_train"]["images"][0]
    assert excluded["path"].startswith("data/")
    assert validate_test_extension(original, root, extended, new_root)["benchmark_coverage"]["complete"]


def test_portable_proof_can_validate_with_unmounted_datasets(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    proof = validate_test_extension(original, root, extended, root)
    shutil.rmtree(root)
    assert validate_test_extension(original, root, extended, root, verify_files=False) == proof
    with pytest.raises(FileNotFoundError):
        validate_test_extension(original, root, extended, root)


def test_rebased_portable_proof_can_validate_without_files(dataset):
    path, original, root, target = dataset
    external = root.parent / "separate-download"
    shutil.move(str(root / "COD10K-v3/Test"), external)
    extended = extend_test_manifest(path, target, external / "Image", external / "GT_Object")
    proof = validate_test_extension(original, root, extended, Path(extended["data_root"]))
    shutil.rmtree(root)
    shutil.rmtree(external)
    assert validate_test_extension(original, root, extended, extended["data_root"], verify_files=False) == proof


def test_does_not_overwrite_evaluation_manifest(dataset):
    path, _, _, target = dataset
    target.write_text("existing artifact")
    with pytest.raises(FileExistsError):
        extend_test_manifest(path, target)
    assert target.read_text() == "existing artifact"
    with pytest.raises(FileExistsError):
        extend_test_manifest(path, path)


@pytest.mark.parametrize("directory,stem", [("Image", "COD10K-CAM-cod10k_test-0"), ("GT_Object", "COD10K-CAM-cod10k_test-0")])
def test_unpaired_foreground_is_not_silently_dropped(dataset, directory, stem):
    path, _, root, target = dataset
    (root / f"COD10K-v3/Test/{directory}/{stem}.png").unlink()
    with pytest.raises(ValueError, match="pairing mismatch"):
        extend_test_manifest(path, target)
    assert not target.exists()


def test_missing_pairs_fail_even_if_training_allowed_nonstandard_counts(dataset):
    path, _, root, target = dataset
    for name in ("Image", "GT_Object"):
        (root / f"COD10K-v3/Test/{name}/COD10K-CAM-cod10k_test-0.png").unlink()
    with pytest.raises(ValueError, match="requires 2 foreground pairs; found 1"):
        extend_test_manifest(path, target)


def test_ambiguous_autodiscovery_requires_explicit_paths(dataset):
    path, _, root, target = dataset
    shutil.copytree(root / "COD10K-v3/Test", root / "COD10K")
    with pytest.raises(ValueError, match="Ambiguous"):
        extend_test_manifest(path, target)
    assert extend_test_manifest(path, target, "COD10K/Image", "COD10K/GT_Object")


def test_missing_autodiscovery_gives_explicit_directory_hint(dataset):
    path, _, root, target = dataset
    shutil.rmtree(root / "COD10K-v3/Test")
    with pytest.raises(FileNotFoundError, match="cod10k-images"):
        extend_test_manifest(path, target)


def test_only_one_explicit_directory_rejected(dataset):
    path, _, root, target = dataset
    with pytest.raises(ValueError, match="both"):
        extend_test_manifest(path, target, cod10k_images=root / "COD10K-v3/Test/Image")


def test_duplicate_stem_rejected(dataset):
    path, _, root, target = dataset
    (root / "COD10K-v3/Test/Image/COD10K-CAM-cod10k_test-0.jpg").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        extend_test_manifest(path, target)


def test_training_duplicate_image_hash_rejected(dataset):
    path, original, root, target = dataset
    old_image = root / next(record["image"] for record in original["records"] if record["split"] == "train")
    (root / "COD10K-v3/Test/Image/COD10K-CAM-cod10k_test-0.png").write_bytes(old_image.read_bytes())
    with pytest.raises(ValueError, match="hashes cross"):
        extend_test_manifest(path, target)


def test_existing_test_duplicate_image_hash_rejected(dataset):
    path, original, root, target = dataset
    old_image = root / next(record["image"] for record in original["records"] if record["split"] == "test")
    (root / "COD10K-v3/Test/Image/COD10K-CAM-cod10k_test-0.png").write_bytes(old_image.read_bytes())
    with pytest.raises(ValueError, match="duplicates an original"):
        extend_test_manifest(path, target)


@pytest.mark.parametrize("field,value", [("split_seed", 999), ("val_fraction", 0.2), ("selection_policy", {"cod10k_prefix": ""})])
def test_changed_split_or_selection_metadata_rejected(dataset, field, value):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    extended[field] = value
    with pytest.raises(ValueError, match="changed original manifest metadata"):
        validate_test_extension(original, root, _rehash(extended), root, verify_files=False)


def test_changed_original_record_metadata_rejected(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    extended["records"][0]["unapproved_annotation"] = "changed"
    with pytest.raises(ValueError, match="original records"):
        validate_test_extension(original, root, _rehash(extended), root, verify_files=False)


def test_same_hash_copy_is_not_same_original_file(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    image = root / extended["records"][0]["image"]
    duplicate = image.with_name("different-file.png")
    duplicate.write_bytes(image.read_bytes())
    extended["records"][0]["image"] = duplicate.relative_to(root).as_posix()
    with pytest.raises(ValueError, match="original records"):
        validate_test_extension(original, root, _rehash(extended), root)


def test_changed_original_audit_exclusions_rejected(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    extended["excluded_files"]["cod10k_train"]["images"] = []
    with pytest.raises(ValueError, match="audit metadata"):
        validate_test_extension(original, root, _rehash(extended), root, verify_files=False)


def test_changed_provenance_rejected(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    extended["test_extension"]["training_manifest_sha256"] = "bad"
    with pytest.raises(ValueError, match="provenance"):
        validate_test_extension(original, root, _rehash(extended), root, verify_files=False)


def test_existing_full_manifest_can_be_reused_with_identity_records(dataset):
    _, _, root, target = dataset
    complete_path = target.with_name("complete-original.json")
    complete = prepare_manifest(root, complete_path, allow_nonstandard_counts=True)
    extended = extend_test_manifest(complete_path, target)
    assert extended["records"] == complete["records"]
    assert validate_test_extension(complete, root, extended, root)["added_records"] == 0
    with pytest.raises(ValueError, match="already present"):
        extend_test_manifest(complete_path, target.with_name("replace.json"), root / "COD10K-v3/Test/Image", root / "COD10K-v3/Test/GT_Object")


def test_full_training_protocol_can_extend_tests_without_creating_validation(dataset):
    _, _, root, target = dataset
    full_path = target.with_name("full-training-original.json")
    full = prepare_manifest(root, full_path, allow_nonstandard_counts=True,
                            test_datasets=["camo", "chameleon", "nc4k"],
                            training_protocol="full_train", val_fraction=0)
    extended = extend_test_manifest(full_path, target)
    assert extended["training_protocol"] == "full_train"
    assert extended["val_fraction"] == 0
    assert not any(record["split"] == "val" for record in extended["records"])
    assert extended["records"][:len(full["records"])] == full["records"]
    assert validate_test_extension(full, root, extended, root)["benchmark_coverage"]["complete"]
    shutil.rmtree(root)
    assert validate_test_extension(full, root, extended, root, verify_files=False)["benchmark_coverage"]["complete"]


def test_other_missing_dataset_cannot_be_mislabeled_complete(dataset):
    _, _, root, target = dataset
    path = target.with_name("only-camo.json")
    prepare_manifest(root, path, allow_nonstandard_counts=True, test_datasets=["camo"])
    with pytest.raises(ValueError, match="other test datasets are missing"):
        extend_test_manifest(path, target)


def test_snapshot_checksum_count_metadata_and_path_escape_checked(dataset):
    path, original, root, target = dataset
    extended = extend_test_manifest(path, target)
    bad = copy.deepcopy(extended)
    bad["content_sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        validate_test_extension(original, root, bad, root, verify_files=False)
    bad = copy.deepcopy(extended)
    bad["split_counts"]["camo/train"] += 1
    with pytest.raises(ValueError, match="count metadata"):
        validate_test_extension(original, root, _rehash(bad), root, verify_files=False)
    bad = copy.deepcopy(extended)
    bad["records"][-1]["image"] = "../escape.png"
    with pytest.raises(ValueError, match="escapes"):
        validate_test_extension(original, root, _rehash(bad), root, verify_files=False)
