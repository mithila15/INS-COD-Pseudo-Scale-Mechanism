"""Data-only full-training/holdout checks; no model, optimizer or training."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fsa_study.data import load_manifest, manifest_digest, prepare_manifest
from fsa_study.training_protocol import (FULL_TRAIN_SELECTION, HOLDOUT_SELECTION,
                                         prepare_full_training_manifest, resolve_val_fraction, training_metadata,
                                         validate_training_protocol)
from test_data import fixture


def records_manifest(protocol="full_train", camo=1000, cod10k=3040):
    """Record-only validation input: no image generation or dataset download."""
    records = [{"id": f"{dataset}_train/{index}", "dataset": dataset,
                "source": f"{dataset}_train", "split": "train"}
               for dataset, count in (("camo", camo), ("cod10k", cod10k))
               for index in range(count)]
    manifest = {"records": records, "val_fraction": 0.0,
                "allow_nonstandard_counts": False, "training_protocol": protocol}
    if protocol == "holdout":
        manifest.pop("training_protocol")
        manifest["val_fraction"] = .1
        for dataset, count in (("camo", camo), ("cod10k", cod10k)):
            selected = [record for record in records if record["dataset"] == dataset]
            for record in selected[:round(count * .1)]:
                record["split"] = "val"
    return manifest


class TrainingProtocolTests(unittest.TestCase):
    def test_fraction_defaults_and_explicit_mismatch(self):
        self.assertEqual(resolve_val_fraction(), .1)
        self.assertEqual(resolve_val_fraction("full_train"), 0.0)
        self.assertEqual(resolve_val_fraction("holdout", .2), .2)
        for protocol, fraction in (("holdout", 0), ("holdout", 1), ("full_train", .1),
                                   ("full_train", -1), ("full_train", float("nan"))):
            with self.subTest(protocol=protocol, fraction=fraction), self.assertRaises(ValueError):
                resolve_val_fraction(protocol, fraction)
        with self.assertRaisesRegex(ValueError, "Unknown training protocol"):
            resolve_val_fraction("automatic")

    def test_full_4040_records_and_fixed_final_selection(self):
        metadata = training_metadata(records_manifest(), "full_train")
        self.assertEqual(metadata, {"training_protocol": "full_train",
                                   "training_split_counts": {"train": {"camo": 1000, "cod10k": 3040}, "val": {}},
                                   "training_sample_count": 4040, "validation_sample_count": 0,
                                   "checkpoint_selection": FULL_TRAIN_SELECTION})

    def test_historical_holdout_counts_and_selection(self):
        metadata = training_metadata(records_manifest("holdout"))
        self.assertEqual(metadata["training_split_counts"],
                         {"train": {"camo": 900, "cod10k": 2736}, "val": {"camo": 100, "cod10k": 304}})
        self.assertEqual(metadata["training_sample_count"], 3636)
        self.assertEqual(metadata["validation_sample_count"], 404)
        self.assertEqual(metadata["checkpoint_selection"], HOLDOUT_SELECTION)

    def test_modes_cannot_silently_switch(self):
        for manifest, protocol in ((records_manifest(), "holdout"),
                                   (records_manifest("holdout"), "full_train")):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "protocol mismatch"):
                validate_training_protocol(manifest, protocol)

    def test_full_train_rejects_validation_records(self):
        manifest = records_manifest()
        manifest["records"][0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "must not contain validation"):
            validate_training_protocol(manifest, "full_train")

    def test_full_train_requires_official_counts_unless_audited_override(self):
        manifest = records_manifest(camo=4, cod10k=4)
        with self.assertRaisesRegex(ValueError, "4040 training pairs"):
            validate_training_protocol(manifest, "full_train")
        manifest["allow_nonstandard_counts"] = True
        self.assertEqual(training_metadata(manifest, "full_train")["training_sample_count"], 8)
        manifest["records"] = [r for r in manifest["records"] if r["dataset"] == "camo"]
        with self.assertRaisesRegex(ValueError, "both CAMO and COD10K"):
            validate_training_protocol(manifest, "full_train")

    def test_test_images_cannot_be_promoted_to_training(self):
        manifest = records_manifest()
        manifest["records"][0]["source"] = "camo_test"
        with self.assertRaisesRegex(ValueError, "non-training source"):
            validate_training_protocol(manifest, "full_train")

    def test_holdout_requires_validation_and_train(self):
        manifest = records_manifest("holdout")
        manifest["records"] = [r for r in manifest["records"] if r["split"] == "train"]
        with self.assertRaisesRegex(ValueError, "nonempty train and validation"):
            validate_training_protocol(manifest)
        manifest["records"] = []
        with self.assertRaisesRegex(ValueError, "nonempty train set"):
            validate_training_protocol(manifest)

    def test_metadata_uses_records_not_untrusted_count_fields(self):
        manifest = records_manifest()
        manifest["source_counts"] = {"camo_train": 99}
        manifest["split_counts"] = {"cod10k/train": 99}
        self.assertEqual(training_metadata(manifest, "full_train")["training_sample_count"], 4040)


class FullTrainingManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"
        self.root.mkdir()
        fixture(self.root)
        self.path = Path(self.temporary.name) / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_manifest_round_trip_keeps_test_and_all_train_records(self):
        manifest = prepare_manifest(self.root, self.path, val_fraction=0,
                                    allow_nonstandard_counts=True, training_protocol="full_train")
        self.assertEqual(manifest["training_protocol"], "full_train")
        self.assertEqual(manifest["split_counts"], {"camo/train": 4, "camo/test": 2,
                                                  "cod10k/train": 4, "cod10k/test": 2})
        self.assertEqual(load_manifest(self.path)[0], manifest)
        self.assertEqual(training_metadata(manifest, "full_train")["validation_sample_count"], 0)

    def test_old_holdout_split_and_serialization_are_preserved(self):
        manifest = prepare_manifest(self.root, self.path, allow_nonstandard_counts=True)
        self.assertNotIn("training_protocol", manifest)
        self.assertEqual([r["id"] for r in manifest["records"] if r["split"] == "val"],
                         ["cod10k_train/COD10K-CAM-cod10k_train_2", "camo_train/camo_train_0"])
        repeat = prepare_manifest(self.root, self.path.with_name("repeat.json"),
                                  allow_nonstandard_counts=True, training_protocol="holdout")
        self.assertEqual(self.path.read_bytes(), self.path.with_name("repeat.json").read_bytes())
        self.assertEqual(manifest["content_sha256"], repeat["content_sha256"])

    def test_full_manifest_cannot_overwrite_historical_manifest(self):
        prepare_manifest(self.root, self.path, allow_nonstandard_counts=True)
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_manifest(self.root, self.path, val_fraction=0,
                             allow_nonstandard_counts=True, training_protocol="full_train")
        self.assertEqual(self.path.read_bytes(), before)

    def test_full_manifest_still_requires_strict_counts_by_default(self):
        with self.assertRaisesRegex(ValueError, "expected 3040"):
            prepare_manifest(self.root, self.path, val_fraction=0, training_protocol="full_train")
        self.assertFalse(self.path.exists())

    def test_loader_rejects_rehashed_validation_leak_in_full_manifest(self):
        manifest = prepare_manifest(self.root, self.path, val_fraction=0,
                                    allow_nonstandard_counts=True, training_protocol="full_train")
        edited = copy.deepcopy(manifest)
        record = next(r for r in edited["records"] if r["split"] == "train")
        record["split"] = "val"
        key = record["dataset"]
        edited["split_counts"][f"{key}/train"] -= 1
        edited["split_counts"][f"{key}/val"] = 1
        edited["content_sha256"] = manifest_digest(edited)
        self.path.write_text(json.dumps(edited))
        with self.assertRaisesRegex(ValueError, "must not contain validation"):
            load_manifest(self.path)

    def test_test_bytes_remain_disjoint_from_full_training(self):
        source = self.root / "COD10K-v3/Train/Image/COD10K-CAM-cod10k_train_0.png"
        target = self.root / "COD10K-v3/Test/Image/COD10K-CAM-cod10k_test_0.png"
        target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "hashes cross"):
            prepare_manifest(self.root, self.path, val_fraction=0,
                             allow_nonstandard_counts=True, training_protocol="full_train")


class PromoteManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"
        self.root.mkdir()
        fixture(self.root, tests=("camo", "chameleon", "cod10k", "nc4k"))
        self.source = Path(self.temporary.name) / "source.json"
        self.output = Path(self.temporary.name) / "full_train.json"
        self.expected_counts = {name: 2 for name in ("camo", "chameleon", "cod10k", "nc4k")}

    def tearDown(self):
        self.temporary.cleanup()

    def prepare_source(self):
        return prepare_manifest(self.root, self.source, allow_nonstandard_counts=True)

    def promote(self):
        with patch("fsa_study.benchmark.EXPECTED_TEST_COUNTS", self.expected_counts):
            return prepare_full_training_manifest(self.source, self.output)

    def test_promote_only_val_preserve_test_records_source_and_provenance(self):
        original = self.prepare_source()
        original["test_extension"] = {"training_manifest": "/historical/source.json",
                                      "training_manifest_sha256": "historical-hash"}
        original["content_sha256"] = manifest_digest(original)
        self.source.write_text(json.dumps(original))
        before = self.source.read_bytes()
        result = self.promote()
        self.assertEqual(self.source.read_bytes(), before)
        self.assertNotEqual(original["content_sha256"], result["content_sha256"])
        for old, new in zip(original["records"], result["records"], strict=True):
            expected = {**old, "split": "train"} if old["split"] == "val" else old
            self.assertEqual(expected, new)
        self.assertNotIn("test_extension", result)
        provenance = result["full_training_source"]
        self.assertEqual(provenance["source_test_extension"], original["test_extension"])
        self.assertEqual(provenance["source_manifest_sha256"], original["content_sha256"])
        self.assertEqual(provenance["promoted_validation_samples"], 2)
        self.assertEqual(load_manifest(self.output)[0], result)
        self.assertEqual(training_metadata(result, "full_train")["training_sample_count"], 8)

    def test_custom_layout_is_reused_without_canonical_rescan(self):
        original = self.prepare_source()
        old_dir = self.root / "COD10K-v3/Test/Image"
        new_dir = self.root / "custom/external_cod10k/images"
        new_dir.parent.mkdir(parents=True)
        old_dir.rename(new_dir)
        for record in original["records"]:
            if record["source"] == "cod10k_test":
                record["image"] = f"custom/external_cod10k/images/{Path(record['image']).name}"
        original["content_sha256"] = manifest_digest(original)
        self.source.write_text(json.dumps(original))
        result = self.promote()
        self.assertFalse(old_dir.exists())
        self.assertEqual([r for r in original["records"] if r["split"] == "test"],
                         [r for r in result["records"] if r["split"] == "test"])

    def test_requires_all_four_test_datasets_even_with_custom_count_override(self):
        prepare_manifest(self.root, self.source, allow_nonstandard_counts=True,
                         test_datasets=["camo", "chameleon", "nc4k"])
        with self.assertRaisesRegex(ValueError, "complete four-dataset"):
            self.promote()
        self.assertFalse(self.output.exists())

    def test_nonstandard_test_counts_not_implicitly_allowed(self):
        self.prepare_source()
        with self.assertRaisesRegex(ValueError, "complete four-dataset"):
            prepare_full_training_manifest(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_source_files_are_verified_and_output_never_overwritten(self):
        original = self.prepare_source()
        self.output.write_text("keep this file")
        with self.assertRaises(FileExistsError):
            self.promote()
        self.assertEqual(self.output.read_text(), "keep this file")
        self.output.unlink()
        image = self.root / original["records"][0]["image"]
        image.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "Changed image"):
            self.promote()
        self.assertFalse(self.output.exists())

    def test_source_manifest_cannot_be_overwritten(self):
        self.prepare_source()
        before = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_full_training_manifest(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_output_symlink_is_rejected_without_creating_its_target(self):
        self.prepare_source()
        target = self.output.with_name("missing-target.json")
        self.output.symlink_to(target)
        with self.assertRaisesRegex(FileExistsError, "existing symlink"):
            self.promote()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
