"""Synthetic input integrity checks; no training and no optimizer updates."""
import importlib.util
import json
import random
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa_study.data import SOURCES, CODDataset, load_manifest, prepare_manifest
from run_study import build_parser


def _png(value):
    """Small valid RGB PNG, built with stdlib to keep manifest tests torch-free."""
    def chunk(kind, payload):
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload) & 0xffffffff)
    rows = b"".join(b"\x00" + bytes((value, y * 40, 255 - value)) * 4 for y in range(4))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def fixture(root, tests=("cod10k", "camo")):
    serial = 1
    for source, (dataset, split, image_rel, mask_rel, _) in SOURCES.items():
        if split == "test" and dataset not in tests:
            continue
        for index in range(4 if split == "train" else 2):
            stem = ("COD10K-CAM-" if dataset == "cod10k" else "") + f"{source}_{index}"
            for folder, value in ((image_rel, serial), (mask_rel, 255)):
                path = root / folder / f"{stem}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_png(value))
            serial += 1


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"
        self.root.mkdir()
        fixture(self.root)
        self.path = Path(self.temporary.name) / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, **kwargs):
        return prepare_manifest(self.root, self.path, allow_nonstandard_counts=True, **kwargs)

    def test_reproducible_holdout_and_missing_test_record(self):
        first = self.prepare()
        second = prepare_manifest(self.root, self.path.with_name("repeat.json"), allow_nonstandard_counts=True)
        self.assertEqual(first, second)
        self.assertEqual(first["test_datasets_selected"], ["camo", "cod10k"])
        self.assertEqual(first["test_datasets_missing"], ["chameleon", "nc4k"])
        self.assertEqual(first["split_counts"]["camo/val"], 1)
        self.assertEqual(first["split_counts"]["cod10k/val"], 1)
        self.assertEqual(load_manifest(self.path)[0]["content_sha256"], first["content_sha256"])
        for record in first["records"]:
            self.assertEqual(record["source"].endswith("test"), record["split"] == "test")

    def test_cod10k_explicit_foreground_filter_audits_exclusions(self):
        excluded = self.root / "COD10K-v3/Train/Image/COD10K-NonCAM-extra.png"
        excluded.write_bytes(_png(180))
        manifest = self.prepare()
        self.assertEqual(manifest["source_counts"]["cod10k_train"], 4)
        self.assertEqual(manifest["raw_source_counts"]["cod10k_train"]["images"], 5)
        self.assertEqual(manifest["excluded_files"]["cod10k_train"]["images"][0]["path"], excluded.relative_to(self.root).as_posix())
        excluded.write_bytes(_png(181))
        with self.assertRaisesRegex(ValueError, "Changed explicitly excluded"):
            load_manifest(self.path)

    def test_manifest_not_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_explicit_missing_test_fails(self):
        with self.assertRaisesRegex(FileNotFoundError, "Explicitly selected"):
            self.prepare(test_datasets=["chameleon"])

    def test_explicit_no_test_preserves_shared_mask_explanation(self):
        manifest = self.prepare(test_datasets=["none"])
        self.assertFalse(any(record["split"] == "test" for record in manifest["records"]))
        self.assertTrue(manifest["unselected_shared_masks"])

    def test_standard_counts_required_without_override(self):
        with self.assertRaisesRegex(ValueError, "expected 3040"):
            prepare_manifest(self.root, self.path)

    def test_missing_mask_is_not_silently_dropped(self):
        (self.root / "COD10K-v3/Train/GT_Object/COD10K-CAM-cod10k_train_0.png").unlink()
        with self.assertRaisesRegex(ValueError, "no mask"):
            self.prepare()

    def test_orphan_mask_is_rejected(self):
        (self.root / "COD10K-v3/Train/GT_Object/COD10K-CAM-orphan.png").write_bytes(_png(255))
        with self.assertRaisesRegex(ValueError, "no matching image"):
            self.prepare()

    def test_duplicate_stem_is_rejected(self):
        (self.root / "COD10K-v3/Train/Image/COD10K-CAM-cod10k_train_0.jpg").write_bytes(_png(123))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_changed_file_and_manifest_detected(self):
        manifest = self.prepare()
        changed = self.root / manifest["records"][0]["image"]
        changed.write_bytes(_png(200))
        with self.assertRaisesRegex(ValueError, "Changed image"):
            load_manifest(self.path)
        manifest["split_seed"] += 1
        self.path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Manifest checksum"):
            load_manifest(self.path)

    def test_duplicate_image_crossing_train_test_rejected(self):
        train_paths = sorted((self.root / "COD10K-v3/Train/Image").glob("*.png"))
        target = self.root / "COD10K-v3/Test/Image/COD10K-CAM-cod10k_test_0.png"
        target.write_bytes(train_paths[0].read_bytes())
        with self.assertRaisesRegex(ValueError, "hashes cross"):
            self.prepare()

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "torch/Pillow unavailable")
    def test_augmentation_does_not_depend_on_model_rng(self):
        import torch
        manifest = self.prepare()
        data = CODDataset(manifest, self.root, "train", size=64, seed=7)
        data.epoch = 4
        first = data[0]
        random.seed(999)
        torch.manual_seed(999)
        torch.rand(123)
        second = data[0]
        self.assertTrue(torch.equal(first["image"], second["image"]))
        self.assertTrue(torch.equal(first["mask"], second["mask"]))
        self.assertEqual(first["edge"].shape, (1, 64, 64))
        self.assertTrue(torch.isfinite(first["image"]).all())


class CommandTests(unittest.TestCase):
    def test_explicit_initialization_and_control_flags(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--manifest", "/tmp/m.json", "--output", "/tmp/run",
                                  "--random-init", "--baseline", "--generator-options", '{"scale": 2}'])
        self.assertTrue(args.baseline)
        self.assertEqual(args.generator_options, {"scale": 2})
        self.assertEqual(args.determinism, "warn")
        self.assertEqual(args.epochs, 150)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "torch/Pillow unavailable")
    def test_checkpoint_variant_and_manifest_mismatch_rejected(self):
        import torch
        from fsa_study.engine import _environment
        from fsa_study.engine import _model_config_digest, validate_checkpoint
        model_config = {"generator": "legacy", "utilizer": "mean", "channels": 48, "seed": 0}
        checkpoint = {"schema_version": 1, "model_config": model_config,
                      "model_config_sha256": _model_config_digest(model_config), "manifest_sha256": "abc"}
        checkpoint["environment"] = _environment()
        checkpoint["model"] = {"dummy": torch.zeros(1)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            torch.save(checkpoint, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, {"content_sha256": "abc"})
        with self.assertRaisesRegex(ValueError, "different dataset"):
            validate_checkpoint(checkpoint, {"content_sha256": "def"})
        checkpoint["model_config"]["utilizer"] = "other"
        with self.assertRaisesRegex(ValueError, "variant metadata"):
            validate_checkpoint(checkpoint, {"content_sha256": "abc"})


if __name__ == "__main__":
    unittest.main()
