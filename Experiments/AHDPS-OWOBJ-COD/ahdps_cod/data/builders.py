from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from torch.utils.data import DataLoader

from .cod_binary_dataset import CODBinaryDataset
from .common import SampleRecord
from .transforms import EvalPairTransform, TrainPairTransform

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

DATASET_TAG_TO_INFO = {
    "cod10k_train": {"cfg_key": ("cod10k", "train_image_root", "train_mask_root"), "dataset_name": "COD10K"},
    "cod10k_test": {"cfg_key": ("cod10k", "test_image_root", "test_mask_root"), "dataset_name": "COD10K"},
    "camo_train": {"cfg_key": ("camo", "train_image_root", "mask_root"), "dataset_name": "CAMO"},
    "camo_test": {"cfg_key": ("camo", "test_image_root", "mask_root"), "dataset_name": "CAMO"},
    "chameleon_test": {"cfg_key": ("chameleon", "image_root", "mask_root"), "dataset_name": "CHAMELEON"},
    "nc4k_test": {"cfg_key": ("nc4k", "image_root", "mask_root"), "dataset_name": "NC4K"},
}

SPLIT_ALIASES = {
    "cod10k_train": "cod10k_train_cam_3040.txt",
    "cod10k_test": "cod10k_test_cam_2026.txt",
    "camo_train": "camo_train_1000.txt",
    "camo_test": "camo_test_250.txt",
    "chameleon_test": "chameleon_test_76.txt",
    "nc4k_test": "nc4k_test_4121.txt",
    "train_concat": "cod_train_concat_4040.txt",
    "train_main": "cod_train_concat_main_3636.txt",
    "train_dev": "cod_train_concat_dev_404.txt",
}

BASE_DATASET_SPLIT_ROOTS = {
    "cod10k_train": ("cod10k", "train_image_root"),
    "cod10k_test": ("cod10k", "test_image_root"),
    "camo_train": ("camo", "train_image_root"),
    "camo_test": ("camo", "test_image_root"),
    "chameleon_test": ("chameleon", "image_root"),
    "nc4k_test": ("nc4k", "image_root"),
}


def _list_stems(root: Path, exts: Sequence[str]) -> List[str]:
    exts = {e.lower() for e in exts}
    if not root.exists():
        return []
    return sorted([p.stem for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _resolve_file(root: Path, stem: str, exts: Sequence[str]) -> Path:
    for ext in exts:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot resolve file for stem={stem} in {root}")


def _splits_root(cfg: Dict) -> Path:
    root = Path(cfg["paths"]["datasets"]["splits_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _split_path(cfg: Dict, split_alias: str) -> Path:
    return _splits_root(cfg) / SPLIT_ALIASES[split_alias]


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_root_for_tag(cfg: Dict, dataset_tag: str) -> Path:
    ds_key, image_root_key = BASE_DATASET_SPLIT_ROOTS[dataset_tag]
    return Path(cfg["paths"]["datasets"][ds_key][image_root_key])


def _build_base_dataset_split_lines(cfg: Dict, dataset_tag: str) -> list[str]:
    root = _image_root_for_tag(cfg, dataset_tag)
    stems = _list_stems(root, IMAGE_EXTS)
    if len(stems) == 0:
        raise FileNotFoundError(
            f"No image files found for split {dataset_tag} at {root} (exists={root.exists()})."
        )
    return [f"{dataset_tag}\t{stem}" for stem in stems]


def _ensure_base_dataset_split(cfg: Dict, dataset_tag: str, force: bool = False) -> Path:
    out_path = _split_path(cfg, dataset_tag)
    if out_path.exists() and not force:
        return out_path
    _write_lines(out_path, _build_base_dataset_split_lines(cfg, dataset_tag))
    return out_path


def parse_split_file(path: str | Path) -> List[Tuple[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    entries: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tag, sample_id = line.split("\t", 1)
            entries.append((tag.strip(), sample_id.strip()))
    return entries


def _ensure_train_concat_split(cfg: Dict, force: bool = False) -> Path:
    out_path = _split_path(cfg, "train_concat")
    if out_path.exists() and not force:
        return out_path

    camo_path = _ensure_base_dataset_split(cfg, "camo_train", force=force)
    cod10k_path = _ensure_base_dataset_split(cfg, "cod10k_train", force=force)
    train_concat_entries = []
    train_concat_entries.extend([f"{tag}\t{sample_id}" for tag, sample_id in parse_split_file(camo_path)])
    train_concat_entries.extend([f"{tag}\t{sample_id}" for tag, sample_id in parse_split_file(cod10k_path)])

    rng = random.Random(int(cfg["project"].get("seed", 3407)))
    rng.shuffle(train_concat_entries)
    _write_lines(out_path, train_concat_entries)
    return out_path


def _ensure_train_subsplits(cfg: Dict, force: bool = False) -> tuple[Path, Path]:
    main_path = _split_path(cfg, "train_main")
    dev_path = _split_path(cfg, "train_dev")
    if main_path.exists() and dev_path.exists() and not force:
        return main_path, dev_path

    train_concat_path = _ensure_train_concat_split(cfg, force=force)
    entries = [f"{tag}\t{sample_id}" for tag, sample_id in parse_split_file(train_concat_path)]
    if len(entries) < 2:
        raise ValueError("train_concat must contain at least 2 samples.")

    dev_ratio = float(cfg.get("data", {}).get("dev_ratio", 0.1))
    dev_count = int(round(len(entries) * dev_ratio))
    dev_count = min(max(1, dev_count), max(1, len(entries) - 1))
    dev_entries = entries[:dev_count]
    main_entries = entries[dev_count:]
    _write_lines(dev_path, dev_entries)
    _write_lines(main_path, main_entries)
    return main_path, dev_path


def ensure_split_file(cfg: Dict, split_alias: str, force: bool = False) -> Path:
    out_path = _split_path(cfg, split_alias)
    if out_path.exists() and not force:
        return out_path

    if split_alias in BASE_DATASET_SPLIT_ROOTS:
        return _ensure_base_dataset_split(cfg, split_alias, force=force)
    if split_alias == "train_concat":
        return _ensure_train_concat_split(cfg, force=force)
    if split_alias in ("train_main", "train_dev"):
        main_path, dev_path = _ensure_train_subsplits(cfg, force=force)
        return main_path if split_alias == "train_main" else dev_path
    raise KeyError(f"Unsupported split alias: {split_alias}")


def _make_record(cfg: Dict, dataset_tag: str, sample_id: str) -> SampleRecord:
    datasets_cfg = cfg["paths"]["datasets"]
    info = DATASET_TAG_TO_INFO[dataset_tag]
    ds_key, image_root_key, mask_root_key = info["cfg_key"]
    image_root = Path(datasets_cfg[ds_key][image_root_key])
    mask_root = Path(datasets_cfg[ds_key][mask_root_key])
    image_path = _resolve_file(image_root, sample_id, IMAGE_EXTS)
    mask_path = _resolve_file(mask_root, sample_id, MASK_EXTS)
    return SampleRecord(str(image_path), str(mask_path), info["dataset_name"], sample_id)


def build_records_from_split(cfg: Dict, split_path: str | Path) -> List[SampleRecord]:
    entries = parse_split_file(split_path)
    return [_make_record(cfg, dataset_tag, sample_id) for dataset_tag, sample_id in entries]


def build_train_loader(cfg: Dict) -> DataLoader:
    split_alias = cfg["data"].get("train_split", "train_main")
    split_path = ensure_split_file(cfg, split_alias)
    records = build_records_from_split(cfg, split_path)
    transform = TrainPairTransform(
        image_size=int(cfg["data"]["image_size"]),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        hflip_prob=float(cfg["data"].get("hflip_prob", 0.5)),
    )
    dataset = CODBinaryDataset(records=records, transform=transform)
    return DataLoader(
        dataset,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        drop_last=True,
    )


def build_dev_loader(cfg: Dict) -> DataLoader:
    split_alias = cfg["data"].get("dev_split", "train_dev")
    split_path = ensure_split_file(cfg, split_alias)
    records = build_records_from_split(cfg, split_path)
    transform = EvalPairTransform(
        image_size=int(cfg["data"]["image_size"]),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )
    dataset = CODBinaryDataset(records=records, transform=transform)
    return DataLoader(
        dataset,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
    )


def build_eval_loaders(cfg: Dict) -> Dict[str, DataLoader]:
    split_aliases = list(cfg["data"].get("eval_splits", []))
    loaders: Dict[str, DataLoader] = {}
    transform = EvalPairTransform(
        image_size=int(cfg["data"]["image_size"]),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )
    for split_alias in split_aliases:
        split_path = ensure_split_file(cfg, split_alias)
        records = build_records_from_split(cfg, split_path)
        dataset = CODBinaryDataset(records=records, transform=transform)
        loaders[split_alias] = DataLoader(
            dataset,
            batch_size=int(cfg["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 4)),
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
        )
    return loaders
