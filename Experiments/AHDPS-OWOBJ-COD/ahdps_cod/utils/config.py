from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

_DATASET_LAYOUT_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "cod10k": {
        "_aliases": ["COD10K-v3", "COD10K", "COD10K_v3", "cod10k-v3", "cod10k"],
        "train_image_root": {"rels": ["Train/Image", "Train/Images", "Train/Imgs"], "exts": _IMAGE_EXTS},
        "train_mask_root": {"rels": ["Train/GT_Object", "Train/GT", "Train/Mask", "Train/Masks"], "exts": _MASK_EXTS},
        "test_image_root": {"rels": ["Test/Image", "Test/Images", "Test/Imgs"], "exts": _IMAGE_EXTS},
        "test_mask_root": {"rels": ["Test/GT_Object", "Test/GT", "Test/Mask", "Test/Masks"], "exts": _MASK_EXTS},
    },
    "camo": {
        "_aliases": ["CAMO-V.1.0-CVIU2019", "CAMO", "camo"],
        "train_image_root": {"rels": ["Images/Train", "Image/Train", "Train/Image", "Train/Images"], "exts": _IMAGE_EXTS},
        "test_image_root": {"rels": ["Images/Test", "Image/Test", "Test/Image", "Test/Images"], "exts": _IMAGE_EXTS},
        "mask_root": {"rels": ["GT", "Mask", "Masks", "Annotations"], "exts": _MASK_EXTS},
    },
    "chameleon": {
        "_aliases": ["CHAMELEON", "Chameleon", "chameleon"],
        "image_root": {"rels": ["Imgs", "Images", "Image"], "exts": _IMAGE_EXTS},
        "mask_root": {"rels": ["GT", "Mask", "Masks"], "exts": _MASK_EXTS},
    },
    "nc4k": {
        "_aliases": ["NC4K", "nc4k"],
        "image_root": {"rels": ["Imgs", "Images", "Image"], "exts": _IMAGE_EXTS},
        "mask_root": {"rels": ["GT", "Mask", "Masks"], "exts": _MASK_EXTS},
    },
}


def _expand_string(value: str, variables: Dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in variables:
            return variables[key]
        return os.environ.get(key, match.group(0))

    value = os.path.expanduser(value)
    value = _VAR_PATTERN.sub(_replace, value)
    return os.path.expandvars(value)


def _expand_object(obj: Any, variables: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_object(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_object(v, variables) for v in obj]
    if isinstance(obj, str):
        return _expand_string(obj, variables)
    return obj


def _absolutize_path(path_str: str, base_dir: Path) -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def guess_default_datasets_root(project_root: str | Path) -> Path:
    env_value = os.environ.get("COD_DATASETS_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    candidates = [
        Path("/workspace/cod/datasets"),
        project_root.parent / "datasets",
        project_root.parent / "Datasets",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _iter_dirs(root: Path, max_depth: int = 2) -> Iterable[Path]:
    if not root.exists():
        return []
    dirs: list[Path] = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            try:
                depth = len(path.relative_to(root).parts)
            except ValueError:
                continue
            if depth <= max_depth:
                dirs.append(path)
    return dirs


def _has_matching_files(root: Path, exts: Sequence[str]) -> bool:
    exts = {e.lower() for e in exts}
    if not root.exists() or not root.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in exts for child in root.iterdir())


def _path_is_usable(path: Path, exts: Sequence[str] | None = None) -> bool:
    if not path.exists():
        return False
    if exts is None:
        return path.is_dir()
    return _has_matching_files(path, exts)


def _find_dataset_base_dirs(datasets_root: Path, aliases: Sequence[str]) -> list[Path]:
    if not datasets_root.exists():
        return []
    alias_norms = {_norm_name(alias) for alias in aliases}
    matches: list[Path] = []
    seen: set[str] = set()
    for path in _iter_dirs(datasets_root, max_depth=2):
        norm = _norm_name(path.name)
        if any(norm == alias or alias in norm or norm in alias for alias in alias_norms):
            resolved = str(path.resolve())
            if resolved not in seen:
                matches.append(path.resolve())
                seen.add(resolved)
    matches.sort(key=lambda p: (len(p.parts), len(str(p))))
    return matches


def _resolve_dataset_layouts(cfg: Dict[str, Any], datasets_root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "datasets_root": str(datasets_root),
        "datasets_root_exists": datasets_root.exists(),
        "changes": [],
    }
    datasets_cfg = cfg.setdefault("paths", {}).setdefault("datasets", {})
    for ds_key, spec in _DATASET_LAYOUT_CANDIDATES.items():
        ds_cfg = datasets_cfg.get(ds_key)
        if not isinstance(ds_cfg, dict):
            continue

        base_dirs = _find_dataset_base_dirs(datasets_root, spec.get("_aliases", [ds_key]))
        if not base_dirs:
            continue

        for path_key, rule in spec.items():
            if path_key.startswith("_"):
                continue
            current_value = ds_cfg.get(path_key)
            if not isinstance(current_value, str) or not current_value.strip():
                continue
            current_path = Path(current_value).expanduser()
            if _path_is_usable(current_path, rule.get("exts")):
                continue

            candidate = None
            for base_dir in base_dirs:
                for rel in rule.get("rels", []):
                    possible = (base_dir / rel).resolve()
                    if _path_is_usable(possible, rule.get("exts")):
                        candidate = possible
                        break
                if candidate is not None:
                    break

            if candidate is not None:
                ds_cfg[path_key] = str(candidate)
                report["changes"].append({
                    "dataset": ds_key,
                    "key": path_key,
                    "from": str(current_path),
                    "to": str(candidate),
                })
    return report


def load_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    project_root = cfg.get("project", {}).get("root")
    if project_root is None:
        # configs/<file>.yaml -> repo root
        project_root = config_path.parent.parent
    project_root = Path(project_root).expanduser().resolve()

    variables = {
        "PROJECT_ROOT": str(project_root),
        "CONFIG_DIR": str(config_path.parent.resolve()),
        "COD_DATASETS_ROOT": os.environ.get("COD_DATASETS_ROOT", str(guess_default_datasets_root(project_root))),
    }
    cfg = _expand_object(cfg, variables)

    cfg.setdefault("project", {})
    cfg["project"]["root"] = str(project_root)

    for section, value in list(cfg.get("paths", {}).items()):
        if not isinstance(value, dict):
            continue
        for key, subvalue in list(value.items()):
            if isinstance(subvalue, str):
                cfg["paths"][section][key] = _absolutize_path(subvalue, project_root)
            elif isinstance(subvalue, dict):
                for subkey, subsub in list(subvalue.items()):
                    if isinstance(subsub, str):
                        cfg["paths"][section][key][subkey] = _absolutize_path(subsub, project_root)

    datasets_root = guess_default_datasets_root(project_root)
    cfg.setdefault("autodetect", {})
    cfg["autodetect"]["datasets"] = _resolve_dataset_layouts(cfg, datasets_root)
    return cfg
