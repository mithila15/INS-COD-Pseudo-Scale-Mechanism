from __future__ import annotations

from pathlib import Path
from typing import Dict

_SPLIT_TO_PRETTY = {
    "cod10k_test": "COD10K",
    "camo_test": "CAMO",
    "chameleon_test": "CHAMELEON",
    "nc4k_test": "NC4K",
    "train_dev": "TRAIN_DEV",
    "train_main": "TRAIN_MAIN",
    "train_concat": "TRAIN_CONCAT",
    "cod10k_train": "COD10K_TRAIN",
    "camo_train": "CAMO_TRAIN",
}


def split_alias_to_pretty_name(split_alias: str) -> str:
    return _SPLIT_TO_PRETTY.get(split_alias, split_alias.upper())


def write_cod_results_txt(path: str | Path, method_name: str, all_metrics: Dict[str, Dict[str, float]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for split_alias, metrics in all_metrics.items():
        dataset_name = split_alias_to_pretty_name(split_alias)
        lines.append(f"{method_name} {dataset_name} {dict(metrics)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def format_metric_line(metrics: Dict[str, float]) -> str:
    return "; ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
