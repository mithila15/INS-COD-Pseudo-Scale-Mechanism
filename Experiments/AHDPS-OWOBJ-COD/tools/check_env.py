from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.utils.config import load_config


def describe_dir(path_str: str) -> str:
    path = Path(path_str)
    exists = path.exists()
    is_dir = path.is_dir()
    count = 0
    if exists and is_dir:
        count = sum(1 for p in path.iterdir() if p.is_file())
    return f"exists={exists}, is_dir={is_dir}, file_count={count}, path={path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["paths"]["datasets"]

    print("== dataset paths ==")
    for ds_key, ds_cfg in datasets.items():
        if not isinstance(ds_cfg, dict):
            continue
        print(f"[{ds_key}]")
        for key, value in ds_cfg.items():
            print(f"  - {key}: {describe_dir(value)}")

    print("\n== autodetect report ==")
    print(cfg.get("autodetect", {}).get("datasets", {}))


if __name__ == "__main__":
    main()
