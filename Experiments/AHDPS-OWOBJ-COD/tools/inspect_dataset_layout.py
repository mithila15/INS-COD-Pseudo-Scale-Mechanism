from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.data.builders import IMAGE_EXTS
from ahdps_cod.utils.config import load_config


def count_stems(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {e.lower() for e in IMAGE_EXTS}
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mapping = {
        "cod10k_train": Path(cfg["paths"]["datasets"]["cod10k"]["train_image_root"]),
        "cod10k_test": Path(cfg["paths"]["datasets"]["cod10k"]["test_image_root"]),
        "camo_train": Path(cfg["paths"]["datasets"]["camo"]["train_image_root"]),
        "camo_test": Path(cfg["paths"]["datasets"]["camo"]["test_image_root"]),
        "chameleon_test": Path(cfg["paths"]["datasets"]["chameleon"]["image_root"]),
        "nc4k_test": Path(cfg["paths"]["datasets"]["nc4k"]["image_root"]),
    }

    print("split,image_root,image_count")
    for tag, root in mapping.items():
        print(f"{tag},{root},{count_stems(root)}")


if __name__ == "__main__":
    main()
