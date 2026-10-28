from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.data.builders import SPLIT_ALIASES, ensure_split_file
from ahdps_cod.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split_alias in SPLIT_ALIASES:
        path = ensure_split_file(cfg, split_alias, force=args.force)
        print(f"{split_alias}: {path}")


if __name__ == "__main__":
    main()
