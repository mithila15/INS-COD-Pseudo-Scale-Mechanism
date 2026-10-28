from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.utils.visualization import save_train_history_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    history_path = Path(args.history)
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if isinstance(history, dict) and "history" in history:
        history = history["history"]

    save_train_history_plots(args.out_dir, history)
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
