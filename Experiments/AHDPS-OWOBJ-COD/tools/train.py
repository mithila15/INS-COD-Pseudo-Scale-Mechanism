from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.data.builders import build_dev_loader, build_eval_loaders, build_train_loader, ensure_split_file
from ahdps_cod.engine.trainer import train
from ahdps_cod.losses.hdps import AHDPSCriterion
from ahdps_cod.models.ahdps_cod import AHDPSCODNet
from ahdps_cod.utils.common import configure_torch_runtime, save_json, set_seed
from ahdps_cod.utils.config import load_config
from ahdps_cod.utils.reporting import write_cod_results_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force-splits", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_torch_runtime(cfg["train"].get("device", "cpu"), cfg["train"].get("cpu_num_threads", 1))
    set_seed(int(cfg["project"].get("seed", 3407)))

    for split in [cfg["data"].get("train_split", "train_main"), cfg["data"].get("dev_split", "train_dev"), *cfg["data"].get("eval_splits", [])]:
        ensure_split_file(cfg, split, force=args.force_splits)

    model = AHDPSCODNet(cfg["model"])
    criterion = AHDPSCriterion(cfg["model"])

    train_loader = build_train_loader(cfg)
    dev_loader = build_dev_loader(cfg)
    eval_loaders = build_eval_loaders(cfg)

    results = train(model, criterion, train_loader, dev_loader, eval_loaders, cfg)
    metrics_root = Path(cfg["paths"]["results"]["metrics"])
    save_json(results, metrics_root / "train_and_eval_results.json")
    if "eval" in results:
        write_cod_results_txt(metrics_root / "evalresults.txt", cfg["project"].get("name", "AHDPSCOD"), results["eval"])


if __name__ == "__main__":
    main()
