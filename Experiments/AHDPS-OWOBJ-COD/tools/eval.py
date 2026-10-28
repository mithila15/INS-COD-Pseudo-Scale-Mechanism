from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahdps_cod.data.builders import build_eval_loaders, ensure_split_file
from ahdps_cod.engine.evaluator import evaluate
from ahdps_cod.models.ahdps_cod import AHDPSCODNet
from ahdps_cod.utils.common import configure_torch_runtime, save_json
from ahdps_cod.utils.config import load_config
from ahdps_cod.utils.reporting import split_alias_to_pretty_name, write_cod_results_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--save-debug-maps", action="store_true")
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--save-sample-metrics", action="store_true")
    parser.add_argument("--max-vis-per-split", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_torch_runtime(cfg["train"].get("device", "cpu"), cfg["train"].get("cpu_num_threads", 1))
    for split in cfg["data"].get("eval_splits", []):
        ensure_split_file(cfg, split, force=False)

    model = AHDPSCODNet(cfg["model"])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    eval_loaders = build_eval_loaders(cfg)

    results_root = Path(cfg["paths"]["results"].get("root", Path(cfg["project"]["root"]) / "results"))
    visualizations_root = Path(cfg["paths"]["results"].get("visualizations", results_root / "visualizations"))

    all_metrics = {}
    pred_root = Path(cfg["paths"]["results"]["predictions"]) if args.save_preds else None
    for split_alias, loader in eval_loaders.items():
        dataset_name = split_alias_to_pretty_name(split_alias)
        save_dir = pred_root / dataset_name if pred_root is not None else None
        analysis_dir = None
        if args.save_debug_maps or args.save_panels or args.save_sample_metrics:
            analysis_dir = visualizations_root / dataset_name
        all_metrics[split_alias] = evaluate(
            model,
            loader,
            device=device,
            save_dir=str(save_dir) if save_dir else None,
            analysis_dir=str(analysis_dir) if analysis_dir else None,
            save_debug_maps=args.save_debug_maps,
            save_panels=args.save_panels,
            save_sample_metrics=args.save_sample_metrics,
            max_visualizations=int(args.max_vis_per_split),
        )

    metrics_root = Path(cfg["paths"]["results"]["metrics"])
    save_json(all_metrics, metrics_root / "eval_only_results.json")
    write_cod_results_txt(metrics_root / "evalresults.txt", cfg["project"].get("name", "AHDPSCOD"), all_metrics)
    print(all_metrics)


if __name__ == "__main__":
    main()
