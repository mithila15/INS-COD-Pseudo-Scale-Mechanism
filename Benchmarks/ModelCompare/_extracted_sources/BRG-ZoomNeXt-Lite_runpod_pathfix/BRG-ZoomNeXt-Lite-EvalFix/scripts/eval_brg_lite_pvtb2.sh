#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_brg_lite_pvtb2.sh /path/to/state_best.pth [extra args]" >&2
  exit 1
fi
LOAD_FROM="$1"
shift

source scripts/_resolve_datasets_root.sh
brg_resolve_datasets_root
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs_brg_lite_eval}"
DEVICE="${DEVICE:-cuda:0}"

python main_for_brg_lite.py \
  --config configs/brg_lite_pvtb2_train.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  --load-from "${LOAD_FROM}" \
  --evaluate \
  --save-results \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
