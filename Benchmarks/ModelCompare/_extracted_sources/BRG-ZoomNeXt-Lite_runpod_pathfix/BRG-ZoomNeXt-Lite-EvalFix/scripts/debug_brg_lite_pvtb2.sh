#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_resolve_datasets_root.sh
brg_resolve_datasets_root
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs_brg_lite_debug}"
DEVICE="${DEVICE:-cuda:0}"

PRETRAINED_ARGS=()
if [[ "${USE_PRETRAINED:-0}" == "1" ]]; then
  PRETRAINED_ARGS+=(--pretrained)
fi

python main_for_brg_lite.py \
  --config configs/brg_lite_pvtb2_debug.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  "${PRETRAINED_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
