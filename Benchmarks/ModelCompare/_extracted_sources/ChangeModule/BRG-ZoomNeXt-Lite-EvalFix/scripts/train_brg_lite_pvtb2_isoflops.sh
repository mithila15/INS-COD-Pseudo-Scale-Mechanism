#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export DATASETS_ROOT="${DATASETS_ROOT:-/workspace/COD/datasets}"
PRETRAINED_FLAG="--pretrained"
if [[ "${USE_PRETRAINED:-1}" == "0" ]]; then
  PRETRAINED_FLAG=""
fi

python main_for_brg_lite.py \
  --config configs/brg_lite_pvtb2_isoflops.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  ${PRETRAINED_FLAG} \
  --output-dir /workspace/outputs_brg_lite_isoflops
