#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_runpod_preflight.sh
fsa_runpod_preflight

source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs_fsa_zoomnext_isoflops}"
DEVICE="${DEVICE:-cuda:0}"

PRETRAINED_ARGS=()
if [[ "${USE_PRETRAINED:-1}" == "1" ]]; then
  PRETRAINED_ARGS+=(--pretrained)
fi

python main_for_fsa.py \
  --config configs/fsa_pvtb2_isoflops.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  "${PRETRAINED_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
