#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_runpod_preflight.sh
fsa_runpod_preflight
source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs_case06}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/case06_av_residual_spatial_d64}"
DEVICE="${DEVICE:-cuda:0}"

PRETRAINED_ARGS=()
if [[ "${USE_PRETRAINED:-1}" == "1" ]]; then
  PRETRAINED_ARGS+=(--pretrained)
fi

echo "[case06] DATASETS_ROOT=${DATASETS_ROOT}"
echo "[case06] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[case06] USE_PRETRAINED=${USE_PRETRAINED:-1}, USE_AMP=${USE_AMP:-from_config}"

"${PYTHON_BIN}" main_for_fsa.py \
  --config configs/case06_train.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  "${PRETRAINED_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --info case06_av_residual_spatial_d64 \
  "$@"
