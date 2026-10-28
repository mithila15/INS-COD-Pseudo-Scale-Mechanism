#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHT="${1:-}"
if [[ -z "${WEIGHT}" ]]; then
  echo "Usage: bash scripts/eval_case06.sh /path/to/state_best.pth [extra main_for_fsa.py args...]" >&2
  exit 2
fi
shift || true

source scripts/_runpod_preflight.sh
fsa_runpod_preflight
source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs_case06_eval/case06_av_residual_spatial_d64}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" main_for_fsa.py \
  --config configs/case06_train.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  --load-from "${WEIGHT}" \
  --evaluate \
  --save-results \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --info case06_eval \
  "$@"
