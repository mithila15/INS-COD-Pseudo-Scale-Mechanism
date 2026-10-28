#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CASE_ID="${1:-}"
WEIGHT="${2:-}"
if [[ -z "${CASE_ID}" || -z "${WEIGHT}" ]]; then
  echo "Usage: bash scripts/eval_fsa_case.sh <case_id> <state_best.pth> [extra main_for_fsa.py args...]" >&2
  exit 2
fi
shift 2 || true
CONFIG="configs/cases/${CASE_ID}.py"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[FSA case][ERROR] Missing config: ${CONFIG}" >&2
  exit 2
fi

export RUNPOD_SKIP_PREFLIGHT="${RUNPOD_SKIP_PREFLIGHT:-1}"
export USE_AMP="${USE_AMP:-0}"
source scripts/_runpod_preflight.sh
fsa_runpod_preflight
source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root

OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PWD/outputs_fsa_cases_eval}"
OUTPUT_DIR="${OUTPUT_ROOT}/${CASE_ID}"
DEVICE="${DEVICE:-cuda:0}"

python main_for_fsa.py \
  --config "${CONFIG}" \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  --metric-names S meanE wF MAE \
  --load-from "${WEIGHT}" \
  --evaluate \
  --save-results \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --info "${CASE_ID}_eval" \
  "$@"
