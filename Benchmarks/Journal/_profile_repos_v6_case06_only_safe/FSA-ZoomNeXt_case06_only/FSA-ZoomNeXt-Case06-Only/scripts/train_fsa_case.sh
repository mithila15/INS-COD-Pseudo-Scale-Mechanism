#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CASE_ID="${1:-}"
if [[ -z "${CASE_ID}" ]]; then
  echo "Usage: bash scripts/train_fsa_case.sh <case_id> [extra main_for_fsa.py args...]" >&2
  echo "Example: bash scripts/train_fsa_case.sh case05_av_residual_spatial" >&2
  exit 2
fi
shift || true

CONFIG="configs/cases/${CASE_ID}.py"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[FSA case][ERROR] Missing config: ${CONFIG}" >&2
  echo "Available cases:" >&2
  ls configs/cases/*.py | sed 's#configs/cases/##; s#.py$##; s#^#  #' >&2
  exit 2
fi

export RUNPOD_SKIP_PREFLIGHT="${RUNPOD_SKIP_PREFLIGHT:-1}"
export USE_AMP="${USE_AMP:-0}"
source scripts/_runpod_preflight.sh
fsa_runpod_preflight

source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root

OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs_fsa_cases}"
OUTPUT_DIR="${OUTPUT_ROOT}/${CASE_ID}"
DEVICE="${DEVICE:-cuda:0}"

PRETRAINED_ARGS=()
if [[ "${USE_PRETRAINED:-1}" == "1" ]]; then
  PRETRAINED_ARGS+=(--pretrained)
fi

echo "[FSA case] Training ${CASE_ID}"
echo "[FSA case] Config: ${CONFIG}"
echo "[FSA case] Output: ${OUTPUT_DIR}"

python main_for_fsa.py \
  --config "${CONFIG}" \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  --metric-names S meanE wF MAE \
  "${PRETRAINED_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --info "${CASE_ID}" \
  "$@"

# Optional shared summary update. This rewrites the CSV from metric JSON files,
# so repeated calls do not create duplicate rows.
if [[ -n "${SHARED_SUMMARY_CSV:-}" ]]; then
  SUMMARY_ROOTS_VALUE="${SUMMARY_ROOTS:-${OUTPUT_ROOT}}"
  OUTPUT_ROOTS="${SUMMARY_ROOTS_VALUE}" \
  SHARED_SUMMARY_CSV="${SHARED_SUMMARY_CSV}" \
  SHARED_SUMMARY_MD="${SHARED_SUMMARY_MD:-${SHARED_SUMMARY_CSV%.csv}.md}" \
  E_SOURCE="${E_SOURCE:-meanE}" \
  bash scripts/update_shared_summary.sh || true
fi
