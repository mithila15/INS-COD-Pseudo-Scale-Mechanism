#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_fsa_pvtb2.sh /path/to/state_best.pth [extra args]" >&2
  exit 2
fi
CKPT="$1"
shift

source scripts/_runpod_preflight.sh
fsa_runpod_preflight

source scripts/_resolve_datasets_root.sh
fsa_resolve_datasets_root
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs_fsa_zoomnext_eval}"
DEVICE="${DEVICE:-cuda:0}"

python main_for_fsa.py \
  --config configs/fsa_pvtb2_train.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  --load-from "${CKPT}" \
  --evaluate \
  --save-results \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
