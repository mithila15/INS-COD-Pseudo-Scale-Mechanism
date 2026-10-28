#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# RunPod default persistent volume path.
# Override when your datasets are elsewhere:
#   DATASETS_ROOT=/workspace/datasets bash scripts/train_brg_lite_pvtb2.sh
source scripts/_resolve_datasets_root.sh
brg_resolve_datasets_root
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs_brg_lite}"
DEVICE="${DEVICE:-cuda:0}"

PRETRAINED_ARGS=()
if [[ "${USE_PRETRAINED:-1}" == "1" ]]; then
  PRETRAINED_ARGS+=(--pretrained)
fi

python main_for_brg_lite.py \
  --config configs/brg_lite_pvtb2_train.py \
  --data-cfg configs/dataset_cod_runpod.yaml \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  "${PRETRAINED_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
