#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TOOL="${TOOL:-fvcore}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_SIZE="${TARGET_SIZE:-384}"
MID_DIM="${MID_DIM:-48}"

python tools/search_iso_flops.py \
  --target-model PvtV2B2_ZoomNeXt \
  --student-model PvtV2B2_BRG_ZoomNeXt_Lite \
  --target-size "$TARGET_SIZE" \
  --mid-dim "$MID_DIM" \
  --candidate-sizes ${CANDIDATE_SIZES:-384 448 512 576 608 640 672 704} \
  --tool "$TOOL" \
  --device "$DEVICE"
