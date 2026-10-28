#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TOOL="${TOOL:-fvcore}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_SIZE="${TARGET_SIZE:-384}"
STUDENT_SIZE="${STUDENT_SIZE:-384}"
MID_DIM="${MID_DIM:-48}"

python tools/measure_flops.py \
  --model-name PvtV2B2_ZoomNeXt \
  --height "$TARGET_SIZE" --width "$TARGET_SIZE" \
  --tool "$TOOL" --device "$DEVICE"

python tools/measure_flops.py \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  --height "$STUDENT_SIZE" --width "$STUDENT_SIZE" \
  --mid-dim "$MID_DIM" \
  --tool "$TOOL" --device "$DEVICE"
