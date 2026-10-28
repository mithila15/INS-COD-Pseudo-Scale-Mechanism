#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"
TOOL="${TOOL:-fvcore}"
python tools/measure_flops.py --model-name PvtV2B2_ZoomNeXt --height 384 --width 384 --device "${DEVICE}" --tool "${TOOL}"
python tools/measure_flops.py --model-name PvtV2B2_FSA_ZoomNeXt --height 384 --width 384 --device "${DEVICE}" --tool "${TOOL}" --mid-dim 48
