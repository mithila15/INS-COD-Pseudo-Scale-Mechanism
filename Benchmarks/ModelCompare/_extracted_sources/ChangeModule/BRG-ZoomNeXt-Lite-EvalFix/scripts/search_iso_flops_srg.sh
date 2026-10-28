#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"
TOOL="${TOOL:-fvcore}"

python tools/search_iso_flops.py \
  --target-model PvtV2B2_ZoomNeXt \
  --student-model PvtV2B2_SRG_ZoomNeXt_Lite \
  --target-size 384 \
  --candidate-sizes 384 448 512 576 608 640 672 704 \
  --device "${DEVICE}" \
  --tool "${TOOL}" \
  --mid-dim 48
