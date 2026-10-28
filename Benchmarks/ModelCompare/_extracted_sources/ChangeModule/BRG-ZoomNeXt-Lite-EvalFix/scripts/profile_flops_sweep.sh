#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
DIMS="${DIMS:-32,48,64,80,96,128,160,192}"
OUT="${OUT:-/workspace/outputs_brg_lite/flops_profile_sweep.json}"
python tools/profile_flops.py \
  --compare \
  --sweep-lite-mid-dims "$DIMS" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --device "$DEVICE" \
  --output "$OUT"
