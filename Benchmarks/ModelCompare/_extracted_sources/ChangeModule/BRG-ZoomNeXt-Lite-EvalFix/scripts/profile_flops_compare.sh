#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
BENCHMARK_ITERS="${BENCHMARK_ITERS:-0}"
WARMUP="${WARMUP:-5}"
OUT="${OUT:-/workspace/outputs_brg_lite/flops_profile_compare.json}"
python tools/profile_flops.py \
  --compare \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --device "$DEVICE" \
  --warmup "$WARMUP" \
  --benchmark-iters "$BENCHMARK_ITERS" \
  --output "$OUT"
