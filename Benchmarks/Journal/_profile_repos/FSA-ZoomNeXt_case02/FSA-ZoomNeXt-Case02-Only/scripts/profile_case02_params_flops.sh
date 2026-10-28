#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cpu}"
TOOL="${TOOL:-fvcore}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
OUT_DIR="${OUT_DIR:-outputs_profile_case02}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python or activate your venv." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[case02 profile] python=${PYTHON_BIN}"
echo "[case02 profile] device=${DEVICE}, tool=${TOOL}, input=${HEIGHT}x${WIDTH}"
echo "[case02 profile] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" tools/profile_case02_params_flops.py \
  --device "${DEVICE}" \
  --tool "${TOOL}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --out-dir "${OUT_DIR}"
