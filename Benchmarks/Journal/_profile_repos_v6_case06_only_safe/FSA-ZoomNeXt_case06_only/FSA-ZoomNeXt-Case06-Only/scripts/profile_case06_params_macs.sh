#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

DEVICE="${DEVICE:-cpu}"
TOOL="${TOOL:-fvcore}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
OUT_DIR="${OUT_DIR:-outputs_profile_case06}"
AUTO_INSTALL_PROFILE_DEPS="${AUTO_INSTALL_PROFILE_DEPS:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python or activate your venv." >&2
  exit 1
fi

if [[ "${AUTO_INSTALL_PROFILE_DEPS}" == "1" ]]; then
  if [[ "${TOOL}" == "fvcore" || "${TOOL}" == "all" ]]; then
    "${PYTHON_BIN}" - <<'PY' || "${PYTHON_BIN}" -m pip install fvcore
import fvcore  # noqa: F401
PY
  fi
  if [[ "${TOOL}" == "thop" || "${TOOL}" == "all" ]]; then
    "${PYTHON_BIN}" - <<'PY' || "${PYTHON_BIN}" -m pip install thop
import thop  # noqa: F401
PY
  fi
fi

mkdir -p "${OUT_DIR}"

echo "[case06 profile] python=${PYTHON_BIN}"
echo "[case06 profile] device=${DEVICE}, tool=${TOOL}, input=${HEIGHT}x${WIDTH}"
echo "[case06 profile] out_dir=${OUT_DIR}"
echo "[case06 profile] case=case06_av_residual_spatial_d64, d=64"

"${PYTHON_BIN}" tools/profile_case06_params_macs.py \
  --device "${DEVICE}" \
  --tool "${TOOL}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --out-dir "${OUT_DIR}"
