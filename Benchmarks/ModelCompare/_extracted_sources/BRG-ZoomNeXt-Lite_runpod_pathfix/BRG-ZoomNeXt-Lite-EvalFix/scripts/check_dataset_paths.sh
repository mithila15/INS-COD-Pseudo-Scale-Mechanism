#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_resolve_datasets_root.sh
brg_resolve_datasets_root

echo
echo "[BRG-Lite] Checking expected folders..."
paths=(
  "${DATASETS_ROOT}/COD10K-v3/Train/Image"
  "${DATASETS_ROOT}/COD10K-v3/Train/GT_Object"
  "${DATASETS_ROOT}/COD10K-v3/Train/GT_Edge"
  "${DATASETS_ROOT}/COD10K-v3/Test/Image"
  "${DATASETS_ROOT}/COD10K-v3/Test/GT_Object"
  "${DATASETS_ROOT}/COD10K-v3/Test/GT_Edge"
  "${DATASETS_ROOT}/CAMO-V.1.0-CVIU2019/Images/Train"
  "${DATASETS_ROOT}/CAMO-V.1.0-CVIU2019/Images/Test"
  "${DATASETS_ROOT}/CAMO-V.1.0-CVIU2019/GT"
  "${DATASETS_ROOT}/CHAMELEON/Imgs"
  "${DATASETS_ROOT}/CHAMELEON/GT"
  "${DATASETS_ROOT}/CHAMELEON/Edge"
  "${DATASETS_ROOT}/NC4K/Imgs"
  "${DATASETS_ROOT}/NC4K/GT"
)
for p in "${paths[@]}"; do
  if [[ -d "${p}" ]]; then
    n="$(find "${p}" -maxdepth 1 -type f | wc -l | tr -d ' ')"
    echo "[OK]      ${p}  files=${n}"
  else
    echo "[MISSING] ${p}"
  fi
done
