#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Safe RunPod setup for FSA-ZoomNeXt.
# It checks/fixes:
#   - Blackwell/sm_120 PyTorch CUDA wheel compatibility
#   - NumPy 2.x ABI breakage with SciPy/albumentations/pysodmetrics/matplotlib
#   - SafeSimpleASPP patch presence
#   - stale __pycache__

apt-get update || true
apt-get install -y libgl1 libglib2.0-0 || true
python -m pip install --upgrade pip

RUNPOD_AUTO_FIX="${RUNPOD_AUTO_FIX:-1}" python tools/runpod_preflight.py --auto-fix --clean-pycache
bash scripts/diagnose_cuda.sh || true
