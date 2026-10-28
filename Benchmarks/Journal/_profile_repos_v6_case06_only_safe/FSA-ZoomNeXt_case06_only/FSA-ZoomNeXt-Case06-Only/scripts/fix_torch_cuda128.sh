#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reinstall PyTorch CUDA 12.8 for Blackwell/sm_120 and then restore
# NumPy<2 project dependency compatibility.
python tools/runpod_preflight.py --auto-fix --skip-deps || true
python -m pip install --no-cache-dir --force-reinstall --no-deps numpy==1.26.4
python tools/runpod_preflight.py --auto-fix --clean-pycache
