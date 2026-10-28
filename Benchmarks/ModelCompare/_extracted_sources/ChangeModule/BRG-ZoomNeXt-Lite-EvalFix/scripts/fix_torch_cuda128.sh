#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio xformers triton || true
python -m pip cache purge || true
python -m pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
bash "$(dirname "$0")/diagnose_cuda.sh"
