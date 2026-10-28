#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run this once in a fresh RunPod container.
# Most PyTorch RunPod images already include torch/torchvision; this installs the project dependencies.
apt-get update || true
apt-get install -y libgl1 libglib2.0-0 || true
python -m pip install --upgrade pip
python -m pip install -r requirements_brg_lite.txt
