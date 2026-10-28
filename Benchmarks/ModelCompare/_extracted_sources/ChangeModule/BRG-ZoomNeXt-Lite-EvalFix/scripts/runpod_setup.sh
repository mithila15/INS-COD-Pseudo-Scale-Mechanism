#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run once in a fresh RunPod container.
# If your RunPod GPU is RTX 50xx / Blackwell, older torch wheels can raise:
# "CUDA error: no kernel image is available for execution on the device".
# This setup checks CUDA first and can reinstall PyTorch with CUDA 12.8 wheels.

apt-get update || true
apt-get install -y libgl1 libglib2.0-0 || true
python -m pip install --upgrade pip

AUTO_INSTALL_TORCH="${AUTO_INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

set +e
python - <<'PY' > /tmp/cuda_test.log 2>&1
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available')
x = torch.ones(1, device='cuda')
y = x + 1
torch.cuda.synchronize()
print('CUDA test OK:', y.detach().cpu().item())
PY
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[runpod_setup] Existing PyTorch CUDA test failed:"
  cat /tmp/cuda_test.log || true
  if [[ "$AUTO_INSTALL_TORCH" == "1" ]]; then
    echo "[runpod_setup] Reinstalling PyTorch from ${TORCH_INDEX_URL}"
    python -m pip uninstall -y torch torchvision torchaudio xformers triton || true
    python -m pip cache purge || true
    python -m pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
  else
    echo "[runpod_setup] AUTO_INSTALL_TORCH=0, skipping PyTorch reinstall."
  fi
else
  cat /tmp/cuda_test.log || true
fi

# Install project dependencies without changing the installed PyTorch build.
python -m pip install --no-cache-dir -r requirements_project_only.txt
bash scripts/diagnose_cuda.sh
