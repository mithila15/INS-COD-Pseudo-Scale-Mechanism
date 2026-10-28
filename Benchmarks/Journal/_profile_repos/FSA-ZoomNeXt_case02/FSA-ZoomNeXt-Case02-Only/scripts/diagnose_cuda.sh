#!/usr/bin/env bash
set -euo pipefail
nvidia-smi || true
python - <<'PY'
import torch, sys
print('python:', sys.version.replace('\n',' '))
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('torch arch list:', torch.cuda.get_arch_list())
    try:
        x = torch.ones(4, device='cuda')
        y = x + 1
        torch.cuda.synchronize()
        print('CUDA test: OK', y.detach().cpu().tolist())
    except Exception as e:
        print('CUDA test: FAIL')
        print(repr(e))
        raise
PY
