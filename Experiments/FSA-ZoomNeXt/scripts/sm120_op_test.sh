#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('arch list:', torch.cuda.get_arch_list())
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    x = torch.rand(4, 3, 384, 384, device='cuda', dtype=dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=dtype).view(1, 3, 1, 1)
    y = x.sub(mean)
    torch.cuda.synchronize()
    print('sub test OK:', dtype, float(y.float().mean().cpu()))
print('all CUDA pointwise tests OK')
PY
