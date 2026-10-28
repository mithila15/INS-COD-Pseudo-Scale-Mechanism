#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[verify] cwd=$(pwd)"
python - <<'PY'
from pathlib import Path
files = [
    Path('methods/zoomnext/srg_lite.py'),
    Path('methods/zoomnext/brg_lite.py'),
    Path('methods/zoomnext/brg_lite_layers.py'),
    Path('methods/zoomnext/layers.py'),
    Path('utils/pipeline/scheduler.py'),
]
for f in files:
    print(f, 'exists=', f.exists(), 'size=', f.stat().st_size if f.exists() else None)
    if f.exists():
        with open(f, 'rb') as fp:
            fp.read(64)
print('SafeSimpleASPP in srg:', 'SafeSimpleASPP' in Path('methods/zoomnext/srg_lite.py').read_text())
print('SafeSimpleASPP class:', 'class SafeSimpleASPP' in Path('methods/zoomnext/brg_lite_layers.py').read_text())
print('scheduler read OK')
PY
grep -R "self.tra_5 =" -n methods/zoomnext/brg_lite.py methods/zoomnext/srg_lite.py
