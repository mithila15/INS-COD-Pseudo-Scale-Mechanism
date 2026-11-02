# 2026-10-28 · FSA Study Environment — Installation and Runtime Checks

**English** | [한국어](README.ko.md) · [Experiment guide](../../../../README.md)

Create a dedicated environment before preparing or training experiments. The commands below target **Linux x86_64 and Python 3.12.13**, using the recorded package versions and distribution hashes. A complete fresh installation from this lock has **not yet been validated**; the existing verification used the local overlay described below.

## 1. Create a new environment

Install `uv` using its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if it is unavailable. The setup commands use uv's [version-specific Python installation](https://docs.astral.sh/uv/guides/install-python/); their flags were checked with local uv **0.12.11** on 2026-10-28.

Use the same terminal for all blocks. Change `REPO` to the absolute checkout path on another machine. `PY` is the interpreter to use in the experiment guide as well. This block downloads Python and packages when executed; it does not download datasets or model weights or start training. An existing environment is preserved.

```bash
REPO=/home/robin/shMin/INS_COD
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study-clean/bin/python"

if [ -e "$REPO/.venv/fsa-study-clean" ]; then
  printf '%s\n' 'Environment already exists; use it after checks, or choose a new environment path.'
else
  uv python install --no-bin 3.12.13 &&
  uv venv --directory "$STUDY" --no-project --managed-python --python 3.12.13 \
    "$REPO/.venv/fsa-study-clean" &&
  uv pip sync --directory "$STUDY" --python "$PY" --require-hashes \
    "$STUDY/environment/requirements.lock"
fi
```

The lock includes CUDA-related packages. Selecting CPU execution does not turn it into a lightweight CPU-only installation. If a different platform or runtime requires a different dependency set, keep it as a separate environment and record the changed versions instead of modifying this study's historical lock. A partial installation is not a successful setup: if installation fails, retain the failure information and resolve it before the checks below.

## 2. Check imports and device availability

These commands only inspect the environment and parse CLI options. They do not construct a training job or update parameters.

```bash
"$PY" - <<'PY'
import json
import platform
import torch
import torchvision
import timm
import einops
import numpy
import PIL
import py_sod_metrics

print(json.dumps({
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__),
    "timm": timm.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY

"$PY" "$STUDY/run_study.py" prepare --help
"$PY" "$STUDY/run_study.py" train --help
"$PY" "$STUDY/plan_study.py" --help
```

The verified runtime was **torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30, Python 3.12.13**. CPU inference and one CUDA inference configuration were checked. The local GPU is **NVIDIA GeForce RTX 5080**; a read-only device check on 2026-10-28 reported driver **595.91.07**. This is the observed host, not a claimed minimum driver requirement. Confirm `cuda_available` on the target machine and use `--device cpu` when running on CPU. The recorded GPU check did not validate GPU backward or complete training.

## 3. Run checks without training

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

The tests use synthetic tensors and temporary image/mask fixtures. Some checks call backward to inspect gradient connectivity; they perform **zero optimizer steps** and do not train or download data. The [historical verification](../results/verification-2026-10-28/README.md) passed **281 tests and 67 subtests** in the existing overlay environment. Those results do not certify a fresh lock installation, actual pretrained weights, full real-data loading, or convergence. Record new results separately when checking a newly installed environment.

## 4. Supply local inputs and start experiments

Continue with the [experiment guide](../../../../README.md) using this `PY` value. Supply an existing local **PVTv2-B2** checkpoint to `--backbone-weights`. The loader accepts a flat state dictionary or `{"state_dict": ...}`, removes only `head.weight` and `head.bias`, then loads the encoder with `strict=True`. It does not silently accept an unrelated or partially matching backbone. The actual pretrained file has not been verified in this workspace.

Datasets and weights must already be present before their respective stages. `--random-init` is available only for an intentional random-initialization comparison, with the same initialization condition across all variants. Training begins when the user executes `train` or a generated `train.sh`; environment setup, `--help`, and these tests do not start it.

## Provenance and existing local environment

| Record | Purpose |
| --- | --- |
| [Archived FSA requirements](../../../FSA-ZoomNeXt/requirements.txt) | Original experiment dependencies, preserved unchanged |
| [Study requirements](../requirements.txt) | Direct dependencies of the new study |
| [Verified constraints](constraints-verified.txt) | Versions used for local verification |
| [Hashed lock](requirements.lock) | Pinned transitive dependencies and allowed distribution hashes |
| [Verification environment](verification.json) | Original installation method and package inventory |

The existing `/home/robin/shMin/INS_COD/.venv/fsa-study` is a **local verification overlay**: its lightweight dependencies are isolated, but a `.pth` file reads PyTorch packages from `/home/robin/shMin/GeneralGodhand/.venv/isaac`. The source environment was not modified. This absolute-path dependency makes the overlay unsuitable as a portable installation recipe, and Isaac packages are not study dependencies. Do not upgrade or sync that shared environment for this experiment; use the new independent environment above.

The archived backbone emits deprecation warnings for older timm import paths and a PyTorch SDPA context manager. It ran with the recorded versions; the original source was retained to preserve its provenance.
