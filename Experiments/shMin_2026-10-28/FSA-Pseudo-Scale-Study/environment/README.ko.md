# 2026-11-04 · FSA Study Environment — 환경 설치와 실행 검사

[English](README.md) | **한국어** · [실험 안내](../../../../README.ko.md)

현재 CNN 실험: 기존 torch/torchvision 잠금 환경에서 ConvNeXt-Tiny와 ResNet50을 지원한다. [CNN 안내](../docs/CNN-BACKBONES.ko.md)의 모델별 **torchvision 가중치**를 사용한다. 아래 PVT 전용 초기화 설명은 기존 PVT 설정에만 해당한다.

실험을 준비하거나 학습하기 전에 전용 환경을 만든다. 아래 명령은 **Linux x86_64, Python 3.12.13**을 기준으로 기록된 패키지 버전과 배포 파일 해시를 사용한다. 이 lock 전체를 완전히 새 환경에 설치하는 과정은 **아직 검증하지 않았다**. 기존 검증은 아래에 설명한 로컬 overlay 환경에서 수행했다.

## 1. 새 환경 만들기

`uv`가 없으면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 준비한다. 아래 명령은 uv의 [Python 버전 지정 설치](https://docs.astral.sh/uv/guides/install-python/)를 사용한다. 명령 옵션은 2026-10-28 로컬 uv **0.12.11**에서 확인했다.

모든 블록은 같은 터미널에서 실행한다. 다른 장비에서는 `REPO`를 해당 저장소의 절대 경로로 바꾼다. 실험 안내에서도 여기서 지정한 `PY`를 사용한다. 이 블록을 실행하면 Python과 패키지를 내려받으며 데이터셋·모델 가중치 다운로드나 학습은 시작하지 않는다. 기존 환경은 보존한다.

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

이 lock에는 CUDA 관련 패키지가 포함된다. CPU 실행을 선택해도 설치 용량이 작은 CPU 전용 환경으로 바뀌지는 않는다. 다른 플랫폼이나 런타임 때문에 패키지 구성이 달라져야 한다면 별도 환경에서 변경 버전을 기록하고 이 연구의 역사적 lock은 보존한다. 설치가 일부 진행됐다고 준비가 끝난 것은 아니다. 설치가 실패하면 실패 정보를 남기고 원인을 해결한 뒤 아래 검사를 진행한다.

## 2. import와 장치 확인

다음 명령은 환경 확인과 CLI 옵션 해석만 수행한다. 학습 작업을 만들거나 파라미터를 업데이트하지 않는다.

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

기존 검증 런타임은 **torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30, Python 3.12.13**이었다. CPU 추론과 CUDA 추론 한 구성을 확인했다. 로컬 GPU는 **NVIDIA GeForce RTX 5080**이며 2026-10-28 읽기 전용 장치 검사에서 드라이버 **595.91.07**을 확인했다. 이는 실제 확인한 장비 정보이며 최소 드라이버 요구 조건이라는 뜻은 아니다. 사용할 장비에서 `cuda_available`을 확인하고 CPU 실행에는 `--device cpu`를 지정한다. 기존 GPU 검사는 GPU backward나 전체 학습을 검증한 기록이 아니다.

## 3. 학습 없이 검사하기

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

검사는 합성 tensor와 임시 이미지·마스크 fixture를 사용한다. 일부 검사는 gradient 연결을 확인하기 위해 backward를 호출하지만 **optimizer step은 0회**이며 학습이나 데이터 다운로드를 하지 않는다. [기존 검증 기록](../results/verification-2026-10-28/README.md)은 overlay 환경에서 **281 tests와 67 subtests**를 통과한 결과다. 이것이 새 lock 설치, 실제 사전학습 가중치, 실제 데이터 전체 로딩이나 수렴을 보장하지는 않는다. 새 환경을 검사하면 결과도 별도로 기록한다.

## 4. 로컬 입력 준비 후 실험 시작

이 `PY` 값을 사용해 [실험 안내](../../../../README.ko.md)를 진행한다. `--backbone-weights`에는 이미 보유한 로컬 **PVTv2-B2** 체크포인트를 지정한다. loader는 flat state dictionary 또는 `{"state_dict": ...}`를 받아 `head.weight`, `head.bias`만 제외하고 encoder를 `strict=True`로 로드한다. 다른 backbone이나 일부만 맞는 가중치를 묵시적으로 허용하지 않는다. 이 작업 공간에서 실제 사전학습 파일은 아직 검증하지 않았다.

데이터셋과 가중치는 해당 단계 전에 로컬에 있어야 한다. 무작위 초기화 자체를 비교하려는 경우에만 `--random-init`을 사용하고 모든 비교군의 초기화 조건을 맞춘다. 사용자가 `train` 또는 생성된 `train.sh`를 실행할 때 학습이 시작된다. 환경 설치, `--help`, 위 검사는 학습을 시작하지 않는다.

## 원본 기록과 기존 로컬 환경

| 기록 | 용도 |
| --- | --- |
| [원본 FSA requirements](../../../FSA-ZoomNeXt/requirements.txt) | 변경하지 않고 보존한 원래 실험 의존성 |
| [새 연구 requirements](../requirements.txt) | 새 연구의 직접 의존성 |
| [검증 constraints](constraints-verified.txt) | 로컬 검증에 사용한 버전 |
| [해시 포함 lock](requirements.lock) | 전이 의존성 버전과 허용 배포 파일 해시 |
| [검증 환경 기록](verification.json) | 원래 설치 방법과 패키지 목록 |

기존 `/home/robin/shMin/INS_COD/.venv/fsa-study`는 **로컬 검증용 overlay**다. 가벼운 의존성은 분리해 설치했지만 `.pth` 파일로 `/home/robin/shMin/GeneralGodhand/.venv/isaac`의 PyTorch 패키지를 읽는다. 원본 환경은 수정하지 않았다. 이 절대 경로에 의존하므로 다른 장비의 재현 설치 방법으로 사용할 수 없으며 Isaac 패키지는 이 연구의 의존성이 아니다. 이 실험을 위해 기존 공용 환경을 업그레이드하거나 sync하지 말고 위의 새 독립 환경을 사용한다.

보존한 backbone은 예전 timm import 경로와 PyTorch SDPA context manager에 관한 deprecation warning을 출력한다. 기록된 버전에서 동작을 확인했으며 출처 보존을 위해 원본 소스를 유지했다.
