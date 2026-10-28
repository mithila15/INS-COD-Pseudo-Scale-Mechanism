# 2026-10-28 · INS_COD 실행 안내 — 분리된 실험과 비교 코드

이 문서는 기존 설정을 보존하면서 새 경로에서 실행할 명령을 제공한다. 저장소 정리 과정에서는 학습·평가·모델 가중치 다운로드·데이터셋 다운로드를 실행하지 않았다. 아래 명령은 사용자가 환경과 입력을 준비한 뒤 직접 실행한다.

## 실행 전 준비

- 각 프로젝트의 `requirements.txt`는 원본 의존성 기록이며, 이번 이전에서 새로 검증한 환경 잠금 파일이 아니다. FSA, BRG, AHDPS, 비교 코드는 별도 환경을 권장한다. 특히 오래된 ZoomNet 요구 버전을 다른 프로젝트 환경에 일괄 설치하지 않는다.
- 학습에는 해당 GPU와 호환되는 PyTorch·torchvision 설치가 필요하다. FSA의 `requirements.txt`에는 PyTorch 설치가 포함되지 않는다. BRG의 원본 요구사항과 현재 PyTorch 환경의 호환성도 따로 확인한다. 아래의 `.venv/.../bin/python`은 사용자가 준비할 환경 경로이며, 이 저장소가 환경 자체를 포함하지 않는다.
- 데이터셋과 학습 체크포인트는 이 이전에 포함하지 않았다. 기존 COD10K/CAMO/CHAMELEON/NC4K의 이미지와 마스크를 재사용한다. `COD_DATASETS_ROOT`를 실제 데이터 위치로 바꾼다. 일부 구현은 다른 폴더 배치를 탐색하지만, 아래 배치를 기준으로 확인한다.
- FSA와 BRG의 `--pretrained`는 PVT-v2 backbone 가중치를 사용한다. 캐시에 없으면 실행 시 다운로드가 발생할 수 있다. AHDPS의 현재 기본 모델에는 사전학습 체크포인트 입력이 없다.
- 학습·분할 생성·평가 전 실제 입력 수, 이미지/마스크 대응, 필요한 edge/instance 정보와 저장 공간을 확인한다. 이전 검증은 실제 데이터 로딩·GPU 초기화·학습 성능 검증을 대신하지 않는다.

```text
COD_DATASETS_ROOT/
  COD10K-v3/Train/{Image,GT_Object,GT_Edge,GT_Instance}/
  COD10K-v3/Test/{Image,GT_Object,GT_Edge,GT_Instance}/
  CAMO-V.1.0-CVIU2019/Images/{Train,Test}/
  CAMO-V.1.0-CVIU2019/GT/
  CHAMELEON/{Imgs,GT}/
  NC4K/{Imgs,GT}/
```

원본 설정에는 CHAMELEON `Edge`, NC4K `Instance` 경로도 있다. 이미지/마스크와 달리 이러한 부가 입력의 필수 여부와 생성·대체 처리는 실험별로 다르므로 원본의 데이터 로더를 따른다.

다음 공통 변수 블록을 먼저 실행한다. 이후 블록은 **같은 터미널에서 이 변수를 사용**하며 현재 작업 디렉터리에 의존하지 않는다. `RUN_ID`를 실행마다 바꾸어 기존 결과와 구분한다. 결과는 각 실험·비교 프로젝트의 `results/$RUN_ID/` 아래에 보존한다.

```bash
export INS_ROOT="/home/robin/shMin/INS_COD"
export COD_DATASETS_ROOT="/home/robin/datasets/COD"
export DATASETS_ROOT="$COD_DATASETS_ROOT"
export RUN_ID="2026-10-28-migration-run01"
export FSA_ROOT="$INS_ROOT/Experiments/FSA-ZoomNeXt"
export BRG_ROOT="$INS_ROOT/Experiments/BRG-ZoomNeXt-Lite-EvalFix"
export AHDPS_ROOT="$INS_ROOT/Experiments/AHDPS-OWOBJ-COD"
export FSA_PYTHON="$INS_ROOT/.venv/fsa/bin/python"
export BRG_PYTHON="$INS_ROOT/.venv/brg/bin/python"
export AHDPS_PYTHON="$INS_ROOT/.venv/ahdps/bin/python"
export PROFILE_PYTHON="$INS_ROOT/.venv/profile/bin/python"
```

`/home/robin/datasets/COD`는 지정 예시이며 이번 정리에서 존재하거나 완전한 데이터셋으로 검증된 위치가 아니다. 위 Python 경로도 실제 준비한 환경에 맞춘다. Python/PyTorch를 먼저 준비한 뒤 각 환경에서 필요한 프로젝트 의존성을 설치할 때의 명령은 다음과 같다. 이는 패키지 설치 명령이며 데이터셋 다운로드나 학습 명령이 아니다.

```bash
"$FSA_PYTHON" -m pip install -r "$FSA_ROOT/requirements.txt"
"$BRG_PYTHON" -m pip install -r "$BRG_ROOT/requirements_brg_lite.txt"
"$AHDPS_PYTHON" -m pip install -r "$AHDPS_ROOT/requirements.txt"
```

비교 환경은 실행할 모델의 원본 요구사항과 비교 도구를 참고해 별도로 준비한다. Journal의 기본 `hook` 측정과 ModelCompare의 측정에 필요한 PyTorch·torchvision·timm 등 모델 의존성을 확인한다. 프로젝트마다 요구 버전이 다르므로 아래 비교 명령의 성공을 환경 설치 완료로 간주하지 않는다.

## FSA-ZoomNeXt 학습

재사용 입력은 위 데이터셋과 선택한 사전학습 backbone 가중치이다. 실행 위치는 [FSA-ZoomNeXt](../Experiments/FSA-ZoomNeXt/)이며 원본 `fsa_pvtb2_train.py`와 `dataset_cod_runpod.yaml`을 사용한다. `DATASETS_ROOT`는 YAML의 `${DATASETS_ROOT}` 문자열을 실행 시 확장하는 데 사용한다.

```bash
export FSA_RUN_ROOT="$FSA_ROOT/results/$RUN_ID"

"$FSA_PYTHON" "$FSA_ROOT/main_for_fsa.py" \
  --config "$FSA_ROOT/configs/fsa_pvtb2_train.py" \
  --data-cfg "$FSA_ROOT/configs/dataset_cod_runpod.yaml" \
  --model-name PvtV2B2_FSA_ZoomNeXt \
  --pretrained \
  --output-dir "$FSA_RUN_ROOT" \
  --device cuda:0 \
  --info "$RUN_ID"
```

이 명령은 **학습과 설정에 따른 주기적 테스트 평가**를 실행한다. 원본 설정은 COD10K train+CAMO train, 150 epochs, 384×384 입력이다. 테스트 대상은 CHAMELEON/CAMO/COD10K/NC4K이며 기본 평가 간격은 5 epochs이다. 원본의 best 선택 지표는 `cod10k_te.wF`이므로 독립 검증셋만 사용하는 모델 선택 방식으로 해석하면 안 된다.

새 결과는 `$FSA_RUN_ROOT/<자동 구성된 실험명>/` 아래에 저장된다. `state_latest.pth`, `state_best.pth`, 전체 학습 상태 및 평가 요약은 원본 경로 생성 규칙을 따른다. 원본 shell wrapper에는 환경 자동 수정용 preflight도 있으므로, 위 명령은 필요한 환경을 미리 준비한 뒤 Python 진입점을 직접 호출한다.

## BRG-ZoomNeXt-Lite-EvalFix 학습

[BRG 실험](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/)의 `dataset_cod_windows.yaml`은 `E:/COD/datasets` 경로를 담고 있고 환경변수를 확장하지 않는다. 아래 블록은 **데이터 경로만 바꾼 실행용 YAML 사본을 생성**한다. 원본 설정과 모델 알고리즘은 그대로 사용한다.

```bash
export BRG_RUN_ROOT="$BRG_ROOT/results/$RUN_ID"
export BRG_DATA_CONFIG="$BRG_RUN_ROOT/dataset.yaml"

"$BRG_PYTHON" - <<'PYTHON'
import os
from pathlib import Path
import yaml

source = Path(os.environ["BRG_ROOT"]) / "configs/dataset_cod_windows.yaml"
target = Path(os.environ["BRG_DATA_CONFIG"])
data_root = Path(os.environ["COD_DATASETS_ROOT"])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
original_prefix = "E:/COD/datasets/"
for name, info in cfg.items():
    old_root = info["root"].replace("\\", "/")
    if not old_root.startswith(original_prefix):
        raise ValueError(f"Unexpected original root for {name}: {old_root}")
    info["root"] = str(data_root / old_root[len(original_prefix):])
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
print(target)
PYTHON
```

위 명령은 같은 `dataset.yaml`이 있으면 덮어쓰지 않고 중단한다. 아래 명령은 **학습과 원본 설정에 따른 주기적·최종 테스트 평가**를 시작한다.

```bash
"$BRG_PYTHON" "$BRG_ROOT/main_for_brg_lite.py" \
  --config "$BRG_ROOT/configs/brg_lite_pvtb2_train.py" \
  --data-cfg "$BRG_DATA_CONFIG" \
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite \
  --pretrained \
  --output-dir "$BRG_RUN_ROOT/outputs" \
  --device cuda:0 \
  --info "$RUN_ID"
```

재사용 입력은 기존 COD10K/CAMO 학습 이미지·마스크와 backbone 가중치이며, 테스트는 CHAMELEON/CAMO/COD10K/NC4K를 사용한다. 결과는 `$BRG_RUN_ROOT/outputs/<자동 구성된 실험명>/` 아래에 저장된다. 원본 best 선택 지표는 `cod10k_te.wF`이다. 원본 shell script도 Windows 데이터 YAML을 사용하므로 Linux에서는 위 실행용 사본을 지정해야 한다.

## AHDPS-OWOBJ-COD 학습

[원본 OWOBJ 설정](../Experiments/AHDPS-OWOBJ-COD/configs/cod_ahdps_owobj_base.yaml)은 결과 경로가 `${PROJECT_ROOT}/results`로 고정되어 있고 결과 루트용 CLI 옵션이 없다. 따라서 아래 첫 블록은 **원본을 읽어 새 실행용 설정 사본만 생성**한다. 데이터 경로는 기존 `COD_DATASETS_ROOT`를 사용하고 split과 결과 경로를 이번 run 폴더로 분리한다. 원본 설정 파일은 수정하지 않는다.

설정 사본이 `configs/` 밖에 생기므로 `project.root`를 실제 프로젝트 절대 경로로 명시한다. 이 구현은 `project.root`를 변수 확장보다 먼저 해석하므로 이 필드에 `${PROJECT_ROOT}` 문자열을 쓰지 않는다.

```bash
export AHDPS_RUN_ROOT="$AHDPS_ROOT/results/$RUN_ID"
export AHDPS_CONFIG="$AHDPS_RUN_ROOT/config.yaml"

"$AHDPS_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

project = Path(os.environ["AHDPS_ROOT"])
run_root = Path(os.environ["AHDPS_RUN_ROOT"])
target = Path(os.environ["AHDPS_CONFIG"])
source = project / "configs/cod_ahdps_owobj_base.yaml"
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
cfg["project"]["root"] = str(project)
cfg["project"]["name"] = "ahdps-owobj-" + os.environ["RUN_ID"]
cfg["paths"]["datasets"]["splits_root"] = str(run_root / "splits")
for key in cfg["paths"]["results"]:
    cfg["paths"]["results"][key] = str(run_root if key == "root" else run_root / key)
run_root.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
print(target)
PY
```

위 명령은 이미 같은 `config.yaml`이 있으면 덮어쓰지 않고 중단한다. 새 실행에는 `RUN_ID`를 바꾼 뒤 공통 변수부터 다시 정의한다. 이어서 아래 명령이 **split 생성·학습·개발셋 평가·마지막 테스트 평가**를 실행한다.

```bash
"$AHDPS_PYTHON" "$AHDPS_ROOT/tools/train.py" \
  --config "$AHDPS_CONFIG"
```

원본 설정은 train/dev 분할 비율 0.1, 50 epochs, 384×384 입력이며 CUDA와 AMP를 사용한다. 체크포인트는 `$AHDPS_RUN_ROOT/checkpoints/`, 학습 기록과 마지막 평가 수치는 `$AHDPS_RUN_ROOT/metrics/`에 저장된다. 모델 초기화·학습·전체 데이터 평가 성공 여부는 이번 이전에서 확인하지 않았다.

## Journal 모델 복잡도 비교

다음은 **bundled 소스와 합성 입력으로 파라미터 수·MACs/FLOPs를 측정**하는 명령이다. 데이터셋이나 학습 체크포인트가 필요하지 않으며 학습·정확도 평가는 하지 않는다. 원본의 `E:\COD\Journal\*.zip` 기본값을 대신해 모두 보존된 소스 폴더를 지정한다.

```bash
export JOURNAL_ROOT="$INS_ROOT/Benchmarks/Journal"
export JOURNAL_RUN_ROOT="$JOURNAL_ROOT/results/$RUN_ID"
export PROFILE_SOURCES="$JOURNAL_ROOT/_profile_repos_v6_case06_only_safe"

"$PROFILE_PYTHON" "$JOURNAL_ROOT/compare_cod_params_flops_v6_case06_only_safe.py" \
  --fsa-root "$PROFILE_SOURCES/FSA-ZoomNeXt_case02/FSA-ZoomNeXt-Case02-Only" \
  --fsa-case06-root "$PROFILE_SOURCES/FSA-ZoomNeXt_case06_only/FSA-ZoomNeXt-Case06-Only" \
  --zoomnet-root "$PROFILE_SOURCES/ZoomNet-main/ZoomNet-main" \
  --zoomnext-root "$PROFILE_SOURCES/ZoomNeXt-main/ZoomNeXt-main" \
  --require-fsa-case06 \
  --extract-dir "$JOURNAL_RUN_ROOT/extracted" \
  --out-dir "$JOURNAL_RUN_ROOT" \
  --height 384 --width 384 --batch-size 1 \
  --device cpu --tool hook --input-mode ones \
  --stop-on-error
```

`--load-pretrained`를 지정하지 않으므로 측정용 모델은 사전학습 가중치를 요청하지 않는 설정이다. 결과는 `$JOURNAL_RUN_ROOT/`에 저장된다. 모델별 입력 scale과 FLOPs/MACs 집계 기준은 측정 코드와 결과 메타데이터를 함께 확인한다.

## ModelCompare 비교

아래는 보존된 ZoomNeXt, BRG pathfix 변형, ChangeModule 변형과 FSA case02를 지정한다. `--lite-root`는 **ModelCompare에 포함된 과거 비교용 변형**이며 COD 루트의 현재 main 실험을 가져오는 명령이 아니다. FSA-Mobile과 일반 FSA 항목은 이 비교에서 제외한다.

```bash
export COMPARE_ROOT="$INS_ROOT/Benchmarks/ModelCompare"
export COMPARE_RUN_ROOT="$COMPARE_ROOT/results/$RUN_ID"
export COMPARE_SOURCES="$COMPARE_ROOT/_extracted_sources"
export CASE02_ROOT="$INS_ROOT/Benchmarks/Journal/_profile_repos_v6_case06_only_safe/FSA-ZoomNeXt_case02/FSA-ZoomNeXt-Case02-Only"
mkdir -p "$COMPARE_RUN_ROOT"

"$PROFILE_PYTHON" "$COMPARE_ROOT/measure_compare_case02_v6.py" \
  --official-root "$COMPARE_SOURCES/ZoomNeXt-main/ZoomNeXt-main" \
  --lite-root "$COMPARE_SOURCES/BRG-ZoomNeXt-Lite_runpod_pathfix/BRG-ZoomNeXt-Lite-EvalFix" \
  --change-root "$COMPARE_SOURCES/ChangeModule/BRG-ZoomNeXt-Lite-EvalFix" \
  --fsa-case02-root "$CASE02_ROOT" \
  --skip-fsa --skip-fsa-mobile \
  --device cpu --h 384 --w 384 \
  --strict-paths \
  --csv "$COMPARE_RUN_ROOT/case02-v6.csv"
```

이 명령 역시 **모델 복잡도 비교**이며 데이터셋 정확도 평가나 학습이 아니다. `--pretrained`를 지정하지 않는다. CPU 측정이라도 모델 초기화의 CUDA 조회나 라이브러리 호환성 문제가 남을 수 있다. 실제 모델 forward를 실행해 확인한 명령은 아니며, 문제가 발생하면 오류 기록과 해당 환경을 보존한다.

## 보존된 과거 버전의 실행 제한

BRG의 데이터 YAML과 일부 과거 실행 스크립트에는 `E:\COD\...` 절대 경로가 남아 있다. ModelCompare/Journal의 Python 기본 인자에도 과거 Windows ZIP 경로가 있다. 이번 분리에서는 원본 조건을 보존했으며, 이 문서는 지원되는 CLI·환경변수 또는 실행용 설정 사본으로 새 경로를 지정한다. 과거 버전을 실행하려면 해당 버전의 설정 로더 지원 범위를 확인하고 실행용 사본에서 데이터·프로젝트·결과 경로를 수정한다.

소스 구문·경로 보존 확인과 실제 모델 로딩·전처리·학습 완료·학습 모델 평가를 구분한다. 본 문서의 명령 제공은 학습 실행 완료나 성능 재현을 뜻하지 않는다.
