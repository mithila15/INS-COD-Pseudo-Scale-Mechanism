# 2026-10-28 · INS_COD 실행 — 진입점과 환경·입력·결과 경로

이 문서는 이관한 소스의 CLI와 설정을 기준으로 실행 진입점을 안내한다. 아래 명령은 사용자가 실행할 때 전처리·학습·프로파일링을 시작한다. 이번 정리에서는 데이터 다운로드와 학습을 실행하지 않았다. 실제 검증 범위와 남은 오류는 [이관 기록](MIGRATION.md)에 별도로 남긴다.

| 계열 | 진입점 | 대표 설정 | 필요한 환경·자료 |
| --- | --- | --- | --- |
| FSA-ZoomNeXt | [main_for_fsa.py](../Experiments/FSA-ZoomNeXt/main_for_fsa.py) | [fsa_pvtb2_train.py](../Experiments/FSA-ZoomNeXt/configs/fsa_pvtb2_train.py), [dataset_cod_runpod.yaml](../Experiments/FSA-ZoomNeXt/configs/dataset_cod_runpod.yaml) | [requirements.txt](../Experiments/FSA-ZoomNeXt/requirements.txt)는 PyTorch 제외. 호환되는 torch/torchvision을 별도 준비. 이미지·마스크, 선택한 초기화에 필요한 가중치 준비 |
| BRG 보존본 | [main_for_brg_lite.py](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/main_for_brg_lite.py) | [brg_lite_pvtb2_train.py](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/configs/brg_lite_pvtb2_train.py), [dataset_cod_windows.yaml](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/configs/dataset_cod_windows.yaml) | [requirements_brg_lite.txt](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/requirements_brg_lite.txt), 데이터·가중치. 기본 데이터 경로는 `E:/COD/datasets` |
| AHDPS-OWOBJ-COD | [train.py](../Experiments/AHDPS-OWOBJ-COD/tools/train.py), [eval.py](../Experiments/AHDPS-OWOBJ-COD/tools/eval.py) | [baseline](../Experiments/AHDPS-OWOBJ-COD/configs/cod_ahdps_baseline.yaml), [AHDPS](../Experiments/AHDPS-OWOBJ-COD/configs/cod_ahdps_base.yaml), [objectness](../Experiments/AHDPS-OWOBJ-COD/configs/cod_ahdps_owobj_base.yaml) | [requirements.txt](../Experiments/AHDPS-OWOBJ-COD/requirements.txt), COD 이미지·마스크·split |
| Journal | [v6 비교](../Benchmarks/Journal/compare_cod_params_flops_v6_case06_only_safe.py) | CLI 옵션으로 입력·모델·측정 도구 지정 | torch와 각 참조 프로젝트의 의존성. 선택한 측정 도구에 따라 fvcore/thop 필요 |
| ModelCompare | [v8 비교](../Benchmarks/ModelCompare/measure_compare_sinet_v8.py) | `--official-root`, `--lite-root`, `--fsa-root`, `--sinet-root` 등 | torch, thop 및 모델별 의존성. 미포함 모델과 과거 ZIP 기본 경로를 확인 |

프로젝트 간 의존성 버전은 서로 다를 수 있다. 각 환경에서 torch/torchvision의 장치 호환성을 먼저 맞추고 해당 requirements를 적용한다. 모든 프로젝트를 같은 환경에 설치한 것으로 간주하지 않는다. `--pretrained` 또는 설정의 `pretrained: true`는 모델 구현에 따라 가중치 다운로드를 유발할 수 있으므로 기존 가중치와 캐시 위치를 먼저 확인한다.

## 실행 명령

[FSA·BRG·AHDPS 학습과 Journal·ModelCompare 비교 명령](RUNNING.md)은 별도 실행 안내를 대표 문서로 사용한다. 여기에는 절대 경로 변수, 환경 설치 조건, 기존 데이터·가중치 재사용, 실행용 설정 사본과 새로운 결과 위치를 정리했다.

[저장소 목차](../README.md) · [실험 목차](../Experiments/README.md) · [비교 목차](../Benchmarks/README.md)
