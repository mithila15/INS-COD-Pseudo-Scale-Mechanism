# 2026-10-28 · INS_COD 실험 — 연구 계열과 버전별 목차

각 폴더는 독립적인 연구 실험이다. 원본 프로젝트 내부 구조를 유지해 로컬 import와 설정 상대 경로를 보존했다. main BRG 실험은 [COD](https://github.com/sh-min/COD)에 있다.

| 연구 계열 | 위치 | 시작 파일 | 설정과 의존성 |
| --- | --- | --- | --- |
| FSA-ZoomNeXt | [FSA-ZoomNeXt](FSA-ZoomNeXt/) | [main_for_fsa.py](FSA-ZoomNeXt/main_for_fsa.py) | [configs](FSA-ZoomNeXt/configs/), [requirements.txt](FSA-ZoomNeXt/requirements.txt) |
| BRG 과거 복사본 | [BRG-ZoomNeXt-Lite-EvalFix](BRG-ZoomNeXt-Lite-EvalFix/) | [main_for_brg_lite.py](BRG-ZoomNeXt-Lite-EvalFix/main_for_brg_lite.py) | [configs](BRG-ZoomNeXt-Lite-EvalFix/configs/), [requirements_brg_lite.txt](BRG-ZoomNeXt-Lite-EvalFix/requirements_brg_lite.txt) |
| AHDPS / objectness | [AHDPS-OWOBJ-COD](AHDPS-OWOBJ-COD/) | [tools/train.py](AHDPS-OWOBJ-COD/tools/train.py), [tools/eval.py](AHDPS-OWOBJ-COD/tools/eval.py) | [configs](AHDPS-OWOBJ-COD/configs/), [requirements.txt](AHDPS-OWOBJ-COD/requirements.txt) |

BRG 보존본 48개 파일은 이관 시점 COD 루트의 대응 파일과 동일하다. main의 대표 위치는 COD이며 이 폴더는 원래 별도로 보관하던 실험 사본의 출처를 유지한다. CoCoNet 전체와 과거 분기 실험은 사용자 지정에 따라 이관 범위에서 제외하고 [COD/CoCoNet](https://github.com/sh-min/COD/tree/main/CoCoNet)에 유지한다. 자세한 범위는 [이관 기록](../docs/MIGRATION.md)을 참고한다.

[실행 안내](../docs/RUNNING.md)에서 입력 경로, 환경, 결과 위치를 설정한다. 데이터 준비, 모델 초기화, 학습, 평가는 서로 다른 단계이며 이번 정리 자체가 이 단계의 성공을 보장하지 않는다.

[저장소 목차](../README.md) · [비교 도구](../Benchmarks/README.md)
