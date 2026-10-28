# 2026-10-28 · INS_COD — 보조 실험과 모델 비교 코드 정리

[sh-min/COD](https://github.com/sh-min/COD)에 함께 있던 보조 실험과 비교 도구를 연구 계열별로 정리한 저장소다. 각 실험의 모델·설정·학습 방법을 독립적으로 보존하며, 전체를 하나의 실행 패키지로 통합하지 않았다.

main 실험인 COD 루트의 BRG/ZoomNeXt 프로젝트는 COD에 유지한다. `main_for_brg_lite.py`, `main_for_image.py`, `methods/`, `configs/`, `utils/` 등이 해당한다. 별도 폴더로 보관하던 `BRG-ZoomNeXt-Lite-EvalFix/`는 과거 복사본으로 이관하며, 해당 48개 파일은 이관 시점 COD 루트의 대응 파일과 동일하다. **CoCoNet은 사용자 지정에 따라 이관하지 않고 COD에 유지한다.** 범위와 파일별 출처는 [이관 기록](docs/MIGRATION.md) 및 [매니페스트](docs/migration-manifest.json)에 남긴다.

| 구분 | 대표 위치 | 내용 | 이관 원본 파일 수 |
| --- | --- | --- | ---: |
| 신규 비교 실험 | [Pseudo-Scale Study](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/) | 생성12 × 활용13 + 2개 대조군, 논문 근거·공통 학습 및 평가 | 신규 작성 |
| 실험 | [FSA-ZoomNeXt](Experiments/FSA-ZoomNeXt/) | Feature-Space Scale Approximation 모델, 학습·평가·연산량 도구 | 56 |
| 보존 실험 | [BRG-ZoomNeXt-Lite-EvalFix](Experiments/BRG-ZoomNeXt-Lite-EvalFix/) | COD 루트 main의 과거 복사본 | 48 |
| 실험 | [AHDPS-OWOBJ-COD](Experiments/AHDPS-OWOBJ-COD/) | AHDPS와 objectness 관련 모델·설정 | 45 |
| 비교 | [Journal](Benchmarks/Journal/) | FSA case02/case06, ZoomNeXt, ZoomNet 연산량 비교 | 672 |
| 비교 | [ModelCompare](Benchmarks/ModelCompare/) | 여러 모델·변형의 파라미터와 연산량 비교 | 256 |

합계 1,077개는 원본 코드·설정·의존성 기록·실행 스크립트·동반 라이선스 파일 수이며, 새로 작성한 문서는 제외한 수치다. 원본 COD가 이미 추적하던 범위를 이관했으므로 데이터셋, 학습 가중치, 데이터 분할 산출물, 실험 결과·로그, 발표자료, ZIP은 포함되지 않는다. 데이터 로더의 Python 소스는 코드이므로 포함한다.

- [실험 목록](Experiments/README.md): FSA, BRG 보존본, AHDPS 위치
- [연구 변경 기록](Progress/README.md): pseudo-scale 논리 분석과 실험 결정
- [비교 도구 목록](Benchmarks/README.md): 비교 스크립트와 참조 소스
- [실행 진입점과 준비 조건](docs/EXPERIMENTS.md): 모델·설정·의존성 위치
- [학습·비교 실행 명령](docs/RUNNING.md): 절대 경로, 설정 사본, 결과 저장 위치
- [이관 범위와 검증 기록](docs/MIGRATION.md): 이전 경로, 보존·제외 범위, 검증 범위

각 프로젝트는 `methods`, `utils`, `datasets` 같은 같은 이름의 모듈을 사용한다. 실험별 환경과 작업 경로를 구분하고 해당 프로젝트의 `requirements.txt`를 기준으로 준비한다. 기존 Windows/RunPod 경로가 남은 설정은 실제 데이터·가중치 경로를 명시해야 한다. 코드 이관과 경로 검증은 모델 초기화, 학습 완료, 평가 재현을 뜻하지 않으며, 실제 확인 범위는 이관 기록에서 구분한다.

`_profile_repos*`와 `_extracted_sources`는 이번 이관에서 원본 비교 코드의 참조 자료로 보존했다. 이름이 캐시처럼 보여도 코드 수정·변형을 포함하므로 폴더 이름만으로 제거하거나 공통 소스로 합치지 않았다. 제3자 코드의 저작권 표시와 포함된 라이선스를 그대로 보존하며, 저장소 전체에 새로운 공통 라이선스를 부여하지 않는다.
