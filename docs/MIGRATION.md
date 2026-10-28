# 2026-10-28 · COD 분리 기록 — main·CoCoNet 유지와 보조 실험 이관

## 범위와 근거

출처는 [sh-min/COD의 d3a4822 커밋](https://github.com/sh-min/COD/tree/d3a482289f66921487a33c9965e6c37cfdab3cb3)이다. 최초 커밋 `d7fb80a`의 루트 BRG/ZoomNeXt 실험과 이후 외장 드라이브에서 가져온 폴더들이 섞여 있어, 사용자가 루트 실험 유지와 추가 폴더 분리를 요청했다. 이어서 CoCoNet은 이번 업로드에서 제외하도록 지정했다.

COD 루트의 실행 코드·설정·의존성 파일과 `CoCoNet/` 전체를 COD에 남긴다. 기존 Git 이력을 다시 쓰거나 브랜치를 합치지 않는다. INS_COD에서 보조 실험을 정리하고 업로드 상태를 확인한 뒤 COD의 이관 대상 폴더를 제거하는 순서로 진행한다. COD의 README는 새 위치를 안내하도록 갱신한다.

| COD의 이전 경로 | INS_COD의 대표 위치 | 원본 파일 수 |
| --- | --- | ---: |
| `AfterComment/FSA-ZoomNeXt/` | [Experiments/FSA-ZoomNeXt](../Experiments/FSA-ZoomNeXt/) | 56 |
| `BRG-ZoomNeXt-Lite-EvalFix/` | [Experiments/BRG-ZoomNeXt-Lite-EvalFix](../Experiments/BRG-ZoomNeXt-Lite-EvalFix/) | 48 |
| `Using_SOTA/AHDPS_OWOBJ_COD/` | [Experiments/AHDPS-OWOBJ-COD](../Experiments/AHDPS-OWOBJ-COD/) | 45 |
| `Journal/` | [Benchmarks/Journal](../Benchmarks/Journal/) | 672 |
| `ModelCompare/` | [Benchmarks/ModelCompare](../Benchmarks/ModelCompare/) | 256 |

이관 원본은 총 **1,077개, 5,223,236바이트**다. 각 프로젝트 내부의 파일 내용·상대 계층·실행 권한을 보존했다. 새 실험 간 공용 모듈을 만들거나 모델·손실·학습 방식을 변경하지 않았다. 원본의 main 실험과 CoCoNet에 해당하는 1,212개 파일도 매니페스트에서 별도 구분한다. 매니페스트의 CoCoNet 항목은 보존 위치와 원본 해시를 기록한 메타데이터이며 CoCoNet 소스 자체는 INS_COD에 포함하지 않는다.

독립된 BRG 폴더 48개 파일은 이관 시점 COD 루트의 대응 파일과 모두 동일하다. 사용자가 추가 폴더 전체를 포함하도록 지정했으므로 이 폴더도 과거 보존본으로 이관한다. main의 대표 위치는 계속 COD 루트다. ModelCompare 내부의 BRG pathfix·ChangeModule 변형은 다른 코드이므로 이 동일성 판단을 적용하지 않는다.

`_profile_repos*`, `_extracted_sources`는 기존 Git에 추적된 참조 소스다. 동일해 보이는 코드도 과거 실험의 문맥과 버전을 유지하기 위해 보존했다. 포함된 ZoomNet LICENSE 4개와 소스 내 저작권 표시도 그대로 보존했다. 새로운 공통 라이선스는 부여하지 않았다.

원본 README가 기록한 최초 복사 출처는 `/media/robin/INS/COD`이다. 이 경로는 역사적 출처이며, 이번 작업에서는 해당 드라이브가 없어 **GitHub에 이미 올라간 파일만** 대상으로 삼았다. 원본에 없는 데이터셋·학습 체크포인트·분할 산출물·결과·로그·발표자료·ZIP을 이번에 추가하거나 복원하지 않았다.

## 파일 보존과 재검사

[migration-manifest.json](migration-manifest.json)은 원본 2,289개 파일 전체의 처리 범위를 기록한다. 각 항목에는 원본 경로, 원본 Git blob, 실행 모드, SHA-256, 바이트 수와 처리 방식이 있다. 이동 파일에는 새 경로, 중복 BRG에는 COD의 대응 경로도 기록했다. 이 해시는 이관 기준 원본 값이며 앞으로 소스를 수정할 때 갱신한 값을 과거 원본 값으로 덮어쓰지 않는다.

다음 명령은 데이터나 모델을 불러오지 않고 파일 보존·Python/셸 구문·문서 상대 링크를 검사한다. 현재 작업 디렉터리에 의존하지 않는다.

```bash
python3 /home/robin/shMin/INS_COD/scripts/verify_migration.py \
  --source /home/robin/shMin/COD
```

`--source`를 생략하면 INS_COD 자체만 검사한다. 지정하면 COD의 **고정된 원본 커밋**과 매니페스트의 전체 경로·Git blob을 대조하므로 이후 COD에서 이관 폴더가 제거되어도 재검사할 수 있다. 새 실험 결과가 생긴 뒤의 수정·추가 파일은 별도 변경으로 확인해야 한다.

## 이번 검증 결과

- 이동 파일 1,077개: 원본 SHA-256·Git blob·바이트 수·실행 권한 일치.
- Python 소스 866개: 전체 AST 구문 검사 통과.
- 셸 스크립트 127개: `bash -n` 구문 검사 통과.
- 기준 커밋 파일 2,289개: 이동/유지 범위가 전체 트리와 일치.
- 스테이징한 트리를 임시 폴더에 풀어 같은 검사를 다시 통과했다. 이동한 FSA 학습 설정 읽기와 AHDPS 설정 로더의 새 프로젝트·결과 경로 확장도 확인했다.
- 원본의 CRLF와 기존 후행 공백은 바이트 보존을 위해 유지했다. 새 문서·검증 스크립트에는 Git 공백 오류가 없다.
- 새 문서의 상대 링크와 README 제목 형식 검사. 최종 검사 수치와 오류 목록은 [validation.json](validation.json)에 기록한다.
- [실행 안내](RUNNING.md)의 BRG/AHDPS 설정 생성 코드를 임시 폴더에서 실행했다. BRG 데이터 경로 치환과 AHDPS 원본 설정 로더의 프로젝트·데이터·split·결과 경로 확장을 확인했다. 실제 학습용 데이터는 사용하지 않았다.

현재 기본 Python에는 PyTorch와 timm 등 연구 환경 의존성이 없다. **실제 데이터 로딩, 모델 초기화·forward, GPU 동작, 학습, 체크포인트 평가, 연산량 측정은 미실행**이다. requirements는 기존 실험의 기록을 보존한 것이며 새로 설치·검증한 잠금 파일이 아니다. 학습·데이터셋 다운로드도 시작하지 않았다.

## 실행 시 남은 조건

모델별 환경, 실제 데이터와 필요한 backbone 가중치를 준비해야 한다. BRG의 Windows 데이터 YAML, 비교 스크립트의 과거 ZIP 기본값, AHDPS의 고정 결과 경로는 원본에 남아 있다. [RUNNING.md](RUNNING.md)에서는 새 경로를 지원하는 CLI·환경변수를 사용하고, 필요한 경우 원본을 보존한 실행용 설정 사본을 만든다. 학습 명령, 재사용 입력, 전처리·평가 동작과 새 결과 위치도 이 문서에 구분했다.

[저장소 목차](../README.md) · [실험별 진입점](EXPERIMENTS.md)
