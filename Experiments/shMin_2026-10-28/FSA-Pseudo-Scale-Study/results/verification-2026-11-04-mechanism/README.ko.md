# 2026-11-04 · Mechanism 브랜치 검증 — 초기화 모델과 실행 경계 확인

[English](README.md) | **한국어** · [실행 안내](../../docs/SCALE-MECHANISM.ko.md)

이 자료는 **코드 검증**이며 COD 학습·성능 평가 결과가 아니다. 데이터셋·사전학습 가중치를 다운로드하지 않았고 optimizer step도 실행하지 않았다. 새 브랜치의 분할 성능 향상은 아직 확인하지 않았다.

| 검사 | 관측 결과 | 근거 |
| --- | --- | --- |
| 전체 회귀 검사 | **813개 테스트 + 하위 검사 74개 통과**, 기존 deprecation 경고 13개 | [pytest 로그](pytest.txt) |
| 이전 model 진입점과 새 기본값 | PVT/CNN 기준·pseudo 4개 구성의 초기 state tensor와 64×64 logits가 정확히 일치 | [모델 검사](models.json), [재현 코드](verify_models.py) |
| GPU 추론 | 5개 구성 × stride 32/16 = 10개 초기화 모델. 384×384·48채널·batch 1에서 유한 출력 | [모델 검사](models.json) |
| 추론 연산량 | 실제 RGB encoder 3회·stride 16 padding을 포함해 10개 graph의 관측 연산 분류 완료 | [모델 검사](models.json) |
| Feature teacher gradient | 두 stride에서 128×128·batch 2, 학생 gradient 유한·연결 확인, teacher 고정, optimizer step 0 | [모델 검사](models.json) |
| 실행·리더보드 | 단계별 4/4/2/2개·seed 하나, 준비 중 run 폴더 없음. 실제 scale 이름·stride/teacher 비교군 분리, teacher hash 확인 | [구조 검사](../../tests/test_scale_mechanism.py), [pipeline 검사](../../tests/test_backbone_pipeline.py) |
| 문서 | 상대 링크·anchor·Bash 문법 확인 | [문서 검사](docs-check.json) |
| 설치 범위 | 설치 대상의 실행 프로세스는 차단, 다른 소스 폴더의 실행은 허용. 백업·입력 보존 확인 | [설치기 검사](../../tests/test_install_study.py) |

GPU 검증 환경은 로컬 **NVIDIA GeForce RTX 5080**, torch 2.11.0+cu130이며 임의 텐서와 초기화 가중치를 사용했다. 사용자 서버에서 teacher를 포함한 batch 4·384×384 학습의 속도·최대 메모리를 검증한 것은 아니다. 실제 manifest·사전학습 checkpoint를 사용한 COD 학습·평가는 사용자가 실행해야 한다.

처음 stride 16 profiling 검사에서 `aten.pad.default`가 미지원으로 남았다. 문서화한 규약에 맞춰 데이터 이동으로 분류하고 floating 연산은 0으로 처리했으며, padding 4개 모드 검사를 추가했다. 실제 지연 측정에는 포함된다. 수정 후 위 전체 검사와 GPU 검사를 통과했다.

Linux 검증 환경에 Windows PowerShell이 없어 새 브랜치 인수를 받는 전송 코드는 검토만 했다. 실제 Windows 실행·인증·SCP는 수행하지 않았다. 소스 ZIP 설치는 로컬 fixture로 검증했다.

이전 model 비교는 이전 `model.py`와 현재의 하위 호환 연산자 구현을 사용한다. 검사한 구성과 소스 hash는 JSON에 기록했다. 모든 과거 checkpoint를 확인했다는 주장은 아니다. 실제 과거 성능 자료는 기존 경로에 보존했다. 초기화 검사 재현 명령은 [영문 기록](README.md), 사용자 학습 명령은 [한국어 실행 안내](../../docs/SCALE-MECHANISM.ko.md#4-ubuntu-bash-core-네-개만-시작)에 있다.
