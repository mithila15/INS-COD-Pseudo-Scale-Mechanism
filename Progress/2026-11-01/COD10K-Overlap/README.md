# 2026-11-01 · COD10K 분할 무결성 — 확인된 중복 7쌍과 명시적 처리 규약

## 질문과 관측

사용자가 새 코드를 `shMin` 아래에 설치한 뒤 기존 priority8 평가에 COD10K-test를 추가하려 했으나 `Identical image paths or byte hashes cross train/val/test splits` 오류로 준비가 중단됐다. 학습이나 평가가 시작된 것이 아니라 입력 검사 단계에서 중단된 상태다. 기존 코드·데이터를 지우는 대신 정확한 경로·SHA256을 비교하도록 진단했다.

[서버 진단 요약](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/server-audit-summary.json)은 사용자 제공 터미널 기록에서 추출했다. COD10K test 이미지·GT는 각각 2,026개, 대응 누락은 0개였다. 학습 원본과 바이트가 같은 7쌍 중 6쌍은 train, 1쌍은 val이었다. 같은 실제 경로는 없었다. 원 manifest SHA256은 `21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a`다.

## 근거와 판단

[공식 SINet-V2 공지](https://github.com/GewelsJI/SINet-V2/blob/5b9844c94d10a03b815c7c7039c194166a9a8ba3/README.md#L44)는 2021-10-09에 Crab-32와 Cat-1506의 중복을 알렸다. 나머지 Deer·Gecko·Giraffe·Owl 쌍을 포함한 전체 7쌍의 이미지 SHA256은 독립 COD-TDQ 배포 자료의 고정 commit과도 일치했다. [14개 포인터·고정 URL·해시](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/upstream-evidence.json)를 보존한다. 외부에서도 동일한 쌍이 배포됐다는 근거이며, 7쌍 전체에 대한 원저자 공식 공지와 구분한다. 공식 공지의 2장에 대한 지표 영향 추정치를 이번 모델·7장에 일반화하지 않는다.

따라서 사용자 서버의 잘못된 복사만으로 설명할 근거는 없다. 엄격한 무결성 검사가 실제 중복을 발견했지만, 코드에는 배포 벤치마크의 알려진 중복을 명시적으로 기록하면서 실행할 방법이 없었다. 기존 이미지·모델 삭제 또는 무조건 중복 무시는 모두 필요하지 않다.

## 변경과 보존

1. `prepare`와 `complete_benchmark.py`의 기본값은 엄격 거부로 유지한다. 명시한 `--acknowledge-cod10k-overlap`만 정확한 7개 COD10K ID 쌍·이미지 SHA256을 허용한다. 다른 중복·동일한 실제 경로·train/val 중복은 계속 거부한다.
2. 새 manifest에 `overlap_policy="acknowledged-cod10k-v1"`과 실제 `overlap_audit`를 기록하고 평가·리더보드에도 전달한다. `disjoint=false`를 숨기지 않는다. 표준 데이터 개수, 실행 완료, 독립성은 서로 다른 속성이다.
3. `full_train` 준비는 명시한 정책을 이어받고 validation 이동 후 audit를 재계산한다. 이 복사본의 6 train/test + 1 val/test가 7 train/test + 0 val/test로 바뀐다. 실제 이미지·GT를 삭제하거나 배포 개수 4,040/2,026을 바꾸지 않는다.
4. 기존 manifest·체크포인트·학습 기록·3개 데이터셋 보고서를 보존하고 새 평가 폴더와 새 학습 root를 사용한다. `shMin`이 활성 코드·결과 위치이며 기존 `INS_COD` 환경·데이터 경로를 재사용한다. 수정 파일 설치는 체크섬을 확인하고 `.code-backups/`에 보존본을 남긴다.

기존 6장의 최적화 노출과 1장의 validation/체크포인트 선택 노출은 인정 표시로 취소되지 않는다. COD10K 결과는 알려진 중복을 포함한 배포 조건으로 명시해야 한다. 이번 발견으로 기존 CAMO·CHAMELEON·NC4K 점수 전체가 무효가 되는 것은 아니지만 전체 데이터의 시각적 유사 중복까지 검증한 것도 아니다. 독립 분할이 필요한 경우 test 2,019장 부분집합이나 새 train 4,033장 조건을 별도로 설계해야 하며 이번 변경에서 자동 적용하지 않는다.

## 검증 범위와 다음 단계

[영어 근거 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/README.md)와 [한국어 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/README.ko.md)에 evidence 산출물을 연결했다. 사용자 서버 로그를 요약하고 14개 작은 외부 포인터의 해시를 대조했다. 이미지·데이터셋·가중치 다운로드, 학습·optimizer 업데이트, 서버 평가를 실행하지 않았다. 코드 회귀 검증은 이 외부 근거 대조와 구분한다. 로컬 전체 테스트 736개·subtest 74개, 브라우저 표시 5개, 새 임시 경로의 패키지 설치·CLI 4개·합성 CPU 추론을 통과했다. 실제 서버의 새 점수는 아직 계산하지 않았다.

다음 단계는 수정 소스 설치 → 명시적 중복 규약으로 새 평가 plan 준비 → 기존 모델의 4개 데이터셋 평가다. 전체 4,040장 학습은 별도 plan 준비와 사용자의 명시적인 시작 단계다. [서버 절대 경로·준비·평가·학습 명령](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md#확인된-cod10k-중복을-명시적으로-기록하고-진행하기).

[당일 목차](../README.md) · [Progress 목차](../../README.md)
