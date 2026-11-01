# 2026-11-01 · COD 비교 조건 — COD10K 누락 복구와 전체 데이터 학습

## 질문과 확인한 근거

사용자는 GitHub에 올린 priority8의 성능을 확인하는 과정에서 테스트 데이터셋 하나가 빠졌는지, ZoomNeXt와 학습 이미지 수가 같은지 확인하고 **문서와 실제 코드를 수정해 업로드**하도록 요청했다. 이후 Windows에서 pull한 수정본을 SCP로 실험 서버에 전송하는 명령도 요청했다.

- [업로드된 priority8 보고서](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-report-20261101_172449/README.ko.md)는 seed 3407의 8개 run이 완료됐지만 CAMO 250·CHAMELEON 76·NC4K 4,121만 포함한다. COD10K-test 2,026장은 없다.
- 기존 split 구현과 사용자가 보여준 validation 개수·epoch당 909 step은 학습 CAMO 900 + COD10K 2,736 = 3,636장, validation 404장과 일치한다. 업로드된 summary에는 manifest 해시가 있으며 서버 원본 manifest 전체를 직접 확인한 것은 아니다.
- [공식 ZoomNeXt 배포 설정](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/configs/icod_train.py#L6-L60)은 전체 4,040장, batch 4, 150 epochs다. [논문 arXiv v3 §4.1](https://arxiv.org/html/2310.20208v3#S4.SS1)은 batch 8이라고 기술하므로 둘을 구분했다.
- 입력 인코더 횟수·채널·증강·손실·AMP·평가 resize·패키지 버전도 다르다. 전체 데이터 사용만으로 ZoomNeXt와 동일한 조건이 되지 않는다. [영어 비교 표](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md), [한국어 비교 표](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md)에 공식 근거를 연결했다.

## 결정과 코드 변경

1. **누락된 테스트 평가와 새 학습을 분리한다.** `complete_benchmark.py`는 완료 체크포인트와 원 학습 분할을 재사용한다. 원본 sample의 동일성·분할·파일 해시를 유지하고 COD10K test만 추가하는 증명을 보존하며, 새 출력 폴더에서 4개 데이터셋을 동일한 현재 evaluator로 다시 평가하도록 plan을 만든다. 생성만으로 평가·학습을 시작하지 않는다. 이 plan은 평가 전용이며 train/profile action을 거부한다.
2. **새 준비에서 누락이 기본 동작이 되지 않게 한다.** `prepare` CLI는 4개 표준 테스트셋을 모두 요구한다. 의도적인 일부 선택은 명시할 수 있지만 리더보드는 실행 완료와 표준 벤치마크 완료를 별도로 표시한다. 일부 점수를 0으로 채우지 않는다.
3. **`holdout`을 보존하고 `full_train`을 추가한다.** 기존 3,636/404와 validation MAE 선택은 유지한다. `prepare_full_training.py`는 원래 holdout 404장만 train으로 옮긴 새 manifest와 별도 결과 root를 만든다. 4,040장·validation 없음·최종 epoch 고정 선택을 명시하며 `best_val_mae`는 null이다. 기존 8개 변형·seed·초기 backbone·150 epochs·batch 4·나머지 설정은 보존한다. 새 모델은 기존 학습 checkpoint에서 이어 학습하지 않는다.
4. **서버의 로컬 수정과 결과를 보존한다.** Windows 전송 스크립트는 GitHub `main`의 커밋된 study와 모델 import에 필요한 형제 디렉터리 `Experiments/FSA-ZoomNeXt`, 이관 출처 `docs/MIGRATION.md`를 ZIP+체크섬으로 전송한다. 로컬 데이터셋·환경·체크포인트 가중치는 포함하지 않는다. 서버의 별도 release 경로에서 `--rebind-entrypoints`로 검증한 plan의 실행 파일 경로를 새 소스로 옮긴다. 기존 checkout·환경·가중치·결과는 덮어쓰지 않는다.

원본 업로드 보고서 8개 파일은 이번 변경에서 수정하지 않는다. 기존 3개 데이터셋 실측과 새 코드의 회귀 검증을 별도 산출물로 유지한다. 추가 모델이나 데이터셋을 다운로드하지 않았다.

## 결과와 해석 범위

현재 보고서는 1개 seed의 3개 테스트셋 결과다. 예를 들어 CAMO에서 `projection__mean`의 S/wF/MAE는 0.8413/0.7921/0.0570, `single`은 0.8327/0.7825/0.0605다. 그러나 `projection`은 동일 해상도 capacity 대조군이므로 이를 pseudo-scale resampling의 효과로 해석하지 않는다. NC4K에서는 `single`의 wF/MAE 0.8239/0.0364가 해당 보고서의 비교군보다 좋다. COD10K 누락·seed 1개·서로 다른 학습 소스 이력 때문에 전체 벤치마크 우승이나 통계적 유의성을 주장하지 않는다. 수치는 [원본 summary](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-report-20261101_172449/summary.json)의 실측을 반올림했다.

4,040장 학습은 비교 조건을 개선하지만 10% 데이터 차이가 기존 성능 차이의 원인임을 입증하지는 않는다. 새 실험은 테스트 점수로 epoch를 고르지 않는다. 최종 모델 비교를 더 엄밀히 하려면 공식 ZoomNeXt도 데이터·평가 조건을 명시해서 실행하고 seed를 늘려야 한다.

## 검증과 다음 실행

[검증 산출물](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-benchmark/README.md)에 최종 **694 tests + 74 subtests**, 브라우저 8개 검사와 로그를 보존한다. 초기 회귀 검사 실패와 수정 기록도 별도 보존한다. 합성 입력과 추론/gradient 검사이며 optimizer 업데이트는 0회다. 기존 모델을 이용한 실제 서버의 4개 테스트셋 평가, full-data 학습, 새 성능 점수는 아직 실행하지 않았다. Windows/SCP 실전송도 사용자 장비에서 확인해야 한다.

학습 진입 조건은 기존 데이터·backbone 경로와 원 plan이 유효하고, COD10K test 2,026장을 포함한 4개 데이터셋 검사와 학습 이미지/GT geometry 검사를 통과하는 것이다. 테스트 geometry는 원 GT를 보존하고 예측을 그 크기로 복원하는 기존 정책을 사용한다. 이 크기 검사가 실제 이미지/GT의 의미적 정합까지 증명하지는 않는다.

사용자 실행 순서와 절대 경로를 포함한 명령: [Windows pull·SCP → 기존 모델 평가 → 별도 전체 학습](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md). 학습 시작 명령을 안내한 것은 학습을 실행했다는 의미가 아니다.

[당일 목차](../README.md) · [Progress 목차](../../README.md)
