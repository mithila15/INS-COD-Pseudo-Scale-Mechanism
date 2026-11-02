# 2026-11-02 · Server Results Review — 업로드 복구와 full4040 실측 검토

## 질문과 결정

사용자는 서버에서 학습한 결과와 수정 코드를 로컬로 옮겨 GitHub에 올리고, 그 자료로 성능을 확인하기를 요청했다. 전달 범위는 **코드·결과·로그 전체**, 제외 범위는 **원본 데이터·가중치·가상환경**이다. 채팅 첨부를 사용할 수 없어 GitHub Release를 전달 경로로 사용했다.

일반 Git 트리에는 읽을 수 있는 결과·로그·메타데이터·curve 산출물을, [Release](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102)에는 받은 원본 ZIP과 체크섬 및 **예측 PNG 54,995개 전체**를 보존한다. 기존 실험 결과와 소스는 삭제하거나 덮어쓰지 않는다. [대표 결과](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md)와 [보존 산출물 목록](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/asset-index.md)에서 실제 게시 위치를 연결한다.

## 업로드 실패와 복구 이력

- [486c3e4](https://github.com/sh-min/INS_COD/commit/486c3e4fe5bc24eedb480395191025d56e2521b1)는 `Sync server study code and experiment results`라는 메시지와 달리 **57개 파일·9,046줄 삭제만** 기록했고 새 full4040 결과는 포함하지 않았다.
- [e79c8d4](https://github.com/sh-min/INS_COD/commit/e79c8d477c7a5b0954d8321e4bd838ce52fc259f)에서 삭제된 파일을 복구했다. 이번 전송 ZIP과 복구 기준을 대조하니 결과 외 소스·문서 114개가 모두 있었고, 4개는 바이트까지 같으며 110개는 CRLF/LF 줄바꿈만 달랐다. 의미상 코드 수정은 없어 기존 Git 소스를 유지한다.
- ZIP의 긴 경로는 Windows 추출 문제의 가능한 요인이지만, 당시 추출 실패의 직접 원인을 이 자료만으로 확정하지 않는다. 서버 ZIP에 소스가 누락되었다는 증거는 없다.

[소스·파일 목록 감사](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/source-audit.json)와 원래 커밋을 함께 보존한다. 이번 검토는 결과 정리이며 별도의 학습·추론·데이터 다운로드를 실행하지 않았다.

## 확인한 실제 실험 상태

서버 내보내기 메타데이터 시각은 **2026-11-02 13:16:05 KST**다. `priority8-full4040-fixed-20261101_200834`에는 `single`, `native_conv__asff`, `native_conv__sk`의 **150 epoch 학습·4개 테스트셋 평가·profiling 완료 결과 3개**가 있다. `native_conv__mean`은 **15/150 epoch**까지 기록되어 있다. `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean`에는 실행 history가 없다. 이는 압축 시점 기록이며 현재 서버가 계속 실행 중인지는 확인하지 않았다.

원본 리더보드의 미시작 5개 표시는 네 번째 실험 history보다 앞선 갱신 상태이므로 그대로 보존하고 요약에서 차이를 설명한다. 각 완료 실험은 학습 CAMO 1,000 + COD10K 3,040 = 4,040장, seed 3407, 입력 384², batch 4, 최종 epoch 150 고정 선택이다. 테스트는 CAMO 250·CHAMELEON 76·COD10K 2,026·NC4K 4,121장이다. 과거 3,636/404 holdout의 8개 실험·3개 테스트셋 결과와 섞지 않는다.

## 관측과 검증

CAMO·CHAMELEON에서는 native-scale ASFF/SK의 S·wF·E_mean·MAE가 baseline보다 개선되었다. CHAMELEON SK의 wF는 **0.85471781 → 0.86523292**, MAE는 **0.02720466 → 0.02359833**이다. 반면 COD10K·NC4K에서는 지표별 우열이 다르고 두 pseudo-scale 모델의 wF가 baseline보다 낮다. 전체 우승 모델을 선언하지 않는다. 측정된 추론 중앙값은 baseline **6.3071 ms**, ASFF **7.7616 ms**, SK **7.4474 ms**로 추가 모듈의 시간 비용도 있다. 자세한 조건·수치·변화량은 [영문](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.md)·[한국어](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md) 보고서에 둔다.

검증은 다음 상태를 구분한다.

- **압축 무결성:** ZIP 전체 CRC 검사 통과. [archive audit](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/archive-audit.json).
- **산출물 바이트:** 새 완료 실험 3개의 기록된 SHA256 **19,509/19,509개**가 ZIP 내부 실제 바이트와 일치. 예측 이미지·curve·CSV·JSON을 포함한다. [artifact audit](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/artifact-audit.json).
- **수치 집계:** CSV·데이터셋 threshold curve에서 **252개 점수**를 재집계했고 JSON 대비 최대 오차 **2.22 × 10⁻¹⁶**. 실험마다 평가 ID 6,473개, 세 실험의 ID 집합과 리더보드 순위가 일치. [protocol audit](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/protocol-audit.json).
- **검증하지 않은 것:** 원본 데이터/GT·학습 가중치는 미포함이므로 GT 픽셀 기반 지표 재실행, 실제 모델 tensor 검사, checkpoint 해시 재계산, 서버 현재 프로세스 상태는 검증하지 않았다. 학습 완료는 서버의 보존 산출물에 근거하며 이번 검토가 새 학습 성공을 의미하지 않는다.

## 남은 한계와 다음 단계

Seed 하나라 변동성과 통계적 유의성을 판단할 수 없다. COD10K 중복 7쌍은 과거 holdout에서 train/test 6·validation/test 1이었고, 새 전체 학습에서는 train/test 7·validation/test 0이다. 보고된 COD10K 점수는 중복 7개를 포함하므로 독립 테스트 근거가 아니다. CHAMELEON `animal-62`, `animal-74`는 이미지·마스크 크기가 달라 예측을 원본 GT 크기에 맞췄지만 의미적 정합은 미검증이다.

다음 단계는 서버에서 기존 실행 상태를 확인하고, 사전 지정된 `native_conv__mean`, `projection__mean`, DySample 세 실험의 결과를 같은 조건으로 확보하는 것이다. 이미 실행 중이면 중복 실행하지 않는다. 중단된 부분 학습은 자동 resume이 지원되지 않으므로 기존 결과를 보존하고 별도 경로를 준비해야 한다. 이번 관측으로 테스트 점수에 맞춰 계획을 바꾸거나 나머지 비교를 생략하지 않는다. 전체 8개가 완료되면 같은 평가 규약으로 다시 요약하고, 재현성 판단을 위해 seed 반복을 별도 계획한다.

[당일 목록](../README.md) · [Progress 목록](../../README.md)
