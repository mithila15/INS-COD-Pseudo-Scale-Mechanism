# 2026-11-02 · Pseudo-Scale Study — Full-4040 서버 결과 검토

[English](README.md)

업로드된 스냅샷에는 **8개 실험 중 3개의 완료 결과**가 있습니다. Native-scale 생성과 ASFF/SK fusion은 단일 인코더 baseline보다 CAMO·CHAMELEON의 주요 네 지표를 개선했습니다. COD10K·NC4K는 지표별 결과가 엇갈립니다. **전체 우승 모델을 정할 단계는 아니며**, 단일 seed의 중간 결과로 통계적 유의성을 주장하지 않습니다.

## 스냅샷과 비교 조건

서버 내보내기 메타데이터 시각은 **2026-11-02 13:16:05 KST** (`2026-11-02T13:16:05.387305+09:00`)입니다. 대상 실행은 `priority8-full4040-fixed-20261101_200834`입니다. 진행 중인 실험에서 복사한 보존본이며 현재 서버 상태를 실시간으로 확인한 결과는 아닙니다.

| 실험 | 이 스냅샷에서 확인한 상태 |
| --- | --- |
| `single` | 150/150 epoch, 4개 데이터셋 평가·추론 프로파일 완료 |
| `native_conv__asff` | 150/150 epoch, 4개 데이터셋 평가·추론 프로파일 완료 |
| `native_conv__sk` | 150/150 epoch, 4개 데이터셋 평가·추론 프로파일 완료 |
| `native_conv__mean` | 15/150 epoch까지 학습 기록 존재, 완료 평가·프로파일 없음 |
| `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean` | 계획에 포함, 스냅샷에 실행 기록 없음 |

원본 리더보드의 완료 3개·미시작 5개 표시는 이전 단계 갱신 시점의 상태입니다. [네 번째 실험의 history](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__mean__seed3407/history.jsonl)와 [run.log](../priority8-full4040-fixed-20261101_200834/run.log)에는 15 epoch가 기록되어 있습니다. 과거 상태를 덮어쓰지 않고 원본 리더보드를 보존했습니다.

완료된 세 실험은 PVT-v2-B2 인코더 하나와 공통 decoder, seed **3407**, 입력 **384 × 384**, 학습 batch size **4**, **150 epoch**, 동일한 로컬 backbone 초기 가중치를 사용합니다. 학습 데이터는 **CAMO 1,000 + COD10K 3,040 = 4,040장**이며 validation holdout이 없습니다. **150 epoch의 마지막 체크포인트**를 평가하므로 validation/test 점수로 체크포인트를 고르지 않습니다. 세 실험의 학습 소스·평가 소스·manifest·기록된 초기 가중치 해시가 동일합니다. 설정과 출처는 [원본 실행 폴더](../priority8-full4040-fixed-20261101_200834/runs/priority8/)에 보존되어 있습니다.

## 주요 테스트 지표

실험별 평가 장수는 **CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121**, 총 6,473장입니다. `S`는 Sα, `wF`는 weighted Fβ, `E_mean`은 threshold curve의 평균 enhanced alignment입니다. MAE는 낮을수록, 나머지는 높을수록 좋습니다. 표는 소수점 8자리로 반올림했으며 원본 JSON은 전체 정밀도를 보존합니다. 평가 프로토콜은 기록된 `official_py_sod_metrics_uint8_v2`이며 과거 학습 로그의 float validation MAE와 구분합니다.

| Dataset | Variant | S ↑ | wF ↑ | E_mean ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| CAMO | single (baseline) | 0.83650563 | 0.79112954 | 0.89889561 | 0.05871336 |
| CAMO | native_conv + ASFF | 0.83918770 | 0.79462401 | 0.90248976 | 0.05817316 |
| CAMO | native_conv + SK | 0.83769063 | 0.79331306 | 0.89937418 | 0.05775508 |
| CHAMELEON | single (baseline) | 0.90062112 | 0.85471781 | 0.94287184 | 0.02720466 |
| CHAMELEON | native_conv + ASFF | 0.90348041 | 0.85724452 | 0.94806074 | 0.02487026 |
| CHAMELEON | native_conv + SK | 0.90742662 | 0.86523292 | 0.95109041 | 0.02359833 |
| COD10K | single (baseline) | 0.85513856 | 0.76665013 | 0.92464114 | 0.02567728 |
| COD10K | native_conv + ASFF | 0.85550535 | 0.76559010 | 0.92470970 | 0.02521642 |
| COD10K | native_conv + SK | 0.85504774 | 0.76657850 | 0.92499880 | 0.02555312 |
| NC4K | single (baseline) | 0.87244166 | 0.82299061 | 0.92479790 | 0.03662558 |
| NC4K | native_conv + ASFF | 0.87188360 | 0.82202524 | 0.92544153 | 0.03643222 |
| NC4K | native_conv + SK | 0.87149105 | 0.82227157 | 0.92337927 | 0.03654112 |

원본 JSON에는 데이터셋별 **21개 지표**가 있습니다. E·Fβ의 adaptive/mean/max, IoU, Dice/F1, precision, recall, specificity, BER, FPR, FNR, accuracy, balanced accuracy, boundary IoU를 포함합니다. 외부 논문 표와 비교할 때 `E_mean`을 `E_max`로 대신 사용하면 안 됩니다.

## 현재 결과 해석

- **CAMO:** ASFF가 S·wF·E_mean에서 가장 높고, SK가 MAE에서 가장 낮습니다. ASFF의 baseline 대비 변화는 ΔS **+0.00268207**, ΔwF **+0.00349447**, ΔE_mean **+0.00359415**, ΔMAE **−0.00054020**입니다.
- **CHAMELEON:** SK가 표시한 네 지표 모두 가장 좋습니다. baseline 대비 ΔS **+0.00680550**, ΔwF **+0.01051510**, ΔE_mean **+0.00821857**, ΔMAE **−0.00360633**입니다. 평가 샘플이 76장이라는 점을 함께 봐야 합니다.
- **COD10K:** S·MAE는 ASFF, E_mean은 SK, wF는 baseline이 가장 좋습니다. SK의 S·wF는 baseline보다 각각 **0.00009082**, **0.00007163** 낮아 차이가 작습니다.
- **NC4K:** S·wF는 baseline이 가장 높고, E_mean·MAE는 ASFF가 가장 좋습니다. 두 pseudo-scale 모델 모두 S·wF는 baseline보다 소폭 낮습니다.

두 데이터셋에서 pseudo-scale fusion의 개선 신호가 보이지만, 큰 벤치마크 전반에서 일관된 이점은 아직 확인되지 않았습니다. 생성 방식과 fusion의 영향을 나누어 판단하려면 남은 mean/projection/DySample 실험이 필요합니다. 테스트 순위는 관측 결과를 설명하며 체크포인트 선택이나 종합 우승 점수로 사용하지 않습니다.

## 추론·학습 비용

| Variant | Parameters | GMAC/image | Median ms/image | Images/s | Latency vs baseline | Training hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single (baseline) | 25,318,163 | 13.9843 | 6.3071 | 158.53 | — | 4.744 |
| native_conv + ASFF | 25,351,775 | 14.0980 | 7.7616 | 128.85 | +23.06% | 5.785 |
| native_conv + SK | 25,351,427 | 14.0680 | 7.4474 | 134.30 | +18.08% | 5.703 |

측정 환경은 **NVIDIA RTX PRO 5000 Blackwell**, FP32, batch 1, 384 × 384 synthetic RGB 입력, warm-up 10회·측정 50회입니다. GPU 입력부터 deep/region/edge를 포함한 모든 출력 head까지 측정하며 IO·CPU/GPU 전송은 제외합니다. Images/s는 평균 지연으로 계산하고, 인접한 지연 열은 중앙값입니다. Training hours는 기록된 학습 경과 시간이며 평가까지 포함한 전체 파이프라인 시간은 아닙니다.

ASFF/SK의 MAC 증가는 약 **0.81% / 0.60%**지만 측정 지연은 **23.06% / 18.08%** 늘었습니다. 산술 연산량 증가가 작아도 실제 실행 시간 증가가 작다고 볼 수 없습니다. 이 프로파일의 측정값이며 모든 GPU나 실제 서비스 전체 처리 속도에 대한 주장은 아닙니다. 프로파일의 `protocol.generator/utilizer/baseline`에는 CLI 기본값이 남아 있고, 실제로 로드한 모델 구성은 `metadata.config`에 기록되어 평가 체크포인트 메타데이터와 일치합니다.

## 검증 범위·한계·보존 자료

제공된 per-image CSV와 데이터셋 threshold curve에서 **252개 점수 = 실험 3개 × 데이터셋 4개 × 지표 21개**를 독립적으로 재집계했습니다. `metrics.json`과의 최대 절대 오차는 **2.22 × 10⁻¹⁶**입니다. 완료된 각 실험에는 서로 다른 데이터셋/이미지 ID가 6,473개 있고, 세 실험의 ID 집합과 지표별 리더보드 순서도 일치합니다. 이는 산출물 내부의 일관성 검증입니다. 원본 이미지·GT 데이터와 체크포인트 가중치는 전송에서 제외되어 **GT 픽셀을 사용한 재평가가 아니며**, 이 검토 과정에서 학습이나 모델 추론을 실행하지 않았습니다.

ZIP 전체 CRC 검사가 통과했습니다. 새 완료 실험 3개의 **기록된 산출물 SHA256 19,509/19,509개**가 ZIP 압축 해제 바이트와 일치합니다. 예측 PNG 19,419개와 기록된 curve·CSV·JSON을 포함합니다. 전송 바이트를 검증한 것이며 GT 정확도나 생성자 신원 인증은 아닙니다. [산출물 해시 감사](artifact-audit.json)와 [산출물 목차](asset-index.md)를 참고하세요.

원본 산출물에 기록된 다음 한계는 유지합니다.

- **seed 3407 하나:** 여러 seed에 대한 변동성·통계적 유의성을 평가하지 않았습니다.
- **COD10K train/test 중복 이미지 7쌍:** `acknowledged-cod10k-v1` 정책으로 포함했습니다. train/test 영향 테스트 이미지 7개, validation/test 0개입니다. 보고 점수는 중복 샘플을 포함하며 완전히 분리된 테스트셋의 증거가 아닙니다.
- **CHAMELEON 이미지/마스크 크기 불일치 2개:** `animal-62`, `animal-74`입니다. GT를 바꾸지 않고 예측을 GT 크기에 맞추는 정책이며, 공간 정합 자체를 별도로 검증하지 않았습니다.
- 과거 3,636장 holdout 실험은 학습 장수와 체크포인트 선택 규칙이 다릅니다. 별도로 보존하며 이번 비교에 섞지 않습니다. ZoomNeXt와 4,040장이라는 장수가 같다고 전체 학습 조건까지 같아지는 것은 아닙니다.

[GitHub Release 보존본](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102)에 검증한 원본 ZIP과 체크섬을 보존합니다. 현재·이전 실험의 **예측 PNG 54,995개 전체**, curve 산출물, 코드·결과·로그를 포함합니다. Git 저장소에는 바로 읽을 수 있는 결과·메타데이터·curve 산출물을 두고, 예측 마스크는 ZIP에 보존합니다. 요청한 대로 원본 데이터셋·가중치·가상환경은 제외했습니다.

전송 범위의 **결과 외 소스·문서 114개 경로**는 모두 있으며, 복구 기준 커밋 `e79c8d477c7a5b0954d8321e4bd838ce52fc259f`와 CRLF/LF 줄바꿈을 정규화한 뒤 일치합니다. 4개는 바이트까지 같고, 110개는 줄바꿈만 다릅니다. 이번 업로드에 병합할 의미상 코드 수정은 없어 기존 Git 소스를 유지합니다. 이전 상태가 남은 리더보드를 포함하여 원본 결과는 보존합니다.

- [Original leaderboard](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.md) · [JSON](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.json) · [HTML file](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.html).
- [single metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/single__seed3407/test-four-benchmarks/metrics.json) · [ASFF metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__asff__seed3407/test-four-benchmarks/metrics.json) · [SK metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__sk__seed3407/test-four-benchmarks/metrics.json).
- [ASFF profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__asff__seed3407/profile.json) · [SK profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__sk__seed3407/profile.json) · [single profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/single__seed3407/profile.json).
- [Full-precision analysis and baseline deltas](analysis-results.json) · [Protocol and aggregation audit](protocol-audit.json) · [Source and archive inventory](source-audit.json) · [Archive integrity audit](archive-audit.json).
