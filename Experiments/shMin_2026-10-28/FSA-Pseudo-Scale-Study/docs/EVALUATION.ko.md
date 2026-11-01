# 2026-11-01 · 평가 — 평가지표·추론 비용·리더보드

[English](EVALUATION.md) | **한국어** | [실행 안내](../../../../README.ko.md)

이 문서는 실제 COD 성능이 아니라 구현한 평가 규약이다. 새 테스트 평가는 `official_py_sod_metrics_uint8_v2`를 사용한다. 기존 체크포인트도 재학습 없이 평가할 수 있다. 기존 6개 값은 보존하며 합성 입력에서 PySODMetrics 1.6.2 직접 계산과 동일함을 검사했다.

## 출력 수치

존재하는 테스트 데이터셋마다 `count`와 **21개 지표 필드**를 저장한다. 데이터셋 안에서는 각 이미지에 같은 가중치를 주며, 서로 다른 데이터셋을 임의의 종합 정확도 하나로 합치지 않는다.

| 키 | 의미·규약 | 좋은 방향 |
| --- | --- | --- |
| `S` | 구조 유사도 S-measure, 기본 구조 가중치 0.5 | 높음 |
| `wF` | 가중 F-measure, 라이브러리 beta 기본값 1. 일반 Fβ와 다름 | 높음 |
| `MAE` | 평균 절대 픽셀 오차 | 낮음 |
| `E_adaptive`, `E_mean`, `E_max` | 적응 임계값 E, 데이터셋 평균 256개 임계값 곡선의 평균·최댓값 | 높음 |
| `Fbeta_adaptive`, `Fbeta_mean`, `Fbeta_max` | beta 제곱 0.3인 일반 F-measure. 적응 임계값과 평균 곡선의 평균·최댓값 | 높음 |
| `IoU_0_5` | 전경 교집합 / 합집합 | 높음 |
| `Dice_0_5`, `F1_0_5` | 전경 교집합의 2배 / 두 전경 크기 합. 서로 같은 값 | 높음 |
| `Precision_0_5`, `Recall_0_5` | 전경 precision·recall(sensitivity) | 높음 |
| `Specificity_0_5` | 배경 true-negative rate | 높음 |
| `BER_0_5` | 전경 false-negative rate와 배경 false-positive rate의 평균, 범위 0–1 | 낮음 |
| `FPR_0_5`, `FNR_0_5` | False-positive / false-negative rate | 낮음 |
| `Accuracy_0_5`, `BalancedAccuracy_0_5` | 픽셀 정확도, balanced accuracy = 1 − BER | 높음 |
| `BoundaryIoU_0_5` | 전경 내부 경계 띠의 교집합 / 합집합 | 높음 |

Dice와 F1을 별개의 개선 근거로 세지 않는다. 이진 overlap은 **전경의 이미지별 점수를 평균한 값**이며 전경·배경 class mIoU나 전체 픽셀을 합친 confusion matrix와 다르다. BER은 비율이므로 논문 표가 퍼센트 규약일 때만 100을 곱한다.

## 예측 처리와 임계값

1. Segmentation logit에 sigmoid를 적용하고 모델 해상도에서 이미지별 min–max 정규화한다.
2. `align_corners=False` bilinear 보간으로 정답 원본 크기에 맞춘다.
3. [0,1]로 제한하고 255를 곱한 뒤 소수부를 버려 uint8로 바꾼다. 이 원본 크기 회색조 확률 PNG를 저장한다.
4. PySODMetrics가 uint8을 255로 나누고, 상수가 아닌 맵에는 다시 min–max 정규화를 적용한다. 상수 맵은 255로 나눈 값을 유지한다. 정답 전경은 원본 mask > 0이다.
5. 모든 테스트 지표는 이 준비된 맵을 사용한다. 고정 이진 지표는 **예측 ≥ 0.5**이다. E/Fβ 적응 임계값은 `min(2 × 평균 예측, 1)`이며 곡선은 라이브러리의 256개 임계값을 uint8 0–255 오름차순으로 내보낸다.

저장 PNG는 이진 마스크가 아닌 지표 라이브러리의 입력이다. 이를 재평가할 때도 동일한 정규화가 필요하다. Validation은 기존 **uint8·라이브러리 정규화 전 float MAE**를 유지하며 학습 holdout의 이미지 수로 가중 평균한다. 기본 `holdout`의 `best.pth`는 이 값만으로 선택하므로 validation/test MAE의 처리 규약이 다르다.

E/Fβ mean/max는 이미지별 곡선을 먼저 평균하고 그 곡선에서 평균·최댓값을 취한다. 이미지별 최댓값을 평균하면 다른 값이다. `per_image_metrics.csv`의 이미지별 max는 진단용이다. 아주 작은 합성 입력에서는 원래 라이브러리 분모 규약 때문에 E가 1을 조금 넘을 수 있으며 임의로 잘라내지 않는다.

Boundary IoU는 원저자의 3×3 erosion을 사용하고 반경을 `max(1, round(0.02 × 이미지 대각선))`로 정한다. Zero padding으로 이미지 테두리에 걸친 경계도 포함한다. Semantic 전경 경계 평가이며 COCO instance AP 전체를 재현하지 않는다.

이진 지표에서 두 마스크가 모두 비면 IoU·Dice/F1·Boundary IoU는 1, 한쪽만 비면 0이다. 예측 전경이 없을 때 precision은 정답도 비었으면 1, 아니면 0이다. 정답 전경/배경 class가 없으면 각각 recall/specificity를 1로 두고 BER 오류 기여를 0으로 둔다. 공식 S/wF/E/Fβ는 각 라이브러리 규약을 유지하며 Fβ의 빈 분모 점수는 0이다. 전체 정의는 `metrics.json`의 `metric_protocol_details`에 기록한다.

## 테스트 평가 산출물

- `metrics.json`: 데이터셋별 점수, 체크포인트·manifest·코드 식별 정보, 프로토콜, 산출물 경로·SHA-256.
- `per_image_metrics.csv`: 데이터셋·sample ID·원본 mask 경로·원본 크기·예측 경로·21개 지표.
- `predictions/<dataset-key>/<sample-key>.png`: 원본 크기 확률 맵. ID 해시로 이름 충돌을 막으며 CSV로 원본과 대응한다.
- `curves/<dataset-key>/threshold_curves.csv`: uint8/정규화 임계값, precision·recall·Fβ·E, 총 256행.
- `curves/<dataset-key>/{pr,fbeta,emeasure}.{svg,png}`: CSV 곡선으로 만든 그림. 파란 선은 해당 데이터셋의 평가 모델이며 축은 임계값·점수 또는 precision·recall이다. 생성 코드는 `fsa_study/metrics.py`에 있다.

`--no-save-predictions`는 확률 PNG만 생략하며 수치·곡선은 저장한다. 기존 평가 폴더는 덮어쓰지 않는다. 원본을 보존하고 새 출력 위치를 지정한다.

## 추론 비용

기존 전체 head latency·CUDA 메모리 측정은 유지한다. 별도의 **시간 측정 밖 추론**에서 실제 ATen 연산을 세므로 CARAFE의 functional 행렬곱/einsum도 포함한다. 시간 측정 loop에 카운터 hook을 섞지 않는다.

| 필드 | 해석 |
| --- | --- |
| `parameters`, `encoder_parameters` | 전체 모델·인코더 파라미터 수 |
| `latency_ms` | 평균·중앙값·표본 표준편차·원시 측정값 |
| `images_per_second` | Batch 크기를 평균 latency로 나눈 처리량 |
| `cuda_memory` | 추론 중 peak allocated byte와 증가분. CPU에서는 null |
| `complexity.macs`, `.flops` | Batch 전체의 합성곱·tensor contraction MACs, FLOPs = 2 × MACs |
| `.macs_per_image`, `.flops_per_image` | Batch 크기로 나눈 이미지당 값 |
| `.estimated_noncontraction_flops` | 산술·정규화·활성화·pooling·보간·샘플링의 별도 추정량 |
| `.estimated_total_flops`, `.estimated_total_flops_per_image` | 모든 관측 연산에 계산 규칙이 있을 때만 제공하는 합성곱/행렬 FLOPs + 기타 추정량 |
| `.coverage`, `.operators` | 분류·미지원 호출 수, 연산 이름·계산 규칙·개수 |

Grouped convolution은 실제 그룹 weight 차원을, transposed convolution은 입력 공간 크기를 사용한다. Bilinear 보간/grid sampling은 출력 원소당 곱 4회+합 3회로 추정하고 내부 좌표·index 계산은 제외한다. Scalar 비교·초월함수도 추정에서는 1회로 센다. 이는 하드웨어 명령어 수가 아니며 호출 커버리지가 FLOP 커버리지 비율을 뜻하지 않는다. 미지원 연산이 있으면 전체 추정량은 null이며 목록에 남긴다. 메모리 이동은 부동소수 연산으로 세지 않지만 latency에는 영향을 줄 수 있다.

장비·입력 크기·batch·dtype·warmup/repeats·측정 범위를 함께 보고한다. 모델의 segmentation/deep/region/edge head를 모두 실행한다. IO·전송·backward·optimizer·학습 메모리·에너지는 제외한다. 무작위 가중치·입력 profiling은 실행 확인이며 정확도나 학습된 branch 유용성의 근거가 아니다.

## 자동 리더보드

생성된 `run_all.sh`를 직접 실행하면 run마다 학습 → 평가 → profiling을 수행한다. `train.sh`는 학습만 수행한다. 둘 다 단계마다 `plans/<phase>/leaderboard/`를 갱신한다. Plan 생성 자체로는 실행하지 않는다.

- `leaderboard.html`: 단독으로 열리는 데이터셋·지표 선택, 변형 검색·정렬 표.
- `leaderboard.md`: validation MAE, 선택한 테스트 지표, 주요 비용 요약.
- `leaderboard.csv` / `.json`: 전체 지표·순위·좋은 방향·프로토콜·seed·누락 seed·run 위치.
- `summary.csv` / `.json`: 기존 평균·표본 표준편차 집계 형식에 새 지표를 추가한 결과.

기본 `holdout`의 전체 선택 기준은 **평균 validation MAE가 낮은 순서**다. `full_train`은 validation 없이 최종 epoch를 고정 선택하므로 validation 순위가 없고 `best_val_mae`가 null이다. 테스트는 데이터셋·지표별 보고용 순위, 비용은 각 측정값별 순위로 구분한다. 이질적인 지표를 임의로 가중 합산하지 않고 테스트로 체크포인트를 고르지 않으며 유의성을 주장하지 않는다. 정확히 같은 값이면 같은 경쟁 순위(1,1,3)를 부여한다.

학습·평가 규약과 **실제로 완료된 seed 집합이 모두 같은 경우만** 같은 순위표에 둔다. 생성기·활용기 option은 비교할 변형의 일부로 보존한다. 장비·입력·dtype·측정 조건이 다르면 비용 표도 분리한다. Seed 하나의 결과는 provisional이며 `--plan`으로 목표 seed·미시작 실험을 표시한다. Plan이 없으면 목표 시행 수는 미확인이다. 한 행뿐인 그룹의 1위는 다른 그룹보다 우수하다는 뜻이 아니다.

완료 단계 건너뛰기는 plan 동작·경로, 학습 설정, 체크포인트, 현재 평가·연산량 프로토콜, 측정 설정, 평가 산출물 해시를 확인한다. 예전 6개 지표나 조건이 다른 profile을 새 출력으로 간주하지 않는다. 부분 학습은 resume·덮어쓰기하지 않는다. 명시적 `--overwrite`로 교체하는 것은 파생 보고서 6개뿐이며 원시 결과는 보존한다.

재평가에는 plan과 집계에 새 `--evaluation-subdir test-v2`, `--profile-name profile-v2.json`을 지정한다. 직접 `evaluate`/`profile` 명령을 쓰면 그에 맞는 새 `--output` 위치를 쓴다. `run_plan.py --dry-run --plan /absolute/path/plan.json`은 명령만 보여주고 실행하지 않는다.

## 참고 문헌과 적용 범위

확인일 **2026-10-29**. 기존 생성·활용 논문 11편은 [방법 목록](../README.ko.md#참고-논문)에 보존했다. 다음은 평가 근거이며 새 학습 모듈의 출처가 아니다.

| English title / 한국어 제목 | 발표·공개처 | 적용 |
| --- | --- | --- |
| [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)<br>Boundary IoU: 객체 중심 이미지 분할 평가의 개선 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021; 학회 | [저자 erosion 구현](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py)을 따른 semantic mask 경계 비교. 전체 instance 평가와 구분 |
| [ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection](https://doi.org/10.1109/TPAMI.2024.3417329)<br>ZoomNeXt: 위장 객체 탐지를 위한 통합 협력 피라미드 네트워크 | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024; 저널 | 이미지 COD 지표·곡선 보고 기준. [§4.1.2](https://arxiv.org/html/2310.20208v4#S4.SS1.SSS2) |
| [PySODMetrics](https://github.com/lartpang/PySODMetrics)<br>회색조·이진 분할 지표 구현 | 공식 구현 저장소; 설치 package 1.6.2 | S/wF/MAE/E/Fβ·임계값 PR 곡선. 개별 지표 논문 링크는 `metric_protocol_details.sources`에 기록 |
| [PyTorch 2.11 FLOP counter](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/flop_counter.py)<br>PyTorch 연산량 카운터 | 공식 소프트웨어 소스 | Dense contraction 계산 규약. 독립 카운터와 값 비교 |

현재 검증은 합성 이미지·체크포인트 또는 초기화 추론이다. 해당 과거 검사는 실제 벤치마크 정확도나 수렴을 측정하지 않았다. 이후 업로드된 priority8 보고서에는 seed 1개의 3개 테스트셋 결과가 있지만 COD10K-test가 없으므로 전체 벤치마크 결과가 아니다. [검증 기록](../results/verification-2026-10-29-metrics/README.md).

## 벤치마크 전체 완료 판정

`benchmark_coverage`는 CAMO 250·CHAMELEON 76·COD10K 2,026·NC4K 4,121을 모두 포함하는지 기록한다. 리더보드의 `execution_complete`는 계획된 단계 완료, `benchmark_complete`는 표준 테스트셋 전체 평가 완료를 뜻한다. 누락·비표준 개수를 별도로 표시하고 일부 테스트셋의 완료를 전체 완료로 바꾸지 않는다. [복구 명령과 manifest 출처 검증](BENCHMARK-PROTOCOL.ko.md)을 참고한다. 기존 manifest로 학습한 체크포인트를 확장된 테스트 manifest에서 평가할 때는 원 학습 manifest와 확장 증명을 함께 보존하며, 다른 학습 조건으로 만든 모델과 같은 그룹에 넣지 않는다.
