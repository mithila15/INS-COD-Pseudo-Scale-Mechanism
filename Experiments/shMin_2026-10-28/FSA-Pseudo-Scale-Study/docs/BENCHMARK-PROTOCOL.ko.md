# 2026-11-01 · 벤치마크 조건 — 테스트 누락 복구와 전체 학습 데이터 비교

[English](BENCHMARK-PROTOCOL.md) | **한국어** | [실험 안내](../README.ko.md)

업로드된 [priority8 보고서](../results/priority8-report-20261101_172449/README.ko.md)는 seed 3407의 8개 실험을 CAMO·CHAMELEON·NC4K에서 평가한 결과다. **COD10K-test가 빠졌다.** 기존 plan의 모든 단계가 끝났다는 상태와 표준 테스트셋 4개를 모두 평가했다는 상태는 다르다. 기존 보고서와 체크포인트는 3개 데이터셋의 역사적 결과로 보존한다.

두 문제는 별도 작업으로 해결한다.

| 작업 | 재사용 자료 | 새 산출물 | 학습 필요 여부 |
| --- | --- | --- | --- |
| 테스트셋 4개 평가 완성 | 기존 holdout 학습의 `best.pth`, 동일 학습 분할 | 새 평가 manifest·plan·지표·리더보드 | 불필요 |
| `full_train` 비교 | 원 학습 이미지 4,040장, backbone 초기 가중치, 실험 설정 | 별도 경로의 새 모델·결과 | 처음부터 필요 |

## ZoomNeXt와 다른 조건

2026-11-01에 공식 ZoomNeXt 이미지 COD 코드의 [`614af4348808734aaf2ec43937cf827410330b67`](https://github.com/lartpang/ZoomNeXt/tree/614af4348808734aaf2ec43937cf827410330b67) 버전을 확인했다. 논문은 **ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection** / **ZoomNeXt: 위장 객체 검출을 위한 통합 협력 피라미드 네트워크**, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (**TPAMI, 2024, 저널**)다. [정식 게재 논문](https://doi.org/10.1109/TPAMI.2024.3417329)과 [arXiv v3 §4.1](https://arxiv.org/html/2310.20208v3#S4.SS1)을 구분해서 참고했다. 한글 제목은 설명을 위한 번역이다.

| 항목 | 기존 priority8 (`holdout`) | ZoomNeXt 공식 배포 이미지 설정 |
| --- | --- | --- |
| 최적화에 쓰는 이미지 | CAMO 900 + COD10K 2,736 = **3,636장** | CAMO 1,000 + COD10K 3,040 = **4,040장** |
| Validation | CAMO 100 + COD10K 304 = **404장** | 배포 학습 루프에 별도 holdout 없음 |
| 체크포인트 선택 | 이미지 수 가중 validation float MAE 최저 | 최종 epoch |
| Epoch / 배포 코드 batch | 150 / 4 | 150 / 4 |
| 이 조건의 업데이트 수 | epoch당 909, 총 136,350 | epoch당 1,010, 총 151,500 |
| Backbone·인코더 입력 | PVTv2-B2, 384×384 RGB 1회 후 pseudo 특징 | PVTv2-B2 공유 인코더에 192/384/576 RGB 3회 |
| 디코더 채널 | 48 | 64 |
| 손실 | weighted BCE+IoU, deep supervision, region, edge Dice, cosine UAL | BCE와 cosine UAL |
| 증강 | 좌우 반전 0.5, 상하 반전 0.2 | 좌우 반전 0.5, ±90° 이내 회전·밝기/대비·HSV 각각 0.5 |
| 학습 GT resize | PIL nearest 후 이진화 | OpenCV bilinear |
| 정밀도 | 학습에 AMP autocast/scaler 없음 | `use_amp=True` |
| Adam | LR 1e-4, encoder LR ×0.1, weight decay 0 | 배포 기본값 동일 |
| LR 감소 | epoch 100 종료 후 | 전체 데이터·batch 4에서 iteration 101,000 |
| 예측 처리 | 이미지별 min-max, PyTorch bilinear로 GT 크기 복원 | 배치 전체 min-max, OpenCV bilinear로 GT 크기 복원 |
| 기록된 환경 | torch 2.11.0, torchvision 0.26.0, timm 1.0.30, PySODMetrics 1.6.2 | README: torch 2.1.2, torchvision 0.16.2; requirements: timm 0.9.12, PySODMetrics 1.4.2 |

**논문에는 batch 8, NVIDIA 2080Ti라고 적혀 있지만 배포 `icod_train.py`는 batch 4다.** 코드의 batch-4 스케줄을 논문의 batch-8 조건과 동일하다고 적지 않는다. 근거: [학습 설정](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/configs/icod_train.py#L6-L60), [증강](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L70-L132), [AMP·학습 루프](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L202-L278), [최종 체크포인트](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L313-L327), [모델·손실](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/methods/zoomnext/zoomnext.py), [평가](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L137-L178), [설치 안내](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/README.md#install-requirements), [의존성](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/requirements.txt).

3,636/404는 분할 구현, 사용자가 보여준 validation 개수, epoch당 909 step 로그로 뒷받침된다. 업로드된 보고서에는 공통 manifest 해시가 있지만 원격 manifest의 전체 records는 없다. 서버 원본 데이터는 로컬에서 직접 보지 않았다. 업로드된 추론 프로파일에 RTX PRO 5000 Blackwell이 기록됐지만 이는 profiling 장비의 직접 근거이며 학습 장비를 독립적으로 확인한 기록은 아니다. 공식 사전학습 가중치를 다운로드해 실제 바이트까지 대조하지 않았다.

**`full_train`은 학습 이미지 수와 최종 epoch 선택을 맞춘다. ZoomNeXt 전체 실험 조건 재현은 아니다.** 구조·손실·증강·정밀도·라이브러리·예측 처리는 위 차이가 남는다. 10% holdout은 비교 조건의 차이이며 성능 격차의 원인으로 입증된 것은 아니다. 더 강한 논문 비교에는 공식 모델도 명시적으로 맞춘 데이터·평가 조건에서 다시 실행해야 한다.

## Windows에서 수정 코드 전송

Ubuntu의 `master@ubuntu` 터미널이 아니라 **Windows PowerShell**(`PS C:\Users\User>` 같은 프롬프트)에서 실행한다. 로컬 상위 폴더는 `C:\Users\User\OneDrive\Desktop\shMin`이며 `$Repo`에는 그 안의 실제 Git checkout, 예를 들어 앞서 만든 `INS_COD-publish-<timestamp>` 폴더를 넣는다. 로컬 변경 없는 `sh-min/INS_COD`의 `main` checkout이어야 한다.

최초 `git -C "$Repo" pull --ff-only origin main`으로 새 스크립트를 받은 뒤 실행한다.

```powershell
& "$Repo\Experiments\shMin_2026-10-28\FSA-Pseudo-Scale-Study\deploy\send_study.ps1" -Repo $Repo
```

[`send_study.ps1`](../deploy/send_study.ps1)은 origin·브랜치·변경 없는 상태를 확인하고 `main`을 pull한 뒤 커밋된 study와 **필수 모델 보존 소스인 `Experiments/FSA-ZoomNeXt`, `docs/MIGRATION.md`를 함께** ZIP으로 만든다. 모델 import가 이 형제 디렉터리를 사용하므로 새 release에도 필요하다. ZIP과 체크섬을 SCP로 `master@192.168.1.200:/home/master/`에 보낸다. 전송한 revision과 새 코드 경로 `/home/master/mithila/INS_COD-releases/<revision12>`를 출력하며 원격 실험을 시작하거나 로컬 데이터셋·환경·체크포인트 가중치를 전송하지 않는다. Windows PC에서 해당 사설 서버 주소에 접근 가능해야 한다. Parsec 접속 성공만으로 이 네트워크 경로까지 보장되지는 않는다.

ZIP 체크섬을 확인하고 출력된 **새 release 디렉터리**에 압축을 푼다. 아래 `STUDY`를 그 release의 `Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study`로 설정하고 기존 환경·데이터·체크포인트를 재사용한다. 서버의 기존 checkout은 압축 해제 대상이 아니므로 로컬 architecture 수정도 보존할 수 있다. 이 study는 알려진 `architecture=common` metadata 호환만 지원하며 실제 PCD 체크포인트에는 원 구현이 필요하다.

전송 스크립트는 코드 수정 환경에서 검토했다. 해당 환경에서는 Windows PowerShell·사설망·실제 SCP를 실행하지 않았으므로 사용자 장비에서 확인해야 한다.

## 표준 테스트셋 4개를 요구하고 확인하기

`run_study.py prepare` CLI의 기본값은 다음 4개를 모두 요구한다.

| 데이터셋 key | 표준 테스트 수 |
| --- | ---: |
| `camo` | 250 |
| `chameleon` | 76 |
| `cod10k` | `COD10K-CAM-` 전경 이미지 2,026 |
| `nc4k` | 4,121 |

의도적인 일부 평가는 `--test-datasets camo chameleon nc4k`, 학습 전용 준비는 `--test-datasets none`으로 명시할 수 있다. 이 경우 4개 벤치마크 전체 완료로 표시하지 않는다. 평가 metadata와 새 리더보드는 `benchmark_coverage`, 누락 데이터셋, 비표준 개수를 표시한다. `execution_complete`와 `benchmark_complete`는 서로 다른 상태다. 누락 점수를 만들거나 0점으로 취급하지 않는다.

### 완료된 priority8 체크포인트 재평가

수정 코드를 설치한 뒤 **Ubuntu Bash**에서 실행한다. 이 절의 명령은 **같은 터미널**의 변수를 사용한다. 별도 source release를 전송했다면 `STUDY`를 그 release의 실험 디렉터리 절대 경로로 바꾸고 `PY`는 기존 환경을 유지한다. `--rebind-entrypoints`는 검증한 기존 plan의 실행 스크립트 경로만 현재 release로 옮기는 명시적 선택이다. Python 실행 파일과 나머지 설정을 유지하고 원본 plan이나 기존 checkout을 덮어쓰지 않는다.

```bash
REPO="/home/master/mithila/INS_COD"
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study-clean/bin/python"
RUN="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-separate-20261030_204647"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
COMPLETE="$RUN/plans/four-benchmarks-$STAMP"

"$PY" "$STUDY/complete_benchmark.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --output "$COMPLETE" \
  --evaluation-subdir "test-four-benchmarks-$STAMP" \
  --rebind-entrypoints
```

이 명령은 **새 평가 manifest·plan 준비와 검사만** 한다. 원래 모든 run의 학습이 완료돼야 한다. 원 학습 manifest와 체크포인트의 동일성을 확인하고 기존 모든 sample의 동일성·분할·파일 해시를 보존한 채 COD10K test만 추가한다. 새 파일 위치 때문에 data root를 넓혀야 한다면 동일 파일을 가리키도록 경로를 다시 표현하고 출처를 기록한다. 표준 테스트 개수와 이미지·GT 크기를 검사한다. 체크포인트 metadata를 수정해서 새 manifest로 학습한 것처럼 만들지 않는다.

COD10K test 파일은 미리 있어야 한다. 기존 data root 아래 인식 가능한 배치가 하나면 자동 탐색한다. 없거나 여러 후보가 있으면 준비 명령에 `--cod10k-images /absolute/path/to/COD10K/Test/Image --cod10k-masks /absolute/path/to/COD10K/Test/GT_Object`를 함께 추가한다. 이는 예시이며 서버에서 확인된 경로가 아니다. 자동 다운로드는 없으며 다른 자료를 대신 넣기 위해 개수·파일명 검사를 우회하지 않는다.

다음은 **평가만 실행**한다. 기존 모델로 **4개 테스트셋 전체를 현재의 동일 평가 코드로 다시 계산**하여 예전 evaluator와 새 evaluator를 한 보고서에 섞지 않는다. 학습·profiling은 실행하지 않는다. 기존 3개 데이터셋 출력도 유지한다.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$COMPLETE/run_evaluation.sh" 2>&1 | tee "$COMPLETE/evaluation.log"
```

새 지표는 `$RUN/runs/priority8/<variant>__seed3407/test-four-benchmarks-$STAMP/metrics.json`, 리더보드는 `$COMPLETE/leaderboard/leaderboard.html` 또는 `leaderboard.md`다. 데이터셋마다 기존 21개 지표·곡선·이미지별 결과를 보존한다. 복구 plan은 `run_plan.py`로 수동 호출해도 train/profile action을 거부한다.

## 별도 전체 데이터 학습 비교

방법 개발의 기본값은 계속 `holdout`이다. 3,636/404 분할과 validation으로 `best.pth`를 고르는 동작은 유지한다. 명시적 `full_train`은 CAMO 1,000 + COD10K 3,040장과 **validation 없음**을 사용한다. 선택 기준은 사전에 정한 **최종 epoch 고정**이다. 파일명 호환성을 위해 `best.pth`에 최종 epoch 모델을 저장하지만 테스트나 validation에서 가장 좋은 모델을 고른 의미가 아니다. `best_val_mae`는 null이고 validation loader·validation 순위·early stopping을 사용하지 않는다.

앞에서 `$COMPLETE/manifest.json`을 준비한 뒤 아래 명령으로 **원래 8개 실험에서 새 전체 학습 plan을 생성**한다. 기존 training holdout records만 train으로 옮기고 모든 test record를 보존한다. 기존 seed·150 epochs·batch 4·backbone 초기화·변형·나머지 학습 설정을 유지하며 학습 완료 체크포인트를 재사용하지 않는다. 이 명령의 입력은 평가 파일이 아니므로 4개 데이터셋 평가가 끝나기 전에도 계획 생성은 가능하다.

```bash
FULL="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-$STAMP"

"$PY" "$STUDY/prepare_full_training.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --manifest "$COMPLETE/manifest.json" \
  --output "$FULL" \
  --rebind-entrypoints
```

`$FULL/plans/priority8/plan.json`의 학습 개수와 설정 변경 기록을 확인한다. **같은 Ubuntu 터미널**에서 아래 명령을 실행하면 8개 실험의 **새 학습 → 테스트 평가 → profiling을 시작**한다. 이전 3,636장 모델에서 이어 학습하는 것이 아니라 새 전체 학습 시간이 필요하다.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$FULL/plans/priority8/run_all.sh" 2>&1 | tee "$FULL/run.log"
```

새 결과는 `$FULL/runs/priority8/<variant>__seed3407/`, 새 리더보드는 `$FULL/plans/priority8/leaderboard/`에 저장한다. 같은 plan 경로의 `train.sh`는 학습만 실행한다. 자원을 따로 배정하지 않았다면 같은 GPU에서 기존 모델 평가와 새 학습을 동시에 시작하지 않는다.

완료된 plan을 옮기는 대신 처음부터 새 실험을 준비한다면 `run_study.py prepare`와 `plan_study.py` **양쪽에** `--training-protocol full_train`을 준다(`--val-fraction` 생략 또는 `0`). 새 `--phase priority8 --seeds 3407`의 순서는 `single`, `native_conv__asff`, `native_conv__sk`, `native_conv__mean`, `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean`이다. 이 phase에는 `legacy_full`을 넣지 않으며 나머지 행렬 phase의 대조군은 그대로다.

## 결과 해석과 검증 한계

학습 소스·manifest·지표·profiling 출처를 유지한다. holdout과 전체 학습을 하나의 순위 그룹에 섞지 않는다. 전체 학습의 테스트 표는 결과 보고용이며 테스트 점수로 epoch를 선택하거나 반복해서 변형을 조정하지 않는다. seed 1개로 seed 간 안정성이나 통계적 유의성을 입증할 수 없다. `projection` 대조군에는 공간 resampling이 없으므로 이 대조군의 향상을 pseudo-scale 생성의 효과라고 볼 수 없다.

최종 회귀 검사 **694 tests + 74 subtests**, 브라우저 8개 검사를 통과했다. 합성 입력·추론·gradient 검사이며 optimizer 업데이트를 하지 않았고 새 모델의 정확도를 측정한 결과가 아니다. [검증 로그와 범위](../results/verification-2026-11-01-benchmark/README.ko.md).

코드 수정 과정에서 새 실제 데이터 학습이나 서버 벤치마크 평가를 실행하지 않았다. 서버 경로·원본 이미지·체크포인트·서버 실제 데이터 CUDA 평가·최종 4개 데이터셋 결과는 사용자가 실행해야 확인된다. 기존 환경을 재사용하며 재설치하지 않는다. [변경 기록](../../../../Progress/2026-11-01/Benchmark-Completion/README.md).
