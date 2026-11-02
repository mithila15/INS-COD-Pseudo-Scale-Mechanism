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

[`send_study.ps1`](../deploy/send_study.ps1)은 origin·브랜치·변경 없는 상태를 확인하고 `main`을 pull한 뒤 커밋된 study와 **모델 import에 필요한 `Experiments/FSA-ZoomNeXt`, `docs/MIGRATION.md`를 함께** ZIP으로 만든다. ZIP과 체크섬을 SCP로 `master@192.168.1.200:/home/master/`에 보내고 전송한 revision 및 `/home/master/mithila/shMin` 아래의 활성 코드 위치를 출력한다. 전송만으로 코드 설치나 실험을 시작하지 않는다. 로컬 데이터셋·환경·체크포인트 가중치는 제외한다. Windows에서 서버 주소에 접근 가능해야 하며 Parsec 접속만으로 해당 경로가 보장되지는 않는다.

전송 후에는 **Ubuntu Bash**에서 실행한다. `VERSION`은 전송 출력의 12자리 revision으로 바꾼다. 아래는 체크섬 확인 → 새 임시 staging 디렉터리에 압축 해제 → 포함된 [소스 설치기](../deploy/install_study.py) 실행이다. 기존 `INS_COD-releases/<revision>` 디렉터리를 다시 만들지 않으므로 그 폴더가 있어도 `mkdir` 충돌이 나지 않는다.

```bash
VERSION="REPLACE_WITH_TRANSFERRED_REVISION12"
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
WORK="/home/master/mithila/shMin"
STUDY="$WORK/$REL"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
ARCHIVE="/home/master/ins-cod-study-$VERSION.zip"

(
set -e
cd /home/master
sha256sum -c "ins-cod-study-$VERSION.zip.sha256"
STAGING=$(mktemp -d /home/master/ins-cod-install.XXXXXX)
"$PY" -m zipfile -e "$ARCHIVE" "$STAGING"
"$PY" "$STAGING/$REL/deploy/install_study.py" \
  --archive "$ARCHIVE" --work "$WORK"
)
```

설치기는 체크섬과 archive commit을 확인하고 교체할 코드·문서를 `$WORK/.code-backups/<revision>-<unique>/`에 보존한 뒤 `shMin` 아래에 설치한다. 기존 결과·데이터셋·체크포인트·가상환경·추가 로컬 PCD 파일은 보존한다. 인식한 study 프로세스가 실행 중이면 설치를 중단한다. 임시 staging은 실제 활성 소스와 별개다. 기존 `INS_COD`는 환경·데이터·이력 경로로 유지한다. 체크포인트 metadata 호환은 알려진 `architecture=common`만 지원하며 실제 PCD 체크포인트는 원 모델 구현이 필요하다.

사용자는 앞서 Windows/SCP 전송과 파일 백업을 동반한 소스 설치 성공을 보고했다. 패키지 설치기는 로컬 합성 파일시스템으로 검사하며, 코드 수정 작업이 사용자 원격 파일시스템에서 새 설치기를 실행한 것은 아니다. 설치에는 평가·학습·의존성 설치가 포함되지 않는다.

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

수정 코드를 설치한 뒤 **Ubuntu Bash**에서 실행한다. 아래 기본값은 사용자가 통합한 활성 코드 위치 `shMin`이며 Python 환경은 기존 `INS_COD`에서 재사용한다. 이 기본 명령은 엄격 중복 검사를 유지하므로 기록된 7쌍이 있으면 중단한다. 해당 경우에는 [명시적 중복 기록 절](#확인된-cod10k-중복을-명시적으로-기록하고-진행하기)을 사용한다. 이 절의 명령은 **같은 터미널**의 변수를 사용한다. 별도 source release를 전송했다면 `STUDY`를 그 release의 실험 디렉터리 절대 경로로 바꾸고 `PY`는 기존 환경을 유지한다. `--rebind-entrypoints`는 검증한 기존 plan의 실행 스크립트 경로만 현재 release로 옮기는 명시적 선택이다. Python 실행 파일과 나머지 설정을 유지하고 원본 plan이나 기존 checkout을 덮어쓰지 않는다.

```bash
REPO="/home/master/mithila/INS_COD"
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
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

## 확인된 COD10K 중복을 명시적으로 기록하고 진행하기

기본값은 계속 엄격 검사다. 서로 다른 split에 같은 이미지 바이트나 동일한 실제 경로가 있으면 준비를 중단한다. 이번 `Identical image paths or byte hashes cross train/val/test splits` 오류는 COD10K 이미지·GT 2,026쌍을 찾은 뒤 발생했다. 서버 진단에서는 **학습 원본과 테스트 사이 바이트가 같은 7쌍**을 찾았고, 기존 분할에서는 **최적화 train 6쌍, validation 1쌍**이었다. 동일한 실제 경로를 공유하는 쌍은 없었다. 코드 설치는 성공했으며 실패한 준비 명령은 평가·학습을 시작하지 않았다.

근거 수준은 둘로 구분한다.

- COD10K/SINet-V2 저자는 [2021-10-09 공식 안내](https://github.com/GewelsJI/SINet-V2/blob/5b9844c94d10a03b815c7c7039c194166a9a8ba3/README.md#L44)에서 **Crab-32**와 **Cat-1506**의 train/test 중복을 명시했다.
- 7쌍 모두 독립적인 COD-TDQ 배포 자료의 고정 버전 `c4e97371fa835cd0a778ba67c071f24ca5deaf11`에서도 이름·SHA256이 서버 기록과 같았다. 작은 Git-LFS 포인터만 읽었으며 이미지 본체를 다운로드하지 않았다. 나머지 5쌍도 외부에 배포된 복사본에서 확인했지만, **COD10K 원저자가 7쌍 전체를 인정한 공지라는 뜻은 아니다.** [고정 URL·해시 근거](../results/verification-2026-11-01-overlap/upstream-evidence.json), [서버 진단 요약](../results/verification-2026-11-01-overlap/server-audit-summary.json).

명시적인 `--acknowledge-cod10k-overlap` 옵션은 학습 원본과 테스트 split 사이에서 목록에 기록한 COD10K **ID 쌍과 이미지 SHA256**이 일치하는 7쌍만 허용한다. 다른 split 사이의 미확인 쌍·해시, 서로 다른 split이 공유하는 실제 경로, 서로 다른 split에 걸쳐 3개 이상인 동일 해시 그룹, train/validation 중복은 계속 오류다. 모든 중복을 무시하는 옵션이 아니다. 기존 records 안에서 같은 split에만 있는 이미지 중복은 원래 엄격 정책의 동작을 유지하며, COD10K 인정 옵션 때문에 제거하거나 오류로 바꾸지 않는다. 별도의 테스트 추가 검사는 새 test 이미지가 원본 이미지 경로 또는 인정되지 않은 원본 이미지 해시를 재사용하면 계속 거부한다. 파일 해시·이미지/GT 대응·개수·크기·manifest·체크포인트 검사는 유지한다. 입력 이미지·GT·이전 manifest·체크포인트·과거 보고서를 변경하지 않는다.

진단한 서버 복사본에서는 수정 코드를 `shMin` 아래에 설치한 뒤 위의 엄격 검사 준비 명령 대신 다음을 **Ubuntu Bash**에서 실행한다. 새 평가 계획만 생성한다. 이후 명령도 **같은 터미널**에서 실행한다.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
RUN="$STUDY/results/priority8-separate-20261030_204647"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
COMPLETE="$RUN/plans/four-benchmarks-ack-$STAMP"

"$PY" "$STUDY/complete_benchmark.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --output "$COMPLETE" \
  --evaluation-subdir "test-four-benchmarks-ack-$STAMP" \
  --rebind-entrypoints \
  --acknowledge-cod10k-overlap
```

새 manifest의 `overlap_policy="acknowledged-cod10k-v1"`과 `overlap_audit`를 확인한다. 기록된 분할에서는 `disjoint=false`, `train_test_count=6`, `val_test_count=1`, `affected_test_count=7`과 정확한 쌍 목록이 나와야 한다. 평가 metadata와 새 보고서에도 이 조건이 전달된다. 이 audit는 서로 다른 split 사이의 동일성을 검사하며 같은 split 내 이미지가 모두 고유하거나 테스트 데이터셋끼리 독립적이라는 뜻은 아니다. 표준 개수 충족·실행 완료와 데이터의 독립성은 다른 판단이다.

아래 별도 명령은 기존 학습 체크포인트로 **평가만 실행**한다.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$COMPLETE/run_evaluation.sh" 2>&1 | tee "$COMPLETE/evaluation.log"
```

지표는 `$RUN/runs/priority8/<variant>__seed3407/test-four-benchmarks-ack-$STAMP/metrics.json`, 리더보드는 `$COMPLETE/leaderboard/`에 저장한다. 아래의 전체 학습 준비 절은 이 manifest를 그대로 입력으로 쓸 수 있다. 정책을 이어받고 validation을 train으로 이동한 뒤 중복 기록을 다시 계산하므로 이 복사본에서는 **train/test 7쌍, validation/test 0쌍**이 된다. 새 실험의 `run_study.py prepare`도 배포 개수를 의도적으로 유지할 때 같은 명시적 옵션을 지원한다.

이 옵션은 배포 기준의 4,040/2,026장을 유지한다. **중복 노출을 없애거나 COD10K-test를 독립 평가셋으로 바꾸는 기능이 아니다.** 기존 모델은 6장의 중복 이미지를 최적화에, 1장을 validation·체크포인트 선택에 이미 사용했다. 인정 표시로 이 이력을 되돌릴 수 없다. 이번 COD10K 중복만으로 기존 CAMO·CHAMELEON·NC4K 점수가 무효가 되지는 않지만, 이 바이트 검사가 모든 시각적 유사 이미지 누출까지 보장하는 것도 아니다. 공식 공지의 2장에 대한 대략적인 지표 영향은 이번 7장이나 이 모델에 일반화하지 않는다.

독립 분할을 사용하려면 별도 이름의 조건과 비교가 필요하다. 테스트에서 7장을 제외하면 **2,019장 부분집합**, 새 학습 전에 대응하는 학습 원본 7장을 제외하면 전체 테스트셋에 대해 **4,033장 학습**이다. 둘 다 표준 4,040/2,026 조건과 다르며 이번 옵션이 몰래 수행하는 동작이 아니다. [판단·근거 기록](../../../../Progress/2026-11-01/COD10K-Overlap/README.md).

### 설치 성공 후 준비 단계에서 중단되는 경우

`8f08f4ff6b61`의 중복 인정 구현에는 기존 test split 내부의 동일 해시 그룹도 7쌍의 교차 split 허용 목록과 대조하는 회귀 오류가 있었다. 따라서 test records만 있는 그룹도 `Unacknowledged image overlap: unknown hash or a group with more than two records` 오류를 낼 수 있었다. 같은 split 내 경로 공유 검사도 원래 엄격 검사보다 강화되는 문제가 있었다. [동일 split 회귀 수정](../results/verification-2026-11-01-same-split/README.ko.md)은 원래 동작을 복구하고 위의 교차 split 검사·새 test 추가 검사는 유지한다. 합성 입력으로 오류를 재현했으며, 사용자 서버에서 정확히 어떤 그룹이 이번 오류를 냈는지는 아직 직접 확인하지 않았다. 이 메시지가 나오는 모든 경우를 무해한 중복으로 판단하지 않는다.

준비 명령이 실패하면 새 plan과 `run_evaluation.sh`가 생성되지 않는다. 뒤이어 Bash나 `tee`에서 나오는 `No such file or directory`는 그 결과이며, 리더보드 경로를 출력하는 것만으로 보고서가 생기지 않는다. 수정 소스를 설치하고 새 출력 경로로 준비를 다시 실행한다. 준비 성공과 plan 생성을 확인한 뒤 평가 또는 새 학습을 시작한다. 재학습으로 manifest 준비 코드의 오류가 해결되지는 않는다. 수정 후에도 교차 split 그룹이 거부된다면 검사를 끄는 대신 해당 records를 확인해야 한다.

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

`$FULL/plans/priority8/plan.json`의 학습 개수와 설정 변경 기록을 확인한다. 바로 전체 데이터 새 학습을 하려면 앞의 `run_evaluation.sh` 실행은 생략해도 된다. manifest·plan 준비 성공만 필요하다. **같은 Ubuntu 터미널**에서 아래 명령을 실행하면 8개 실험의 **새 학습 → 테스트 평가 → profiling을 시작**한다. 이전 3,636장 모델에서 이어 학습하는 것이 아니라 새 전체 학습 시간이 필요하다.

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
