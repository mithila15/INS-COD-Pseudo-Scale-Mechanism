# 2026-11-05 · Pseudo-Scale Mechanism — 소수 대조 실험 실행 방법

[English](SCALE-MECHANISM.md) | **한국어** · [저장소](../../../../README.ko.md) · [검증 기록](../results/verification-2026-11-04-mechanism/README.ko.md)

브랜치: **`research/pseudo-scale-mechanism`**, 시작 커밋 `1a003b9a7becddc0fd844bb85de62f005b96a9f0`. 질문은 **“학습 분기를 추가하는 효과를 넘어, 특징의 scale을 바꾸는 것 자체가 도움이 되는가?”**다. 실제 RGB scale 참고군, 더 조밀한 최종 특징, 원본을 보존하는 fusion, 선택적인 frozen feature teacher를 추가했다. 모두 검증할 가설이며 새 학습 성능은 아직 없다.

## 1. 먼저 네 개만 비교

| 순서 | 실험 | 추론 시 이미지당 encoder 호출 | 확인할 질문 |
| ---: | --- | ---: | --- |
| 1 | `single` | 1 | 보조 분기 없는 ConvNeXt 마지막 stage 기준 모델 |
| 2 | `projection__mean` | 1 | 같은 해상도에서 학습 분기 두 개만 추가하면 좋아지는가? |
| 3 | `native_conv__mean` | 1 | 동일한 분기 연산을 1.5배·0.5배 특징 해상도에서 하면 더 좋아지는가? |
| 4 | `image_pyramid__mean` | 3 | 같은 최종 stage decoder로 RGB 1배·1.5배·0.5배를 실제 인코딩하면 어느 정도인가? |

공통 조건은 ConvNeXt-Tiny 마지막 stage만 사용, seed 3407, 150 epoch, batch 4, 입력 384×384, projection 채널 48, 같은 로컬 ImageNet 초기 가중치와 기존 loss·optimizer·증강이다. Projection과 native convolution은 분기 파라미터 수와 초기값이 같지만 중간 공간 연산과 연산량이 다르다. 실제 scale 참고군은 encoder와 projection을 공유해 RGB를 세 번 처리하고, 1배 격자에 맞춰 평균낸다. Pseudo 생성기는 없다. **이 참고군은 ZoomNeXt 전체 구조의 재현이 아니다.**

`core` 명령은 **4개만** 실행한다. 이후 단계를 자동 실행하지 않는다. 이전 priority8 실행기와 전체 후보 목록은 별도로 보존했다.

| 직접 지정하는 단계 | 실험 수 | 최종 output stride | 목적 |
| --- | ---: | ---: | --- |
| `core` | 4 | 32 | 기준·용량 대조·pseudo·실제 RGB scale |
| `resolution` | 4 | 16 | 조밀한 최종 특징으로 같은 네 가지 반복 |
| `fusion` | 2 | 기본 32 | Projection/native에 `anchor_gated`를 적용하고 core의 mean과 비교 |
| `distill` | 2 | 기본 32 | Projection/native + mean에 실제 scale feature teacher 감독 추가 |

## 2. 구조와 학습 신호

**Stride 16.** ConvNeXt의 모든 stage와 사전학습 파라미터 텐서를 유지한다. 마지막 2×2 downsampling convolution만 stride 2에서 1로 바꾸고, 앞에서 오른쪽·아래에 replicate padding을 넣는다. 입력 384×384에서 마지막 768채널 특징이 12×12에서 24×24가 된다. 공식 분류 모델에 적용한 **이번 연구의 변경**이며 sampling·수용영역·경계 동작이 달라진다. Dilation이나 앞 stage skip은 추가하지 않는다. Decoder의 다섯 convolution block은 같고, stride 16일 때 첫 block은 크기를 유지하고 이후 네 block이 확대한다.

**원본 보존 fusion.** 같은 `B,C,H,W` 크기로 정렬된 main/large/small 특징 `M,L,S`를 다음처럼 결합한다.

```python
aux = projection_1x1(cat([L, S], dim=1))  # 2C -> C, bias 없음
output = M + sigmoid(channel_gate) * aux
```

채널별 학습 gate는 0.01로 시작하고 이미지·위치 사이에서 공유한다. 입력에 따라 바뀌는 attention gate가 아니며 뺄셈을 강제하지 않는다. Main 특징이 직접 흐르는 경로를 보존하는 이번 연구용 모듈이다. 실제 이득은 projection 대조군과 native 생성기 모두에서 mean 대비 확인한다.

**선택적 feature distillation.** 먼저 core의 실제 RGB scale 참고군을 학습한다. 나중에 distill 단계에서 같은 ImageNet 가중치로 학생을 새로 초기화하고, 참고군의 최종 epoch `best.pth`를 teacher로 읽는다. 고정 teacher는 학생과 같은 증강된 학습 이미지를 1배·1.5배·0.5배로 인코딩한다. Projection한 보조 특징을 main 격자에 정렬하고 대응하는 학생 보조 특징과 채널 벡터 cosine distance를 계산한다. 픽셀·batch·두 분기 평균에 **0.1**을 곱해 기존 loss에 더한다. Main 특징에는 feature matching loss를 걸지 않는다.

Teacher는 eval 모드이며 gradient를 받지 않는다. 학생 optimizer와 state dictionary 밖에 두므로 학생 평가·profiling에는 학생만 복원한다. **학습 비용은 증가**한다. 이미지마다 학생 1회와 고정 teacher 3회 encoder 호출이 필요하고 teacher 선행 학습도 필요하다. 절감 여부를 확인하는 대상은 학생의 **추론 비용**이다. Teacher checkpoint SHA256·모델 설정·manifest hash·epoch·loss 계수를 기록한다. 최종 checkpoint가 아니거나 manifest·backbone·stride·채널·입력 크기·full-train 조건이 다르면 거부한다. Plan의 완료 상태 재검사를 위해 teacher 파일은 보관한다. 학생 단독 evaluate는 teacher 없이 가능하다.

**Pseudo-mask label은 생성하지 않는다.** Dataset mask가 분할·deep·region 정답이며 mask에서 파생한 edge가 edge head의 정답이다. 추가 감독 신호는 teacher 특징 텐서다. [감독 신호와 pseudo-label 설명](PSEUDO-LABELS.ko.md).

## 3. 환경·입력과 별도 코드 설치

기존 서버 가상환경을 재사용한다.

```text
/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python
```

기존 Python 3.12.13 / torch 2.11.0 / torchvision 0.26.0 고정 프로필이며 새 의존성은 없다. 새 환경이 필요하면 [한국어 환경 안내](../environment/README.ko.md)를 따른다. 기존 `/home/master/mithila/shMin`의 full4040 manifest와 데이터 파일을 그대로 참조한다. 데이터셋을 다시 받거나 분할을 다시 만들지 않는다.

Manifest는 기존 study의 `results/priority8-full4040-fixed-20261101_200834/manifest.json`이다. 학습은 CAMO 1,000 + COD10K 3,040장, test는 CAMO 250·CHAMELEON 76·COD10K 2,026·NC4K 4,121장이다. 기존에 명시한 COD10K train/test 바이트 중복 7쌍과 CHAMELEON 크기 불일치 2건을 그대로 기록한다. 이 변경으로 train/test 독립성이 확보되는 것은 아니다. [조건과 한계](BENCHMARK-PROTOCOL.ko.md).

기존 study의 `weights/convnext_tiny-983f1562.pth`에 공식 TorchVision ConvNeXt-Tiny 가중치 약 109 MB를 준비한다. PVT 가중치는 호환되지 않는다. 없으면 [CNN 초기화 안내](CNN-BACKBONES.ko.md)의 **사용자가 직접 실행하는** 다운로드·hash 확인 명령을 사용한다. 모델 생성 중 자동 다운로드하지 않는다.

### Windows PowerShell: 새 브랜치를 받아 서버로 전송

**Windows PowerShell에서만 실행한다. Ubuntu Bash에 붙여 넣지 않는다.** 새 checkout을 사용한다. 아래 정확한 폴더가 이미 있으면 해당 checkout이 이 브랜치인지 확인한 뒤 두 번째 블록부터 사용한다.

```powershell
$Base = "C:\Users\User\OneDrive\Desktop\shMin"
$Repo = Join-Path $Base "INS_COD-scale-study"
New-Item -ItemType Directory -Force -Path $Base | Out-Null
git clone -c core.longpaths=true --single-branch --branch research/pseudo-scale-mechanism "https://github.com/sh-min/INS_COD.git" "$Repo"
if ($LASTEXITCODE -ne 0) { throw "Clone failed; stop here." }
```

**같은 PowerShell 창에서** 다음을 실행한다.

```powershell
$Script = Join-Path $Repo "Experiments\shMin_2026-10-28\FSA-Pseudo-Scale-Study\deploy\send_study.ps1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& "$Script" -Repo "$Repo" -Branch "research/pseudo-scale-mechanism"
```

지정 브랜치를 pull하고 커밋된 ZIP과 SHA256을 SCP로 보낸다. 출력의 `Version for server installation: <12자리 버전>`을 다음 단계에서 사용한다. 실험은 시작하지 않는다.

### Ubuntu Bash: 별도 코드 경로에 설치

`VERSION`만 방금 전송 출력의 정확한 12자리 버전으로 바꾼 뒤 블록 전체를 실행한다. Checksum 실패 시 설치도 중단한다.

```bash
(
set -e
VERSION="REPLACE_WITH_TRANSFERRED_12_CHARACTER_VERSION"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
ARCHIVE="/home/master/ins-cod-study-$VERSION.zip"
cd /home/master
sha256sum -c "$ARCHIVE.sha256"
mkdir -p /home/master/mithila/INS_COD-releases
RELEASE=$(mktemp -d "/home/master/mithila/INS_COD-releases/scale-$VERSION-XXXXXX")
"$PY" -m zipfile -e "$ARCHIVE" "$RELEASE"
"$PY" "$RELEASE/$REL/deploy/install_study.py" \
  --archive "$ARCHIVE" --work /home/master/mithila/shMin-scale-study
)
```

설치기는 데이터·결과·checkpoint·가상환경을 보존하고 교체되는 소스는 백업한다. 대상 경로의 코드를 쓰는 실험 프로세스가 실행 중이면 교체를 거부한다. 새 활성 코드는 `/home/master/mithila/shMin-scale-study`이며, 아래 결과는 기존 `shMin` 실험 트리 안의 새 폴더에 저장한다.

### 이미 설치한 CRLF 실행 스크립트 복구

2026-11-05 서버 출력은 `Manifest and backbone weights ready.`까지 진행한 뒤 Bash `invalid option name`에서 멈췄다. 새 tmux 세션·학습 프로세스가 생기기 전이다. `core.autocrlf=true`로 ZIP을 만드는 재현 검사에서 같은 오류가 나왔다. 옵션 끝의 CR 때문에 `pipefail\r`를 잘못된 옵션으로 읽는다. 서버 원본 바이트를 직접 읽은 것은 아니며, 재현 결과에 근거한 CRLF 진단이다.

수정한 전송기는 archive 줄바꿈 변환을 끄고 `.gitattributes`는 shell 파일을 LF로 고정한다. 설치기는 원본 ZIP checksum 검증 후 **소스 `.sh`만** CRLF에서 LF로 바꾸며, `normalizations`에 원본·설치 hash를 기록하고 교체 전 파일을 백업한다. 다른 소스 바이트와 보존 대상 입력·결과·checkpoint는 그대로 둔다. [Git 줄바꿈 규칙](https://git-scm.com/docs/gitattributes). [재현·검증 기록](../results/verification-2026-11-05-deployment/README.ko.md).

보고된 상태처럼 코드·가중치는 준비됐고 **활성 `scale-core` 세션·학습이 없을 때**, **Ubuntu Bash에서** 아래 블록 전체를 실행한다. 현재 launcher를 백업·복구하고, 새 네 가지 plan을 만든 뒤 detached tmux shell에서 학습·평가·profiling을 시작하고 화면에 연결한다. 기존 환경·manifest·가중치를 재사용하며 다운로드는 없다. 학습 실행 단계에서 오류가 나면 세션에 출력이 남는다.

```bash
(
set -e
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
STUDY="/home/master/mithila/shMin-scale-study/$REL"
EXISTING="/home/master/mithila/shMin/$REL"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
export MANIFEST="$EXISTING/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$EXISTING/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)

cp -p "$STUDY/launch_scale_mechanism.sh" "$STUDY/launch_scale_mechanism.sh.crlf-backup-$STAMP"
sed -i 's/\r$//' "$STUDY/launch_scale_mechanism.sh"
bash -n "$STUDY/launch_scale_mechanism.sh"

OUT="$EXISTING/results/scale-mechanism-core-os32-seed3407-$STAMP"
DEVICE=cuda:0 OUTPUT="$OUT" \
  bash "$STUDY/launch_scale_mechanism.sh" --prepare-only core "$WEIGHTS"

tmux new-session -d -s scale-core -c /home/master /bin/bash
printf -v START_CMD '%q ' env PYTHONUNBUFFERED=1 bash "$OUT/plans/core/run_all.sh"
printf -v LOG_ARG '%q' "$OUT/run.log"
tmux send-keys -t scale-core \
  'set +e' C-m 'set -o pipefail' C-m \
  "$START_CMD 2>&1 | tee $LOG_ARG" C-m
tmux attach -t scale-core
)
```

새 결과는 `Result root:`에 출력된다. 그 아래 `plans/core/leaderboard/leaderboard.html`에 표를 저장한다. 실행 범위는 네 가지·seed 3407·각 150 epoch다. Optimizer 상태를 이어가는 resume 기능을 추가한 것은 아니다. 이후 설치에는 수정한 브랜치 설치기를 사용한다. 과거 `bb6f040` ZIP 안의 설치기에는 이번 줄바꿈 보정이 없다.

## 4. Ubuntu Bash: core 네 개만 시작

이 블록은 **학습 → 4개 데이터셋 평가 → 추론 profiling → 리더보드 갱신**을 실제로 시작한다. 설치·데이터·가중치 준비 후 사용자가 실행한다.

```bash
STUDY="/home/master/mithila/shMin-scale-study/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
EXISTING="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
export MANIFEST="$EXISTING/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$EXISTING/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
CORE="$EXISTING/results/scale-mechanism-core-os32-seed3407-$STAMP"

OUTPUT="$CORE" bash "$STUDY/launch_scale_mechanism.sh" core "$WEIGHTS"
```

명령 준비만 하려면 `core` 앞에 `--prepare-only`를 넣는다. 나중에 출력된 `run_all.sh`를 실행한다. Launcher는 새 출력 폴더를 요구하므로 같은 출력으로 다시 호출하지 않는다. 생성한 runner는 완료·설정 일치 단계를 건너뛰지만 미완료 출력은 거부한다. **Optimizer 상태를 복원하는 resume은 구현하지 않았다.** 중단되면 해당 폴더를 보존하고 어느 단계인지 확인한 뒤 별도 실행을 준비한다. 과거 PVT checkpoint를 이어 학습하는 명령은 아니다.

전체 데이터 학습에는 **early stopping이 없다.** 150 epoch 고정이며 `best.pth`는 최종 checkpoint다. Epoch 로그의 `validation: null`은 정상이다. 전체 test 지표는 각 모델 학습 종료 후 평가 단계에서 나온다.

## 5. 다음 단계는 각각 따로 실행

아래 명령은 4절의 `STUDY, EXISTING, PY, MANIFEST, WEIGHTS` 변수를 **같은 Bash 터미널에서** 재사용한다. 각 블록은 지정한 단계만 시작하며 전부 한꺼번에 붙여 넣는 스크립트가 아니다.

**해상도 비교: 새 학습 4개**

```bash
OUTPUT="$EXISTING/results/scale-mechanism-resolution-os16-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" resolution "$WEIGHTS"
```

**원본 보존 fusion: 새 학습 2개, core의 mean 대조군과 비교**

```bash
OUTPUT_STRIDE=32 OUTPUT="$EXISTING/results/scale-mechanism-fusion-os32-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" fusion "$WEIGHTS"
```

**Feature distillation: core의 네 번째 모델 학습 완료 후 새 학습 2개**

```bash
TEACHER="$CORE/runs/core/image_pyramid__mean__seed3407/best.pth"
OUTPUT_STRIDE=32 OUTPUT="$EXISTING/results/scale-mechanism-distill-os32-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" distill "$WEIGHTS" "$TEACHER"
```

터미널을 닫았다면 `CORE`를 **정확히 완료된 결과 경로**로 다시 지정한다. 가장 최신 폴더를 무조건 teacher로 고르면 안 된다. Stride 16 fusion/distillation은 `OUTPUT_STRIDE=16`과 `os16` 출력 이름을 사용한다. Teacher도 `<resolution-result-root>/runs/resolution/image_pyramid__mean__seed3407/best.pth`여야 한다. Stride가 다르면 loader가 거부한다.

## 6. 결과 위치와 판단 기준

Core 결과는 `$CORE` 아래에 있다.

| 파일 | 내용 |
| --- | --- |
| `run.log` | 전체 실행 로그 |
| `runs/core/<case>__seed3407/history.jsonl` | Epoch별 loss. 이후 distill 단계에는 가중치를 적용한 `distill` 항도 기록 |
| `runs/core/<case>__seed3407/test/metrics.json` | 데이터셋별 지표 21개. Sα(`S`), weighted Fβ(`wF`), MAE, E/Fβ 각 버전, IoU/Dice·경계 지표 등 |
| `runs/core/<case>__seed3407/profile.json` | 추론 파라미터·MACs/FLOPs·지연·메모리·분기 진단 |
| `plans/core/leaderboard/leaderboard.md` | 터미널에서 읽는 표 |
| `plans/core/leaderboard/leaderboard.html` | 브라우저에서 읽는 표 |

다른 단계는 경로의 `core`가 해당 단계 이름으로 바뀐다. 같은 터미널에서 core 표를 보려면:

```bash
less "$CORE/plans/core/leaderboard/leaderboard.md"
```

Core 네 모델은 같은 분할 성능 비교군에 들어간다. Stride·teacher 조건이 달라진 이후 실험은 별도 비교군으로 기록하므로 다른 구조·목표를 섞어 평균내지 않는다. 단계별 대응 결과를 명시적으로 비교한다. Profiling에는 참고군 encoder 3회와 학생 1회의 전체 추론을 각각 포함하고 teacher는 학생 추론에서 제외한다. Padding은 연산량 추정에서 데이터 이동으로 분류하며 floating 연산으로 세지 않는다. 실제 지연 측정에는 포함된다.

먼저 native mean이 **single뿐 아니라 projection mean도 이기는지** 확인한다. Single만 이기면 추가 용량의 효과일 수 있다. 실제 RGB scale 참고군부터 이득이 없으면 전체 조합을 늘리기 전에 특징·decoder·scale 범위를 재검토한다. Stride 16은 공간 정보 보존, anchored fusion은 보조 분기의 원본 교란, distillation은 실제 scale 특징으로 직접 감독하는 효과를 확인한다. Distillation 성능 향상만으로 scale 고유 효과를 증명할 수 없으므로 projection 대조군이 필요하다. Seed 하나는 탐색 결과다. 모든 데이터셋·비용·후속 반복 seed를 함께 보고한다. Feature matching loss 감소만으로 의미적 동등성이나 사라진 영상 디테일 복원을 주장하지 않는다.

## 7. 참고 논문과 적용 범위

확인일 **2026-11-04**. 한국어 제목은 설명용 번역이다.

- **A ConvNet for the 2020s / 2020년대를 위한 합성곱 신경망.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), **2022**, 학회. [공식 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), [저자 프리프린트](https://arxiv.org/abs/2201.03545), [TorchVision 0.26.0 구현](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py). Backbone을 사용한다. Stride 16의 padding·stride 변경과 segmentation head는 이번 연구의 변형이다.
- **ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection / ZoomNeXt: 위장 객체 검출을 위한 통합 협력 피라미드 네트워크.** **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, **2024**, 저널. [게재 DOI](https://doi.org/10.1109/TPAMI.2024.3417329), [저자 논문 v4](https://arxiv.org/abs/2310.20208v4). 실제 resize RGB 참고군의 동기다. 이번 ConvNeXt·마지막 stage·mean 참고군은 ZoomNeXt MHSIU/RGPU 모델 재현이 아니다.
- **FitNets: Hints for Thin Deep Nets / FitNets: 얇고 깊은 신경망을 위한 힌트.** 읽은 버전: **arXiv:1412.6550v4 (2015)**, 프리프린트. 정식 발표처는 이번 확인에서 별도로 검증하지 않았다. [저자 논문](https://arxiv.org/abs/1412.6550v4). Teacher 중간 표현을 학습 신호로 쓰는 동기를 참고한다. 분기별 cosine loss·scale target·계수·학습 절차는 이번 설계이며 FitNets 절차 재현이 아니다.

`anchor_gated`는 특정 fusion 논문을 재현한 것으로 귀속하지 않는 연구 제안이다. 기존 생성·활용 논문과 adaptation 기록은 [실험 참고문헌](../README.ko.md#참고-논문), [생성 안내](GENERATION.md), [활용 안내](UTILIZATION.md)에 보존했다. [구현·변경 기록](../../../../Progress/2026-11-04/Pseudo-Scale-Mechanism/README.md).
