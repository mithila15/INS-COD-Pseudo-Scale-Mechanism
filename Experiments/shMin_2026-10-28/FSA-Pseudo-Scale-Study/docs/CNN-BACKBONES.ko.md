# 2026-11-04 · CNN Backbone Study — ConvNeXt Priority8 비교실험

[English](CNN-BACKBONES.md) | **한국어** · [실험 안내](../README.ko.md) · [지도 신호와 pseudo-label](PSEUDO-LABELS.ko.md)

이번에는 기존 **priority8 후보만 ConvNeXt-Tiny의 마지막 stage로 바꾸어 seed `3407` 하나로 실행**한다. 총 8개는 pseudo를 쓰지 않는 baseline, `native_conv + asff/sk/mean`, `projection + mean`, `dysample + asff/sk/mean`이다. 사용자가 지정한 후보 목록이며, 전체 generator×utilizer 조합이나 추가 seed, ResNet 학습은 이 계획에 포함하지 않는다. 업로드된 full4040 기록의 기존 조건을 사용하며, 그 기록을 서버의 실시간 상태로 해석하지 않는다.

**연구 브랜치 갱신:** [새 mechanism 실행 안내](SCALE-MECHANISM.ko.md)는 대조 실험 네 개부터 시작하고, 실제 RGB scale 참고군·stride 16·선택적 feature teacher를 추가한다. 아래 8개 명령은 보존한 기존 priority8을 별도로 실행하는 방법이다.

## 백본 변경으로 무엇을 확인하는가

이 연구에서 scale은 다음 세 가지 의미를 구분해야 한다.

| 구분 | 계산 방식 | 이번 ConvNeXt 후보 실험 |
| --- | --- | --- |
| 입력 이미지의 여러 scale | RGB를 여러 크기로 바꾸고 각각 인코더에 입력 | 사용하지 않음. 이미지당 인코더 호출 1회 |
| 인코더의 여러 stage | 깊이·공간 해상도가 다른 여러 인코더 출력을 디코더에 전달 | `--feature-mode last`에서는 마지막 stage만 전달 |
| Pseudo-scale branch | 하나의 인코더 특징을 여러 중간 해상도에서 변환한 뒤 정렬·융합 | `native_conv`·`dysample`에 사용. `projection`은 원래 해상도의 학습 분기 대조군, baseline은 우회 |

PVTv2뿐 아니라 **ConvNeXt와 ResNet도 계층적 백본**이다. 따라서 PVTv2를 CNN으로 바꾸는 것만으로 다중 stage 특징이 제거되는 것은 아니다. 디코더가 어떤 인코더 출력을 사용하는지가 별도 조건이다. `last`에서도 앞선 backbone layer는 마지막 특징을 계산하기 위해 실행되지만, 그 출력이 디코더로 직접 연결되지는 않는다. [ConvNeXt 논문](https://arxiv.org/abs/2201.03545), [버전을 고정한 ConvNeXt 구현](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py), [PVTv2 논문](https://arxiv.org/abs/2106.13797).

연구 가설은 특징 수준 scale 변환으로 유용한 scale 정보를 만들면서 이미지를 여러 번 인코딩하는 비용을 줄일 수 있다는 것이다. 인코딩한 특징을 resize한 값과 resize한 이미지를 인코딩한 값은 일반적으로 같지 않다. 아래 priority8의 loss는 분할 정답을 사용하며 teacher 특징 일치 loss가 없다. 연구 브랜치에서는 선택적 distillation 단계를 따로 제공한다. 따라서 실제 다중 scale 특징과의 유사도와 계산 비용 절감은 앞으로 측정해야 하는 대상이다.

## 구현된 백본과 특징 선택

| `--backbone` | 4개 stage의 채널 수 | `--feature-mode last`의 입력 특징 |
| --- | --- | --- |
| `pvt_v2_b2` | 64, 128, 320, 512 | `reduction_5`, 512채널, stride 32 |
| `convnext_tiny` | 96, 192, 384, 768 | TorchVision `features.7`, 768채널, stride 32 |
| `resnet50` | 256, 512, 1024, 2048 | TorchVision `layer4`, 2048채널, stride 32 |

각 stage의 nominal stride는 4, 8, 16, 32다. TorchVision ResNet50은 bottleneck의 stride 위치가 원본과 다른 V1.5 변형이다. 원 논문의 Caffe 구현을 그대로 재현한다고 표현하지 않는다. [ResNet50 구현과 변형 설명](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

`--feature-mode multistage`는 기존 4개 stage projection, prior, top-down refinement 설계를 유지하며 별도 대조 실험에 사용할 수 있다. 기존 기본값인 `pvt_v2_b2 + multistage`와 과거 checkpoint도 지원한다. 보존된 `--legacy-reference`는 그 기존 백본·mode 조합에서만 유효하다.

`last`의 처리 과정을 priority8의 `native_conv + sk`를 예로 들면 다음과 같다.

```text
RGB 이미지 (384 × 384)
  → ImageNet 채널 정규화
  → ConvNeXt-Tiny 인코더 1회
  → 마지막 특징 (768 × 12 × 12)
  → 1 × 1 convolution + BatchNorm + ReLU (48 × 12 × 12)
  → native_conv: main / 1.5배 branch / 0.5배 branch
  → 세 branch 모두 48 × 12 × 12로 정렬
  → SK 채널별 branch 선택
  → region·edge 보조 head
  → region sigmoid gate + 3 × 3 depthwise-separable refinement
  → stride 32에서 deep supervision head 1개
  → [bilinear 2배 확대 + 3 × 3 depthwise-separable convolution] 5회
  → 1 × 1 mask predictor → 입력 이미지 크기의 logits
```

region/edge head는 각각 1×1 convolution block, 3×3 depthwise-separable block 2개, 1×1 logit layer를 쓴다. 마지막 stage 전용 디코더에는 ASPP, 서로 다른 kernel의 병렬 refinement, 앞선 인코더 stage의 skip이 없다. 마지막 특징을 복사하여 독립된 인코더 stage 4개처럼 취급하지도 않는다. 분할·deep·region·edge·UAL loss는 유지하며, deep loss는 head 평균에 계수 `0.25`를 적용하므로 head가 1개여도 총 계수는 같다.

384×384 입력에서 `native_conv`의 중간 크기는 18×18과 6×6이며, 이후 모두 12×12로 돌아온다. 확대가 인코더에서 이미 사라진 세부 정보를 복원한다고 보장하지 않는다. SK는 전역 문맥에서 채널별 branch 가중치를 구하며 뺄셈 residual을 계산하지 않는다. 세부 연산 정의와 기존 논문 근거는 [생성 방식](GENERATION.md)과 [활용 방식](UTILIZATION.md)에 보존되어 있다.

## 로컬 사전학습 가중치

기존 [잠금 환경](../environment/README.ko.md)에 TorchVision `0.26.0`이 포함되어 있어 새 백본 때문에 패키지를 추가할 필요는 없다. **TorchVision 형식의 로컬 checkpoint**를 제공한다. 코드에서 CNN은 `weights=None`으로 생성하며 가중치를 자동으로 다운로드하지 않는다.

| 사용할 초기화 | 공식 파일 | TorchVision 표기 용량 |
| --- | --- | --- |
| ConvNeXt-Tiny `IMAGENET1K_V1` | [convnext_tiny-983f1562.pth](https://download.pytorch.org/models/convnext_tiny-983f1562.pth) | 약 109.1 MB |
| 추후 ResNet50 `IMAGENET1K_V2` | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | 약 97.8 MB |

URL은 설치된 TorchVision 0.26.0의 weight enum과 대조했다. 파일이 없으면 사용자가 직접 받아 SHA256을 기록한다. 학습 설정에도 제공한 파일의 hash가 저장된다. 이 가중치는 TorchVision의 ImageNet 학습 recipe이며 원 논문 사전학습을 완전히 재현한 것이라고 표현하지 않는다. [ConvNeXt 가중치 안내](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), [ResNet 가중치 안내](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

loader는 flat state dictionary 또는 `{"state_dict": ...}`를 받는다. 알려진 classification head만 제거한 뒤 나머지는 strict 로딩한다. ConvNeXt는 `classifier.0.{weight,bias}`와 `classifier.2.{weight,bias}`, ResNet은 `fc.{weight,bias}`, PVT는 `head.{weight,bias}`만 제외한다. 특징 가중치 누락, 무관한 추가 key, tensor shape 불일치, timm/Meta ConvNeXt 형식이나 `{"model": ...}` wrapper를 자동으로 무시하지 않는다. `--random-init`는 명시적인 별도 비교 조건이며 사전학습 파일이 없을 때의 자동 대체 동작이 아니다.

해당 가중치가 아직 없을 때만 아래 블록을 **직접 실행**한다. 약 109 MB 모델 파일을 받고 공식 파일명의 hash prefix를 검사한 뒤 전체 SHA256을 출력한다. 데이터셋 다운로드·학습은 하지 않으며 기존 파일은 덮어쓰지 않고 검사한다.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
"$PY" - "$STUDY/weights/convnext_tiny-983f1562.pth" <<'WEIGHT_PY'
import hashlib
from pathlib import Path
import sys
from torch.hub import download_url_to_file
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    download_url_to_file("https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
                         str(path), hash_prefix="983f1562", progress=True)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if not digest.startswith("983f1562"):
    raise SystemExit("Existing weight file does not match the official SHA256 prefix; preserved unchanged.")
print(f"Verified local weights: {path}\nSHA256: {digest}")
WEIGHT_PY
```

## 정확히 8개 실험 준비·실행

아래는 **수정된 소스를 설치한 Linux 실험 서버의 Bash 터미널** 명령이다. 기존 full4040 manifest와 데이터 파일을 재사용한다. 이미 기록된 COD10K 중복 인정 정책을 그대로 유지하며 audit를 변경하거나 완화하지 않는다. test는 CAMO 250장, CHAMELEON 76장, COD10K 2,026장, NC4K 4,121장이다. 기존 CHAMELEON geometry 불일치 2건에 대한 `resize-prediction-to-gt` 정책도 명시한다.

첫 블록은 실행 명령만 만든다. 학습·다운로드·평가를 시작하지 않는다. 백본과 로컬 가중치 종류가 일치해야 한다.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
MANIFEST="$STUDY/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$STUDY/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
OUT="$STUDY/results/convnext-last-priority8-seed3407-$STAMP"

test -s "$MANIFEST" && test -s "$WEIGHTS" &&
"$PY" "$STUDY/plan_study.py" \
  --phase priority8 --seeds 3407 \
  --backbone convnext_tiny --feature-mode last \
  --training-protocol full_train --manifest "$MANIFEST" \
  --backbone-weights "$WEIGHTS" \
  --epochs 150 --batch-size 4 --size 384 --channels 48 --workers 2 \
  --device cuda:0 --python "$PY" --test-geometry resize-prediction-to-gt \
  --output "$OUT/plans/priority8" --results "$OUT/runs/priority8"
```

출력의 `variant_count`, `run_count`가 모두 **8**, `seeds`가 `3407` 하나인지 확인한다. 이어 **같은 터미널**에서 다음 블록을 실행하면 8개 job을 순차적으로 진행한다. 각 job은 학습 → 최종 checkpoint의 전체 지표 평가 → 속도·복잡도 측정 순서로 실행되며 리더보드를 갱신한다.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$OUT/plans/priority8/run_all.sh" 2>&1 | tee "$OUT/run.log"
```

각 결과는 `$OUT/runs/single__seed3407`, `$OUT/runs/native_conv__sk__seed3407`처럼 별도 폴더에 저장한다. 리더보드는 `$OUT/plans/priority8/leaderboard/`에 생긴다. 기존 PVT 결과는 원래 폴더에 보존한다. `full_train`은 고정 최종 epoch checkpoint를 사용하므로 validation 기반 early stopping이나 test 기반 checkpoint 선택을 하지 않는다. 학습 로그의 `validation: null`은 정상이며 전체 test 지표는 별도 평가 단계에서 생성된다. 데이터셋별 기존 21개 지표에는 S-measure, weighted F-measure, MAE, adaptive/mean/max E-measure·F-measure, 고정 threshold와 경계 지표가 포함된다.

## 이번 8개 실험으로 판단할 수 있는 범위

목록에 포함된 no-pseudo baseline은 pseudo 후보들과 **동일한 ConvNeXt, 마지막 stage 디코더, 초기화, 학습 데이터, seed, 입력 크기, loss**를 사용한다. 따라서 이 백본·seed 조건에서 pseudo 후보가 분할 성능을 개선하는지 비교한다. `projection + mean`은 `native_conv + mean`에 대해 같은 해상도에서 학습 branch만 추가한 대조군이다.

별도 [mechanism 실험](SCALE-MECHANISM.ko.md)에 실제 RGB 멀티스케일 reference와 선택적 feature teacher를 구현했다. 아래 priority8 명령의 8개에는 포함하지 않는다. ResNet50 `layer4` 지원은 추후 명시적인 백본 비교용이며 이번 계획에는 없다. 안정적인 개선을 주장하려면 추가 seed 검증도 필요하다.

과거 PVT multistage와 이번 ConvNeXt last의 비교는 백본과 디코더가 받는 정보가 함께 달라진다. 그 점수 차이를 pseudo-scale 생성 하나의 효과로 설명할 수 없다. 같은 profiling 조건에서 파라미터 수·latency·메모리·FLOPs와 분할 지표를 함께 보고한다. 리더보드에서는 백본과 feature mode가 다른 조건을 별도 cohort로 둔다. 구현 완료를 새 ConvNeXt·ResNet 학습 결과로 주장하지 않는다.

## 참고 논문과 구현

확인일: **2026-11-04**. 한국어 제목은 이해를 돕기 위한 번역이다.

- **A ConvNet for the 2020s / 2020년대를 위한 합성곱 신경망.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), **2022**, 학회. [공식 proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), [저자 공개본](https://arxiv.org/abs/2201.03545), [TorchVision 0.26.0 구현](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py). CNN 백본에 사용하며 pseudo 생성기와 분할 디코더는 이번 연구에서 추가한 부분이다.
- **Deep Residual Learning for Image Recognition / 영상 인식을 위한 심층 잔차 학습.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**), **2016**, 학회. [공식 proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), [2015 technical-report 공개본](https://arxiv.org/abs/1512.03385), [TorchVision 0.26.0 구현](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/resnet.py). 선택 가능한 ResNet50 백본의 근거이며 `last`에서는 `layer4`만 head에 전달한다.
- **PVT v2: Improved Baselines with Pyramid Vision Transformer / PVT v2: 피라미드 비전 트랜스포머를 통한 개선된 기준 모델.** **Computational Visual Media**, **2022**, 저널, 8(3):415–424. [게재 DOI](https://doi.org/10.1007/s41095-022-0274-8), [저자 공개본과 journal record](https://arxiv.org/abs/2106.13797). 과거 백본을 별도 실험 조건으로 유지한다.
