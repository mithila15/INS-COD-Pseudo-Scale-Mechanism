# 2026-10-28 · INS_COD — Pseudo-scale 비교 실험 실행 안내

[English](README.md) | **한국어**

단일 인코더 위장 객체 검출 모델에서 **pseudo-scale 특징을 생성하는 방법**과 **생성된 특징을 활용하는 방법**을 비교한다. 활용에는 차이 계산뿐 아니라 직접 fusion, 채널·공간 attention을 포함한다. **생성 12종 × 활용 13종**에 no-pseudo 기준 모델과 기존 FSA 전체 구조를 더해 **158개 구성**, seed 3개로 **474회**를 실행할 수 있다.

이 README는 실험 실행 순서다. 각 변형이 무엇을 비교하는지는 [방법 목록과 비교 기준](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.ko.md)을 참고한다. 참고 논문과 적용 범위는 아래에 보존했다. 아직 실제 정확도 순위는 측정하지 않았다.

## 1. 환경 설정

이 저장소의 checkout을 준비하고 `REPO`를 해당 장비의 절대 경로로 바꾼다. 아래 블록들은 **같은 터미널**에서 실행한다. 환경은 Python 3.12.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.30과 저장소의 dependency lock을 사용한다.

[uv 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 uv를 먼저 준비한다. 다음은 **새 환경** 생성과 Python·패키지 다운로드 단계다. 이미 준비된 환경이 있으면 `PY` 경로를 재사용하고 의도적으로 재구축할 때만 생성/sync를 수행한다. GPU에는 lock의 CUDA 런타임과 호환되는 드라이버가 필요하고 CPU 실행은 `DEVICE=cpu`를 지정한다. 이 lock은 Linux CUDA 프로필이며 별도의 CPU 전용·macOS lock은 아니다.

```bash
REPO=/home/robin/shMin/INS_COD
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study-clean/bin/python"

if [ -e "$REPO/.venv/fsa-study-clean" ]; then
  printf '%s\n' 'Environment already exists; run the checks below before reusing it.'
else
  uv python install --no-bin 3.12.13 &&
  uv venv --directory "$STUDY" --no-project --managed-python --python 3.12.13 \
    "$REPO/.venv/fsa-study-clean" &&
  uv pip sync --directory "$STUDY" --python "$PY" --require-hashes \
    "$STUDY/environment/requirements.lock"
fi

"$PY" -c 'import torch, torchvision, timm, einops, py_sod_metrics; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

이 검사는 의존성 import와 연산 검증이며 학습을 실행하지 않는다. 설치 조건, CUDA 확인, 문제 해결, 기존 로컬 검증 환경과 신규 설치의 차이는 [환경 설정 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.ko.md)에 있다. 이번 작업에서 완전히 새 환경의 lock 전체 설치와 실제 사전학습·데이터 학습은 검증하지 않았다.

## 2. 기존 데이터와 backbone 가중치 준비

`DATA`, `WEIGHTS`를 실제 보유 자료의 절대 경로로 바꾼다. 가중치는 full FSA 체크포인트가 아닌 **PVTv2-B2 backbone** state dict여야 한다. flat 형식과 `state_dict` wrapper를 지원하며 classification head 키를 제거한 뒤 나머지는 strict하게 로드한다. 아래 명령은 데이터나 모델 가중치를 다운로드하지 않는다.

```bash
DATA=/absolute/path/to/COD-Data
WEIGHTS=/absolute/path/to/pvt_v2_b2.pth
DEVICE=cuda:0
RUN="$STUDY/results/$(date +%Y%m%d_%H%M%S)"
MANIFEST="$RUN/data/manifest.json"

"$PY" "$STUDY/run_study.py" prepare \
  --data-root "$DATA" --manifest "$MANIFEST" \
  --split-seed 1729 --val-fraction 0.1
```

지원하는 데이터 배치는 다음과 같다.

```text
COD-Data/
  COD10K-v3/Train/{Image,GT_Object}/
  COD10K-v3/Test/{Image,GT_Object}/
  CAMO-V.1.0-CVIU2019/Images/{Train,Test}/
  CAMO-V.1.0-CVIU2019/GT/
  CHAMELEON/{Imgs,GT}/
  NC4K/{Imgs,GT}/
```

`prepare`는 이미지·마스크 대응과 해시를 검사하고 COD10K의 `COD10K-CAM-` prefix를 선택한 뒤 고정 분할을 만든다. CAMO와 COD10K 학습 자료는 필수다. 존재하는 테스트셋을 포함하고 CHAMELEON/NC4K가 없으면 누락을 기록한다. **비교하는 모든 run에서 같은 manifest를 사용한다.** 테스트 평가에는 manifest에 테스트셋이 하나 이상 있어야 한다.

기본 10% holdout은 원 학습 4,040장에서 **학습 3,636장, validation 404장**을 만든다. 4,040장을 전부 학습한 논문 수치와 조건이 다르다. 파일명 변경본, 테스트셋 명시 선택, 평가 규약은 [상세 실행 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md)에 있다.

## 3. 비교 범위 선택과 명령 생성

**전체 실험은 `cross`로 진행한다.** 이 안에 활용기를 고정한 생성 비교와 생성기를 고정한 활용 비교가 모두 들어 있다. 나머지 phase는 한 축부터 따로 분석할 때 선택하며, 세 phase를 모두 추가 실행해야 하는 것은 아니다.

| Phase | 바꾸는 항목 | 고정 항목 | 구성 수 | seed 3개 실행 수 |
| --- | --- | --- | ---: | ---: |
| `generation` | 생성기 12개 | 기본 `--utilizer mean` | 14 | 42 |
| `utilization` | 활용기 13개 | `--generator legacy` 또는 validation으로 선택한 생성기 | 15 | 45 |
| `cross` | 모든 생성기·활용기 조합 | 없음 | 158 | 474 |

각 phase에는 `single`, `legacy_full` 대조군이 포함된다. 단계별로 진행한다면 mean fusion 아래 생성기를 비교하고 validation 결과를 본 뒤, 활용 비교에 쓸 생성기를 직접 선택한다. phase마다 결과 경로를 분리한다. 자동 우승 조합 선택 기능은 없다. `cross`에서는 아래 `GENERATOR`·`UTILIZER` 값이 전체 조합을 제한하지 않는다.

```bash
PHASE=cross
GENERATOR=legacy
UTILIZER=mean
PLAN="$RUN/plans/$PHASE"
RUNS="$RUN/runs/$PHASE"

"$PY" "$STUDY/plan_study.py" \
  --phase "$PHASE" --generator "$GENERATOR" --utilizer "$UTILIZER" \
  --seeds 3407 3408 3409 --manifest "$MANIFEST" \
  --backbone-weights "$WEIGHTS" --python "$PY" --device "$DEVICE" \
  --size 384 --channels 48 --epochs 150 --batch-size 4 --workers 2 \
  --output "$PLAN" --results "$RUNS"
```

이 단계는 `plan.json`, `train.sh`, `evaluate.sh`, `profile.sh`를 생성하며 **학습을 시작하지 않는다.** 비교군의 크기, 채널 수, epochs, batch size, seeds, manifest와 backbone 초기화 조건을 고정한다. 기본 Adam lr 1e-4, encoder lr ×0.1, 첫 patch embedding 고정, 공통 loss를 사용한다. [전체 공통 조건](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.ko.md).

## 4. 학습 시작

다음 줄을 실행하면 계획된 run을 순차 학습한다. 각 run의 `best.pth`는 **validation MAE만으로 선택**하고 `last.pth`와 설정·이력을 저장한다. 테스트 평가는 따로 실행한다.

```bash
bash "$PLAN/train.sh"
```

기존 출력은 덮어쓰지 않고 중단한다. 실패하면 shell이 멈추며 자동 resume는 없다. 중간 결과를 보존하고 재시작하는 run에는 새 출력 경로를 사용한다. 전체 script를 처음부터 다시 돌리기보다 `plan.json`에서 아직 실행하지 않은 명령을 선택한다.

## 5. Validation 결과와 추론 비용 확인

학습 후 각 run의 best 체크포인트를 profiling하고 validation 지표와 비용을 집계한다.

```bash
bash "$PLAN/profile.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --output "$RUN/summary/validation-$PHASE"
```

낮은 validation MAE를 기준으로 품질을 비교하고 같은 장비·입력 조건에서 파라미터 수, median latency, 메모리도 확인한다. Profiling은 합성 RGB, batch 1, FP32, 모든 예측 head의 추론을 측정한다. 데이터 로딩·전송·학습 메모리·FLOPs는 포함하지 않으며 정확도 검사도 아니다. seed 간 평균·표본 표준편차를 함께 보고 작은 차이를 곧바로 통계적 유의성으로 해석하지 않는다.

## 6. 비교 조건을 확정한 뒤 테스트 평가

Validation으로 비교 구성과 판단 기준을 정한 뒤 실행한다. 아래는 **선택한 계획의 모든 run**을 테스트하고 집계하는 명령이며 테스트 점수로 모델을 선택하지 않는다.

```bash
bash "$PLAN/evaluate.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --output "$RUN/summary/test-$PHASE"
```

데이터셋별 **S-measure, weighted F-measure, E-measure는 높을수록**, **MAE는 낮을수록** 좋다. 추론 비용을 함께 기록한다. 없는 데이터셋의 점수는 만들지 않는다. 활용기를 고정한 생성 비교, 생성기를 고정한 활용 비교, 전체 조합의 상호작용을 차례로 읽는다. `single` 기준 모델을 포함하고 `legacy_full`은 전체 구조 참고군으로 해석한다. 테스트 점수를 보면서 선택된 방법을 반복 수정하지 않는다.

집계는 **phase별 root 하나씩** 수행한다. 서로 다른 phase의 동일 seed 대조군을 합쳐 독립 시행 수로 세면 안 된다. 결과 저장 구조는 다음과 같다.

```text
results/<run-id>/
  data/manifest.json
  plans/<phase>/{plan.json,train.sh,evaluate.sh,profile.sh}
  runs/<phase>/<variant>__seed<seed>/
    config.json, manifest.json, history.jsonl, status.json
    best.pth, last.pth
    test/metrics.json
    profile.json
  summary/{validation-<phase>,test-<phase>}/{summary.json,summary.csv}
```

## 참고 논문

확인일 **2026-10-28**. 한국어 제목은 이해를 위한 번역이다. 원논문 전체 재현이 아니라 핵심 아이디어를 이 연구에 적용한 비교 실험이다. 수식·원문과의 차이는 [생성 방식](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/GENERATION.md), [활용 방식](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/UTILIZATION.md)에 있다.

| English title / 한국어 제목 | Official venue and publication status / 발표·공개처 | Applied idea / 적용 범위 |
| --- | --- | --- |
| [Fast Feature Pyramids for Object Detection](https://doi.org/10.1109/TPAMI.2014.2300479)<br>객체 검출을 위한 빠른 특징 피라미드 | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2014, journal / 저널 | FFP의 scale power law. 원래 저수준 특징의 통계적 보정을 학습 가능한 CNN 채널 지수로 바꾼 새 가설 / new deep-feature adaptation |
| [Exploring Plain Vision Transformer Backbones for Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_17)<br>객체 검출을 위한 비계층적 비전 트랜스포머 백본 탐구 | European Conference on Computer Vision (ECCV), 2022, conference / 학회 | SFP의 up/down feature pyramid를 각 PVT stage에 적용 / per-stage SFP adaptation |
| [Pyramid Scene Parsing Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)<br>피라미드 장면 파싱 네트워크 | IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, conference / 학회 | Pooling 문맥. 실제 이미지 스케일 근사와 구분 / pooled context, not verified image-scale approximation |
| [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)<br>의미론적 영상 분할을 위한 팽창 합성곱의 재고찰 | arXiv:1706.05587, 2017, preprint / 프리프린트; 정식 발표처 미확인 | 두 dilation branch로 축소 / reduced ASPP branches |
| [CARAFE: Content-Aware ReAssembly of FEatures](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html)<br>CARAFE: 내용 인지적 특징 재조립 | IEEE/CVF International Conference on Computer Vision (ICCV), 2019, conference / 학회 | 순수 PyTorch kernel prediction/reassembly. 공식 CUDA의 속도 재현 아님 / no fused-CUDA efficiency claim |
| [Learning to Upsample by Learning to Sample](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html)<br>샘플링을 학습하여 업샘플링하기 | IEEE/CVF International Conference on Computer Vision (ICCV), 2023, conference / 학회 | Offset sampling을 생성에 사용 / offsets for pseudo-scale generation |
| [Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)<br>선택적 커널 네트워크 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, conference / 학회 | Fuse/select만 차용 / channel selection on external branches |
| [Learning Spatial Fusion for Single-Shot Object Detection](https://arxiv.org/abs/1911.09516)<br>단일 단계 객체 탐지를 위한 공간 융합 학습 | arXiv:1911.09516, 2019, preprint / 프리프린트; 정식 발표처 미확인 | ASFF 공간 가중치 / spatial scale weights |
| [EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html)<br>EfficientDet: 확장 가능하고 효율적인 객체 탐지 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, conference / 학회 | BiFPN의 빠른 정규화 혼합만 차용 / fusion operator, not full BiFPN |
| [Attentional Feature Fusion](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html)<br>어텐션 기반 특징 융합 | IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, conference / 학회 | 논문 Eq.4 convex blend. 공식 코드의 ×2 gain 생략 / paper blending convention |
| [ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection](https://doi.org/10.1109/TPAMI.2024.3417329)<br>ZoomNeXt: 위장 객체 탐지를 위한 통합 협력 피라미드 네트워크 | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024, journal / 저널 | MHSIU-inspired 활용. 원래 실제 세 입력 인코딩을 재현하지 않음 / pseudo-feature adaptation |

## 검증 상태와 기존 실험

기록된 코드 검증은 **281 tests + 67 subtests 통과**, CPU/GPU 추론 확인, optimizer step 0회다. 실제 데이터 학습·수렴·방법 간 정확도 순위는 미검증이다. [검증 산출물](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-28/README.md)을 참고한다.

기존 BRG main 실험은 [COD](https://github.com/sh-min/COD)에 유지하고 CoCoNet은 이번 업로드에서 제외했다. 보존 코드·라이선스는 유지한다. 이전 정리 과정은 [이관 기록](docs/MIGRATION.md), [실험 목록](Experiments/README.md), [비교 도구](Benchmarks/README.md), [연구 결정](Progress/README.md)에서 확인한다.
