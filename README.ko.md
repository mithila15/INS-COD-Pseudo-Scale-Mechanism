# 2026-11-02 · INS_COD — Pseudo-scale 비교 실험 실행 안내

[English](README.md) | **한국어**

단일 인코더 위장 객체 검출 모델에서 **pseudo-scale 특징을 생성하는 방법**과 **생성된 특징을 활용하는 방법**을 비교한다. 활용에는 차이 계산뿐 아니라 직접 fusion, 채널·공간 attention을 포함한다. **생성 12종 × 활용 13종**에 no-pseudo 기준 모델과 기존 FSA 전체 구조를 더해 **158개 구성**, seed 3개로 **474회**를 실행할 수 있다.

**최신 실제 결과:** [2026-11-02 full-4040 스냅샷 검토](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md) — **8개 중 3개 실험의 4개 테스트셋 평가 완료**, 1개는 15/150 epoch, 나머지 4개는 내보내기 자료에 실행 기록이 없다. Native-scale ASFF/SK fusion은 CAMO·CHAMELEON에서 개선되고 COD10K·NC4K는 혼재한다. Seed 하나의 중간 결과이며 전체 우승 모델은 정하지 않는다. [서버 전체 ZIP·예측 마스크](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102).

이 README는 실험 실행 순서다. 각 변형이 무엇을 비교하는지는 [방법 목록과 비교 기준](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.ko.md)을 참고한다. 참고 논문과 적용 범위는 아래에 보존했다. 과거 holdout의 [priority8 결과](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-report-20261101_172449/README.ko.md)는 seed 1개의 8개 실험이며 **테스트셋 3개만 포함하고 COD10K-test가 빠져 있다**.

기존 결과에는 [4개 테스트셋 복구·전체 학습 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md)를 사용한다. 공식 ZoomNeXt와 비교하면 과거 holdout 학습은 **3,636장 + validation 404장**, ZoomNeXt는 **4,040장 학습**이다. 새 `full_train`은 4,040장 전체와 최종 epoch 고정 체크포인트를 사용하며 나머지 구조·학습 조건 차이는 남는다. [Windows pull·SCP 전송](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md#windows에서-수정-코드-전송)은 기존 서버 checkout과 결과를 보존한다.

서버 검사에서 **COD10K 학습 원본/test의 정확한 중복 7쌍**을 확인했다. 과거 holdout에서는 **train/test 6쌍·validation/test 1쌍**, 새 `full_train`에서는 **train/test 7쌍·validation/test 0쌍**이다. 준비의 기본값은 중단이다. [명시적 중복 처리 규약](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md#확인된-cod10k-중복을-명시적으로-기록하고-진행하기)은 확인한 ID·해시 쌍만 기록하고 배포 개수를 유지하며 `disjoint=false`를 표시한다. COD10K 평가가 독립적이 되는 것은 아니다. [근거와 한계](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/README.ko.md).

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

이 검사는 의존성 import와 연산 검증이며 학습을 실행하지 않는다. 설치 조건, CUDA 확인, 문제 해결, 기존 로컬 검증 환경과 신규 설치의 차이는 [환경 설정 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.ko.md)에 있다. 최초 환경 검사에서는 새 환경의 lock 전체 설치와 실제 데이터 학습을 검증하지 않았다. 2026-11-02 서버 스냅샷에는 사용자가 실행한 실제 학습·평가가 포함되며, 이번 자료 검토에서 환경을 다시 설치하거나 학습을 재실행하지 않았다.

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

`prepare`는 이미지·마스크 대응과 해시를 검사하고 COD10K의 `COD10K-CAM-` prefix를 선택한 뒤 고정 분할을 만든다. CAMO와 COD10K 학습 자료는 필수다. CLI 기본값은 **테스트셋 4개 전체**를 요구한다: CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121. 의도적인 일부 선택은 `--test-datasets`로 명시하고 `none`은 학습·validation만 준비한다. 일부 테스트셋만 평가한 상태는 리더보드에서도 전체 벤치마크 미완료로 표시한다. **비교하는 모든 run에서 같은 manifest를 사용한다.** 테스트 평가에는 manifest에 테스트셋이 하나 이상 있어야 한다.

기본 10% holdout은 원 학습 4,040장에서 **학습 3,636장, validation 404장**을 만든다. 아래 명령은 이 기본 `holdout` 프로토콜이다. 전체 학습을 새로 시작하려면 준비와 plan 양쪽에 `--training-protocol full_train`을 주고 `--val-fraction`을 생략하거나 `0`으로 지정한다. [기존 priority8 plan 변환](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md#별도-전체-데이터-학습-비교)은 과거 결과를 보존한다. 파일명 변경본, 테스트셋 명시 선택, 평가 규약은 [상세 실행 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md)에 있다.

## 3. 비교 범위 선택과 명령 생성

**전체 실험은 `cross`로 진행한다.** 이 안에 활용기를 고정한 생성 비교와 생성기를 고정한 활용 비교가 모두 들어 있다. 나머지 phase는 한 축부터 따로 분석할 때 선택하며, 세 phase를 모두 추가 실행해야 하는 것은 아니다.

| Phase | 바꾸는 항목 | 고정 항목 | 구성 수 | seed 3개 실행 수 |
| --- | --- | --- | ---: | ---: |
| `generation` | 생성기 12개 | 기본 `--utilizer mean` | 14 | 42 |
| `utilization` | 활용기 13개 | `--generator legacy` 또는 validation으로 선택한 생성기 | 15 | 45 |
| `cross` | 모든 생성기·활용기 조합 | 없음 | 158 | 474 |
| `priority8` | 고정한 우선 비교 8종 | 사전에 정한 변형 | 8 | 24; `--seeds 3407`이면 8 |

세 행렬 phase에는 `single`, `legacy_full` 대조군이 포함된다. `priority8`은 `single`과 고정한 7개 변형이며 `legacy_full`은 제외한다. 단계별로 진행한다면 mean fusion 아래 생성기를 비교하고 validation 결과를 본 뒤, 활용 비교에 쓸 생성기를 직접 선택한다. phase마다 결과 경로를 분리한다. 자동 우승 조합 선택 기능은 없다. `cross`에서는 아래 `GENERATOR`·`UTILIZER` 값이 전체 조합을 제한하지 않는다.

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

이 단계는 `plan.json`, `train.sh`, `evaluate.sh`, `profile.sh`, `run_all.sh`를 생성하며 **학습을 시작하지 않는다.** 비교군의 크기, 채널 수, epochs, batch size, seeds, manifest와 backbone 초기화 조건을 고정한다. 기본 Adam lr 1e-4, encoder lr ×0.1, 첫 patch embedding 고정, 공통 loss를 사용한다. [전체 공통 조건](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.ko.md).

## 4. 학습·평가·리더보드 자동 실행

다음 명령은 각 run의 **학습 → 테스트 평가 → 추론 비용 측정**을 순차 실행하고, 단계가 끝날 때마다 리더보드를 갱신한다. 기본 holdout의 `best.pth`는 **validation MAE만으로 선택**한다. `full_train`에서는 **최종 epoch 모델**이며 validation MAE는 null이고 validation 순위도 없다. 전체 비교군은 실행 전에 plan으로 확정하며, 테스트 순위는 보고용이고 체크포인트 선택에는 쓰지 않는다.

```bash
bash "$PLAN/run_all.sh"
```

브라우저로 **`$PLAN/leaderboard/leaderboard.html`**을 열면 된다. 전체 선택 순위는 validation MAE 기준이고, 테스트 표는 데이터셋별로 분리한다(기본 S-measure 순위). 비용 표는 같은 장비·조건끼리 비교한다. seed 평균·표본 표준편차·시행 수·완료 상태·비교 그룹을 표시하며, 프로토콜이나 완료된 seed 집합이 다르면 별도 그룹으로 둔다. HTML 정렬·검색을 지원하고 CSV·JSON·Markdown도 저장한다.

학습만 하고 validation 리더보드를 자동 갱신하려면 대신 `bash "$PLAN/train.sh"`를 실행한다. 이후 `run_all.sh`는 조건이 일치하는 완료 단계를 건너뛴다. 중단된 학습의 optimizer resume는 지원하지 않는다. 부분 결과를 보존하고 새 run 경로·plan을 사용한다. 체크포인트·예측·프로파일 원본은 덮어쓰지 않으며, 리더보드 파생 파일만 갱신한다.

## 5. Validation 결과와 추론 비용 확인

필요하면 다음 명령으로 profiling을 마치고 별도 집계 스냅샷을 남긴다.

```bash
bash "$PLAN/profile.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --plan "$PLAN/plan.json" \
  --output "$RUN/summary/validation-$PHASE"
```

파라미터 수·latency·throughput·CUDA 최대 메모리와 **MACs·FLOPs**, 별도로 표시한 비행렬 연산 추정량·연산 커버리지를 기록한다. 기본 규약은 **합성곱·행렬 multiply-accumulate 1회 = 2 FLOPs**이다. 보간·샘플링·비선형 연산은 명시한 근사 규칙으로 계산하며 하드웨어 명령어 수가 아니다. 미지원 연산은 목록에 남긴다. 합성 RGB, batch 1, FP32, 모든 head를 포함하며 데이터 로딩·전송·학습 메모리는 제외한다. [지표 정의와 한계](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/EVALUATION.ko.md).

## 6. 테스트 지표와 논문 표 확인

학습만 따로 실행했다면 아래 명령으로 테스트 평가를 완료하고 별도 보고서를 저장한다. 이미 완료된 단계는 동일성 검사 후 건너뛴다.

```bash
bash "$PLAN/evaluate.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --plan "$PLAN/plan.json" \
  --test-metric S --output "$RUN/summary/test-$PHASE"
```

데이터셋별 **21개 수치**를 출력한다. S·weighted F·MAE, E와 일반 Fβ 각각 adaptive/mean/max, 고정 임계값 IoU·Dice/F1·precision·recall·specificity·BER·FPR·FNR·accuracy·balanced accuracy·Boundary IoU이다. Dice와 F1은 같은 값이므로 별개의 근거로 세지 않는다. 임계값 평균·최댓값은 데이터셋 평균 곡선에서 계산한다. **PR/Fβ/E 곡선 CSV+SVG+PNG, 이미지별 점수, 원본 크기 확률 PNG**도 저장한다. 저장 공간을 줄이려면 직접 평가 명령에 `--no-save-predictions`를 지정한다. 수치·곡선은 계속 저장한다.

`--test-metric`에 `wF`, `Fbeta_max`, `MAE`, `BoundaryIoU_0_5` 등을 주면 해당 지표로 테스트 표를 정렬한다. 높음/낮음 방향을 함께 기록하고, 없는 데이터나 점수는 채워 넣지 않는다. 순위 자체는 통계적 유의성 검정이 아니다. 기존 체크포인트를 재평가할 때는 **새 폴더** `test-v2` 등에 저장하고 집계에 `--evaluation-subdir test-v2`를 지정한다. 기존 profile도 보존하려면 새 `profile-v2.json` 경로와 planner/집계의 `--profile-name profile-v2.json`을 사용한다. 기존·확장 지표 프로토콜은 별도로 묶는다.

**phase root 하나씩** 집계한다. 활용기 고정 생성 비교와 생성기 고정 활용 비교를 구분하고 `single`, 전체 구조 참고군 `legacy_full`을 포함한다. 동일 seed·데이터 분할·초기화·측정 조건을 유지한다.

```text
results/<run-id>/
  data/manifest.json
  plans/<phase>/{plan.json,train.sh,evaluate.sh,profile.sh,run_all.sh}
  plans/<phase>/leaderboard/{leaderboard.html,leaderboard.md,leaderboard.csv,leaderboard.json,summary.json,summary.csv}
  runs/<phase>/<variant>__seed<seed>/
    config.json, manifest.json, history.jsonl, status.json
    best.pth, last.pth
    test/{metrics.json,per_image_metrics.csv}
    test/predictions/<dataset-key>/*.png
    test/curves/<dataset-key>/*.{csv,svg,png}
    profile.json
  summary/<snapshot>/{summary.json,summary.csv,leaderboard.html,leaderboard.md,leaderboard.csv,leaderboard.json}
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
| [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)<br>Boundary IoU: 객체 중심 이미지 분할 평가의 개선 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, conference / 학회 | 추가 경계 평가지표. 저자 erosion 규약 적용; 확인일 2026-10-29 / additional boundary evaluation |

## 검증 상태와 기존 실험

기록된 코드 검증은 **281 tests + 67 subtests 통과**, CPU/GPU 추론 확인, optimizer step 0회다. 이 초기 검사는 실제 데이터 수렴·순위를 측정하지 않았다. 과거 8개 holdout 실험 보고서는 3개 테스트셋 결과다. [2026-11-02 서버 검토](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md)에는 **전체 데이터 학습·4개 테스트셋 평가를 끝낸 3개 실험**, 부분 실행 1개와 남은 계획 4개가 추가되며 두 프로토콜의 결과를 섞지 않는다. [검증 산출물](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-28/README.md)을 참고한다.

기존 BRG main 실험은 [COD](https://github.com/sh-min/COD)에 유지하고 CoCoNet은 이번 업로드에서 제외했다. 보존 코드·라이선스는 유지한다. 이전 정리 과정은 [이관 기록](docs/MIGRATION.md), [실험 목록](Experiments/README.md), [비교 도구](Benchmarks/README.md), [연구 결정](Progress/README.md)에서 확인한다.

2026-10-29 평가·리더보드 검증: **522 tests + 67 subtests**, 전체 158개 초기화 모델 연산 커버리지, 브라우저 8개 검사 통과. 학습·다운로드는 실행하지 않았다. [2026-10-29 검증 기록](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-29-metrics/README.md).

[2026-11-01 변경 기록](Progress/2026-11-01/Benchmark-Completion/README.md)은 프로토콜 변경을 기록한다. [2026-11-02 검토 기록](Progress/2026-11-02/Server-Results-Review/README.md)은 업로드 복구·실제 중간 결과·압축 및 해시 검증·남은 사전 계획 실험을 정리한다. 이번 검토에서 학습·추론은 실행하지 않았다.
