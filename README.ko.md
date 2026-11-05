# 2026-11-05 · INS_COD — Pseudo-scale 자체의 효과를 확인하는 비교

[English](README.md) | **한국어**

연구 브랜치 **`research/pseudo-scale-mechanism`**에서 pseudo-scale을 같은 해상도의 학습 분기 및 실제 RGB 멀티스케일과 비교한다. **처음에는 seed 하나로 네 개만 실행**한다. 기존 main/priority8 실험과 결과는 보존했다.

| 첫 실행 | 달라지는 부분 | 추론 encoder 호출 |
| --- | --- | ---: |
| `single` | Pseudo 분기 없음 | 1 |
| `projection__mean` | 같은 해상도에서 학습 분기 두 개 추가 | 1 |
| `native_conv__mean` | 같은 분기 연산을 1.5배·0.5배 특징 해상도에서 수행 | 1 |
| `image_pyramid__mean` | 실제 RGB 1배·1.5배·0.5배를 공유 encoder로 처리한 뒤 mean fusion | 3 |

ConvNeXt-Tiny의 마지막 stage만 decoder에 전달한다. 이후 **각각 따로 실행하는** stride 16 비교, 원본 특징을 보존하는 `anchor_gated` fusion, 실제 RGB scale 참고군으로 학습한 frozen feature teacher도 추가했다. Teacher 감독 대상은 **특징이며 pseudo-mask label이 아니다.** 학생 추론에는 teacher가 없다. 새 모델의 학습 성능이나 향상을 확인했다고 주장하지 않는다.

**2026-11-05 배포 복구:** Windows ZIP의 CRLF 줄바꿈으로 학습 전 launcher가 중단되는 문제를 재현하고 전송기·설치기를 수정했다. [현재 서버 복구 명령](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md#이미-설치한-crlf-실행-스크립트-복구).

## 설치와 실행

[한국어: 환경·브랜치 전송·설치·전체 단계별 명령·결과 위치](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md) · [English guide](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md).

기존 [고정 환경](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.ko.md), full4040 manifest와 데이터 파일, 공식 TorchVision ConvNeXt-Tiny 로컬 가중치를 재사용한다. 안내서는 새 코드를 **`/home/master/mithila/shMin-scale-study`**에 별도로 설치한다. 결과는 기존 **`/home/master/mithila/shMin`** 실험 트리의 새 폴더에 저장한다.

브랜치 설치와 입력 준비 후 **Ubuntu Bash에서** 실행한다.

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

**네 개의** 학습·평가·profiling을 순서대로 시작한다. Seed 3407, 150 epoch, batch 4, 입력 384×384, 채널 48, full4040 학습이다. Early stopping과 test 기반 checkpoint 선택은 없으며 `best.pth`는 최종 checkpoint다. 명령 준비만 하려면 `core` 앞에 `--prepare-only`를 넣는다. 이후 `resolution` 4개, `fusion` 2개, `distill` 2개 명령은 안내서에 따로 있으며 자동 시작하지 않는다. Distillation은 같은 stride·manifest로 학습을 완료한 실제 scale 참고군 checkpoint가 필요하다.

## 결과와 해석

Launcher가 결과 루트를 출력한다. Core 리더보드는 **`$CORE/plans/core/leaderboard/leaderboard.html`** 또는 `.md`다. 각 모델의 `runs/core/<case>__seed3407/test/metrics.json`에는 Sα·weighted Fβ·MAE·E/Fβ 각 버전 등 **데이터셋별 21개 지표**가 나온다. 학습 로그에는 loss를 표시하고 전체 test 지표는 평가 후 기록한다. 추론 비용은 참고군 encoder 3회, 학생 1회를 실제로 포함해 측정한다.

학습은 CAMO 1,000 + COD10K 3,040장이다. 평가는 CAMO 250·CHAMELEON 76·COD10K 2,026·NC4K 4,121장이다. [기존 COD10K 분할 간 중복 7쌍과 CHAMELEON 크기 불일치 2건](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md)은 계속 명시한다. Backbone·stride·teacher 조건이 다르면 별도 비교군으로 기록하며 seed 하나로 통계적 유의성을 주장하지 않는다. 기준 모델만 이겼다는 이유로 scale 효과라고 결론내리지 않고 학습 분기 대조군도 비교한다.

[브랜치 검증](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-04-mechanism/README.ko.md) · [연구 판단 기록](Progress/2026-11-04/Pseudo-Scale-Mechanism/README.md) · [이전 ConvNeXt priority8](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.ko.md) · [최근 업로드된 실제 PVT 결과](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md).

이번 **ConvNeXt·ZoomNeXt·FitNets**의 참고 범위는 [영문](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md#7-references-and-adaptation-boundaries)·[한글](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md#7-참고-논문과-적용-범위) 안내서에 적었다. 기존 참고문헌은 아래에 보존한다.

## 참고 논문

**2026-11-04 추가:** [A ConvNet for the 2020s / 2020년대를 위한 합성곱 신경망](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)(IEEE/CVF CVPR 2022, 학회; ConvNeXt 백본), [Deep Residual Learning for Image Recognition / 영상 인식을 위한 심층 잔차 학습](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)(IEEE CVPR 2016, 학회; 후속 ResNet50 비교), [PVT v2: Improved Baselines with Pyramid Vision Transformer / 피라미드 비전 트랜스포머를 통한 개선된 기준 모델](https://doi.org/10.1007/s41095-022-0274-8)(*Computational Visual Media*, 2022, 저널; 기존 백본 보존). 실제 CNN은 기록한 TorchVision 변형을 사용하며 pseudo 생성기와 decoder는 이 연구의 적용이다.

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
