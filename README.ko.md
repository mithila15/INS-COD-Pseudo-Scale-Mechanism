# 2026-11-04 · INS_COD — ConvNeXt 마지막 Stage Priority8 실험

[English](README.md) | **한국어**

입력 이미지 한 장에서 pseudo-scale 특징을 만들고 결합 방식을 비교한다. 이번에는 **ConvNeXt-Tiny의 마지막 stage만 사용하고, 기존 priority8 × seed 3407 = 총 8회**를 실행한다. Planner 기본값도 이 범위다. 생성 12종 × 활용 13종 전체 목록은 나중에 명시적으로 요청하는 확장 실험용으로 보존했다.

ConvNeXt도 내부적으로 여러 stage를 만든다. 따라서 CNN으로 바꾸는 것만으로 다중 stage 효과가 사라지지는 않는다. 이번 `last` 모드는 stride 32의 마지막 특징 하나만 decoder에 전달하며 앞 stage의 skip, ASPP, 여러 kernel을 혼합하는 decoder를 사용하지 않는다. [구조·비교의 한계](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.ko.md). 후속 비교용 ResNet50 Layer4도 지원하지만 이번 8회에는 포함하지 않는다.

Pseudo-scale은 중간 특징 텐서다. **현재 모델은 pseudo-label을 생성하지 않는다.** 분할·deep·region은 GT mask, edge는 GT에서 계산한 경계로 학습하고 UAL은 예측 확신도를 정규화한다. [Pseudo-label과 감독 신호를 구분한 설명](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/PSEUDO-LABELS.ko.md).

## 1. 환경과 기존 데이터

[환경 설정](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.ko.md)의 Python 3.12.13 / torch 2.11.0 / torchvision 0.26.0 프로필을 설치하거나 서버의 기존 `fsa-study-clean`을 재사용한다. CNN 때문에 추가할 의존성은 없다. 먼저 [Windows pull·SCP·설치 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md#windows에서-수정-코드-전송)로 서버 소스를 갱신한다.

실행기는 기존 `results/priority8-full4040-fixed-20261101_200834/manifest.json`과 데이터 파일을 재사용한다. CAMO 1,000 + COD10K 3,040 = **4,040장 전체 학습**, validation holdout 없음이다. 테스트는 CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121장이다. 다른 manifest를 쓰려면 `MANIFEST` 환경변수에 절대 경로를 지정한다. 새 데이터 준비는 [상세 실행 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md)를 따른다.

이미 확인한 COD10K train/test 중복 7쌍과 CHAMELEON 이미지/GT 크기 불일치 2건은 기록을 유지한다. 모델 변경으로 독립 분할이 되는 것은 아니다. 평가는 기존 명시적 규약에 따라 예측을 원본 GT 크기로 맞춘다. [벤치마크 규약](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md).

## 2. ConvNeXt 가중치 준비

**torchvision ConvNeXt-Tiny IMAGENET1K_V1**의 `convnext_tiny-983f1562.pth`(약 109 MB)를 아래 위치에 준비한다. 기존 PVT 가중치는 호환되지 않는다. 로더는 flat state dict 또는 `state_dict` wrapper를 받으며 알려진 classification head만 제외하고 엄격하게 검사한다. 모델 생성·학습 과정에서 자동 다운로드하지 않는다. [공식 출처·직접 실행할 다운로드 명령](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.ko.md#로컬-사전학습-가중치).

## 3. 기존 우선순위 8개만 시작

서버 소스 갱신과 가중치 준비 후 **Ubuntu 실험 서버 터미널**에서 실행한다. 각 실험을 학습 → 4개 데이터셋 평가 → 추론 비용 측정 순서로 처리하고 단계마다 리더보드를 갱신한다. 기존 PVT checkpoint를 이어 학습하는 과정은 아니다.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
WEIGHTS="$STUDY/weights/convnext_tiny-983f1562.pth"

bash "$STUDY/launch_convnext_priority8.sh" "$WEIGHTS"
```

학습 없이 명령만 확인하려면 `"$WEIGHTS"` 앞에 `--prepare-only`를 넣는다. 출력된 새 plan의 `run_all.sh`로 이후 시작할 수 있다. 새 결과 위치를 지정하려면 `OUTPUT` 환경변수에 아직 없는 절대 경로를 넣는다. 기존 결과가 있으면 덮어쓰지 않고 중단한다.

| 순서 | 실험 | 비교 목적 |
| ---: | --- | --- |
| 1 | `single` | 같은 ConvNeXt/마지막 stage decoder에서 pseudo 생성·결합 제외 |
| 2 | `native_conv__asff` | Resize → convolution → 정렬 후 공간 가중 결합 |
| 3 | `native_conv__sk` | 같은 생성기 + 채널 선택 결합 |
| 4 | `native_conv__mean` | 같은 생성기 + 평균 결합 |
| 5 | `projection__mean` | 원래 특징 해상도에서 동일한 분기 합성곱 사용 |
| 6 | `dysample__asff` | 학습형 업샘플링 + 공간 가중 결합 |
| 7 | `dysample__sk` | 학습형 업샘플링 + 채널 선택 결합 |
| 8 | `dysample__mean` | 학습형 업샘플링 + 평균 결합 |

공통 조건은 **150 epoch, batch 4, 384×384 입력, 특징 채널 48, seed 3407**이다. Adam lr 1e-4, encoder lr ×0.1, 첫 stem/encoder BN 고정과 기존 loss 가중치를 유지한다. `full_train`은 **early stopping 없이 최종 epoch를 고정 선택**하며 validation/test 점수로 checkpoint를 선택하지 않는다. 이때 `best.pth`는 최종 checkpoint다. [전체 plan 명령·구조 변경](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.ko.md).

## 4. 결과 위치와 해석

새 결과는 `Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/convnext-last-priority8-seed3407-<시각>/`에 저장되고 실행기가 절대 경로를 출력한다. 통합 로그는 `run.log`, 리더보드는 `plans/priority8/leaderboard/leaderboard.html`(브라우저)과 `leaderboard.md`(터미널)다.

각 `runs/priority8/<case>__seed3407/`에 설정·학습 로그·checkpoint가 저장된다. `test/metrics.json`은 **21개 지표**를 포함한다: Sα(`S`), weighted Fβ(`wF`), MAE, adaptive/mean/max E·Fβ, IoU, Dice/F1, 경계 지표 등이다. 곡선·이미지별 지표도 유지하고 `profile.json`에 장비·조건과 함께 연산량·추론 비용을 기록한다. [평가 상세](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/EVALUATION.ko.md).

Backbone과 feature mode가 다르면 리더보드 비교 그룹을 분리한다. 우선 같은 ConvNeXt/마지막 stage 구조의 8개를 비교한다. 과거 PVT/multistage와는 encoder와 decoder가 함께 달라졌으므로 점수 차이를 백본이나 pseudo-scale 한 요소의 효과로 단정하지 않는다. 실제 입력 스케일의 근사와 비용 절감을 주장하려면 동일 조건의 다중 입력 스케일 참고군이 추가로 필요하며, seed 하나로 통계적 유의성을 확정할 수 없다.

업로드된 실제 학습 결과는 아직 [2026-11-02 PVT full4040 검토](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md)다. 당시 3/8 완료, 부분 실행 1개, 실행 이력 없는 4개였으며 ConvNeXt 결과나 현재 서버 상태를 뜻하지 않는다. [전체 서버 ZIP](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102). 새 CNN 검사는 초기화·추론·gradient 연결·실행 절차 검증이며 이번 코드 수정에서 학습을 시작하지 않았다. [2026-11-04 변경·검증 기록](Progress/2026-11-04/ConvNeXt-Priority8/README.md).

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
