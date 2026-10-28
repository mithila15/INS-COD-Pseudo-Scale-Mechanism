# 2026-10-28 · Pseudo-Scale Generation — 생성 방식과 논문 근거

이 문서는 단일 인코더 특징에서 세 branch를 만드는 비교 구현의 정의다. 아래 논문의 **연산 아이디어를 차용한 독립 구현**이며 원 논문의 전체 모델·성능 재현은 아니다. 문헌과 공식 코드의 확인일은 **2026-10-28**이다. 이 파일에 적힌 한국어 논문 제목은 이해를 위한 번역이며 저자의 공식 번역 제목을 의미하지 않는다.

## 공통 계약 / Common interface

`build_generator(name, channels, **kwargs)`의 출력은 `nn.Module`이며 `num_branches=3`, `forward(x) -> [main, large_or_local, small_or_context]`이다. 모든 출력은 입력과 같은 `B,C,H,W`이고 첫 출력은 입력 `x` 자체다. 출력 순서는 기존 FSA의 `[large, main, small]`에서 변경했으므로 혼용하지 않는다. 각 모듈의 `metadata`는 JSON으로 저장할 수 있다.

`R(F,s)`는 특징을 `max(1, round(H*s)), max(1, round(W*s))`로 bilinear 보간하는 연산이다. Python `round`를 사용한다. `A`는 입력의 `H,W`로 정렬하는 bilinear 보간이며 모두 `align_corners=False`이다. 정수 반올림 때문에 작은 특징에서 nominal scale과 실제 공간 배율은 다를 수 있다. 실험에서 원래 크기와 실제 branch 크기를 함께 고려해야 한다.

```math
F\in\mathbb{R}^{B\times C\times H\times W},\qquad
G(F)=[F,A(G_l(F)),A(G_s(F))].
```

`B`는 batch 크기, `C`는 채널 수, `H,W`는 공간 크기다. `G_l,G_s`는 아래 표의 두 비주 branch 생성 연산이다. 이 식은 비교 코드의 공통 인터페이스를 설명하는 **보충식**이다. receptive field가 다른 branch를 실제 입력 해상도 변경으로 얻은 feature와 같다고 보장하지 않는다. 특히 `E(resize(image,s))`와 `resize(E(image),s)`는 일반적으로 같지 않다.

새 learned projection은 channel LayerNorm을 사용해 batch 1과 1×1 공간 입력을 처리한다. LayerNorm은 위치마다 채널을 정규화한다. `C=1`에서는 정규화된 값이 상수가 되므로 API edge-case로만 지원하며 연구 기본 채널 수를 대체하지 않는다. 기존 동작을 보존하는 `legacy`만 BatchNorm을 사용한다.

## 비교 목록 / Registry

| 이름 | English description | 한국어 설명 | 기본 설정 |
|---|---|---|---|
| `identity` | Repeated feature control | 동일 특징 세 번: scale 생성 없는 대조군 | `[F,F,F]` |
| `projection` | Same-resolution learned branch control | 같은 해상도에서 독립 projection 두 개; 추가 용량 대조군 | `native_conv`와 projection 구조·파라미터 수 동일 |
| `legacy` | Original FSA pseudo-scale operations | 큰 branch는 원래 특징에 projection; 작은 branch는 평균 pool→원래 크기→projection | 원본 PS 연산 순서, BN |
| `resample` | Parameter-free resampling round trip | 보간으로 native 크기를 만든 뒤 바로 재정렬 | 배율 `1.5,0.5` |
| `native_conv` | Projection at each native resolution | native 크기로 보간→독립 projection→재정렬 | 배율 `1.5,0.5` |
| `shared_native` | Shared projection across native resolutions | 두 native 해상도에서 같은 projection 사용 | 배율 `1.5,0.5`; weight 공유 |
| `ffp_gain` | Learned power-law channel gain | FFP에서 착안한 심층 특징 채널별 scale gain | 두 branch에서 같은 exponent; 0 초기화 |
| `sfp` | Simple feature-pyramid adaptation | 학습 deconvolution과 max-pooling 후 각각 neck | native `2.0,0.5` |
| `ppm` | Region-pooling context pyramid | 서로 다른 구역 수로 평균 pool→1×1 projection→재정렬 | bins `2,4`; 출력은 finer `4` 먼저 |
| `aspp` | Atrous receptive-field pyramid | 병렬 팽창 합성곱으로 수용 영역 변경 | rates `1,3` |
| `carafe` | Content-aware kernel reassembly | 내용으로 예측한 국소 kernel로 2배 업샘플링 | kernel `5`, compression `64` |
| `dysample` | Learned grouped point sampling | 학습 offset으로 sampling 위치 이동 | groups 최대 `4`, static scope `0.25` |

`projection`, `native_conv`, `shared_native`의 projection은 `depthwise 3×3→channel LN→GELU→1×1→channel LN→GELU`다. shared 모듈은 weight를 공유하므로 파라미터 수가 독립 branch보다 작다. legacy projection은 원본대로 `depthwise 3×3→BN→ReLU→1×1→BN→ReLU→1×1→BN→ReLU`다. 따라서 legacy와 새 projection의 차이에는 normalization/activation도 포함된다.

`legacy`는 [보존된 원본 PS](../../../FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py)의 수학적 연산을 재현한다. 단, training mode에서 `B*H*W=1`이면 원본 BN은 오류를 내므로 이번 구현은 **그 경우만 running statistics를 사용하는 fallback**을 둔다. 일반 training 크기와 eval의 연산은 동일하다. 원본 model의 다른 decoder·gate까지 재현한다는 뜻은 아니다.

## 1. Fast Feature Pyramids

**English:** *Fast Feature Pyramids for Object Detection*<br>
**한국어:** *객체 검출을 위한 빠른 특징 피라미드*<br>
**게재처:** IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**), **2014**, 저널, 36(8):1532–1545.<br>
**공식 근거:** [DOI 10.1109/TPAMI.2014.2300479](https://doi.org/10.1109/TPAMI.2014.2300479), [저자 공개 PDF](https://pdollar.github.io/files/papers/DollarPAMI14pyramids.pdf), §5.1 Eq.(7), §5.2. 확인일 2026-10-28.

**EN:** The paper approximates low-level feature channels at nearby image scales by spatial resampling and a scale-dependent power-law gain. Our version learns a deep-feature exponent instead of reproducing the original empirical calibration.

**KO:** 원문은 저수준 특징 채널의 scale 반응을 경험적으로 추정하고 인접 scale의 특징을 보간과 power-law 진폭 보정으로 근사한다. 원문의 Eq.(7)을 표기만 바꿔 옮기면 다음과 같다.

```math
C_s\approx R(C,s)s^{-\lambda_\Omega}.
```

여기서 `C`는 특징 채널 영상, `s`는 scale 비율, `lambda_Omega`는 해당 특징 유형의 scale exponent다. 원문은 실제 계산한 sparse scale 사이를 근사하며 자연 영상의 저수준 특징 통계를 분석한다. 임의의 CNN 특징에서 동일 정확도를 보장하는 결과가 아니다.

비교 코드 `ffp_gain`은 `R(F,s) * exp(-lambda_c*log(s))` 후 정렬한다. `lambda_c`는 채널별로 학습되며 두 scale에서 공유하고 0으로 초기화한다. 초기 출력은 `resample`과 같다. **학습하는 심층 특징 exponent는 이번 adaptation**이며 원문의 통계적 추정 재현으로 부르지 않는다. 뒤의 강한 정규화가 진폭 차이를 지울 수 있고 exponent에 별도 clipping을 하지 않으므로 실제 실행에서 비유한 값 여부를 점검해야 한다.

입력이 float16/bfloat16이면 log·exponential은 float32로 계산하고 gain을 입력 dtype으로 돌려 세 branch의 dtype을 맞춘다. float64 입력은 float64 계산을 유지한다. scale 인자는 유한한 양수만 허용한다.

## 2. ViTDet Simple Feature Pyramid

**English:** *Exploring Plain Vision Transformer Backbones for Object Detection*<br>
**한국어:** *객체 검출을 위한 비계층적 비전 트랜스포머 백본 탐구*<br>
**발표처:** European Conference on Computer Vision (**ECCV**), **2022**, 학회.<br>
**공식 근거:** [Springer proceedings](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_17), [ECVA 논문](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690276.pdf), §3, Fig.2. [저자 SimpleFeaturePyramid 코드](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py). 확인일 2026-10-28.

**EN:** ViTDet creates a feature pyramid from one final ViT feature map. We retain the twofold transposed-convolution and half-resolution max-pooling ideas, then align their outputs for controlled fusion experiments.

**KO:** 원문은 마지막 ViT 특징 하나에서 convolution/deconvolution으로 여러 해상도를 만들며, detector가 각 해상도를 사용한다. 공식 코드는 4배/2배 deconvolution, 1배 identity, 0.5배 max-pool 뒤 각각 1×1+LN 및 3×3+LN neck을 둔다.

이번 `sfp`는 원래 `F`를 보존하고 `2×2 stride-2 ConvTranspose(C→max(1,C/2))`와 반해상도 max-pool branch만 만든다. 각 native branch에서 1×1+LN→3×3+LN neck을 적용한 뒤 `H,W`로 정렬한다. **ViT backbone, 네 scale detector, 원문 학습 설정은 재현하지 않는다.** 홀수 크기나 축 길이 1에서도 동작하도록 small branch는 floor-half 크기의 adaptive max-pool을 사용한다. 홀수 크기에서는 원문의 고정 2×2 stride-2 max-pool과 pooling 구역이 다르다.

## 3. PSPNet Pyramid Pooling

**English:** *Pyramid Scene Parsing Network*<br>
**한국어:** *피라미드 장면 파싱 네트워크*<br>
**발표처:** IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**), **2017**, 학회.<br>
**공식 근거:** [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html), [논문 PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf), §3.2, Fig.3. [저자 PyTorch PPM](https://github.com/hszhao/semseg/blob/master/model/pspnet.py). 확인일 2026-10-28.

**EN:** Region pooling summarizes context at several grid granularities. The controlled implementation uses two pooled branches while preserving the original input as the main branch.

**KO:** 원문 PPM은 bins `(1,2,3,6)`에서 average pooling, 축소 채널 1×1 convolution, BN/ReLU 후 다시 원래 크기로 보간하고 입력과 concatenate한다. 비교 코드는 bins `(2,4)` 두 개를 사용하고 채널 수를 `C`로 유지하며 LN/GELU로 바꾼다. 출력 순서상 finer bin 4가 local branch, coarser bin 2가 context branch다. 입력보다 큰 bin은 각 축 길이로 제한한다.

다음은 **비교 코드의 보충식**이다. `b`는 구역 개수이며 이미지 확대 배율이 아니다. `P_b`는 adaptive average pooling, `Q_b`는 1×1+LN+GELU projection이다.

```math
G_b(F)=A(Q_b(P_b(F))).
```

전역 bin, 원문의 채널 축소, 고정 concatenation을 제거한 것은 생성과 활용의 두 축을 분리하기 위한 adaptation이다. pooling이 작은 물체·경계 정보를 제거할 수 있어 평균화 branch가 자동으로 더 좋은 scale을 의미하지 않는다.

## 4. DeepLabv3 ASPP

**English:** *Rethinking Atrous Convolution for Semantic Image Segmentation*<br>
**한국어:** *의미론적 영상 분할을 위한 팽창 합성곱의 재고찰*<br>
**공개처:** **arXiv:1706.05587**, **2017**, 프리프린트. 확인한 공개본은 v3이며 정식 학회·저널 게재처는 **미확인**으로 남긴다.<br>
**공식 근거:** [arXiv](https://arxiv.org/abs/1706.05587), [논문 PDF](https://arxiv.org/pdf/1706.05587), §3.3, Fig.5. [저자 TensorFlow 코드](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py). 확인일 2026-10-28.

**EN:** Parallel atrous convolutions capture multiple receptive fields without rescaling the spatial grid. We compare two dilation branches rather than reproducing the complete ASPP module.

**KO:** 원문 improved ASPP는 output stride 16에서 1×1 convolution, rates `(6,12,18)`의 3×3 convolution, image-level average pooling branch를 구성하고 concatenate+1×1 projection을 적용한다. 비교 코드는 **rates `(1,3)`의 3×3 convolution 두 개**, LN/GELU와 원래 입력만 사용하며 fusion은 별도 축에 맡긴다.

이는 공간 해상도 변화가 아닌 **수용 영역 문맥 피라미드**다. 큰 dilation이 작은 feature에 적용되면 대부분 zero padding을 읽어 중심 weight 외의 정보가 줄어든다는 문제가 원문에도 설명되어 있다. 구현은 dilation을 자동 축소하지 않으므로 stage 크기와 rate를 기록하고 해석해야 한다. 원문의 결과나 다른 output stride 설정을 이 코드의 결과로 대체해서 보고하지 않는다.

## 5. CARAFE

**English:** *CARAFE: Content-Aware ReAssembly of FEatures*<br>
**한국어:** *CARAFE: 내용 인지적 특징 재조립*<br>
**발표처:** IEEE/CVF International Conference on Computer Vision (**ICCV**), **2019**, 학회.<br>
**공식 근거:** [CVF proceedings](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html), [논문 PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.pdf), §3.1–3.3, Eq.(1)–(3). [공식 MMCV 연산](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/carafe.py). 확인일 2026-10-28.

**EN:** CARAFE predicts a normalized local reassembly kernel for each target position. Our pure PyTorch operation uses low-resolution unfold and a weighted contraction instead of the official fused CUDA implementation.

**KO:** 1×1 convolution으로 채널을 압축하고 3×3 convolution으로 각 low-resolution 위치에 대응하는 2×2 subpixel 각각의 5×5 kernel logits를 만든다. 각 kernel의 25개 원소에 softmax를 적용한 뒤 입력의 해당 5×5 이웃을 채널별로 가중합한다. kernel은 채널 사이에서 공유한다. 다음은 원문 Eq.(3)을 좌표 표현만 바꾼 식이다.

```math
F'_{c,p}=\sum_{q\in\mathcal{N}(p)}w_{p,q}F_{c,q},
\qquad \sum_qw_{p,q}=1.
```

`p`는 출력 위치, `q`는 대응하는 입력 이웃, `c`는 채널, `w`는 내용에서 예측한 weight다. local patch 밖은 zero padding이다. 따라서 경계에서 상수 입력 보존을 가정하지 않는다.

비교 코드는 low-resolution `unfold`와 `einsum` 후 pixel shuffle로 업샘플링하고 다시 입력 크기로 정렬한다. small branch는 별도의 평균 pooling이다. **원문은 업샘플러이며 이런 전체 scale bank와 재정렬은 이번 adaptation**이다. unfold workspace는 원소 수 기준 `B*C*k*k*H*W`다. 고해상도 전체 patch 복제를 피했어도 메모리 사용량이 크며 공식 CUDA 구현의 속도·메모리 성능을 주장할 수 없다.

내부 kernel-logit의 subpixel/이웃 채널 순서는 공식 구현과 다르다. 수학적 연산을 독립 구현한 것이므로 **공식 CARAFE 가중치를 그대로 복사해 사용할 수 없다**. 이를 위한 채널 재배열·checkpoint 변환기는 제공하지 않는다.

## 6. DySample

**English:** *Learning to Upsample by Learning to Sample*<br>
**한국어:** *샘플링을 학습하여 업샘플링하기*<br>
**발표처:** IEEE/CVF International Conference on Computer Vision (**ICCV**), **2023**, 학회.<br>
**공식 근거:** [CVF proceedings](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html), [논문 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.pdf), [저자 코드](https://github.com/tiny-smart/dysample/blob/main/dysample.py). 확인일 2026-10-28.

**EN:** DySample predicts sampling offsets, then samples the input with bilinear grid sampling. The comparison uses the LP-style static-scope construction with a grouped twofold upsampling branch.

**KO:** 1×1 convolution이 low-resolution 위치마다 grouped offset을 예측한다. pixel shuffle로 2배 grid의 offset을 배치하고 기본 bilinear grid에 더한 뒤 `grid_sample`로 입력을 읽는다. offset weight는 표준편차 `0.001`, bias 0으로 초기화하고 static scope 계수 `0.25`를 곱한다. `align_corners=False`, `padding_mode=border`다. 다음은 구현 설명을 위한 보충식이다.

```math
G(F)=\operatorname{GridSample}(F,\,g_{\rm bilinear}+0.25\,\Delta(F)).
```

`g_bilinear`는 정규 2배 보간의 sampling 좌표, `Delta(F)`는 예측 offset이다. 실제 구현은 입력 픽셀 단위 offset을 `[-1,1]` 좌표로 변환한다. groups는 요청값 이하에서 `C`를 나누는 최대 정수로 선택한다. 기본값 4가 채널 수를 나누면 4를 사용한다. small branch의 평균 pooling 및 최종 원래 크기 정렬, divisor-safe grouping은 이번 adaptation이며 원문의 전체 network 비교가 아니다. offset을 모두 0으로 놓으면 native upsampler가 표준 bilinear 2배 보간과 일치해야 한다.

내부 offset의 group/좌표 채널 순서는 공식 코드와 다르며 **공식 DySample 가중치의 직접 복사는 지원하지 않는다**. 연산의 의미를 유지하는 weight 재배열은 별도 작업이며 이 저장소에 해당 checkpoint 변환기는 없다.

## 해석과 코드 출처

FFP 외 후보는 multi-scale 입력 encoder의 근사 오차를 검증한 방법으로 묶지 않는다. ViTDet는 학습 해상도 피라미드, PSPNet/ASPP는 문맥 피라미드, CARAFE/DySample은 업샘플링 연산이다. 확대 후 원래 크기로 줄인 feature가 더 세밀한 입력 관측을 얻었다고 표현하지 않는다.

생성 효과와 활용 효과는 분리한다. native resolution에서 convolution을 적용한 뒤 정렬하는 경우와 정렬한 뒤 convolution을 적용하는 경우는 같지 않다. 독립 projection 뒤 성분별 뺄셈에는 channel 의미가 정렬됐다는 보장이 없다. identity·같은 해상도 projection·단순 resample 대조군과 shared projection 설정이 이 문제를 구분하도록 마련되어 있다. 방법 간 파라미터 수와 계산량이 다르므로 같은 encoder라는 사실만으로 같은 자원 예산이라고 결론 내리지 않는다.

이 파일이 참고한 코드의 라이선스는 Detectron2와 MMCV가 Apache-2.0, 저자 `hszhao/semseg`와 `tiny-smart/dysample`이 MIT다. 이번 파일은 논문 연산을 설명하기 위해 이들을 검토했으며 외부 소스 파일을 복사·vendor하지 않았다. 링크: [Detectron2 license](https://github.com/facebookresearch/detectron2/blob/main/LICENSE), [MMCV license](https://github.com/open-mmlab/mmcv/blob/main/LICENSE), [semseg license](https://github.com/hszhao/semseg/blob/master/LICENSE), [DySample license](https://github.com/tiny-smart/dysample/blob/main/LICENSE).

검증은 무작위 텐서의 shape, 유한값, autograd 연결, 알려진 연산적 성질 및 기존 legacy eval 동치 확인이다. 이는 **학습 완료·정책 업데이트·COD 정확도 향상 검증이 아니다**. 연산 단위 검증은 [test_generators.py](../tests/test_generators.py)에 있다.
