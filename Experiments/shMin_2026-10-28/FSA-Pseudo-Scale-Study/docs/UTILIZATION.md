# 2026-10-28 · Pseudo-Scale Utilization — 차이와 융합 비교 / Difference and Fusion Ablations

이 문서는 [utilizers.py](../fsa_study/utilizers.py)의 **생성 이후 활용 방식**을 설명한다. 모든 방법은 같은 생성기가 만든 특징을 받아 하나의 융합 특징만 반환한다. 별도의 residual/edge head를 내부에서 생성하지 않는다. 성능 우열은 아직 검증되지 않았으며, 아래 설명은 구현과 논문 근거에 관한 것이다.

**English.** This document describes how the comparison suite uses already generated pseudo-scale features. Each utilizer returns one fused feature map for the shared decoder. The implementations are controlled adaptations of selected operators, not reproductions of complete published networks. No accuracy improvement is claimed before training and evaluation.

## 공통 계약 / Shared contract

```python
module = build_utilizer(name, channels=C, num_branches=3, **method_options)
fused = module([M, L, S])  # each input and output: B,C,H,W
```

- `M`: 생성기 입력의 main/anchor 특징. bank의 첫 번째 항목이다.
- `L`: large 또는 local 역할의 보조 특징. 이름이 실제 확대 입력을 encoder에 통과시켰다는 뜻은 아니다.
- `S`: small 또는 context 역할의 보조 특징.
- 세 입력은 크기·채널 수·dtype·device가 같아야 한다. 이 모듈은 입력 resize를 몰래 수행하지 않는다.
- `P`: 학습 가능한 **bias 없는 1×1 convolution**. `cat`은 채널 방향 연결, `GAP`는 공간 평균, `sum_scale`은 bank 방향 합이다.
- `forward`의 출력은 `B,C,H,W` tensor 하나다. 융합 뒤에 강제로 `fused-M`을 만들지 않는다.
- 각 인스턴스의 `.metadata`는 JSON으로 저장할 수 있으며 citation, adaptation, normalization과 설정을 포함한다. 인식하지 못한 설정 키는 무시하지 않고 오류를 낸다.

**English.** The bank order is `[main, large/local, small/context]`. Spatial and channel alignment belongs to the generator. All utilization variants share the downstream decoder and supervision. A fusion variant is not followed by a mandatory subtraction or an auxiliary residual decoder.

`mean`, `concat`, `signed`, `absolute`, `relative`, `normalized`, `bifpn`, `sk`, `asff`, `mhsiu`는 생성기가 제공하는 다른 branch 수에도 대응한다. 차이 방식은 두 개 이상의 branch가 필요하다. `adjacent`, `legacy_gated`, `aff`는 정확히 `[M,L,S]` 세 입력을 요구한다. 본 실험의 기본 비교는 세 입력을 고정한다.

## 논문과 적용 범위 / References and adaptation scope

확인일 / checked: **2026-10-28**. 한국어 제목은 설명용 번역이며 공식 번역 제목이 아니다. arXiv 공개 연도와 정식 발표 연도를 구분한다.

| 방법 | English title / 한국어 제목 | 정식 발표처·분류 / venue and status | 근거 |
| --- | --- | --- | --- |
| `sk` | **Selective Kernel Networks** / **선택적 커널 네트워크** | IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2019**, 학회 / conference) | [공식 논문 페이지](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html), [읽은 PDF §3.1 Eq.(1)–(6)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf), [공식 코드](https://github.com/implus/SKNet) |
| `asff` | **Learning Spatial Fusion for Single-Shot Object Detection** / **단일 단계 객체 탐지를 위한 공간 융합 학습** | **arXiv:1911.09516, 2019**, 프리프린트 / preprint. 정식 학회·저널 게재처는 이번 조사에서 미확인 | [arXiv v2 기록](https://arxiv.org/abs/1911.09516v2), [수식 확인용 v1 PDF §3.2 Eq.(1)–(2)](https://arxiv.org/pdf/1911.09516v1), [공식 코드](https://github.com/GOATmessi8/ASFF) |
| `bifpn` | **EfficientDet: Scalable and Efficient Object Detection** / **EfficientDet: 확장 가능하고 효율적인 객체 탐지** | IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2020**, 학회 / conference) | [공식 논문 페이지](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html), [읽은 PDF §3.3](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf), [공식 코드](https://github.com/google/automl/tree/master/efficientdet) |
| `aff` | **Attentional Feature Fusion** / **어텐션 기반 특징 융합** | IEEE/CVF Winter Conference on Applications of Computer Vision (**WACV 2021**, 학회 / conference) | [공식 논문 페이지](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html), [읽은 PDF §3.2, §4.1 Eq.(4)](https://openaccess.thecvf.com/content/WACV2021/papers/Dai_Attentional_Feature_Fusion_WACV_2021_paper.pdf), [공식 코드](https://github.com/YimianDai/open-aff) |
| `mhsiu` | **ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection** / **ZoomNeXt: 위장 객체 탐지를 위한 통합 협력 피라미드 네트워크** | IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI 2024**, 저널 / journal) | [정식 게재 DOI](https://doi.org/10.1109/TPAMI.2024.3417329), [읽은 공개본 arXiv v4 §3.3 Eq.(1)](https://arxiv.org/html/2310.20208v4), [공식 코드](https://github.com/lartpang/ZoomNeXt) |

검색 기록: 위 다섯 논문을 대상으로 Exa 검색 5회, 각 `numResults=5`, 반환 항목 검토 수 `sources_reviewed=25`(중복 포함). 논문 원문과 저자 저장소를 추가로 읽어 수식 및 구현 차이를 확인했다. 문헌 전체에 대한 체계적 문헌고찰을 뜻하지 않는다.

## 기준선 / Basic controls

| 이름 | 계산 / computation | 해석 / interpretation |
| --- | --- | --- |
| `mean` | `Y=(M+L+S)/3` | 고정 균등 융합 / fixed equal weighting |
| `concat` | `Y=P(cat(M,L,S))` | 학습 가능한 선형 채널 혼합 / learned linear channel mixing |

`concat`에는 추가 activation이나 main skip을 넣지 않는다. 생성기의 효과를 해석할 때 별도의 no-pseudo main-only 기준선과 identity-triplet 생성기를 함께 비교한다. 같은 입력을 세 번 넣었을 때 `mean`은 입력을 보존하지만, `concat`은 자유로운 학습 projection이므로 입력 보존이 보장되지 않는다.

## 차이 기반 활용 / Difference-based utilization

다섯 방법 모두 공통 식을 사용한다. 아래는 구현을 설명하는 표기이며, 특정 논문의 원문 식으로 주장하지 않는다.

```text
Y = M + activation(P(cat(dL,dS)))
P: bias-free Conv1x1(2C -> C)
activation: identity (default), ReLU, or tanh
```

| 이름 | `dL` | `dS` | English description |
| --- | --- | --- | --- |
| `signed` | `L-M` | `S-M` | Signed differences against the anchor |
| `absolute` | `abs(L-M)` | `abs(S-M)` | Magnitude without directional sign |
| `relative` | `(L-M)/(abs(L)+abs(M)+eps)` | `(S-M)/(abs(S)+abs(M)+eps)` | Elementwise amplitude-normalized signed differences |
| `normalized` | `norm_C(L)-norm_C(M)` | `norm_C(S)-norm_C(M)` | Differences of pixelwise channel L2-normalized vectors |
| `adjacent` | `L-M` | `M-S` | Differences with a consistently oriented large-to-small path |

`eps` 기본값은 `1e-6`이다. `norm_C(X)`는 각 `(B,H,W)` 위치에서 C차원 벡터를 그 L2 norm과 `eps` 중 큰 값으로 나눈 결과다. 공간 전체나 배치 전체의 표준화가 아니다. half/bfloat16 입력에서는 정규화 계산을 float32로 수행한 뒤 원래 dtype으로 돌린다. `relative`는 분자·분모를 `max(abs(aux),abs(main),eps)`로 나누는 동치식을 써 큰 값의 덧셈 overflow와 영점 주변의 분모 문제를 줄인다.

정규화된 차이는 원래 `M`과 크기 단위가 다를 수 있다. 공통 학습 projection `P`가 이를 anchor 특징의 좌표와 크기에 매핑한다. 이것이 최적 매핑이라는 보장은 없으며 학습으로 평가할 가설이다. `P`에 bias가 없고 세 activation 모두 0을 0으로 보내므로 **동일 입력 세 개에서는 모든 차이 방법이 정확히 M을 반환**한다. residual projection 입력에 raw M을 별도로 연결하지 않는다.

**English.** These are diagnostic ablations, not formulas claimed by the cited fusion papers. The anchor is retained, and a bias-free learned projection maps the differences into anchor feature units. Identical branches produce zero correction, including after training. Pixelwise channel L2 normalization removes vector magnitude, whereas relative normalization is applied independently to every element.

해석 시 주의할 대수적 성질:

```text
(L-M) + (M-S) = L-S          # M cancels
(L-M) + (S-M) = L+S-2M      # second-difference-like expression
```

구현은 이 둘을 먼저 합하지 않고 연결한 뒤 학습 projection을 적용한다. 단순 합에서 일어나는 고정적인 상쇄를 강요하지 않는다. 일반적인 bias-free 1×1 projection에서는 signed와 adjacent가 두 번째 입력 부호를 projection weight에 흡수할 수 있어 **표현 가능한 함수군은 같다**. 따라서 이 두 비교는 모델 표현력보다 initialization·최적화의 부호 관례를 살피는 통제 실험으로 해석해야 한다.

## 원래 FSA의 융합 경로 / Original FSA fused branch

`legacy_gated`는 [원래 ScaleResidualIntegration](../../../FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py)의 다음 경로를 유지한다.

```text
dL = L-M; dS = S-M
[gL,gS] = sigmoid(Conv1x1(ReLU(BN(Conv1x1(cat(M,dL,dS))))))
Y = out_proj(M + gL*dL + gS*dS)
out_proj = DWConv3x3 -> BN -> ReLU -> PWConv1x1 -> BN -> ReLU
hidden = max(8, C // gate_reduction), gate_reduction=2 by default
```

두 sigmoid는 독립적이다. main의 유효 계수는 `1-gL-gS`이므로 음수가 될 수 있으며 convex fusion이 아니다. 원래 BN을 그대로 두므로 train mode에서 `B*H*W=1`인 입력은 지원하지 않는다. `out_proj` 때문에 `L=M=S`라도 출력이 M과 같다는 조건은 적용되지 않는다.

원본의 별도 `residual_proj(cat(abs(dL),abs(dS),M))`는 생성도 계산도 하지 않는다. raw M이 residual 경로에 들어갈 때 차이가 0이어도 별도 residual이 0이 아닐 수 있다는 기존 구조의 효과를 이 비교군에 숨겨 넣지 않는다. **전체 FSA 재현 `legacy_full`과 구분**해야 한다. 공통 decoder에 들어가는 융합 연산만 보존한 것이다.

**English.** `legacy_gated` preserves the original independent sigmoid gates and depthwise-separable output block, but excludes the unused residual head and the original multi-head decoder. Numerical equivalence of this fused branch is assessed after copying its matching weights and BN state. Removing the residual module changes random-number consumption during construction; equal random seeds alone do not create identical original weights.

## 논문 기반 융합 / Paper-inspired fusion

### BiFPN-inspired fast normalized weights

```text
w = ReLU(theta), theta initialized to ones
a = w / (sum(w) + eps), eps=1e-4
Y = sum_scale(a * [M,L,S])
```

English: **Nonnegative learned scalar importance per branch**, independent of image content. 한국어: **입력 내용과 무관하게 학습되는 scale별 비음수 스칼라 가중치**다. EPS 때문에 합은 정확히 1보다 조금 작다. 모든 `theta`가 0 이하가 되면 ReLU gradient와 출력이 모두 0이 될 수 있다. 양수 초기화는 이 상태에서 시작하지 않게 한다.

원문 §3.3의 fast normalized fusion 식만 적용한다. 반복적인 top-down/bottom-up 연결, multi-level 출력, separable output convolution을 포함하는 **전체 BiFPN을 구현한 것이 아니다**.

### SK-inspired channel selection

```text
U = M+L+S
z = ReLU(Conv1x1(GAP(U)))
a = softmax_scale(reshape(Conv1x1(z), B,3,C,1,1))
Y = sum_scale(a * [M,L,S])
```

English: **Global context chooses a scale distribution independently for each channel.** 한국어: **세 특징의 전역 문맥으로 채널별 scale 선택 비중을 정한다.** 원문 SK unit의 Fuse/Select 개념을 외부 생성기가 만든 특징에 적용한다. 원문 Split의 kernel 구조는 이 모듈이 생성하지 않는다. 원문의 pooled BN을 제외해 batch 1을 지원한다. hidden은 `max(min_hidden,C//reduction)`이며 기본값은 `min_hidden=8`, `reduction=4`로 원문 네트워크의 고정 설정을 재현하지 않는다.

### ASFF-inspired spatial selection

```text
h_i = ReLU(GN(Conv1x1(X_i)))  # one compression module per branch
a = softmax_scale(Conv1x1(cat(h_M,h_L,h_S)))
Y = a_M*M + a_L*L + a_S*S
```

English: **Every spatial position chooses a scale distribution shared by all channels.** 한국어: **위치마다 scale 비중을 선택하고 채널 전체에 같은 비중을 적용한다.** 기본 압축 채널은 16이다. 공식 ASFF 구현처럼 각 branch를 압축한 뒤 logits를 만든다. 원문 detector의 서로 다른 pyramid level을 한 level로 옮기는 연산과 마지막 output expansion은 없다. BN 대신 local GN을 사용하며, 이미 정렬된 pseudo branch 한 묶음만 융합한다.

### Pairwise AFF-inspired local/global selection

```text
A(V) = sigmoid(Local(V) + Global(GAP(V)))
Local  = Conv1x1 -> GN -> ReLU -> Conv1x1 -> GN
Global = Conv1x1 -> ReLU -> Conv1x1
AFF(X,Y) = A(X+Y)*X + (1-A(X+Y))*Y
aux = AFF_aux(L,S)
fused = AFF_main(M,aux)
```

English: **Local and global context select features in two pairwise fusions, with the anchor fused last.** 한국어: **지역·전역 문맥으로 두 보조 특징을 융합하고 main 특징을 마지막에 결합한다.** 두 gate는 다른 파라미터를 갖는다. 이 순서의 삼중 결합은 이번 연구의 adaptation이며 원문 iAFF가 아니다. local GN은 group당 두 채널 이상을 사용하고, 채널 수 1에서는 Identity를 쓴다. global path에는 BN/GN이 없어 batch 1·1×1 feature에서도 동작한다. hidden은 `max(1,C//4)`가 기본값이다.

논문 Eq.(4)의 convex blend를 따른다. [공식 PyTorch AFF 코드](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py)는 출력에 2배 gain을 곱하지만, 본 구현은 그 gain을 적용하지 않는다. 따라서 원본 코드의 완전 복제가 아니다.

### ZoomNeXt-inspired multi-head scale integration

```text
F = pack_by_head([M,L,S])
V_h = grouped_value_Conv1x1(F)_h       # three transformed values per head
a_h = softmax_scale(grouped_attention(F)_h)
Y_h = sum_scale(a_h * V_h)
Y = concat_heads(Y_h)
```

English: **Independent heads learn spatial scale-selection patterns and combine transformed branch values.** 한국어: **head마다 공간 scale 선택 패턴을 학습하고 변환된 value들을 합친다.** attention은 grouped 1×1 convolution, head별 GN, ReLU, grouped 1×1 logits로 만든다. value transform은 각 head 안에서 세 branch 채널을 함께 변환한다. 따라서 가중치가 붙는 value는 원래 M/L/S 자체와 동일하지 않다.

기본 요청 head 수는 4이고 실제 head 수는 `gcd(C,requested_heads)`다. 실제 값을 metadata에 남긴다. `C`를 head별로 나눈 뒤 각 head 안에서 branch를 묶는다. plain `cat(M,L,S)`에 group convolution을 바로 적용하는 잘못된 branch/head 배치를 피한다. head당 attention hidden은 `max(4,(C/heads)//reduction)`, reduction 기본값은 4다.

원문 ZoomNeXt는 **이미지를 0.5×/1×/1.5×로 바꿔 공유 encoder에 세 번 입력**한다. 원문 MHSIU의 정렬은 large의 max+average pooling, small의 bilinear interpolation을 포함한다. 본 구현은 이미 정렬된 단일 encoder pseudo 특징을 사용하고 간소화한 grouped transforms를 적용하므로 **원본 MHSIU 또는 ZoomNeXt의 정확한 재현이 아니다**. `L=M=S`여도 value transform 때문에 출력이 M일 필요가 없다.

## 진단과 해석 / Diagnostics and interpretation

attention 방법의 `weights(bank)`는 tensor를 반환하며 live graph를 멤버에 보관하지 않는다. 공통 shape 관례는 `B,branches,selection_channels,H,W`다.

| 방법 | weights shape | 합과 의미 |
| --- | --- | --- |
| `bifpn` | `B,N,1,1,1` | branch 합이 EPS 때문에 1보다 작음 |
| `sk` | `B,N,C,1,1` | branch 방향 합 1 |
| `asff` | `B,N,1,H,W` | branch 방향 합 1 |
| `aff` | `B,3,C,H,W` | 두 pairwise gate를 합성한 M/L/S의 유효 가중치, 합 1 |
| `mhsiu` | `B,N,heads,H,W` | head마다 branch 방향 합 1, 변환된 value에 적용 |
| `legacy_gated` | `B,3,C,H,W` | `[1-gL-gS,gL,gS]`, 합 1이나 main 계수 음수 가능. out_proj 이전 계수 |

평가·시각화용 가중치를 조회할 때는 `eval()`과 `torch.no_grad()`를 사용한다. 특히 legacy의 `weights()`는 실제 BN gate를 통과하므로 train mode에서 별도 진단 호출은 running statistics를 추가로 바꾼다. 동일 branch 입력 보존 검사는 `mean`, 차이 다섯 방식, `sk`, `asff`, `aff`에 적용한다. `bifpn`은 EPS 수축을 허용해야 하고 `concat`, `legacy_gated`, `mhsiu`에는 그 성질을 요구하지 않는다.

**Pseudo 차이는 approximation error의 정답이 아니다.** 단일 특징을 resize하거나 다른 필터로 변환한 결과의 차이는 보간, 저역통과, receptive field, projection 또는 normalization 차이일 수 있다. 실제 resized image를 encoder에 넣은 reference를 별도로 계산하고 비교해야 reference discrepancy를 측정할 수 있다. 이 utilizer는 그러한 reference나 teacher를 생성하지 않는다.

**English.** Pseudo-feature differences are representation discrepancies, not ground-truth approximation errors. Independently learned projections need not preserve channel semantics. Difference magnitude can reflect interpolation and normalization rather than useful object evidence. Compare generated feature similarity, normalized difference energy, attention distributions, parameter count, runtime, and memory alongside held-out segmentation metrics.

최종 평가는 동일 encoder·학습 데이터·증강·loss·decoder·평가 코드에서 수행한다. 전체 generator×utilizer 조합을 통해 상호작용을 확인하고, seed를 반복한다. 성능 기준선과 원래 전체 FSA 구조의 결과를 구분한다. 학습을 실행하기 전의 shape/유한값/gradient 검사는 **학습 성능 검증이 아니다**.
