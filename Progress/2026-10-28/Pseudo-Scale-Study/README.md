# 2026-10-28 · Pseudo-Scale Study — 기존 논리와 생성·활용 비교 결정

## 질문과 사용자 관측

동일한 단일 인코더에서 pseudo-scale/residual 모듈을 뺐을 때 더 좋았다는 사용자 관측에서 시작했다. 정확도·latency 수치와 학습 로그는 제공되지 않았으므로 어느 지표에서 얼마나 우세했는지는 미확인이다. 사용자는 pseudo-scale을 유지하면서 **생성 방식**과 **생성된 특징의 활용 방식**을 다양하게 비교하기로 했다. 활용은 뺄셈뿐 아니라 fusion도 포함한다.

## 기존 코드에서 확인한 사실

원본은 [fsa_layers.py](../../../Experiments/FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py)와 [fsa_zoomnext.py](../../../Experiments/FSA-ZoomNeXt/methods/zoomnext/fsa_zoomnext.py)다.

1. 인코더는 main RGB 한 번만 처리한다. 각 stage에서 projection한 F를 pseudo 생성기에 넣는다.
2. large는 실제 확대 없이 독립 learned projection(F)이다. small은 절반 pooling → 원래 크기로 보간 → 별도 projection이다. 작은 native 해상도에서 convolution하는 구현이 아니다.
3. `dl=L-M`, `ds=S-M`이고 `Y=out_proj(M+gl*dl+gs*ds)`다. 두 sigmoid gate의 합이 1로 제한되지 않아 M의 유효 계수 `1-gl-gs`가 음수가 될 수 있다.
4. residual 특징은 `Psi([abs(dl),abs(ds),M])`다. M이 입력에 포함되므로 dl=ds=0이어도 residual이 0일 필요가 없다.
5. residual은 edge prior, cross-level compensation, region refinement의 residual 재주입으로 이어진다. 비용은 pseudo 생성 두 분기만의 비용이 아니다.
6. 실제 확대/축소 이미지의 encoder 특징을 reference로 사용하지 않는다. optional KD도 logits 수준이고 기본 weight는 0이다. 따라서 dl/ds는 **학습된 분기 특징의 차이**이며 검증된 scale approximation error가 아니다.
7. `use_compensation=False`나 `residual_refine_weight=0`만으로 모든 pseudo/residual 연산이 제거되지 않는다. 기존 case02/case06은 width와 gate도 달라 순수 모듈 ablation으로 해석하기 어렵다.

## 해석과 새 가설

서로 다른 projection을 거친 채널을 바로 빼면 scale 변화와 representation 변화가 섞일 수 있다. 이는 가능한 설명이지, 현재 성능 저하의 실증된 원인은 아니다. 단일 특징에서 만든 추가 분기가 유용한 정보를 학습하는지, 기존 정보를 재표현하면서 비용만 늘리는지 분리해 확인해야 한다.

FFP는 다른 scale 특징을 근사하는 근거에 가깝지만 원래 gradient/color 같은 저수준 특징의 통계적 보정이다. 이를 깊은 CNN 특징에 학습 가능한 channel exponent로 적용하는 것은 새 가설이다. PSPNet/ASPP는 문맥 범위 변형이며 실제 scale-feature 근사와 동일시하지 않는다. CARAFE/DySample은 upsampling 방법이고 그대로 COD 성능 개선을 보장하지 않는다.

## 결정한 실험 설계

- [새 코드와 영어·한국어 참고 논문](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.md): 생성12 × 활용13 =156조합, no-pseudo 및 전체 FSA 참고군 추가158구성.
- 공통 encoder/projection/decoder/loss와 입력 크기384·width48·동일 seed를 고정한다. 공통 edge head는 모든 구성에서 fused 특징을 사용한다. 차이·fusion 선택과 downstream residual 재주입을 혼합하지 않는다.
- `projection` 대 `native_conv`는 같은 파라미터 수로 연산 위치를 비교한다. `single` 대 `identity+mean`은 수치 동치 대조다. 나머지 방법 간 파라미터·연산량은 다를 수 있으므로 측정한다.
- `signed` 대 `adjacent`는 부호 재매개변수화로 동일 함수군이며 최적화/초기화 통제로 기록한다.
- 생성만 비교할 때 활용 고정, 활용만 비교할 때 생성 고정, 최종 전체 교차 비교를 모두 명령 생성기로 지원한다. 선택은 validation에서 하고 test는 별도 보고한다.
- CAMO+COD10K의 원 학습4,040장에서 고정404장을 validation으로 분리한다. 기본 학습3,636장이라 원 논문의 전체 학습 protocol과 직접 비교하지 않는다. COD10K의 camouflage prefix만 선택하고 제외 목록·해시를 남긴다.
- S-measure, weighted F, E-measure와 MAE는 공식 PySODMetrics 구현으로 평가한다. seed별 결과와 표본 표준편차를 기록한다. 추론 latency, 메모리, 파라미터는 별도로 측정한다.

## 보존·변경·현재 상태

원본 FSA와 나머지 이관 파일1,077개는 보존한다. 새 파일은 날짜별 Experiments와 본 Progress에만 추가하며, 이관 검사기는 원본 프로젝트들만 manifest 대상으로 검사하도록 범위를 수정했다. 과거 검증 보고서는 덮어쓰지 않는다.

[이번 검증 결과](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-28/README.md)는 실제 Git 추적 대상이며 이 문서와 실험 README에서 연결한다. 원본 가중치를 매핑한 연산 동치, 합성 입력 전후 shape와 유한값, gradient 연결, 단일 encoder 호출, 공통 초기화, 데이터 split 및 strict checkpoint 검증을 수행했다. 이때 optimizer step은 0이다. 합성 inference profile은 학습·COD 성능 측정과 별도로 표시한다.

실제 데이터셋 다운로드, pretrained weight 다운로드, 학습·재학습은 실행하지 않았다. 현재 특정 생성/fusion 조합을 최종 권장할 근거는 없다. 다음 단계는 사용자가 로컬 데이터·가중치 경로를 지정하여 manifest를 준비하고, 동일 조건의 학습을 시작하는 것이다. 실행 명령은 [실험 실행 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md)에 있다.
