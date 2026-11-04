# 2026-11-04 · Supervision Audit — Pseudo-scale 특징과 Pseudo-label 구분

[English](PSEUDO-LABELS.md) | **한국어** · [CNN 실험](CNN-BACKBONES.ko.md) · [실험 안내](../README.ko.md)

**현재 구현에는 모델 예측으로 pseudo-label을 생성해 학습하는 과정이 없다.** 데이터셋이 제공한 mask를 정답으로 사용하는 지도 분할 학습이다. 여기서 *pseudo-scale*의 pseudo는 생성된 **특징 표현**을 가리키며 생성된 label을 뜻하지 않는다. 과거 PVT 실험과 이번 CNN adapter 모두 같은 구분을 적용한다.

## 무엇을 생성하고 어디에 사용하는가

| 항목 | 출처 | 코드에서의 사용 |
| --- | --- | --- |
| 분할 mask | 검증한 manifest에 지정된 데이터셋 annotation | 최종 mask, deep head, region-prior head의 정답 |
| Edge 정답 | 학습 mask에 5×5 morphological gradient 적용 | edge-prior head의 정답 |
| Pseudo-scale 특징 | 인코더 특징에 학습·공간 변환 적용 | fusion/utilizer에 전달한 뒤 디코더에서 사용 |
| Region/edge 예측 | 모델의 보조 head | 정답으로 지도하는 보조 출력. region 예측은 decoder gate에도 사용 |
| UAL 항 | 최종 예측 확률 | confidence 정규화. 이산 label이나 teacher target을 만들지 않음 |
| 선택적 feature target | 학습 이미지에 적용한 고정 image-pyramid teacher의 최종 epoch 특징 | 보조 분기 cosine loss. 예측 mask 정답이 아님 |
| 저장된 예측 PNG | test 시점 모델 추론 | 평가 산출물이며 학습 입력으로 사용하지 않음 |

코드 근거: [데이터와 정답 생성](../fsa_study/data.py), [loss와 학습](../fsa_study/engine.py), [특징 생성기](../fsa_study/generators.py), [특징 활용 모듈](../fsa_study/utilizers.py), [모델 head](../fsa_study/model.py).

## 현재 정답 준비 과정

1. 같은 manifest record의 이미지와 mask를 읽는다. 학습 이미지와 mask의 크기는 일치해야 한다.
2. deterministic sampling으로 정한 좌우·상하 flip을 이미지와 mask에 함께 적용한다. RGB는 bilinear, mask는 nearest-neighbor로 학습 크기에 맞춘다.
3. `mask > 0` 조건으로 이진 정답을 만든다.
4. resize된 이진 mask에서 max pooling으로 5×5 dilation과 erosion을 구한다. Edge 정답은 `dilation(mask) - erosion(mask) > 0`이다. padding은 기존 구현의 규칙을 따르며 이미지 바깥을 별도 label로 추가하지 않는다.

Edge 정답은 이미 존재하는 annotation에서 결정적으로 파생된다. 이를 teacher가 생성한 pseudo-label이라고 부르면 코드와 다른 설명이 된다. 지도 신호의 출처는 데이터셋 mask이며 그 파일 hash는 manifest에 보존된다.

## 현재 loss

다음은 이번 연구 코드의 실제 loss 정의다. 새 loss를 백본 논문의 기여로 귀속하는 설명이 아니다.

| 로그 항목 | 정답과 계산 | 적용 계수 |
| --- | --- | --- |
| `seg` | mask에 대한 경계 가중 binary cross-entropy와 weighted IoU | 1 |
| `deep` | 존재하는 deep head 각각의 같은 mask loss를 평균 | 0.25 |
| `region` | region logits `po`의 같은 mask loss | 0.5 |
| `edge` | edge logits `pe`와 파생 edge 정답의 Dice loss | 1 |
| `distill` (선택) | 대응하는 고정 실제 scale 보조 특징과 cosine distance | 별도 distill launcher에서 0.1 |
| `ual` | `1 - (2 * sigmoid(logits) - 1)^2`의 평균에 학습 진행률에 따른 cosine 계수 적용 | 0에서 1로 증가 |

지도 loss 안에서는 logits를 정답 크기로 bilinear 정렬한다. `last`에는 deep head가 1개이고 기존 common multistage에는 4개다. Head 평균에 0.25를 적용하므로 총 deep-loss 계수는 유지된다. UAL은 확실한 예측을 유도하지만 저장된 target을 만들거나 threshold로 pseudo mask를 생성하지 않는다. 따라서 UAL 사용만으로 teacher–student 학습이 되는 것은 아니다.

## Pseudo-scale 생성은 별도 과정

priority8의 `native_conv`에서는 projection한 인코더 특징 하나를 원래 main branch로 유지하고, 나머지 두 branch를 독립적으로 변환한다. 특징을 1.5배 또는 0.5배 크기로 바꾼 뒤 해당 해상도에서 연산하고 main 격자로 다시 정렬한다. 선택한 utilizer는 이 특징들을 결합해 분할 예측에 사용한다. 어떤 branch도 mask label로 취급하지 않는다. Core와 기존 priority8에는 teacher matching이 없으며, 별도 distill 단계에서 resize RGB teacher 특징과의 cosine 감독을 추가한다.

원래 동기와 adaptation의 범위는 [생성 방식](GENERATION.md), [활용 방식](UTILIZATION.md)에 보존한다. *근사*라는 표현은 연구 목표를 설명한다. 실제 multi-input-scale 특징과 얼마나 일치하는지는 reference를 통해 측정해야 한다.

## 나중에 pseudo-labeling을 추가한다면

Pseudo-labeling은 **보류된 별도 실험**이다. ConvNeXt 후보 목록을 실행하는 데 필요하지 않으며, 이 브랜치에는 선택적 frozen feature teacher가 있지만 pseudo-mask cache는 없다. Teacher target·출처·고정 상태·학생 단독 추론은 [mechanism 안내](SCALE-MECHANISM.ko.md)에 명시했다. 추후 구현한다면 학습 전에 다음을 명시해야 한다.

- 확률을 생성하는 teacher checkpoint와 학습 이력, label 생성 대상 이미지. Test 이미지·정답이 학습 지도 신호로 들어가서는 안 된다.
- Soft probability와 hard mask 중 무엇을 사용할지, confidence·threshold·ignore 규칙, 기하학적 변환 후 정렬 방법.
- Teacher 고정·EMA 갱신·주기적 재생성 중 어떤 방식을 쓸지, 고정 teacher로 gradient가 흐르지 않게 하는 방법.
- 생성 target, 원본 이미지 ID, teacher/checkpoint hash, 생성 설정의 기록 위치.
- 기존 지도 loss 대비 pseudo-label loss의 가중치와, 효과를 확인할 no-pseudo-label 대조군.

위 항목은 향후 설계 조건이며 이미 구현된 기능을 설명하는 것이 아니다. 현재 branch fusion이나 학습된 region gate는 pseudo-label 생성 과정에 해당하지 않는다.
