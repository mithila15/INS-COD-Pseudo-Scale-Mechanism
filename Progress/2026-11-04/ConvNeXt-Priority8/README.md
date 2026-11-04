# 2026-11-04 · CNN Pseudo-Scale Study — ConvNeXt 마지막 Stage와 기존 Priority8 재비교

## 질문과 결정

기존 PVTv2의 여러 stage가 이미 서로 다른 해상도를 제공하므로 pseudo-scale의 필요성과 효과가 섞이는지 확인한다. ConvNeXt도 내부는 계층적이므로 CNN 교체만으로 문제를 해결했다고 보지 않는다. **ConvNeXt-Tiny 마지막 stage 하나만 decoder에 전달**하는 모드를 구현했다. ResNet50 Layer4는 후속 비교용으로 지원하며 지금 실행 목록에는 넣지 않는다.

사용자의 “기존에 몇 개만 먼저 진행하던 케이스”는 마지막 완료 모델 하나가 아니라 **priority8**이다. 원래 순서인 `single`, `native_conv__asff`, `native_conv__sk`, `native_conv__mean`, `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean`을 **seed 3407 하나로 총 8회** 실행하도록 확정했다. 이번 작업에서는 학습을 시작하지 않았다.

## 전후 조건과 구현

| 항목 | 기존 전체 학습 실험 | 이번 준비 |
| --- | --- | --- |
| Encoder | PVTv2-B2 한 번 | ConvNeXt-Tiny 한 번 |
| Decoder 입력 | 4개 encoder stage | 최종 stage 하나, stride 32 |
| Decoder | ASPP와 다중 kernel region-guided refinement | 단일 특징 region gate + 순차 2배 보간/3×3 DSConv, 이전 stage skip 없음 |
| Deep supervision | 4개 head 평균 | 최종 특징의 1개 head; 같은 총 loss 계수 |
| 학습 데이터/선택 | 4,040장, final epoch 고정 | 같은 manifest, 같은 선택 규칙 |
| 실행 | priority8, seed 3407 | 같은 목록·seed, 새로운 결과 폴더 |
| 설정 | 150 epoch, batch4, 384², channels48 | 유지 |

공통 학습률·손실 가중치·encoder BN 고정 정책을 유지하고, PVT 첫 patch embedding 고정을 CNN 첫 stem 고정으로 대응시켰다. 모델 구성에 backbone과 feature mode를 저장해 평가·profiling 때 정확하게 복원한다. 서로 다른 조건의 리더보드 순위를 섞지 않는다. 기존 PVT 기본 모드의 state dict와 초기 출력은 유지한다.

Pseudo-scale은 feature 생성이며 pseudo-label이 아니다. 실제 코드에는 teacher나 pseudo-label 생성 단계가 없고, GT mask·GT 파생 경계·예측 확신도 정규화로 학습한다. [영문 설명](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/PSEUDO-LABELS.md) · [한국어 설명](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/PSEUDO-LABELS.ko.md).

## 검증과 남은 한계

전체 테스트 **780개 + subtest 74개 통과**. CPU CNN forward/gradient 연결, 엄격한 synthetic checkpoint 로딩, optimizer 그룹/고정 stem 구성, checkpoint 재구성, 기존 PVT 호환, 정확한 8-job 계획과 리더보드 비교 그룹을 확인했다. Backward는 연결성 검사이며 optimizer step은 0회다. [검증 산출물](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-04-cnn/README.md).

실제 사전학습 CNN 파일·서버 원본 데이터·학습 수렴은 이 컴퓨터에서 검증하지 않았다. PVT/multistage → ConvNeXt/last는 encoder와 decoder 경로가 함께 바뀌므로 기존 점수 차이를 한 요인의 효과로 해석하지 않는다. 기존 COD10K 중복 7쌍과 CHAMELEON geometry 예외는 그대로 남는다. 실제 입력 스케일 feature의 근사 여부·정확도 대비 비용 절감은 동일 조건의 다중 입력 스케일 참고군을 나중에 추가해야 판단할 수 있다.

## 다음 실행

서버 소스를 갱신하고 matching torchvision ConvNeXt-Tiny 가중치를 준비한 뒤 아래를 직접 실행한다. [가중치 준비·환경·상세 명령](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.ko.md). 데이터와 manifest만 재사용하고 기존 PVT checkpoint는 재사용하지 않는다. 학습 → 4개 테스트셋 평가 → profiling이며 결과는 새 `results/convnext-last-priority8-seed3407-<시각>/`, 리더보드는 그 안의 `plans/priority8/leaderboard/`다.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
WEIGHTS="$STUDY/weights/convnext_tiny-983f1562.pth"
bash "$STUDY/launch_convnext_priority8.sh" "$WEIGHTS"
```

[당일 목차](../README.md) · [연구 기록](../../README.md)
