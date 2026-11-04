# 2026-11-04 · Pseudo-Scale Mechanism — 별도 브랜치에서 본질적 효과 비교

질문은 “같은 단일 encoder에서 pseudo-scale을 생성·활용하는 것이 추가 분기 용량을 넘어 도움이 되는가?”다. 사용자가 별도 브랜치의 코드와 영문·한글 실행 안내를 요청했다. **`research/pseudo-scale-mechanism`**을 기존 main `1a003b9a7becddc0fd844bb85de62f005b96a9f0`에서 만들었다. main 통합은 하지 않는다.

## 결정과 변경 이유

먼저 ConvNeXt-Tiny 마지막 stage를 쓰는 네 가지를 비교한다: `single`, 같은 해상도 학습 분기 `projection__mean`, 특징을 resize한 뒤 동일 연산을 하는 `native_conv__mean`, 실제 RGB를 세 번 인코딩하는 `image_pyramid__mean`. 모두 seed 3407이며 150 epoch·full4040 최종 checkpoint 고정이다. 기존 priority8와 전체 catalog는 보존하지만 새 launcher는 네 가지부터 실행한다.

이후 변수별로 따로 실행한다. `resolution`은 네 가지를 stride 16으로 반복한다. 최종 stage를 제거하지 않고 마지막 downsampler stride/padding만 바꿔 384 입력에서 최종 특징을 12×12에서 24×24로 유지한다. 파라미터와 decoder block 수는 같지만 sampling·수용영역·경계 동작은 달라진다. `fusion`은 projection/native 두 모델에 원본 특징을 직접 더하는 작은 채널 gate를 적용한다. `distill`은 학습 완료된 실제 scale 참고군을 고정 teacher로 사용해 같은 두 학생의 보조 특징에 cosine loss를 추가한다. 후속 단계는 자동 실행하지 않는다.

Teacher 입력은 같은 증강된 학습 이미지이며 test 예측을 학습 대상으로 가져오지 않는다. 선택은 최종 epoch 고정이다. 그러나 기존 COD10K train/test 중복 7쌍은 그대로 있어 데이터 독립성을 새로 주장할 수 없다. Feature target을 pseudo-mask label과 구분했고, teacher 파라미터·gradient·추론 비용이 학생에 섞이지 않도록 구현했다. Distillation은 teacher 선행 학습과 학습 중 encoder 3회 추가가 필요하다.

## 근거와 검증

[검증 자료](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-04-mechanism/README.ko.md)에 전체 **813개 + 하위 74개**, 이전 기본 동작 4개 비교, 384 입력 GPU 추론 10개, 두 stride의 teacher/student gradient 검사를 연결했다. 초기 profiling에서 padding 미지원 문제가 있어 데이터 이동으로 분류하고 회귀 검사를 추가했다. Optimizer step·데이터 다운로드·실제 학습은 수행하지 않았다.

[소스·영문 실행 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md)와 [한국어 실행 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md)에 환경·가중치·Windows branch 전송·서버 별도 설치·단계별 명령·결과 위치·참고 논문을 적었다. 문서와 검증 JSON/로그/재현 코드를 이 브랜치에 함께 게시한다. 기존 결과·checkpoint·학습 데이터는 덮어쓰지 않는다.

## 현재 판단과 다음 단계

실험의 첫 판단은 native mean이 baseline뿐 아니라 projection mean도 이기는지, 그 이득과 비용이 실제 RGB scale 참고군에 얼마나 접근하는지다. 같은 해상도 분기 대비 차이가 없으면 scale 고유 효과를 뒷받침하지 못한다. 실제 RGB scale 참고군도 이득이 없으면 특징·decoder·scale 범위를 먼저 검토한다.

Stride·fusion·teacher는 각각 대응 대조군으로 해석하며 통합 우승을 자동 선언하지 않는다. Teacher loss 감소만으로 semantic scale equivalence를 주장하지 않는다. 한 seed는 탐색이므로 최종 주장에는 반복 seed가 필요하다. 실제 서버의 batch 4·384 학습 메모리·속도와 성능은 미검증이다. Windows 전송은 로컬 실행할 수 없어 코드 검토와 ZIP 설치 fixture까지만 확인했다.

새 코드 위치는 `/home/master/mithila/shMin-scale-study`, 새 결과는 기존 `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/scale-mechanism-*`이다. 다음 실제 동작은 사용자가 안내서의 `core` 명령으로 네 실험을 시작하는 것이다. [당일 목록](../README.md).
