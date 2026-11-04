# 2026-11-04 · CNN 검증 — 구조·Priority8·기존 모델 호환 확인

**학습 결과가 아닌 코드·초기화 모델 검증 자료**다. [English](README.md) | 한국어.

- 전체 **780 tests + subtest 74개 통과**, 13개 deprecation warning, 20.48초. Synthetic forward/gradient 연결, strict checkpoint, checkpoint 재구성·optimizer 그룹을 확인했다. Optimizer step은 **0회**다.
- 과거 Git 소스의 PVT baseline(659 tensors), native_conv+SK(723), legacy reference(1,283)와 state key·seed 초기 텐서·출력 텐서가 정확하게 같았다. 서버의 실제 학습 checkpoint를 다시 로딩한 검사는 아니다.
- ConvNeXt-Tiny / last / native_conv+SK, **384×384·채널48 CUDA 추론**을 로컬 RTX 5080에서 확인했다. Complexity 미지원 연산은 없었다. 입력·가중치는 무작위이며 warmup2회 뒤 측정5회는 실행 진단으로만 기록한다. 실험 서버의 학습 모델 속도·정확도 결과로 해석하지 않는다.
- Launcher `--prepare-only`가 **8개·seed3407** 명령만 생성하고 실제 run 폴더를 만들지 않는 것을 확인했다. 임시 가중치 파일은 명령 생성용 placeholder로 모델에 로딩하지 않았다.

[검증 JSON·소스 해시](verification.json) · [계획 요약](launcher-plan-check.json) · [무작위 초기화 profile](convnext-last-random-profile.json).

데이터·가중치 다운로드, 학습·optimizer update는 실행하지 않았다. 공식 CNN 사전학습 가중치 실파일·서버 데이터 접근·수렴은 아직 확인하지 않았다. 기존 PVT 결과는 보존했다.

기존 로컬 `fsa-study` overlay 환경(torch2.11.0+cu130 / torchvision0.26.0+cu130)을 사용했으며 fresh 설치 검증은 아니다. [환경 출처](../../environment/README.ko.md). 재현 추론 명령은 [영문 기록](README.md), 실제 8개 학습 명령은 [CNN 실행 안내](../../docs/CNN-BACKBONES.ko.md)에 있다.
