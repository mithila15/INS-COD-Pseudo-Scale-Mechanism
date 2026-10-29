# 2026-10-29 · Pseudo-scale 연구 — 평가 확장과 자동 리더보드

사용자는 현재 평가 항목이 논문용으로 충분한지 확인한 뒤 부족한 지표를 모두 추가하고, 학습 종료 시 어떤 실험이 잘 되었는지 리더보드로 보여주며 GitHub에 올리도록 요청했다.

기존 테스트 출력은 S·wF·MAE·E(adaptive/mean/max)의 6개 값이었다. 이번에는 일반 Fβ(adaptive/mean/max), 이진 영역·confusion 지표와 Boundary IoU를 더해 21개 필드를 출력한다. Dice/F1은 같은 값이므로 독립적인 지표 개선으로 해석하지 않는다. 데이터셋 평균 곡선과 이미지별 최댓값을 구분하고 원본 크기 확률 PNG, 이미지별 CSV, PR/Fβ/E 곡선 CSV·SVG·PNG와 해시를 저장한다. 기존 6개 값은 공식 구현과 동일함을 검사했다.

Boundary IoU는 [원논문](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)의 저자 erosion 규약을 적용한다. 기존 생성·fusion 참고 논문 11편은 보존하고 새 평가 출처를 영어·한국어 README에 추가했다. [평가 정의](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/EVALUATION.ko.md)에 임계값·정규화·빈 마스크·macro 평균·출처를 명시했다.

추론 비용에는 MACs와 2×MACs 규약의 FLOPs, 보간·샘플링 등 별도 추정량, 연산별 커버리지를 추가했다. 전체 추정치는 하드웨어 명령어 수가 아니며, 미지원 연산이 있으면 null로 남긴다. 시간 측정과 별도 forward에서 계산하여 카운터가 latency를 오염시키지 않는다.

`run_all.sh`는 직접 실행했을 때 run별 학습→평가→profiling을 수행하며 단계마다 HTML·Markdown·CSV·JSON 리더보드를 갱신한다. `train.sh`는 학습과 validation 리더보드만 수행한다. 전체 선택 기준은 validation MAE, 테스트는 데이터셋·지표별 보고 순위, 비용은 장비·조건별 표로 둔다. 같은 프로토콜과 같은 완료 seed 집합만 비교하며 미완료·미시작 상태를 표시한다. 원시 결과는 보존하고 파생 리더보드 6개만 갱신한다.

검토 중 예전 6개 지표를 새 평가 완료로 잘못 간주하는 문제와 수정된 plan의 action/출력 경로 불일치를 발견하여 검증을 추가했다. 오래된 지표·profiling 조건, 사라진 artifact, 다른 체크포인트는 조용히 건너뛰지 않는다. 기존 결과 재평가는 새 평가·profile 경로를 사용한다. 중단된 optimizer 학습 재개 기능은 추가하지 않았다.

[검증 산출물](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-29-metrics/README.md)에 522 tests + 67 subtests, 전체 158개 초기화 모델 연산 커버리지, 대표 CPU/CUDA profile, 브라우저 8개 상호작용 검사와 재현 스크립트를 보존한다. 코드·문서·해당 산출물을 함께 업로드한다. 과거 2026-10-28 검증 기록은 변경하지 않았다.

현재 확인한 것은 합성 입력 지표·초기화 모델 추론·평가 CLI·집계/표시다. 실제 데이터 다운로드와 학습은 실행하지 않았다. 실제 정확도·수렴·최종 우승 조합은 아직 판단할 수 없다. 다음 단계는 사용자가 데이터를 정해진 구조로 준비하고 동일 manifest·backbone으로 계획을 생성한 뒤 학습을 시작하는 것이다. [영어 실행 안내](../../../README.md) · [한국어 실행 안내](../../../README.ko.md).
