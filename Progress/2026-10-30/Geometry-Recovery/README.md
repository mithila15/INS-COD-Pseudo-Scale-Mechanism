# 2026-10-30 · Pseudo-scale 연구 — CHAMELEON 평가 오류와 학습 결과 재사용

사용자가 전달한 서버 로그에서 `priority8-seed3407-20261030_114816`의 첫 `single` 실험은 150 epoch를 완료했고 최고 validation MAE는 0.028608166038820583이었다. 그다음 테스트 평가에서 `chameleon_test/animal-62` 이미지 1000×628과 정답 마스크 998×628의 크기 불일치가 발생했다. 실행기는 평가 실패로 중단되어 이후 7개 실험을 시작하지 않았다. 이는 전달받은 로그의 관측이며 로컬에서 서버 모델의 정확도를 재측정한 결과가 아니다.

원인은 `CODDataset`이 모든 split에 동일 크기를 강제한 것이다. 기존 evaluator는 이미 예측을 원본 GT 크기로 bilinear 보간하므로, 학습 데이터의 크기 검사가 테스트용 평가 경로까지 차단했다. [SINet 원본 평가](https://github.com/DengPingFan/SINet/blob/master/MyTest.py#L24-L45)도 GT의 크기를 기준으로 예측을 보간한다. 이 규약은 서로 다른 해상도를 처리하지만 원본 이미지와 정답의 공간적 정합을 증명하지는 않는다.

수정은 명시적인 테스트 옵션 `--test-geometry resize-prediction-to-gt`로 제한했다. 기본값과 train/val은 `strict`로 유지한다. 모델·최적화·학습 증강·validation 선택·150 epoch 설정은 바꾸지 않았다. 평가 시작 전 선택된 전체 test pair를 읽어 실제 데이터셋 수와 크기 불일치를 기록하며, 정답·이미지·manifest·checkpoint를 수정하거나 샘플을 제외하지 않는다. `geometry_audit.json`과 그 해시를 지표 결과에 연결한다. strict와 opt-in 평가 결과는 같은 프로토콜로 합산하지 않는다.

새 `recover_evaluation.py`는 기존 plan을 검증하고 완료된 학습은 재사용하며, 실패한 평가 폴더와 겹치지 않는 새 경로를 가진 plan을 생성한다. 계획 생성은 학습·추론을 실행하지 않는다. 사용자가 생성된 `run_all.sh`를 실행하면 완료된 첫 모델의 학습을 건너뛰고 평가→profiling→나머지 7개 학습/평가로 진행한다. 중단된 optimizer 학습 자체의 재개 기능은 추가하지 않았다.

과거 모델의 소스 해시를 덮어쓰지 않는다. 따라서 이전 코드로 학습한 baseline과 수정 후 학습한 모델이 리더보드의 별도 source-protocol cohort에 표시될 수 있다. 이는 자동 비교의 한계로 남기며 해시를 지워 하나의 순위로 만들지 않는다. `manifest_no_cod10k_test.json`이라는 이름만으로 데이터셋 구성을 단정하지 않고 복구 사전 검사에서 실제 test records를 출력한다.

[검증 기록과 로그](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-30-geometry/README.md)는 보고된 크기를 재현한 합성 입력, 추론 전용 지표 생성, 이전 checkpoint schema 호환성과 복구 순서 검사다. 실제 서버의 이미지/GT 정합, 학습 모델의 테스트 점수와 CUDA 장시간 실행은 아직 확인하지 않았다. 데이터 다운로드와 학습은 자동 실행하지 않았다.

사용자가 서버에서 코드 갱신 후 실행할 명령은 [한국어 실험 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.ko.md)와 [영어 실험 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.md)에 있다. 코드·문서·검증 로그를 동일 변경으로 GitHub에 게시한다.
