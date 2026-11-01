# 2026-11-01 · COD10K 준비 오류 — 동일 split 중복 판정 회귀 수정

## 질문과 관측

사용자는 `8f08f4ff6b61` 설치 후에도 `Unacknowledged image overlap: unknown hash or a group with more than two records`로 준비가 중단되자 검사를 무시하고 학습을 다시 시작해도 되는지 물었다. 설치 기록은 `installed`였지만 평가 plan 생성은 실패했다. 뒤이어 나온 `run_evaluation.sh`·로그 디렉터리 부재는 준비 실패의 결과이며, 리더보드 경로를 출력한 것은 결과 파일 생성의 근거가 아니다. 해당 시도에서 평가와 새 학습은 실행되지 않았다.

## 원인 분석과 결정

코드 검사와 합성 재현에서 인정 정책이 기존 test split 내부의 동일 해시 그룹을 학습 원본/test 7쌍의 허용 목록과 대조하는 회귀를 확인했다. 같은 split의 경로 공유도 원래 엄격 정책보다 강하게 거부했다. 사용자 서버에서 실제 오류를 낸 그룹은 직접 확인하지 않았으므로 이번 오류 전체를 새로운 학습/test 누출 또는 무해한 test 내부 중복으로 단정하지 않는다.

중복을 전부 무시하거나 기존 모델을 삭제하는 대신 원래 동일 split 동작을 복구한다. `overlap.py`, `data.py`, `benchmark.py`의 검사를 일치시키며 기존 동일 split records는 보존한다. 알려진 7쌍의 ID·이미지 SHA256·명시적 인정·audit 규약은 유지한다. 미확인 교차 split 해시, 교차 split 실제 경로 공유, train/val 중복, 교차 split 그룹의 세 번째 record는 계속 거부한다. 새 COD10K test가 원본 경로 또는 인정되지 않은 원본 해시를 재사용하는 추가 검사도 유지한다.

이는 학습 알고리즘·평가 지표·데이터셋 구성을 바꾸는 수정이 아니다. 기존 manifest·체크포인트·학습 로그·평가 결과와 [7쌍의 앞선 근거](../COD10K-Overlap/README.md)를 보존한다. audit는 split 간 동일성 검사이며 같은 split 내부의 고유성이나 테스트 데이터셋 간 독립성을 주장하지 않는다.

## 검증 범위와 다음 단계

[영어 검증 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-same-split/README.md)와 [한국어 검증 안내](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-same-split/README.ko.md)에 새 회귀 검사와 한계를 기록한다. 합성 코드 검사와 사용자 서버 실험은 구분한다. 이번 수정으로 데이터셋·가중치 다운로드, optimizer 업데이트, 실제 학습 또는 서버 평가를 실행하지 않았다.

다음 단계는 수정 소스 전송·설치 → 새 경로에 평가/전체 학습 plan 준비 → 준비 성공 확인 → 사용자가 선택한 평가 또는 학습 실행이다. 기존 체크포인트의 재평가는 재학습 없이 가능하며 새 4,040장 학습 준비도 기존 모델 평가 완료를 기다릴 필요가 없다. 재학습 자체로 준비 코드 오류가 해결되지는 않는다. 수정 후 미확인 교차 split 중복이 남으면 실제 records를 추가 확인한다. [서버 경로·준비·학습 명령](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md).

[당일 목차](../README.md) · [Progress 목차](../../README.md)
