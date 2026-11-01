# 2026-11-01 · COD10K 중복 검사 — 기존 동일 split records 보존

[English](README.md) | **한국어** | [실행 안내](../../docs/BENCHMARK-PROTOCOL.ko.md#설치-성공-후-준비-단계에서-중단되는-경우)

사용자 서버에서 `8f08f4ff6b61` 소스 설치는 성공했지만 준비 단계에서 `Unacknowledged image overlap: unknown hash or a group with more than two records`로 중단됐다. 새 평가 plan이 생성되지 않아 뒤이어 `run_evaluation.sh`와 로그 디렉터리가 없다는 오류가 나온 것이다. 이 시도에서 평가나 새 학습은 시작되지 않았다.

## 재현한 문제와 수정

중복 인정 정책이 기존 test split 내부의 동일 해시 그룹도 정확한 학습 원본/test 쌍을 위한 목록에 대조하고 있었다. 따라서 원래 엄격 정책이 허용한 test 내부 그룹을 인정 정책에서 거부할 수 있었다. 같은 split의 동일 경로 records도 엄격 정책과 달리 거부하는 문제가 있었다. 합성 records로 이 회귀를 재현했으며 사용자 서버의 실제 오류 그룹을 직접 확인한 것은 아니다.

중복 metadata 계산, 전체 manifest 검사와 snapshot 검사에서 동일 split에 대한 원래 동작을 복구했다. 데이터셋 구성·sample 순서·해시·geometry 처리·모델·지표·7쌍의 근거 목록은 바꾸지 않는다.

| 경우 | 수정 후 동작 |
| --- | --- |
| 기존 records에서 하나의 split 안에만 있는 동일 바이트·경로 | 두 정책 모두 records 보존 |
| 서로 다른 split에서 같은 실제 이미지 경로 사용 | 두 정책 모두 거부 |
| 확인된 정확한 COD10K 학습 원본/test 쌍 | 기본값에서는 거부, 명시적 인정 때만 허용하고 기록 |
| 미확인 교차 split 해시, 다른 ID/source/stem, train/val 중복, 서로 다른 split에 걸쳐 3개 이상인 동일 해시 그룹 | 거부 |
| 테스트 추가 과정에서 새 test record가 원본 이미지 경로나 인정되지 않은 원본 이미지 해시를 재사용 | 기존 추가 검사에서 계속 거부 |

audit는 계속 **서로 다른 split 사이**의 동일성을 표시한다. 같은 split 내부 이미지의 고유성이나 테스트 데이터셋 사이의 독립성을 인증하지 않는다. 기존 동일 split records를 보존하는 것은 복구 도중 평가 조건을 바꾸지 않기 위한 것이며 중복 records가 집계 지표에 영향이 없다는 주장이 아니다.

## 검증과 한계

**747개 테스트와 74개 subtest가 통과했다.** [Pytest 출력](pytest.txt), [JUnit 기록](pytest.xml), [검증 범위·소스 해시](verification.json)에 검사를 기록한다. 동일 split records 처리, 교차 split 거부 유지, 명시적 인정과 준비 흐름을 검사한다. 합성 코드 검사이며 새 성능 측정은 아니다. [앞선 7쌍의 근거](../verification-2026-11-01-overlap/README.ko.md)와 [기존 3개 데이터셋 실험 보고서](../priority8-report-20261101_172449/README.ko.md)는 그대로 보존한다.

이번 수정에서 데이터셋·가중치 다운로드, optimizer 업데이트, 실제 학습 또는 서버 평가를 실행하지 않았다. 실행 전 수정본 설치와 사용자 실제 manifest의 준비 성공을 확인해야 한다. 이후에도 미확인 교차 split 중복이 나오면 해당 records를 확인해야 하며, 재학습이나 검사 우회로 준비 문제가 해결되지는 않는다.

[연구 판단 기록](../../../../../Progress/2026-11-01/COD10K-Same-Split-Fix/README.md) · [실험 안내](../../README.ko.md)
