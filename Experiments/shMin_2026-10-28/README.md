# 2026-11-02 · shMin 실험 — Pseudo-scale 생성·활용과 전체 벤치마크 비교

| 실험 | 내용 | 결과 |
| --- | --- | --- |
| [FSA-Pseudo-Scale-Study](FSA-Pseudo-Scale-Study/) | 12 generators × 13 utilizers, 공통 단일 인코더·디코더와 2개 대조군; priority8 screening | [과거 holdout 8개·3개 데이터셋 보고서](FSA-Pseudo-Scale-Study/results/priority8-report-20261101_172449/README.md), [새 full4040 3/8 완료 결과](FSA-Pseudo-Scale-Study/results/server-review-20261102/README.ko.md), [코드 검증](FSA-Pseudo-Scale-Study/results/verification-2026-11-01-benchmark/README.md) |

[COD10K-test 평가 복구·전체 학습 4,040장·ZoomNeXt 조건 대조](FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.ko.md)는 기존 결과를 보존하고 새 결과 경로를 사용한다. 2026-11-02 13:16 KST 서버 스냅샷에는 새 전체 학습·4개 테스트셋 평가를 끝낸 3개 실험, 15 epoch까지 실행한 1개, 실행 기록이 없는 4개가 있다. 과거 3,636/404 holdout 결과와 새 4,040장·최종 epoch 고정 결과를 분리한다. 중복 7쌍의 위치도 과거 train/test 6·validation/test 1에서 새 train/test 7·validation/test 0으로 달라진다. [검토·산출물 목록](FSA-Pseudo-Scale-Study/results/server-review-20261102/asset-index.md).

[전체 실험 목록](../README.md)
