# 2026-11-01 · COD10K 중복 검사 — 근거와 명시적 벤치마크 규약

[English](README.md) | **한국어** | [실행 안내](../../docs/BENCHMARK-PROTOCOL.ko.md#확인된-cod10k-중복을-명시적으로-기록하고-진행하기)

사용자가 제공한 서버 검사에서 **COD10K 학습 원본/test의 바이트 중복 7쌍**이 확인됐다. 학습 원본의 6장은 최적화 train, 1장은 validation에 배정돼 있었다. 테스트 이미지 2,026장은 모두 대응 GT가 있었고, 7쌍 중 동일한 실제 경로를 공유하는 경우는 없었다. 코드 설치 후 엄격 검사에서 중단된 이유이며 데이터셋이나 기존 모델을 삭제해야 한다는 근거는 아니다.

## 근거 산출물

- [server-audit-summary.json](server-audit-summary.json): 사용자 터미널 출력에서 원 manifest 해시·분할 개수·정확한 sample ID·이미지 해시를 추린 기록이다. 전체 터미널 내용과 개인 로컬 경로는 옮기지 않았다. 사용자 서버 보고이며 로컬에서 데이터셋을 독립 검사한 결과와 구분한다.
- [upstream-evidence.json](upstream-evidence.json): 독립적인 COD-TDQ 배포 자료의 고정 commit `c4e97371fa835cd0a778ba67c071f24ca5deaf11`에서 7쌍을 대조했다. 14개 원본 URL·Git-LFS 포인터 텍스트·SHA256·파일 크기를 보존했다. 모두 사용자 이미지 해시와 같다. 작은 metadata 텍스트만 읽었으며 이미지·GT·데이터셋·모델 가중치는 다운로드하지 않았다.

[2021-10-09 SINet-V2 공식 공지](https://github.com/GewelsJI/SINet-V2/blob/5b9844c94d10a03b815c7c7039c194166a9a8ba3/README.md#L44)는 Crab-32와 Cat-1506을 명시한다. 독립 미러에서 7쌍 모두를 확인했지만 나머지 5쌍까지 COD10K 원저자가 공식 인정했다는 뜻은 아니다. 공지의 2장에 대한 대략적인 지표 영향으로 이번 7장이나 이 실험의 영향을 추정하지 않는다.

| Test ID 끝부분 | 학습 원본 ID 끝부분 | 기존 split | 공식 공지에 해당 쌍 명시? |
| --- | --- | --- | --- |
| Crab-32 | Crab-32 | train | 예 |
| Cat-1506 | Cat-1506 | train | 예 |
| Deer-1762 | Deer-1796 | train | 아니오 |
| Gecko-1892 | Chameleon-1694 | train | 아니오 |
| Gecko-1895 | Gecko-1928 | train | 아니오 |
| Giraffe-1932 | Giraffe-1930 | val | 아니오 |
| Owl-4633 | Bird-3205 | train | 아니오 |

## 처리 규약과 해석

기본값은 엄격한 거부다. 준비 명령의 명시적 `--acknowledge-cod10k-overlap` 옵션은 목록에 기록된 정확한 COD10K ID·해시 쌍만 허용한다. 실제 경로 재사용·알려지지 않은 중복·나머지 무결성 문제는 계속 오류다. 새 manifest에는 `overlap_policy="acknowledged-cod10k-v1"`, `disjoint=false`인 `overlap_audit`가 들어간다. 평가·보고서에도 중복을 표시하며 표준 개수나 벤치마크 완료가 독립성을 뜻하지 않는다.

기존 holdout은 train/test 6쌍·val/test 1쌍이다. 전체 학습 준비는 명시한 규약을 이어받고 validation을 train으로 이동한 뒤 train/test 7쌍·val/test 0쌍으로 다시 계산한다. 학습 원본 4,040장·COD10K test 2,026장을 보존하며 벤치마크를 조용히 바꾸지 않는다. 기존 체크포인트와 원래 3개 테스트셋 산출물도 유지한다. 인정 표시는 과거 최적화·validation 선택에 사용된 사실을 되돌리지 않는다.

이번 COD10K 중복만으로 기존 CAMO·CHAMELEON·NC4K 점수가 무효가 되지는 않는다. 바이트 검사로 재인코딩·시각적 유사 이미지까지 모두 검증한 것도 아니다. 중복을 제외한 test 2,019장 또는 새 train 4,033장 조건은 표준 4,040/2,026과 구분해 별도 실험으로 다뤄야 하며 이번 복구가 수행하는 동작이 아니다.

## 검증 범위

**코드 검증 736개 테스트와 74개 subtest를 통과했다.** [pytest 출력](pytest.txt), [JUnit 기록](pytest.xml), [소스 해시·범위](verification.json), [브라우저 검사 5개](browser-verification.json)를 보존한다. 엄격 거부, 정확한 7쌍 인정, 기존 학습 자료 보존, 전체 학습 audit 승격, 보고서 무결성, 21개 지표와 백업 설치를 검사했다. 합성 입력만 사용했으며 optimizer 업데이트와 실제 학습은 실행하지 않았다.

[패키지 검증](archive-verification.json)은 staged 소스의 Git archive를 새 임시 작업 공간에 설치하고 CLI 4개와 포함된 backbone의 64×64 CPU 추론을 확인했다. Python/PowerShell 소스 해시는 [verification.json](verification.json)과 같다. 검증 ZIP은 이 최종 문서 추가 전 snapshot이며 서버 설치를 실행한 것은 아니다.

[문서 검사 기록](documentation-checks.json)은 로컬 링크 대상, 실행하지 않은 Bash 블록의 문법, 7쌍의 근거 해시 일치를 확인한다.

고정된 14개 Git-LFS metadata와 7쌍의 이미지 해시가 제공된 로그와 일치하는지 확인했다. 이미지/GT의 의미적 정합 확인, 서버 평가, optimizer 업데이트, 새 학습은 하지 않았다. 코드 회귀 검증과 외부 자료 대조는 별도 상태이며 여기서는 새 성능이나 시간 측정을 주장하지 않는다. [기존 3개 데이터셋 priority8 보고서](../priority8-report-20261101_172449/README.ko.md)는 그대로 유지한다.

[연구 판단 기록](../../../../../Progress/2026-11-01/COD10K-Overlap/README.md) · [실험 안내](../../README.ko.md)
