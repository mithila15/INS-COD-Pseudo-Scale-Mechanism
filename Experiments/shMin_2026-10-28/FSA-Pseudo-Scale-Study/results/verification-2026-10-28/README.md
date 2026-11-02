# 2026-10-28 · Pseudo-Scale Verification — 연산·초기화·입력·추론 검사

**최종 281 tests + 67 subtests 통과. optimizer step은 0회이며 학습을 실행하지 않았다.** 156개 생성×활용 조합의 forward/backward 연결을 검사했으며, full model 단위 검사는 6개 대표 구성에서 단일 encoder 호출과 출력을 확인했다. 별도 전체 모델 profile은 7개 구성에서 수행했다. 원본 PS와 gated fused branch의 가중치 매핑 후 eval 출력은 원본과 bit-exact였다.

English: 281 tests and 67 subtests passed with **zero optimizer steps**. Operator pairing tests cover all 156 combinations; full-model unit tests cover six representative configurations and inference profiles cover seven configurations. This verifies execution and invariants, not convergence or COD accuracy.

## 검사 산출물

- [최종 로그](tests-final.txt), [최종 JUnit XML](tests-final.xml): 데이터 pairing/고정 split/변경 감지, 공식 지표 합성 예측, strict checkpoint, 계획 42·45·474회, 결과 집계 조건 분리, 모든 module 조합 및 RNG 검사.
- [중간 로그](tests.txt), [중간 JUnit XML](tests.xml): workflow 집계 검사를 추가하기 전 261 tests+67 subtests 통과 기록을 보존했다.
- [상태 보고서](validation.json), [검증 코드 SHA-256](source-sha256.json), [산출물 체크섬](checksums.sha256).
- [원본 이관 무결성과 문서 링크 검사](migration-verification.json): 원본 1,077개 source blob/내용/권한 보존 및 pinned COD source tree 확인. [초기 검사 기록](migration-verification-initial.json)의 pytest cache README 오인과 생성 전 보고서 링크 오류는 캐시 제외 및 보고서 생성 후 재검사로 해결했다.
- [환경 기록](../../environment/verification.json), [환경 프로필](../../environment/README.md).
- [검증 코드](../../tests/), [추론 측정 코드](../../profile_study.py).

## 합성 입력 추론 확인

CPU, FP32, batch 1, 384×384, width48, 2 threads, warmup3, repeats10, random initialization, 동일 seed의 synthetic uniform RGB다. IO와 CPU/GPU 전송은 제외하고 region/edge/deep head를 포함한 전체 forward를 측정했다. CPU의 실제 운영 부하·frequency를 격리하지 않았으므로 작은 시간 차이를 성능 순위로 해석하지 않는다. 이 숫자는 COD 정확도나 학습 시간, 학습 메모리가 아니다.

| 실행 산출물 | 전체 파라미터 | median ms |
| --- | ---: | ---: |
| [profile-carafe-aff-cpu](profile-carafe-aff-cpu.json) | 25,581,379 | 334.803 |
| [profile-dysample-sk-cpu](profile-dysample-sk-cpu.json) | 25,334,275 | 237.534 |
| [profile-legacy-full-cpu](profile-legacy-full-cpu.json) | 25,595,795 | 250.919 |
| [profile-legacy-gated-cpu](profile-legacy-gated-cpu.json) | 25,396,115 | 235.885 |
| [profile-native-asff-cpu](profile-native-asff-cpu.json) | 25,351,775 | 240.083 |
| [profile-single-cpu](profile-single-cpu.json) | 25,318,163 | 247.473 |
| [profile-single-sequential-cpu](profile-single-sequential-cpu.json) | 25,318,163 | 228.726 |

`profile-single-cpu`는 초반 테스트와 실행 시간이 겹친 예비 측정이다. 비교용 `single-sequential`과 나머지 CPU 구성은 후속 순차 측정으로 따로 보존했다. CARAFE의 순수 PyTorch 구현 비용은 공식 fused CUDA 구현의 비용과 동일하지 않다.

[DySample+ASFF CUDA 검사](profile-dysample-asff-cuda.json)는 NVIDIA GeForce RTX 5080에서 같은 입력 조건으로 실행했고 median 6.857 ms, peak allocated 215,608,832 bytes였다. 다른 GPU 구성과의 비교는 수행하지 않았다. GPU backward·optimizer·학습 전체 단계 검증은 아니다.

## 재실행

새 결과 폴더를 지정해 기존 기록을 보존한다. 아래는 검사이며 학습을 시작하지 않는다.

```bash
STUDY=/home/robin/shMin/INS_COD/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study
PY=/home/robin/shMin/INS_COD/.venv/fsa-study/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
python3 /home/robin/shMin/INS_COD/scripts/verify_migration.py --source /home/robin/shMin/COD
```

실제 dataset 전체 로딩, 실제 사전학습 파일 로딩, 완전한 새 lock 환경 설치, 학습 수렴·정확도는 아직 확인하지 않았다. [학습 시작 명령과 입력 조건](../../docs/RUNNING.md)을 따른다.
