# 2026-11-01 · 벤치마크 검증 — 4개 평가셋 복구와 전체 학습 데이터 계획

[English](README.md) | **한국어** | [프로토콜·실행 안내](../../docs/BENCHMARK-PROTOCOL.ko.md)

최종 로컬 검사에서 **694개 테스트와 74개 하위 테스트가 통과**했습니다. COD10K test만 추가하는 manifest, 기존 학습·체크포인트 근거 보존, `architecture=common` 메타데이터 호환, 합성 입력 추론 평가, 21개 지표, 평가셋 누락 표시, 별도 코드 폴더로의 실행 경로 변경, 전체 학습 데이터 계획 생성을 확인했습니다. 모델 테스트의 합성 텐서 미분 검사 외에 옵티마이저 업데이트나 실제 학습은 수행하지 않았습니다.

- [최종 출력](pytest.txt), [JUnit 기록](pytest.xml).
- [최초 실패 출력](pytest-initial.txt), [JUnit 기록](pytest-initial.xml): 새 추론 테스트가 남긴 PyTorch deterministic 설정 때문에 후속 CUDA 샘플러 경로가 달라졌습니다. 테스트 종료 시 RNG·deterministic·cuDNN·cuBLAS 환경을 복원하도록 수정했습니다. 미지원 연산이 있을 때 총 FLOPs를 표시하지 않는 기존 계산 규칙은 변경하지 않았습니다.
- [브라우저 검증](browser-verification.json): 합성 리더보드의 정렬·필터·표시 8개 검사. 기록된 해시는 최종 보고서 코드와 일치하며 임시 브라우저 프로필은 제거했습니다.
- [압축본 검증](archive-verification.json): 스테이징된 소스를 별도 폴더에 압축 해제해 CLI 4개, 모델 최초 import와 64×64 합성 입력 CPU 추론을 확인했습니다. 필수 형제 폴더 `Experiments/FSA-ZoomNeXt` 소스도 포함합니다. ZIP 크기는 이 검증 기록을 추가하기 전 검사한 스냅샷의 값입니다.
- [소스 체크섬·검증 범위](verification.json).

로컬 재현 명령은 다음과 같습니다. 기존 환경을 사용합니다.

```bash
REPO="/home/robin/shMin/INS_COD"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$REPO/.venv/fsa-study/bin/python" -m pytest -q \
  "$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/tests"
```

기존에 업로드된 priority8 보고서는 바이트 단위로 보존했습니다. 서버 원본 이미지·기존 모델 가중치·새 4개 평가셋 점수는 여기서 직접 검증하지 못했습니다. Windows 전송 스크립트는 검토했지만 실제 Windows 실행·SCP 전송·원격 명령 실행은 하지 않았습니다. 데이터·백본 다운로드와 환경 변경도 없습니다. 서버에서 파일·개수·해시·크기 검사를 통과해야 평가나 학습을 시작할 수 있습니다. 위 안내 문서에 기존 모델 재평가와 새 학습 명령을 분리해 두었습니다.
