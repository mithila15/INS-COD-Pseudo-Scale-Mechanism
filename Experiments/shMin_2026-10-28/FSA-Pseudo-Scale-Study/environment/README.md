# 2026-10-28 · FSA Study Environment — 원본 환경과 신규 검증 프로필

이 연구는 [원본 FSA requirements](../../../FSA-ZoomNeXt/requirements.txt)를 보존하고 별도 환경에서 동작한다. 원본 환경을 업그레이드하거나 다른 연구 환경에 패키지를 설치하지 않는다.

- Python **3.12.13**, PyTorch **2.11.0** (`2.11.0+cu130` runtime), torchvision **0.26.0**, timm **1.0.30**.
- [직접 의존성](../requirements.txt), [검증 버전 constraints](constraints-verified.txt), [해시 포함 transitive lock](requirements.lock), [실제 설치·검증 기록](verification.json).
- 이번 로컬 검증은 `/home/robin/shMin/INS_COD/.venv/fsa-study`에 lightweight 의존성을 설치하고, 기존 `/home/robin/shMin/GeneralGodhand/.venv/isaac`의 PyTorch 패키지를 `.pth`로 읽는 overlay를 사용했다. 원본 환경은 수정하지 않았다. 이 방식은 해당 로컬 경로가 있어야 하므로 다른 장비의 재현 설치 명령으로 사용하지 않는다. 다른 연구의 Isaac 패키지는 이 연구의 의존성이 아니다.
- lock은 이번에 실제 사용한 관련 패키지 버전에 고정하고 배포 파일 해시를 추가했다. 완전히 새 환경에서 lock 전체를 설치하는 작업은 수행하지 않았다. CPU/GPU 런타임과 드라이버 호환은 해당 장비에서 확인한다.

**English.** Archived dependencies remain unchanged. Verification used an isolated environment with a read-only overlay for existing PyTorch binaries; it did not modify the shared environment. The portable lock pins the relevant installed package versions and distribution hashes. A fresh full lock installation has not been tested.

새 장비의 설치 예시다. 필요한 Python 3.12.13을 먼저 준비한다. 패키지 다운로드를 수행하며 데이터셋이나 모델 가중치를 다운로드하지 않는다.

```bash
REPO=/home/robin/shMin/INS_COD
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
python3.12 -m venv "$REPO/.venv/fsa-study-clean"
PY="$REPO/.venv/fsa-study-clean/bin/python"
"$PY" -m pip install --require-hashes -r "$STUDY/environment/requirements.lock"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

원본 PVT 구현은 timm의 옛 import 경로 및 torch의 옛 SDPA context manager에 대한 deprecation warning을 발생시키지만 검증된 버전에서 실행된다. 원본 소스 무결성을 유지하기 위해 해당 파일을 수정하지 않았다.
