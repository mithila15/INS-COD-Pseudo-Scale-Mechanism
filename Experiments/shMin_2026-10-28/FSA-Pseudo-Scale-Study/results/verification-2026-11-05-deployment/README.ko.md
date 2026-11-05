# 2026-11-05 · 배포 검증 — Windows CRLF 실행 오류 복구

[English](README.md) | **한국어** · [사용자 실행 복구 명령](../../docs/SCALE-MECHANISM.ko.md#이미-설치한-crlf-실행-스크립트-복구)

서버에서 `bb6f040` 설치와 기존 manifest·backbone 가중치 확인은 완료됐지만, Bash 옵션 오류로 새 학습 세션 생성 전에 종료됐다. 앞서 받은 프로세스 목록에도 학습이 없었다. 서버 파일의 원본 바이트는 직접 읽지 못했다. 같은 오류를 로컬에서 재현한 결과가 CRLF 진단의 근거다.

임시 Git 저장소에 LF launcher 3줄을 커밋한 뒤 `core.autocrlf=true`로 내보내면 CRLF 3줄·Bash exit 2 (`pipefail\\r: invalid option name`)가 발생했다. `core.autocrlf=false`·`core.eol=lf`를 적용하면 CRLF 0줄·exit 0이었다. [관측 기록](reproduction.json). [Git 줄바꿈 규칙](https://git-scm.com/docs/gitattributes).

세 곳을 보완했다. `.gitattributes`에서 shell을 LF로 고정하고, 전송기는 ZIP 생성 시 자동 줄바꿈 변환을 끈다. 설치기는 원본 ZIP SHA256을 검증한 뒤 소스 shell만 LF로 바꾸고 원본·설치 hash를 남긴다. 기존 파일은 교체 전에 백업한다. 과거 결과·데이터·checkpoint·가상환경은 계속 설치 대상에서 제외하며 Python 소스 바이트는 설치기에서 바꾸지 않는다.

**관련 테스트 52개 통과**, 기존 timm 경고 1개다. [로그](pytest.txt), [archive 회귀 검사](../../tests/test_deploy_line_endings.py), [설치기 회귀 검사](../../tests/test_install_study.py). 두 archive 방어를 각각 확인하고 binary 보존, 과거 CRLF ZIP 복원·Bash 실행, 교체 전 바이트 백업, 반복 설치, 결과 shell 보존, 기존 구조·plan·report 연동을 검사했다.

배포 검사이며 모델 학습 결과가 아니다. 가중치·데이터 다운로드와 optimizer step은 없었다. Linux Git에서 Windows의 줄바꿈 설정을 모사했으며 실제 Windows PowerShell·SCP·서버 실행은 수행하지 않았다. 배포에 한정된 변경이므로 이전 전체 모델 검사는 반복하지 않았다. 사용자 복구 블록의 Bash 문법은 검사했으며 실제 서버 학습·메모리·속도 확인은 남아 있다.
