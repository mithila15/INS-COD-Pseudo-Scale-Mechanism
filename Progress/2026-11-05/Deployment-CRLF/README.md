# 2026-11-05 · Windows Deployment — CRLF로 중단된 학습 시작 복구

사용자 서버는 `bb6f040` 설치·가중치 검증 이후 launcher의 Bash 옵션 오류로 중단됐다. 새 `scale-core` tmux 세션이나 학습 프로세스가 생성되지 않은 상태였다. 설치·입력 준비 완료를 실제 학습 시작과 구분한다.

## 확인과 변경

Git의 Windows 설정 `core.autocrlf=true`를 Linux에서 적용해 archive를 만들자 LF Git blob도 CRLF로 내보내졌고, Bash가 `pipefail\\r`를 옵션으로 읽어 같은 오류가 재현됐다. 서버 파일을 직접 읽은 확정 검사는 아니며, 관측 오류와 재현으로 판단한 원인이다.

기존 전송 코드에서 Linux 줄바꿈을 보장하지 못한 부분을 수정했다. Shell LF attribute, archive 명령의 EOL 변환 해제, 설치기의 소스 shell 한정 보정과 hash 기록을 추가했다. 원본 ZIP·교체 전 파일·이전 결과는 보존한다. 기존 서버에는 재전송 없이 launcher를 백업·보정한 뒤 네 실험을 시작하는 복구 명령을 준비했다. 새 세션은 학습 실패 후에도 오류를 읽을 수 있는 Bash shell을 유지한다.

## 근거와 다음 단계

[검증 기록](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-05-deployment/README.ko.md)에 재현 JSON·52개 관련 검사 로그를 함께 게시한다. 학습·가중치 다운로드·SCP는 에이전트가 실행하지 않았다. 실제 Windows 셸은 미검증이며 Git의 Windows EOL 설정을 모사했다.

[한국어 복구·시작 명령](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md#이미-설치한-crlf-실행-스크립트-복구) · [영문](../../../Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md#recover-an-already-installed-crlf-launcher). 기존 데이터·환경·가중치를 참조하고 새 `scale-mechanism-core-os32-seed3407-<timestamp>`에 결과를 쓴다. 실제 서버에서 CPU 사전 검사를 거쳐 GPU 학습에 들어가는지 확인해야 한다.
