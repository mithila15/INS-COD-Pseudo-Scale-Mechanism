# 2026-10-28 · INS_COD 비교 — 연산량 도구와 참조 소스 목차

파라미터 수와 MACs/FLOPs를 비교하는 스크립트 및 당시 사용한 모델 소스다. 합성 입력으로 측정하는 연산량과 데이터셋에서 측정하는 COD 성능은 서로 다른 결과다.

| 위치 | 주요 스크립트 | 비교 대상·특징 |
| --- | --- | --- |
| [Journal](Journal/) | [v3 CUDA safe](Journal/compare_cod_params_flops_v3_cuda_safe.py) | FSA case02, ZoomNeXt, ZoomNet |
| [Journal](Journal/) | [v6 case06](Journal/compare_cod_params_flops_v6_case06_only_safe.py) | case06을 추가한 비교, 추출된 프로젝트 경로 지정 지원 |
| [ModelCompare](ModelCompare/) | [case02 v6](ModelCompare/measure_compare_case02_v6.py) | BRG/FSA 등 변형 비교 |
| [ModelCompare](ModelCompare/) | [SINet v8](ModelCompare/measure_compare_sinet_v8.py) | SINet/SINet-V2 비교 입력 옵션 추가 |

Journal의 [_profile_repos](Journal/_profile_repos/), [_profile_repos_v2](Journal/_profile_repos_v2/), [_profile_repos_v3_cuda_safe](Journal/_profile_repos_v3_cuda_safe/), [_profile_repos_v6_case06_only_safe](Journal/_profile_repos_v6_case06_only_safe/)와 ModelCompare의 [_extracted_sources](ModelCompare/_extracted_sources/)는 당시 비교에 사용한 코드다. 일부 파일이 중복되더라도 수정 이력과 변형의 구분을 위해 원본 구조를 보존한다. 여기의 BRG 파생 코드에는 별도 수정이 있어 COD 루트 main과 같다고 간주하지 않는다.

스크립트 기본값에는 과거 `E:\COD\...zip`와 Windows 다운로드 경로가 남아 있다. ZIP을 새로 받기 전에 포함된 참조 소스의 경로를 CLI로 명시할 수 있는지 확인한다. SINet 등 스크립트가 지원하는 모든 모델 소스가 저장소에 들어 있는 것은 아니다. 없는 비교 대상은 별도로 준비하거나 해당 `--skip-*` 옵션을 사용해야 한다.

[실행 안내](../docs/RUNNING.md)에 포함 소스 경로를 쓰는 Journal 예제가 있다. 비교 결과를 보고할 때 입력 크기·스케일·배치 크기, 모델 너비, 측정 도구, MACs와 FLOPs의 환산 기준을 함께 기록한다.

[저장소 목차](../README.md) · [실험 목록](../Experiments/README.md)
