# 2026-11-01 · Pseudo-Scale Study — 실행과 재현 / Running and Reproduction

이 문서의 명령은 **입력 준비, 명령 생성, 학습, 평가, 추론 측정, 결과 집계**를 구분한다. `prepare`와 `plan_study.py`는 학습하지 않는다. `train` 또는 생성된 `train.sh`/`run_all.sh`를 사용자가 실행해야 파라미터 업데이트가 시작된다. 데이터셋과 backbone 가중치는 자동 다운로드하지 않는다.

The commands distinguish **input preparation, command generation, training, evaluation, inference profiling, and aggregation**. Only explicit `train`/`train.sh` execution starts optimization. Dataset and backbone files must already exist locally; no command downloads them.

## 공통 경로 / Shared paths

아래 변수는 이후 모든 블록에서 **같은 터미널**을 사용한다. 명령은 현재 작업 디렉터리에 의존하지 않는다. `STUDY_DATA_ROOT`와 `STUDY_BACKBONE`은 실제 보유 자료의 절대 경로로 바꾼다. 아래 데이터·가중치 경로는 예시이며, 해당 자료가 이 위치에 존재한다는 검증은 하지 않았다. 새 실행마다 `STUDY_RUN_ID`를 바꾼다. 기존 manifest·run·plan·summary는 덮어쓰지 않는다.

Use the **same terminal** for these variable definitions and subsequent blocks. Replace the example data/backbone paths with existing absolute paths. These example inputs have not been verified on this host. Use a new run ID for each experiment; existing artifacts are rejected rather than overwritten.

```bash
export INS_ROOT="/home/robin/shMin/INS_COD"
export STUDY_ROOT="$INS_ROOT/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export STUDY_PYTHON="$INS_ROOT/.venv/fsa-study/bin/python"
export STUDY_DATA_ROOT="/home/robin/datasets/COD"
export STUDY_BACKBONE="$INS_ROOT/.venv/fsa-study/assets/pvt_v2_b2.pth"
export STUDY_RUN_ID="2026-10-28-study01"
export STUDY_RESULTS="$STUDY_ROOT/results/$STUDY_RUN_ID"
export STUDY_MANIFEST="$STUDY_RESULTS/data/manifest.json"
export STUDY_DEVICE="cuda:0"
```

`STUDY_PYTHON`은 이번 코드 검증에 사용한 격리 환경 경로다. 다른 장비에서는 그 장비의 PyTorch·torchvision·timm·einops·NumPy·Pillow 환경 절대 경로로 바꾼다. 공식 테스트 지표에는 `pysodmetrics` 배포 패키지의 `py_sod_metrics` 모듈이 필요하다. CPU만 있으면 `STUDY_DEVICE="cpu"`를 지정한다. 환경 기록은 [environment](../environment/)를 참조한다.

`STUDY_PYTHON` identifies the isolated verification environment on this host. On another machine, use its prepared environment with PyTorch, torchvision, timm, einops, NumPy and Pillow. Official test evaluation additionally requires `py_sod_metrics` from the `pysodmetrics` distribution. Select `cpu` explicitly on a CPU-only host. See the [environment records](../environment/).

## 1. 데이터와 고정 분할 준비 / Prepare audited data and a fixed split

데이터와 평가 도구 출처: [SINet 저자 저장소](https://github.com/DengPingFan/SINet), [PySODMetrics](https://github.com/lartpang/PySODMetrics). SINet의 testing set 안내에는 COD10K-test, CAMO-test, CHAMELEON이 포함된다. 이번 작업에서 해당 데이터 다운로드는 실행하지 않았다. / Author-provided data/tool pointers; no dataset download was started.

지원하는 원본 배치는 다음과 같다. 학습 원본은 COD10K+CAMO가 필수이고 CLI는 기본적으로 CAMO·CHAMELEON·COD10K·NC4K 테스트셋 4개를 모두 요구한다. 하나라도 없으면 준비를 중단한다. `--test-datasets cod10k camo nc4k`처럼 이름을 명시하면 지정한 데이터가 없을 때 중단한다. `--test-datasets none`은 테스트 데이터를 manifest에 넣지 않는다.

COD10K and CAMO training inputs are required. By default, the preparation CLI requires all four test datasets: CAMO, CHAMELEON, COD10K and NC4K. Missing any of them stops preparation. An explicit list requires every named dataset to exist; `--test-datasets none` prepares training/validation only.

```text
STUDY_DATA_ROOT/
  COD10K-v3/
    Train/{Image,GT_Object}/
    Test/{Image,GT_Object}/
  CAMO-V.1.0-CVIU2019/
    Images/{Train,Test}/
    GT/
  CHAMELEON/{Imgs,GT}/
  NC4K/{Imgs,GT}/
```

```bash
"$STUDY_PYTHON" "$STUDY_ROOT/run_study.py" prepare \
  --data-root "$STUDY_DATA_ROOT" \
  --manifest "$STUDY_MANIFEST" \
  --split-seed 1729 --val-fraction 0.1
```

이 단계는 파일을 읽어 SHA-256과 분할 목록을 기록하는 **전처리**다. 원본 이미지·마스크를 복사·변경하지 않는다. COD10K 전체 배포본의 비위장·배경 이미지가 섞여 있으면 기본 `--cod10k-prefix COD10K-CAM-`으로 위장 이미지·마스크를 먼저 선택한다. 제외 파일도 경로·해시·원본 수를 기록한다. 이미 이름이 바뀐 위장 전용 복사본에만 `--cod10k-prefix ''`를 명시한다. CAMO는 Train/Test가 하나의 GT를 공유하므로 합집합으로 대응을 검사한다.

Preparation reads files and records hashes/split IDs without modifying or copying originals. The default explicit `COD10K-CAM-` prefix selects foreground COD10K examples before strict pair/count checks; excluded source files are also listed and hashed. Use an empty prefix only for an audited renamed foreground-only copy. CAMO's shared GT directory is checked against the union of its selected image splits.

정상 원본 수는 COD10K train 3,040/test 2,026, CAMO train 1,000/test 250, CHAMELEON 76, NC4K 4,121이다. 학습 원본마다 10%를 고정 분리하므로 실제 최적화에는 **3,636장**(2,736+900), 검증에는 **404장**(304+100)을 사용한다. 모델 seed가 바뀌어도 manifest가 같으면 분할은 같다. 이 설정을 전체 4,040장 학습 논문 수치와 동일 조건이라고 표시하면 안 된다. `--allow-nonstandard-counts`는 작은 검증 fixture·명시적인 사용자 데이터에만 사용하고 표준 벤치마크 수를 우회하는 용도로 쓰지 않는다.

The standard sources contain COD10K 3,040 train/2,026 test, CAMO 1,000 train/250 test, CHAMELEON 76, and NC4K 4,121 examples. The per-source 10% holdout yields **3,636 optimization examples** and **404 validation examples**. Model seeds do not change this manifest split. This is not the same training protocol as papers using all 4,040 images. The nonstandard-count override is intended for explicit custom datasets or tiny verification fixtures.

이미지/마스크 stem 중복, 대응 누락, 선택 범위의 고아 마스크, split 간 동일 이미지 해시, manifest 수정, 이후 입력 파일 변경을 검사한다. 사용하지 않는 CAMO 공유 GT 파일은 해당 미선택 source와 함께 별도 기록한다. 검증·테스트 마스크는 원래 해상도에서 평가한다. 학습은 동일한 seed/epoch/sample ID로 좌우·상하 반전을 결정하고, 384×384로 resize한 마스크에서 5×5 morphological gradient edge를 생성한다. 기존 GT_Edge·instance·교사 출력을 사용하지 않는다.

Integrity checks reject duplicate stems, unmatched selected pairs, orphan masks, identical image hashes crossing splits, changed manifests, and changed files. Unused CAMO shared masks are explicitly recorded with their unselected source. Evaluation retains original mask resolution. Training flips are keyed by seed/epoch/sample ID, independent of model RNG. Edges are generated by a 5×5 morphological gradient after mask resize; stored GT_Edge, instance labels and teacher outputs are not required.

## 2. 한 조합 학습 / Train one variant

아래 명령은 **실제 학습**이다. 기존 PVTv2-B2 backbone 체크포인트를 재사용하며 classification head 두 키를 제외한 나머지는 strict하게 로드한다. 기존 full FSA 체크포인트를 backbone 파일로 지정하지 않는다. 파일은 flat state dict 또는 `state_dict` wrapper 형태여야 한다.

The following command **starts optimization**. It reuses a local PVTv2-B2 backbone checkpoint, strictly loading all backbone keys after removing the two classification-head keys. A full FSA checkpoint is not a backbone checkpoint; use a flat backbone state dict or a `state_dict` wrapper.

```bash
"$STUDY_PYTHON" "$STUDY_ROOT/run_study.py" train \
  --manifest "$STUDY_MANIFEST" \
  --generator sfp --utilizer asff --seed 3407 \
  --backbone-weights "$STUDY_BACKBONE" \
  --size 384 --channels 48 --epochs 150 --batch-size 4 \
  --workers 2 --device "$STUDY_DEVICE" \
  --output "$STUDY_RESULTS/single-runs/sfp__asff__seed3407"
```

무작위 초기화 비교를 의도할 때만 `--backbone-weights "$STUDY_BACKBONE"` 대신 `--random-init`을 사용한다. 초기화 조건은 모든 비교군에서 맞춘다. `--baseline`은 pseudo 생성·활용 모듈이 없는 공통 decoder 대조군이며, `--legacy-reference`는 원래 전체 FSA 구조다. 두 옵션은 동시에 사용할 수 없다. 생성기·활용기의 선택적 인자는 JSON 객체로 전달한다. 예를 들어 차이 연산의 작은 분모 조건은 `--utilizer relative --utilizer-options '{"eps":1e-6}'`로 바꿀 수 있다.

Use `--random-init` instead of `--backbone-weights` only for an intentional random-initialization study, consistently across comparison arms. `--baseline` disables pseudo generation/use with the common decoder; `--legacy-reference` selects the complete original FSA architecture. They are mutually exclusive. Optional generator/utilizer parameters accept JSON objects, for example `--utilizer relative --utilizer-options '{"eps":1e-6}'`.

기본값은 Adam, lr 1e-4, encoder lr 0.1배, weight decay 0, patch_embed1 고정, encoder BN 통계·affine 고정, 150 epochs 중 약 2/3 시점 lr 0.1배 감소이다. 모든 조합에 같은 segmentation/deep/region/edge/UAL 가중치 1/0.25/0.5/1/1을 적용한다. 기본 `holdout`에서는 매 epoch **학습 원본에서 분리한 검증셋의 이미지 수 가중 평균 MAE**만으로 `best.pth`를 선택한다. `last.pth`, `config.json`, `manifest.json`, `history.jsonl`, 완료 시 `status.json`도 저장한다. 테스트 평가를 자동으로 실행하지 않는다.

Defaults use Adam, lr 1e-4, encoder lr factor 0.1, zero weight decay, frozen patch_embed1 and encoder BN statistics/affine, and a 0.1 lr step at approximately two thirds of 150 epochs. All variants share segmentation/deep/region/edge/UAL weights 1/0.25/0.5/1/1. In the default `holdout` protocol, only sample-weighted validation MAE selects `best.pth`; `last.pth` and complete configuration/history are also saved. Training never invokes test evaluation.

`--determinism warn`이 기본이며 deterministic 연산을 요청하되 CUDA `grid_sample` backward 등 미지원 연산은 경고한다. 엄격한 실패가 필요하면 `--determinism strict`를 사용한다. seed를 고정해도 하드웨어·버전 간 bitwise 동일성을 주장하지 않는다. 체크포인트에 optimizer/scheduler 상태가 있지만 현재 CLI는 중간 재개 기능을 제공하지 않는다.

The default `--determinism warn` requests deterministic algorithms and warns when an operator such as CUDA grid_sample backward lacks support. `strict` rejects such operators. Fixed seeds do not imply bitwise equality across hardware/software. Optimizer/scheduler states are archived, but the CLI does not currently implement interrupted-run resumption.

## 3. 비교 행렬 생성과 실행 / Generate and execute an experiment matrix

[생성기](GENERATION.md)와 [활용기](UTILIZATION.md)의 논문 근거·변형 범위를 먼저 확인한다. `generation`은 고정 활용기 아래 12개 생성기와 두 대조군, `utilization`은 고정 생성기 아래 13개 활용기와 두 대조군, `cross`는 12×13 조합과 두 대조군이다. seed 3407/3408/3409를 쓰면 각각 **42 / 45 / 474회** 학습이다. phase 간 중복 대조군은 각기 다른 결과 루트에 저장한다.

Read the [generator](GENERATION.md) and [utilization](UTILIZATION.md) references and adaptation notes. `generation` includes 12 generators plus two controls; `utilization` includes 13 utilizers plus two controls; `cross` includes 12×13 pairs plus two controls. With three seeds, these plans contain **42 / 45 / 474 training runs**, respectively. Keep different phases under separate result roots.

다음은 전체 행렬의 **명령 생성만** 수행한다. `--phase generation --utilizer mean` 또는 `--phase utilization --generator legacy`로 단계별 계획을 만들 수 있다. 단계별 plan/results 경로도 새로 지정해야 한다. 명령 파일은 shell quoting이 적용된 절대 경로를 사용한다.

This block **writes commands only**. Use `--phase generation --utilizer mean` or `--phase utilization --generator legacy` for separate phases, with new plan/result paths for each phase. Generated shell commands use quoted absolute paths.

```bash
export STUDY_PLAN="$STUDY_RESULTS/plans/cross"
export STUDY_RUNS="$STUDY_RESULTS/runs/cross"

"$STUDY_PYTHON" "$STUDY_ROOT/plan_study.py" \
  --phase cross --seeds 3407 3408 3409 \
  --manifest "$STUDY_MANIFEST" \
  --python "$STUDY_PYTHON" --backbone-weights "$STUDY_BACKBONE" \
  --epochs 150 --batch-size 4 --size 384 --channels 48 --workers 2 \
  --device "$STUDY_DEVICE" \
  --results "$STUDY_RUNS" --output "$STUDY_PLAN"
```

`plan.json`과 `train.sh` 내용을 검토한 뒤 다음 명령으로 학습과 validation 리더보드 갱신을 시작한다. 완료된 동일 조건 단계는 건너뛰지만 부분 학습은 resume하지 않고 보존·중단한다. 전체 학습→테스트→profiling 자동 실행은 `run_all.sh`를 사용한다. 실행 스크립트를 실제 호출하기 전에는 학습하지 않는다.

After reviewing `plan.json`/`train.sh`, run the following to train sequentially and refresh the validation leaderboard after each run. Completed matching stages skip, while partial training is preserved and rejected; optimizer resume is not implemented. Use `run_all.sh` for the complete training → test evaluation → profiling sequence. Creating a plan never executes these actions.

```bash
bash "$STUDY_PLAN/train.sh"
```

## 4. 독립 테스트와 추론 비용 / Explicit test evaluation and inference cost

다음은 **체크포인트 평가**이며 파라미터 업데이트가 없다. 학습 때와 동일한 manifest 해시·변형 메타데이터·정확한 state key를 검사한다. `--split test`는 manifest에 포함된 테스트셋만 평가하며 없는 CHAMELEON 점수를 만들지 않는다. S·weighted F·MAE·E(adaptive/mean/max)에 일반 Fβ(adaptive/mean/max)와 고정 임계값 IoU·Dice/F1·precision·recall·specificity·BER·FPR·FNR·accuracy·balanced accuracy·Boundary IoU를 더해 21개 값을 출력한다. 이미지별 CSV·확률 PNG와 PR/Fβ/E 곡선 CSV/SVG/PNG를 저장한다. [평가 규약](EVALUATION.ko.md)을 참고한다. `py_sod_metrics`가 없으면 대체 지표를 넣지 않고 중단한다.

This is **checkpoint evaluation without updates**. Manifest identity, variant metadata and exact state keys are checked. `test` evaluates only datasets present in the manifest; unavailable CHAMELEON is not fabricated. The expanded protocol exports 21 scalar scores: S/wF/MAE, adaptive/mean/max E and Fβ, plus IoU/Dice/F1 and fixed-threshold confusion/boundary metrics. Per-image CSV, probability PNG and PR/Fβ/E curve CSV/SVG/PNG are saved. See [evaluation definitions](EVALUATION.md). Missing `py_sod_metrics` causes a clear error rather than substituted scores.

```bash
"$STUDY_PYTHON" "$STUDY_ROOT/run_study.py" evaluate \
  --manifest "$STUDY_MANIFEST" \
  --checkpoint "$STUDY_RESULTS/single-runs/sfp__asff__seed3407/best.pth" \
  --split test --batch-size 4 --workers 2 --device "$STUDY_DEVICE" \
  --output "$STUDY_RESULTS/single-runs/sfp__asff__seed3407/test"
```

예측은 sigmoid → 모델 해상도에서 이미지별 min-max → 원본 mask 크기로 bilinear resize → 공식 지표 입력용 uint8 변환 순서다. 검증 선택에는 uint8 변환 전 float MAE를 사용한다. 이는 원본 코드의 min-max 순서를 보존한 명시적 평가 규약이며, 다른 논문의 resize·정규화 순서와 확인 없이 동일하다고 주장하면 안 된다.

Predictions use sigmoid, per-image min-max at model resolution, bilinear resize to the original mask, then uint8 conversion for official metrics. Validation selection uses float MAE before uint8 conversion. This preserves the archived evaluation order; verify resize/normalization conventions before comparing numbers with other papers.

전체 행렬의 평가·추론 비용 측정은 각각 아래 명령이다. 학습 완료 후 사용할 수 있다. `profile.sh`는 batch 1, 학습 해상도, 동일 seed의 합성 RGB 입력으로 모델 전체 출력 head를 포함한 장치 내 추론 latency·파라미터·GPU peak memory·branch diagnostics를 기록한다. MACs와 2×MACs 규약의 FLOPs, 별도 비행렬 연산 추정량과 지원/미지원 연산 목록도 저장한다. 데이터 로딩/CPU↔GPU 전송 시간·학습 메모리는 제외한다. checkpoint를 지정하지 않은 profile은 무작위 초기화 검사이며 COD 성능이나 학습된 branch 유용성의 근거가 아니다.

After training, these scripts separately perform test evaluation and inference profiling. Profiling uses batch 1, the training resolution and identically seeded synthetic RGB input, including all prediction heads. It reports latency, parameters, CUDA peak memory and branch diagnostics; it also reports MACs, 2×MACs FLOPs, separate noncontraction estimates and operation coverage. IO/transfer time and training memory are excluded. Profiling without a checkpoint is an initialization check, not COD quality evidence.

```bash
bash "$STUDY_PLAN/evaluate.sh"
bash "$STUDY_PLAN/profile.sh"
```

학습 전 합성 입력으로 **추론만** 확인하려면 별도 출력에 다음을 실행한다. CPU의 작은 입력 실행은 논문용 GPU 384×384 비용 비교를 대신하지 않는다.

For an **inference-only** initialization check before training, use a separate output. A small CPU-input check does not replace a GPU 384×384 benchmark.

```bash
"$STUDY_PYTHON" "$STUDY_ROOT/profile_study.py" \
  --generator sfp --utilizer asff --size 64 --channels 48 \
  --device cpu --batch-size 1 --threads 2 --warmup 2 --repeats 5 \
  --output "$STUDY_RESULTS/initialization/sfp__asff_cpu64.json"
```

## 5. 결과 집계와 검증 범위 / Aggregate results and understand verification limits

각 run에 `profile.json`이 있으면 파라미터·latency·throughput·CUDA peak memory도 집계한다. 평가와 profiling 모두 해당 run의 `best.pth` 해시를 확인하며, 측정 장비·입력·dtype·반복 조건이 다르면 같은 평균으로 합치지 않는다. / Optional inference profiles are grouped by matching hardware and measurement conditions and checked against the selected checkpoint.

```bash
"$STUDY_PYTHON" "$STUDY_ROOT/summarize_study.py" \
  --results "$STUDY_RUNS" --output "$STUDY_RESULTS/summary/cross"
```

이 명령은 완료된 run의 검증·테스트 지표를 조건별로 분리하고 seed 간 평균·표본 표준편차를 JSON/CSV로 기록하고 HTML/Markdown/CSV/JSON 리더보드를 생성한다. Validation MAE 순위, 데이터셋별 테스트 순위, 비용 표를 분리한다. 같은 조건의 동일 seed가 중복되면 중단하며, 미완료 run은 따로 나열한다. 동일 프로토콜·동일 완료 seed 집합끼리만 순위를 계산한다. 테스트 점수는 보고용이며 체크포인트를 선택하지 않고 통계적 유의성을 주장하지 않는다. 전체 행렬을 탐색한 뒤 최종 조합은 검증 결과로 정하고 독립 테스트로 보고한다. 같은 테스트 결과를 보고 반복 수정했다면 독립 최종 검증이라고 해석하지 않는다.

Aggregation separates comparison conditions and reports seed means/sample standard deviations in JSON/CSV, plus HTML/Markdown/CSV/JSON leaderboards. Under the `holdout` protocol, validation MAE supplies the selection ranking; test and cost ranks are separate. Duplicate seeds fail; incomplete runs are listed. Only matching protocols and completed seed sets share a rank. Test ranks are descriptive and do not select checkpoints or establish significance. Choose final variants using validation evidence before interpreting independent test results; repeated redesign after inspecting those tests breaks that interpretation.

현재 준비된 것은 실행 가능한 비교 코드, 합성 데이터 무결성 검사, 모델 초기화/연산 검증이다. 실제 데이터 전체 로딩·사전학습 원본 파일 로딩·학습 수렴·성능 순위·전체 GPU 학습 시간은 해당 단계를 실행해야 확인된다. 이 문서의 명령 제공은 학습 실행 기록이 아니다.

The delivered artifacts prepare executable comparisons, synthetic data-integrity checks and initialization/operator verification. Full real-data loading, an actual pretrained backbone file, convergence, accuracy rankings and total GPU training time require their corresponding explicit runs. Providing these commands is not evidence that training occurred.

Automatic reports: `$STUDY_PLAN/leaderboard/leaderboard.html`. `run_all.sh` runs every stage; `train.sh` runs training only. Both refresh the derived leaderboard after every stage. Add `--plan "$STUDY_PLAN/plan.json"` when summarizing manually to include not-started runs. / 자동 리더보드는 같은 경로에 저장하며 수동 집계에 `--plan`을 추가하면 미시작 실험도 표시한다.

## 전체 데이터 학습과 빠진 테스트셋 / Full-data training and missing tests

기본 `holdout` 분할은 유지한다. `--training-protocol full_train`은 준비와 plan/train 양쪽에 명시해야 하며 `--val-fraction`은 생략하거나 `0`으로 준다. 학습 4,040장·validation 없음·최종 epoch 고정 선택을 사용하고 `best_val_mae`는 null이다. 테스트 점수로 체크포인트를 선택하지 않는다. [기존 8개 실험의 평가 복구와 새 전체 학습 명령](BENCHMARK-PROTOCOL.ko.md)은 원본 결과를 보존하고 새 경로를 사용한다.

The default holdout split is preserved. Explicit `--training-protocol full_train` is required in both preparation and planning/training; omit `--val-fraction` or set it to `0`. It uses 4,040 training images, no validation, fixed-final-epoch selection and null `best_val_mae`. Test scores never select checkpoints. The [existing-run recovery and fresh full-data training guide](BENCHMARK-PROTOCOL.md) preserves previous outputs and uses separate destinations.

`priority8`는 고정한 8개 변형의 phase이며 `--seeds 3407`이면 8회다. 기존 `generation`·`utilization`·`cross`의 구성 수는 그대로다. `priority8` is a fixed eight-variant phase; `--seeds 3407` produces eight runs. The existing matrix phase counts remain unchanged.
