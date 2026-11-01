# 2026-11-01 · INS_COD — Pseudo-scale experiment guide

**English** | [한국어](README.ko.md)

Compare **how to generate pseudo-scale features** and **how to use them** in a single-encoder camouflaged object detector. Utilization includes subtraction, direct fusion, and channel/spatial attention. The study provides **12 generators × 13 utilizers**, plus a no-pseudo baseline and the archived full FSA reference: **158 configurations, 474 runs with three seeds**.

This README is the execution guide. See the [method catalog and comparison rationale](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.md) for what each variant tests. Referenced papers and adaptations are retained below. The uploaded [priority8 results](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-report-20261101_172449/README.md) cover eight runs with one seed and **three test datasets; COD10K-test is missing**.

For existing results, use the [four-dataset recovery and full-data training guide](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md). Existing training uses **3,636 images + 404 validation**, while official ZoomNeXt uses **4,040 training images**. The new `full_train` option uses all 4,040 and a fixed final-epoch checkpoint; other architecture/training differences remain. [Windows pull and SCP transfer](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md#transfer-the-updated-source-from-windows) preserves the existing server checkout and results.

The server audit also found **seven exact COD10K train-source/test duplicates (six train, one validation)**. Strict preparation stops by default. The [explicit overlap policy](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md#acknowledge-the-verified-cod10k-overlaps) records only the verified ID/hash pairs, preserves published sample counts and exposes `disjoint=false`; it does not make COD10K evaluation independent. [Evidence and limits](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-01-overlap/README.md).

## 1. Set up the environment

Start from a checkout of this repository. Set `REPO` to its absolute path on your machine. Use the **same terminal** for all blocks below. The documented profile uses Python 3.12.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.30 and the checked-in dependency lock.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if it is unavailable. The following block creates a **new** environment and downloads Python/packages only. If this environment already exists, reuse its `PY` path and skip creation/sync unless deliberately rebuilding it. GPU execution requires a driver compatible with the locked CUDA runtime; CPU execution can use `DEVICE=cpu`. This lock is the documented Linux CUDA profile, not a separate CPU-only or macOS lock.

```bash
REPO=/home/robin/shMin/INS_COD
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study-clean/bin/python"

if [ -e "$REPO/.venv/fsa-study-clean" ]; then
  printf '%s\n' 'Environment already exists; run the checks below before reusing it.'
else
  uv python install --no-bin 3.12.13 &&
  uv venv --directory "$STUDY" --no-project --managed-python --python 3.12.13 \
    "$REPO/.venv/fsa-study-clean" &&
  uv pip sync --directory "$STUDY" --python "$PY" --require-hashes \
    "$STUDY/environment/requirements.lock"
fi

"$PY" -c 'import torch, torchvision, timm, einops, py_sod_metrics; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

The checks import dependencies and test operators; they do not run training. For installation prerequisites, CUDA checks, troubleshooting, and the distinction between the historical local verification environment and a fresh installation, see [environment setup](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.md). A fresh full lock installation and actual pretrained-data training have not been validated in this task.

## 2. Prepare existing data and backbone weights

Replace `DATA` and `WEIGHTS` with existing absolute paths. `WEIGHTS` must be a **PVTv2-B2 backbone** state dictionary (flat or `state_dict` wrapper), not a full FSA checkpoint. The loader removes the classification-head keys and loads the backbone strictly. Data and model weights are not downloaded by these commands.

```bash
DATA=/absolute/path/to/COD-Data
WEIGHTS=/absolute/path/to/pvt_v2_b2.pth
DEVICE=cuda:0
RUN="$STUDY/results/$(date +%Y%m%d_%H%M%S)"
MANIFEST="$RUN/data/manifest.json"

"$PY" "$STUDY/run_study.py" prepare \
  --data-root "$DATA" --manifest "$MANIFEST" \
  --split-seed 1729 --val-fraction 0.1
```

Expected layout:

```text
COD-Data/
  COD10K-v3/Train/{Image,GT_Object}/
  COD10K-v3/Test/{Image,GT_Object}/
  CAMO-V.1.0-CVIU2019/Images/{Train,Test}/
  CAMO-V.1.0-CVIU2019/GT/
  CHAMELEON/{Imgs,GT}/
  NC4K/{Imgs,GT}/
```

`prepare` checks image/mask pairing and hashes, selects COD10K foreground files with the `COD10K-CAM-` prefix, and creates one fixed split. CAMO and COD10K training data are required. The CLI now requires **all four test datasets by default**: CAMO 250, CHAMELEON 76, COD10K 2,026 and NC4K 4,121. An intentional subset must be explicit with `--test-datasets`; `none` prepares training/validation only. Partial test coverage remains visibly incomplete in the leaderboard. Keep the **same manifest for every compared run**. Test evaluation requires at least one test dataset in it.

The default 10% holdout gives **3,636 training images and 404 validation images** from the original 4,040 training images. The commands below use this default `holdout` protocol. For fresh full-data training, set `--training-protocol full_train` in both preparation and planning, and omit `--val-fraction` or use `0`; [migration of existing priority8 plans](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md#train-a-separate-full-data-comparison) preserves old results. Renamed data, explicit test selection and the full evaluation protocol are described in [advanced execution details](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md).

## 3. Choose the experiment scope and generate commands

Use **`cross` for the complete study**. It already contains the fixed-utilizer and fixed-generator comparisons; the other phases are optional ways to investigate one axis first, not additional mandatory runs.

| Phase | What changes | Fixed choice | Configurations | Runs with 3 seeds |
| --- | --- | --- | ---: | ---: |
| `generation` | All 12 generators | `--utilizer mean` by default | 14 | 42 |
| `utilization` | All 13 utilizers | `--generator legacy` or a generator selected on validation | 15 | 45 |
| `cross` | Every generator/utilizer pairing | Neither | 158 | 474 |
| `priority8` | The fixed eight-job screening sequence | Prespecified variants | 8 | 24; use `--seeds 3407` for 8 |

The three matrix phases include `single` and `legacy_full`. `priority8` includes `single` and seven prespecified variants, without `legacy_full`. For a staged study, compare generation with mean fusion, inspect validation results, then manually choose the generator for utilization. Use separate phase output roots. The script does not select a winner automatically. In `cross`, `GENERATOR`/`UTILIZER` below do not restrict the grid.

```bash
PHASE=cross
GENERATOR=legacy
UTILIZER=mean
PLAN="$RUN/plans/$PHASE"
RUNS="$RUN/runs/$PHASE"

"$PY" "$STUDY/plan_study.py" \
  --phase "$PHASE" --generator "$GENERATOR" --utilizer "$UTILIZER" \
  --seeds 3407 3408 3409 --manifest "$MANIFEST" \
  --backbone-weights "$WEIGHTS" --python "$PY" --device "$DEVICE" \
  --size 384 --channels 48 --epochs 150 --batch-size 4 --workers 2 \
  --output "$PLAN" --results "$RUNS"
```

This writes `plan.json`, `train.sh`, `evaluate.sh`, `profile.sh`, and `run_all.sh`; it does **not** train. Keep size, channels, epochs, batch size, seeds, manifest and backbone initialization fixed across comparisons. Defaults use Adam, lr 1e-4, encoder lr ×0.1, frozen first patch embedding, and shared losses. [Full common protocol](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/README.md).

## 4. Start training with automatic evaluation and leaderboards

The following starts each planned run's **training → test evaluation → inference profiling**, sequentially, and refreshes the leaderboard after every stage. Under the default holdout protocol, `best.pth` is selected using **validation MAE only**. Under `full_train`, it is the **fixed final-epoch model**, with null validation MAE and no validation ranking. The full comparison is specified in the plan before execution; test rankings are descriptive and never select checkpoints.

```bash
bash "$PLAN/run_all.sh"
```

Open **`$PLAN/leaderboard/leaderboard.html`** in a browser. Validation MAE supplies the overall selection ranking; test tables rank each dataset separately (default S-measure), while cost tables show matching hardware/settings. Tables include seed means, sample SD, seed count, completion status and a comparison-group identifier. Different protocols or seed sets are kept in separate groups. The HTML supports sorting/filtering; CSV, JSON and Markdown are also saved.

To train only and automatically update the validation leaderboard, use `bash "$PLAN/train.sh"` instead. `run_all.sh` later skips completed matching stages. Interrupted/partial training does **not** resume: preserve it and create a new run destination/plan. Raw checkpoints, predictions and profiles are never overwritten. Only the derived leaderboard files are refreshed.

## 5. Inspect validation results and inference cost

These optional commands complete profiling and save an additional immutable summary snapshot:

```bash
bash "$PLAN/profile.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --plan "$PLAN/plan.json" \
  --output "$RUN/summary/validation-$PHASE"
```

Profiles report parameters, latency, throughput, CUDA peak memory, **MACs and FLOPs**, plus a separately labeled estimate of noncontraction operations and an operator coverage table. The common convention is **2 FLOPs per convolution/matrix multiply-accumulate**. Interpolation, sampling and nonlinear operations have explicit approximate counting rules; these are not hardware instruction counts. Unsupported operations are listed, never silently treated as free. Profiling includes all heads, batch 1, FP32 and synthetic RGB; it excludes data loading, transfers and training memory. [Metric definitions and limitations](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/EVALUATION.md).

## 6. Inspect test metrics and export paper tables

If training was run separately, the following completes test evaluation and saves an additional report. Completed stages are skipped after identity checks.

```bash
bash "$PLAN/evaluate.sh"
"$PY" "$STUDY/summarize_study.py" \
  --results "$RUNS" --plan "$PLAN/plan.json" \
  --test-metric S --output "$RUN/summary/test-$PHASE"
```

Each available dataset reports **21 scalar scores**: S, weighted F, MAE; adaptive/mean/max E and ordinary Fβ; fixed-threshold IoU, Dice/F1, precision, recall, specificity, BER, FPR, FNR, accuracy, balanced accuracy and Boundary IoU. Dice and F1 are aliases, not independent evidence. Mean/max threshold scores are computed from the dataset-average curve. Also saved: **PR/Fβ/E curves as CSV+SVG+PNG, per-image scores, and original-size grayscale probability PNGs**. For disk-limited runs, direct evaluation accepts `--no-save-predictions`; numeric scores and curves remain available.

`--test-metric` chooses the descriptive test table ordering, e.g. `wF`, `Fbeta_max`, `MAE` or `BoundaryIoU_0_5`; metric direction is recorded. Missing datasets/scores remain missing. Rankings alone are not significance tests. For existing checkpoints, evaluate into a **new** folder such as `test-v2`, then pass `--evaluation-subdir test-v2` to the summarizer. Use a new profile path such as `profile-v2.json` and `--profile-name profile-v2.json` in the planner/summarizer to preserve an older profile. Old and expanded metric protocols are separated.

Summarize **one phase root at a time**. Compare generators at a fixed utilizer and vice versa, keeping `single` and the whole-architecture `legacy_full` reference. Keep the same seed set, data split, initialization and measurement conditions.

```text
results/<run-id>/
  data/manifest.json
  plans/<phase>/{plan.json,train.sh,evaluate.sh,profile.sh,run_all.sh}
  plans/<phase>/leaderboard/{leaderboard.html,leaderboard.md,leaderboard.csv,leaderboard.json,summary.json,summary.csv}
  runs/<phase>/<variant>__seed<seed>/
    config.json, manifest.json, history.jsonl, status.json
    best.pth, last.pth
    test/{metrics.json,per_image_metrics.csv}
    test/predictions/<dataset-key>/*.png
    test/curves/<dataset-key>/*.{csv,svg,png}
    profile.json
  summary/<snapshot>/{summary.json,summary.csv,leaderboard.html,leaderboard.md,leaderboard.csv,leaderboard.json}
```

## Referenced papers

Checked on **2026-10-28**. Korean title translations are provided in the [Korean version](README.ko.md). These are adaptations of individual ideas to this study, not reproductions of the complete published networks. Details and mathematical definitions are in [generation](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/GENERATION.md) and [utilization](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/UTILIZATION.md).

| Paper | Official venue and publication status | Applied idea |
| --- | --- | --- |
| [Fast Feature Pyramids for Object Detection](https://doi.org/10.1109/TPAMI.2014.2300479) | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2014; journal | FFP scale power law; adapting calibrated low-level features to learnable deep-feature channel exponents is a new hypothesis. |
| [Exploring Plain Vision Transformer Backbones for Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_17) | European Conference on Computer Vision (ECCV), 2022; conference | Apply the Simple Feature Pyramid idea per PVT stage, with aligned branches; not a ViT detector reproduction. |
| [Pyramid Scene Parsing Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html) | IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017; conference | Pooled context branches; these do not establish approximation of real image-scale features. |
| [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | arXiv:1706.05587, 2017; preprint. Formal venue unverified. | Two atrous branches with reduced dilation settings; no complete DeepLabv3 reproduction. |
| [CARAFE: Content-Aware ReAssembly of FEatures](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html) | IEEE/CVF International Conference on Computer Vision (ICCV), 2019; conference | Content-aware reassembly in pure PyTorch; no claim of the official fused CUDA implementation's efficiency. |
| [Learning to Upsample by Learning to Sample](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html) | IEEE/CVF International Conference on Computer Vision (ICCV), 2023; conference | LP/static-scope offset sampling for pseudo-scale generation, followed by alignment. |
| [Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019; conference | Fuse/select channel attention on externally generated branches; not the original SK split kernels. |
| [Learning Spatial Fusion for Single-Shot Object Detection](https://arxiv.org/abs/1911.09516) | arXiv:1911.09516, 2019; preprint. Formal venue unverified. | ASFF-inspired spatial scale weights for already aligned pseudo features. |
| [EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020; conference | Fast normalized fusion operator only; not the complete bidirectional BiFPN network. |
| [Attentional Feature Fusion](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html) | IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021; conference | Paper Eq. 4 convex blend with nested three-branch fusion; omit the official code's x2 gain. |
| [ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection](https://doi.org/10.1109/TPAMI.2024.3417329) | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024; journal | MHSIU-inspired grouped attention/value transforms; not the original three-image encoder pipeline. |
| [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021; conference | Additional boundary metric using the authors' erosion convention; checked 2026-10-29. |

## Verification and existing experiments

The recorded code verification passed **281 tests and 67 subtests**, with CPU/GPU inference checks and zero optimizer steps. That initial check did not measure real-data convergence or rankings. The later uploaded priority8 report supplies three-dataset results; four-dataset completion and full-data retraining still require execution. See the [verification artifacts](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-28/README.md).

The original BRG main experiment remains in [COD](https://github.com/sh-min/COD); CoCoNet was excluded from this upload. Archived sources and their licenses remain intact. Historical organization details are in the [migration record](docs/MIGRATION.md), [experiment index](Experiments/README.md), [benchmark index](Benchmarks/README.md), and [research decisions](Progress/README.md).

Metric/leaderboard verification on 2026-10-29: **522 tests + 67 subtests**, operation coverage for all 158 initialized models, and eight browser checks passed. No training or downloads were run. [2026-10-29 verification](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-29-metrics/README.md).

The [2026-11-01 change record](Progress/2026-11-01/Benchmark-Completion/README.md) distinguishes code validation, uploaded historical results and pending server execution.
