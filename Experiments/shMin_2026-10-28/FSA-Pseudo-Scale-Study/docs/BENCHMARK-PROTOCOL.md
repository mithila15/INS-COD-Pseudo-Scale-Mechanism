# 2026-11-01 · Benchmark protocol — Complete test coverage and full-data training

**English** | [한국어](BENCHMARK-PROTOCOL.ko.md) | [Study guide](../README.md)

The uploaded [priority8 report](../results/priority8-report-20261101_172449/README.md) contains eight completed runs with seed 3407, evaluated on CAMO, CHAMELEON and NC4K. **COD10K-test is absent.** Completed execution of the original plan does not mean that all four benchmarks have been evaluated. Keep that report and its checkpoints as the historical three-dataset result.

Two separate actions address the data issues:

| Action | Reuse | New output | Training needed? |
| --- | --- | --- | --- |
| Complete four-dataset evaluation | Existing holdout-trained `best.pth` files and unchanged training split | New evaluation manifest, plan, metrics and leaderboard | No |
| Run `full_train` | Original 4,040 training-source images, backbone initialization and experiment settings | Fresh models and results in a separate root | Yes, from the original initialization |

## What differs from ZoomNeXt

The comparison below uses the official ZoomNeXt image-COD code at commit [`614af4348808734aaf2ec43937cf827410330b67`](https://github.com/lartpang/ZoomNeXt/tree/614af4348808734aaf2ec43937cf827410330b67), checked on 2026-11-01. The paper is **ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection**, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (**TPAMI, 2024; journal**): [published article](https://doi.org/10.1109/TPAMI.2024.3417329), [arXiv v3 §4.1](https://arxiv.org/html/2310.20208v3#S4.SS1). The journal article and arXiv version are distinct publication records.

| Setting | Existing priority8 study (`holdout`) | Official released ZoomNeXt image configuration |
| --- | --- | --- |
| Optimization images | CAMO 900 + COD10K 2,736 = **3,636** | CAMO 1,000 + COD10K 3,040 = **4,040** |
| Validation | CAMO 100 + COD10K 304 = **404** | No training holdout in the released training loop |
| Checkpoint | Minimum sample-weighted validation float MAE | Final epoch |
| Epochs / released batch size | 150 / 4 | 150 / 4 |
| Steps at these settings | 909/epoch; 136,350 total | 1,010/epoch; 151,500 total |
| Backbone and encoder inputs | PVTv2-B2; one 384×384 RGB pass, then pseudo features | PVTv2-B2; shared encoder applied to 192/384/576 RGB inputs |
| Decoder channels | 48 | 64 |
| Loss | Weighted BCE+IoU, deep supervision, region, edge Dice, cosine UAL | BCE and cosine UAL |
| Augmentation | Horizontal flip 0.5; vertical flip 0.2 | Horizontal flip 0.5; rotation up to ±90°, brightness/contrast and HSV, each with probability 0.5 |
| Training mask resize | PIL nearest, then binary mask | OpenCV bilinear |
| Precision | No AMP autocast/scaler in study training | `use_amp=True` |
| Adam | LR 1e-4; encoder LR ×0.1; weight decay 0 | Same released defaults |
| LR reduction | After epoch 100 | At iteration 101,000 with full-data batch 4 |
| Prediction processing | Per-image min-max; PyTorch bilinear resize to GT | Batch-wide min-max; OpenCV bilinear resize to GT |
| Recorded software | torch 2.11.0, torchvision 0.26.0, timm 1.0.30, PySODMetrics 1.6.2 | README: torch 2.1.2, torchvision 0.16.2; requirements: timm 0.9.12, PySODMetrics 1.4.2 |

The **paper states batch size 8 and NVIDIA 2080Ti**, while the released `icod_train.py` sets batch size 4. Do not label the released batch-4 schedule as the paper's batch-8 setting. Sources: [training configuration](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/configs/icod_train.py#L6-L60), [augmentation](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L70-L132), [AMP/training loop](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L202-L278), [final checkpoint](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L313-L327), [model/loss](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/methods/zoomnext/zoomnext.py), [evaluation](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/main_for_image.py#L137-L178), [installation](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/README.md#install-requirements) and [requirements](https://github.com/lartpang/ZoomNeXt/blob/614af4348808734aaf2ec43937cf827410330b67/requirements.txt).

The 3,636/404 split is supported by the study's split implementation and the reported validation counts and 909-step epoch logs. The uploaded report records a shared manifest hash; it does not contain the remote manifest's full records. The remote raw data were not inspected locally. The uploaded inference profile identifies an RTX PRO 5000 Blackwell; that is direct profiling evidence, not an independent audit of training hardware. The official pretrained weights were not downloaded to compare their bytes.

**`full_train` aligns the number of training images and fixed-final-epoch selection. It does not reproduce the entire ZoomNeXt recipe.** Architecture, losses, augmentation, precision, libraries and prediction processing remain as above. The removed 10% is a comparison confound, not a proven cause of any accuracy gap. A stronger published comparison should also run the official model under explicitly matched data and evaluation conditions.

## Transfer the updated source from Windows

Use **Windows PowerShell** (a prompt such as `PS C:\Users\User>`), not the Ubuntu `master@ubuntu` shell. The local parent folder is `C:\Users\User\OneDrive\Desktop\shMin`; set `$Repo` to the actual Git checkout inside it, such as the previously created `INS_COD-publish-<timestamp>` directory. It must be a clean `main` checkout of `sh-min/INS_COD`.

After an initial `git -C "$Repo" pull --ff-only origin main` has obtained the new script, run:

```powershell
& "$Repo\Experiments\shMin_2026-10-28\FSA-Pseudo-Scale-Study\deploy\send_study.ps1" -Repo $Repo
```

[`send_study.ps1`](../deploy/send_study.ps1) verifies origin/branch/clean state, pulls `main`, and archives the committed study **together with its required archived model source at `Experiments/FSA-ZoomNeXt` and `docs/MIGRATION.md`**. The model imports need that sibling source directory in the extracted release. It hashes the ZIP and sends ZIP + checksum to `master@192.168.1.200:/home/master/` using SCP. It prints the transferred revision and the new source destination `/home/master/mithila/INS_COD-releases/<revision12>`. It does not run experiments remotely or transfer local datasets, environments or checkpoint weights. The private server address must be reachable from the Windows PC; successful Parsec access alone does not establish that route.

Extract into the printed **new release directory**, verify the ZIP checksum, and point `STUDY` below at that release's `Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study` directory. Reuse the old environment, data and checkpoints. The old server checkout can retain its local architecture changes; it is not the extraction destination. Only the known `architecture=common` metadata compatibility is supported by this study; an actual PCD checkpoint requires its original implementation.

The transfer script was reviewed in the code-editing environment; Windows PowerShell, the private-network connection and the actual SCP transfer were not executed there. The user must perform those steps on their own machines.

## Require and verify all four test datasets

The `run_study.py prepare` CLI now requires all four by default:

| Dataset key | Standard test count |
| --- | ---: |
| `camo` | 250 |
| `chameleon` | 76 |
| `cod10k` | 2,026 foreground `COD10K-CAM-` images |
| `nc4k` | 4,121 |

An intentional subset remains possible with `--test-datasets camo chameleon nc4k`, or training-only preparation with `--test-datasets none`. These cannot be reported as complete four-benchmark results. Evaluation metadata and refreshed leaderboards expose `benchmark_coverage`, missing datasets and nonstandard counts. `execution_complete` and `benchmark_complete` describe different conditions. No missing metric is fabricated or used as a zero score.

### Re-evaluate completed priority8 checkpoints

Run the following in **Ubuntu Bash**, after installing the updated source. Use the same terminal for the blocks in this section. If using a transferred isolated source release, set `STUDY` to that release's absolute study directory; keep `PY` pointed at the existing environment. `--rebind-entrypoints` explicitly changes validated plan script paths to the current release while preserving the declared Python executable and other settings. It does not rewrite the original plan or install over the old checkout.

```bash
REPO="/home/master/mithila/INS_COD"
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study-clean/bin/python"
RUN="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-separate-20261030_204647"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
COMPLETE="$RUN/plans/four-benchmarks-$STAMP"

"$PY" "$STUDY/complete_benchmark.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --output "$COMPLETE" \
  --evaluation-subdir "test-four-benchmarks-$STAMP" \
  --rebind-entrypoints
```

This **only prepares and audits** a new evaluation manifest and plan. Every original run must have completed training. The helper verifies the original training manifest and checkpoint identities, preserves every original sample identity, split and file hash, and adds only COD10K test records. If new test files require a broader data root, paths are rebased with the same file identities and recorded provenance. It checks all four standard test counts and audits image/GT dimensions. It never edits checkpoint metadata to pretend that the checkpoint was trained with the new manifest.

COD10K test files must already exist. Unique recognized layouts below the original data root can be discovered automatically. If absent or ambiguous, add both `--cod10k-images /absolute/path/to/COD10K/Test/Image` and `--cod10k-masks /absolute/path/to/COD10K/Test/GT_Object` to the preparation command. These are examples, not verified server paths. There is no automatic download, and count/filename checks must not be bypassed to substitute other images.

The next block **executes evaluation only**, reusing the old models and recomputing **all four datasets with the current common evaluation code**, so the new report does not silently mix old and new evaluators. It performs no training or profiling. The original three-dataset outputs remain in place.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$COMPLETE/run_evaluation.sh" 2>&1 | tee "$COMPLETE/evaluation.log"
```

New metrics: `$RUN/runs/priority8/<variant>__seed3407/test-four-benchmarks-$STAMP/metrics.json`. New leaderboard: `$COMPLETE/leaderboard/leaderboard.html` or `leaderboard.md`. Each dataset retains all 21 scalar metrics, curves and per-image outputs. The recovery plan rejects train/profile actions, even if invoked manually through `run_plan.py`.

## Train a separate full-data comparison

`holdout` remains the default for method development. Its 3,636/404 split and validation-selected `best.pth` behavior are unchanged. The explicit `full_train` protocol uses CAMO 1,000 + COD10K 3,040 and **no validation split**. Its checkpoint selection is declared in advance as **fixed final epoch**. For filename compatibility, `best.pth` contains that final-epoch model; it is not a test-selected or validation-selected best checkpoint. `best_val_mae` is null, no validation loader runs, and there is no validation ranking or early stopping for this protocol.

After preparing `$COMPLETE/manifest.json` above, the following **creates a new full-data plan** from the original eight jobs. It promotes only the original training holdout records to train, preserves all test records, and keeps the old seeds, 150 epochs, batch size 4, backbone initialization, variants and other declared training settings. It reuses no trained checkpoint. The full-data plan can be prepared before the four-dataset evaluation has finished; evaluation files are not its input.

```bash
FULL="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-$STAMP"

"$PY" "$STUDY/prepare_full_training.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --manifest "$COMPLETE/manifest.json" \
  --output "$FULL" \
  --rebind-entrypoints
```

Review `$FULL/plans/priority8/plan.json`, including the recorded sample counts and settings audit. In the **same Ubuntu terminal**, the following **starts fresh training → test evaluation → profiling** for those eight jobs. It will take a full new training schedule; it does not resume the previous 3,636-image models.

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$FULL/plans/priority8/run_all.sh" 2>&1 | tee "$FULL/run.log"
```

Outputs are `$FULL/runs/priority8/<variant>__seed3407/` and `$FULL/plans/priority8/leaderboard/`. `train.sh` at the same plan location runs training only. Do not launch old-model evaluation and new training concurrently on the same GPU unless resources are deliberately scheduled.

For a new study rather than migration of completed runs, pass `--training-protocol full_train` to **both** `run_study.py prepare` and `plan_study.py` (omit `--val-fraction`, or use `0`). The new `--phase priority8 --seeds 3407` selects, in order: `single`, `native_conv__asff`, `native_conv__sk`, `native_conv__mean`, `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean`. It excludes `legacy_full`; the other matrix phases retain their original controls.

## Interpretation and validation limits

Preserve training-source, manifest, metric and profiling provenance. A holdout run and a full-data run must not share one ranking cohort. Full-data test tables are descriptive; do not choose an epoch or tune repeated variants on those test scores. A single seed cannot establish cross-seed stability or statistical significance. The `projection` control has no spatial resampling, so a gain from that control is not evidence that pseudo-scale generation caused the gain.

The final regression suite passed **694 tests and 74 subtests**, and eight browser checks passed. These are synthetic/inference/gradient checks with no optimizer updates; they do not measure the new model's accuracy. [Verification logs and scope](../results/verification-2026-11-01-benchmark/README.md).

No new real-data training or server benchmark evaluation was run by the code-editing task. Server paths, raw images, checkpoints, server CUDA evaluation on real data and final four-dataset results still need the user's execution. The documented environment is reused, not reinstalled. [Change record](../../../../Progress/2026-11-01/Benchmark-Completion/README.md).
