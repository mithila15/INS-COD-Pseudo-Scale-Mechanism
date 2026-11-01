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

[`send_study.ps1`](../deploy/send_study.ps1) verifies origin/branch/clean state, pulls `main`, and archives committed study source **together with the required archived model source at `Experiments/FSA-ZoomNeXt` and `docs/MIGRATION.md`**. It hashes the ZIP and sends ZIP + checksum to `master@192.168.1.200:/home/master/` using SCP. It prints the transferred revision and active server source directory under `/home/master/mithila/shMin`. Transfer does not install the code or start experiments. Local datasets, environments and checkpoint weights are excluded. The server address must be reachable from Windows; Parsec access alone does not establish that route.

After transfer, use **Ubuntu Bash**. Replace `VERSION` with the printed 12-character revision. This block verifies the checksum, extracts into a fresh temporary staging directory and runs the packaged [source installer](../deploy/install_study.py). It does not reuse an existing `INS_COD-releases/<revision>` directory, so that directory's presence causes no `mkdir` conflict.

```bash
VERSION="REPLACE_WITH_TRANSFERRED_REVISION12"
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
WORK="/home/master/mithila/shMin"
STUDY="$WORK/$REL"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
ARCHIVE="/home/master/ins-cod-study-$VERSION.zip"

(
set -e
cd /home/master
sha256sum -c "ins-cod-study-$VERSION.zip.sha256"
STAGING=$(mktemp -d /home/master/ins-cod-install.XXXXXX)
"$PY" -m zipfile -e "$ARCHIVE" "$STAGING"
"$PY" "$STAGING/$REL/deploy/install_study.py" \
  --archive "$ARCHIVE" --work "$WORK"
)
```

The installer validates the checksum and archive commit, saves replaced code/document files under `$WORK/.code-backups/<revision>-<unique>/`, then installs source under `shMin`. Existing results, datasets, checkpoints, virtual environments and extra local PCD files are preserved. It refuses installation while recognized study processes are active. The temporary staging copy is separate from the active source. The old `INS_COD` checkout remains available for its environment, data and recorded provenance. Only `architecture=common` checkpoint metadata compatibility is supported; an actual PCD checkpoint still requires its original model implementation.

The user previously confirmed a successful Windows/SCP transfer and backup-preserving source installation. The packaged installer has local synthetic filesystem checks; the code-editing task did not run the new installer against the user's remote filesystem. Installation performs no evaluation, training or dependency installation.

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

Run the following in **Ubuntu Bash**, after installing the updated source. The default below uses the user-consolidated active `shMin` source and reuses Python from the old `INS_COD` environment. This default command keeps strict overlap checks and stops on the recorded seven pairs; use the [explicit acknowledgment section](#acknowledge-the-verified-cod10k-overlaps) in that case. Use the same terminal for the blocks in this section. If using a transferred isolated source release, set `STUDY` to that release's absolute study directory; keep `PY` pointed at the existing environment. `--rebind-entrypoints` explicitly changes validated plan script paths to the current release while preserving the declared Python executable and other settings. It does not rewrite the original plan or install over the old checkout.

```bash
REPO="/home/master/mithila/INS_COD"
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
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

## Acknowledge the verified COD10K overlaps

The default remains strict: preparation rejects identical image bytes or resolved paths crossing splits. The reported `Identical image paths or byte hashes cross train/val/test splits` error occurred after finding all 2,026 COD10K image/GT pairs. The server audit found **seven byte-identical train-source/test pairs**, with **six in the optimization split and one in the validation split**. None shared the same resolved path. The code installation succeeded; the failed preparation did not begin evaluation or training.

Two evidence levels must stay separate:

- The COD10K/SINet-V2 authors explicitly acknowledge **Crab-32** and **Cat-1506** in their [2021-10-09 notice](https://github.com/GewelsJI/SINet-V2/blob/5b9844c94d10a03b815c7c7039c194166a9a8ba3/README.md#L44).
- All seven exact name/hash pairs also appear in the independent COD-TDQ dataset bundle at pinned revision `c4e97371fa835cd0a778ba67c071f24ca5deaf11`. Only its small Git-LFS metadata pointers were fetched. This corroborates published copies of the remaining five pairs; it is **not an official COD10K acknowledgment of all seven**. [Pinned URLs and hash evidence](../results/verification-2026-11-01-overlap/upstream-evidence.json), [server audit summary](../results/verification-2026-11-01-overlap/server-audit-summary.json).

The explicit `--acknowledge-cod10k-overlap` option permits only the seven catalogued COD10K **ID pairs and image SHA256 values**. Different pairs/hashes, shared resolved paths, other dataset overlap and train/validation overlap remain errors. This is not a general `ignore-overlap` switch. Normal file-hash, pairing, count, geometry, manifest and checkpoint checks remain active. No input image, mask, old manifest, checkpoint or historical report is changed.

For the audited server copy, after installing the corrected source under `shMin`, use this **Ubuntu Bash** block instead of the strict preparation block above. It creates a fresh evaluation plan only. Run the subsequent blocks in the **same terminal**.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
RUN="$STUDY/results/priority8-separate-20261030_204647"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
COMPLETE="$RUN/plans/four-benchmarks-ack-$STAMP"

"$PY" "$STUDY/complete_benchmark.py" \
  --plan "$RUN/plans/priority8/plan.json" \
  --output "$COMPLETE" \
  --evaluation-subdir "test-four-benchmarks-ack-$STAMP" \
  --rebind-entrypoints \
  --acknowledge-cod10k-overlap
```

Inspect the new manifest's `overlap_policy="acknowledged-cod10k-v1"` and `overlap_audit`. For the recorded split, expect `disjoint=false`, `train_test_count=6`, `val_test_count=1`, `affected_test_count=7`, and the exact groups. Evaluation metadata and refreshed reports carry this disclosure. Standard dataset counts and completed execution do **not** establish disjoint data.

This separate block starts **evaluation only**, using the existing trained checkpoints:

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$COMPLETE/run_evaluation.sh" 2>&1 | tee "$COMPLETE/evaluation.log"
```

Metrics go to `$RUN/runs/priority8/<variant>__seed3407/test-four-benchmarks-ack-$STAMP/metrics.json`; the leaderboard goes to `$COMPLETE/leaderboard/`. The following full-data preparation section can reuse this manifest: it inherits the policy and recomputes the audit after promoting validation to train, giving **seven train/test pairs and zero validation/test pairs** for this copy. A fresh `run_study.py prepare` command also supports the explicit acknowledgement option when intentionally retaining these published sample counts.

This option retains the published 4,040/2,026 counts; it does **not** remove exposure or make COD10K-test independent. Existing models already used six duplicate images for optimization and one for validation/checkpoint selection. Acknowledgment cannot undo that history. The earlier CAMO/CHAMELEON/NC4K scores are not invalidated by this specific COD10K finding, but this audit is not a guarantee against every form of near-duplicate leakage. The upstream notice's approximate metric effect for two images must not be extrapolated to all seven or to this study.

A different, disjoint protocol would need separately named outputs and comparisons: removing seven test records yields a **2,019-image subset**, while removing their seven training-source counterparts before fresh training yields **4,033 training images** with the full test set. Neither is the standard 4,040/2,026 setting, and neither is silently performed by this option. [Decision and evidence record](../../../../Progress/2026-11-01/COD10K-Overlap/README.md).

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
