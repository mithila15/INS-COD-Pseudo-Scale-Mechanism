# 2026-11-05 · Pseudo-Scale Mechanism — Small Controlled Experiments

**English** | [한국어](SCALE-MECHANISM.ko.md) · [Repository](../../../../README.md) · [Verification](../results/verification-2026-11-04-mechanism/README.md)

Branch: **`research/pseudo-scale-mechanism`**, based on `1a003b9a7becddc0fd844bb85de62f005b96a9f0`. This branch tests whether changing feature scale adds value beyond extra learned branches. It adds an actual RGB-scale reference, a denser final feature, optional anchored fusion, and optional frozen feature supervision. These are hypotheses to measure; there are no newly trained results yet.

## 1. Start with four cases

| Order | Case | Encoder passes per image at inference | Question |
| ---: | --- | ---: | --- |
| 1 | `single` | 1 | ConvNeXt final-stage baseline without auxiliary branches |
| 2 | `projection__mean` | 1 | Do two learned branches at unchanged resolution help? |
| 3 | `native_conv__mean` | 1 | Do the same branch operators help more when applied at 1.5× / 0.5× feature resolution? |
| 4 | `image_pyramid__mean` | 3 | What does actual 1× / 1.5× / 0.5× RGB encoding achieve with this same final-stage decoder? |

All four use ConvNeXt-Tiny, only the final stage, seed 3407, 150 epochs, batch 4, 384×384 RGB, 48 projected channels, the same local ImageNet initialization, and the existing losses/optimizer/augmentation. The projection and native-convolution controls have the same branch parameters and initialization, but different intermediate spatial operations and costs. The reference shares one encoder/projection across three RGB inputs, aligns the features to the 1× grid and averages them. It does not contain a pseudo generator. This is a controlled reference, **not a reproduction of the complete ZoomNeXt architecture**.

The default `core` command schedules **four runs only**. It never schedules the later stages. The earlier priority8 launcher and historical catalog remain available separately.

| Explicit stage | Runs | Final output stride | Purpose |
| --- | ---: | ---: | --- |
| `core` | 4 | 32 | Baseline, capacity control, pseudo scale, actual RGB scales |
| `resolution` | 4 | 16 | Repeat those four while retaining a denser final feature |
| `fusion` | 2 | 32 by default | Projection/native branches with `anchor_gated`, compared with mean in core |
| `distill` | 2 | 32 by default | Projection/native branches with mean and frozen real-scale feature targets |

## 2. Architecture and supervision changes

**Stride 16.** All ConvNeXt stages and pretrained parameter tensors remain. Only the final 2×2 downsampling convolution changes from stride 2 to stride 1, preceded by right/bottom replicate padding. At 384×384, the last 768-channel map becomes 24×24 instead of 12×12. This is our adaptation, not the official classification architecture. It changes sampling, receptive fields and boundary behavior; it adds no dilation or early-stage skips. The decoder keeps five identical convolution blocks; at stride 16 the first preserves spatial size and the following four upsample.

**Anchored fusion.** For aligned main/large/small features `M, L, S` of shape `B,C,H,W`, the implementation is:

```python
aux = projection_1x1(cat([L, S], dim=1))  # 2C -> C, no bias
output = M + sigmoid(channel_gate) * aux
```

The learned per-channel gate starts at 0.01 and is shared across images and positions. It is not an input-dependent attention gate, and no subtraction is required. This is a study-specific module, with an explicit direct path from the main feature. Whether it helps must be measured against `mean` for both the capacity control and pseudo generator.

**Optional feature distillation.** First train the actual RGB-scale reference in core. The later distill stage starts fresh students from the same ImageNet initialization and reads the reference's fixed-final-epoch `best.pth`. The frozen teacher encodes the same augmented training image at 1×, 1.5× and 0.5×. Its projected auxiliary features are aligned to the main grid. Each corresponding student auxiliary feature receives channel-vector cosine-distance supervision, averaged over pixels, batch and the two branches, with coefficient **0.1**. The main feature has no feature-matching loss.

The teacher is in evaluation mode with gradients disabled. It is external to the student's optimizer and state dictionary; student evaluation/profiling reconstructs only the student. Training does cost more: one student encoder pass plus three frozen teacher passes per image, and the teacher must first be trained. The claimed saving concerns **student inference**, not total research/training cost. Teacher checkpoint SHA256, model configuration, manifest hash, epoch and loss weight are recorded. The loader rejects a non-final checkpoint, a different manifest/backbone/stride/channel count/input size, or a non-full-train teacher. Retain the teacher file to rerun plan identity checks; standalone student evaluation does not need it.

**No pseudo-mask labels are generated.** Dataset masks still supervise segmentation, deep and region outputs; derived mask edges supervise the edge head. Feature targets are tensors, not predicted labels. [Detailed supervision account](PSEUDO-LABELS.md).

## 3. Environment, inputs and separate source installation

Reuse the server environment:

```text
/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python
```

It uses the existing pinned Python 3.12.13 / torch 2.11.0 / torchvision 0.26.0 profile. No new dependency is added. For a new environment follow the [English environment guide](../environment/README.md). The existing full4040 manifest and dataset files are reused from `/home/master/mithila/shMin`; no dataset download or split rewrite is performed.

The manifest is `results/priority8-full4040-fixed-20261101_200834/manifest.json` inside the old study. Training is CAMO 1,000 + COD10K 3,040; test counts are CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121. The acknowledged seven COD10K cross-split byte duplicates and two CHAMELEON geometry mismatches remain recorded. This update does not establish independent train/test data. [Protocol and limitations](BENCHMARK-PROTOCOL.md).

Supply the local official TorchVision ConvNeXt-Tiny file `weights/convnext_tiny-983f1562.pth` in the old study (about 109 MB). PVT weights are incompatible. If absent, use the **user-run** download/hash-check block in [CNN initialization](CNN-BACKBONES.md#local-pretrained-initialization). Model construction never downloads it.

### Windows PowerShell: clone this branch and send it

Run only in **Windows PowerShell**, not the Ubuntu Bash terminal. This uses a new checkout, leaving an existing main checkout separate. If that exact folder already exists, run the second block from that checkout after verifying it belongs to this branch.

```powershell
$Base = "C:\Users\User\OneDrive\Desktop\shMin"
$Repo = Join-Path $Base "INS_COD-scale-study"
New-Item -ItemType Directory -Force -Path $Base | Out-Null
git clone -c core.longpaths=true --single-branch --branch research/pseudo-scale-mechanism "https://github.com/sh-min/INS_COD.git" "$Repo"
if ($LASTEXITCODE -ne 0) { throw "Clone failed; stop here." }
```

In the **same PowerShell**:

```powershell
$Script = Join-Path $Repo "Experiments\shMin_2026-10-28\FSA-Pseudo-Scale-Study\deploy\send_study.ps1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& "$Script" -Repo "$Repo" -Branch "research/pseudo-scale-mechanism"
```

This pulls the named branch and transfers its committed ZIP plus SHA256 through SCP. It prints `Version for server installation: <12-character revision>`. No experiment starts.

### Ubuntu Bash: install into a separate source directory

Replace the value of `VERSION` with the exact 12-character version printed by the transfer. Run the entire block; a failed checksum stops installation.

```bash
(
set -e
VERSION="REPLACE_WITH_TRANSFERRED_12_CHARACTER_VERSION"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
ARCHIVE="/home/master/ins-cod-study-$VERSION.zip"
cd /home/master
sha256sum -c "$ARCHIVE.sha256"
mkdir -p /home/master/mithila/INS_COD-releases
RELEASE=$(mktemp -d "/home/master/mithila/INS_COD-releases/scale-$VERSION-XXXXXX")
"$PY" -m zipfile -e "$ARCHIVE" "$RELEASE"
"$PY" "$RELEASE/$REL/deploy/install_study.py" \
  --archive "$ARCHIVE" --work /home/master/mithila/shMin-scale-study
)
```

The installer preserves data, results, checkpoints and environments, and backs up replaced source files. It refuses to replace files used by a running study in the destination. The new active source is `/home/master/mithila/shMin-scale-study`; results below still go into the existing `shMin` experiment tree.

### Recover an already installed CRLF launcher

The 2026-11-05 server report reached `Manifest and backbone weights ready.` and then stopped with Bash `invalid option name`, before a new tmux session or training process existed. A controlled `core.autocrlf=true` export reproduces this error: `pipefail\r` is interpreted as an unknown shell option. Raw server bytes have not been read here; CRLF is the diagnosis supported by that reproduction.

The corrected sender disables archive EOL conversion, and `.gitattributes` fixes shell scripts to LF. The installer also converts **source `.sh` files only** from CRLF to LF after validating the original ZIP checksum, records archive/installed hashes in `normalizations`, and preserves original files in its backup. Other source bytes and preserved input/result/checkpoint paths are unchanged. [Git attribute rules](https://git-scm.com/docs/gitattributes). [Reproduction and checks](../results/verification-2026-11-05-deployment/README.md).

For the reported state (installed source and verified weights, **no active `scale-core` session/training**), run the complete block below in **Ubuntu Bash**. It backs up and repairs the existing launcher, creates a new four-case plan, starts training/evaluation/profiling in a detached tmux shell and attaches to it. It reuses the existing environment, manifest and weights; there is no download. Errors after training starts remain visible in the session.

```bash
(
set -e
REL="Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
STUDY="/home/master/mithila/shMin-scale-study/$REL"
EXISTING="/home/master/mithila/shMin/$REL"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
export MANIFEST="$EXISTING/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$EXISTING/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)

cp -p "$STUDY/launch_scale_mechanism.sh" "$STUDY/launch_scale_mechanism.sh.crlf-backup-$STAMP"
sed -i 's/\r$//' "$STUDY/launch_scale_mechanism.sh"
bash -n "$STUDY/launch_scale_mechanism.sh"

OUT="$EXISTING/results/scale-mechanism-core-os32-seed3407-$STAMP"
DEVICE=cuda:0 OUTPUT="$OUT" \
  bash "$STUDY/launch_scale_mechanism.sh" --prepare-only core "$WEIGHTS"

tmux new-session -d -s scale-core -c /home/master /bin/bash
printf -v START_CMD '%q ' env PYTHONUNBUFFERED=1 bash "$OUT/plans/core/run_all.sh"
printf -v LOG_ARG '%q' "$OUT/run.log"
tmux send-keys -t scale-core \
  'set +e' C-m 'set -o pipefail' C-m \
  "$START_CMD 2>&1 | tee $LOG_ARG" C-m
tmux attach -t scale-core
)
```

The new result path is printed as `Result root:`; its leaderboard is `plans/core/leaderboard/leaderboard.html`. The actual user-run scope remains four cases, seed 3407, 150 epochs each. This repair does not implement partial optimizer-state resume. For subsequent installations use the corrected branch installer; the old `bb6f040` ZIP's embedded installer does not include the new normalization.

## 4. Ubuntu Bash: start only the four core runs

This block explicitly starts **training → four-dataset evaluation → inference profiling → leaderboard updates** for the four cases. Run it yourself after installation, dataset availability and weight verification:

```bash
STUDY="/home/master/mithila/shMin-scale-study/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
EXISTING="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
export MANIFEST="$EXISTING/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$EXISTING/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
CORE="$EXISTING/results/scale-mechanism-core-os32-seed3407-$STAMP"

OUTPUT="$CORE" bash "$STUDY/launch_scale_mechanism.sh" core "$WEIGHTS"
```

To prepare commands only, insert `--prepare-only` before `core`. Then start the printed `run_all.sh` later. Do not invoke the launcher again with the same output: it requires a new directory. The generated runner skips completed matching stages and rejects incomplete outputs; **optimizer-state resume is not implemented**. After an interruption, preserve that directory and inspect the stage before creating another run. None of these commands resume old PVT checkpoints.

There is **no early stopping** in this full-data protocol. Selection is fixed at epoch 150; `best.pth` is that final checkpoint. Epoch logs have `validation: null`. Full test metrics arrive after each model finishes training.

## 5. Run later stages separately

The blocks in this section reuse `STUDY, EXISTING, PY, MANIFEST, WEIGHTS` from section 4 in the **same Bash terminal**. Each invocation below starts only its named stage. They are alternatives to run deliberately, not a script to paste as one batch.

**Resolution (four fresh runs):**

```bash
OUTPUT="$EXISTING/results/scale-mechanism-resolution-os16-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" resolution "$WEIGHTS"
```

**Anchored fusion (two fresh runs, compared with the mean controls in core):**

```bash
OUTPUT_STRIDE=32 OUTPUT="$EXISTING/results/scale-mechanism-fusion-os32-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" fusion "$WEIGHTS"
```

**Feature distillation (two fresh runs; only after core's fourth model has completed training):**

```bash
TEACHER="$CORE/runs/core/image_pyramid__mean__seed3407/best.pth"
OUTPUT_STRIDE=32 OUTPUT="$EXISTING/results/scale-mechanism-distill-os32-seed3407-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" \
  bash "$STUDY/launch_scale_mechanism.sh" distill "$WEIGHTS" "$TEACHER"
```

If the terminal was closed, redefine `CORE` as the **exact completed result directory**, rather than guessing the newest folder. For stride-16 fusion/distillation use `OUTPUT_STRIDE=16` and an `os16` output name; the teacher must then be the completed stride-16 reference at `<resolution-result-root>/runs/resolution/image_pyramid__mean__seed3407/best.pth`. The loader rejects a stride mismatch.

## 6. Results and decisions

For core, results are under `$CORE`:

| File | Meaning |
| --- | --- |
| `run.log` | Combined live stage log |
| `runs/core/<case>__seed3407/history.jsonl` | Per-epoch losses, including weighted `distill` in that later stage |
| `runs/core/<case>__seed3407/test/metrics.json` | 21 scalar metrics per dataset, including Sα (`S`), weighted Fβ (`wF`), MAE, E/Fβ variants, IoU/Dice and boundary metrics |
| `runs/core/<case>__seed3407/profile.json` | Inference parameters, MACs/FLOPs, latency, memory and branch diagnostics |
| `plans/core/leaderboard/leaderboard.md` | Terminal-readable tables |
| `plans/core/leaderboard/leaderboard.html` | Browser tables |

Other stages use their stage name instead of `core`. To view the core report in the same terminal:

```bash
less "$CORE/plans/core/leaderboard/leaderboard.md"
```

All four core variants share the segmentation comparison cohort. Output stride and teacher configuration separate later cohorts; the report does not silently average changed architectures or training objectives. Compare the paired conditions explicitly across stage reports. Costs are measured on the complete inference graph: three encodings for the reference and one for a student, with no teacher included. The operation estimate classifies padding as memory movement, not floating arithmetic; measured latency still includes it.

First compare `native_conv__mean` with both `single` and `projection__mean`. If it only beats single, extra branch capacity may explain the gain. If the actual-scale reference does not help this backbone/decoder, reconsider the chosen feature/decoder and scale range before increasing the sweep. Stride 16 tests retained spatial detail; anchored fusion tests whether auxiliary branches disturb the main feature; distillation tests explicit real-scale supervision. A distillation improvement alone does not prove a scale-specific effect—its projection control is necessary. One seed is exploratory; report all datasets, trade-offs and later repeated seeds before a robust claim. Teacher alignment losses are diagnostics, not proof of semantic equivalence or recovery of discarded image detail.

## 7. References and adaptation boundaries

Checked **2026-11-04**. Korean titles are explanatory translations.

- **A ConvNet for the 2020s / 2020년대를 위한 합성곱 신경망.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), **2022**, conference. [Official paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), [author preprint](https://arxiv.org/abs/2201.03545), [TorchVision 0.26.0 implementation](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py). Used for the backbone. The stride-16 padding/stride edit and segmentation head are this study's adaptations.
- **ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection / ZoomNeXt: 위장 객체 검출을 위한 통합 협력 피라미드 네트워크.** **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, **2024**, journal. [Publication DOI](https://doi.org/10.1109/TPAMI.2024.3417329), [author paper v4](https://arxiv.org/abs/2310.20208v4). Motivates using resized RGB inputs as a real-scale reference. Our shared ConvNeXt/final-stage/mean reference does not reproduce ZoomNeXt's MHSIU/RGPU model.
- **FitNets: Hints for Thin Deep Nets / FitNets: 얇고 깊은 신경망을 위한 힌트.** Read version: **arXiv:1412.6550v4 (2015)**, preprint; formal venue not independently verified here. [Author paper](https://arxiv.org/abs/1412.6550v4). Motivates intermediate teacher representations as supervision. The branch-wise cosine loss, scale targets, coefficient and training protocol here are our design, not a reproduction of FitNets' training procedure.

The `anchor_gated` formula is a study proposal, not an implementation attributed to a named fusion paper. Existing generation/fusion papers and adaptation notes remain in the [study bibliography](../README.md#referenced-papers), [generation guide](GENERATION.md), and [utilization guide](UTILIZATION.md). [Implementation/change record](../../../../Progress/2026-11-04/Pseudo-Scale-Mechanism/README.md).
