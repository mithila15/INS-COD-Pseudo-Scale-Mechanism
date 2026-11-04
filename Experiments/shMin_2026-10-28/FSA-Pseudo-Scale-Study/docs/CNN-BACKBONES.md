# 2026-11-04 · CNN Backbone Study — ConvNeXt Priority8 Before Further Comparisons

**English** | [한국어](CNN-BACKBONES.ko.md) · [Study guide](../README.md) · [Supervision and pseudo-labels](PSEUDO-LABELS.md)

This preserved priority8 protocol reuses the previous **priority8 shortlist with ConvNeXt-Tiny, the final encoder stage, and one seed (`3407`)**. It schedules exactly eight runs: the no-pseudo baseline, `native_conv + asff/sk/mean`, `projection + mean`, and `dysample + asff/sk/mean`. This is the user-selected shortlist; it does not schedule the full generator–utilizer sweep, additional seeds, or ResNet training. The uploaded full4040 snapshot supplies the historical settings, not the live state of the remote server.

**Research-branch update:** the new [mechanism guide](SCALE-MECHANISM.md) starts four controls and adds an actual RGB-scale reference, stride-16 adaptation and optional feature teacher. The eight-case command below remains the separate original priority8 experiment.

## What the backbone change tests

There are three different meanings of scale in this study:

| Meaning | Computation | Present in the ConvNeXt shortlist? |
| --- | --- | --- |
| Multiple input scales | Resize an RGB image several ways and run the encoder on every image | No; one encoder invocation per input |
| Multiple encoder stages | Feed features from several encoder depths/resolutions into a decoder | No; `--feature-mode last` consumes only the final stage |
| Pseudo-scale branches | Transform one encoder feature at several intermediate resolutions and align the branches for fusion | `native_conv` and `dysample`; `projection` is a same-resolution learned-branch control, and the baseline bypasses the bank |

PVTv2, ConvNeXt, and ResNet are all hierarchical backbones. **Changing PVTv2 to a CNN does not itself remove multi-stage features.** The distinction comes from selecting which encoder outputs the segmentation head consumes. Earlier backbone layers still execute to compute the final feature; `last` removes their direct skip connections to the head. See the [ConvNeXt paper](https://arxiv.org/abs/2201.03545), [versioned ConvNeXt implementation](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py), and [PVTv2 paper](https://arxiv.org/abs/2106.13797).

The research hypothesis is that feature-level scale transformations can provide useful scale-dependent information with less cost than repeated image encoding. A feature resized after encoding is not generally equal to a feature obtained by encoding a resized image. The priority8 loss below supervises segmentation and does **not** match teacher features. The research branch offers a separate optional distillation stage. Similarity to true multi-scale features and an efficiency advantage remain measurements to establish.

## Implemented backbone and feature interfaces

| `--backbone` | Four stage channel counts | Final feature used by `--feature-mode last` |
| --- | --- | --- |
| `pvt_v2_b2` | 64, 128, 320, 512 | `reduction_5`, 512 channels, stride 32 |
| `convnext_tiny` | 96, 192, 384, 768 | TorchVision `features.7`, 768 channels, stride 32 |
| `resnet50` | 256, 512, 1024, 2048 | TorchVision `layer4`, 2048 channels, stride 32 |

The four stages have nominal strides 4, 8, 16, and 32. TorchVision's ResNet50 uses its documented V1.5 bottleneck stride placement; it is not a claim to reproduce the original Caffe implementation. [ResNet50 implementation and variant](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

`--feature-mode multistage` retains the existing four-stage projection, prior, and top-down refinement design. It is available for explicit controls. The historical default `pvt_v2_b2 + multistage` and old checkpoints remain supported. The archived `--legacy-reference` is only valid for that historical backbone and mode.

`--feature-mode last` has this data flow, shown for the `native_conv + sk` member of priority8:

```text
RGB image (384 × 384)
  → ImageNet channel normalization
  → one ConvNeXt-Tiny encoder pass
  → final feature (768 × 12 × 12)
  → 1 × 1 convolution + BatchNorm + ReLU (48 × 12 × 12)
  → native_conv: main / 1.5× branch / 0.5× branch
  → align all three to 48 × 12 × 12
  → SK channel-wise branch selection
  → region and edge auxiliary heads
  → region sigmoid gate + 3 × 3 depthwise-separable refinement
  → one deep-supervision head at stride 32
  → five [bilinear 2× upsample + 3 × 3 depthwise-separable convolution] blocks
  → 1 × 1 mask predictor → input-size logits
```

Each region/edge head uses a 1×1 convolution block, two 3×3 depthwise-separable blocks, and a 1×1 logit layer. The final-stage decoder has no ASPP, parallel mixed-kernel refinement, or earlier encoder skips. It also does not fabricate four independent encoder stages from repeated copies of the last feature. The loss still uses the segmentation, deep, region, edge, and UAL terms; the deep loss averages the available heads, so its coefficient remains `0.25` with one head.

At 384×384 input, the `native_conv` intermediate spatial sizes are 18×18 and 6×6 before both return to 12×12. Enlargement does not recover details that the encoder discarded. SK chooses branch weights per channel from global context; it does not compute a subtraction residual. The full definitions and existing paper references remain in [generation](GENERATION.md) and [utilization](UTILIZATION.md).

## Local pretrained initialization

Use the existing [locked study environment](../environment/README.md), which already includes TorchVision `0.26.0`; no additional package is required for these adapters. Supply a **local TorchVision-format checkpoint**. The code constructs CNNs with `weights=None` and never downloads weights implicitly.

| Intended initialization | Official file | Size reported by TorchVision |
| --- | --- | --- |
| ConvNeXt-Tiny `IMAGENET1K_V1` | [convnext_tiny-983f1562.pth](https://download.pytorch.org/models/convnext_tiny-983f1562.pth) | about 109.1 MB |
| Future ResNet50 `IMAGENET1K_V2` | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | about 97.8 MB |

The URLs were checked against the installed TorchVision 0.26.0 weight enums. Obtain the file yourself if it is absent and record its SHA256; the training config also records the supplied file's hash. These are TorchVision's ImageNet recipes, not identical reproductions of the papers' original pretraining. [ConvNeXt weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), [ResNet weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

The loader accepts a flat state dictionary or `{"state_dict": ...}`. It removes only the known classification-head parameters and loads the remaining encoder state strictly. ConvNeXt excludes `classifier.0.{weight,bias}` and `classifier.2.{weight,bias}`; ResNet excludes `fc.{weight,bias}`; PVT excludes `head.{weight,bias}`. Missing feature weights, extra unrelated keys, mismatched tensor shapes, timm/Meta ConvNeXt layouts, and a `{"model": ...}` wrapper are not silently accepted. `--random-init` is an explicit separate initialization condition, not a fallback for a missing pretrained file.

Run this optional block yourself **only if the matching weights are not already available**. It downloads the approximately 109 MB model file, verifies the official filename hash prefix and prints its full SHA256. It does not download datasets or start training. Existing files are checked, not overwritten.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
"$PY" - "$STUDY/weights/convnext_tiny-983f1562.pth" <<'WEIGHT_PY'
import hashlib
from pathlib import Path
import sys
from torch.hub import download_url_to_file
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    download_url_to_file("https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
                         str(path), hash_prefix="983f1562", progress=True)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if not digest.startswith("983f1562"):
    raise SystemExit("Existing weight file does not match the official SHA256 prefix; preserved unchanged.")
print(f"Verified local weights: {path}\nSHA256: {digest}")
WEIGHT_PY
```

## Prepare and execute exactly eight cases

Run the following in the **Linux experiment server's Bash terminal after installing the updated study source**. It reuses the existing full4040 manifest and the same data files. The manifest retains the previously acknowledged COD10K overlaps; this step does not edit or relax its audit. Its test split contains CAMO 250, CHAMELEON 76, COD10K 2,026, and NC4K 4,121 images. The existing `resize-prediction-to-gt` policy remains explicit for the two recorded CHAMELEON geometry mismatches.

This first block creates commands only. It does not train, download, or evaluate. The selected backbone and local weight file must match.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
MANIFEST="$STUDY/results/priority8-full4040-fixed-20261101_200834/manifest.json"
WEIGHTS="$STUDY/weights/convnext_tiny-983f1562.pth"
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
OUT="$STUDY/results/convnext-last-priority8-seed3407-$STAMP"

test -s "$MANIFEST" && test -s "$WEIGHTS" &&
"$PY" "$STUDY/plan_study.py" \
  --phase priority8 --seeds 3407 \
  --backbone convnext_tiny --feature-mode last \
  --training-protocol full_train --manifest "$MANIFEST" \
  --backbone-weights "$WEIGHTS" \
  --epochs 150 --batch-size 4 --size 384 --channels 48 --workers 2 \
  --device cuda:0 --python "$PY" --test-geometry resize-prediction-to-gt \
  --output "$OUT/plans/priority8" --results "$OUT/runs/priority8"
```

Confirm that the printed `variant_count` and `run_count` are both **8**, and `seeds` contains only `3407`. Then, in the **same terminal**, the following block runs those eight jobs sequentially. For each job it trains, evaluates the final checkpoint with all existing metrics, profiles it, and updates the leaderboard:

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash "$OUT/plans/priority8/run_all.sh" 2>&1 | tee "$OUT/run.log"
```

Each variant has its own directory under `$OUT/runs/priority8/`, such as `single__seed3407` or `native_conv__sk__seed3407`; the leaderboard is under `$OUT/plans/priority8/leaderboard/`. Existing PVT results are preserved in their original folders. With `full_train`, checkpoint selection remains the fixed final epoch, with no validation-based early stopping or test-based checkpoint selection. Training logs have `validation: null`; full test metrics appear during the separate evaluation stage. The 21 existing metrics per dataset include S-measure, weighted F-measure, MAE, adaptive/mean/max E-measure and F-measure, and threshold/boundary metrics.

## What these eight runs can establish

The included no-pseudo baseline uses the **same ConvNeXt, final-stage decoder, initialization, training data, seed, input size, and losses** as the pseudo variants. It tests whether each variant improves segmentation under this one backbone/seed condition. `projection + mean` supplies a same-resolution learned-branch control against `native_conv + mean`.

The separate [mechanism experiment](SCALE-MECHANISM.md) now implements the actual multi-input-scale reference and optional feature teacher. They are not part of the eight-case priority8 command below. ResNet50 `layer4` is available for a later explicit cross-backbone experiment, but is not in this plan. Additional seeds are needed before claiming a stable improvement.

Changing from the historical PVT multistage model to ConvNeXt final-stage changes both the backbone and decoder information path. A score difference between them cannot be attributed solely to pseudo-scale generation. Report parameters, latency, memory, and FLOPs together with segmentation metrics under the same profiling protocol. The leaderboard keeps backbone and feature-mode conditions in separate cohorts. No new trained ConvNeXt or ResNet result is claimed by this implementation.

## Paper and implementation references

Reference check date: **2026-11-04**. Korean titles below are explanatory translations.

- **A ConvNet for the 2020s / 2020년대를 위한 합성곱 신경망.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), **2022**, conference. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), [author preprint](https://arxiv.org/abs/2201.03545), [TorchVision 0.26.0 adapter source](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/convnext.py). Used for the CNN backbone; the pseudo generator and segmentation decoder are this study's additions.
- **Deep Residual Learning for Image Recognition / 영상 인식을 위한 심층 잔차 학습.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**), **2016**, conference. [Official proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), [2015 technical-report preprint](https://arxiv.org/abs/1512.03385), [TorchVision 0.26.0 source](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/resnet.py). Used for the optional ResNet50 backbone; `layer4` is the only head input in `last` mode.
- **PVT v2: Improved Baselines with Pyramid Vision Transformer / PVT v2: 피라미드 비전 트랜스포머를 통한 개선된 기준 모델.** **Computational Visual Media**, **2022**, journal, 8(3):415–424. [Publication DOI](https://doi.org/10.1007/s41095-022-0274-8), [author preprint and journal record](https://arxiv.org/abs/2106.13797). The historical backbone is retained as a separate experimental condition.
