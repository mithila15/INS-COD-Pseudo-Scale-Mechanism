# 2026-11-04 · INS_COD — ConvNeXt Final-Stage Priority8 Experiments

**English** | [한국어](README.ko.md)

Generate pseudo-scale features from one input image, then compare how those features are fused. The current experiment uses **ConvNeXt-Tiny's final stage only**, with **the original priority8 shortlist × seed 3407 = eight training runs**. The planner defaults to this eight-case scope. The larger 12-generator × 13-utilizer catalog remains available for explicitly requested later studies.

ConvNeXt is itself hierarchical: changing PVTv2 to a CNN does not remove multiple stages inside the encoder. Here the decoder receives only the final stride-32 feature; earlier-stage skips, ASPP and multikernel decoder fusion are excluded from this mode. [Architecture and comparison limits](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.md). ResNet50 Layer4 is implemented for a later comparison and is not included in this run.

Pseudo-scale features are intermediate tensors. **This implementation does not generate pseudo-labels:** segmentation, deep and region targets use ground-truth masks, edges come from those masks, and UAL regularizes prediction confidence. [Separate supervision/pseudo-label account](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/PSEUDO-LABELS.md).

## 1. Environment and existing data

Install the pinned Python 3.12.13 / torch 2.11.0 / torchvision 0.26.0 profile using the [environment guide](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.md), or reuse the server's existing `fsa-study-clean` environment. No new dependency is required for the CNN adapters. Update the server source first using [Windows pull/SCP and installation](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md#transfer-the-updated-source-from-windows).

The launcher reuses the existing `results/priority8-full4040-fixed-20261101_200834/manifest.json` and its dataset files. Training uses CAMO 1,000 + COD10K 3,040 = **4,040 images**, without a validation holdout. Tests remain CAMO 250, CHAMELEON 76, COD10K 2,026 and NC4K 4,121. A different absolute manifest can be supplied through the `MANIFEST` environment variable. Dataset preparation instructions remain in [advanced execution](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/RUNNING.md).

The acknowledged seven COD10K train/test duplicates and two CHAMELEON geometry mismatches remain recorded. The new model does not make those splits independent. Predictions are resized to untouched GT dimensions under the existing explicit evaluation policy. [Benchmark protocol](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md).

## 2. Supply the matching backbone checkpoint

Use **torchvision ConvNeXt-Tiny IMAGENET1K_V1**, `convnext_tiny-983f1562.pth` (about 109 MB), at the path below. The old PVT checkpoint cannot initialize this model. The strict local loader accepts a flat state dictionary or a `state_dict` wrapper and excludes only known classification-head keys. Nothing downloads automatically during model construction or training. [Official weight source and user-run download command](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.md#local-pretrained-initialization).

## 3. Start exactly eight experiments

Run this block in the **Ubuntu experiment server terminal** after installing the updated code and supplying the weights. It starts training → four-dataset evaluation → inference profiling sequentially, updating the leaderboard after each stage. It does not resume the old PVT checkpoints.

```bash
STUDY="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
export PY="/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python"
WEIGHTS="$STUDY/weights/convnext_tiny-983f1562.pth"

bash "$STUDY/launch_convnext_priority8.sh" "$WEIGHTS"
```

To inspect commands without starting experiments, insert `--prepare-only` before `"$WEIGHTS"`. It prints the new plan's `run_all.sh` command. To choose a result root, set `OUTPUT` to a new absolute directory before launching. Existing outputs are rejected rather than overwritten.

| Order | Case | Purpose |
| ---: | --- | --- |
| 1 | `single` | Same ConvNeXt/final-stage decoder, no pseudo generation or fusion |
| 2 | `native_conv__asff` | Resize → convolution → align; spatial fusion |
| 3 | `native_conv__sk` | Same generator; channel-selective fusion |
| 4 | `native_conv__mean` | Same generator; uniform fusion |
| 5 | `projection__mean` | Matching branch convolutions at the original feature resolution |
| 6 | `dysample__asff` | Learned upsampling; spatial fusion |
| 7 | `dysample__sk` | Learned upsampling; channel-selective fusion |
| 8 | `dysample__mean` | Learned upsampling; uniform fusion |

Common settings: **150 epochs, batch 4, 384×384 input, 48 feature channels, seed 3407**, Adam lr 1e-4, encoder lr ×0.1, frozen first stem/encoder BN, and the existing loss weights. `full_train` uses the fixed final epoch; it has **no early stopping and no validation/test-based checkpoint selection**. `best.pth` is the selected final checkpoint. [Explicit plan command and architectural changes](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.md).

## 4. Results and interpretation

The launcher prints a fresh root under `Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/convnext-last-priority8-seed3407-<timestamp>/`. The combined log is `run.log`. The leaderboard is `plans/priority8/leaderboard/leaderboard.html` (browser) or `leaderboard.md` (terminal).

Each `runs/priority8/<case>__seed3407/` contains configuration, training history and checkpoints. Test scores are in `test/metrics.json`: **21 scalar metrics**, including Sα (`S`), weighted Fβ (`wF`), MAE, adaptive/mean/max E and Fβ, IoU, Dice/F1 and boundary metrics. Curves and per-image scores are retained. `profile.json` reports model complexity and inference costs with measured hardware/settings. [Evaluation details](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/EVALUATION.md).

Different backbones and feature modes have separate comparison cohorts. The valid first comparison is among these eight ConvNeXt/final-stage cases. Historical PVT/multistage scores involve a different encoder and decoder; changes cannot be attributed solely to the backbone or pseudo-scale operator. Showing approximation of real input-scale features and the claimed cost advantage still requires a matched multiple-input-scale reference. A single seed does not establish statistical significance.

The latest uploaded actual training snapshot remains [the 2026-11-02 PVT full4040 review](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.md): 3/8 complete at export, one partial, four without run history. It is not a ConvNeXt result or live server status. [Complete exported server ZIP](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102). New CNN checks are initialization/inference/gradient and workflow checks only; no training is launched by this code update. [2026-11-04 change and verification record](Progress/2026-11-04/ConvNeXt-Priority8/README.md).

## Referenced papers

Added on **2026-11-04**: [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) (IEEE/CVF CVPR 2022, conference; ConvNeXt backbone), [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) (IEEE CVPR 2016, conference; later ResNet50 comparison), and [PVT v2: Improved Baselines with Pyramid Vision Transformer](https://doi.org/10.1007/s41095-022-0274-8) (*Computational Visual Media*, 2022, journal; preserved historical backbone). The CNN implementation uses TorchVision's recorded variants; the pseudo generator and decoder are study adaptations.

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

The recorded code verification passed **281 tests and 67 subtests**, with CPU/GPU inference checks and zero optimizer steps. That initial check did not measure real-data convergence or rankings. The historical eight-run holdout report supplies three-dataset results. The [2026-11-02 server review](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.md) adds **three completed full-data, four-test runs**, one partial run and four remaining planned runs; it does not merge the two protocols. See the [verification artifacts](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-28/README.md).

The original BRG main experiment remains in [COD](https://github.com/sh-min/COD); CoCoNet was excluded from this upload. Archived sources and their licenses remain intact. Historical organization details are in the [migration record](docs/MIGRATION.md), [experiment index](Experiments/README.md), [benchmark index](Benchmarks/README.md), and [research decisions](Progress/README.md).

Metric/leaderboard verification on 2026-10-29: **522 tests + 67 subtests**, operation coverage for all 158 initialized models, and eight browser checks passed. No training or downloads were run. [2026-10-29 verification](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-10-29-metrics/README.md).

The [2026-11-01 change record](Progress/2026-11-01/Benchmark-Completion/README.md) records the protocol change. The [2026-11-02 review record](Progress/2026-11-02/Server-Results-Review/README.md) records the recovered upload, actual partial results, archive/hash checks and remaining preplanned experiments. No training or inference was run during this review.
