# 2026-11-04 · INS_COD — Testing Whether Pseudo Scale Adds Value

**English** | [한국어](README.ko.md)

This research branch, **`research/pseudo-scale-mechanism`**, tests pseudo-scale features against both a same-resolution learned-branch control and actual multi-input-scale encoding. **Start with four cases**, one seed, instead of the full sweep. The earlier main/priority8 experiments and their results remain preserved.

| First run | What changes | Encoder calls at inference |
| --- | --- | ---: |
| `single` | No pseudo branches | 1 |
| `projection__mean` | Two learned branches at unchanged resolution | 1 |
| `native_conv__mean` | Same branch operators at 1.5× / 0.5× feature resolution | 1 |
| `image_pyramid__mean` | Actual 1× / 1.5× / 0.5× RGB inputs, shared encoder, mean fusion | 3 |

ConvNeXt-Tiny's final stage is the sole decoder input. The branch also provides **separately invoked** stride-16 comparisons, original-feature-preserving `anchor_gated` fusion, and a frozen feature teacher trained from the actual RGB-scale reference. The optional teacher supervises features, **not pseudo-mask labels**, and is absent from student inference. No new trained result or accuracy gain is claimed.

## Installation and execution

[English: environment, branch transfer, installation, all stage commands and result locations](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md) · [한국어 실행 안내](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md).

Use the existing [pinned environment](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/environment/README.md), full4040 manifest and dataset files, plus the local official TorchVision ConvNeXt-Tiny checkpoint. The guide installs this branch in **`/home/master/mithila/shMin-scale-study`**, independently of the old source. Results still go into new directories under the original **`/home/master/mithila/shMin`** experiment tree.

After branch installation and input preparation, run in **Ubuntu Bash**:

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

This starts **four** sequential training/evaluation/profile jobs: seed 3407, 150 epochs, batch 4, 384×384 input, 48 channels, full4040 training. There is no early stopping or test-based checkpoint selection; `best.pth` is the fixed final checkpoint. To create commands only, insert `--prepare-only` before `core`. Separate `resolution` (4), `fusion` (2) and `distill` (2) commands are in the guide; none starts automatically. Distillation requires the completed reference checkpoint with matching stride and manifest.

## Results and limits

The launcher prints the result root. For core, the leaderboard is **`$CORE/plans/core/leaderboard/leaderboard.html`** or `.md`; each model's `runs/core/<case>__seed3407/test/metrics.json` has **21 metrics per dataset**, including Sα, weighted Fβ, MAE and E/Fβ variants. Training logs show losses; full test scores appear after evaluation. Inference profiles count all three encodings for the reference and one for the student.

Training uses CAMO 1,000 + COD10K 3,040 images. Evaluation covers CAMO 250, CHAMELEON 76, COD10K 2,026 and NC4K 4,121. [The existing seven COD10K cross-split duplicates and two CHAMELEON geometry mismatches](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/BENCHMARK-PROTOCOL.md) remain disclosed. Backbone/stride/teacher conditions have separate report cohorts, and one seed cannot establish statistical significance. A gain over the no-branch baseline alone does not prove a scale-specific benefit; compare the learned-branch control too.

[Branch verification](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/verification-2026-11-04-mechanism/README.md) · [Research decision record](Progress/2026-11-04/Pseudo-Scale-Mechanism/README.md) · [Historical ConvNeXt priority8](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/CNN-BACKBONES.md) · [Latest uploaded actual PVT results](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/server-review-20261102/README.md).

New reference/adaptation details for **ConvNeXt, ZoomNeXt and FitNets** are in the [English](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.md#7-references-and-adaptation-boundaries) and [Korean](Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/docs/SCALE-MECHANISM.ko.md#7-참고-논문과-적용-범위) guides. The existing bibliography follows.

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
