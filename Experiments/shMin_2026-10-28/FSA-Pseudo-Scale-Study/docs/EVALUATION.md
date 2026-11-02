# 2026-11-01 · Evaluation — Metrics, inference cost and leaderboards

**English** | [한국어](EVALUATION.ko.md) | [Execution guide](../../../../README.md)

This describes the implemented protocol, not measured COD performance. New test evaluations use `official_py_sod_metrics_uint8_v2`. Existing checkpoints can be evaluated without retraining. The original six scores are preserved; synthetic regression tests compare them directly with PySODMetrics 1.6.2.

## Scalar outputs

Every available test dataset receives `count` and the following **21 scalar fields**. Aggregate each image equally within its dataset. Keep datasets separate; no combined cross-dataset accuracy score is invented.

| Keys | Meaning and convention | Better |
| --- | --- | --- |
| `S` | S-measure, default structure weight 0.5 | Higher |
| `wF` | Weighted F-measure, library default beta parameter 1; distinct from ordinary Fβ | Higher |
| `MAE` | Mean absolute pixel error | Lower |
| `E_adaptive`, `E_mean`, `E_max` | Adaptive threshold E-measure; mean/max of the dataset-average 256-threshold curve | Higher |
| `Fbeta_adaptive`, `Fbeta_mean`, `Fbeta_max` | Ordinary F-measure with beta squared 0.3; adaptive threshold, and mean/max of the dataset-average curve | Higher |
| `IoU_0_5` | Foreground intersection / union | Higher |
| `Dice_0_5`, `F1_0_5` | Twice foreground intersection / foreground-size sum; identical aliases | Higher |
| `Precision_0_5`, `Recall_0_5` | Foreground precision and sensitivity | Higher |
| `Specificity_0_5` | Background true-negative rate | Higher |
| `BER_0_5` | Mean of foreground false-negative and background false-positive rates, range 0–1 | Lower |
| `FPR_0_5`, `FNR_0_5` | False-positive and false-negative rates | Lower |
| `Accuracy_0_5`, `BalancedAccuracy_0_5` | Pixel accuracy; balanced accuracy = 1 − BER | Higher |
| `BoundaryIoU_0_5` | Intersection/union of foreground inner-boundary bands | Higher |

Do not count Dice and F1 as independent evidence. Binary overlap metrics are foreground **image macro means**, not foreground/background class mIoU or a globally pooled confusion matrix. BER is a fraction; multiply by 100 only when a paper table explicitly uses percent.

## Prediction preparation and thresholds

1. Apply sigmoid to the segmentation logits, then per-image min–max normalization at model resolution.
2. Bilinearly resize to the original ground-truth dimensions with `align_corners=False`.
3. Clip to [0,1], multiply by 255 and truncate to uint8. Save this original-size grayscale probability PNG.
4. PySODMetrics prepares the uint8 map: divide by 255, then min–max normalize nonconstant maps. Constant maps remain divided by 255. Ground truth is the source mask greater than zero.
5. Compute all test metrics on that prepared map. Fixed binary fields use **prediction ≥ 0.5**. Adaptive E/Fβ uses `min(2 × mean(prediction), 1)`. Curves contain 256 library thresholds and are exported in ascending uint8 order 0–255.

The saved PNG is the input to the metric library, not a thresholded mask. Reproducing scores from it must apply the same preparation. Validation remains the original **float MAE before uint8/library preparation**, weighted by image count across the training holdout; only that score selects `best.pth` under the default `holdout` protocol. Validation/test MAE therefore use distinct documented conventions.

For E/Fβ mean/max, first average the curves across images, then take the mean/max across thresholds. Averaging per-image maxima would produce a different score. `per_image_metrics.csv` includes per-image maxima only as diagnostics. Tiny fixtures can give E slightly above 1 because the original library uses its published denominator convention; values are not silently clipped.

Boundary IoU follows the authors' 3×3 erosion with radius `max(1, round(0.02 × image diagonal))` and zero padding so image-border contours are included. It evaluates the semantic foreground boundary and does not reproduce COCO instance AP.

For binary metrics, two empty masks have IoU/Dice/F1/Boundary IoU 1. If only one mask is empty, those overlap scores are 0. No predicted foreground gives precision 1 only when ground truth is also empty, otherwise 0. An absent ground-truth foreground/background class has recall/specificity 1 and contributes zero error to BER. Official S/wF/E/Fβ retain their own library empty-mask rules, including zero Fβ on empty denominators. The complete definition is embedded in `metrics.json` under `metric_protocol_details`.

## Files produced by test evaluation

- `metrics.json`: all dataset scores, checkpoint/manifest/code identity, protocol definitions, artifact paths and SHA-256 hashes.
- `per_image_metrics.csv`: dataset, sample ID, source mask path, original dimensions, prediction path and all 21 scores.
- `predictions/<dataset-key>/<sample-key>.png`: original-size probability maps; names include an ID hash to prevent collisions. The CSV maps them back to original examples.
- `curves/<dataset-key>/threshold_curves.csv`: uint8/normalized threshold, precision, recall, Fβ and E, with 256 rows.
- `curves/<dataset-key>/{pr,fbeta,emeasure}.{svg,png}`: plots generated directly from the CSV curves. Blue lines represent the evaluated model on the named dataset; axes state score/threshold or precision/recall. Plot creation lives in `fsa_study/metrics.py`.

`--no-save-predictions` omits only probability PNGs. It does not disable numeric/curve exports. Existing evaluation directories are rejected; preserve the previous files and choose another destination.

## Inference cost

`profile_study.py` preserves its original full-head latency and CUDA-memory measurement. An additional **untimed** inference counts executed ATen operators, including functional matrix multiplication/einsum used by CARAFE. It does not add hooks to the timing loop.

| Field | Interpretation |
| --- | --- |
| `parameters`, `encoder_parameters` | All model parameters and encoder-only parameters |
| `latency_ms` | Mean, median, sample SD and raw timed samples |
| `images_per_second` | Batch size divided by mean latency |
| `cuda_memory` | Peak allocated bytes and increment during inference; null on CPU |
| `complexity.macs`, `.flops` | Dense convolution and tensor-contraction MACs; FLOPs = 2 × MACs, for the entire batch |
| `.macs_per_image`, `.flops_per_image` | Counts divided by batch size |
| `.estimated_noncontraction_flops` | Separate analytical estimate for arithmetic, normalization, activations, pooling, interpolation and sampling |
| `.estimated_total_flops`, `.estimated_total_flops_per_image` | Contraction FLOPs plus estimates, available only when every observed operator has a rule |
| `.coverage`, `.operators` | Classified/unsupported call counts, operator names, counting rules and counts |

Grouped convolutions use the actual grouped weight dimensions; transposed convolutions count the input spatial grid. Bilinear interpolation/grid sampling use 4 multiplications + 3 additions per output element, excluding internal coordinate/index construction. Scalar comparisons/transcendentals each count as 1 in the estimate. These conventions are analytical, not hardware instruction counts; the call-coverage fraction is not a FLOP-coverage fraction. Unknown operations leave the estimated total null and remain listed. Memory movement is not counted as floating arithmetic but can still affect latency.

Always report device, input size, batch, dtype, warmup/repeats and timing scope. The model runs all segmentation/deep/region/edge heads. IO, transfer, backward, optimizer, training memory and energy are excluded. Profiling random weights/input measures execution only; it cannot establish segmentation quality or learned branch utility.

## Automatic leaderboards

The generated `run_all.sh` explicitly executes training → evaluation → profiling per planned run. `train.sh` runs training only. Both refresh `plans/<phase>/leaderboard/` after each stage; generating a plan does not execute it.

- `leaderboard.html`: standalone interactive dataset/metric selector, variant filter and sortable tables.
- `leaderboard.md`: validation MAE, chosen test metric and principal costs for review.
- `leaderboard.csv` / `.json`: all metrics/ranks, directions, protocols, seed sets, missing seeds and run locations.
- `summary.csv` / `.json`: original mean/sample-SD summary format, extended with the new fields.

Under default `holdout`, overall selection uses **lowest mean validation MAE**. `full_train` fixes selection to the final epoch, has null `best_val_mae` and has no validation ranking. Each test dataset/metric and each inference-cost measure has a separate descriptive ranking. No weighted combination across unlike metrics, no automatic checkpoint selection from tests, and no statistical-significance claim. Exact ties share competition ranks (1,1,3).

Only identical training/evaluation protocols **and identical observed seed sets** share a ranking. Generator/utilizer options identify the compared variant; their values are retained. Hardware, input, dtype and measurement differences separate cost tables. Single-seed results are provisional; `--plan` records expected seeds and not-started jobs. Without a plan, the target seed count is unknown. A one-row cohort's rank 1 is not evidence of outperforming other cohorts.

Completed-stage skips verify plan actions/destinations, training configuration, checkpoint identity, current evaluation/cost protocol, measurement settings and evaluation artifact hashes. A stale metric set or different profile cannot silently count as completed. Partial training is never resumed or overwritten. Only six derived report files are replaced by explicit `--overwrite`; raw results remain intact.

For re-evaluation, use a new `--evaluation-subdir test-v2` and `--profile-name profile-v2.json` in the plan and summarizer. Direct `evaluate`/`profile` commands use corresponding new `--output` paths. Use `run_plan.py --dry-run --plan /absolute/path/plan.json` to inspect commands without executing anything.

## References and adaptation scope

Checked **2026-10-29**. The [method catalog](../README.md#referenced-papers) retains all 11 pseudo-scale/fusion references. These additional sources define evaluation; they do not supply new training modules.

| Reference | Venue/status | Use here |
| --- | --- | --- |
| [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021; conference | Semantic-mask boundary comparison following the [authors' erosion implementation](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py); not full instance evaluation |
| [ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection](https://doi.org/10.1109/TPAMI.2024.3417329) | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024; journal | Image-COD metric/curve reporting context; [§4.1.2](https://arxiv.org/html/2310.20208v4#S4.SS1.SSS2) |
| [PySODMetrics](https://github.com/lartpang/PySODMetrics) | Official implementation repository; installed package 1.6.2 | S/wF/MAE/E/Fβ and threshold PR curves; individual metric-paper links are stored in `metric_protocol_details.sources` |
| [PyTorch 2.11 FLOP counter](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/flop_counter.py) | Official software source | Dense contraction counting convention, checked independently against our counter |

The recorded 2026-10-29 verification used synthetic images/checkpoints or initialization-only inference. The later uploaded priority8 report contains real three-dataset scores for one seed, with COD10K-test absent; it is not a complete four-benchmark result. [Verification records](../results/verification-2026-10-29-metrics/README.md).

## Complete benchmark coverage

`benchmark_coverage` checks for CAMO 250, CHAMELEON 76, COD10K 2,026 and NC4K 4,121. Leaderboard `execution_complete` means the planned stages completed; `benchmark_complete` means all four standard benchmarks were evaluated. Missing datasets and nonstandard counts remain explicit. See [recovery commands and manifest provenance checks](BENCHMARK-PROTOCOL.md). Evaluating a checkpoint against an extended test manifest preserves both the original training manifest and the extension proof; it does not merge models trained under different conditions into one cohort.
