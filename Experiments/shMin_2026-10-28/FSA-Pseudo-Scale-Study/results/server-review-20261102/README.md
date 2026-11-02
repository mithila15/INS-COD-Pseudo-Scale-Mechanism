# 2026-11-02 · Pseudo-Scale Study — Full-4040 Server Snapshot Review

[한국어](README.ko.md)

The uploaded snapshot contains **3 completed experiments out of 8**. Native-scale generation with ASFF or SK improves the four principal metrics on CAMO and CHAMELEON relative to the single-encoder baseline. COD10K and NC4K results are mixed. **There is no overall winner**, and this single-seed partial result does not establish statistical significance.

## Snapshot and comparison conditions

The server export metadata records **2026-11-02 13:16:05 KST** (`2026-11-02T13:16:05.387305+09:00`). The run is `priority8-full4040-fixed-20261101_200834`. Files were copied from an ongoing experiment; this is an archive, not a live server-status check.

| Variant | Evidence in this snapshot |
| --- | --- |
| `single` | 150/150 epochs; four-dataset evaluation and inference profile complete |
| `native_conv__asff` | 150/150 epochs; four-dataset evaluation and inference profile complete |
| `native_conv__sk` | 150/150 epochs; four-dataset evaluation and inference profile complete |
| `native_conv__mean` | History reaches epoch 15/150; no completed evaluation/profile |
| `projection__mean`, `dysample__asff`, `dysample__sk`, `dysample__mean` | Planned; no run history in the snapshot |

The original leaderboard reports 3 complete and 5 not started because it reflects an earlier stage refresh. The [fourth run history](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__mean__seed3407/history.jsonl) and [run log](../priority8-full4040-fixed-20261101_200834/run.log) record epoch 15. Original reports are preserved without rewriting this historical status.

All three completed comparisons use one PVT-v2-B2 encoder and the controlled common decoder, seed **3407**, **384 × 384** input, training batch size **4**, **150 epochs**, and the same local backbone initialization. Training uses **CAMO 1,000 + COD10K 3,040 = 4,040 images**, with no validation holdout. The evaluated checkpoint is fixed at **epoch 150**; no validation/test score selects it. Training source hashes, evaluation source hashes, manifest hash and recorded initialization hash agree across the completed runs. Configuration and provenance are retained in the [original run directory](../priority8-full4040-fixed-20261101_200834/runs/priority8/).

## Principal test metrics

Test counts per experiment: **CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121**, totalling 6,473 images. `S` is Sα, `wF` is weighted Fβ, and `E_mean` is mean enhanced alignment over the threshold curve. Higher is better except MAE. Values below are rounded to eight decimals; JSON artifacts retain full precision. The evaluation uses the recorded `official_py_sod_metrics_uint8_v2` protocol; these are not the earlier float validation-MAE logs.

| Dataset | Variant | S ↑ | wF ↑ | E_mean ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| CAMO | single (baseline) | 0.83650563 | 0.79112954 | 0.89889561 | 0.05871336 |
| CAMO | native_conv + ASFF | 0.83918770 | 0.79462401 | 0.90248976 | 0.05817316 |
| CAMO | native_conv + SK | 0.83769063 | 0.79331306 | 0.89937418 | 0.05775508 |
| CHAMELEON | single (baseline) | 0.90062112 | 0.85471781 | 0.94287184 | 0.02720466 |
| CHAMELEON | native_conv + ASFF | 0.90348041 | 0.85724452 | 0.94806074 | 0.02487026 |
| CHAMELEON | native_conv + SK | 0.90742662 | 0.86523292 | 0.95109041 | 0.02359833 |
| COD10K | single (baseline) | 0.85513856 | 0.76665013 | 0.92464114 | 0.02567728 |
| COD10K | native_conv + ASFF | 0.85550535 | 0.76559010 | 0.92470970 | 0.02521642 |
| COD10K | native_conv + SK | 0.85504774 | 0.76657850 | 0.92499880 | 0.02555312 |
| NC4K | single (baseline) | 0.87244166 | 0.82299061 | 0.92479790 | 0.03662558 |
| NC4K | native_conv + ASFF | 0.87188360 | 0.82202524 | 0.92544153 | 0.03643222 |
| NC4K | native_conv + SK | 0.87149105 | 0.82227157 | 0.92337927 | 0.03654112 |

There are **21 metrics per dataset** in the original JSON, including adaptive/mean/max E and Fβ, IoU, Dice/F1, precision, recall, specificity, BER, FPR, FNR, accuracy, balanced accuracy and boundary IoU. `E_mean` must not be substituted for `E_max` when matching an external paper's table.

## Interpretation of this snapshot

- **CAMO:** ASFF is highest on S, wF and E_mean; SK has the lowest MAE. ASFF versus baseline: ΔS **+0.00268207**, ΔwF **+0.00349447**, ΔE_mean **+0.00359415**, ΔMAE **−0.00054020**.
- **CHAMELEON:** SK is strongest on all four displayed metrics. Versus baseline: ΔS **+0.00680550**, ΔwF **+0.01051510**, ΔE_mean **+0.00821857**, ΔMAE **−0.00360633**. This set has only 76 images.
- **COD10K:** ASFF leads S and MAE; SK leads E_mean; the baseline retains the highest wF. SK S and wF differ from baseline by only **−0.00009082** and **−0.00007163** respectively.
- **NC4K:** the baseline retains the highest S and wF; ASFF leads E_mean and MAE. Both pseudo-scale variants have slightly lower S and wF than baseline.

Pseudo-scale fusion has a useful signal on two datasets, but these numbers do not yet establish a consistent benefit across the larger benchmarks. The pending mean/projection/DySample experiments are needed to distinguish generation effects from fusion effects. Rankings describe the test observations; they do not select a checkpoint or form a composite winner.

## Inference and training cost

| Variant | Parameters | GMAC/image | Median ms/image | Images/s | Latency vs baseline | Training hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single (baseline) | 25,318,163 | 13.9843 | 6.3071 | 158.53 | — | 4.744 |
| native_conv + ASFF | 25,351,775 | 14.0980 | 7.7616 | 128.85 | +23.06% | 5.785 |
| native_conv + SK | 25,351,427 | 14.0680 | 7.4474 | 134.30 | +18.08% | 5.703 |

Inference was recorded on **NVIDIA RTX PRO 5000 Blackwell**, FP32, batch 1, 384 × 384 synthetic RGB input, 10 warm-up iterations and 50 timed repeats. It measures device input through all output heads, including deep/region/edge heads, and excludes IO and host/device transfer. Images/s uses measured mean latency; the adjacent latency column is the median. Training hours are recorded elapsed training times, not complete pipeline runtimes.

The ASFF/SK additions increase MACs by approximately **0.81% / 0.60%** but measured latency by **23.06% / 18.08%**. Thus small arithmetic-count overhead does not imply small wall-clock overhead. These are measurements from this profile, not a claim about all hardware or production end-to-end throughput. Recorded profile CLI defaults under `protocol.generator/utilizer/baseline` do not identify the loaded model; the effective configuration is in `metadata.config` and matches the evaluated checkpoint metadata.

## Verification, limits and retained evidence

We independently reaggregated **252 scores = 3 experiments × 4 datasets × 21 metrics** from the supplied per-image CSVs and dataset threshold curves. The maximum absolute difference from `metrics.json` was **2.22 × 10⁻¹⁶**. Each completed run has 6,473 distinct dataset/image IDs, identical ID sets across runs, and correctly ordered metric leaderboards. This verifies internal artifact consistency. It does **not** rerun evaluation against ground-truth pixels: raw input/GT datasets and checkpoint weights were excluded from the transfer, and no training or model inference was performed during this review.

The complete ZIP CRC check passed. For the three newly completed runs, **19,509/19,509 recorded artifact SHA256 hashes** match the decompressed ZIP bytes, including 19,419 prediction PNGs and the recorded curve/CSV/JSON artifacts. These checks verify transferred bytes, not GT correctness or producer authenticity. See the [artifact hash audit](artifact-audit.json) and [asset index](asset-index.md).

Known data limitations remain visible in the original artifacts:

- **One seed (3407):** no multi-seed uncertainty estimate or statistical significance claim.
- **Seven COD10K train/test duplicate-image pairs:** retained under `acknowledged-cod10k-v1`, with 7 train/test and 0 validation/test affected test images. The reported COD10K score includes them and is not evidence from a disjoint test set.
- **Two CHAMELEON image/mask dimension mismatches:** `animal-62` and `animal-74`. The recorded policy resizes predictions to the unchanged GT grid; spatial registration has not been independently verified.
- The previous 3,636-image holdout experiment uses a different training sample count and checkpoint-selection rule. It is preserved separately and is not pooled into this comparison. Matching ZoomNeXt's 4,040-image count alone does not establish an identical training recipe.

The [GitHub Release archive](https://github.com/sh-min/INS_COD/releases/tag/server-transfer-20261102) retains the original verified ZIP and checksum, including **all 54,995 prediction PNGs** from the current and earlier experiments, plus curve assets, code, results and logs. The repository contains browsable results/metadata and curve assets; prediction masks remain in that archive. Dataset originals, model weights and environments were excluded as requested.

All **114 non-results source/document paths** in the transferred scopes are present and match the restored reference `e79c8d477c7a5b0954d8321e4bd838ce52fc259f` after CRLF/LF normalization: 4 byte-identical and 110 differing only in line endings. There is no semantic code change to merge from this upload; existing repository source versions are retained. Original artifacts, including the stale leaderboard status, are preserved.

- [Original leaderboard](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.md) · [JSON](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.json) · [HTML file](../priority8-full4040-fixed-20261101_200834/plans/priority8/leaderboard/leaderboard.html).
- [single metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/single__seed3407/test-four-benchmarks/metrics.json) · [ASFF metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__asff__seed3407/test-four-benchmarks/metrics.json) · [SK metrics](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__sk__seed3407/test-four-benchmarks/metrics.json).
- [ASFF profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__asff__seed3407/profile.json) · [SK profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__sk__seed3407/profile.json) · [single profile](../priority8-full4040-fixed-20261101_200834/runs/priority8/single__seed3407/profile.json).
- [Full-precision analysis and baseline deltas](analysis-results.json) · [Protocol and aggregation audit](protocol-audit.json) · [Source and archive inventory](source-audit.json) · [Archive integrity audit](archive-audit.json).
