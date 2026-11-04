# 2026-11-04 · Supervision Audit — Pseudo-Scale Features and Pseudo-Labels

**English** | [한국어](PSEUDO-LABELS.ko.md) · [CNN experiment](CNN-BACKBONES.md) · [Study guide](../README.md)

**The implemented study does not generate or train on model-predicted pseudo-labels.** It is supervised segmentation using the dataset's supplied masks. Here, “pseudo” in *pseudo-scale* refers to generated feature representations, not generated labels. This distinction applies to the historical PVT experiments and the new CNN adapters.

## What is generated and how it is used

| Item | Source | Use in the implementation |
| --- | --- | --- |
| Segmentation mask | Dataset annotation named in the audited manifest | Target for the final mask, deep heads, and region-prior head |
| Edge target | A deterministic 5×5 morphological gradient of the training mask | Target for the edge-prior head |
| Pseudo-scale features | Learned/spatial transformations of an encoder feature | Inputs to the fusion/utilization module, then the decoder |
| Region/edge predictions | Auxiliary model heads | Supervised auxiliary outputs; the region prediction also gates decoder features |
| UAL term | Final prediction probabilities | Confidence regularization; no discrete label or teacher target is constructed |
| Saved prediction PNGs | Test-time model inference | Evaluation artifacts, not inputs to training |

Implementation evidence: [dataset and target preparation](../fsa_study/data.py), [losses and training](../fsa_study/engine.py), [feature generators](../fsa_study/generators.py), [feature utilization](../fsa_study/utilizers.py), and [model heads](../fsa_study/model.py).

## Current target preparation

1. Load the image and its mask from the same manifest record. Training image/mask geometry must agree.
2. Apply the same deterministic sampled horizontal/vertical flips to both. Resize RGB bilinearly and the mask with nearest-neighbor sampling to the chosen training size.
3. Convert the mask to a binary target using `mask > 0`.
4. On that resized binary mask, compute 5×5 dilation and erosion through max pooling. The edge target is `dilation(mask) - erosion(mask) > 0`. Pool padding follows the existing implementation and does not create extra outside-image labels.

The edge target is derived from an existing annotation. Calling it a teacher-generated pseudo-label would misdescribe the code. Dataset masks remain the source of supervision, and their hashes are retained in the manifest.

## Current losses

The following is a code-level description of this study's implementation, not a new loss attributed to the backbone papers.

| Logged term | Target and computation | Effective coefficient |
| --- | --- | --- |
| `seg` | Boundary-weighted binary cross-entropy plus weighted IoU against the mask | 1 |
| `deep` | Mean of the same mask loss over all available deep heads | 0.25 |
| `region` | Same mask loss for region logits `po` | 0.5 |
| `edge` | Dice loss for edge logits `pe` against the derived edge target | 1 |
| `ual` | Mean of `1 - (2 * sigmoid(logits) - 1)^2`, with a cosine coefficient increasing from 0 to 1 over training | Scheduled 0–1 |

Spatial logits are bilinearly aligned to the target size inside supervised losses. In `last` mode there is one deep head; in the historical common multistage decoder there are four. Averaging the heads keeps the overall deep-loss coefficient at 0.25. The UAL term encourages confident predictions but does not produce stored targets, threshold predictions into pseudo masks, or make this a teacher–student method.

## Pseudo-scale generation is a separate process

For the `native_conv` cases in priority8, one projected encoder feature supplies an unchanged main branch and two independently transformed branches. The feature is resized to 1.5× or 0.5× spatial resolution, processed at that resolution, and aligned back to the main grid. The selected utilizer combines these features before segmentation prediction. No branch is treated as a label, and no loss currently forces a branch to equal an encoder output from a differently scaled RGB image.

The original motivations and adaptation limits remain in [generation](GENERATION.md) and [utilization](UTILIZATION.md). The word *approximation* describes the research objective; agreement with true multi-input-scale features still requires a reference measurement.

## If pseudo-labeling is added later

Pseudo-labeling is a **deferred, separate experiment**. It is not required to run the new ConvNeXt shortlist, and this update introduces no teacher model or pseudo-mask cache. A later implementation would need to declare the following before training:

- Which teacher checkpoint produces the probabilities, how it was trained, and which images are eligible. Test images or test annotations must not become training supervision.
- Whether targets are soft probabilities or hard masks, how confidence is measured, the threshold/ignore policy, and how geometric transforms keep targets aligned.
- Whether the teacher is frozen, updated by EMA, or periodically regenerated, and how gradients are prevented from entering a frozen teacher.
- Where generated targets, source image IDs, teacher/checkpoint hashes, and generation settings are recorded.
- How pseudo-label loss is weighted relative to the supervised losses and which no-pseudo-label control establishes its benefit.

These are requirements for a future design, not a description of behavior already implemented. Neither the current branch fusion nor the learned region gate constitutes pseudo-label generation.
