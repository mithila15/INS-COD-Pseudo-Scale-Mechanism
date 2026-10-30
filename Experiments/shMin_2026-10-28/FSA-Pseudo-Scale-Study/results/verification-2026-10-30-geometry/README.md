# 2026-10-30 · Test geometry recovery — Synthetic inference and preserved checkpoints

The user's remote log reported successful `single` training through epoch 150, best validation MAE 0.028608166038820583, followed by a test-loader error for `chameleon_test/animal-62`: RGB 1000×628 versus GT 998×628. This is a reported remote observation, not locally measured model accuracy. The original full log remains with the user; no raw remote data or trained checkpoint was available locally.

The fix adds explicit test-only `--test-geometry resize-prediction-to-gt`. It preserves the existing prediction-to-original-GT interpolation, original GT and manifest bytes, train/val size checks and checkpoint format. `recover_evaluation.py` prepares a fresh plan and evaluation directory while preserving old training and failed evaluation outputs. It performs no training/inference. The generated runner executes only when the user invokes it.

## Verification

- [Full test log](tests.txt): **557 tests + 67 subtests passed**, 13 dependency deprecation warnings, no failures; [JUnit XML](tests.xml).
- [Additional validation and provenance](validation.json): training-function identity, changed-document links, source and verification artifact checksums.
- [Geometry regression](../../tests/test_geometry.py): synthetic reported dimensions; strict preflight failure before checkpoint loading/output; opt-in test inference from a schema-1 synthetic checkpoint; saved prediction 998×628; all 21 finite scores; original image/GT/manifest/checkpoint hashes unchanged; strict train/val and matching-pair tensor identity.
- [Recovery regression](../../tests/test_recover_evaluation.py): 26 cases, including completed-baseline reuse and subsequent jobs in order through a mocked runner; original plan/partial outputs retained, malformed data/config/path overlap rejected before new output creation.
- [Completion checks](../../tests/test_run_plan.py) reject policy mismatch and missing geometry audits; [summary checks](../../tests/test_workflow.py) separate test geometry policies without changing training-metric cohorts.

The complete suite also retains operator, model, metric, complexity and leaderboard coverage. Neural-network forward/backward consistency tests do not call an optimizer. No optimizer steps, new training, dataset downloads, actual trained-model scoring or server restart occurred. Browser behavior was not re-tested in this patch; its previous verification remains in the dated 2026-10-29 record.

The local test environment was the existing `.venv/fsa-study` profile; no dependencies were changed. Reproduce from any directory after setting the actual checkout path:

```bash
REPO=/home/robin/shMin/INS_COD
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
"$REPO/.venv/fsa-study/bin/python" -m pytest -q "$STUDY/tests"
```

## Limits and next execution

The real `animal-62` image/GT registration has not been inspected. Mapping predictions to GT dimensions is an explicit evaluation convention, not a verified alignment correction. The recovery preflight prints the actual selected test datasets; a filename such as `manifest_no_cod10k_test.json` does not establish those contents.

Original source hashes remain in existing checkpoints/configurations. The prior baseline and newly trained variants may therefore appear in separate leaderboard cohorts; this patch does not erase provenance to force a combined ranking. Full test metrics are emitted only after evaluation completes. Early stopping and interrupted-optimizer resume were not introduced.

See the [English recovery commands](../../README.md#recover-a-test-imagemask-size-mismatch), [Korean guide](../../README.ko.md), and [research change record](../../../../../Progress/2026-10-30/Geometry-Recovery/README.md). The user must explicitly run the generated script to evaluate the existing baseline and train the seven remaining variants.
