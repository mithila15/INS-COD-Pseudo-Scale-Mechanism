# 2026-11-01 · COD10K overlap validation — Preserve existing same-split records

**English** | [한국어](README.ko.md) | [Execution guide](../../docs/BENCHMARK-PROTOCOL.md#preparation-fails-after-successful-installation)

Source installation at `8f08f4ff6b61` succeeded on the user's server, but preparation then stopped with `Unacknowledged image overlap: unknown hash or a group with more than two records`. The new evaluation plan was not created, so the subsequent missing `run_evaluation.sh` and log-directory errors did not indicate a failed model evaluation. Neither evaluation nor fresh training started in that attempt.

## Reproduced defect and repair

The acknowledged policy examined pre-existing test-only hash groups against a list intended for exact training-source/test pairs. A group entirely inside the test split could therefore fail, although the original strict policy accepted it. The acknowledged policy also rejected same-split path aliases that the strict policy had preserved. Synthetic records reproduce these regressions; the exact offending group on the server has not been inspected.

The repair restores the same-split behavior in the overlap metadata calculation, full manifest validator and snapshot validator. It does not change dataset membership, sample order, hashes, geometry handling, models, metrics or the seven-pair evidence catalog.

| Case | Behavior after repair |
| --- | --- |
| Existing repeated bytes or paths entirely inside one split | Preserve records under both policies |
| Resolved image path shared across splits | Reject under both policies |
| Exact known COD10K training-source/test pair | Reject by default; permit and disclose only with explicit acknowledgement |
| Unknown cross-split hash, wrong ID/source/stem, train/val overlap, or cross-split group with more than two records | Reject |
| New test record reusing an original image path or an unacknowledged original-image hash during additive evaluation | Reject; extension guards remain active |

The audit continues to describe **cross-split** equality. It does not certify uniqueness within a split or independence between test datasets. Preserving existing same-split records prevents a protocol change during recovery; it is not a claim that duplicate records have no effect on aggregate statistics.

## Verification and limits

**747 tests and 74 subtests passed.** [Pytest output](pytest.txt), [JUnit record](pytest.xml) and [verification scope and source hashes](verification.json) record the checks. They cover same-split records, preserved cross-split rejection, explicit acknowledgement and the preparation flow. They are synthetic code checks, not a new accuracy measurement. The [earlier seven-pair evidence](../verification-2026-11-01-overlap/README.md) and [three-dataset experiment report](../priority8-report-20261101_172449/README.md) remain unchanged.

No dataset or weight download, optimizer update, real training or server evaluation was performed for this repair. Installing the fix and successfully preparing the user's actual manifest remain necessary before execution. If preparation still reports unknown cross-split overlap, its records need inspection; retraining or bypassing validation cannot resolve the preparation issue.

[Research decision record](../../../../../Progress/2026-11-01/COD10K-Same-Split-Fix/README.md) · [Study guide](../../README.md)
