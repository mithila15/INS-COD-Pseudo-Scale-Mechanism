# 2026-11-01 · COD10K overlap audit — Evidence and explicit benchmark policy

**English** | [한국어](README.ko.md) | [Execution guide](../../docs/BENCHMARK-PROTOCOL.md#acknowledge-the-verified-cod10k-overlaps)

The user-supplied server audit found **seven byte-identical COD10K train-source/test pairs**. Six training-source records were in optimization and one was in validation. All 2,026 test images had matching masks, and none of the seven pairs shared a resolved path. This explains the strict preparation failure after source installation; it does not support deleting the dataset or previous models.

## Evidence artifacts

- [server-audit-summary.json](server-audit-summary.json): compact derivation of the supplied terminal output, including the original manifest hash, split counts, exact sample IDs and image hashes. The full terminal log and private local paths are omitted. This is user-reported server evidence, not an independent local scan of the dataset.
- [upstream-evidence.json](upstream-evidence.json): the seven corresponding train/test pairs from the independent COD-TDQ research bundle at pinned commit `c4e97371fa835cd0a778ba67c071f24ca5deaf11`, including the 14 source URLs, Git-LFS pointer texts, SHA256s and sizes. All match the user's image hashes. Only small metadata text was fetched; no images, masks, datasets or model weights were downloaded.

The [official SINet-V2 notice dated 2021-10-09](https://github.com/GewelsJI/SINet-V2/blob/5b9844c94d10a03b815c7c7039c194166a9a8ba3/README.md#L44) acknowledges Crab-32 and Cat-1506. The independent mirror corroborates all seven but is not COD10K-author acknowledgment of the other five. The notice's estimated metric impact for two images cannot establish the impact of seven images on this study.

| Test ID suffix | Training-source ID suffix | Existing split | Official notice names pair? |
| --- | --- | --- | --- |
| Crab-32 | Crab-32 | train | Yes |
| Cat-1506 | Cat-1506 | train | Yes |
| Deer-1762 | Deer-1796 | train | No |
| Gecko-1892 | Chameleon-1694 | train | No |
| Gecko-1895 | Gecko-1928 | train | No |
| Giraffe-1932 | Giraffe-1930 | val | No |
| Owl-4633 | Bird-3205 | train | No |

## Policy and interpretation

Strict rejection remains the default. The explicit `--acknowledge-cod10k-overlap` preparation option allows only the known exact COD10K ID/hash pairs, while resolved-path reuse, unknown overlap and other integrity failures remain errors. New manifests record `overlap_policy="acknowledged-cod10k-v1"` and `overlap_audit` with `disjoint=false`. Evaluation and reports disclose the overlap; standard counts or a completed benchmark do not imply independence.

The original holdout audit has six train/test pairs and one validation/test pair. Full-data preparation inherits the acknowledged policy and recomputes seven train/test and zero validation/test pairs after promoting validation into train. It keeps all 4,040 training-source and 2,026 COD10K test records, rather than silently modifying the benchmark. Existing checkpoints and original three-dataset outputs remain untouched. Acknowledgment does not reverse past optimization or validation-based selection exposure.

This particular COD10K finding does not invalidate the previously reported CAMO, CHAMELEON and NC4K scores. It also does not prove all data are free of re-encoded or visually near-duplicate images. A disjoint 2,019-image test subset or a fresh 4,033-image training protocol would be separately named experiments, not the standard 4,040/2,026 setting and not actions performed by this repair.

## Verification scope

**Code checks passed: 736 tests and 74 subtests.** [Pytest output](pytest.txt), [JUnit record](pytest.xml), [source hashes and scope](verification.json), and [five browser checks](browser-verification.json) are preserved. Tests cover strict rejection, exact seven-pair acknowledgement, unchanged training artifacts, full-training audit promotion, report integrity, all 21 metrics, and backup installation. Tests use synthetic inputs without optimizer updates or real training.

[Package verification](archive-verification.json) installed a staged-source Git archive into a fresh temporary workspace, checked four CLI entrypoints and imported the archived backbone for finite 64×64 CPU inference. Its Python/PowerShell source hashes match [verification.json](verification.json). The temporary archive predates this final verification document; no remote installation was executed.

[Documentation checks](documentation-checks.json) verify local link targets, Bash block syntax without execution, and the agreement of all seven evidence hashes.

The evidence collection verified 14 pinned Git-LFS metadata records and all seven image-hash equalities against the supplied log. It did not inspect image/mask semantics, execute server evaluation, run an optimizer step or train a new model. Code regression verification is separate from this source audit; no new accuracy or timing measurements are claimed here. The original [three-dataset priority8 report](../priority8-report-20261101_172449/README.md) remains unchanged.

[Research decision record](../../../../../Progress/2026-11-01/COD10K-Overlap/README.md) · [Study guide](../../README.md)
