# 2026-10-29 · Verification — Expanded metrics and automatic leaderboards

This directory records **synthetic/inference-only verification**, not trained COD accuracy. No dataset download, optimizer step, teacher generation or training was performed. The environment is the existing isolated study interpreter, Python 3.12.13 with PyTorch 2.11.0+cu130 and PySODMetrics 1.6.2; no environment rebuild was needed.

| Evidence | Result and scope |
| --- | --- |
| [Final pytest log](tests-final.txt), [JUnit](tests-final.xml) | **522 tests + 67 subtests passed**. Includes old operators, 21 metrics, artifact export, inference CLI on an untrained schema-1 checkpoint, ranking/cohort integrity and runner orchestration with substituted training commands. |
| [Initial integration log](tests.txt), [JUnit](tests.xml) | Earlier 514-test integration checkpoint, before additional completed-stage and plan-identity regression tests. Preserved as history. |
| [All 158 full-model coverage](all158-fullmodel-coverage.json) | 156 generator/utilizer pairs plus two controls, random weights/input, CPU batch 1, 64×64, channels 8. Every observed operator classified. This tests coverage, not 384×384 accuracy/latency. |
| [Baseline profile](baseline.json), [CARAFE profile](carafe.json), [CUDA DySample profile](dysample_cuda.json) | Random initialization/input, 384×384, channels 48, warmup 1/repeats 2. Integration checks only; too few repetitions and different devices for a paper latency comparison. |
| [Browser verification](browser-verification.json) | Eight isolated Chrome checks: three-variant ranking, test selection, BER direction, filtering, cost view and numeric rendering. Synthetic report scores are test fixtures, not experiment results. |
| [Documentation checks](documentation-checks.json) | Bash syntax, matching English/Korean command blocks and retained research references. |
| [Migration and links](migration-verification.json) | Original archived sources preserved; local documentation links and README title rules checked. |
| [Source hashes](source-sha256.json), [artifact checksums](checksums.sha256) | Current code and verification artifact identity; earlier reports retain their original embedded metadata. |

The original six COD metric fields agree bit-for-bit with direct PySODMetrics accumulation in regression tests. Fβ curves agree with the library reference; Boundary IoU erosion matches the authors' implementation. Empty/full/disjoint masks, threshold order, per-image versus dataset maxima, PNG roundtrip, artifact collisions and no-overwrite behavior are covered. Full-model contraction counts agree with PyTorch's official counter for representative baseline/CARAFE/DySample/SFP/legacy models.

Leaderboards keep validation selection, descriptive test rankings and hardware-specific costs separate. Tests check same-seed cohorts, missing experiments, incompatible metric versions, corrupt checkpoint identity, exact ties and failed/partial stage preservation. Planned training is tested with substituted commands; the real model CLI smoke test is evaluation only.

To repeat non-training tests from any directory, with an existing environment:

```bash
STUDY=/home/robin/shMin/INS_COD/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study
PY=/home/robin/shMin/INS_COD/.venv/fsa-study/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q "$STUDY/tests"
```

Optional standalone reproducibility scripts, writing to **new** output paths:

```bash
VERIFY_OUT="/tmp/ins-cod-verification-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VERIFY_OUT"
"$PY" "$STUDY/results/verification-2026-10-29-metrics/verify_browser.py" \
  --study "$STUDY" --output "$VERIFY_OUT/browser.json"
"$PY" "$STUDY/results/verification-2026-10-29-metrics/verify_complexity_grid.py" \
  --study "$STUDY" --output "$VERIFY_OUT/complexity-grid.json"
```

Use the same terminal for both blocks. The browser script requires `/usr/bin/google-chrome`; it uses an isolated temporary browser profile. The grid script performs inference only. These commands do not establish real-data loading, pretrained-weight compatibility, convergence, real accuracy rankings or significance. [Metric definitions](../../docs/EVALUATION.md) · [실행 안내](../../../../../README.ko.md).
