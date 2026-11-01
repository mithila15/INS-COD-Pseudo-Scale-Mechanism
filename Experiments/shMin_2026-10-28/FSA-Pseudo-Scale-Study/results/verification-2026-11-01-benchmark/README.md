# 2026-11-01 · Benchmark verification — Four-test recovery and full-data training preparation

**English** | [한국어](README.ko.md) | [Protocol and execution guide](../../docs/BENCHMARK-PROTOCOL.md)

The final local suite passed **694 tests and 74 subtests**. It covers additive COD10K test manifests, immutable checkpoint/training provenance, legacy `architecture=common` metadata, inference-only evaluation of synthetic inputs, all 21 scalar metrics, report coverage, isolated-release entrypoint rebinding, and preparation of fresh full-data plans. Model tests also check synthetic tensor gradients; no optimizer update or real training was executed.

- [Final pytest output](pytest.txt) and [JUnit record](pytest.xml).
- [Initial failing output](pytest-initial.txt) and [JUnit record](pytest-initial.xml), preserved as diagnostic evidence. The new inference test left PyTorch deterministic settings enabled, changing a later CUDA sampler dispatch. Its fixture now restores RNG, deterministic, cuDNN and cuBLAS environment state. The existing FLOPs counter was not changed: an unsupported operation still prevents a fabricated total.
- [Browser verification](browser-verification.json): eight checks on synthetic leaderboard sorting, filtering and display. Recorded hashes match the final reporting implementation. The browser profile was isolated and removed.
- [Archive verification](archive-verification.json): unpacked a staged-source ZIP into a separate directory, checked four CLI entrypoints, and ran a cold model import plus CPU inference on a synthetic 64×64 input. The package includes the required sibling `Experiments/FSA-ZoomNeXt` source. ZIP size refers to that tested snapshot, before adding this verification record.
- [Source checksums and scope](verification.json).

Reproduction from the local repository, using the existing environment:

```bash
REPO="/home/robin/shMin/INS_COD"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$REPO/.venv/fsa-study/bin/python" -m pytest -q \
  "$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/tests"
```

The original uploaded priority8 report remains byte-for-byte unchanged. The actual server images, existing model tensors and new four-dataset scores were not available for end-to-end validation here. The Windows PowerShell transfer script was reviewed but not executed on Windows; no SCP transfer or remote command was run. No dataset or backbone download was started, and no experiment environment was changed. Actual server preparation must pass file/count/hash/geometry checks before evaluation or training can start. The guide above provides separate commands for evaluation reuse and new training.
