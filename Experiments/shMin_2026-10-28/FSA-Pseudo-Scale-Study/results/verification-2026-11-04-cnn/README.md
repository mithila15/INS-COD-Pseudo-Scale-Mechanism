# 2026-11-04 · CNN Verification — Architecture, Priority8 and Compatibility Checks

These are **code and untrained-model verification artifacts**, not COD benchmark results. English is the default; [한국어](README.ko.md).

- Full test suite: **780 passed, 74 subtests passed**, 13 deprecation warnings, 20.48 s. Tests cover synthetic forward/gradient connectivity, strict local checkpoint loading, checkpoint reconstruction and optimizer grouping; optimizer steps: **0**.
- Historical PVT baseline (659 state tensors), native_conv+SK (723), and full legacy reference (1,283) matched the original Git model's state keys, seeded tensors and output tensors exactly. This checks compatibility with the prior source, not restoration of unavailable trained server weights.
- ConvNeXt-Tiny / last / native_conv+SK: **384×384, channels48, CUDA inference** passed on the local NVIDIA GeForce RTX 5080, with no unsupported complexity operators. Weights and input are random. Five timing samples after two warmups are diagnostic only; they do not establish trained-model latency on the experiment server or accuracy.
- Launcher `--prepare-only`: exactly **8 jobs, seed3407**, and no run directories created. A temporary placeholder weight file was used only to check command construction; it was never loaded or executed.

[Machine-readable verification and source hashes](verification.json) · [Plan summary](launcher-plan-check.json) · [Random initialization profile](convnext-last-random-profile.json).

No dataset/model download, optimizer update or training was performed. The official pretrained CNN file, real server input access and convergence remain unverified here. Existing PVT results were preserved.

The suite used the existing local `fsa-study` overlay and its recorded torch 2.11.0+cu130 / torchvision0.26.0+cu130 environment; this does not certify a clean installation. [Environment provenance](../../environment/README.md).

To reproduce the inference diagnostic, choose a **new** output path:

```bash
STUDY="/home/robin/shMin/INS_COD/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="/home/robin/shMin/INS_COD/.venv/fsa-study/bin/python"
"$PY" "$STUDY/profile_study.py" \
  --backbone convnext_tiny --feature-mode last --generator native_conv --utilizer sk \
  --size 384 --channels 48 --device cuda:0 --warmup 2 --repeats 5 \
  --output "$STUDY/results/convnext-random-check-$(date +%Y%m%d_%H%M%S).json"
```

[Training guide](../../docs/CNN-BACKBONES.md) · [Study index](../../README.md).
