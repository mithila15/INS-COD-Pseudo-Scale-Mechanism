# 2026-11-04 · Mechanism Branch Verification — Initialized Models and Execution Boundaries

**English** | [한국어](README.ko.md) · [Run guide](../../docs/SCALE-MECHANISM.md)

This is **code verification**, not a COD training/evaluation result. No datasets or pretrained weights were downloaded, and no optimizer step was performed. The new branch does not establish a segmentation improvement.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Full study regression suite | **813 tests + 74 subtests passed**, 13 existing deprecation warnings | [pytest log](pytest.txt) |
| Previous model entry point vs new defaults | Identical initialized state tensors and exactly matching 64×64 logits for four PVT/CNN baseline/pseudo cases | [Model report](models.json), [reproduction script](verify_models.py) |
| GPU inference | Ten initialized models: five variants × strides 32/16, 384×384 RGB, 48 channels, batch 1, finite output | [Model report](models.json) |
| Inference operation accounting | All observed operators classified for those ten graphs, including three real RGB encodings and stride-16 padding | [Model report](models.json) |
| Feature teacher gradients | At both strides, batch 2 at 128×128: finite connected student gradients, frozen teacher, zero optimizer steps | [Model report](models.json) |
| Workflow/report checks | 4/4/2/2 stage plans, one seed, no run directory during preparation; real-scale variant kept distinct; stride/teacher cohorts separated; checkpoint teacher hash checked | [Mechanism tests](../../tests/test_scale_mechanism.py), [pipeline tests](../../tests/test_backbone_pipeline.py) |
| Documentation | Local relative links/anchors and Bash syntax checked | [Documentation check](docs-check.json) |
| Installation boundary | Active process inside destination rejected; process using sibling source permitted; backup/input preservation checks | [Installer tests](../../tests/test_install_study.py) |

The GPU checks use the local **NVIDIA GeForce RTX 5080**, torch 2.11.0+cu130, random tensors and initialized weights. They do not validate training throughput or peak memory for batch 4 at 384×384 with a teacher on the user's server. The actual input manifest, pretrained checkpoint and all COD training/evaluation remain to be run by the user.

An initial stride-16 profiling test found `aten.pad.default` unclassified. It is now explicitly accounted as data movement (zero floating arithmetic under the documented convention), with four padding-mode checks; its runtime is still included in latency. The corrected full suite and model checks above passed.

Windows PowerShell is unavailable in this Linux validation environment. The branch-aware transfer script was reviewed, but Windows execution, authentication and SCP were not executed here. Source ZIP installation is tested with local fixtures.

Reproduce initialized-model checks from a Git checkout that contains the base revision (not a source-only server ZIP). This command runs inference and backward connectivity only; it does not train. Use a new output filename:

```bash
REPO="/home/robin/shMin/INS_COD"
STUDY="$REPO/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
PY="$REPO/.venv/fsa-study/bin/python"
"$PY" "$STUDY/results/verification-2026-11-04-mechanism/verify_models.py" \
  --device cuda:0 --output "/tmp/scale-mechanism-verification-$(date +%Y%m%d_%H%M%S).json"
```

The compatibility check executes the previous `model.py` entry point against the current backward-compatible operator implementations; the compared cases and source hashes are in the report. It is not a claim to test every historical checkpoint. Real benchmark scores remain in their earlier result directories. [User-run training commands](../../docs/SCALE-MECHANISM.md#4-ubuntu-bash-start-only-the-four-core-runs).
