# 2026-11-05 · Deployment Verification — Windows CRLF Launcher Recovery

**English** | [한국어](README.ko.md) · [User-run recovery](../../docs/SCALE-MECHANISM.md#recover-an-already-installed-crlf-launcher)

The server installed revision `bb6f040` and verified the existing manifest/backbone weights, then Bash rejected a launcher option before a new training session started. Its earlier process listing had no training process. Raw bytes of the server file were not available; the CRLF diagnosis is supported by reproducing the same Bash error locally.

A three-line LF launcher was committed into a temporary Git repository. Exporting with `core.autocrlf=true` produced three CRLF lines and Bash exit 2 (`pipefail\\r: invalid option name`). Exporting with `core.autocrlf=false` and `core.eol=lf` produced no CRLF and exit 0. [Recorded observations](reproduction.json). [Git EOL rules](https://git-scm.com/docs/gitattributes).

The fix has three independent boundaries: `.gitattributes` forces LF for shell files; the sender disables automatic EOL conversion during ZIP export; the installer normalizes source shell scripts after checking the original ZIP SHA256 and records both hashes. Existing files are backed up before replacement. Archived results, datasets, checkpoints and environments are still excluded from installation. Python source bytes are not normalized by the installer.

**52 targeted tests passed**, with one existing timm deprecation warning. [Test log](pytest.txt), [archive regression](../../tests/test_deploy_line_endings.py), [installer regression](../../tests/test_install_study.py). Checks cover each archive guard independently, binary preservation, old CRLF ZIP installation, executable Bash startup, original-byte backups, repeated-install idempotence, preserved result scripts, and existing architecture/plan/report wiring.

This is a deployment check, not a trained-model result. No weights/datasets were downloaded and no optimizer step ran. Linux Git emulated the Windows EOL configuration; actual Windows PowerShell, SCP and the experiment server were not executed here. The previous full model suite is not rerun for this deployment-only change. The user-run repair block was checked with `bash -n`; actual server training and its memory/throughput remain to be observed.
