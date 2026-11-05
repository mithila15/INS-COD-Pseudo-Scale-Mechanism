"""Reproduce Windows-style git archive conversion; run only a printf launcher."""
from pathlib import Path
import re
import shlex
import subprocess
import zipfile

import pytest

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]


@pytest.mark.parametrize("attributes,export_override", [(False, False), (True, False),
                                                       (False, True), (True, True)])
def test_windows_archive_shell_runs_with_each_independent_eol_guard(tmp_path, attributes, export_override):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    launcher = b'#!/usr/bin/env bash\nset -euo pipefail\nprintf "ready\\n"\n'
    binary = b'\x00preserve\r\n\xff'
    (tmp_path / "start.sh").write_bytes(launcher)
    (tmp_path / "input.bin").write_bytes(binary)
    if attributes:
        (tmp_path / ".gitattributes").write_bytes((REPO / ".gitattributes").read_bytes())
    git("add", ".")
    git("-c", "user.name=verification", "-c", "user.email=verification@example.invalid",
        "commit", "-qm", "archive fixture")
    options = []
    if export_override:
        sender = (STUDY / "deploy/send_study.ps1").read_text()
        command = re.search(r"^Invoke-GitChecked (.*?) -C \$Repo archive ", sender, flags=re.M)
        assert command is not None, "Expected explicit sender archive options"
        options = shlex.split(command.group(1))
    archive = tmp_path / "source.zip"
    git("-c", "core.autocrlf=true", *options, "archive", "--format=zip", "--output", str(archive), "HEAD")
    with zipfile.ZipFile(archive) as source:
        script = source.read("start.sh")
        assert source.read("input.bin") == binary
    process = subprocess.run(["bash"], input=script, capture_output=True)
    if attributes or export_override:
        assert script == launcher
        assert process.returncode == 0 and process.stdout == b"ready\n", process.stderr
    else:
        assert b"\r\n" in script
        assert process.returncode != 0 and b"invalid option name" in process.stderr
