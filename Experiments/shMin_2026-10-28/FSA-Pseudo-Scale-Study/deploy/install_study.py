#!/usr/bin/env python3
"""Install committed study sources with backups; never run experiments."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

STUDY = "Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
SIBLING = "Experiments/FSA-ZoomNeXt"
ENTRYPOINTS = {"run_study.py", "run_plan.py", "profile_study.py", "main_for_fsa.py"}
PRESERVED = {"results", "datasets", "data", "checkpoints", "weights", ".venv", "__pycache__"}


def check_processes(work):
    """Refuse to replace source files used by a process in this destination."""
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = proc.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        for arg in args:
            if Path(arg).name not in ENTRYPOINTS:
                continue
            try:
                source = Path(arg) if Path(arg).is_absolute() else (proc.parent / "cwd").resolve(strict=True) / arg
            except OSError:
                continue
            if source.resolve().is_relative_to(work):
                raise ValueError(f"Experiment PID {proc.parent.name} is active; stop it before installing.")


def install(archive, work):
    archive, work = Path(archive).resolve(strict=True), Path(work).absolute()
    if work.resolve() != work or (work.exists() and not work.is_dir()):
        raise ValueError("--work must be a real directory path without symlink components")
    blob = archive.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    checksum = Path(str(archive) + ".sha256").read_text().split()
    if not checksum or digest != checksum[0].lower():
        raise ValueError("ZIP checksum mismatch; no files changed")
    check_processes(work)
    pending, seen = [], set()
    with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
        revision = bundle.comment.decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Expected a git archive with its full commit in the ZIP comment")
        required = {STUDY + "/complete_benchmark.py", STUDY + "/prepare_full_training.py",
                    STUDY + "/fsa_study/overlap.py", SIBLING + "/methods/backbone/pvt_v2_eff.py"}
        if not required.issubset(bundle.namelist()):
            raise ValueError("Required source files missing from ZIP; no files changed")
        for info in bundle.infolist():
            if info.is_dir():
                continue
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts or "\\" in info.filename:
                raise ValueError(f"Invalid ZIP path: {info.filename}")
            if not any(info.filename.startswith(prefix + "/") for prefix in (STUDY, SIBLING)):
                continue
            if any(part.lower() in PRESERVED for part in rel.parts):
                continue
            if rel.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}:
                continue
            if str(rel) in seen:
                raise ValueError(f"Duplicate ZIP path: {rel}")
            seen.add(str(rel))
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP contains a symlink: {rel}")
            target = work / str(rel)
            if target.resolve() != target or target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"Unexpected destination path: {target}; no files changed")
            data = bundle.read(info)
            if not target.exists() or target.read_bytes() != data:
                pending.append((target, data, mode & 0o777 or 0o644))
    backup_root = work / ".code-backups"
    if backup_root.is_symlink():
        raise ValueError("Backup directory must not be a symlink")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=revision[:12] + "-", dir=backup_root))
    for target, _, _ in pending:
        if target.exists():
            saved = backup / target.relative_to(work)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    receipt = {"commit": revision, "zip_sha256": digest, "target": str(work),
               "files": [str(t.relative_to(work)) for t, _, _ in pending],
               "status": "backed_up_before_install"}
    receipt_file = backup / "deployment.json"
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    for target, data, mode in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(data)
            temporary = Path(stream.name)
        try:
            temporary.chmod(mode)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    receipt["status"] = "installed"
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "backup": str(backup), "active_study": str(work / STUDY)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--work", default="/home/master/mithila/shMin")
    args = parser.parse_args()
    try:
        result = install(args.archive, args.work)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2))
    print("Source installation only. Existing results, datasets, checkpoints and environments preserved.")


if __name__ == "__main__":
    main()
