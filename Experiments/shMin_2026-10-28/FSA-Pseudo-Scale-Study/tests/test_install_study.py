import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("study_installer", Path(__file__).resolve().parents[1] / "deploy/install_study.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def bundle(tmp_path, extra=None):
    archive = tmp_path / "source.zip"
    members = {
        installer.STUDY + "/complete_benchmark.py": b"updated",
        installer.STUDY + "/prepare_full_training.py": b"prepare",
        installer.STUDY + "/fsa_study/overlap.py": b"overlap",
        installer.SIBLING + "/methods/backbone/pvt_v2_eff.py": b"backbone",
    }
    members.update(extra or {})
    with zipfile.ZipFile(archive, "w") as stream:
        stream.comment = b"a" * 40
        for path, contents in members.items():
            stream.writestr(path, contents)
    Path(str(archive) + ".sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  source.zip\n")
    return archive


def test_install_preserves_inputs_and_backs_up_changed_source(tmp_path):
    work = tmp_path / "work"
    relative = installer.STUDY + "/complete_benchmark.py"
    target = work / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old source")
    preserved = [installer.STUDY + "/results/run/metrics.json", installer.STUDY + "/data/image.jpg",
                 installer.STUDY + "/best.pth", ".venv/bin/python", "Datasets/COD/test.jpg",
                 installer.STUDY + "/pcd.py"]
    for path in preserved:
        file = work / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"preserve me")
    archive = bundle(tmp_path, {path: b"do not install" for path in preserved[:-1]})
    result = installer.install(archive, work)
    assert target.read_bytes() == b"updated"
    assert (Path(result["backup"]) / relative).read_bytes() == b"old source"
    assert all((work / path).read_bytes() == b"preserve me" for path in preserved)
    assert result["status"] == "installed"
    assert installer.install(archive, work)["files"] == []


def test_checksum_failure_does_not_create_destination(tmp_path):
    archive = bundle(tmp_path)
    Path(str(archive) + ".sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        installer.install(archive, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_path_traversal_rejected_before_writes(tmp_path):
    archive = bundle(tmp_path, {installer.STUDY + "/../../escape.py": b"bad"})
    with pytest.raises(ValueError, match="Invalid ZIP path"):
        installer.install(archive, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_destination_symlink_rejected_before_writes(tmp_path):
    archive = bundle(tmp_path)
    work = tmp_path / "work"
    (work / installer.STUDY).mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / installer.STUDY / "fsa_study").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unexpected destination"):
        installer.install(archive, work)
    assert not (work / installer.STUDY / "complete_benchmark.py").exists()
    assert not list(outside.iterdir())


def test_active_process_blocks_before_changes(tmp_path, monkeypatch):
    archive = bundle(tmp_path)
    def active(_work):
        raise ValueError("Experiment PID 123 is active")
    monkeypatch.setattr(installer, "check_processes", active)
    with pytest.raises(ValueError, match="active"):
        installer.install(archive, tmp_path / "work")
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("source_dir,blocked", [("branch", True), ("main", False)])
def test_process_guard_is_scoped_to_install_destination(tmp_path, monkeypatch, source_dir, blocked):
    work = tmp_path / "branch"
    proc = tmp_path / "fake-proc" / "123" / "cmdline"
    proc.parent.mkdir(parents=True)
    proc.write_bytes(f"python\0{tmp_path / source_dir / 'run_study.py'}\0train\0".encode())
    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([proc]) if self == Path("/proc")
                        else real_glob(self, pattern))
    if blocked:
        with pytest.raises(ValueError, match="PID 123 is active"):
            installer.check_processes(work)
    else:
        installer.check_processes(work)
