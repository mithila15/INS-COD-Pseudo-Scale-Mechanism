"""Recovery plans audit real fixture files without loading or training any model."""
import copy
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from fsa_study.data import manifest_digest, sha256_file
import recover_evaluation as recovery
from run_plan import execute_plan, read_plan
from run_study import build_parser

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def original(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index, split in enumerate(("train", "val", "test")):
        image, mask = data / f"{split}.jpg", data / f"{split}.png"
        Image.new("RGB", (10 if split == "test" else 8, 6), (10 + index, 20, 30)).save(image)
        Image.new("L", (8, 6), 255).save(mask)
        source = "chameleon_test" if split == "test" else "camo_train"
        records.append({
            "id": f"{source}/{split}", "dataset": "chameleon" if split == "test" else "camo",
            "source": source, "split": split, "image": image.name, "mask": mask.name,
            "image_sha256": sha256_file(image), "mask_sha256": sha256_file(mask),
        })
    manifest = {"schema_version": 1, "data_root": str(data), "records": records,
                "source_counts": dict(Counter(record["source"] for record in records)),
                "split_counts": dict(Counter(f"{record['dataset']}/{record['split']}" for record in records))}
    manifest["content_sha256"] = manifest_digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    jobs = []
    for seed in (7, 8):
        run = tmp_path / "runs" / f"single__seed{seed}"
        base = [sys.executable, str(HERE / "run_study.py")]
        jobs.append({
            "run": str(run), "variant": "single", "seed": seed,
            "train": base + ["train", "--manifest", str(manifest_path), "--output", str(run),
                             "--seed", str(seed), "--baseline", "--random-init", "--device", "cpu"],
            "evaluate": base + ["evaluate", "--manifest", str(manifest_path), "--output", str(run / "test"),
                                "--checkpoint", str(run / "best.pth"), "--split", "test", "--device", "cpu"],
            "profile": [sys.executable, str(HERE / "profile_study.py"), "--checkpoint", str(run / "best.pth"),
                        "--output", str(run / "profile.json"), "--device", "cpu"],
        })
    plan = {"jobs": jobs, "run_count": 2, "variant_count": 1, "seeds": [7, 8],
            "results": str(tmp_path / "runs"), "evaluation_subdir": "test", "profile_name": "profile.json"}
    path = tmp_path / "original-plan.json"
    path.write_text(json.dumps(plan))
    run = Path(jobs[0]["run"])
    run.mkdir(parents=True)
    config = vars(build_parser().parse_args(jobs[0]["train"][2:]))
    config.update({"manifest_sha256": manifest["content_sha256"], "model_config": {"seed": 7}})
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"status": "training_complete"}))
    (run / "best.pth").write_bytes(b"fixture-checkpoint-never-load")
    (run / "test").mkdir()
    (run / "test" / "partial.txt").write_text("preserve failed evaluation")
    return path, plan, manifest_path, data


def test_recovery_preserves_training_and_partial_evaluation_and_caches_audit(original, tmp_path, monkeypatch):
    path, old, manifest_path, data = original
    original_bytes = path.read_bytes()
    output = tmp_path / "plans" / "recovered"
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls = []
    original_audit = recovery.audit_geometry
    original_load = recovery.load_manifest

    def audited(*args, **kwargs):
        calls.append(("audit", kwargs))
        return original_audit(*args, **kwargs)

    def loaded(*args, **kwargs):
        calls.append(("load", kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(recovery, "audit_geometry", audited)
    monkeypatch.setattr(recovery, "load_manifest", loaded)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Recovery executed a subprocess"))
    plan = recovery.recover_plan(path, output)
    assert read_plan(output / "plan.json") == plan
    assert path.read_bytes() == original_bytes
    assert all(p.read_bytes() == before for p, before in snapshots.items())
    assert sorted(p.name for p in output.iterdir()) == ["plan.json", "run_all.sh"]
    assert [name for name, _ in calls] == ["load", "audit"]
    assert calls[0][1]["verify_files"] is True
    assert calls[1][1] == {"split": "test", "policy": "resize-prediction-to-gt"}
    assert plan["recovery"]["source_plan_sha256"] == sha256_file(path)
    assert [entry["state"] for entry in plan["recovery"]["training_states"]] == [
        "completed_training_reused", "training_not_started"]
    audit = plan["recovery"]["data_audits"][0]
    assert audit["test_datasets_selected"] == ["chameleon"]
    assert audit["geometry_audit"]["mismatch_count"] == 1
    assert audit["geometry_audit"]["checked_pairs"] == 1
    for before, after in zip(old["jobs"], plan["jobs"]):
        for key in ("train", "profile", "run", "variant", "seed"):
            assert before[key] == after[key]
        args = build_parser().parse_args(after["evaluate"][2:])
        assert args.test_geometry == "resize-prediction-to-gt"
        assert args.checkpoint == str(Path(before["run"]) / "best.pth")
        assert args.output == str(Path(before["run"]) / "test-geometry-v1")
        assert not Path(args.output).exists()
    assert not Path(old["jobs"][1]["run"]).exists()
    runner = shlex.split((output / "run_all.sh").read_text().splitlines()[-1])
    assert runner == [sys.executable, str(HERE / "run_plan.py"), "--plan", str(output / "plan.json"),
                      "--actions", "train", "evaluate", "profile", "--skip-completed"]


@pytest.mark.parametrize("relative", ["", " ", ".", "..", "../escape", "test/../escape", "/tmp/test",
                                       "best.pth", "config.json/child", "profile.json", "profile.json/child", "test", "test/recovered"])
def test_rejects_unsafe_or_reused_evaluation_directories_before_output(original, tmp_path, relative):
    output = tmp_path / "not-created" / "plan"
    with pytest.raises(ValueError):
        recovery.recover_plan(original[0], output, relative)
    assert not output.parent.exists()


@pytest.mark.parametrize("run_index,child", [(0, "recovery"), (1, ""), (1, "plans/recovery")])
def test_rejects_plan_destination_inside_training_runs_without_creating_outputs(original, run_index, child):
    path, plan, _, _ = original
    run = Path(plan["jobs"][run_index]["run"])
    output = run / child
    files_before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    directories_before = {p for p in path.parent.rglob("*") if p.is_dir()}
    with pytest.raises(ValueError, match="outside every training run"):
        recovery.recover_plan(path, output)
    assert {p for p in path.parent.rglob("*") if p.is_dir()} == directories_before
    assert {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == files_before
    assert not Path(plan["jobs"][1]["run"]).exists()


def test_rejects_new_evaluation_ancestor_of_old_evaluation_without_creating_outputs(original, tmp_path):
    path, plan, _, _ = original
    plan["evaluation_subdir"] = "evaluations/original"
    for job in plan["jobs"]:
        command = job["evaluate"]
        command[command.index("--output") + 1] = str(Path(job["run"]) / "evaluations/original")
    path.write_text(json.dumps(plan))
    output = tmp_path / "recovered"
    with pytest.raises(ValueError, match="preserve the original evaluation"):
        recovery.recover_plan(path, output, "evaluations")
    assert not output.exists()
    assert not Path(plan["jobs"][1]["run"]).exists()


def test_incomplete_training_is_preserved_and_rejected(original, tmp_path):
    path, plan, _, _ = original
    run = Path(plan["jobs"][1]["run"])
    run.mkdir()
    (run / "history.jsonl").write_text('{"epoch": 1}\n')
    output = tmp_path / "recovered"
    with pytest.raises(ValueError, match="Incomplete training"):
        recovery.recover_plan(path, output)
    assert (run / "history.jsonl").read_text() == '{"epoch": 1}\n'
    assert not output.exists()


def test_changed_completed_training_configuration_is_rejected(original, tmp_path):
    path, plan, _, _ = original
    config_path = Path(plan["jobs"][0]["run"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["epochs"] -= 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration disagrees"):
        recovery.recover_plan(path, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


def test_changed_dataset_file_is_rejected_before_plan_output(original, tmp_path):
    path, _, _, data = original
    Image.new("L", (8, 6), 0).save(data / "test.png")
    output = tmp_path / "recovered"
    with pytest.raises(ValueError, match="Changed mask content"):
        recovery.recover_plan(path, output)
    assert not output.exists()


def test_occupied_new_evaluation_target_and_existing_plan_are_preserved(original, tmp_path):
    path, plan, _, _ = original
    target = Path(plan["jobs"][0]["run"]) / "test-geometry-v1"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("keep")
    output = tmp_path / "recovered"
    with pytest.raises(FileExistsError, match="evaluation output already exists"):
        recovery.recover_plan(path, output)
    assert marker.read_text() == "keep" and not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError, match="plan output already exists"):
        recovery.recover_plan(path, output)
    assert list(output.iterdir()) == []


def test_requires_absolute_paths(original, tmp_path):
    with pytest.raises(ValueError, match="--plan must be an absolute"):
        recovery.recover_plan("relative.json", tmp_path / "recovered")
    with pytest.raises(ValueError, match="--output must be an absolute"):
        recovery.recover_plan(original[0], "relative")


def test_help_imports_no_torch_and_runs_no_training():
    code = ("import runpy, sys; sys.argv = ['recover_evaluation.py', '--help']; "
            f"sys.path.insert(0, {str(HERE)!r}); "
            "\ntry: runpy.run_module('recover_evaluation', run_name='__main__')"
            "\nexcept SystemExit as error: assert error.code == 0"
            "\nassert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--evaluation-subdir" in result.stdout


def test_existing_geometry_option_is_replaced_without_duplicates(original, tmp_path):
    path, plan, _, _ = original
    revised = copy.deepcopy(plan)
    for job in revised["jobs"]:
        job["evaluate"].extend(["--test-geometry", "strict"])
    path.write_text(json.dumps(revised))
    recovered = recovery.recover_plan(path, tmp_path / "recovered", "test-geometry-v2")
    for job in recovered["jobs"]:
        assert job["evaluate"].count("--test-geometry") == 1
        args = build_parser().parse_args(job["evaluate"][2:])
        assert args.test_geometry == "resize-prediction-to-gt"
        assert args.output.endswith("/test-geometry-v2")


def test_runner_skips_completed_baseline_then_continues_remaining_jobs_without_executing(original, tmp_path):
    output = tmp_path / "recovered"
    plan = recovery.recover_plan(original[0], output)
    calls = []

    def record_only(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    result = execute_plan(output / "plan.json", skip_completed=True, run_command=record_only)
    assert result == 0
    actual_actions = [command for command in calls if not command[1].endswith("summarize_study.py")]
    first, second = plan["jobs"]
    assert actual_actions == [first["evaluate"], first["profile"], second["train"],
                              second["evaluate"], second["profile"]]
    assert first["train"] not in calls
    assert not Path(second["run"]).exists()
    assert not (Path(first["run"]) / "test-geometry-v1").exists()
