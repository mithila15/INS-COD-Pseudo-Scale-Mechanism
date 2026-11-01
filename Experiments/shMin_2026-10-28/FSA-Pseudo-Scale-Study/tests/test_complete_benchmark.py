"""Complete-benchmark preparation/orchestration checks; no inference or training."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from fsa_study import benchmark
from fsa_study.data import SOURCES, geometry_protocol, prepare_manifest, sha256_file
from fsa_study.metrics import METRIC_KEYS, METRIC_PROTOCOL, protocol_details
from complete_benchmark import complete_benchmark_plan, main
from run_plan import completed, execute_plan, read_plan
from run_study import build_parser

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "EXPECTED_TEST_COUNTS", {name: 2 for name in benchmark.EXPECTED_TEST_COUNTS})
    root = tmp_path / "data"
    serial = 1
    for source, (name, split, images, masks, _) in SOURCES.items():
        for index in range(4 if split == "train" else 2):
            stem = ("COD10K-CAM-" if name == "cod10k" else "") + f"{source}-{index}"
            for kind, folder in (("image", images), ("mask", masks)):
                path = root / folder / f"{stem}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB" if kind == "image" else "L", (8, 6), serial if kind == "image" else 255).save(path)
            serial += 1
    manifest_path = tmp_path / "original-manifest.json"
    manifest = prepare_manifest(root, manifest_path, allow_nonstandard_counts=True,
                                test_datasets=["camo", "chameleon", "nc4k"])
    jobs = []
    for seed in (7, 8):
        run = tmp_path / "runs" / f"single__seed{seed}"
        run.mkdir(parents=True)
        base = [sys.executable, str(HERE / "run_study.py")]
        job = {"run": str(run), "variant": "single", "seed": seed,
               "train": base + ["train", "--manifest", str(manifest_path), "--output", str(run), "--seed", str(seed),
                                "--baseline", "--random-init", "--device", "cpu"],
               "evaluate": base + ["evaluate", "--manifest", str(manifest_path), "--output", str(run / "test"),
                                   "--checkpoint", str(run / "best.pth"), "--split", "test", "--device", "cpu"],
               "profile": [sys.executable, str(HERE / "profile_study.py"), "--checkpoint", str(run / "best.pth"),
                           "--output", str(run / "profile.json"), "--device", "cpu"]}
        config = vars(build_parser().parse_args(job["train"][2:]))
        config.update({"manifest_sha256": manifest["content_sha256"], "model_config": {"seed": seed}})
        (run / "config.json").write_text(json.dumps(config))
        (run / "manifest.json").write_bytes(manifest_path.read_bytes())
        (run / "status.json").write_text(json.dumps({"status": "training_complete"}))
        (run / "best.pth").write_bytes(f"synthetic completed checkpoint {seed}".encode())
        (run / "profile.json").write_text("old profile retained")
        (run / "test").mkdir()
        (run / "test/old-metrics.json").write_text("old results retained")
        jobs.append(job)
    plan = {"jobs": jobs, "run_count": 2, "results": str(tmp_path / "runs"), "seeds": [7, 8],
            "evaluation_subdir": "test", "profile_name": "profile.json"}
    path = tmp_path / "original-plan.json"
    path.write_text(json.dumps(plan))
    return path, plan, manifest_path, root


def test_preparation_preserves_all_original_data_checkpoints_and_commands(original, tmp_path, monkeypatch):
    source, original_plan, manifest_path, root = original
    before = {path: sha256_file(path) for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("preparation started a subprocess"))
    output = tmp_path / "new-plan"
    plan = complete_benchmark_plan(source, output)
    assert plan["evaluation_only"]
    assert plan["status"] == "commands_only_not_executed"
    assert plan["benchmark_repair"]["manifest_extension"]["benchmark_coverage"]["complete"]
    assert plan["benchmark_repair"]["geometry_audit"]["checked_pairs"] == 8
    for old, new in zip(original_plan["jobs"], plan["jobs"]):
        assert new["run"] == old["run"]
        assert new["train"] == old["train"]
        assert new["profile"] == old["profile"]
        args = build_parser().parse_args(new["evaluate"][2:])
        assert args.manifest == str(output / "manifest.json")
        assert args.training_manifest == str(manifest_path)
        assert args.test_geometry == "resize-prediction-to-gt"
        assert args.data_root is None
        assert completed(new, "train")
        assert not Path(args.output).exists()
    assert all(sha256_file(path) == digest for path, digest in before.items())
    assert read_plan(output / "plan.json")["jobs"] == plan["jobs"]
    assert "--actions evaluate --skip-completed" in (output / "run_evaluation.sh").read_text()
    assert not (output / "run_all.sh").exists()


def test_new_script_syntax_and_dry_run_only_print_evaluation(original, tmp_path, capsys):
    source, _, _, _ = original
    output = tmp_path / "new-plan"
    complete_benchmark_plan(source, output)
    assert subprocess.run(["bash", "-n", str(output / "run_evaluation.sh")]).returncode == 0
    assert execute_plan(output / "plan.json", actions=["evaluate"], dry_run=True,
                        run_command=lambda *a, **k: pytest.fail("executed")) == 0
    text = capsys.readouterr().out
    assert "run_study.py evaluate" in text
    assert "run_study.py train" not in text
    assert "profile_study.py" not in text


@pytest.mark.parametrize("actions", [("train",), ("profile",), ("train", "evaluate", "profile")])
def test_evaluation_only_plan_rejects_training_or_profile_before_subprocess(original, tmp_path, actions):
    source, _, _, _ = original
    output = tmp_path / "new-plan"
    complete_benchmark_plan(source, output)
    with pytest.raises(ValueError, match="evaluation-only"):
        execute_plan(output / "plan.json", actions=actions, run_command=lambda *a, **k: pytest.fail("executed"))


def test_only_evaluation_and_summary_commands_execute(original, tmp_path):
    source, _, _, _ = original
    output = tmp_path / "new-plan"
    plan = complete_benchmark_plan(source, output)
    calls = []
    assert execute_plan(output / "plan.json", actions=["evaluate"], skip_completed=True,
                        run_command=lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0)) == 0
    assert [command for command in calls if command[1].endswith("run_study.py")] == [job["evaluate"] for job in plan["jobs"]]
    assert all(command[1].endswith("summarize_study.py") or command[2] == "evaluate" for command in calls)


@pytest.mark.parametrize("state", ["missing_run", "missing_checkpoint", "incomplete_training"])
def test_all_runs_must_be_completed_before_writing_plan(original, tmp_path, state):
    import shutil
    source, plan, _, _ = original
    run = Path(plan["jobs"][-1]["run"])
    if state == "missing_run":
        shutil.rmtree(run)
    elif state == "missing_checkpoint":
        (run / "best.pth").unlink()
    else:
        (run / "status.json").write_text(json.dumps({"status": "training"}))
    output = tmp_path / "new-plan"
    with pytest.raises(ValueError, match="Every run must have completed"):
        complete_benchmark_plan(source, output)
    assert not output.exists()


@pytest.mark.parametrize("relative", ["test", "test/nested", "..", "/absolute", "best.pth", "profile.json", "."])
def test_unsafe_or_overlapping_evaluation_destination_rejected(original, tmp_path, relative):
    source, _, _, _ = original
    with pytest.raises(ValueError):
        complete_benchmark_plan(source, tmp_path / "new-plan", relative)
    assert not (tmp_path / "new-plan").exists()


def test_existing_new_evaluation_and_plan_directories_are_not_overwritten(original, tmp_path):
    source, plan, _, _ = original
    target = Path(plan["jobs"][0]["run"]) / "test-benchmark-v1"
    target.mkdir()
    (target / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="New evaluation output"):
        complete_benchmark_plan(source, tmp_path / "new-plan")
    assert (target / "keep.txt").read_text() == "preserve"
    with pytest.raises(FileExistsError, match="plan output already exists"):
        complete_benchmark_plan(source, tmp_path)


def test_plan_directory_inside_run_is_rejected(original, tmp_path):
    source, plan, _, _ = original
    with pytest.raises(ValueError, match="outside every training run"):
        complete_benchmark_plan(source, Path(plan["jobs"][0]["run"]) / "new-plan")


def test_dangling_plan_destination_symlink_is_preserved(original, tmp_path):
    source, _, _, _ = original
    target = tmp_path / "absent-target"
    output = tmp_path / "new-plan"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="existing symlink"):
        complete_benchmark_plan(source, output)
    assert output.is_symlink()
    assert output.readlink() == target
    assert not target.exists()


def test_root_override_is_retained_only_for_original_training_data(original, tmp_path):
    source, plan, _, root = original
    for job in plan["jobs"]:
        job["train"].extend(["--data-root", str(root)])
        job["evaluate"].extend(["--data-root", str(root)])
    source.write_text(json.dumps(plan))
    prepared = complete_benchmark_plan(source, tmp_path / "new-plan")
    args = build_parser().parse_args(prepared["jobs"][0]["evaluate"][2:])
    assert args.training_data_root == str(root)
    assert args.data_root is None


def _write_completed_evaluation(job, manifest_path, original_path):
    manifest = json.loads(Path(manifest_path).read_text())
    training = json.loads(Path(original_path).read_text())
    args = build_parser().parse_args(job["evaluate"][2:])
    target = Path(args.output)
    target.mkdir()
    proof = benchmark.validate_test_extension(training, training["data_root"], manifest, manifest["data_root"], verify_files=False)
    artifacts = {"per_image_metrics": "per_image.csv", "curves": {"fixture": "curves.csv"}, "predictions_saved": True,
                 "geometry_audit": "geometry_audit.json", "sha256": {}}
    for name, value in (("manifest.json", manifest), ("training_manifest.json", training),
                        ("geometry_audit.json", {"policy": "resize-prediction-to-gt"})):
        (target / name).write_text(json.dumps(value))
        artifacts["sha256"][name] = sha256_file(target / name)
    for name in ("per_image.csv", "curves.csv"):
        (target / name).write_text("synthetic artifact for metadata-only verification")
        artifacts["sha256"][name] = sha256_file(target / name)
    result = {"checkpoint_sha256": sha256_file(Path(job["run"]) / "best.pth"), "split": "test",
              "manifest_sha256": manifest["content_sha256"], "training_manifest_sha256": training["content_sha256"],
              "model_config": {"seed": job["seed"]}, "manifest_extension": proof,
              "benchmark_coverage": proof["benchmark_coverage"],
              "geometry_protocol": geometry_protocol("resize-prediction-to-gt"),
              "geometry_audit": {"policy": "resize-prediction-to-gt"},
              "datasets": {name: {**dict.fromkeys(METRIC_KEYS, .5), "count": count}
                           for name, count in proof["benchmark_coverage"]["observed_counts"].items()},
              "metric_protocol": METRIC_PROTOCOL, "metric_protocol_details": protocol_details(), "artifacts": artifacts}
    (target / "metrics.json").write_text(json.dumps(result))
    return target, result


def test_extension_completion_verifies_both_manifests_snapshots_and_counts(original, tmp_path):
    source, _, original_path, _ = original
    output = tmp_path / "new-plan"
    plan = complete_benchmark_plan(source, output)
    job = plan["jobs"][0]
    target, result = _write_completed_evaluation(job, output / "manifest.json", original_path)
    assert completed(job, "evaluate")
    result["datasets"]["cod10k"]["count"] -= 1
    (target / "metrics.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="complete audited benchmark"):
        completed(job, "evaluate")


@pytest.mark.parametrize("field", ["manifest_sha256", "training_manifest_sha256", "manifest_extension"])
def test_changed_extension_identity_rejected(original, tmp_path, field):
    source, _, original_path, _ = original
    output = tmp_path / "new-plan"
    plan = complete_benchmark_plan(source, output)
    job = plan["jobs"][0]
    target, result = _write_completed_evaluation(job, output / "manifest.json", original_path)
    result[field] = "changed"
    (target / "metrics.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="metadata disagrees|extension provenance"):
        completed(job, "evaluate")


@pytest.mark.parametrize("name", ["manifest.json", "training_manifest.json"])
def test_changed_saved_manifest_snapshot_rejected(original, tmp_path, name):
    source, _, original_path, _ = original
    output = tmp_path / "new-plan"
    plan = complete_benchmark_plan(source, output)
    job = plan["jobs"][0]
    target, _ = _write_completed_evaluation(job, output / "manifest.json", original_path)
    (target / name).write_text("changed")
    with pytest.raises(ValueError, match="manifest snapshot"):
        completed(job, "evaluate")


def test_cli_reports_commands_only_and_full_benchmark(original, tmp_path, capsys):
    source, _, _, _ = original
    assert main(["--plan", str(source), "--output", str(tmp_path / "new-plan")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "commands_only_not_executed"
    assert report["evaluation_only"]
    assert report["benchmark_coverage"]["complete"]


def test_isolated_code_rebinding_preserves_source_plan_and_completed_run_identity(original, tmp_path):
    source, plan, _, _ = original
    for job in plan["jobs"]:
        for action in ("train", "evaluate", "profile"):
            job[action][1] = str(tmp_path / "original-code" / Path(job[action][1]).name)
    source.write_text(json.dumps(plan))
    before = source.read_bytes()
    output = tmp_path / "new-plan"
    with pytest.raises(ValueError, match="intended study entrypoint"):
        complete_benchmark_plan(source, output)
    rebound = complete_benchmark_plan(source, output, rebind_entrypoints=True)
    assert source.read_bytes() == before
    assert (output / "source_plan.json").read_bytes() == before
    assert rebound["benchmark_repair"]["source_plan_sha256"] == sha256_file(source)
    assert rebound["benchmark_repair"]["entrypoint_rebinding"]["enabled"]
    for old, new in zip(plan["jobs"], rebound["jobs"]):
        for action in ("train", "profile"):
            assert old[action][0] == new[action][0]
            assert old[action][2:] == new[action][2:]
            assert Path(new[action][1]).parent == HERE
        assert completed(new, "train")
    assert read_plan(output / "plan.json")["jobs"] == rebound["jobs"]
    with pytest.raises(ValueError, match="evaluation-only"):
        execute_plan(output / "plan.json", run_command=lambda *a, **k: pytest.fail("executed"))
