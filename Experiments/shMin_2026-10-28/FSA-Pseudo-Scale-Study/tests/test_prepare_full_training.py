"""New full-training plans inspect generated files only; no model is trained."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from fsa_study.benchmark import EXPECTED_TEST_COUNTS, extend_test_manifest
from fsa_study.data import prepare_manifest, sha256_file
from fsa_study.training_protocol import training_metadata
import prepare_full_training as preparation
import plan_study
from run_plan import read_plan
from run_study import build_parser
from test_data import fixture

HERE = Path(__file__).resolve().parents[1]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def original(tmp_path, monkeypatch):
    data = tmp_path / "data"
    fixture(data, tests=tuple(EXPECTED_TEST_COUNTS))
    for dataset in EXPECTED_TEST_COUNTS:
        monkeypatch.setitem(EXPECTED_TEST_COUNTS, dataset, 2)
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_manifest(data, manifest_path, allow_nonstandard_counts=True,
                                test_datasets=list(EXPECTED_TEST_COUNTS))
    weights = tmp_path / "backbone.pth"
    weights.write_bytes(b"initial-weights-never-loaded")
    jobs = []
    for variant, seed, flags in [("single", 3407, ["--baseline"]),
                                 ("native_conv__asff", 3407, ["--generator", "native_conv", "--utilizer", "asff"])]:
        run = tmp_path / "old-results" / "runs" / "priority8" / f"{variant}__seed{seed}"
        base = [sys.executable, str(HERE / "run_study.py")]
        jobs.append({"variant": variant, "seed": seed, "run": str(run),
                     "train": base + ["train", "--manifest", str(manifest_path), "--data-root", str(data),
                                       "--output", str(run), "--seed", str(seed), "--epochs", "150",
                                       "--batch-size", "4", "--channels", "48", "--size", "384",
                                       "--lr", "0.0001", "--workers", "2", "--device", "cpu",
                                       "--backbone-weights", str(weights), *flags],
                     "evaluate": base + ["evaluate", "--manifest", str(manifest_path), "--data-root", str(data),
                                          "--checkpoint", str(run / "best.pth"), "--split", "test",
                                          "--output", str(run / "old-test"), "--device", "cpu"],
                     "profile": [sys.executable, str(HERE / "profile_study.py"), "--checkpoint", str(run / "best.pth"),
                                 "--output", str(run / "profile.json"), "--device", "cpu", "--repeats", "73"]})
    plan = {"phase": "priority8", "jobs": jobs, "run_count": 2, "variant_count": 2, "seeds": [3407],
            "results": str(Path(jobs[0]["run"]).parent), "evaluation_subdir": "old-test", "profile_name": "profile.json"}
    path = tmp_path / "old-results" / "plans" / "priority8" / "plan.json"
    _write(path, plan)
    first_run = Path(jobs[0]["run"])
    first_run.mkdir(parents=True)
    (first_run / "best.pth").write_bytes(b"old-selected-model-never-loaded")
    (first_run / "history.jsonl").write_text('{"epoch": 150}\n')
    (first_run / "old-test").mkdir()
    (first_run / "old-test" / "preserved.txt").write_text("old scores")
    return path, plan, manifest_path, manifest, data


def test_prepares_fresh_runs_preserving_inputs_and_training_settings(original, tmp_path, monkeypatch):
    source, old, manifest_path, manifest, _ = original
    snapshots = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Preparation executed a subprocess"))
    destination = tmp_path / "new-full-training"
    plan = preparation.prepare_full_training_plan(source, manifest_path, destination)
    plan_file = destination / "plans/priority8/plan.json"
    assert read_plan(plan_file) == plan
    assert (plan_file.parent / "source_plan.json").read_bytes() == snapshots[source]
    assert all(path.read_bytes() == before for path, before in snapshots.items())
    assert not (destination / "runs").exists()
    assert not Path(old["jobs"][1]["run"]).exists()
    generated = json.loads((destination / "manifest.json").read_text())
    assert training_metadata(generated, "full_train")["training_sample_count"] == 8
    assert training_metadata(generated, "full_train")["validation_sample_count"] == 0
    assert [record for record in generated["records"] if record["split"] == "test"] == [
        record for record in manifest["records"] if record["split"] == "test"]
    parser = build_parser()
    for before, after in zip(old["jobs"], plan["jobs"]):
        original_args = vars(parser.parse_args(before["train"][2:]))
        new_args = vars(parser.parse_args(after["train"][2:]))
        assert {key: value for key, value in original_args.items() if key not in preparation.CHANGED_TRAIN_SETTINGS} == {
            key: value for key, value in new_args.items() if key not in preparation.CHANGED_TRAIN_SETTINGS}
        assert new_args["training_protocol"] == "full_train"
        assert new_args["data_root"] is None
        assert new_args["manifest"] == str(destination / "manifest.json")
        expected = destination / "runs/priority8" / f"{after['variant']}__seed{after['seed']}"
        assert after["run"] == str(expected)
        evaluation = parser.parse_args(after["evaluate"][2:])
        assert evaluation.data_root is None and evaluation.training_manifest is None
        assert evaluation.training_data_root is None
        assert evaluation.test_geometry == "resize-prediction-to-gt"
        assert evaluation.checkpoint == str(expected / "best.pth")
        assert evaluation.output == str(expected / "test-four-benchmarks")
        assert after["profile"][after["profile"].index("--repeats") + 1] == "73"
    audit = plan["full_training"]
    assert audit["source_plan_sha256"] == sha256_file(source)
    assert audit["source_manifest_file_sha256"] == sha256_file(manifest_path)
    assert audit["benchmark_coverage"]["complete"]
    assert audit["training_geometry"]["checked_pairs"] == 8
    assert audit["test_geometry"]["checked_pairs"] == 8
    assert audit["execution"] == "not_started"
    assert plan["status"] == "commands_only_not_executed"


def test_generated_scripts_are_explicit_and_bash_syntax_valid(original, tmp_path):
    destination = tmp_path / "new-full-training"
    preparation.prepare_full_training_plan(original[0], original[2], destination)
    folder = destination / "plans/priority8"
    for filename, actions in [("run_all.sh", ["train", "evaluate", "profile"]), ("train.sh", ["train"])]:
        script = folder / filename
        process = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert shlex.split(script.read_text().splitlines()[-1]) == [
            sys.executable, str(HERE / "run_plan.py"), "--plan", str(folder / "plan.json"),
            "--actions", *actions, "--skip-completed"]


def test_generated_priority8_plan_keeps_all_variants_seed_and_schedule(original, tmp_path, monkeypatch):
    _, _, manifest, _, _ = original
    source_folder = tmp_path / "generated-priority8"
    monkeypatch.setattr(sys, "argv", ["plan_study.py", "--phase", "priority8", "--seeds", "3407",
                                     "--manifest", str(manifest), "--output", str(source_folder),
                                     "--results", str(tmp_path / "original-priority8-runs"),
                                     "--python", sys.executable, "--epochs", "150", "--batch-size", "4",
                                     "--backbone-weights", str(tmp_path / "backbone.pth")])
    plan_study.main()
    source_file = source_folder / "plan.json"
    before = source_file.read_bytes()
    fresh = preparation.prepare_full_training_plan(source_file, manifest, tmp_path / "fresh-priority8")
    assert source_file.read_bytes() == before
    assert fresh["run_count"] == fresh["variant_count"] == 8
    assert [job["variant"] for job in fresh["jobs"]] == [
        "single", "native_conv__asff", "native_conv__sk", "native_conv__mean", "projection__mean",
        "dysample__asff", "dysample__sk", "dysample__mean"]
    for job in fresh["jobs"]:
        arguments = build_parser().parse_args(job["train"][2:])
        assert arguments.epochs == 150 and arguments.batch_size == 4 and arguments.seed == 3407
        assert arguments.training_protocol == "full_train"
        assert not Path(job["run"]).exists()


def test_test_extension_inputs_retain_proof_but_remove_old_evaluation_flags(original, tmp_path):
    source, plan, _, _, data = original
    old_manifest = tmp_path / "partial-manifest.json"
    prepare_manifest(data, old_manifest, allow_nonstandard_counts=True,
                     test_datasets=["camo", "chameleon", "nc4k"])
    complete_path = tmp_path / "extended.json"
    extend_test_manifest(old_manifest, complete_path)
    for job in plan["jobs"]:
        preparation._put(job["train"], "--manifest", old_manifest)
        preparation._put(job["evaluate"], "--manifest", complete_path)
        job["evaluate"] += ["--training-manifest", str(old_manifest), "--training-data-root", str(data)]
    plan["evaluation_only"] = True
    plan["recovery"] = {"training_states": "historical only"}
    plan["benchmark_repair"] = {"training_states": [{"state": "completed_training_reused"}]}
    _write(source, plan)
    result = preparation.prepare_full_training_plan(source, complete_path, tmp_path / "fresh")
    assert "evaluation_only" not in result and "recovery" not in result
    assert "benchmark_repair" not in result
    assert result["full_training"]["source_inputs"][0]["extension_proof"]["added_records"] == 2
    for job in result["jobs"]:
        assert "--training-manifest" not in job["evaluate"]
        assert "--training-data-root" not in job["evaluate"]
        assert "--data-root" not in job["evaluate"]


@pytest.mark.parametrize("which", ["plan", "manifest", "output"])
def test_requires_absolute_paths(original, tmp_path, which):
    values = {"plan": original[0], "manifest": original[2], "output": tmp_path / "fresh"}
    values[which] = "relative-path"
    with pytest.raises(ValueError, match="absolute path"):
        preparation.prepare_full_training_plan(values["plan"], values["manifest"], values["output"])
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("which", ["existing", "inside_old_run", "future_old_run", "inside_old_results", "dangling_symlink"])
def test_rejects_overlap_and_existing_outputs_without_mutation(original, tmp_path, which):
    source, plan, manifest, _, _ = original
    output = tmp_path / "fresh"
    if which == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
    elif which == "inside_old_run":
        output = Path(plan["jobs"][0]["run"]) / "fresh"
    elif which == "future_old_run":
        output = Path(plan["jobs"][1]["run"])
    elif which == "inside_old_results":
        output = Path(plan["results"]) / "fresh"
    else:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    directories = {path for path in tmp_path.rglob("*") if path.is_dir()}
    with pytest.raises((ValueError, FileExistsError)):
        preparation.prepare_full_training_plan(source, manifest, output)
    assert {path: path.read_bytes() for path in files} == files
    assert {path for path in tmp_path.rglob("*") if path.is_dir()} == directories


def test_rejects_incomplete_benchmark_without_creating_destination(original, tmp_path):
    source, plan, _, _, data = original
    partial_path = tmp_path / "partial.json"
    prepare_manifest(data, partial_path, allow_nonstandard_counts=True,
                     test_datasets=["camo", "chameleon", "nc4k"])
    output = tmp_path / "not-created" / "fresh"
    with pytest.raises(ValueError, match="complete four-dataset"):
        preparation.prepare_full_training_plan(source, partial_path, output)
    assert not output.parent.exists()


def test_changed_source_image_is_rejected_before_any_new_outputs(original, tmp_path):
    source, _, manifest, payload, data = original
    record = payload["records"][0]
    (data / record["image"]).write_bytes(b"changed")
    output = tmp_path / "not-created" / "fresh"
    with pytest.raises(ValueError, match="Changed image content"):
        preparation.prepare_full_training_plan(source, manifest, output)
    assert not output.parent.exists()


def test_missing_initialization_is_rejected_without_loading_or_writing_models(original, tmp_path):
    source, plan, manifest, _, _ = original
    command = plan["jobs"][0]["train"]
    Path(command[command.index("--backbone-weights") + 1]).unlink()
    output = tmp_path / "not-created" / "fresh"
    with pytest.raises(FileNotFoundError, match="backbone weights are missing"):
        preparation.prepare_full_training_plan(source, manifest, output)
    assert not output.parent.exists()


def test_original_plan_can_be_unstarted_or_partially_trained(original, tmp_path):
    source, plan, manifest, _, _ = original
    second = Path(plan["jobs"][1]["run"])
    second.mkdir()
    (second / "history.jsonl").write_text('{"epoch": 3}\n')
    snapshot = (second / "history.jsonl").read_bytes()
    new = preparation.prepare_full_training_plan(source, manifest, tmp_path / "fresh")
    assert len(new["jobs"]) == 2
    assert (second / "history.jsonl").read_bytes() == snapshot


def test_isolated_release_requires_explicit_rebinding_and_preserves_source_snapshot(original, tmp_path):
    source, plan, manifest, _, _ = original
    old_code = tmp_path / "old-code"
    for job in plan["jobs"]:
        for action in ("train", "evaluate", "profile"):
            job[action][1] = str(old_code / Path(job[action][1]).name)
    _write(source, plan)
    source_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="intended study entrypoint"):
        preparation.prepare_full_training_plan(source, manifest, tmp_path / "refused")
    destination = tmp_path / "fresh"
    result = preparation.prepare_full_training_plan(source, manifest, destination, rebind_entrypoints=True)
    assert result["full_training"]["entrypoints_rebound"] is True
    assert (destination / "plans/priority8/source_plan.json").read_bytes() == source_bytes
    for job in result["jobs"]:
        for action in ("train", "evaluate", "profile"):
            assert job[action][0] == sys.executable
            assert Path(job[action][1]).parent == HERE
    assert read_plan(destination / "plans/priority8/plan.json") == result


def test_help_imports_no_torch_and_executes_no_training():
    code = ("import runpy, sys; sys.argv=['prepare_full_training.py', '--help']; "
            f"sys.path.insert(0, {str(HERE)!r}); "
            "\ntry: runpy.run_module('prepare_full_training', run_name='__main__')"
            "\nexcept SystemExit as error: assert error.code == 0"
            "\nassert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--rebind-entrypoints" in result.stdout
