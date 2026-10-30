"""Orchestration tests substitute commands; no optimizer is ever executed."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from fsa_study.data import sha256_file
from run_plan import completed, execute_plan, read_plan
from run_study import build_parser

HERE = Path(__file__).resolve().parents[1]


def plan_fixture(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"content_sha256": "fixture"}))
    run = tmp_path / "runs" / "single__seed7"
    base = [sys.executable, str(HERE / "run_study.py")]
    job = {"run": str(run), "variant": "single", "seed": 7,
           "train": base + ["train", "--manifest", str(manifest), "--output", str(run),
                            "--seed", "7", "--baseline", "--random-init", "--device", "cpu"],
           "evaluate": base + ["evaluate", "--manifest", str(manifest), "--checkpoint", str(run / "best.pth"),
                               "--split", "test", "--output", str(run / "test"), "--device", "cpu"],
           "profile": [sys.executable, str(HERE / "profile_study.py"), "--checkpoint", str(run / "best.pth"),
                       "--output", str(run / "profile.json"), "--device", "cpu"]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"run_count": 1, "jobs": [job], "seeds": [7], "results": str(run.parent)}))
    return path, job


def completed_train(job):
    run = Path(job["run"])
    run.mkdir(parents=True)
    config = vars(build_parser().parse_args(job["train"][2:]))
    config.update({"manifest_sha256": "fixture", "model_config": {"seed": 7}})
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"status": "training_complete", "best_val_mae": .1}))
    (run / "best.pth").write_bytes(b"synthetic-checkpoint-identity-only")
    return run


def test_dry_run_has_no_side_effects(tmp_path, capsys):
    path, job = plan_fixture(tmp_path)
    assert execute_plan(path, dry_run=True, run_command=lambda *a, **k: pytest.fail("executed")) == 0
    assert not Path(job["run"]).exists()
    assert not (tmp_path / "leaderboard").exists()
    output = capsys.readouterr().out
    assert "train" in output and "evaluate" in output and "profile_study.py" in output
    assert "--overwrite" in output


def test_refresh_after_every_action_and_finish(tmp_path):
    path, job = plan_fixture(tmp_path)
    calls = []
    def fake_command(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": False}
        return SimpleNamespace(returncode=0)
    assert execute_plan(path, run_command=fake_command) == 0
    assert calls[1] == job["train"] and calls[3] == job["evaluate"] and calls[5] == job["profile"]
    assert len(calls) == 8  # initial, 3 stages + 3 refreshes, final
    assert all("summarize_study.py" in call[1] for call in (calls[0], calls[2], calls[4], calls[6], calls[7]))


def test_failure_preserves_outputs_refreshes_and_stops(tmp_path):
    path, job = plan_fixture(tmp_path)
    calls = []
    def fake_command(command, **kwargs):
        calls.append(command)
        if command == job["train"]:
            run = Path(job["run"])
            run.mkdir(parents=True)
            (run / "partial.txt").write_text("keep")
            return SimpleNamespace(returncode=9)
        return SimpleNamespace(returncode=0)
    assert execute_plan(path, run_command=fake_command) == 9
    assert len(calls) == 3 and calls[-1][1].endswith("summarize_study.py")
    assert (Path(job["run"]) / "partial.txt").read_text() == "keep"


def test_skip_completed_checks_matching_plan_and_checkpoint(tmp_path):
    path, job = plan_fixture(tmp_path)
    run = completed_train(job)
    assert completed(job, "train")
    assert not completed(job, "evaluate")
    (run / "test").mkdir()
    from fsa_study.metrics import MetricRecorder, METRIC_PROTOCOL, protocol_details
    recorder = MetricRecorder(run / "test")
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 255
    recorder.step(image, image, "fixture", "sample")
    datasets, artifacts = recorder.finish()
    metrics = {"checkpoint_sha256": sha256_file(run / "best.pth"), "split": "test",
               "manifest_sha256": "fixture", "model_config": {"seed": 7},
               "datasets": datasets, "artifacts": artifacts,
               "metric_protocol": METRIC_PROTOCOL, "metric_protocol_details": protocol_details()}
    (run / "test/metrics.json").write_text(json.dumps(metrics))
    assert completed(job, "evaluate")
    # An old strict evaluation must not be silently reused for an opt-in policy.
    job["evaluate"].extend(["--test-geometry", "resize-prediction-to-gt"])
    with pytest.raises(ValueError, match="different geometry policy"):
        completed(job, "evaluate")
    del job["evaluate"][-2:]
    calls = []
    assert execute_plan(path, actions=["train", "evaluate"], skip_completed=True,
                        run_command=lambda command, **k: calls.append(command) or SimpleNamespace(returncode=0)) == 0
    assert len(calls) == 2 and all(call[1].endswith("summarize_study.py") for call in calls)
    (run / "best.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="different checkpoint"):
        completed(job, "evaluate")


def test_skip_rejects_changed_training_configuration(tmp_path):
    _, job = plan_fixture(tmp_path)
    run = completed_train(job)
    path = run / "config.json"
    config = json.loads(path.read_text())
    config["epochs"] += 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration disagrees"):
        completed(job, "train")


def test_opt_in_completion_requires_saved_geometry_audit(tmp_path):
    from fsa_study.data import geometry_protocol
    _, job = plan_fixture(tmp_path)
    run = completed_train(job)
    job["evaluate"].extend(["--test-geometry", "resize-prediction-to-gt"])
    (run / "test").mkdir()
    metrics = {"checkpoint_sha256": sha256_file(run / "best.pth"), "split": "test",
               "manifest_sha256": "fixture", "model_config": {"seed": 7},
               "geometry_protocol": geometry_protocol("resize-prediction-to-gt")}
    (run / "test/metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="missing its geometry audit"):
        completed(job, "evaluate")


def test_plan_rejects_duplicate_run_destinations(tmp_path):
    path, job = plan_fixture(tmp_path)
    path.write_text(json.dumps({"jobs": [job, job], "run_count": 2}))
    with pytest.raises(ValueError, match="distinct absolute"):
        read_plan(path)


@pytest.mark.parametrize("corruption", ["action", "output", "checkpoint", "variant", "script"])
def test_preflight_rejects_plan_action_or_destination_mismatch_before_any_execution(tmp_path, corruption):
    path, job = plan_fixture(tmp_path)
    if corruption == "action":
        job["evaluate"] = job["train"]
    elif corruption == "output":
        job["train"][job["train"].index("--output") + 1] = str(tmp_path / "wrong")
    elif corruption == "checkpoint":
        job["profile"][job["profile"].index("--checkpoint") + 1] = str(tmp_path / "wrong.pth")
    elif corruption == "variant":
        job["variant"] = "legacy__mean"
    else:
        job["profile"][1] = str(tmp_path / "wrong.py")
    path.write_text(json.dumps({"jobs": [job], "run_count": 1}))
    with pytest.raises(ValueError):
        execute_plan(path, actions=["evaluate"], run_command=lambda *a, **k: pytest.fail("executed"))


def test_skip_rejects_old_metrics_and_stale_profile_conditions(tmp_path):
    _, job = plan_fixture(tmp_path)
    run = completed_train(job)
    identity = {"checkpoint_sha256": sha256_file(run / "best.pth")}
    (run / "test").mkdir()
    (run / "test/metrics.json").write_text(json.dumps({**identity, "split": "test", "manifest_sha256": "fixture",
                                                      "model_config": {"seed": 7}, "metric_protocol": "official_py_sod_metrics_uint8"}))
    with pytest.raises(ValueError, match="older/different metric protocol"):
        completed(job, "evaluate")
    (run / "profile.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="older/different complexity protocol"):
        completed(job, "profile")
    from fsa_study.complexity import COMPLEXITY_PROTOCOL
    protocol = {"size": 384, "batch_size": 1, "threads": 2, "warmup": 10,
                "repeats": 50, "seed": 3407, "device": "cpu", "image": None}
    profile = {**identity, "complexity": COMPLEXITY_PROTOCOL, "protocol": protocol}
    (run / "profile.json").write_text(json.dumps(profile))
    assert completed(job, "profile")
    profile["protocol"]["size"] = 64
    (run / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="requested --size"):
        completed(job, "profile")


def test_generated_scripts_include_automatic_reporting_runner(tmp_path):
    output = tmp_path / "plan"
    result = subprocess.run([sys.executable, str(HERE / "plan_study.py"), "--phase", "generation",
                             "--manifest", str(tmp_path / "manifest.json"), "--output", str(output),
                             "--results", str(tmp_path / "runs"), "--random-init"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ("train", "evaluate", "profile", "run_all"):
        script = output / f"{name}.sh"
        assert "run_plan.py" in script.read_text() and "--skip-completed" in script.read_text()
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert "--actions train evaluate profile" in (output / "run_all.sh").read_text()
    assert not (tmp_path / "runs").exists()
