"""Synthetic inference/gradient/compatibility verification. Never trains or downloads."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))
BASE = "1a003b9a7becddc0fd844bb85de62f005b96a9f0"


def main():
    import torch
    from fsa_study.model import StudyModel
    from fsa_study.distillation import FrozenScaleTeacher
    from fsa_study.engine import study_loss
    from fsa_study.complexity import measure_complexity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Verification output already exists; preserve it and use a new path")
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    report = {"status": "initialized_models_only", "optimizer_steps": 0, "dataset_downloads": 0,
              "pretrained_weights": False, "device": args.device, "torch": torch.__version__,
              "base_revision": BASE, "compatibility": [], "inference": [], "gradients": []}
    if args.device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name(args.device)
    # Compare the previous model entry point, with backward-compatible default
    # generator/backbone implementations, against the new default behavior.
    old_path = str(STUDY.relative_to(STUDY.parents[2]) / "fsa_study/model.py")
    source = subprocess.check_output(["git", "-C", str(STUDY.parents[2]), "show", BASE + ":" + old_path], text=True)
    previous = types.ModuleType("fsa_study._pre_mechanism")
    previous.__file__ = str(STUDY / "fsa_study/model.py")
    exec(compile(source, "predecessor-model.py", "exec"), previous.__dict__)
    cases = [
        {"baseline": True},
        {"generator": "native_conv", "utilizer": "sk"},
        {"backbone": "convnext_tiny", "feature_mode": "last", "baseline": True},
        {"backbone": "convnext_tiny", "feature_mode": "last", "generator": "native_conv", "utilizer": "mean"},
    ]
    for case in cases:
        old = previous.StudyModel(channels=8, seed=3407, **case).eval()
        new = StudyModel(channels=8, seed=3407, **case).eval()
        assert old.state_dict().keys() == new.state_dict().keys()
        assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
        x = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            before, after = old(x)["logits"], new(x)["logits"]
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        report["compatibility"].append({"case": case, "state_tensors_equal": True, "max_logit_difference": 0.0})
        del old, new
    for stride in (32, 16):
        for variant in ("single", "projection__mean", "native_conv__mean", "image_pyramid__mean", "native_conv__anchor_gated"):
            options = {"baseline": variant == "single",
                       "input_mode": "image_pyramid" if variant.startswith("image_pyramid") else "single"}
            if "__" in variant and options["input_mode"] == "single":
                options["generator"], options["utilizer"] = variant.split("__")
            model = StudyModel(backbone="convnext_tiny", feature_mode="last", channels=48,
                               output_stride=stride, seed=3407, **options).to(args.device).eval()
            x = torch.rand(1, 3, 384, 384, device=args.device)
            with torch.inference_mode():
                logits = model(x)["logits"]
            assert logits.shape == (1, 1, 384, 384) and torch.isfinite(logits).all()
            cost = measure_complexity(model, x)
            assert cost["coverage"]["all_observed_ops_classified"], cost["coverage"]
            report["inference"].append({"variant": variant, "output_stride": stride,
                "input_shape": list(x.shape), "finite_logits": True, "parameters": sum(p.numel() for p in model.parameters()),
                "encoder_calls": model.metadata["encoder_calls_per_image"],
                "macs_per_image": cost["macs_per_image"], "unsupported_operators": cost["coverage"]["unsupported_operators"]})
            del model, x, logits
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    for stride in (32, 16):
        student = StudyModel(backbone="convnext_tiny", feature_mode="last", channels=48, seed=3407,
                             output_stride=stride, generator="native_conv").to(args.device).train()
        teacher_model = StudyModel(backbone="convnext_tiny", feature_mode="last", channels=48, seed=3407,
                                   output_stride=stride, input_mode="image_pyramid").to(args.device)
        teacher = FrozenScaleTeacher(teacher_model, {})
        # Only a backward-connectivity check; no optimizer is constructed.
        x = torch.rand(2, 3, 128, 128, device=args.device)
        out = student(x, return_features=True)
        mask = (torch.rand(2, 1, 128, 128, device=args.device) > .5).float()
        supervised, _ = study_loss(out, mask, mask, .5)
        feature = teacher.loss(out["feature_banks"], x)
        loss = supervised + .1 * feature
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        assert all(p.grad is None and not p.requires_grad for p in teacher.model.parameters())
        report["gradients"].append({"output_stride": stride, "input_shape": list(x.shape),
                                    "finite_connected_student": True, "frozen_teacher": True, "optimizer_steps": 0})
        del student, teacher, teacher_model, x, out, mask, loss, feature, supervised
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    report["source_sha256"] = {
        str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(STUDY.glob("fsa_study/*.py"))
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output), "compatibility": len(report["compatibility"]),
                      "inference": len(report["inference"]), "gradients": len(report["gradients"]), "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
