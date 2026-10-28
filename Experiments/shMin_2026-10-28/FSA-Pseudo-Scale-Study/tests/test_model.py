"""Forward, invariants and autograd only: never train or update parameters."""
import itertools
import pytest
import torch

from fsa_study.generators import GENERATOR_NAMES, build_generator
from fsa_study.utilizers import UTILIZER_NAMES, build_utilizer


@pytest.mark.parametrize("generator,utilizer", list(itertools.product(GENERATOR_NAMES, UTILIZER_NAMES)))
def test_all_pairings_forward_backward(generator, utilizer):
    x = torch.randn(2, 8, 9, 11, requires_grad=True)
    g, u = build_generator(generator, 8), build_utilizer(utilizer, 8)
    y = u(g(x))
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in itertools.chain(g.named_parameters(), u.named_parameters()):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_common_initialization_and_global_rng_independent():
    from fsa_study.model import StudyModel
    torch.manual_seed(99)
    state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    baseline = StudyModel(seed=42, baseline=True, channels=8)
    assert torch.equal(torch.get_rng_state(), state)
    variant = StudyModel(seed=42, generator="dysample", utilizer="asff", channels=8)
    common = baseline.state_dict()
    assert all(torch.equal(value, variant.state_dict()[key]) for key, value in common.items())
    assert not hasattr(baseline, "generators") and not hasattr(baseline, "utilizers")
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(before, after) for before, after in zip(cuda_states, torch.cuda.get_rng_state_all()))


@pytest.mark.parametrize("kwargs", [
    {"baseline": True}, {"legacy_reference": True},
    {"generator": "identity", "utilizer": "mean"},
    {"generator": "legacy", "utilizer": "legacy_gated"},
    {"generator": "dysample", "utilizer": "mhsiu"},
    {"generator": "carafe", "utilizer": "aff"},
])
def test_full_model_single_encoder(kwargs):
    from fsa_study.model import StudyModel
    m = StudyModel(**kwargs, channels=8).eval()
    calls = []
    hook = m.encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        output = m(torch.rand(1, 3, 64, 64))
    hook.remove()
    assert len(calls) == 1
    assert output["logits"].shape == (1, 1, 64, 64)
    assert len(output["deep_logits"]) == 4
    assert all(torch.isfinite(output[k]).all() for k in ("logits", "po", "pe"))


def test_identity_mean_matches_bypass():
    from fsa_study.model import StudyModel
    a = StudyModel(baseline=True, channels=8, seed=14).eval()
    b = StudyModel(generator="identity", utilizer="mean", channels=8, seed=14).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(a(x)["logits"], b(x)["logits"], atol=1e-6, rtol=1e-5)


def test_checkpoint_strict_roundtrip_and_backbone_head(tmp_path):
    from fsa_study.model import StudyModel, load_backbone
    model = StudyModel(channels=8, generator="sfp", utilizer="sk").eval()
    path = tmp_path / "state.pth"
    torch.save({"model_config": model.config, "model": model.state_dict()}, path)
    ckpt = torch.load(path, weights_only=True)
    restored = StudyModel(**ckpt["model_config"]).eval()
    restored.load_state_dict(ckpt["model"], strict=True)
    invalid = dict(ckpt["model"])
    invalid.pop(next(iter(invalid)))
    with pytest.raises(RuntimeError):
        restored.load_state_dict(invalid, strict=True)
    state = dict(model.encoder.state_dict())
    state.update({"head.weight": torch.zeros(1000, 512), "head.bias": torch.zeros(1000)})
    torch.save(state, path)
    load_backbone(restored.encoder, path)
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_backbone(restored.encoder, path)


def test_plan_counts():
    from plan_study import variants
    assert len(variants("cross", "legacy", "mean")) == 158
    assert len(variants("generation", "legacy", "mean")) == 14
    assert len(variants("utilization", "legacy", "mean")) == 15
