"""CNN adapter, final-stage isolation and checkpoint checks; zero optimizer steps."""
import pytest
import torch

from fsa_study.backbones import (BACKBONE_CHANNELS, ConvNeXtFeatures,
                                 ResNetFeatures, load_backbone)
from fsa_study.model import LastStageDecoder, StudyModel


@pytest.fixture(autouse=True)
def forbid_weight_downloads(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Model construction must not download weights")
    monkeypatch.setattr(torch.hub, "download_url_to_file", fail)


@pytest.mark.parametrize("backbone", ["convnext_tiny", "resnet50"])
@pytest.mark.parametrize("feature_mode", ["last", "multistage"])
def test_cnn_outputs_and_single_encoder_call(backbone, feature_mode):
    model = StudyModel(backbone=backbone, feature_mode=feature_mode, channels=8,
                       generator="native_conv", utilizer="sk").eval()
    endpoints, projected = [], []
    hooks = [model.encoder.register_forward_hook(lambda module, inputs, output: endpoints.append(output))]
    for layer in model.projections:
        hooks.append(layer.register_forward_pre_hook(lambda module, inputs: projected.append(inputs[0])))
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 96))
    for hook in hooks:
        hook.remove()
    keys = ("reduction_5",) if feature_mode == "last" else tuple(f"reduction_{i}" for i in range(2, 6))
    assert len(endpoints) == 1
    assert tuple(endpoints[0]) == keys
    assert len(projected) == len(keys) == len(model.generators)
    assert projected[-1] is endpoints[0]["reduction_5"]
    channels = BACKBONE_CHANNELS[backbone]
    assert projected[-1].shape == (1, channels[-1], 2, 3)
    assert output["logits"].shape == (1, 1, 64, 96)
    assert len(output["deep_logits"]) == len(keys)
    assert all(torch.isfinite(value).all() for name, value in output.items() if name != "deep_logits")
    assert model.metadata["encoder_calls_per_image"] == 1
    assert model.metadata["encoder_stage_fusion"] is (feature_mode == "multistage")
    assert model.feature_levels == ((5,) if feature_mode == "last" else (2, 3, 4, 5))


@pytest.mark.parametrize("backbone", ["convnext_tiny", "resnet50"])
def test_final_stage_loss_gradients_without_parameter_updates(backbone):
    from fsa_study.engine import study_loss
    model = StudyModel(backbone=backbone, feature_mode="last", channels=8,
                       generator="native_conv", utilizer="sk").eval()
    output = model(torch.rand(2, 3, 64, 64))
    mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss, terms = study_loss(output, mask, mask, .25)
    assert set(terms) == {"seg", "deep", "region", "edge", "ual"}
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("backbone", ["convnext_tiny", "resnet50"])
def test_final_stage_common_initialization_and_bypass(backbone):
    torch.manual_seed(59)
    before = torch.get_rng_state().clone()
    baseline = StudyModel(backbone=backbone, feature_mode="last", channels=8, seed=17, baseline=True).eval()
    identity = StudyModel(backbone=backbone, feature_mode="last", channels=8, seed=17,
                          generator="identity", utilizer="mean").eval()
    variant = StudyModel(backbone=backbone, feature_mode="last", channels=8, seed=17,
                         generator="native_conv", utilizer="sk").eval()
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(value, variant.state_dict()[key]) for key, value in baseline.state_dict().items())
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(baseline(image)["logits"], identity(image)["logits"], atol=1e-6, rtol=1e-5)
    assert len(baseline.frozen_stem_modules) == (2 if backbone == "resnet50" else 1)
    assert not hasattr(baseline, "generators")
    assert isinstance(baseline.last_decoder, LastStageDecoder)
    assert not hasattr(baseline, "refinement")
    assert all(module.kernel_size in {(1, 1), (3, 3)} for module in baseline.last_decoder.modules()
               if isinstance(module, torch.nn.Conv2d))


@pytest.mark.parametrize("backbone,builder", [("convnext_tiny", ConvNeXtFeatures), ("resnet50", ResNetFeatures)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_official_cnn_checkpoint_keys_and_strict_rejection(tmp_path, monkeypatch, backbone, builder, wrapped):
    """Exercise strict loading with synthetic tensors, no external weight file."""
    from fsa_study.backbones import CLASSIFIER_KEYS
    encoder = builder(feature_mode="last")
    state = dict(encoder.state_dict())
    for key in CLASSIFIER_KEYS[backbone]:
        state[key] = torch.zeros(1)  # Classification tensors are explicitly excluded.
    path = tmp_path / "checkpoint.pth"
    path.touch()
    def checkpoint(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return {"state_dict": state} if wrapped else state
    monkeypatch.setattr(torch, "load", checkpoint)
    load_backbone(encoder, path)
    key = next(iter(encoder.state_dict()))
    missing = state.pop(key)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_backbone(encoder, path)
    state[key] = missing
    state["unexpected_feature.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_backbone(encoder, path)
    state.pop("unexpected_feature.weight")
    state[key] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_backbone(encoder, path)


def test_cnn_rejects_foreign_checkpoint_format(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pth"
    path.touch()
    encoder = ConvNeXtFeatures("last")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": encoder.state_dict()})
    with pytest.raises(ValueError, match="torchvision keys"):
        load_backbone(encoder, path)


@pytest.mark.parametrize("kwargs", [
    {"backbone": "unknown"}, {"feature_mode": "unknown"},
    {"backbone": "convnext_tiny", "legacy_reference": True},
    {"feature_mode": "last", "legacy_reference": True},
])
def test_reject_invalid_model_combinations(kwargs):
    with pytest.raises(ValueError):
        StudyModel(**kwargs)


def test_final_stage_checkpoint_roundtrip_and_odd_size():
    model = StudyModel(backbone="convnext_tiny", feature_mode="last", channels=8,
                       generator="native_conv", utilizer="sk", seed=15).eval()
    restored = StudyModel(**model.config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 65, 97)
    with torch.inference_mode():
        expected = model(image)["logits"]
        actual = restored(image)["logits"]
    assert actual.shape == (1, 1, 65, 97)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_historical_config_defaults_and_pvt_last_mode():
    old_config = dict(generator="native_conv", utilizer="sk", channels=8, seed=4,
                      baseline=False, legacy_reference=False, generator_options={}, utilizer_options={})
    historical = StudyModel(**old_config).eval()
    explicit = StudyModel(**old_config, backbone="pvt_v2_b2", feature_mode="multistage").eval()
    explicit.load_state_dict(historical.state_dict(), strict=True)
    assert all(torch.equal(value, explicit.state_dict()[key]) for key, value in historical.state_dict().items())
    last = StudyModel(**old_config, feature_mode="last").eval()
    with torch.inference_mode():
        output = last(torch.rand(1, 3, 64, 64))
    assert output["logits"].shape == (1, 1, 64, 64)
    assert len(output["deep_logits"]) == 1
