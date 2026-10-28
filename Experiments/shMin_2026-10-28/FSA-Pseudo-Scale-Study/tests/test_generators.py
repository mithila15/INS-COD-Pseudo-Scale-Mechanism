"""Mathematical/synthetic generator checks; no optimizer or dataset access."""

import ast
import json
from pathlib import Path
import sys
import typing

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsa_study.generators import (
    GENERATOR_NAMES, ContentAwareUpsampler, DynamicSampleUpsampler, build_generator,
)


@pytest.mark.parametrize("name", GENERATOR_NAMES)
def test_aligned_contract_and_differentiability(name):
    torch.manual_seed(19)
    model = build_generator(name, 8)
    x = torch.randn(2, 8, 7, 9, requires_grad=True)
    outputs = model(x)
    assert len(outputs) == model.num_branches == 3
    assert outputs[0] is x
    for output in outputs:
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    loss = sum((i + 1) * output.square().mean() for i, output in enumerate(outputs))
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert json.loads(json.dumps(model.metadata))["name"] == name


@pytest.mark.parametrize("name", GENERATOR_NAMES)
@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (1, 5, 1, 7), (1, 7, 9, 1)])
def test_singleton_axes_and_nondivisible_channels(name, shape):
    # Forward-only in training mode specifically exercises safe normalization.
    model = build_generator(name, shape[1]).train()
    x = torch.randn(*shape)
    outputs = model(x)
    assert all(output.shape == x.shape for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_zero_power_gain_reduces_to_plain_resampling():
    x = torch.randn(2, 7, 9, 13)
    baseline = build_generator("resample", 7)(x)
    power = build_generator("ffp_gain", 7)(x)
    for expected, actual in zip(baseline, power):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_power_gain_preserves_input_dtype_and_parameter_gradients(dtype):
    model = build_generator("ffp_gain", 7)
    x = torch.randn(1, 7, 5, 9, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        model.scale_exponent.fill_(0.3)
    outputs = model(x)
    assert all(output.dtype == dtype for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.float().square().mean() for output in outputs).backward()
    assert torch.isfinite(model.scale_exponent.grad).all()
    assert torch.count_nonzero(model.scale_exponent.grad) > 0


@pytest.mark.parametrize("shape", [(2, 8, 7, 5), (1, 1, 1, 1), (1, 5, 1, 7)])
def test_zero_dysample_offsets_equal_bilinear(shape):
    module = DynamicSampleUpsampler(shape[1])
    with torch.no_grad():
        module.offset.weight.zero_()
        module.offset.bias.zero_()
    x = torch.randn(*shape)
    actual = module(x)
    expected = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_uniform_carafe_kernels_are_local_averages():
    module = ContentAwareUpsampler(3, kernel_size=3, compressed=4)
    with torch.no_grad():
        module.encode.weight.zero_()
        module.encode.bias.zero_()
    x = torch.randn(2, 3, 5, 7)
    expected = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=True)
    expected = expected.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)


def test_projection_capacity_control_matches_native_branches():
    counts = {
        name: sum(p.numel() for p in build_generator(name, 8).parameters())
        for name in ("projection", "native_conv", "shared_native")
    }
    assert counts["projection"] == counts["native_conv"]
    assert counts["native_conv"] == 2 * counts["shared_native"]


def _load_original_legacy_class():
    """Execute only the three original class definitions, without backbone imports.

    AST selection leaves the legacy operation definitions untouched. This is a
    test reference to the preserved repository file, not vendored source code.
    """
    original = Path(__file__).resolve().parents[3] / "FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py"
    if not original.is_file():
        pytest.skip("preserved FSA source is absent; cannot verify legacy equivalence")
    tree = ast.parse(original.read_text(encoding="utf-8"))
    names = {"DSConvBNReLU", "ConvBNAct", "PseudoScaleGenerator"}
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(selected) == 3
    namespace = {
        "torch": torch, "nn": nn, "F": F, "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "resize_to": lambda x, tgt_hw: F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(original), "exec"), namespace)
    return namespace["PseudoScaleGenerator"]


def test_legacy_eval_exactly_matches_preserved_fsa_weights():
    torch.manual_seed(71)
    reference = _load_original_legacy_class()(8).eval()
    candidate = build_generator("legacy", 8).eval()
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.normal_()
                module.running_var.uniform_(0.3, 1.7)
    old_state = list(reference.state_dict().items())
    new_state = list(candidate.state_dict().items())
    assert len(old_state) == len(new_state)
    remapped = {}
    for (old_key, old_tensor), (new_key, new_tensor) in zip(old_state, new_state):
        assert old_tensor.shape == new_tensor.shape, (old_key, new_key)
        remapped[new_key] = old_tensor
    candidate.load_state_dict(remapped, strict=True)
    x = torch.randn(2, 8, 9, 11)
    with torch.no_grad():
        old_large, old_main, old_small = reference(x)
        actual = candidate(x)
    for expected, result in zip([old_main, old_large, old_small], actual):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name,kwargs", [
    ("resample", {"scale_factors": (0, 0.5)}),
    ("resample", {"scale_factors": (float("inf"), 0.5)}),
    ("resample", {"scale_factors": (float("nan"), 0.5)}),
    ("ppm", {"ppm_bins": (1, 0)}),
    ("aspp", {"aspp_rates": (1, 0)}),
    ("carafe", {"carafe_kernel": 2}),
    ("dysample", {"dysample_groups": 0}),
])
def test_invalid_hyperparameters_fail_explicitly(name, kwargs):
    with pytest.raises(ValueError):
        build_generator(name, 8, **kwargs)


def test_unknown_name_and_typo_fail_explicitly():
    with pytest.raises(ValueError):
        build_generator("not_a_generator", 8)
    with pytest.raises(TypeError):
        build_generator("ppm", 8, bins_typo=(2, 4))
