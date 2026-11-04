"""Analytical complexity checks; inference only, no optimizer or data downloads."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fsa_study.complexity import OperationCounter, measure_complexity
from fsa_study.generators import GENERATOR_NAMES, build_generator
from fsa_study.utilizers import UTILIZER_NAMES, build_utilizer


def count(function, *args, batch_size=1, **kwargs):
    with torch.inference_mode(), OperationCounter() as counter:
        function(*args, **kwargs)
    return counter.result(batch_size=batch_size)


@pytest.mark.parametrize("groups", [1, 3, 6])
def test_grouped_and_depthwise_convolution(groups):
    layer = nn.Conv2d(6, 12, 3, padding=2, dilation=2, groups=groups, bias=True).eval()
    result = measure_complexity(layer, torch.randn(2, 6, 5, 7))
    expected = 2 * 5 * 7 * 12 * (6 // groups) * 3 * 3
    assert result["macs"] == expected
    assert result["flops"] == 2 * expected
    assert result["macs_per_image"] == expected / 2
    assert result["estimated_noncontraction_flops"] == 2 * 12 * 5 * 7


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_transposed_convolution_counts_input_spatial_positions(groups):
    layer = nn.ConvTranspose2d(4, 8, 3, stride=2, groups=groups, bias=False).eval()
    result = measure_complexity(layer, torch.randn(2, 4, 5, 7))
    assert result["macs"] == 2 * 5 * 7 * 4 * (8 // groups) * 3 * 3


def test_linear_and_bias_are_separate():
    result = measure_complexity(nn.Linear(4, 5).eval(), torch.randn(2, 3, 4))
    assert result["macs"] == 2 * 3 * 5 * 4
    assert result["estimated_noncontraction_flops"] == 2 * 3 * 5


@pytest.mark.parametrize("left,right,expected", [
    ((2, 3), (3, 4), 24), ((2, 5, 3), (2, 3, 4), 120),
    ((2, 1, 5, 3), (1, 4, 3, 7), 840), ((3,), (3, 4), 12),
    ((2, 3), (3,), 6), ((3,), (3,), 3),
])
def test_functional_matmul_including_broadcast_and_vectors(left, right, expected):
    result = count(torch.matmul, torch.randn(*left), torch.randn(*right))
    assert result["macs"] == expected
    assert result["coverage"]["all_observed_ops_classified"]


def test_carafe_functional_einsum_reassembly_is_counted():
    # B=2, C=3, K=25, H=4, W=5, S=4 subpixels; one K-term
    # product/reduction for each B,C,S,H,W output, not a Conv module.
    result = count(torch.einsum, "bckhw,bskhw->bcshw",
                   torch.randn(2, 3, 25, 4, 5), torch.randn(2, 4, 25, 4, 5))
    assert result["macs"] == 2 * 3 * 4 * 4 * 5 * 25
    assert result["coverage"]["all_observed_ops_classified"]


def test_scaled_dot_product_attention_counts_both_matrix_products():
    result = count(F.scaled_dot_product_attention,
                   torch.randn(2, 3, 5, 4), torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 9))
    assert result["macs"] == 2 * 3 * 5 * 7 * (4 + 9)
    assert result["estimated_noncontraction_flops"] == 2 * 3 * 5 * 7 * 6


def test_interpolation_and_grid_sampling_are_separate_estimates():
    x = torch.randn(1, 2, 3, 4)
    interpolate = count(F.interpolate, x, size=(6, 8), mode="bilinear", align_corners=False)
    grid = count(F.grid_sample, x, torch.rand(1, 6, 8, 2), mode="bilinear", align_corners=False)
    for result in (interpolate, grid):
        assert result["macs"] == 0
        assert result["estimated_noncontraction_flops"] == 7 * 1 * 2 * 6 * 8
        assert result["coverage"]["all_observed_ops_classified"]
        assert "internal coordinates" in next(iter(result["operators"].values()))["rule"]


def test_adaptive_pool_counts_overlapping_odd_sized_bins():
    x = torch.randn(1, 2, 5, 7)
    average = count(F.adaptive_avg_pool2d, x, (3, 4))
    maximum = count(F.adaptive_max_pool2d, x, (3, 4))
    # Height bin lengths 2,3,2 and width bin lengths 2,3,3,2.
    assert average["estimated_noncontraction_flops"] == 2 * 7 * 10
    assert maximum["estimated_noncontraction_flops"] == 2 * 7 * 10 - 2 * 3 * 4


def test_fixed_pooling_counts_dense_kernels_and_ignores_index_outputs():
    x = torch.randn(1, 2, 6, 8)
    average = count(F.avg_pool2d, x, 2)
    maximum = count(F.max_pool2d, x, 2, return_indices=True)
    assert average["estimated_noncontraction_flops"] == 2 * 3 * 4 * 4
    assert maximum["estimated_noncontraction_flops"] == 2 * 3 * 4 * 3
    assert average["coverage"]["all_observed_ops_classified"]
    assert maximum["coverage"]["all_observed_ops_classified"]


def test_unknown_operator_invalidates_total_instead_of_silently_counting_zero():
    result = count(torch.special.airy_ai, torch.ones(3))
    assert result["coverage"]["unsupported_calls"] == 1
    assert result["coverage"]["unsupported_operators"]
    assert not result["coverage"]["all_observed_ops_classified"]
    assert result["estimated_total_flops"] is None
    assert result["estimated_total_flops_per_image"] is None


@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
def test_padding_copies_data_without_floating_arithmetic(mode):
    result = count(F.pad, torch.randn(1, 3, 4, 4), (0, 1, 0, 1), mode=mode)
    assert result["coverage"]["all_observed_ops_classified"]
    assert result["macs"] == result["estimated_total_flops"] == 0


def test_noncontraction_einsum_is_explicitly_unsupported():
    result = count(torch.einsum, "ij,ij->ij", torch.ones(2, 3), torch.ones(2, 3))
    assert result["coverage"]["unsupported_calls"] == 1
    assert result["estimated_total_flops"] is None


def test_no_training_or_state_updates():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    x = torch.rand(2, 3, 8, 8)
    with pytest.raises(ValueError, match="eval"):
        measure_complexity(model, x)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    measure_complexity(model, x)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not model.training


@pytest.mark.parametrize("generator", GENERATOR_NAMES)
@pytest.mark.parametrize("utilizer", UTILIZER_NAMES)
def test_all_generator_utilizer_pairs_have_explicit_operator_coverage(generator, utilizer):
    model = nn.Sequential(build_generator(generator, 8), build_utilizer(utilizer, 8)).eval()
    result = measure_complexity(model, torch.rand(1, 8, 7, 9))
    assert result["coverage"]["all_observed_ops_classified"], result["coverage"]["unsupported_operators"]
    assert result["estimated_total_flops"] is not None


@pytest.mark.parametrize("config", [
    {"baseline": True}, {"generator": "carafe", "utilizer": "mhsiu"},
    {"generator": "dysample", "utilizer": "sk"}, {"legacy_reference": True},
])
def test_full_model_inference_coverage(config):
    from fsa_study.model import StudyModel
    from torch.utils.flop_counter import FlopCounterMode
    model = StudyModel(channels=8, **config).eval()
    x = torch.rand(1, 3, 64, 64)
    result = measure_complexity(model, x)
    assert result["macs"] > 0
    assert result["coverage"]["all_observed_ops_classified"], result["coverage"]["unsupported_operators"]
    with torch.inference_mode(), FlopCounterMode(display=False) as official:
        model(x)
    assert result["flops"] == official.get_total_flops()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_and_cpu_accounting_match_for_dynamic_sampling():
    model = nn.Sequential(build_generator("dysample", 8), build_utilizer("sk", 8)).eval()
    x = torch.rand(1, 8, 8, 8)
    cpu = measure_complexity(model, x)
    cuda = measure_complexity(model.cuda(), x.cuda())
    assert cpu["macs"] == cuda["macs"]
    assert cpu["estimated_total_flops"] == cuda["estimated_total_flops"]
    assert cuda["coverage"]["all_observed_ops_classified"]
