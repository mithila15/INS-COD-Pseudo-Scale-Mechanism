"""Independent reference calculations, offline and without optimizer steps.

DySample reference: coordinate-major sampling in the author's LP/static-scope
implementation https://github.com/tiny-smart/dysample/blob/main/dysample.py
CARAFE reference: per-target neighborhood contraction, ICCV 2019 Eq.(3).
No external code, CUDA extension, dataset, or pretrained weight is loaded.
"""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa_study.generators import ContentAwareUpsampler, DynamicSampleUpsampler


def _dysample_coordinate_major_reference(module, feature):
    """Build the low-resolution grid before shuffle, unlike the implementation.

    The study predictor orders channels as (group, xy, subpixel). The original
    coordinate-major derivation orders them as (xy, group, subpixel). Explicit
    permutation is essential: copying logits without reordering would compare
    different sampling offsets whenever there is more than one group.
    """
    batch, channels, height, width = feature.shape
    groups, scale = module.groups, module.scale
    raw = module.offset(feature)
    coordinate_major = raw.reshape(batch, groups, 2, scale * scale, height, width)
    coordinate_major = coordinate_major.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, 2 * groups * scale * scale, height, width
    )

    subpixel_positions = torch.arange(
        (-scale + 1) / 2, (scale - 1) / 2 + 1,
        device=feature.device, dtype=feature.dtype,
    ) / scale
    initial_positions = torch.stack(torch.meshgrid(
        subpixel_positions, subpixel_positions, indexing="ij"
    )).transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1)
    offsets = (0.25 * coordinate_major + initial_positions).reshape(
        batch, 2, -1, height, width
    )

    center_y = torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5
    center_x = torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5
    centers = torch.stack(torch.meshgrid(center_x, center_y, indexing="ij"))
    centers = centers.transpose(1, 2).unsqueeze(1).unsqueeze(0)
    normalizer = feature.new_tensor([width, height]).view(1, 2, 1, 1, 1)
    coordinates = 2 * (centers + offsets) / normalizer - 1

    shuffled = F.pixel_shuffle(coordinates.reshape(batch, -1, height, width), scale)
    grid = shuffled.reshape(batch, 2, -1, scale * height, scale * width)
    grid = grid.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
    sampled = F.grid_sample(
        feature.reshape(batch * groups, channels // groups, height, width), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return sampled.reshape(batch, channels, scale * height, scale * width)


def _carafe_explicit_neighborhood_reference(module, feature):
    """Enumerate high-resolution target pixels and low-resolution neighbors.

    There is no unfold, einsum, or pixel shuffle in this reference reassembly.
    Kernel predictor channels are ordered (subpixel, neighborhood); the same
    predicted kernel is shared by all feature channels at each target pixel.
    """
    batch, channels, height, width = feature.shape
    scale, kernel = module.scale, module.kernel_size
    logits = module.encode(module.compress(feature)).reshape(
        batch, scale * scale, kernel * kernel, height, width
    )
    weights = logits.softmax(dim=2)
    output = feature.new_zeros(batch, channels, scale * height, scale * width)
    for target_y in range(scale * height):
        for target_x in range(scale * width):
            parent_y, parent_x = target_y // scale, target_x // scale
            subpixel = (target_y % scale) * scale + target_x % scale
            for kernel_y in range(kernel):
                for kernel_x in range(kernel):
                    source_y = parent_y + kernel_y - kernel // 2
                    source_x = parent_x + kernel_x - kernel // 2
                    # Outside the source image, CARAFE reads zero padding.
                    if 0 <= source_y < height and 0 <= source_x < width:
                        neighbor = kernel_y * kernel + kernel_x
                        weight = weights[:, subpixel, neighbor, parent_y, parent_x].unsqueeze(1)
                        output[:, :, target_y, target_x] += weight * feature[:, :, source_y, source_x]
    return output


class ReferenceOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(987)

    @torch.no_grad()
    def test_dysample_nonzero_grouped_offsets_match_coordinate_major_reference(self):
        # Different batch sizes, group counts, axes, and scale factors expose
        # group/xy/subpixel permutations that zero-offset tests cannot reveal.
        cases = [
            (2, 12, 3, 5, 4, 2),
            (1, 6, 4, 3, 3, 2),
            (1, 4, 2, 3, 2, 3),
            (1, 4, 1, 3, 2, 2),
        ]
        for batch, channels, height, width, groups, scale in cases:
            with self.subTest(shape=(batch, channels, height, width), groups=groups, scale=scale):
                module = DynamicSampleUpsampler(channels, groups=groups, scale=scale).double()
                # Fixed synthetic fixture with substantial learned-offset-like
                # displacements; this is parameter assignment, not learning.
                module.offset.weight.normal_(std=0.3)
                module.offset.bias.normal_(std=0.2)
                feature = torch.randn(batch, channels, height, width, dtype=torch.float64)
                actual = module(feature)
                expected = _dysample_coordinate_major_reference(module, feature)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    @torch.no_grad()
    def test_carafe_matches_explicit_subpixel_and_neighborhood_contraction(self):
        for batch, channels, height, width, kernel, scale in [
            (2, 3, 3, 4, 3, 2),
            (1, 2, 2, 3, 5, 2),
            (1, 2, 2, 2, 3, 3),
        ]:
            with self.subTest(shape=(batch, channels, height, width), kernel=kernel, scale=scale):
                module = ContentAwareUpsampler(channels, kernel_size=kernel, compressed=4, scale=scale).double()
                feature = torch.randn(batch, channels, height, width, dtype=torch.float64)
                actual = module(feature)
                expected = _carafe_explicit_neighborhood_reference(module, feature)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
