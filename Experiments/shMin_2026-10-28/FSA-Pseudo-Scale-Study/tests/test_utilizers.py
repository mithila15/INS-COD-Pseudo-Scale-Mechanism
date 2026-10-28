"""Semantic utilization checks; no datasets, optimizer, or parameter updates."""
import ast
import json
import sys
import unittest
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa_study.utilizers import UTILIZER_NAMES, build_utilizer


class UtilizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(73)

    def test_every_method_connects_all_branches_without_optimizer(self):
        for name in UTILIZER_NAMES:
            with self.subTest(name=name):
                module = build_utilizer(name, channels=8)
                bank = [torch.randn(1, 8, 5, 7, requires_grad=True) for _ in range(3)]
                output = module(bank)
                self.assertEqual(output.shape, bank[0].shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for feature in bank:
                    self.assertIsNotNone(feature.grad)
                    self.assertTrue(torch.isfinite(feature.grad).all())
                    self.assertGreater(feature.grad.abs().sum().item(), 0)
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                json.dumps(module.metadata)

    def test_identical_branches_preserve_anchor_when_required(self):
        anchor = torch.randn(1, 8, 3, 5)
        exact = ("mean", "signed", "absolute", "relative", "normalized", "adjacent", "sk", "asff", "aff")
        for name in exact:
            with self.subTest(name=name):
                module = build_utilizer(name, channels=8)
                output = module([anchor, anchor, anchor])
                torch.testing.assert_close(output, anchor, rtol=2e-6, atol=2e-6)
        weighted = build_utilizer("bifpn", channels=8)
        output = weighted([anchor, anchor, anchor])
        torch.testing.assert_close(output, anchor * (3 / (3 + weighted.eps)))

    def test_difference_zero_property_for_every_allowed_activation(self):
        anchor = torch.randn(1, 4, 3, 3)
        for name in ("signed", "absolute", "relative", "normalized", "adjacent"):
            for activation in ("identity", "relu", "tanh"):
                with self.subTest(name=name, activation=activation):
                    module = build_utilizer(name, 4, activation=activation)
                    self.assertIsNone(module.projection.bias)
                    torch.testing.assert_close(module([anchor] * 3), anchor, rtol=0, atol=0)

    def test_normalized_means_pixelwise_channel_l2_not_spatial(self):
        main = torch.tensor([[[[3., 0.]], [[4., 2.]]]])
        large = 2 * main
        small = torch.tensor([[[[0., 2.]], [[2., 0.]]]])
        module = build_utilizer("normalized", 2)
        large_delta, small_delta = module.differences([main, large, small])
        torch.testing.assert_close(large_delta, torch.zeros_like(main))
        expected = torch.tensor([[[[-0.6, 1.]], [[0.2, -1.]]]])
        torch.testing.assert_close(small_delta, expected)

    def test_relative_formula_and_zero_safety(self):
        main = torch.tensor([[[[0., 1., -2., 3.]]]])
        large = torch.tensor([[[[0., 4., 2., -2.]]]])
        small = torch.tensor([[[[0., -4., -2., 1.]]]])
        module = build_utilizer("relative", 1)
        for aux, actual in zip([large, small], module.differences([main, large, small])):
            expected = (aux - main) / (aux.abs() + main.abs() + module.eps)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for name in ("relative", "normalized"):
                with self.subTest(dtype=dtype, name=name):
                    zeros = torch.zeros(1, 2, 2, 2, dtype=dtype)
                    tiny = torch.full_like(zeros, 1e-7)
                    module = build_utilizer(name, 2, eps=1e-12)
                    for delta in module.differences([zeros, tiny, zeros]):
                        self.assertTrue(torch.isfinite(delta).all())
                        self.assertEqual(delta.dtype, dtype)

    def test_signed_and_adjacent_are_equivalent_with_projection_sign_change(self):
        signed = build_utilizer("signed", 4)
        adjacent = build_utilizer("adjacent", 4)
        with torch.no_grad():
            adjacent.projection.weight.copy_(signed.projection.weight)
            adjacent.projection.weight[:, 4:].neg_()
        bank = [torch.randn(1, 4, 3, 3) for _ in range(3)]
        torch.testing.assert_close(signed(bank), adjacent(bank))

    def test_convex_attention_weights_have_correct_branch_sum(self):
        bank = [torch.randn(1, 8, 3, 5) for _ in range(3)]
        expected_shapes = {
            "bifpn": (1, 3, 1, 1, 1), "sk": (1, 3, 8, 1, 1),
            "asff": (1, 3, 1, 3, 5), "aff": (1, 3, 8, 3, 5),
            "mhsiu": (1, 3, 4, 3, 5),
        }
        for name, expected_shape in expected_shapes.items():
            with self.subTest(name=name):
                module = build_utilizer(name, 8).eval()
                weights = module.weights(bank)
                self.assertEqual(weights.shape, expected_shape)
                self.assertTrue((weights >= 0).all())
                target = torch.ones_like(weights.sum(dim=1))
                if name == "bifpn":
                    target *= 3 / (3 + module.eps)
                torch.testing.assert_close(weights.sum(dim=1), target)

    def test_legacy_independent_gates_allow_negative_main_coefficient(self):
        module = build_utilizer("legacy_gated", 4).eval()
        with torch.no_grad():
            module.gate[1].weight.zero_()
            module.gate[1].bias.fill_(10)
        bank = [torch.randn(1, 4, 3, 3) for _ in range(3)]
        weights = module.weights(bank)
        self.assertTrue((weights[:, 0] < 0).all())
        torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(bank[0]))
        self.assertFalse(hasattr(module, "residual_proj"))

    def test_legacy_fused_output_matches_preserved_source_after_copying_state(self):
        source = ROOT.parents[1] / "FSA-ZoomNeXt/methods/zoomnext/fsa_layers.py"
        parsed = ast.parse(source.read_text())
        required = {"DSConvBNReLU", "ConvBNAct", "ScaleResidualIntegration"}
        selected = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in required], type_ignores=[])
        namespace = {
            "torch": torch, "nn": nn, "Optional": Optional, "Tuple": Tuple,
            "resize_to": lambda x, tgt_hw: F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False),
        }
        exec(compile(selected, str(source), "exec"), namespace)
        reference = namespace["ScaleResidualIntegration"](4).eval()
        actual = build_utilizer("legacy_gated", 4).eval()
        actual.load_state_dict({k: v for k, v in reference.state_dict().items() if not k.startswith("residual_proj.")})
        bank = [torch.randn(2, 4, 3, 5) for _ in range(3)]
        with torch.no_grad():
            expected, _ = reference(bank[1], bank[0], bank[2])
            result = actual(bank)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_mhsiu_group_layout_keeps_matching_channels_together(self):
        module = build_utilizer("mhsiu", channels=12, heads=4)
        with torch.no_grad():
            nn.init.dirac_(module.value_transform.weight, groups=module.heads)
            module.attention[-1].weight.zero_()
            module.attention[-1].bias.zero_()
        bank = [torch.randn(1, 12, 3, 5) for _ in range(3)]
        expected = torch.stack(bank, dim=1).mean(dim=1)
        torch.testing.assert_close(module(bank), expected)
        self.assertEqual(build_utilizer("mhsiu", channels=6, heads=4).heads, 2)

    def test_new_attention_handles_batch_one_single_pixel(self):
        for channels in (1, 2, 7):
            for name in ("sk", "asff", "aff", "mhsiu"):
                with self.subTest(channels=channels, name=name):
                    module = build_utilizer(name, channels).train()
                    bank = [torch.randn(1, channels, 1, 1) for _ in range(3)]
                    self.assertTrue(torch.isfinite(module(bank)).all())

    def test_bad_configuration_and_bank_are_rejected(self):
        with self.assertRaises(ValueError):
            build_utilizer("not_a_method", 4)
        with self.assertRaises(TypeError):
            build_utilizer("mean", 4, misspelled_option=1)
        with self.assertRaises(ValueError):
            build_utilizer("relative", 4, eps=0)
        with self.assertRaises(ValueError):
            build_utilizer("aff", 4, num_branches=4)
        module = build_utilizer("mean", 4)
        bank = [torch.randn(1, 4, 3, 3) for _ in range(3)]
        with self.assertRaises(ValueError):
            module(bank[:2])
        with self.assertRaises(ValueError):
            module([bank[0], bank[1].double(), bank[2]])
        with self.assertRaises(ValueError):
            module([bank[0], bank[1][:, :, :1], bank[2]])


if __name__ == "__main__":
    unittest.main()
