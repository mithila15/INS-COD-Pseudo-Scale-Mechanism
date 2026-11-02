"""Inference operation accounting, including functional pseudo-scale operators.

The paper-comparable ``macs`` count dense convolution and tensor contractions;
``flops`` is twice that count. This is the convention in PyTorch's official
counter, not a hardware instruction count. Grouped/depthwise weights already
contain C_in/groups; transposed convolutions use input spatial size:
https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/flop_counter.py

Other operators are reported separately as analytical *estimates*. In particular
interpolation includes weighted samples, but omits coordinate/index construction
inside the kernel. Grid sampling is counted even though conventional MAC tools
often omit it. Norms, activations and reductions use documented scalar-operation
estimates. An unsupported operation makes the total estimate unavailable instead
of silently claiming complete FLOPs. Memory movement has no floating arithmetic
in this convention, although it can dominate latency.
"""
from __future__ import annotations

from collections import Counter
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode


COMPLEXITY_PROTOCOL = {
    "version": "aten_analytical_v1",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
    "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
    "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
}


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((t for v in value if (t := _first_tensor(v)) is not None), None)
    if isinstance(value, dict):
        return _first_tensor(list(value.values()))
    return None


def _arguments(func, args, kwargs):
    bound = {arg.name: arg.default_value for arg in func._schema.arguments if arg.has_default_value()}
    bound.update({arg.name: value for arg, value in zip(func._schema.arguments, args)})
    bound.update(kwargs)
    return bound


def _einsum_macs(equation, operands, output):
    # Count a two-input product/reduction, independently of einsum's contraction
    # implementation. Other equations are left unsupported rather than guessed.
    if len(operands) != 2 or "..." in equation or "->" not in equation:
        raise ValueError("only explicit two-input einsum equations are covered")
    left, right = equation.replace(" ", "").split("->")
    terms = left.split(",")
    if len(terms) != 2 or any(len(set(term)) != len(term) for term in terms):
        raise ValueError("einsum diagonal/repeated-label case is not covered")
    dimensions = {}
    for term, tensor in zip(terms, operands):
        for label, dimension in zip(term, tensor.shape):
            dimensions[label] = max(dimensions.get(label, 1), dimension)
    reduced = set("".join(terms)) - set(right)
    # A reduction label occurring in only one operand requires a separate
    # reduction and is not a direct contraction under this MAC convention.
    if not reduced or not reduced.issubset(set(terms[0]) & set(terms[1])):
        raise ValueError("einsum must contract shared reduction labels")
    return output.numel() * math.prod(dimensions[label] for label in reduced)


_MOVEMENT = {
    "alias", "detach", "detach_", "lift_fresh", "lift_fresh_copy", "clone", "copy_", "_to_copy", "to",
    "view", "view_as", "_unsafe_view", "reshape", "reshape_as", "flatten", "unflatten", "permute",
    "transpose", "t", "movedim", "contiguous", "select", "slice", "split", "split_with_sizes", "chunk",
    "unbind", "unsqueeze", "squeeze", "expand", "expand_as", "repeat", "repeat_interleave", "as_strided",
    "cat", "stack", "pixel_shuffle", "pixel_unshuffle", "im2col", "unfold", "meshgrid", "arange",
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
    "new_empty", "new_zeros", "new_full", "constant_pad_nd", "reflection_pad2d", "replication_pad2d",
    "index", "index_select", "gather", "slice_scatter", "select_scatter", "_local_scalar_dense",
}
_ELEMENTWISE = {
    "add", "add_", "sub", "sub_", "rsub", "mul", "mul_", "div", "div_", "true_divide", "neg", "abs",
    "square", "pow", "sqrt", "rsqrt", "reciprocal", "exp", "expm1", "log", "log1p", "erf", "tanh",
    "relu", "relu_", "clamp", "clamp_min", "clamp_max", "maximum", "minimum", "where",
    "eq", "ne", "gt", "ge", "lt", "le", "isfinite", "logical_and", "logical_or",
}


def _classify(func, args, kwargs, output):
    """Return category, MACs, estimated extra scalar operations, and rule."""
    name = str(func).split(".")[1]
    a = _arguments(func, args, kwargs)
    out = _first_tensor(output)
    n = out.numel() if out is not None else 0
    x = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if name in _MOVEMENT:
        return "non_arithmetic", 0, 0, "shape/index/allocation/data movement; memory cost excluded"
    if name == "dropout":
        if a.get("train", False):
            raise ValueError("training dropout is outside inference counting scope")
        return "non_arithmetic", 0, 0, "evaluation dropout is identity"
    if name in {"conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                "convolution", "_convolution", "cudnn_convolution", "convolution_overrideable"}:
        weight = a["weight"]
        transposed = name.startswith("conv_transpose") or a.get("transposed", False)
        spatial = x.shape[2:] if transposed else out.shape[2:]
        macs = x.shape[0] * math.prod(spatial) * weight.numel()
        bias = n if a.get("bias") is not None else 0
        return "contraction", macs, bias, "batch*spatial*weight.numel; input spatial for transpose; bias adds separate"
    if name == "linear":
        return "contraction", n * x.shape[-1], n if a.get("bias") is not None else 0, "output.numel*input_features; bias adds separate"
    if name in {"mm", "bmm", "matmul", "mv", "dot", "addmm", "baddbmm"}:
        lhs = args[1] if name in {"addmm", "baddbmm"} else args[0]
        extra = n if name in {"addmm", "baddbmm"} and a.get("beta", 1) != 0 else 0
        extra += n if name in {"addmm", "baddbmm"} and a.get("alpha", 1) != 1 else 0
        extra += n if name in {"addmm", "baddbmm"} and a.get("beta", 1) not in {0, 1} else 0
        return "contraction", n * lhs.shape[-1], extra, "output.numel*contracted_dimension; add/scale separate"
    if name == "einsum":
        return "contraction", _einsum_macs(args[0], args[1], out), 0, "two-input product summed over shared reduction labels"
    if name == "scaled_dot_product_attention" or name.startswith("_scaled_dot_product_"):
        query, key, value = args[:3]
        score_elements = math.prod(out.shape[:-2]) * query.shape[-2] * key.shape[-2]
        macs = score_elements * (query.shape[-1] + value.shape[-1])
        mask = a.get("attn_mask") is not None or a.get("is_causal", False)
        return "contraction", macs, score_elements * (6 + int(mask)), "QK and attention*V contractions; scale+softmax estimate and optional mask"
    if name in _ELEMENTWISE:
        extra = n * (2 if name in {"add", "add_", "sub", "sub_"} and a.get("alpha", 1) != 1 else 1)
        return "estimated", 0, extra, "one scalar operation per output element (alpha multiply adds one)"
    if name in {"sigmoid", "sigmoid_"}:
        return "estimated", 0, 4 * n, "negation, exp, add, reciprocal each cost 1"
    if name == "gelu":
        return "estimated", 0, n * (9 if a.get("approximate", "none") == "tanh" else 5), "erf GELU 5 or tanh GELU 9 scalar operations per element"
    if name in {"softmax", "_softmax", "log_softmax", "_log_softmax"}:
        return "estimated", 0, n * (6 if "log" in name else 5), "max, subtract, exp, sum, divide; optional log; 1 cost per scalar"
    if name in {"sum", "mean", "amax", "amin", "max", "min"}:
        reduction = max(0, x.numel() - n)
        return "estimated", 0, reduction + (n if name == "mean" else 0), "reduction inputs-outputs operations; mean adds output divisions"
    if name in {"linalg_vector_norm", "norm"}:
        order = a.get("ord", a.get("p", 2))
        if order != 2:
            raise ValueError("only L2 vector norm is covered")
        return "estimated", 0, 2 * x.numel(), "L2: square inputs, reduce inputs-outputs, sqrt outputs"
    if name in {"batch_norm", "native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training"}:
        if a.get("training", False):
            raise ValueError("training BatchNorm is outside inference counting scope")
        channels = x.shape[1]
        affine = a.get("weight") is not None
        return "estimated", 0, 2 * n + 2 * channels + (2 * n if affine else 0), "running variance add+rsqrt per channel; center/scale; optional affine multiply/add"
    if name in {"layer_norm", "native_layer_norm", "group_norm", "native_group_norm"}:
        # Mean/variance calculation is ~5N; centering/scaling ~2N; affine ~2N.
        affine = a.get("weight") is not None
        return "estimated", 0, n * (9 if affine else 7), "mean/variance and normalization ~7 operations/element; affine adds 2"
    if name in {"adaptive_avg_pool2d", "_adaptive_avg_pool2d", "adaptive_max_pool2d"}:
        h, w = x.shape[-2:]
        oh, ow = out.shape[-2:]
        # Adaptive bins overlap on non-divisible shapes. Count each actual bin.
        rows = sum(math.ceil((i + 1) * h / oh) - math.floor(i * h / oh) for i in range(oh))
        cols = sum(math.ceil((i + 1) * w / ow) - math.floor(i * w / ow) for i in range(ow))
        samples = math.prod(x.shape[:-2]) * rows * cols
        return "estimated", 0, samples if "avg" in name else samples - n, "sum/comparison within exact adaptive bins; average adds one division per bin"
    if name in {"avg_pool1d", "avg_pool2d", "avg_pool3d", "max_pool1d", "max_pool2d", "max_pool3d",
                "max_pool1d_with_indices", "max_pool2d_with_indices", "max_pool3d_with_indices"}:
        kernel = a["kernel_size"]
        kernel = (kernel,) * (x.ndim - 2) if isinstance(kernel, int) else kernel
        elements = math.prod(kernel)
        return "estimated", 0, n * (elements if "avg" in name else elements - 1), "dense kernel reduction, including padded positions; average adds division; max comparisons cost 1"
    if name.startswith("upsample_nearest"):
        return "non_arithmetic", 0, 0, "nearest sampling is copy/indexing"
    if name.startswith("upsample_bilinear2d"):
        return "estimated", 0, 7 * n, "4 weighted samples: 4 multiplies+3 additions/output; internal coordinates excluded"
    if name in {"grid_sampler", "grid_sampler_2d"}:
        if a.get("interpolation_mode", 0) != 0:
            raise ValueError("only bilinear 2D grid sampling is covered")
        return "estimated", 0, 7 * n, "4 weighted samples: 4 multiplies+3 additions/output; internal coordinates/padding/indexing excluded"
    raise ValueError("no registered analytical rule")


class OperationCounter(TorchDispatchMode):
    """Observe executed ATen operators without module-hook blind spots."""
    def __init__(self):
        super().__init__()
        self.operators = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        try:
            category, macs, extra, rule = _classify(func, args, kwargs, output)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            category, macs, extra, rule = "unsupported", 0, 0, str(exc)
        key = str(func)
        row = self.operators.setdefault(key, {"category": category, "calls": 0, "macs": 0,
                                             "estimated_noncontraction_flops": 0, "rule": rule})
        # Conservatively retain unsupported status if an overload has both
        # supported and unsupported argument configurations within one forward.
        if category == "unsupported":
            row["category"], row["rule"] = category, rule
        row["calls"] += 1
        row["macs"] += int(macs)
        row["estimated_noncontraction_flops"] += int(extra)
        return output

    def result(self, batch_size=1):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        counts = Counter()
        for row in self.operators.values():
            counts[row["category"]] += row["calls"]
        macs = sum(row["macs"] for row in self.operators.values())
        extra = sum(row["estimated_noncontraction_flops"] for row in self.operators.values())
        complete = not counts["unsupported"]
        return {**COMPLEXITY_PROTOCOL, "batch_size": batch_size,
                "macs": macs, "flops": 2 * macs,
                "macs_per_image": macs / batch_size, "flops_per_image": 2 * macs / batch_size,
                "estimated_noncontraction_flops": extra, "estimated_counted_flops": 2 * macs + extra,
                "estimated_total_flops": 2 * macs + extra if complete else None,
                "estimated_total_flops_per_image": (2 * macs + extra) / batch_size if complete else None,
                "coverage": {"supported_calls": counts["contraction"] + counts["estimated"],
                             "contraction_calls": counts["contraction"], "estimated_calls": counts["estimated"],
                             "non_arithmetic_calls": counts["non_arithmetic"], "unsupported_calls": counts["unsupported"],
                             "all_observed_ops_classified": complete,
                             "unsupported_operators": sorted(k for k, row in self.operators.items() if row["category"] == "unsupported"),
                             "note": "Call coverage is not a fraction of FLOPs; full classification still uses documented estimates/exclusions."},
                "operators": dict(sorted(self.operators.items()))}


def measure_complexity(model, x):
    """Run exactly one separate inference; never change mode or update weights."""
    if any(module.training for module in model.modules()):
        raise ValueError("complexity requires model.eval(); training forwards are not allowed")
    with torch.inference_mode(), OperationCounter() as counter:
        model(x)
    return counter.result(batch_size=x.shape[0])
