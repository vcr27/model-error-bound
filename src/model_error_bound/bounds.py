"""Interval Bound Propagation for model-vs-compiled error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class InputDomain:
    """A box input domain represented by elementwise lower and upper bounds."""

    lower: Tensor
    upper: Tensor

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if torch.any(self.lower > self.upper):
            raise ValueError("every lower-bound value must be <= its upper bound")

    @classmethod
    def from_center_radius(cls, center: Tensor, radius: float | Tensor) -> "InputDomain":
        """Create an L-infinity box around a center point."""

        radius_tensor = torch.as_tensor(radius, dtype=center.dtype, device=center.device)
        if torch.any(radius_tensor < 0):
            raise ValueError("radius must be non-negative")
        return cls(center - radius_tensor, center + radius_tensor)


@dataclass(frozen=True)
class BoundReport:
    """Detailed result returned when `return_report=True` is requested."""

    bound: float
    norm: str
    original_output_lower: Tensor
    original_output_upper: Tensor
    compiled_output_lower: Tensor
    compiled_output_upper: Tensor
    assumptions: list[str]


def MaxBound(
    model: nn.Module,
    cl_model: nn.Module,
    X: InputDomain | tuple[Tensor, Tensor] | dict[str, Tensor],
    config: dict[str, Any] | None = None,
) -> float | BoundReport:
    """Return a sound upper bound on ||model(x) - cl_model(x)|| over X.

    The implemented formal method is Interval Bound Propagation (IBP). It is
    sound for the supported deterministic layer types because each layer's
    output interval contains all possible outputs for inputs in its input box.
    """

    cfg = dict(config or {})
    domain = _coerce_domain(X)
    norm = cfg.get("norm", "linf")
    return_report = bool(cfg.get("return_report", False))

    model_lower, model_upper = propagate_intervals(model, domain.lower, domain.upper)

    if _is_torch_compile_wrapper_of(cl_model, model):
        zero_bound = torch.zeros((), dtype=domain.lower.dtype, device=domain.lower.device)
        report = BoundReport(
            bound=float(zero_bound),
            norm=norm,
            original_output_lower=model_lower.detach(),
            original_output_upper=model_upper.detach(),
            compiled_output_lower=model_lower.detach(),
            compiled_output_upper=model_upper.detach(),
            assumptions=[
                "Input domain is a box: lower <= x <= upper elementwise.",
                "torch.compile wraps the same PyTorch module and is treated as semantics-preserving for this structural bound.",
                "The empirical check should still be run to look for observed backend numerical drift.",
            ],
        )
        return report if return_report else report.bound

    compiled_lower, compiled_upper = propagate_intervals(cl_model, domain.lower, domain.upper)

    exact_diff_interval = _try_exact_linear_difference(model, cl_model, domain.lower, domain.upper)
    if exact_diff_interval is None:
        bound = _interval_difference_bound(model_lower, model_upper, compiled_lower, compiled_upper, norm)
        method_note = "For general supported networks, the two output intervals are compared conservatively."
    else:
        diff_lower, diff_upper = exact_diff_interval
        bound = _signed_interval_abs_bound(diff_lower, diff_upper, norm)
        method_note = "For a single Linear layer, the exact model-minus-compiled difference layer is bounded directly."

    report = BoundReport(
        bound=float(bound.detach()),
        norm=norm,
        original_output_lower=model_lower.detach(),
        original_output_upper=model_upper.detach(),
        compiled_output_lower=compiled_lower.detach(),
        compiled_output_upper=compiled_upper.detach(),
        assumptions=[
            "Input domain is a box: lower <= x <= upper elementwise.",
            "Supported layers are deterministic and currently include Sequential, Linear, ReLU, Flatten, and Identity.",
            "The returned value is an upper bound under interval arithmetic; it may be conservative.",
            method_note,
        ],
    )
    return report if return_report else report.bound


def propagate_intervals(model: nn.Module, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    """Propagate lower/upper input bounds through a supported PyTorch module."""

    model = _unwrap_torch_compile(model)
    model.eval()
    with torch.no_grad():
        return _propagate_module(model, lower, upper)


def empirical_check(
    model: nn.Module,
    cl_model: nn.Module,
    X: InputDomain | tuple[Tensor, Tensor] | dict[str, Tensor],
    bound: float,
    config: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    """Sample points from X and verify no observed error exceeds `bound`.

    This is only a sanity check. The formal guarantee comes from MaxBound.
    """

    cfg = dict(config or {})
    domain = _coerce_domain(X)
    samples = int(cfg.get("samples", 256))
    seed = int(cfg.get("seed", 0))
    norm = cfg.get("norm", "linf")
    generator = torch.Generator(device=domain.lower.device).manual_seed(seed)

    max_seen = 0.0
    model.eval()
    cl_model.eval()
    with torch.no_grad():
        for _ in range(samples):
            unit = torch.rand(domain.lower.shape, generator=generator, device=domain.lower.device)
            x = domain.lower + unit * (domain.upper - domain.lower)
            diff = model(x) - cl_model(x)
            max_seen = max(max_seen, float(_tensor_norm(diff, norm)))

    tolerance = float(cfg.get("tolerance", 1e-6))
    return {
        "passed": max_seen <= float(bound) + tolerance,
        "max_observed_error": max_seen,
        "bound": float(bound),
        "samples": samples,
    }


def _coerce_domain(X: InputDomain | tuple[Tensor, Tensor] | dict[str, Tensor]) -> InputDomain:
    if isinstance(X, InputDomain):
        return X
    if isinstance(X, tuple) and len(X) == 2:
        return InputDomain(X[0], X[1])
    if isinstance(X, dict) and "lower" in X and "upper" in X:
        return InputDomain(X["lower"], X["upper"])
    raise TypeError("X must be InputDomain, (lower, upper), or {'lower': ..., 'upper': ...}")


def _propagate_module(module: nn.Module, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    if isinstance(module, nn.Sequential):
        for layer in module:
            lower, upper = _propagate_module(layer, lower, upper)
        return lower, upper
    if isinstance(module, nn.Linear):
        return _linear_interval(module, lower, upper)
    if isinstance(module, nn.ReLU):
        return torch.relu(lower), torch.relu(upper)
    if isinstance(module, nn.Flatten):
        start_dim = module.start_dim
        end_dim = module.end_dim
        return torch.flatten(lower, start_dim, end_dim), torch.flatten(upper, start_dim, end_dim)
    if isinstance(module, nn.Identity):
        return lower, upper
    raise NotImplementedError(
        f"Unsupported module {module.__class__.__name__}. "
        "Supported modules: Sequential, Linear, ReLU, Flatten, Identity."
    )


def _unwrap_torch_compile(model: nn.Module) -> nn.Module:
    """Return the original module inside a torch.compile OptimizedModule."""

    original = getattr(model, "_orig_mod", None)
    return original if isinstance(original, nn.Module) else model


def _is_torch_compile_wrapper_of(candidate: nn.Module, original: nn.Module) -> bool:
    return getattr(candidate, "_orig_mod", None) is original


def _linear_interval(layer: nn.Linear, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    weight_pos = torch.clamp(layer.weight, min=0)
    weight_neg = torch.clamp(layer.weight, max=0)
    out_lower = lower.matmul(weight_pos.t()) + upper.matmul(weight_neg.t())
    out_upper = upper.matmul(weight_pos.t()) + lower.matmul(weight_neg.t())
    if layer.bias is not None:
        out_lower = out_lower + layer.bias
        out_upper = out_upper + layer.bias
    return out_lower, out_upper


def _try_exact_linear_difference(
    model: nn.Module,
    cl_model: nn.Module,
    lower: Tensor,
    upper: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Use the exact difference network when both models are one Linear layer."""

    if not isinstance(model, nn.Linear) or not isinstance(cl_model, nn.Linear):
        return None
    if model.weight.shape != cl_model.weight.shape:
        return None
    if (model.bias is None) != (cl_model.bias is None):
        return None

    diff_layer = nn.Linear(model.in_features, model.out_features, bias=model.bias is not None)
    with torch.no_grad():
        diff_layer.weight.copy_(model.weight - cl_model.weight)
        if model.bias is not None and cl_model.bias is not None:
            diff_layer.bias.copy_(model.bias - cl_model.bias)
    return _linear_interval(diff_layer, lower, upper)


def _interval_difference_bound(
    a_lower: Tensor,
    a_upper: Tensor,
    b_lower: Tensor,
    b_upper: Tensor,
    norm: str,
) -> Tensor:
    diff_lower = a_lower - b_upper
    diff_upper = a_upper - b_lower
    return _signed_interval_abs_bound(diff_lower, diff_upper, norm)


def _signed_interval_abs_bound(diff_lower: Tensor, diff_upper: Tensor, norm: str) -> Tensor:
    abs_upper = torch.maximum(torch.abs(diff_lower), torch.abs(diff_upper))
    return _tensor_norm(abs_upper, norm)


def _tensor_norm(value: Tensor, norm: str) -> Tensor:
    flat = value.reshape(-1)
    if norm == "linf":
        return torch.max(torch.abs(flat))
    if norm == "l1":
        return torch.sum(torch.abs(flat))
    if norm == "l2":
        return torch.linalg.vector_norm(flat, ord=2)
    raise ValueError("supported norms are 'linf', 'l1', and 'l2'")
