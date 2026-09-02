from __future__ import annotations

import torch
from torch import nn

from model_error_bound import InputDomain, MaxBound, empirical_check, quantize_linear_model


def test_hand_checkable_single_linear_layer_bound() -> None:
    model = nn.Linear(1, 1)
    compiled = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[2.0]])
        model.bias[:] = torch.tensor([1.0])
        compiled.weight[:] = torch.tensor([[1.5]])
        compiled.bias[:] = torch.tensor([0.5])

    X = InputDomain(torch.tensor([-1.0]), torch.tensor([2.0]))

    assert MaxBound(model, compiled, X) == 1.5


def test_relu_network_bound_is_not_violated_by_samples() -> None:
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    compiled = quantize_linear_model(model, decimals=1)
    X = InputDomain(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 0.5]))

    bound = MaxBound(model, compiled, X)
    check = empirical_check(model, compiled, X, bound, {"samples": 1000, "seed": 4})

    assert check["passed"]


def test_domain_validation_rejects_bad_bounds() -> None:
    try:
        InputDomain(torch.tensor([1.0]), torch.tensor([0.0]))
    except ValueError as exc:
        assert "lower-bound" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_unsupported_layer_fails_loudly() -> None:
    model = nn.Sequential(nn.Sigmoid())
    X = InputDomain(torch.tensor([0.0]), torch.tensor([1.0]))

    try:
        MaxBound(model, model, X)
    except NotImplementedError as exc:
        assert "Unsupported module" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
