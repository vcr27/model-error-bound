"""Run a small end-to-end bound computation."""

from __future__ import annotations

import torch
from torch import nn

from model_error_bound import InputDomain, MaxBound, empirical_check, quantize_linear_model


def build_model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def main() -> None:
    model = build_model()
    cl_func = lambda original: quantize_linear_model(original, decimals=2)
    cl_model = cl_func(model)
    X = InputDomain(
        lower=torch.tensor([-1.0, -0.5, 0.0]),
        upper=torch.tensor([1.0, 0.5, 2.0]),
    )

    report = MaxBound(model, cl_model, X, {"norm": "linf", "return_report": True})
    check = empirical_check(model, cl_model, X, report.bound, {"samples": 500, "seed": 11})

    print(f"formal L-infinity bound: {report.bound:.6f}")
    print(f"max sampled error:       {check['max_observed_error']:.6f}")
    print(f"empirical check passed:  {check['passed']}")


if __name__ == "__main__":
    main()
