"""Small compile/optimization examples used by the package and quickstart."""

from __future__ import annotations

import copy

import torch
from torch import nn


def quantize_linear_model(model: nn.Module, decimals: int = 2) -> nn.Module:
    """Return a copied model whose Linear weights/biases are rounded.

    This stands in for a simple compiler or quantization pass. It deliberately
    preserves the architecture, making the model-vs-compiled difference easy to
    reason about while still producing non-identical outputs.
    """

    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    compiled = copy.deepcopy(model)
    scale = float(10**decimals)
    with torch.no_grad():
        for module in compiled.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(torch.round(module.weight * scale) / scale)
                if module.bias is not None:
                    module.bias.copy_(torch.round(module.bias * scale) / scale)
    return compiled
