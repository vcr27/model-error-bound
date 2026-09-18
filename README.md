# Model Error Bound

This package implements Task 3: a small formal-method toolbox for bounding the
maximum output difference between a PyTorch DNN and a compiled version of the
same model. The implementation targets small `torch.nn.Module` networks or
well-defined sub-networks, which matches the task's allowed scope because full
formal verification of a large LLM is generally intractable.

The public API is:

```python
from model_error_bound import MaxBound

cl_model = cl_func(model)
bound = MaxBound(model, cl_model, X)
```

## Method

The implemented formal method is Interval Bound Propagation (IBP). The input
domain `X` is a box: every input feature has a lower and upper bound. Instead of
testing only individual samples, IBP pushes the whole input interval through the
network layer by layer.

For a `Linear` layer, positive weights multiply the lower bound to produce the
new lower bound, while negative weights multiply the upper bound. This is the
standard interval arithmetic rule that safely contains every possible linear
output. For `ReLU`, the lower and upper bounds are passed through `max(0, x)`.

After computing output intervals for both models, the package bounds the
difference interval:

```text
model(x) - compiled_model(x)
```

The default norm is `L-infinity`, so the returned value is the largest possible
absolute output-coordinate difference under the interval relaxation.

## Scope and Assumptions

This implementation supports deterministic feed-forward PyTorch networks made
from:

- `torch.nn.Sequential`
- `torch.nn.Linear`
- `torch.nn.ReLU`
- `torch.nn.Flatten`
- `torch.nn.Identity`

The bound is sound for these supported layers and box-shaped input domains. It
can be conservative because interval methods lose correlations between neurons.
For example, if two neurons depend on the same input, IBP may treat their worst
cases as if they can happen independently.

The included `quantize_linear_model` helper rounds `Linear` weights and biases.
It is used as a small, explainable stand-in for a compiler or quantization pass.
The package also includes `torch_compile_model`, which calls `torch.compile` and
lets the same `MaxBound(model, cl_model, X)` API compare the original module with
the compiled wrapper.

The Hugging Face demonstration loads a real fine-tuned tiny BERT model and
verifies one well-defined supported slice: its classification head
(`Linear`; evaluation-time dropout is the identity). It does **not** claim to
verify the complete transformer. Extending the formal guarantee through the full
model would require sound rules for embeddings, attention, LayerNorm, residual
connections, and other operations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

```bash
python examples/quickstart.py
python examples/torch_compile_demo.py
python examples/huggingface_demo.py
```

Example usage:

```python
import torch
from torch import nn

from model_error_bound import InputDomain, MaxBound, quantize_linear_model

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
)

cl_func = lambda original: quantize_linear_model(original, decimals=2)
cl_model = cl_func(model)
X = InputDomain(
    lower=torch.tensor([-1.0, -0.5, 0.0]),
    upper=torch.tensor([1.0, 0.5, 2.0]),
)

bound = MaxBound(model, cl_model, X)
print(bound)
```

## Results

On the quickstart network, `MaxBound` returns a formal upper bound and the
empirical checker samples points from `X` to confirm that no sampled difference
exceeds it. Sampling is only a sanity check; the guarantee comes from interval
propagation.

| Experiment | Formal method | Norm | Expected result |
| --- | --- | --- | --- |
| Hand-checkable 1D linear model | Exact interval arithmetic | `L-infinity` | Bound equals `1.5` |
| Small ReLU network with rounded weights | IBP | `L-infinity` | Sampled errors stay below the bound |
| Small ReLU network with `torch.compile` | IBP over the wrapped original module | `L-infinity` | Structural bound is `0.0`; sampled error was `0.0` on the tested CPU run |
| Real Hugging Face tiny BERT classification head | IBP over a supported model slice | `L-infinity` | Sampled errors stay below the formal bound |

## Design Walkthrough

- `InputDomain` stores `lower` and `upper` tensors and validates that they form
  a real box.
- `MaxBound(model, cl_model, X)` computes output intervals for both models and
  returns a sound upper bound on their difference.
- `empirical_check(...)` randomly samples inputs from `X` to make sure observed
  errors do not exceed the formal bound.
- `quantize_linear_model(...)` creates a compiled model by rounding weights and
  biases, giving the example a concrete `cl_func`.
- `torch_compile_model(...)` creates a `torch.compile` version of the model. In
  this case the compiled object wraps the same module parameters, so the formal
  structural bound is zero under the assumption that `torch.compile` preserves
  the model semantics. The empirical check is still useful for observing backend
  numerical differences.
- `extract_sequence_classification_head(...)` adapts the deterministic head of
  a Hugging Face BERT or DistilBERT sequence classifier. Dropout is omitted
  because the model is placed in evaluation mode, where dropout is the identity.
  The formal claim applies to this head and its hidden-feature box `X`, not to
  token inputs or the full transformer.

## Running Tests

```bash
pytest
```
