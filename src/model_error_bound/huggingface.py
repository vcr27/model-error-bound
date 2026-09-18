"""Adapters for small, explicitly supported slices of Hugging Face models."""

from __future__ import annotations

from torch import nn


def extract_sequence_classification_head(model: nn.Module) -> nn.Sequential:
    """Extract a supported deterministic Hugging Face classification head.

    BERT applies dropout and a Linear classifier to its pooled representation;
    DistilBERT applies Linear, ReLU, dropout, and another Linear layer to its
    first-token representation. In evaluation mode dropout is the identity, so
    both become slices supported by this package. The returned layers share
    parameters with ``model``; compiler helpers copy before modifying them.
    """

    pre_classifier = getattr(model, "pre_classifier", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError(
            "expected a Hugging Face sequence classifier with a Linear classifier module"
        )
    model.eval()
    if isinstance(pre_classifier, nn.Linear):
        return nn.Sequential(pre_classifier, nn.ReLU(), classifier)
    return nn.Sequential(classifier)
