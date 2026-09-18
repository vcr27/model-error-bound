"""Bound a supported classification-head slice from a real Hugging Face model."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification

from model_error_bound import (
    InputDomain,
    MaxBound,
    empirical_check,
    extract_sequence_classification_head,
    quantize_linear_model,
)


MODEL_ID = "takedarn/bert-tiny-sst2"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    huggingface_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    head = extract_sequence_classification_head(huggingface_model)
    compiled_head = quantize_linear_model(head, decimals=2)

    hidden_size = head[0].in_features
    X = InputDomain.from_center_radius(torch.zeros(hidden_size), radius=0.1)
    report = MaxBound(head, compiled_head, X, {"norm": "linf", "return_report": True})
    check = empirical_check(
        head,
        compiled_head,
        X,
        report.bound,
        {"samples": 500, "seed": 11},
    )

    print(f"Hugging Face model:       {MODEL_ID}")
    print("verified slice:           BERT classification head")
    print(f"formal L-infinity bound:  {report.bound:.6f}")
    print(f"max sampled error:        {check['max_observed_error']:.6f}")
    print(f"empirical check passed:   {check['passed']}")


if __name__ == "__main__":
    main()
