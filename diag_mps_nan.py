"""
Diagnose where the NaN is coming from in train_nli.py.

Runs ONE forward+backward on CPU and MPS with the same inputs+seed,
and reports:
  - max |logit| at init            → is the forward pass producing extreme values?
  - loss                           → is cross-entropy sane?
  - max |grad| across all params   → is the backward producing NaN/Inf?
  - any NaN/Inf in encoder output  → which device is the culprit?

Usage:
    python diag_mps_nan.py
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
)


def run_on(device_str: str) -> dict:
    torch.manual_seed(0)
    tok = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=3
    ).to(device_str)
    model.train()

    batch = tok(
        ["A man is walking."] * 4,
        ["A person moves."] * 4,
        padding=True, truncation=True, max_length=64, return_tensors="pt",
    ).to(device_str)
    labels = torch.tensor([0, 1, 2, 0], device=device_str)

    out = model(**batch, output_hidden_states=True, labels=labels)
    last_hidden = out.hidden_states[-1]
    logits = out.logits

    stats = {
        "device": device_str,
        "hidden_max_abs": last_hidden.abs().max().item(),
        "hidden_has_nan": torch.isnan(last_hidden).any().item(),
        "logit_max_abs": logits.abs().max().item(),
        "loss": out.loss.item(),
    }

    out.loss.backward()
    grad_max = 0.0
    grad_has_nan = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad
            if torch.isnan(g).any().item() or torch.isinf(g).any().item():
                grad_has_nan = True
            gm = g.abs().max().item()
            if gm > grad_max:
                grad_max = gm
    stats["grad_max_abs"] = grad_max
    stats["grad_has_nan_or_inf"] = grad_has_nan
    return stats


if __name__ == "__main__":
    print("=== CPU ===")
    cpu_stats = run_on("cpu")
    for k, v in cpu_stats.items():
        print(f"  {k}: {v}")

    if torch.backends.mps.is_available():
        print("\n=== MPS ===")
        mps_stats = run_on("mps")
        for k, v in mps_stats.items():
            print(f"  {k}: {v}")
    else:
        print("\nMPS not available on this machine.")
