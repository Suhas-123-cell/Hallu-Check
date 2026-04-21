"""
Step-by-step training trace on real MNLI data.

Reproduces the exact training setup from train_nli.py (batch=16, max_length=128,
AdamW, lr=1e-5, real MNLI rows, DataCollatorWithPadding) but logs every step
and prints per-step stats. Aborts at the first NaN/Inf so you see exactly
which batch and which tensor triggers the divergence.

Usage:
    python diag_step_by_step.py              # runs on MPS
    python diag_step_by_step.py --device cpu # runs on CPU for comparison
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
)

TRAIN_PARQUET = os.path.expanduser("~/Downloads/train-00000-of-00001.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(42)

    tok = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=3
    ).to(args.device)

    # Shrink head init so fresh pooler+classifier don't destabilize training
    with torch.no_grad():
        for module in (model.pooler.dense, model.classifier):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.train()

    n_rows = args.steps * args.batch_size + args.batch_size
    df = pd.read_parquet(TRAIN_PARQUET).head(n_rows)
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
    print(f"Loaded {len(df)} rows. Label dist: {df['label'].value_counts().to_dict()}")

    def tokenize(examples):
        return tok(
            examples["premise"], examples["hypothesis"],
            truncation=True, max_length=args.max_length, padding=False,
        )

    ds = Dataset.from_pandas(df[["premise", "hypothesis", "label"]], preserve_index=False)
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["premise", "hypothesis"])

    collator = DataCollatorWithPadding(tokenizer=tok)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optim = AdamW(model.parameters(), lr=1e-5, eps=1e-6, weight_decay=0.01)

    print(f"\n{'step':>4} | {'loss':>10} | {'logit_max':>10} | {'grad_max':>12} | {'seq_len':>8} | notes")
    print("-" * 80)

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        batch = {k: v.to(args.device) for k, v in batch.items()}
        seq_len = batch["input_ids"].shape[1]

        optim.zero_grad()
        out = model(**batch)
        loss = out.loss
        logit_max = out.logits.abs().max().item()

        notes = []
        if torch.isnan(loss) or torch.isinf(loss):
            notes.append("LOSS_NAN")

        loss.backward()
        grad_max = 0.0
        grad_nan = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    grad_nan = True
                gm = p.grad.abs().max().item()
                if gm > grad_max:
                    grad_max = gm
        if grad_nan:
            notes.append("GRAD_NAN")

        # Same clipping as train_nli.py (max_grad_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        print(f"{step:>4} | {loss.item():>10.4f} | {logit_max:>10.4f} | {grad_max:>12.4f} | {seq_len:>8} | {' '.join(notes)}")

        if notes:
            print(f"\n❌ Diverged at step {step}. Dumping batch for inspection:")
            print(f"   input_ids shape: {batch['input_ids'].shape}")
            print(f"   attention_mask sum per row: {batch['attention_mask'].sum(dim=1).tolist()}")
            print(f"   labels: {batch['labels'].tolist()}")
            break
    else:
        print(f"\n✅ Completed {args.steps} steps with no NaN on device={args.device}.")


if __name__ == "__main__":
    main()
