"""
Diagnostic script for DeBERTa-v3 NLI training issue.
Checks: LayerNorm weights, forward pass logits, label distribution, MPS issues.
"""
import torch
import numpy as np
import sys

print("=" * 60)
print("DeBERTa-v3 NLI Training Diagnostic")
print("=" * 60)

# ── 1. Check checkpoint key naming ──────────────────────────────────────
print("\n[1/5] Checking checkpoint key naming convention...")
model_name = "microsoft/deberta-v3-large"

try:
    from huggingface_hub import hf_hub_download
    checkpoint_path = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            checkpoint_path = hf_hub_download(
                repo_id=model_name, filename=fname, local_files_only=True
            )
            print(f"  Found cached checkpoint: {fname}")
            break
        except Exception:
            continue

    if checkpoint_path:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(checkpoint_path, device="cpu")
        else:
            raw_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        gamma_keys = [k for k in raw_sd if ".gamma" in k or ".beta" in k]
        weight_bias_keys = [k for k in raw_sd if "LayerNorm.weight" in k or "LayerNorm.bias" in k]
        print(f"  Checkpoint has {len(gamma_keys)} gamma/beta keys (old naming)")
        print(f"  Checkpoint has {len(weight_bias_keys)} weight/bias keys (new naming)")
        if gamma_keys:
            print(f"  ⚠️  Checkpoint uses LEGACY gamma/beta naming!")
            print(f"  Sample keys: {gamma_keys[:3]}")
        if weight_bias_keys:
            print(f"  ✅ Checkpoint uses standard weight/bias naming")
    else:
        print("  ❌ Could not find cached checkpoint!")
except Exception as e:
    print(f"  Error checking checkpoint: {e}")

# ── 2. Load model and check LayerNorm weights ───────────────────────────
print("\n[2/5] Loading model and checking LayerNorm weights...")
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

layernorm_default_weight = 0
layernorm_default_bias = 0
layernorm_pretrained_weight = 0
layernorm_pretrained_bias = 0
total_ln = 0

for name, param in model.named_parameters():
    if "LayerNorm" not in name:
        continue
    total_ln += 1
    if "weight" in name:
        is_default = torch.allclose(param.data, torch.ones_like(param.data), atol=1e-6)
        if is_default:
            layernorm_default_weight += 1
        else:
            layernorm_pretrained_weight += 1
    elif "bias" in name:
        is_default = torch.allclose(param.data, torch.zeros_like(param.data), atol=1e-6)
        if is_default:
            layernorm_default_bias += 1
        else:
            layernorm_pretrained_bias += 1

print(f"  Total LayerNorm params: {total_ln}")
print(f"  Weight - pretrained: {layernorm_pretrained_weight}, default(all-1s): {layernorm_default_weight}")
print(f"  Bias   - pretrained: {layernorm_pretrained_bias}, default(all-0s): {layernorm_default_bias}")

if layernorm_default_weight > 0 or layernorm_default_bias > 0:
    print(f"  ❌ CRITICAL: {layernorm_default_weight + layernorm_default_bias} LayerNorm params at DEFAULT init!")
    print(f"     This is WHY the model cannot learn!")
else:
    print(f"  ✅ All LayerNorm weights appear to be pretrained (non-default)")

# ── 3. Test forward pass on CPU ─────────────────────────────────────────
print("\n[3/5] Testing forward pass (CPU)...")
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model.eval()

test_pairs = [
    ("The capital of France is Paris.", "Paris is the capital of France."),
    ("The sky is blue.", "The sky is red."),
    ("Dogs are animals.", "Cats can fly."),
]

for premise, hypothesis in test_pairs:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        probs = torch.softmax(logits, dim=-1)

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"  Premise: {premise[:50]}")
    print(f"  Hypothesis: {hypothesis[:50]}")
    print(f"  Logits: [{', '.join(f'{l:.4f}' for l in logits.tolist())}]")
    print(f"  Probs:  [{', '.join(f'{p:.4f}' for p in probs.tolist())}]")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    # Compute cross-entropy loss for label=0 (entailment)
    ce_loss = -torch.log(probs[0]).item()
    print(f"  CE loss (if label=entailment): {ce_loss:.4f}")
    print()

# ── 4. Check MPS gradient checkpointing ─────────────────────────────────
print("[4/5] Checking MPS + gradient checkpointing compatibility...")
if torch.backends.mps.is_available():
    print(f"  MPS is available")
    try:
        model_mps = model.to("mps")
        model_mps.train()
        model_mps.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}  # DEFAULT
        )

        inputs_mps = {k: v.to("mps") for k, v in inputs.items()}
        labels = torch.tensor([0], device="mps")

        # Forward + backward with use_reentrant=True
        outputs = model_mps(**inputs_mps, labels=labels)
        loss_reentrant = outputs.loss.item()
        outputs.loss.backward()

        grad_norm_reentrant = sum(
            p.grad.norm().item() ** 2 for p in model_mps.parameters() if p.grad is not None
        ) ** 0.5

        model_mps.zero_grad()

        # Now test with use_reentrant=False
        model_mps.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        outputs2 = model_mps(**inputs_mps, labels=labels)
        loss_non_reentrant = outputs2.loss.item()
        outputs2.loss.backward()

        grad_norm_non_reentrant = sum(
            p.grad.norm().item() ** 2 for p in model_mps.parameters() if p.grad is not None
        ) ** 0.5

        print(f"  use_reentrant=True:  loss={loss_reentrant:.4f}, grad_norm={grad_norm_reentrant:.4f}")
        print(f"  use_reentrant=False: loss={loss_non_reentrant:.4f}, grad_norm={grad_norm_non_reentrant:.4f}")

        if abs(grad_norm_reentrant - grad_norm_non_reentrant) / max(grad_norm_non_reentrant, 1e-8) > 0.1:
            print(f"  ⚠️  Gradient norms differ significantly! use_reentrant=True may cause issues on MPS.")
        else:
            print(f"  ✅ Gradient norms are similar between reentrant modes.")

        model_mps.to("cpu")
        del model_mps
        torch.mps.empty_cache()

    except Exception as e:
        print(f"  Error testing MPS gradient checkpointing: {e}")
else:
    print("  MPS not available, skipping.")

# ── 5. Check dataset labels ─────────────────────────────────────────────
print("\n[5/5] Checking MNLI dataset labels...")
try:
    from datasets import load_dataset
    mnli = load_dataset("nyu-mll/multi_nli", split="train[:1000]")
    labels = mnli["label"]
    unique_labels = set(labels)
    print(f"  Unique labels in first 1000 samples: {sorted(unique_labels)}")
    from collections import Counter
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        print(f"    Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    if unique_labels == {0, 1, 2}:
        print(f"  ✅ Labels are correct (0, 1, 2)")
    else:
        print(f"  ❌ Unexpected labels! Expected {{0, 1, 2}}, got {unique_labels}")
except Exception as e:
    print(f"  Error checking dataset: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete.")
print("=" * 60)
