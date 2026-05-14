from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Reduce fragmentation on small-VRAM GPUs (e.g. 8GB RTX 4060 mobile).
# Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class NaNSkipTrainer(Trainer):
    _nan_skip_count = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            ):
                nan_grad = True
                break

        if nan_grad:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            self._nan_skip_count += 1
            if self._nan_skip_count % 10 == 1:
                logger.warning(
                    "NaN gradients detected at step %d — zeroed and skipped "
                    "(total skipped: %d)",
                    self.state.global_step, self._nan_skip_count,
                )
            # Return a loss of 0 so logging/accumulation don't see NaN
            return torch.tensor(0.0, device=loss.device)

        return loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("train_nli")


# ── LayerNorm weight restoration ─────────────────────────────────────────────
# DeBERTa-v3 checkpoints on HuggingFace store LayerNorm params as
# gamma/beta (legacy naming) but transformers defines the model with
# PyTorch-standard weight/bias.  from_pretrained() silently DROPS the
# gamma/beta tensors, leaving ~50 LayerNorm layers at defaults (1/0).
# This function reloads the raw checkpoint and injects those values.
def _fix_deberta_layernorm_keys(model, model_name_or_path: str) -> None:
    # ── 1. Load the raw checkpoint tensors ────────────────────────────
    raw: dict | None = None

    # Method A: Try loading from HuggingFace cache / hub
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        # microsoft/deberta-v3-large only has pytorch_model.bin (no safetensors)
        for filename in ("pytorch_model.bin", "model.safetensors"):
            try:
                weight_file = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=filename,
                    local_files_only=True,  # use cache, don't re-download
                )
                logger.info("Loading raw checkpoint from: %s", weight_file)
                if filename.endswith(".safetensors"):
                    from safetensors.torch import load_file  # type: ignore
                    raw = load_file(weight_file, device="cpu")
                else:
                    # Try with weights_only=False for legacy .bin files
                    try:
                        raw = torch.load(weight_file, map_location="cpu", weights_only=True)
                    except Exception:
                        raw = torch.load(weight_file, map_location="cpu", weights_only=False)
                if raw is not None:
                    logger.info("Raw checkpoint loaded: %d keys", len(raw))
                    break
            except Exception as e:
                logger.debug("Could not load %s: %s", filename, e)
                continue
    except ImportError:
        pass

    # Method B: Search HuggingFace cache directory directly
    if raw is None:
        import glob
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.cache/huggingface/transformers"),
        ]
        model_slug = model_name_or_path.replace("/", "--")
        for cache_dir in cache_dirs:
            pattern = os.path.join(cache_dir, f"models--{model_slug}", "**", "pytorch_model.bin")
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                pattern = os.path.join(cache_dir, f"models--{model_slug}", "**", "model.safetensors")
                matches = glob.glob(pattern, recursive=True)
            for match in matches:
                try:
                    logger.info("Loading raw checkpoint from cache: %s", match)
                    if match.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        raw = load_file(match, device="cpu")
                    else:
                        try:
                            raw = torch.load(match, map_location="cpu", weights_only=True)
                        except Exception:
                            raw = torch.load(match, map_location="cpu", weights_only=False)
                    if raw is not None:
                        logger.info("Raw checkpoint loaded from cache: %d keys", len(raw))
                        break
                except Exception as e:
                    logger.debug("Could not load %s: %s", match, e)
            if raw is not None:
                break

    if raw is None:
        logger.error(
            "❌ CRITICAL: Could not load raw checkpoint for LayerNorm fix. "
            "Training WILL fail (loss ~4-17, accuracy ~33%%). "
            "Ensure the model is cached: python -c "
            "\"from transformers import AutoModel; AutoModel.from_pretrained('%s')\"",
            model_name_or_path,
        )
        return

    # ── 2. Find gamma/beta keys and build a suffix-based lookup ───────
    # The raw checkpoint may use different prefixes (e.g. "deberta." vs
    # "deberta.deberta."), so we match by SUFFIX rather than full key.
    gamma_beta_map: dict = {}  # suffix -> tensor
    for raw_key, value in raw.items():
        if ".LayerNorm.gamma" in raw_key:
            suffix = raw_key[raw_key.index(".LayerNorm.gamma"):]
            new_suffix = suffix.replace(".LayerNorm.gamma", ".LayerNorm.weight")
            gamma_beta_map[new_suffix] = value
        elif ".LayerNorm.beta" in raw_key:
            suffix = raw_key[raw_key.index(".LayerNorm.beta"):]
            new_suffix = suffix.replace(".LayerNorm.beta", ".LayerNorm.bias")
            gamma_beta_map[new_suffix] = value

    if not gamma_beta_map:
        logger.info("Raw checkpoint uses standard weight/bias naming — no LayerNorm fix needed")
        return

    logger.info("Found %d gamma/beta params in raw checkpoint to restore", len(gamma_beta_map))

    # ── 3. Inject into model state dict ───────────────────────────────
    current_state = model.state_dict()
    fixed = 0
    for model_key in list(current_state.keys()):
        for suffix, value in gamma_beta_map.items():
            if model_key.endswith(suffix) and current_state[model_key].shape == value.shape:
                current_state[model_key] = value
                fixed += 1
                break

    if fixed > 0:
        model.load_state_dict(current_state, strict=True)
        logger.info(
            "✅ Restored %d pretrained LayerNorm params (gamma→weight, beta→bias)",
            fixed,
        )
    else:
        logger.warning(
            "⚠️  Found gamma/beta keys in checkpoint but could not match "
            "any to model state dict. LayerNorm layers may be uninitialised!"
        )

# ── Label semantics ──────────────────────────────────────────────────────────
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
NUM_LABELS = 3

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"
DEFAULT_OUTPUT_DIR = "models/nli-deberta-v3-mnli"
DEFAULT_TRAIN_PARQUET = os.path.expanduser(
    "~/Downloads/train-00000-of-00001.parquet"
)
DEFAULT_VAL_MATCHED = os.path.expanduser(
    "~/Downloads/validation_matched-00000-of-00001.parquet"
)
DEFAULT_VAL_MISMATCHED = os.path.expanduser(
    "~/Downloads/validation_mismatched-00000-of-00001.parquet"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_mnli_parquet(path: str) -> Dataset:
    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_parquet(path)
    df = df[["premise", "hypothesis", "label"]].copy()
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ── CRITICAL: Filter out invalid labels ──────────────────────────
    # MNLI contains label=-1 for unlabeled examples.  If these reach
    # CrossEntropyLoss(num_labels=3), the negative index causes an
    # out-of-bounds access → garbage logits → NaN gradients → loss
    # explosion (3e+13).  Keep only valid labels {0, 1, 2}.
    valid_mask = df["label"].isin([0, 1, 2])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(
            "⚠️  Dropping %d rows with invalid labels from %s "
            "(label distribution before filter: %s)",
            n_invalid, path, df["label"].value_counts().to_dict(),
        )
        df = df[valid_mask].reset_index(drop=True)

    # Sanitise text columns — drop rows with empty/whitespace-only text
    for col in ("premise", "hypothesis"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["premise"].str.len() > 0) & (df["hypothesis"].str.len() > 0)]

    logger.info(
        "Loaded %d rows from %s  | label dist: %s",
        len(df), path, df["label"].value_counts().sort_index().to_dict(),
    )
    return Dataset.from_pandas(df, preserve_index=False)


def load_mnli_from_hub() -> tuple:
    from datasets import load_dataset  # type: ignore

    logger.info("Downloading MNLI from HuggingFace Hub (this may take a minute)...")
    mnli = load_dataset("nyu-mll/multi_nli")

    # Keep only the columns we need
    columns_to_keep = ["premise", "hypothesis", "label"]

    train_ds = mnli["train"].select_columns(columns_to_keep)
    val_matched_ds = mnli["validation_matched"].select_columns(columns_to_keep)
    val_mismatched_ds = mnli["validation_mismatched"].select_columns(columns_to_keep)

    logger.info(
        "Downloaded MNLI: train=%d, val_matched=%d, val_mismatched=%d",
        len(train_ds), len(val_matched_ds), len(val_mismatched_ds),
    )
    return train_ds, val_matched_ds, val_mismatched_ds


def tokenize_function(examples, tokenizer, max_length: int = 256):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=max_length,
        padding=False,  # DataCollator handles dynamic padding
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    # ── Device selection ─────────────────────────────────────────────
    use_bf16 = False
    use_fp16 = False
    use_gc = False  # gradient checkpointing — only on CUDA

    if args.device:  # explicit --device override
        device_str = args.device
        logger.info("Using explicit device: %s", device_str)
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    if device_str == "cuda":
        # bf16 is only fast on Ampere+ (compute capability >= 8.0).
        # Turing (T4, 7.5) "supports" bf16 but emulates it — ~2x slower than fp16.
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            use_bf16 = True
            logger.info("Using CUDA backend (bf16, Ampere+)")
        else:
            use_fp16 = True
            logger.info("Using CUDA backend (fp16 mixed-precision, pre-Ampere)")
        use_gc = True  # gradient checkpointing saves VRAM on CUDA

        # Pick optimizer. fp32 Adam stores 8 bytes of state per param (m + v),
        # which OOMs deberta-v3-large (435M) on 8GB cards. Paged 8-bit AdamW
        # cuts optimizer state ~4× and offloads to CPU under pressure.
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if args.optim == "auto":
            args.optim = "paged_adamw_8bit" if total_vram_gb < 10 else "adamw_torch"
            logger.info("Auto-selected optim=%s (GPU VRAM %.1f GB)", args.optim, total_vram_gb)
        if args.optim in ("adamw_bnb_8bit", "paged_adamw_8bit"):
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    f"optim={args.optim} requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                ) from e
    elif device_str == "mps":
        logger.info("Using Apple MPS backend (fp32)")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        # NO gradient checkpointing on MPS — it interacts badly with
        # disentangled attention and produces NaN gradients on ~97% of
        # batches. Without it, batch=16 + max_length=128 still fits in
        # 16GB unified memory.

        # ── CRITICAL: MPS minimum batch size ─────────────────────
        # DeBERTa's disentangled attention produces extremely noisy
        # single-sample gradients that destabilise AdamW (loss
        # explodes to 1e+13, grad_norm → NaN → segfault).  This
        # happens on CPU too — it's an architecture property, not
        # an MPS bug.  batch≥16 averages the gradient enough for
        # stable training.  We enforce this minimum here and remove
        # gradient accumulation (trades effective batch size for
        # gradient quality).
        MPS_MIN_BATCH = 16
        if args.batch_size < MPS_MIN_BATCH:
            logger.warning(
                "⚠️  MPS: batch_size %d → %d (DeBERTa requires ≥%d "
                "for stable gradients on any device)",
                args.batch_size, MPS_MIN_BATCH, MPS_MIN_BATCH,
            )
            args.batch_size = MPS_MIN_BATCH
            args.gradient_accumulation_steps = 1
    else:
        logger.info("Using CPU backend (fp32)")

    # 8-bit optimizers need CUDA. If user picked one on MPS/CPU, or left
    # it on 'auto' off-CUDA, fall back to the plain torch AdamW.
    if args.optim == "auto" or (device_str != "cuda" and args.optim != "adamw_torch"):
        args.optim = "adamw_torch"

    # ── Load tokenizer & model ───────────────────────────────────────
    # Retry with local_files_only=True if the Hub is unreachable — newer
    # transformers pings huggingface.co from the tokenizer constructor
    # (_patch_mistral_regex → model_info), which fails on offline boxes
    # even when the checkpoint is fully cached.
    logger.info("Loading tokenizer & model: %s", args.base_model)

    def _load_offline_fallback(loader, *a, **kw):
        try:
            return loader(*a, **kw)
        except Exception as e:
            msg = str(e).lower()
            if "name resolution" in msg or "connecterror" in msg or "offline" in msg or "connection" in msg:
                logger.warning("Hub unreachable (%s); retrying with local_files_only=True", e.__class__.__name__)
                return loader(*a, local_files_only=True, **kw)
            raise

    tokenizer = _load_offline_fallback(DebertaV2Tokenizer.from_pretrained, args.base_model)
    model = _load_offline_fallback(
        AutoModelForSequenceClassification.from_pretrained,
        args.base_model,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
        # Force fp32 master weights. fp16/bf16 mixed-precision training
        # needs fp32 params + autocast for activations — if the model
        # itself loads in fp16, GradScaler raises "Attempting to unscale
        # FP16 gradients" on the first backward.
        torch_dtype=torch.float32,
    )

    # ── CRITICAL: Fix LayerNorm key naming mismatch ──────────────
    # Without this, all LayerNorm layers stay at default init
    # and the model cannot learn (loss stays ~17+ instead of ~1.1)
    _fix_deberta_layernorm_keys(model, args.base_model)

    # ── CRITICAL: Small-std init for fresh head ──────────────────
    # pooler.dense and classifier are MISSING from the pretrained
    # checkpoint and get HF's default init (std=0.02).  On step 1,
    # AdamW's update is ~lr*sign(grad) for every param, which pushes
    # logits from ~0.5 → ~20 in a single step and eventually NaNs.
    # Shrinking head init to std=0.001 keeps step-1 gradients small
    # enough that the encoder stays stable during warmup.
    with torch.no_grad():
        for module in (model.pooler.dense, model.classifier):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    logger.info("Re-initialized pooler+classifier with std=0.001 (stability fix)")
    
    # When using gradient checkpointing with use_reentrant=False,
    # the inputs to the first checkpointed block must require gradients.
    if use_gc:
        model.enable_input_require_grads()

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %.1fM params (%.1fM trainable)",
        param_count / 1e6,
        trainable / 1e6,
    )

    # ── Load datasets ────────────────────────────────────────────────
    # Try local parquets first, fall back to HuggingFace Hub
    use_local = all(
        os.path.exists(p)
        for p in [args.train_parquet, args.val_matched, args.val_mismatched]
    )

    if use_local:
        logger.info("Loading MNLI from local parquet files…")
        train_ds = load_mnli_parquet(args.train_parquet)
        val_matched_ds = load_mnli_parquet(args.val_matched)
        val_mismatched_ds = load_mnli_parquet(args.val_mismatched)
    else:
        logger.info("Local parquets not found — downloading from HuggingFace Hub…")
        train_ds, val_matched_ds, val_mismatched_ds = load_mnli_from_hub()

    # ── Optional: limit training samples for quick validation ────────
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.max_train_samples))
        logger.info(
            "🔬 Limited training to %d samples (--max-train-samples)",
            args.max_train_samples,
        )

    logger.info("Tokenizing datasets…")
    tok_fn = lambda examples: tokenize_function(
        examples, tokenizer, max_length=args.max_length
    )
    train_ds = train_ds.map(tok_fn, batched=True, desc="Tokenizing train")
    val_matched_ds = val_matched_ds.map(
        tok_fn, batched=True, desc="Tokenizing val_matched"
    )
    val_mismatched_ds = val_mismatched_ds.map(
        tok_fn, batched=True, desc="Tokenizing val_mismatched"
    )

    # ── Data collator (dynamic padding → GPU-efficient) ──────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ───────────────────────────────────────────
    # Optimised for Apple M5 16 GB:
    #   batch_size=16, grad_accum=2 → effective batch 32
    #   3 epochs ≈ 36,800 steps
    #   ~30-40 min on MPS
    total_train_samples = len(train_ds)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = total_train_samples // effective_batch
    total_steps = steps_per_epoch * args.epochs
    warmup_ratio = 0.06  # 6% warmup — standard for DeBERTa-v3 fine-tuning

    logger.info(
        "Training config: batch=%d, grad_accum=%d, effective_batch=%d, "
        "epochs=%d, total_steps=%d, warmup_ratio=%.2f, lr=%.1e",
        args.batch_size,
        args.gradient_accumulation_steps,
        effective_batch,
        args.epochs,
        total_steps,
        warmup_ratio,
        args.learning_rate,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # ── Training ─────────────────────────────────────────────
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",      # cosine >> linear for DeBERTa
        # ── Optimizer (critical for DeBERTa-v3 stability) ────────
        optim=args.optim,
        adam_epsilon=1e-6,        # DeBERTa paper uses 1e-6 (not 1e-8)
        max_grad_norm=1.0,        # CRITICAL: clip exploding gradients
        # ── Precision & Memory ───────────────────────────────────
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_gc,
        # use_reentrant=False is required on MPS/CUDA for correct gradients.
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        # ── Evaluation ───────────────────────────────────────────
        # Evaluate once per epoch — NO early stopping so all 3
        # epochs run to completion.  (EarlyStopping was killing
        # training during the warmup plateau.)
        # On MPS: save every 1000 steps for crash recovery, eval per epoch
        # (eval is ~10 min on MPS — too slow to run with save). Final model
        # is the last checkpoint, not necessarily the best by eval accuracy.
        # On CUDA: save+eval per epoch, keep best-model selection.
        eval_strategy="epoch",
        save_strategy="steps" if device_str == "mps" else "epoch",
        save_steps=1000 if device_str == "mps" else 500,
        save_total_limit=3,
        load_best_model_at_end=(device_str != "mps"),
        metric_for_best_model="accuracy",
        greater_is_better=True,
        # ── Logging ──────────────────────────────────────────────
        logging_steps=50,
        report_to="none",
        # ── Performance ──────────────────────────────────────────
        dataloader_num_workers=2 if device_str == "cuda" else 0,
        dataloader_pin_memory=(device_str == "cuda"),
        # ── Misc ─────────────────────────────────────────────────
        seed=42,
        remove_unused_columns=True,
        # Force CPU when MPS is detected but disabled (HF Trainer
        # auto-selects MPS otherwise, ignoring our device_str).
        use_cpu=(device_str == "cpu"),
    )

    # ── Trainer ──────────────────────────────────────────────────────
    # NO EarlyStoppingCallback — we want all 3 epochs to run.
    # Early stopping was prematurely killing training during the
    # warmup phase when accuracy temporarily plateaus.
    TrainerClass = NaNSkipTrainer if device_str == "mps" else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_matched_ds,  # eval on matched during training
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training…")
    logger.info("=" * 60)
    t0 = time.time()

    train_result = trainer.train()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Training complete in %.1f min", elapsed / 60)
    logger.info(
        "  Train loss: %.4f",
        train_result.training_loss,
    )
    logger.info("=" * 60)

    # ── Evaluate on both validation sets ─────────────────────────────
    logger.info("Evaluating on val_matched…")
    matched_metrics = trainer.evaluate(eval_dataset=val_matched_ds)
    logger.info("  val_matched accuracy: %.4f", matched_metrics["eval_accuracy"])

    logger.info("Evaluating on val_mismatched…")
    mismatched_metrics = trainer.evaluate(eval_dataset=val_mismatched_ds)
    logger.info(
        "  val_mismatched accuracy: %.4f",
        mismatched_metrics["eval_accuracy"],
    )

    # ── Detailed classification report ───────────────────────────────
    logger.info("Generating detailed classification report (val_matched)…")
    predictions = trainer.predict(val_matched_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, digits=4,
        zero_division=0,  # suppress UndefinedMetricWarning
    )
    logger.info("\n%s", report)

    cm = confusion_matrix(labels, preds)
    logger.info("Confusion matrix:\n%s", cm)

    # ── Save final model ─────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    logger.info("Saving final model to %s…", final_path)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save label mapping for inference
    import json
    label_map = {
        "id2label": {str(i): name for i, name in enumerate(LABEL_NAMES)},
        "label2id": {name: i for i, name in enumerate(LABEL_NAMES)},
        "nli_to_verdict": {
            "entailment": "SUPPORTED",
            "neutral": "UNVERIFIABLE",
            "contradiction": "CONTRADICTED",
        },
    }
    map_path = os.path.join(final_path, "label_mapping.json")
    with open(map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Label mapping saved to %s", map_path)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info("  Model:              %s", args.base_model)
    logger.info("  Training samples:   %d", len(train_ds))
    logger.info("  Training time:      %.1f min", elapsed / 60)
    logger.info("  Val matched acc:    %.4f", matched_metrics["eval_accuracy"])
    logger.info("  Val mismatched acc: %.4f", mismatched_metrics["eval_accuracy"])
    logger.info("  Saved to:           %s", final_path)
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-base on MNLI for NLI classification"
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for trained model (default: %(default)s)",
    )
    parser.add_argument(
        "--train-parquet",
        default=DEFAULT_TRAIN_PARQUET,
        help="Path to MNLI train parquet file",
    )
    parser.add_argument(
        "--val-matched",
        default=DEFAULT_VAL_MATCHED,
        help="Path to MNLI val_matched parquet file",
    )
    parser.add_argument(
        "--val-mismatched",
        default=DEFAULT_VAL_MISMATCHED,
        help="Path to MNLI val_mismatched parquet file",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = batch × accum)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, recommended for DeBERTa-v3)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for premise+hypothesis",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training to N samples (for quick pipeline validation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force device (default: auto-detect, prefers CUDA > CPU on Apple Silicon)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="auto",
        choices=["auto", "adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit"],
        help=(
            "Optimizer. 'auto' = paged_adamw_8bit on CUDA with <10GB VRAM "
            "(needed for deberta-v3-large on 8GB cards), else adamw_torch. "
            "The 8-bit variants require bitsandbytes."
        ),
    )
    args, _ = parser.parse_known_args()

    # Check if local parquets exist (if not, script auto-downloads from Hub)
    has_local = all(
        os.path.exists(p)
        for p in [args.train_parquet, args.val_matched, args.val_mismatched]
    )
    if has_local:
        logger.info("Found local parquet files — will use those.")
    else:
        logger.info("Local parquets not found — will download MNLI from HuggingFace Hub.")

    train(args)


if __name__ == "__main__":
    main()
