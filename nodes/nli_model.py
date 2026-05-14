from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
)

logger = logging.getLogger("hallu-check.nli_model")

# ── Label mappings 
NLI_TO_VERDICT = {
    "entailment": "SUPPORTED",
    "neutral": "UNVERIFIABLE",
    "contradiction": "CONTRADICTED",
}
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# ── Singleton state 
_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[DebertaV2Tokenizer] = None
_device: Optional[torch.device] = None
_loaded: bool = False


def _detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_path: str | None = None,
    device: str | None = None,
) -> bool:
    global _model, _tokenizer, _device, _loaded

    if _loaded:
        return True

    if model_path is None:
        from config import NLI_MODEL_PATH  # type: ignore[import-not-found]
        model_path = NLI_MODEL_PATH

    model_dir = Path(model_path)
    if not model_dir.exists():
        logger.warning(
            "NLI model not found at %s. Run train_nli.py first.", model_path
        )
        return False

    try:
        logger.info("Loading NLI model from %s…", model_path)

        _device = torch.device(device) if device else _detect_device()

        _tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.to(_device)
        _model.eval()

        # Read label mapping from model config
        global LABEL_NAMES
        id2label = _model.config.id2label
        LABEL_NAMES = [id2label[i] for i in range(len(id2label))]
        logger.info("NLI label mapping: %s", {i: l for i, l in enumerate(LABEL_NAMES)})

        param_count = sum(p.numel() for p in _model.parameters())
        logger.info(
            "NLI model loaded: %.1fM params on %s",
            param_count / 1e6,
            _device,
        )
        _loaded = True
        return True

    except Exception as e:
        logger.error("Failed to load NLI model: %s", e)
        _model = None
        _tokenizer = None
        _loaded = False
        return False


def is_loaded() -> bool:
    return _loaded



def classify_nli(
    premise: str,
    hypothesis: str,
    max_length: int = 256,
) -> Dict:
    if not _loaded or _model is None or _tokenizer is None:
        raise RuntimeError(
            "NLI model not loaded. Call load_model() first or run train_nli.py."
        )

    # Tokenize
    inputs = _tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Inference (no gradient computation)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits  # shape: (1, 3)

    # Softmax → calibrated probabilities
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (3,)
    probs_dict = {
        LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
    }

    # Predicted label
    pred_idx = int(torch.argmax(probs).item())
    pred_label = LABEL_NAMES[pred_idx]
    pred_verdict = NLI_TO_VERDICT[pred_label]
    confidence = float(probs[pred_idx])

    return {
        "label": pred_label,
        "verdict": pred_verdict,
        "probabilities": probs_dict,
        "confidence": confidence,
    }


def classify_nli_batch(
    pairs: List[Tuple[str, str]],
    max_length: int = 256,
    batch_size: int = 16,
) -> List[Dict]:
    if not _loaded or _model is None or _tokenizer is None:
        raise RuntimeError(
            "NLI model not loaded. Call load_model() first or run train_nli.py."
        )

    if not pairs:
        return []

    results: List[Dict] = []

    # Process in batches
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]

        # Tokenize batch
        inputs = _tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits  # shape: (batch, 3)

        # Softmax per item
        probs_batch = torch.softmax(logits, dim=-1)  # shape: (batch, 3)

        for j in range(len(batch_pairs)):
            probs = probs_batch[j]
            probs_dict = {
                LABEL_NAMES[k]: float(probs[k]) for k in range(len(LABEL_NAMES))
            }
            pred_idx = int(torch.argmax(probs).item())
            pred_label = LABEL_NAMES[pred_idx]

            results.append({
                "label": pred_label,
                "verdict": NLI_TO_VERDICT[pred_label],
                "probabilities": probs_dict,
                "confidence": float(probs[pred_idx]),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# NLI-based alignment scoring (replaces BERTScore)
# ─────────────────────────────────────────────────────────────────────────────
def compute_nli_alignment(
    candidate: str,
    reference: str,
    max_sentences: int = 10,
) -> Dict:
    import re

    if not _loaded:
        logger.warning("NLI model not loaded, returning default alignment.")
        return {"alignment_score": 0.5, "details": []}

    # Split candidate into sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", candidate)
        if len(s.strip()) > 15  # skip very short fragments
    ]

    if not sentences:
        return {"alignment_score": 0.5, "details": []}

    # Limit to max_sentences to avoid slowdown
    sentences = sentences[:max_sentences]

    # Truncate reference to fit in model context
    ref_truncated = reference[:2000] if len(reference) > 2000 else reference

    # Build pairs: (reference, sentence)
    pairs = [(ref_truncated, sent) for sent in sentences]

    # Batch classify
    results = classify_nli_batch(pairs)

    # Compute alignment score = mean P(entailment)
    entailment_probs = [r["probabilities"]["entailment"] for r in results]
    alignment_score = sum(entailment_probs) / len(entailment_probs)

    details = [
        {
            "sentence": sent,
            "entailment_prob": r["probabilities"]["entailment"],
            "verdict": r["verdict"],
        }
        for sent, r in zip(sentences, results)
    ]

    return {
        "alignment_score": round(alignment_score, 4),
        "details": details,
    }
