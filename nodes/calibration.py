"""
hallu-check | nodes/calibration.py
Platt Scaling confidence calibration for NLI model outputs.

The raw softmax probabilities from the NLI model are often
overconfident. Platt scaling (logistic regression on logits)
produces properly calibrated probabilities where a 0.8 confidence
actually means 80% accuracy.

Usage:
    # Train calibration (run once):
    python -m nodes.calibration --train

    # Calibration is then auto-applied in nli_model.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("hallu-check.calibration")

CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "nli-deberta-v3-mnli", "final", "calibration_params.json",
)

# Cached calibration parameters
_params: Optional[Dict] = None


def _load_params() -> Optional[Dict]:
    """Load calibration parameters from disk."""
    global _params
    if _params is not None:
        return _params

    if not os.path.exists(CALIBRATION_PATH):
        return None

    try:
        with open(CALIBRATION_PATH, "r") as f:
            _params = json.load(f)
        logger.info("Loaded calibration params from %s", CALIBRATION_PATH)
        return _params
    except Exception as e:
        logger.warning("Failed to load calibration params: %s", e)
        return None


def calibrate(probs: Dict[str, float]) -> Dict[str, float]:
    """
    Apply Platt scaling to raw NLI probabilities.

    If calibration parameters are not available, returns raw probs unchanged.

    Args:
        probs: Dict with keys 'entailment', 'neutral', 'contradiction'
              and raw softmax probability values.

    Returns:
        Calibrated probability dict (same keys).
    """
    params = _load_params()
    if params is None:
        return probs  # no calibration available

    try:
        calibrated = {}
        for label in ["entailment", "neutral", "contradiction"]:
            raw = probs.get(label, 0.0)
            # Platt scaling: p_calibrated = 1 / (1 + exp(-(a * logit + b)))
            # where logit = log(p / (1 - p))
            a = params.get(f"{label}_a", 1.0)
            b = params.get(f"{label}_b", 0.0)

            # Clip to avoid log(0) or log(inf)
            raw_clipped = np.clip(raw, 1e-7, 1 - 1e-7)
            logit = np.log(raw_clipped / (1 - raw_clipped))
            calibrated_logit = a * logit + b
            calibrated[label] = float(1.0 / (1.0 + np.exp(-calibrated_logit)))

        # Re-normalize to sum to 1
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in calibrated.items()}

        return calibrated
    except Exception as e:
        logger.warning("Calibration failed: %s. Using raw probabilities.", e)
        return probs


def train_calibration(
    val_parquet: str = os.path.expanduser("~/Downloads/validation_matched-00000-of-00001.parquet"),
    max_samples: int = 2000,
) -> Dict:
    """
    Train Platt scaling parameters on the MNLI validation set.

    Runs the NLI model on val samples, collects (raw_prob, true_label) pairs,
    and fits logistic regression per class.

    Args:
        val_parquet: Path to MNLI validation parquet file.
        max_samples: Number of samples to use for calibration.

    Returns:
        Dict with calibration parameters.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import pandas as pd
    from sklearn.linear_model import LogisticRegression  # type: ignore

    from nodes.nli_model import load_model, classify_nli_batch  # type: ignore

    if not load_model():
        raise RuntimeError("NLI model not loaded.")

    logger.info("Loading validation data for calibration...")
    df = pd.read_parquet(val_parquet)
    df = df[["premise", "hypothesis", "label"]].dropna().head(max_samples)

    # Model's label mapping (may differ from MNLI standard)
    from nodes.nli_model import _model  # type: ignore
    model_id2label = _model.config.id2label
    # Map MNLI labels (0=ent, 1=neut, 2=contra) to model label names
    mnli_to_name = {0: "entailment", 1: "neutral", 2: "contradiction"}

    logger.info("Running NLI model on %d validation samples...", len(df))
    pairs = list(zip(df["premise"].tolist(), df["hypothesis"].tolist()))
    results = classify_nli_batch(pairs, batch_size=32)

    # Collect per-class raw probabilities and true binary labels
    params = {}
    label_names = ["entailment", "neutral", "contradiction"]

    for label_name in label_names:
        raw_probs = []
        true_binary = []

        for i, (result, true_label) in enumerate(zip(results, df["label"].tolist())):
            raw_prob = result["probabilities"].get(label_name, 0.0)
            is_true = (mnli_to_name[int(true_label)] == label_name)
            raw_probs.append(raw_prob)
            true_binary.append(1 if is_true else 0)

        raw_probs = np.array(raw_probs)
        true_binary = np.array(true_binary)

        # Clip and convert to logits
        raw_clipped = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(raw_clipped / (1 - raw_clipped)).reshape(-1, 1)

        # Fit Platt scaling (logistic regression)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(logits, true_binary)

        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        params[f"{label_name}_a"] = round(a, 6)
        params[f"{label_name}_b"] = round(b, 6)

        # Report accuracy
        preds = lr.predict(logits)
        acc = np.mean(preds == true_binary)
        logger.info(
            "  %s: a=%.4f, b=%.4f, calibration_acc=%.4f",
            label_name, a, b, acc,
        )

    # Save parameters
    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Calibration params saved to %s", CALIBRATION_PATH)

    # Reset cached params so they're reloaded next time
    global _params
    _params = None

    return params


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train or test confidence calibration")
    parser.add_argument("--train", action="store_true", help="Train calibration params")
    parser.add_argument("--val-parquet", default=os.path.expanduser("~/Downloads/validation_matched-00000-of-00001.parquet"))
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    if args.train:
        params = train_calibration(args.val_parquet, args.max_samples)
        print("\nCalibration parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
    else:
        params = _load_params()
        if params:
            print("Current calibration parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")
        else:
            print("No calibration params found. Run with --train first.")


if __name__ == "__main__":
    main()
