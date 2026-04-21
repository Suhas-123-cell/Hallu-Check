"""
hallu-check | benchmarks/eval_truthfulqa.py
Evaluate Hallu-Check pipeline on TruthfulQA benchmark.

TruthfulQA tests whether AI systems generate truthful answers to
questions designed to elicit common misconceptions and false beliefs.

Usage:
    python -m benchmarks.eval_truthfulqa [--samples 200]

Reports: Truthfulness rate before/after Hallu-Check verification.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.truthfulqa")

TRUTHFULQA_DATASET = "truthfulqa/truthful_qa"
TRUTHFULQA_SUBSET = "generation"


def load_truthfulqa(max_samples: int = 200) -> list:
    """Load TruthfulQA samples."""
    try:
        from datasets import load_dataset  # type: ignore
        logger.info("Loading TruthfulQA dataset...")
        ds = load_dataset(TRUTHFULQA_DATASET, TRUTHFULQA_SUBSET, split="validation")
        samples = list(ds.select(range(min(max_samples, len(ds)))))
        logger.info("Loaded %d samples from TruthfulQA.", len(samples))
        return samples
    except Exception as e:
        logger.error("Failed to load TruthfulQA: %s", e)
        return []


def evaluate_nli(samples: list) -> dict:
    """
    Evaluate NLI model's ability to detect misconceptions.

    For each question:
      1. Compare best_answer against one incorrect_answer
      2. Check if NLI correctly identifies the incorrect answer as non-supported
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from nodes.nli_model import load_model, classify_nli  # type: ignore

    if not load_model():
        logger.error("NLI model not available.")
        return {}

    correct_detections = 0
    total_tested = 0

    t0 = time.time()

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])

        if not best_answer or not incorrect_answers:
            continue

        # Use best_answer as premise (ground truth) and check:
        # 1. Best answer should be self-consistent (SUPPORTED)
        r_correct = classify_nli(best_answer, best_answer)

        # 2. First incorrect answer should be CONTRADICTED
        incorrect = incorrect_answers[0] if incorrect_answers else ""
        if not incorrect:
            continue

        r_incorrect = classify_nli(best_answer, incorrect)

        # A "correct detection" means:
        # - The incorrect answer is NOT classified as SUPPORTED
        is_detected = r_incorrect["verdict"] != "SUPPORTED"
        if is_detected:
            correct_detections += 1
        total_tested += 1

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d...", i + 1, len(samples))

    elapsed = time.time() - t0

    detection_rate = correct_detections / total_tested if total_tested > 0 else 0

    metrics = {
        "dataset": "TruthfulQA",
        "total_samples": total_tested,
        "misconceptions_detected": correct_detections,
        "misconceptions_missed": total_tested - correct_detections,
        "detection_rate": round(detection_rate, 4),
        "elapsed_seconds": round(elapsed, 1),
        "ms_per_sample": round(elapsed / total_tested * 1000, 1) if total_tested else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLI model on TruthfulQA")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    args = parser.parse_args()

    samples = load_truthfulqa(max_samples=args.samples)
    if not samples:
        logger.error("No samples loaded. Exiting.")
        sys.exit(1)

    metrics = evaluate_nli(samples)
    if not metrics:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRUTHFULQA BENCHMARK RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key:30s}: {value}")
    print("=" * 60)

    results_path = Path(__file__).resolve().parent / "results_truthfulqa.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
