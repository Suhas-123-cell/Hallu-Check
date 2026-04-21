"""
hallu-check | benchmarks/eval_halueval.py
Evaluate Hallu-Check's NLI model on the HaluEval benchmark.

HaluEval provides (question, correct_answer, hallucinated_answer) triples.
We test whether our NLI model can distinguish correct answers (SUPPORTED)
from hallucinated ones (CONTRADICTED/UNVERIFIABLE).

Usage:
    python -m benchmarks.eval_halueval [--samples 500]

Reports: Precision, Recall, F1, Accuracy for hallucination detection.
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
logger = logging.getLogger("benchmark.halueval")

# HaluEval QA dataset on HuggingFace
HALUEVAL_DATASET = "pminervini/HaluEval"
HALUEVAL_SUBSET = "qa_samples"


def load_halueval(max_samples: int = 500) -> list:
    """Load HaluEval QA samples from HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore
        logger.info("Loading HaluEval dataset from HuggingFace...")
        ds = load_dataset(HALUEVAL_DATASET, HALUEVAL_SUBSET, split="data")
        samples = list(ds.select(range(min(max_samples, len(ds)))))
        logger.info("Loaded %d samples from HaluEval.", len(samples))
        return samples
    except Exception as e:
        logger.error("Failed to load HaluEval: %s", e)
        logger.info("Trying local fallback...")
        return []


def evaluate_nli(samples: list) -> dict:
    """
    Run NLI model on HaluEval samples.

    HaluEval schema:
      - knowledge: ground truth context
      - question: the question asked
      - answer: the generated answer (may be hallucinated)
      - hallucination: "yes" or "no"

    We test: given (knowledge, answer), does NLI correctly identify
    hallucinated answers as CONTRADICTED and correct ones as SUPPORTED?
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from nodes.nli_model import load_model, classify_nli  # type: ignore

    if not load_model():
        logger.error("NLI model not available. Run train_nli.py first.")
        return {}

    true_positive = 0   # correctly detected hallucination
    false_positive = 0  # marked correct answer as hallucination
    true_negative = 0   # correctly marked good answer as good
    false_negative = 0  # missed a hallucination

    t0 = time.time()

    for i, sample in enumerate(samples):
        knowledge = sample.get("knowledge", "")
        answer = sample.get("answer", "")
        is_hallucinated = sample.get("hallucination", "").lower() == "yes"

        if not knowledge or not answer:
            continue

        # NLI: premise=knowledge (evidence), hypothesis=answer (claim)
        r = classify_nli(knowledge, answer)

        # Our prediction: is this a hallucination?
        predicted_hallucination = r["verdict"] != "SUPPORTED"

        if is_hallucinated and predicted_hallucination:
            true_positive += 1
        elif is_hallucinated and not predicted_hallucination:
            false_negative += 1
        elif not is_hallucinated and predicted_hallucination:
            false_positive += 1
        else:
            true_negative += 1

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d samples...", i + 1, len(samples))

    elapsed = time.time() - t0
    total = true_positive + false_positive + true_negative + false_negative

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / total if total > 0 else 0

    metrics = {
        "dataset": "HaluEval-QA",
        "total_samples": total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "elapsed_seconds": round(elapsed, 1),
        "ms_per_sample": round(elapsed / total * 1000, 1) if total else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLI model on HaluEval benchmark")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate")
    args = parser.parse_args()

    samples = load_halueval(max_samples=args.samples)
    if not samples:
        logger.error("No samples loaded. Exiting.")
        sys.exit(1)

    metrics = evaluate_nli(samples)
    if not metrics:
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("HALUEVAL BENCHMARK RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key:25s}: {value}")
    print("=" * 60)

    # Save results
    results_path = Path(__file__).resolve().parent / "results_halueval.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
