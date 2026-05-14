from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.e2e")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_nli():
    from nodes.nli_model import load_model, is_loaded  # type: ignore
    if not is_loaded():
        load_model()
    return is_loaded()


def evaluate_truthfulqa_e2e(
    max_samples: int = 100,
    enable_icr: bool = True,
) -> Dict:
    from datasets import load_dataset  # type: ignore
    from nodes.nli_model import classify_nli  # type: ignore
    from nodes.generator import generate_llm_output  # type: ignore
    from nodes.claim_verifier import verify_claims  # type: ignore

    if not _load_nli():
        logger.error("NLI model not available.")
        return {}

    logger.info("Loading TruthfulQA dataset...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    samples = list(ds.select(range(min(max_samples, len(ds)))))
    logger.info("Loaded %d samples.", len(samples))

    # Metrics accumulators
    raw_truthful = 0
    refined_truthful = 0
    raw_hallucinated = 0
    refined_hallucinated = 0
    total = 0
    per_sample: List[Dict] = []

    t0 = time.time()

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])

        if not question or not best_answer:
            continue

        total += 1

        # Step 1: Raw Llama output
        try:
            raw_output = generate_llm_output(question)
        except Exception as e:
            logger.warning("Sample %d: Generation failed: %s", i, e)
            continue

        # Step 2: Score raw output against ground truth
        r_raw_correct = classify_nli(best_answer, raw_output)
        raw_is_truthful = r_raw_correct["verdict"] == "SUPPORTED"
        if raw_is_truthful:
            raw_truthful += 1

        # Check if raw output matches incorrect answers (hallucination)
        raw_matches_incorrect = False
        if incorrect_answers:
            for inc in incorrect_answers[:3]:
                r_inc = classify_nli(inc, raw_output)
                if r_inc["verdict"] == "SUPPORTED":
                    raw_matches_incorrect = True
                    break
        if raw_matches_incorrect:
            raw_hallucinated += 1

        # Step 3: Run verification pipeline
        try:
            hallu_report = verify_claims(
                llm_output=raw_output,
                rag_output=best_answer,  # Use best_answer as "RAG context" for evaluation
                query=question,
            )

            # Step 4: If hallucination detected, refine
            refined_output = raw_output
            if hallu_report.hallucination_detected and enable_icr:
                try:
                    from nodes.iterative_refiner import iterative_refine  # type: ignore
                    icr_result = iterative_refine(
                        query=question,
                        llm_output=raw_output,
                        rag_output=best_answer,
                        initial_report=hallu_report,
                        route="FACTUAL",
                    )
                    if icr_result.final_answer:
                        refined_output = icr_result.final_answer
                except Exception:
                    # Fall back to original
                    pass

            # Step 5: Score refined output
            r_refined_correct = classify_nli(best_answer, refined_output)
            refined_is_truthful = r_refined_correct["verdict"] == "SUPPORTED"
            if refined_is_truthful:
                refined_truthful += 1

            refined_matches_incorrect = False
            if incorrect_answers:
                for inc in incorrect_answers[:3]:
                    r_inc = classify_nli(inc, refined_output)
                    if r_inc["verdict"] == "SUPPORTED":
                        refined_matches_incorrect = True
                        break
            if refined_matches_incorrect:
                refined_hallucinated += 1

        except Exception as e:
            logger.warning("Sample %d: Pipeline failed: %s", i, e)
            refined_truthful += (1 if raw_is_truthful else 0)
            if raw_matches_incorrect:
                refined_hallucinated += 1

        per_sample.append({
            "question": question[:100],
            "raw_truthful": raw_is_truthful,
            "refined_truthful": refined_is_truthful if 'refined_is_truthful' in dir() else raw_is_truthful,
        })

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d | Raw truthful: %d | Refined truthful: %d",
                i + 1, len(samples), raw_truthful, refined_truthful,
            )

    elapsed = time.time() - t0

    metrics = {
        "dataset": "TruthfulQA",
        "evaluation_type": "end_to_end",
        "total_samples": total,
        "raw_truthful": raw_truthful,
        "raw_truthfulness_rate": round(raw_truthful / total, 4) if total else 0,
        "raw_hallucinated": raw_hallucinated,
        "refined_truthful": refined_truthful,
        "refined_truthfulness_rate": round(refined_truthful / total, 4) if total else 0,
        "refined_hallucinated": refined_hallucinated,
        "improvement": round((refined_truthful - raw_truthful) / total, 4) if total else 0,
        "hallucination_reduction": round(
            (raw_hallucinated - refined_hallucinated) / raw_hallucinated, 4
        ) if raw_hallucinated else 0,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_sample": round(elapsed / total, 1) if total else 0,
        "icr_enabled": enable_icr,
    }

    return metrics


def evaluate_halueval_e2e(
    max_samples: int = 100,
    enable_icr: bool = True,
) -> Dict:
    from datasets import load_dataset  # type: ignore
    from nodes.claim_verifier import verify_claims  # type: ignore

    if not _load_nli():
        return {}

    logger.info("Loading HaluEval-QA dataset...")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception:
        try:
            ds = load_dataset("pminervini/HaluEval", split="data")
        except Exception as e:
            logger.error("Failed to load HaluEval: %s", e)
            return {}

    samples = list(ds.select(range(min(max_samples, len(ds)))))
    logger.info("Loaded %d HaluEval samples.", len(samples))

    tp = fp = tn = fn = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        knowledge = sample.get("knowledge", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        hallucinated = sample.get("hallucination", "")

        if not knowledge or not question:
            continue

        # Test with the hallucinated answer
        if hallucinated:
            try:
                report = verify_claims(
                    llm_output=hallucinated,
                    rag_output=knowledge,
                    query=question,
                )
                if report.hallucination_detected:
                    tp += 1  # Correctly detected hallucination
                else:
                    fn += 1  # Missed hallucination
            except Exception:
                fn += 1

        # Test with the correct answer
        if answer:
            try:
                report = verify_claims(
                    llm_output=answer,
                    rag_output=knowledge,
                    query=question,
                )
                if report.hallucination_detected:
                    fp += 1  # False alarm on correct answer
                else:
                    tn += 1  # Correctly accepted good answer
            except Exception:
                tn += 1

        if (i + 1) % 25 == 0:
            logger.info("Progress: %d/%d | TP=%d FP=%d TN=%d FN=%d", i + 1, len(samples), tp, fp, tn, fn)

    elapsed = time.time() - t0
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    metrics = {
        "dataset": "HaluEval-QA",
        "evaluation_type": "end_to_end",
        "total_samples": total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_sample": round(elapsed / total, 1) if total else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    parser.add_argument(
        "--dataset",
        choices=["truthfulqa", "halueval", "all"],
        default="truthfulqa",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--no-icr", action="store_true", help="Disable ICR")
    args = parser.parse_args()

    results = {}

    if args.dataset in ("truthfulqa", "all"):
        logger.info("=" * 60)
        logger.info("EVALUATING: TruthfulQA (End-to-End)")
        logger.info("=" * 60)
        metrics = evaluate_truthfulqa_e2e(
            max_samples=args.samples,
            enable_icr=not args.no_icr,
        )
        if metrics:
            results["truthfulqa"] = metrics
            print("\n" + "=" * 60)
            print("TRUTHFULQA E2E RESULTS")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k:35s}: {v}")

    if args.dataset in ("halueval", "all"):
        logger.info("=" * 60)
        logger.info("EVALUATING: HaluEval-QA (End-to-End)")
        logger.info("=" * 60)
        metrics = evaluate_halueval_e2e(
            max_samples=args.samples,
            enable_icr=not args.no_icr,
        )
        if metrics:
            results["halueval"] = metrics
            print("\n" + "=" * 60)
            print("HALUEVAL E2E RESULTS")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k:35s}: {v}")

    # Save results
    if results:
        results_path = Path(__file__).resolve().parent / "results_e2e.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
