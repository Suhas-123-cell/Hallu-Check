from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.gsm8k")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _extract_number(text: str) -> Optional[float]:
    # GSM8K format: "#### <number>"
    hash_match = re.search(r"####?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if hash_match:
        return float(hash_match.group(1).replace(",", ""))

    # "Answer: <number>" pattern
    answer_match = re.search(
        r"(?:answer|result|total)\s*(?:is|=|:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return float(answer_match.group(1).replace(",", ""))

    # Last number in the text (common LLM pattern)
    all_numbers = re.findall(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if all_numbers:
        return float(all_numbers[-1].replace(",", ""))

    return None


def evaluate_gsm8k(max_samples: int = 100) -> Dict:
    from datasets import load_dataset  # type: ignore
    from nodes.generator import generate_llm_output  # type: ignore

    logger.info("Loading GSM8K dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        logger.error("Failed to load GSM8K: %s", e)
        return {}

    samples = list(ds.select(range(min(max_samples, len(ds)))))
    logger.info("Loaded %d GSM8K problems.", len(samples))

    raw_correct = 0
    refined_correct = 0
    total = 0
    per_problem: List[Dict] = []

    t0 = time.time()

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answer_text = sample.get("answer", "")

        if not question or not answer_text:
            continue

        # Extract ground truth answer
        gt_answer = _extract_number(answer_text)
        if gt_answer is None:
            continue

        total += 1

        # Step 1: Generate with Llama
        query = (
            f"Solve this math problem step by step. "
            f"End with 'Answer: <number>'.\n\n{question}"
        )
        try:
            raw_output = generate_llm_output(query)
        except Exception as e:
            logger.warning("Problem %d: Generation failed: %s", i, e)
            per_problem.append({
                "question": question[:80],
                "gt_answer": gt_answer,
                "raw_correct": False,
                "refined_correct": False,
            })
            continue

        # Step 2: Extract and compare raw answer
        raw_answer = _extract_number(raw_output)
        raw_is_correct = (
            raw_answer is not None
            and abs(raw_answer - gt_answer) < 1e-6
        )
        if raw_is_correct:
            raw_correct += 1

        # Step 3: If wrong, try EGV math verification + correction
        refined_is_correct = raw_is_correct
        if not raw_is_correct:
            try:
                from nodes.execution_verifier import verify_math  # type: ignore
                verdict = verify_math(raw_output, query)

                if verdict.verdict == "FAIL":
                    # The execution shows the actual computed answer
                    # Use it to refine
                    from nodes.refiner import refine_response  # type: ignore
                    evidence = (
                        f"The math problem is:\n{question}\n\n"
                        f"The correct computation yields: {verdict.execution_output}\n"
                        f"The LLM claimed the answer was: {raw_answer}"
                    )
                    refined_output = refine_response(query, evidence)

                    if refined_output:
                        refined_answer = _extract_number(refined_output)
                        refined_is_correct = (
                            refined_answer is not None
                            and abs(refined_answer - gt_answer) < 1e-6
                        )
            except Exception as e:
                logger.debug("Problem %d: EGV/refinement failed: %s", i, e)

        if refined_is_correct:
            refined_correct += 1

        per_problem.append({
            "question": question[:80],
            "gt_answer": gt_answer,
            "raw_answer": raw_answer,
            "raw_correct": raw_is_correct,
            "refined_correct": refined_is_correct,
        })

        if (i + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d | Raw acc: %d/%d (%.1f%%) | Refined acc: %d/%d (%.1f%%)",
                i + 1, len(samples),
                raw_correct, total, raw_correct / total * 100,
                refined_correct, total, refined_correct / total * 100,
            )

    elapsed = time.time() - t0

    metrics = {
        "dataset": "GSM8K",
        "evaluation_type": "math_reasoning",
        "total_problems": total,
        "raw_correct": raw_correct,
        "raw_accuracy": round(raw_correct / total, 4) if total else 0,
        "refined_correct": refined_correct,
        "refined_accuracy": round(refined_correct / total, 4) if total else 0,
        "improvement": round((refined_correct - raw_correct) / total, 4) if total else 0,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_problem": round(elapsed / total, 1) if total else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GSM8K math benchmark")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    metrics = evaluate_gsm8k(max_samples=args.samples)
    if not metrics:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GSM8K BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")
    print("=" * 60)

    results_path = Path(__file__).resolve().parent / "results_gsm8k.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
