"""
hallu-check | benchmarks/eval_humaneval.py
Code Correctness Benchmark (HumanEval)

Evaluates whether the Hallu-Check pipeline (with EGV) improves code
correctness from Llama 3.2-1B on OpenAI's HumanEval dataset.

Methodology:
  1. Llama generates code for each HumanEval problem
  2. EGV (Execution-Grounded Verification) runs the code against tests
  3. If EGV detects failures, ICR refines the code
  4. Measure pass@1 before and after pipeline

Usage:
    python -m benchmarks.eval_humaneval [--samples 50]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.humaneval")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _extract_function(llm_output: str, entry_point: str) -> str:
    """Extract a Python function from LLM output."""
    # Try fenced code blocks first
    code_match = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", llm_output)
    if code_match:
        return code_match.group(1).strip()

    # Try finding the function definition
    func_match = re.search(
        rf"(def\s+{re.escape(entry_point)}\s*\([\s\S]*?)(?=\ndef\s|\Z)",
        llm_output,
    )
    if func_match:
        return func_match.group(1).strip()

    return llm_output.strip()


def evaluate_humaneval(max_samples: int = 50) -> Dict:
    """
    Run HumanEval benchmark with and without EGV.

    For each problem:
      1. Generate code with Llama
      2. Test against provided test cases
      3. If EGV detects failure, refine and re-test
    """
    from datasets import load_dataset  # type: ignore
    from nodes.generator import generate_llm_output  # type: ignore
    from nodes.tools.python_exec import run_python  # type: ignore

    logger.info("Loading HumanEval dataset...")
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        logger.error("Failed to load HumanEval: %s", e)
        return {}

    samples = list(ds.select(range(min(max_samples, len(ds)))))
    logger.info("Loaded %d HumanEval problems.", len(samples))

    raw_passed = 0
    refined_passed = 0
    total = 0
    per_problem: List[Dict] = []

    t0 = time.time()

    for i, sample in enumerate(samples):
        task_id = sample.get("task_id", f"task_{i}")
        prompt = sample.get("prompt", "")
        test = sample.get("test", "")
        entry_point = sample.get("entry_point", "")
        canonical = sample.get("canonical_solution", "")

        if not prompt or not test or not entry_point:
            continue

        total += 1

        # Step 1: Generate code with Llama
        query = f"Complete this Python function:\n\n{prompt}"
        try:
            raw_output = generate_llm_output(query)
        except Exception as e:
            logger.warning("Problem %s: Generation failed: %s", task_id, e)
            per_problem.append({"task_id": task_id, "raw_pass": False, "refined_pass": False})
            continue

        # Extract the function
        code = _extract_function(raw_output, entry_point)

        # Step 2: Test raw output
        raw_test_code = f"{prompt}\n{code}\n\n{test}\n\ncheck({entry_point})"
        raw_pass = False
        try:
            result = run_python(raw_test_code)
            raw_pass = not result.error
        except Exception:
            raw_pass = False

        if raw_pass:
            raw_passed += 1

        # Step 3: If failed, try EGV + refinement
        refined_pass = raw_pass
        if not raw_pass:
            try:
                from nodes.execution_verifier import verify_code  # type: ignore
                verdict = verify_code(raw_output, query)

                if verdict.verdict == "FAIL" and verdict.test_results:
                    # Build failure context for refinement
                    failures = "\n".join(
                        f"Test '{r.test_case.description}': expected {r.test_case.expected_output}, "
                        f"got {r.actual_output}"
                        for r in verdict.test_results if not r.passed
                    )

                    # Refine using Gemini with specific failure info
                    from nodes.refiner import refine_with_evidence  # type: ignore
                    evidence = (
                        f"The code has the following test failures:\n{failures}\n\n"
                        f"Original prompt:\n{prompt}"
                    )
                    refined_code = refine_with_evidence(
                        query=query,
                        rag_output=evidence,
                        claim_report={"claim_verdicts": [], "original_output": raw_output},
                        route="REASONING",
                    )

                    if refined_code:
                        refined_func = _extract_function(refined_code, entry_point)
                        refined_test_code = f"{prompt}\n{refined_func}\n\n{test}\n\ncheck({entry_point})"
                        try:
                            r2 = run_python(refined_test_code)
                            refined_pass = not r2.error
                        except Exception:
                            refined_pass = False

            except Exception as e:
                logger.debug("Problem %s: EGV/refinement failed: %s", task_id, e)

        if refined_pass:
            refined_passed += 1

        per_problem.append({
            "task_id": task_id,
            "raw_pass": raw_pass,
            "refined_pass": refined_pass,
        })

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d | Raw pass@1: %d | Refined pass@1: %d",
                i + 1, len(samples), raw_passed, refined_passed,
            )

    elapsed = time.time() - t0

    metrics = {
        "dataset": "HumanEval",
        "evaluation_type": "code_correctness",
        "total_problems": total,
        "raw_pass_at_1": raw_passed,
        "raw_pass_rate": round(raw_passed / total, 4) if total else 0,
        "refined_pass_at_1": refined_passed,
        "refined_pass_rate": round(refined_passed / total, 4) if total else 0,
        "improvement": round((refined_passed - raw_passed) / total, 4) if total else 0,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_problem": round(elapsed / total, 1) if total else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="HumanEval code benchmark")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    metrics = evaluate_humaneval(max_samples=args.samples)
    if not metrics:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("HUMANEVAL BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")
    print("=" * 60)

    results_path = Path(__file__).resolve().parent / "results_humaneval.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
