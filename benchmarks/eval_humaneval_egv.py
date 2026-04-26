"""
hallu-check | benchmarks/eval_humaneval_egv.py
EGV Pipeline Ablation Benchmark on HumanEval (164 problems)

Compares the FULL pipeline (EGV + NLI + surgical correction) against the
NLI-ONLY baseline on code-generation tasks from OpenAI's HumanEval dataset.

For each of the 164 HumanEval problems:
  1. Llama 3.2-1B generates an answer
  2. Run the FULL pipeline (claim classification → EGV for code/math, NLI for factual)
  3. Run the BASELINE (NLI-only, no claim classification, no EGV)
  4. Measure per-problem:
     • catch_rate: Did the pipeline detect a real bug?  (TP / (TP + FN))
     • false_positive_rate: Did it flag correct code?   (FP / (FP + TN))
     • correction_induced_hallucination_rate: Did surgical correction
       introduce NEW bugs? (re-run EGV on the corrected output)

Outputs a pandas DataFrame and saves to CSV.

Usage:
    python -m benchmarks.eval_humaneval_egv [--samples 164]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

# ── Fix 2: Hard override — disable surgical correction at module level ───────
# Bypasses whatever config issue causes surgical correction to fire.
# Patches both config.py AND the by-value copies in claim_verifier/iterative_refiner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as _cfg_override  # type: ignore
import nodes.claim_verifier as _cv_override  # type: ignore
import nodes.iterative_refiner as _ir_override  # type: ignore
_cfg_override.ENABLE_SURGICAL_CORRECTION = False
_cv_override.ENABLE_SURGICAL_CORRECTION = False
_ir_override.ENABLE_SURGICAL_CORRECTION = False

logging.basicConfig(
    level=logging.WARNING,  # Keep quiet — tqdm handles progress
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.humaneval_egv")

# NOTE: project root already added to sys.path above (surgical correction override)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_function(llm_output: str, entry_point: str) -> str:
    """Extract a Python function from LLM output."""
    # Fenced code blocks
    code_match = re.search(r"```(?:python|py)?\s*\n([\s\S]*?)```", llm_output)
    if code_match:
        return code_match.group(1).strip()

    # Bare function definition
    func_match = re.search(
        rf"(def\s+{re.escape(entry_point)}\s*\([\s\S]*?)(?=\ndef\s|\Z)",
        llm_output,
    )
    if func_match:
        return func_match.group(1).strip()

    return llm_output.strip()


def _run_tests(prompt: str, code: str, test: str, entry_point: str) -> bool:
    """
    Run HumanEval test suite against generated code.
    Returns True if all tests pass, False otherwise.
    Uses subprocess.run() — never eval/exec.
    """
    full_script = f"{prompt}\n{code}\n\n{test}\n\ncheck({entry_point})\n"

    try:
        with tempfile.TemporaryDirectory(prefix="hallu_eval_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", full_script],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
                check=False,
            )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Llama output cache (disk-backed, persists across benchmark runs)
# ─────────────────────────────────────────────────────────────────────────────
_llama_cache: Optional[Dict[str, str]] = None
_llama_cache_path: Optional[str] = None


def enable_llama_cache(cache_file: str = "llama_cache.json") -> int:
    """Activate disk-backed Llama output caching."""
    global _llama_cache, _llama_cache_path
    _llama_cache_path = cache_file
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                _llama_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _llama_cache = {}
    else:
        _llama_cache = {}
    return len(_llama_cache or {})


def _save_llama_cache() -> None:
    if _llama_cache is not None and _llama_cache_path:
        try:
            with open(_llama_cache_path, "w") as f:
                json.dump(_llama_cache, f, indent=0)
        except OSError:
            pass


def cached_generate(query: str, generate_fn) -> str:
    """Generate LLM output with optional disk cache."""
    import hashlib
    if _llama_cache is not None:
        key = hashlib.md5(query.encode()).hexdigest()
        if key in _llama_cache:
            return _llama_cache[key]
        result = generate_fn(query)
        _llama_cache[key] = result
        _save_llama_cache()
        return result
    return generate_fn(query)


# ─────────────────────────────────────────────────────────────────────────────
# Config toggle helper
# ─────────────────────────────────────────────────────────────────────────────

def _set_pipeline_flags(enable_egv: bool, enable_correction: bool) -> None:
    """
    Toggle ENABLE_EGV and ENABLE_SURGICAL_CORRECTION at runtime.

    Must patch BOTH the config module AND the claim_verifier module,
    because ``from config import X`` captures by value at import time.
    """
    import config as _cfg  # type: ignore
    import nodes.claim_verifier as _cv  # type: ignore

    _cfg.ENABLE_EGV = enable_egv
    _cfg.ENABLE_SURGICAL_CORRECTION = enable_correction

    # Patch the by-value imports inside claim_verifier
    _cv.ENABLE_EGV = enable_egv
    _cv.ENABLE_SURGICAL_CORRECTION = enable_correction


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_full_pipeline(
    llm_output: str,
    query: str,
    rag_output: str = "",
) -> Dict[str, Any]:
    """
    Run the FULL pipeline: claim extraction → classify → EGV/NLI.
    Surgical correction is controlled by _BENCH_ENABLE_CORRECTION env var.
    """
    from nodes.claim_verifier import verify_claims  # type: ignore

    enable_correction = (
        os.environ.get("_BENCH_ENABLE_CORRECTION", "false").lower()
        in ("true", "1", "yes")
    )
    _set_pipeline_flags(enable_egv=True, enable_correction=enable_correction)

    try:
        report = verify_claims(
            llm_output=llm_output,
            rag_output=rag_output or "No external context for code problems.",
            query=query,
        )
    except Exception as e:
        logger.warning("Full pipeline failed: %s", e)
        return {
            "hallucination_detected": False,
            "hallucination_score": 0.0,
            "verdicts": [],
            "corrected_output": llm_output,
            "verification_methods": set(),
        }

    methods = {v.verification_method for v in report.claim_verdicts}

    return {
        "hallucination_detected": report.hallucination_detected,
        "hallucination_score": report.hallucination_score,
        "verdicts": [
            {
                "claim": v.claim[:100],
                "verdict": v.verdict,
                "verification_method": v.verification_method,
                "confidence": v.confidence,
            }
            for v in report.claim_verdicts
        ],
        "corrected_output": llm_output,
        "verification_methods": methods,
    }


def _run_nli_baseline(
    llm_output: str,
    query: str,
    rag_output: str = "",
) -> Dict[str, Any]:
    """
    Run NLI-ONLY baseline: claim extraction → NLI for ALL claims (no EGV, no classification).

    Returns same schema as _run_full_pipeline.
    """
    from nodes.claim_verifier import verify_claims  # type: ignore

    _set_pipeline_flags(enable_egv=False, enable_correction=False)

    try:
        report = verify_claims(
            llm_output=llm_output,
            rag_output=rag_output or "No external context for code problems.",
            query=query,
        )
    except Exception as e:
        logger.warning("NLI baseline failed: %s", e)
        return {
            "hallucination_detected": False,
            "hallucination_score": 0.0,
            "verdicts": [],
            "corrected_output": llm_output,
            "verification_methods": {"NLI"},
        }

    return {
        "hallucination_detected": report.hallucination_detected,
        "hallucination_score": report.hallucination_score,
        "verdicts": [
            {
                "claim": v.claim[:100],
                "verdict": v.verdict,
                "verification_method": v.verification_method,
                "confidence": v.confidence,
            }
            for v in report.claim_verdicts
        ],
        "corrected_output": llm_output,
        "verification_methods": {"NLI"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Correction-induced hallucination check
# ─────────────────────────────────────────────────────────────────────────────

def _check_correction_hallucination(
    original_output: str,
    corrected_output: str,
    prompt: str,
    test: str,
    entry_point: str,
) -> bool:
    """
    Check if surgical correction introduced NEW hallucinations (bugs).

    Logic:
      - Extract code from the corrected output
      - Run HumanEval tests against it
      - If original passed but corrected FAILS → correction introduced a bug
      - If original failed and corrected also fails → no new hallucination
        (it was already wrong)

    Returns True if the correction INTRODUCED a new bug (broke something that was working).
    """
    original_code = _extract_function(original_output, entry_point)
    corrected_code = _extract_function(corrected_output, entry_point)

    # If code didn't change, no correction-induced hallucination possible
    if original_code == corrected_code:
        return False

    original_passes = _run_tests(prompt, original_code, test, entry_point)
    corrected_passes = _run_tests(prompt, corrected_code, test, entry_point)

    # Correction introduced a bug: original passed, corrected fails
    return original_passes and not corrected_passes


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_humaneval_egv(
    max_samples: int = 50,  # Fix 3: Validate with 50 first, full run after
    enable_cache: bool = True,
    enable_correction: bool = False,
) -> pd.DataFrame:
    """
    Run the full EGV ablation benchmark on HumanEval.

    For each of 164 problems:
      1. Generate answer with Llama 3.2
      2. Run full pipeline (EGV + NLI)
      3. Run baseline (NLI-only)
      4. Measure catch_rate, false_positive_rate, correction_induced_hallucination_rate

    Args:
        max_samples:       Number of problems to evaluate (max 164).
        enable_cache:      If True, cache Gemini responses to disk (saves ~30-40% tokens).
        enable_correction: If True, run surgical correction (expensive, off by default).

    Returns a pandas DataFrame with per-problem results.
    """
    from datasets import load_dataset  # type: ignore
    from nodes.generator import generate_llm_output  # type: ignore

    # ── Optimization 1: Enable Gemini response cache ──────────────────
    if enable_cache:
        from nodes.claim_verifier import enable_gemini_cache  # type: ignore
        cache_dir = Path(__file__).resolve().parent

        gemini_cache_path = str(cache_dir / "gemini_cache.json")
        n_gemini = enable_gemini_cache(gemini_cache_path)
        print(f"Gemini cache: {n_gemini} entries loaded from {gemini_cache_path}")

        llama_cache_path = str(cache_dir / "llama_cache.json")
        n_llama = enable_llama_cache(llama_cache_path)
        print(f"Llama cache:  {n_llama} entries loaded from {llama_cache_path}")

    # ── Optimization 3: Skip surgical correction during benchmarking ──
    if enable_correction:
        os.environ["_BENCH_ENABLE_CORRECTION"] = "true"
        print("Surgical correction: ENABLED (slower, measures correction quality)")
    else:
        os.environ["_BENCH_ENABLE_CORRECTION"] = "false"
        print("Surgical correction: DISABLED (faster, verdicts-only mode)")

    print("\nLoading HumanEval dataset...")
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")

    samples = list(ds.select(range(min(max_samples, len(ds)))))
    print(f"Loaded {len(samples)} HumanEval problems.\n")

    rows: List[Dict[str, Any]] = []

    for i, sample in enumerate(tqdm(samples, desc="HumanEval EGV Benchmark", unit="problem")):
        sample = cast(Dict[str, Any], sample)
        # Fix 4: 3-second delay between problems to prevent burst rate limiting
        if i > 0:
            time.sleep(3)
        task_id = sample.get("task_id", "")
        prompt = sample.get("prompt", "")
        test = sample.get("test", "")
        entry_point = sample.get("entry_point", "")
        canonical = sample.get("canonical_solution", "")

        if not prompt or not test or not entry_point:
            continue

        # ── Step 1: Generate code with Llama (cached) ─────────────────
        query = f"Complete this Python function:\n\n{prompt}"
        try:
            llm_output = cached_generate(query, generate_llm_output)
        except Exception as e:
            logger.warning("Problem %s: Generation failed: %s", task_id, e)
            rows.append({
                "task_id": task_id,
                "generation_ok": False,
                "ground_truth_passes": False,
                "raw_code_passes": False,
                "full_detected": False,
                "baseline_detected": False,
                "full_score": 0.0,
                "baseline_score": 0.0,
                "full_methods": "",
                "correction_introduced_bug": False,
                "full_n_claims": 0,
                "full_n_contradicted": 0,
                "baseline_n_contradicted": 0,
            })
            continue

        # ── Step 2: Test the generated code against ground truth ──────
        raw_code = _extract_function(llm_output, entry_point)
        raw_passes = _run_tests(prompt, raw_code, test, entry_point)

        # ── Step 3: Run FULL pipeline (EGV + NLI) ─────────────────────
        full_result = _run_full_pipeline(llm_output, query)

        # ── Step 4: Run BASELINE (NLI-only) ───────────────────────────
        baseline_result = _run_nli_baseline(llm_output, query)

        # ── Step 5: Check correction-induced hallucination ────────────
        correction_bug = False
        if full_result["corrected_output"] != llm_output:
            correction_bug = _check_correction_hallucination(
                llm_output,
                full_result["corrected_output"],
                prompt, test, entry_point,
            )

        # ── Collect per-problem metrics ───────────────────────────────
        full_contradicted = sum(
            1 for v in full_result["verdicts"] if v["verdict"] == "CONTRADICTED"
        )
        full_unverifiable = sum(
            1 for v in full_result["verdicts"] if v["verdict"] == "UNVERIFIABLE"
        )
        baseline_contradicted = sum(
            1 for v in baseline_result["verdicts"] if v["verdict"] == "CONTRADICTED"
        )
        baseline_unverifiable = sum(
            1 for v in baseline_result["verdicts"] if v["verdict"] == "UNVERIFIABLE"
        )

        # Fix 2: If ALL verdicts are UNVERIFIABLE, pipeline had no real signal.
        # Don't count these as detections — they inflate catch rates.
        full_has_signal = full_contradicted > 0 or any(
            v["verdict"] == "SUPPORTED" for v in full_result["verdicts"]
        )
        baseline_has_signal = baseline_contradicted > 0 or any(
            v["verdict"] == "SUPPORTED" for v in baseline_result["verdicts"]
        )

        # Only count as "detected" if the pipeline actually had real verdicts
        full_detected_clean = full_result["hallucination_detected"] and full_has_signal
        baseline_detected_clean = baseline_result["hallucination_detected"] and baseline_has_signal

        rows.append({
            "task_id": task_id,
            "generation_ok": True,
            "ground_truth_passes": raw_passes,
            "raw_code_passes": raw_passes,
            # Full pipeline (EGV + NLI)
            "full_detected": full_detected_clean,
            "full_score": full_result["hallucination_score"],
            "full_methods": ",".join(sorted(full_result["verification_methods"])),
            "full_n_claims": len(full_result["verdicts"]),
            "full_n_contradicted": full_contradicted,
            "full_n_unverifiable": full_unverifiable,
            # Baseline (NLI-only)
            "baseline_detected": baseline_detected_clean,
            "baseline_score": baseline_result["hallucination_score"],
            "baseline_n_contradicted": baseline_contradicted,
            "baseline_n_unverifiable": baseline_unverifiable,
            # Correction quality
            "correction_introduced_bug": correction_bug,
        })

    df = pd.DataFrame(rows)
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate metrics from the per-problem DataFrame.

    Definitions:
      - TP (True Positive):  Code has real bug AND pipeline detected it
      - FP (False Positive): Code is correct AND pipeline flagged it
      - FN (False Negative): Code has real bug AND pipeline missed it
      - TN (True Negative):  Code is correct AND pipeline did NOT flag it

    ground_truth: raw_code_passes (True = code is correct, False = code has bugs)
    pipeline_flag: full_detected (True = pipeline flagged hallucination)
    """
    valid = df[df["generation_ok"]].copy()

    if valid.empty:
        return {"error": "No valid samples"}

    # Ground truth: code has a real bug (does NOT pass tests)
    has_bug = ~valid["raw_code_passes"]
    is_correct = valid["raw_code_passes"]

    # ── Full pipeline metrics ─────────────────────────────────────────
    full_tp = ((has_bug) & (valid["full_detected"])).sum()
    full_fp = ((is_correct) & (valid["full_detected"])).sum()
    full_fn = ((has_bug) & (~valid["full_detected"])).sum()
    full_tn = ((is_correct) & (~valid["full_detected"])).sum()

    full_catch_rate = full_tp / (full_tp + full_fn) if (full_tp + full_fn) > 0 else 0.0
    full_fpr = full_fp / (full_fp + full_tn) if (full_fp + full_tn) > 0 else 0.0

    # ── Baseline (NLI-only) metrics ───────────────────────────────────
    base_tp = ((has_bug) & (valid["baseline_detected"])).sum()
    base_fp = ((is_correct) & (valid["baseline_detected"])).sum()
    base_fn = ((has_bug) & (~valid["baseline_detected"])).sum()
    base_tn = ((is_correct) & (~valid["baseline_detected"])).sum()

    base_catch_rate = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
    base_fpr = base_fp / (base_fp + base_tn) if (base_fp + base_tn) > 0 else 0.0

    # ── Correction-induced hallucination rate ─────────────────────────
    n_corrected = (valid["full_detected"]).sum()
    n_correction_bugs = valid["correction_introduced_bug"].sum()
    correction_hallucination_rate = (
        n_correction_bugs / n_corrected if n_corrected > 0 else 0.0
    )

    # ── Summary ───────────────────────────────────────────────────────
    total = len(valid)
    n_buggy = has_bug.sum()
    n_correct = is_correct.sum()

    return {
        "total_problems": int(total),
        "n_buggy_code": int(n_buggy),
        "n_correct_code": int(n_correct),
        # Full pipeline (EGV + NLI)
        "full_TP": int(full_tp),
        "full_FP": int(full_fp),
        "full_FN": int(full_fn),
        "full_TN": int(full_tn),
        "full_catch_rate": round(full_catch_rate, 4),
        "full_false_positive_rate": round(full_fpr, 4),
        # Baseline (NLI-only)
        "baseline_TP": int(base_tp),
        "baseline_FP": int(base_fp),
        "baseline_FN": int(base_fn),
        "baseline_TN": int(base_tn),
        "baseline_catch_rate": round(base_catch_rate, 4),
        "baseline_false_positive_rate": round(base_fpr, 4),
        # Improvement
        "catch_rate_improvement": round(full_catch_rate - base_catch_rate, 4),
        "fpr_improvement": round(base_fpr - full_fpr, 4),
        # Correction quality
        "n_corrections_attempted": int(n_corrected),
        "n_correction_induced_bugs": int(n_correction_bugs),
        "correction_induced_hallucination_rate": round(correction_hallucination_rate, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HumanEval EGV ablation benchmark (EGV+NLI vs NLI-only)"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of HumanEval problems to evaluate (max 164)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Gemini response caching (slower, costs more)",
    )
    parser.add_argument(
        "--with-correction", action="store_true",
        help="Enable surgical correction (expensive but measures correction quality)",
    )
    args = parser.parse_args()

    t0 = time.time()
    df = evaluate_humaneval_egv(
        max_samples=args.samples,
        enable_cache=not args.no_cache,
        enable_correction=args.with_correction,
    )
    elapsed = time.time() - t0

    # Print cache stats
    if not args.no_cache:
        try:
            import nodes.claim_verifier as _cv
            if _cv._gemini_cache is not None:
                print(f"\n  Gemini cache: {len(_cv._gemini_cache)} total entries")
        except (ImportError, AttributeError):
            pass
        if _llama_cache is not None:
            print(f"  Llama cache:  {len(_llama_cache)} total entries")

    if df.empty:
        print("No results. Check errors above.")
        sys.exit(1)

    # ── Compute aggregate metrics ─────────────────────────────────────
    metrics = compute_metrics(df)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["seconds_per_problem"] = round(elapsed / len(df), 1) if len(df) > 0 else 0

    # ── Print results table ───────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  HUMANEVAL EGV ABLATION BENCHMARK RESULTS")
    print("═" * 70)
    print(f"\n  {'Metric':<45s} {'Value':>10s}")
    print(f"  {'─' * 45} {'─' * 10}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<45s} {v:>10.4f}")
        else:
            print(f"  {k:<45s} {v:>10}")

    print(f"\n  {'─' * 56}")
    print(f"  Full pipeline catch rate:     {metrics['full_catch_rate']:.1%}")
    print(f"  Baseline catch rate:          {metrics['baseline_catch_rate']:.1%}")
    print(f"  Catch rate improvement:       {metrics['catch_rate_improvement']:+.1%}")
    print(f"  Full FPR:                     {metrics['full_false_positive_rate']:.1%}")
    print(f"  Baseline FPR:                 {metrics['baseline_false_positive_rate']:.1%}")
    print(f"  Correction hallucination:     {metrics['correction_induced_hallucination_rate']:.1%}")
    print("═" * 70)

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent
    csv_path = results_dir / "results_humaneval_egv.csv"
    json_path = results_dir / "results_humaneval_egv.json"

    df.to_csv(csv_path, index=False)
    print(f"\n  Per-problem results saved to: {csv_path}")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Aggregate metrics saved to:   {json_path}")

    # ── Show sample of the DataFrame ──────────────────────────────────
    print(f"\n  DataFrame shape: {df.shape}")
    print(f"\n{df.head(10).to_string(index=False)}\n")
    


if __name__ == "__main__":
    main()
