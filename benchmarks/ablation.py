"""
hallu-check | benchmarks/ablation.py
Automated Ablation Study Runner

Runs the pipeline with each component toggled on/off to measure
individual contribution. This is essential for the paper — reviewers
need to see which components actually help.

Ablation configs:
  - Full pipeline (all contributions enabled)
  - − ICR (one-shot refinement only)
  - − EGV (NLI for code/math)
  - − Surgical (one-shot edit instead of per-claim)
  - − Self-Consistency
  - − RLM
  - Raw Llama (no pipeline at all)

Usage:
    python -m benchmarks.ablation --dataset truthfulqa --samples 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("benchmark.ablation")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Ablation configurations
ABLATION_CONFIGS = [
    {
        "name": "Full Pipeline",
        "env": {
            "ENABLE_ICR": "true",
            "ENABLE_EGV": "true",
            "ENABLE_SURGICAL_CORRECTION": "true",
            "ENABLE_SELF_CONSISTENCY": "true",
            "ENABLE_RLM_REASONING": "true",
        },
    },
    {
        "name": "− ICR (one-shot refine)",
        "env": {
            "ENABLE_ICR": "false",
            "ENABLE_EGV": "true",
            "ENABLE_SURGICAL_CORRECTION": "true",
            "ENABLE_SELF_CONSISTENCY": "true",
            "ENABLE_RLM_REASONING": "true",
        },
    },
    {
        "name": "− EGV (NLI for code/math)",
        "env": {
            "ENABLE_ICR": "true",
            "ENABLE_EGV": "false",
            "ENABLE_SURGICAL_CORRECTION": "true",
            "ENABLE_SELF_CONSISTENCY": "true",
            "ENABLE_RLM_REASONING": "true",
        },
    },
    {
        "name": "− Surgical (one-shot edit)",
        "env": {
            "ENABLE_ICR": "true",
            "ENABLE_EGV": "true",
            "ENABLE_SURGICAL_CORRECTION": "false",
            "ENABLE_SELF_CONSISTENCY": "true",
            "ENABLE_RLM_REASONING": "true",
        },
    },
    {
        "name": "− Self-Consistency",
        "env": {
            "ENABLE_ICR": "true",
            "ENABLE_EGV": "true",
            "ENABLE_SURGICAL_CORRECTION": "true",
            "ENABLE_SELF_CONSISTENCY": "false",
            "ENABLE_RLM_REASONING": "true",
        },
    },
    {
        "name": "− RLM",
        "env": {
            "ENABLE_ICR": "true",
            "ENABLE_EGV": "true",
            "ENABLE_SURGICAL_CORRECTION": "true",
            "ENABLE_SELF_CONSISTENCY": "true",
            "ENABLE_RLM_REASONING": "false",
        },
    },
    {
        "name": "Raw Llama (no pipeline)",
        "env": {
            "ENABLE_ICR": "false",
            "ENABLE_EGV": "false",
            "ENABLE_SURGICAL_CORRECTION": "false",
            "ENABLE_SELF_CONSISTENCY": "false",
            "ENABLE_RLM_REASONING": "false",
        },
    },
]


def _set_env(env_dict: Dict[str, str]):
    """Set environment variables for an ablation config."""
    for key, value in env_dict.items():
        os.environ[key] = value


def _reload_config():
    """Reload config module to pick up new env vars."""
    import importlib
    import config  # type: ignore
    importlib.reload(config)


def run_ablation(
    dataset: str = "truthfulqa",
    max_samples: int = 50,
) -> List[Dict]:
    """
    Run all ablation configs against a benchmark dataset.

    Args:
        dataset: "truthfulqa" or "halueval"
        max_samples: Number of samples per config

    Returns:
        List of result dicts, one per ablation config.
    """
    results: List[Dict] = []

    for config_idx, ablation in enumerate(ABLATION_CONFIGS):
        config_name = ablation["name"]
        logger.info(
            "\n" + "=" * 70 +
            f"\n  ABLATION {config_idx + 1}/{len(ABLATION_CONFIGS)}: {config_name}" +
            "\n" + "=" * 70,
        )

        # Set environment variables
        _set_env(ablation["env"])
        _reload_config()

        t0 = time.time()

        try:
            if dataset == "truthfulqa":
                from benchmarks.eval_e2e import evaluate_truthfulqa_e2e  # type: ignore
                metrics = evaluate_truthfulqa_e2e(
                    max_samples=max_samples,
                    enable_icr=ablation["env"].get("ENABLE_ICR") == "true",
                )
            elif dataset == "halueval":
                from benchmarks.eval_e2e import evaluate_halueval_e2e  # type: ignore
                metrics = evaluate_halueval_e2e(
                    max_samples=max_samples,
                    enable_icr=ablation["env"].get("ENABLE_ICR") == "true",
                )
            else:
                logger.error("Unknown dataset: %s", dataset)
                continue

        except Exception as e:
            logger.error("Ablation '%s' failed: %s", config_name, e)
            metrics = {"error": str(e)}

        elapsed = time.time() - t0
        metrics["config_name"] = config_name
        metrics["config_env"] = ablation["env"]
        metrics["elapsed_seconds"] = round(elapsed, 1)

        results.append(metrics)

        logger.info(
            "  Config '%s' complete in %.1fs",
            config_name, elapsed,
        )

    return results


def _format_table(results: List[Dict], dataset: str):
    """Format results as a readable table."""
    print("\n" + "=" * 90)
    print(f"  ABLATION STUDY RESULTS — {dataset.upper()}")
    print("=" * 90)

    if dataset == "truthfulqa":
        header = f"{'Config':<30} {'Raw Truth%':>10} {'Refined%':>10} {'Δ':>8} {'Time(s)':>8}"
        print(header)
        print("-" * 66)
        for r in results:
            name = r.get("config_name", "?")
            raw = r.get("raw_truthfulness_rate", 0)
            refined = r.get("refined_truthfulness_rate", 0)
            delta = r.get("improvement", 0)
            elapsed = r.get("elapsed_seconds", 0)
            print(f"{name:<30} {raw:>9.1%} {refined:>9.1%} {delta:>+7.1%} {elapsed:>7.1f}")

    elif dataset == "halueval":
        header = f"{'Config':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Time(s)':>8}"
        print(header)
        print("-" * 70)
        for r in results:
            name = r.get("config_name", "?")
            prec = r.get("precision", 0)
            rec = r.get("recall", 0)
            f1 = r.get("f1", 0)
            acc = r.get("accuracy", 0)
            elapsed = r.get("elapsed_seconds", 0)
            print(f"{name:<30} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%} {acc:>7.1%} {elapsed:>7.1f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument(
        "--dataset",
        choices=["truthfulqa", "halueval"],
        default="truthfulqa",
    )
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    results = run_ablation(
        dataset=args.dataset,
        max_samples=args.samples,
    )

    _format_table(results, args.dataset)

    # Save results
    results_path = Path(__file__).resolve().parent / f"results_ablation_{args.dataset}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
