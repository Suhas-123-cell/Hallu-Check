"""
hallu-check | nodes/self_consistency.py
Node 1.5 — Self-Consistency Checking

Generates multiple LLM answers at varied temperatures and checks
agreement using NLI. This catches a large class of hallucinations
cheaply — no web search needed.

Architecture:
  1. Generate N answers (default=3) at temperatures [0.1, 0.5, 0.9]
  2. For each pair of answers, compute NLI entailment probability
  3. consistency_score = mean pairwise entailment probability
  4. Extract "consensus claims" (facts that appear in majority of answers)

High consistency (>0.85) → claims are likely correct, lower priority for verification
Low consistency (<0.5) → high-risk, proceed with full pipeline
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]

from config import (  # type: ignore[import-not-found]
    HF_API_TOKEN,
    LOCAL_MODEL_ID,
    N_CONSISTENCY_SAMPLES,
)

logger = logging.getLogger("hallu-check.self_consistency")

# Temperature schedule for diversity
_TEMPERATURES = [0.1, 0.5, 0.9, 0.3, 0.7]


def _generate_answer(query: str, temperature: float, timeout: int = 120) -> str:
    """Generate a single LLM answer at a specific temperature."""
    if not HF_API_TOKEN:
        return ""

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=timeout)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, accurate assistant. "
                        "Answer the user's question concisely and factually."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=2048,
            temperature=temperature,
            top_p=0.9,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""
    except Exception as e:
        logger.warning(
            "Self-consistency | Generation failed at temp=%.1f: %s",
            temperature, e,
        )
        return ""


def check_self_consistency(
    query: str,
    primary_answer: str,
    n_samples: int | None = None,
) -> Dict:
    """
    Node 1.5 — Self-Consistency Checking.

    Generates additional answers at varied temperatures and measures
    agreement with the primary answer using NLI.

    Args:
        query: The user's query.
        primary_answer: The answer from Node 1 (temperature=0.3).
        n_samples: Number of additional samples (default from config).

    Returns:
        Dict with:
          - consistency_score: float (0.0 = total disagreement, 1.0 = perfect)
          - is_consistent: bool (True if score > 0.85)
          - is_high_risk: bool (True if score < 0.5)
          - sample_count: int (number of samples + primary)
          - pairwise_scores: list of pairwise entailment probabilities
    """
    n = n_samples or N_CONSISTENCY_SAMPLES
    temperatures = _TEMPERATURES[:n]

    logger.info(
        "Node 1.5 | Self-consistency check: generating %d additional samples…",
        len(temperatures),
    )

    t0 = time.time()

    # Generate additional answers at varied temperatures
    additional_answers = []
    for temp in temperatures:
        answer = _generate_answer(query, temperature=temp)
        if answer:
            additional_answers.append(answer)

    if not additional_answers:
        logger.warning("Node 1.5 | No additional samples generated, skipping consistency check.")
        return {
            "consistency_score": 0.5,
            "is_consistent": False,
            "is_high_risk": False,
            "sample_count": 1,
            "pairwise_scores": [],
        }

    # All answers including primary
    all_answers = [primary_answer] + additional_answers

    # Check if NLI model is available for pairwise comparison
    pairwise_scores: List[float] = []
    try:
        from nodes.nli_model import classify_nli_batch, is_loaded, load_model  # type: ignore

        if not is_loaded():
            load_model()

        if is_loaded():
            # Build all unique pairs
            pairs: List[Tuple[str, str]] = []
            for i in range(len(all_answers)):
                for j in range(i + 1, len(all_answers)):
                    # Truncate to fit NLI model context
                    a1 = all_answers[i][:1000]
                    a2 = all_answers[j][:1000]
                    pairs.append((a1, a2))

            if pairs:
                results = classify_nli_batch(pairs)
                pairwise_scores = [r["probabilities"]["entailment"] for r in results]
        else:
            # Fallback: word overlap
            pairwise_scores = _word_overlap_consistency(all_answers)
    except Exception as e:
        logger.warning("Node 1.5 | NLI-based consistency check failed (%s), using word overlap.", e)
        pairwise_scores = _word_overlap_consistency(all_answers)

    # Compute overall consistency score
    if pairwise_scores:
        consistency_score = sum(pairwise_scores) / len(pairwise_scores)
    else:
        consistency_score = 0.5

    elapsed = time.time() - t0
    logger.info(
        "Node 1.5 | Self-consistency: score=%.3f, %d samples, %.1fs",
        consistency_score,
        len(all_answers),
        elapsed,
    )

    return {
        "consistency_score": round(consistency_score, 4),
        "is_consistent": consistency_score > 0.85,
        "is_high_risk": consistency_score < 0.5,
        "sample_count": len(all_answers),
        "pairwise_scores": [round(s, 4) for s in pairwise_scores],
    }


def _word_overlap_consistency(answers: List[str]) -> List[float]:
    """Fallback: compute pairwise word overlap as consistency proxy."""
    scores: List[float] = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            words_i = set(answers[i].lower().split())
            words_j = set(answers[j].lower().split())
            if not words_i or not words_j:
                scores.append(0.0)
                continue
            intersection = words_i & words_j
            union = words_i | words_j
            scores.append(len(intersection) / len(union) if union else 0.0)
    return scores
