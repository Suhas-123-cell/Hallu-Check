"""
hallu-check | nodes/iterative_refiner.py
Contribution 1 — Iterative Convergent Refinement (ICR)

The single-pass verify→refine approach fails when the refiner itself
halluccinates (BPE→BERT failure). ICR runs multiple rounds:

  Round 0: Generate → Verify → score₀
  Round 1: Refine (surgical) → Re-verify → score₁
  Round 2: Refine (surgical) → Re-verify → score₂
  ...until convergence or max rounds.

Convergence criterion:
  |scoreₙ - scoreₙ₋₁| < ε (default ε = 0.05)

Divergence safeguard:
  if scoreₙ > scoreₙ₋₁ → rollback to best-scoring round

This is the paper's primary novel contribution:
  - Nobody has demonstrated NLI-based convergence detection
    for iterative hallucination correction
  - The divergence safeguard prevents the "flawed reasoning"
    amplification problem identified in Self-Refine literature
  - Specifically designed for small models where BOTH the
    generator AND refiner hallucinate
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (  # type: ignore[import-not-found]
    ICR_MAX_ROUNDS,
    ICR_CONVERGENCE_EPSILON,
    HALLUCINATION_THRESHOLD,
    ENABLE_SURGICAL_CORRECTION,
)

logger = logging.getLogger("hallu-check.iterative_refiner")


@dataclass
class RoundResult:
    """Telemetry for a single refinement round."""
    round_num: int
    hallucination_score: float
    hallucination_detected: bool
    claim_verdicts: list
    answer_text: str
    method: str  # "surgical" or "one-shot"
    elapsed_seconds: float = 0.0
    n_supported: int = 0
    n_contradicted: int = 0
    n_unverifiable: int = 0


@dataclass
class IterativeResult:
    """Full result of the iterative refinement process."""
    final_answer: str
    rounds: List[RoundResult] = field(default_factory=list)
    converged: bool = False
    diverged: bool = False
    best_round: int = 0
    total_rounds: int = 0
    initial_score: float = 0.0
    final_score: float = 0.0
    total_elapsed: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "final_answer": self.final_answer,
            "converged": self.converged,
            "diverged": self.diverged,
            "best_round": self.best_round,
            "total_rounds": self.total_rounds,
            "initial_score": round(self.initial_score, 4),
            "final_score": round(self.final_score, 4),
            "improvement": round(self.initial_score - self.final_score, 4),
            "total_elapsed": round(self.total_elapsed, 1),
            "scores_per_round": [
                round(r.hallucination_score, 4) for r in self.rounds
            ],
        }


def iterative_refine(
    query: str,
    llm_output: str,
    rag_output: str,
    initial_report: object,
    route: str = "FACTUAL",
    max_rounds: int = ICR_MAX_ROUNDS,
    epsilon: float = ICR_CONVERGENCE_EPSILON,
) -> IterativeResult:
    """
    Contribution 1 — Multi-round verify→refine loop with NLI convergence.

    Algorithm:
      1. Start with the initial HallucinationReport (from the first verify pass)
      2. If hallucination detected, apply surgical correction
      3. Re-verify the corrected text
      4. Check convergence: |Δscore| < ε → stop
      5. Check divergence: score increased → rollback to best round → stop
      6. Repeat until converged, diverged, or max_rounds reached

    Args:
        query:          The user's original question.
        llm_output:     The raw LLM answer (Round 0 input).
        rag_output:     The RAG-retrieved evidence context.
        initial_report: The HallucinationReport from the first verify_claims() call.
        route:          "FACTUAL" or "REASONING".
        max_rounds:     Maximum refinement iterations (default 3).
        epsilon:        Convergence threshold (default 0.05).

    Returns:
        IterativeResult with the best answer, round-by-round telemetry,
        and convergence/divergence status.
    """
    total_start = time.time()

    # ── Extract initial state from the first verification report ──────
    if isinstance(initial_report, dict):
        initial_score = initial_report.get("hallucination_score", 0.0)
        initial_detected = initial_report.get("hallucination_detected", False)
        initial_verdicts = initial_report.get("claim_verdicts", [])
    else:
        initial_score = getattr(initial_report, "hallucination_score", 0.0)
        initial_detected = getattr(initial_report, "hallucination_detected", False)
        initial_verdicts = getattr(initial_report, "claim_verdicts", [])

    # Convert verdicts to dicts if they're dataclass instances
    def _verdict_to_dict(v):
        if isinstance(v, dict):
            return v
        return {
            "claim": getattr(v, "claim", ""),
            "verdict": getattr(v, "verdict", ""),
            "evidence": getattr(v, "evidence", ""),
            "confidence": getattr(v, "confidence", 0.0),
            "reasoning": getattr(v, "reasoning", ""),
        }

    initial_verdicts_dicts = [_verdict_to_dict(v) for v in initial_verdicts]

    # ── Round 0: Initial state ───────────────────────────────────────
    round_0 = RoundResult(
        round_num=0,
        hallucination_score=initial_score,
        hallucination_detected=initial_detected,
        claim_verdicts=initial_verdicts_dicts,
        answer_text=llm_output,
        method="initial",
        n_supported=sum(1 for v in initial_verdicts_dicts if v.get("verdict") == "SUPPORTED"),
        n_contradicted=sum(1 for v in initial_verdicts_dicts if v.get("verdict") == "CONTRADICTED"),
        n_unverifiable=sum(1 for v in initial_verdicts_dicts if v.get("verdict") == "UNVERIFIABLE"),
    )

    rounds: List[RoundResult] = [round_0]
    best_round = 0
    best_score = initial_score
    best_answer = llm_output

    logger.info(
        "ICR | Starting iterative refinement: initial_score=%.4f, max_rounds=%d, ε=%.3f",
        initial_score, max_rounds, epsilon,
    )

    # If no hallucination detected, skip refinement entirely
    if not initial_detected or initial_score < HALLUCINATION_THRESHOLD:
        logger.info("ICR | No hallucination detected — skipping refinement.")
        return IterativeResult(
            final_answer=llm_output,
            rounds=rounds,
            converged=True,
            best_round=0,
            total_rounds=1,
            initial_score=initial_score,
            final_score=initial_score,
            total_elapsed=time.time() - total_start,
        )

    # ── Iterative refinement loop ────────────────────────────────────
    current_text = llm_output
    current_verdicts = initial_verdicts_dicts
    prev_score = initial_score

    for round_num in range(1, max_rounds + 1):
        round_start = time.time()
        logger.info("ICR | ── Round %d ──", round_num)

        # ── Step 1: Apply correction ─────────────────────────────────
        if ENABLE_SURGICAL_CORRECTION:
            try:
                from nodes.surgical_corrector import surgical_correct  # type: ignore[import-not-found]
                corrected = surgical_correct(
                    original_output=current_text,
                    claim_verdicts=current_verdicts,
                    rag_output=rag_output,
                    query=query,
                )
                method = "surgical"
            except Exception as e:
                logger.warning("ICR | Surgical correction failed (%s), trying one-shot.", e)
                corrected = _one_shot_refine(query, rag_output, current_verdicts, route)
                method = "one-shot"
        else:
            corrected = _one_shot_refine(query, rag_output, current_verdicts, route)
            method = "one-shot"

        if not corrected or corrected == current_text:
            logger.info("ICR | No changes in round %d — stopping.", round_num)
            break

        # ── Step 2: Re-verify the corrected text ─────────────────────
        try:
            from nodes.claim_verifier import verify_claims  # type: ignore[import-not-found]
            re_report = verify_claims(
                llm_output=corrected,
                rag_output=rag_output,
                query=query,
            )
            new_score = re_report.hallucination_score
            new_detected = re_report.hallucination_detected
            new_verdicts = [_verdict_to_dict(v) for v in re_report.claim_verdicts]
        except Exception as e:
            logger.warning("ICR | Re-verification failed in round %d: %s", round_num, e)
            break

        round_elapsed = time.time() - round_start

        round_result = RoundResult(
            round_num=round_num,
            hallucination_score=new_score,
            hallucination_detected=new_detected,
            claim_verdicts=new_verdicts,
            answer_text=corrected,
            method=method,
            elapsed_seconds=round_elapsed,
            n_supported=sum(1 for v in new_verdicts if v.get("verdict") == "SUPPORTED"),
            n_contradicted=sum(1 for v in new_verdicts if v.get("verdict") == "CONTRADICTED"),
            n_unverifiable=sum(1 for v in new_verdicts if v.get("verdict") == "UNVERIFIABLE"),
        )
        rounds.append(round_result)

        logger.info(
            "ICR | Round %d: score %.4f → %.4f (Δ=%.4f), method=%s, elapsed=%.1fs",
            round_num, prev_score, new_score, prev_score - new_score,
            method, round_elapsed,
        )

        # Track best round
        if new_score < best_score:
            best_score = new_score
            best_round = round_num
            best_answer = corrected

        # ── Step 3: Check convergence ────────────────────────────────
        delta = abs(new_score - prev_score)

        if delta < epsilon:
            logger.info(
                "ICR | Converged at round %d (|Δ|=%.4f < ε=%.3f).",
                round_num, delta, epsilon,
            )
            return IterativeResult(
                final_answer=best_answer,
                rounds=rounds,
                converged=True,
                best_round=best_round,
                total_rounds=len(rounds),
                initial_score=initial_score,
                final_score=best_score,
                total_elapsed=time.time() - total_start,
            )

        # ── Step 4: Check divergence ─────────────────────────────────
        if new_score > prev_score:
            logger.warning(
                "ICR | DIVERGENCE at round %d (score %.4f > %.4f). "
                "Rolling back to best round %d.",
                round_num, new_score, prev_score, best_round,
            )
            return IterativeResult(
                final_answer=best_answer,
                rounds=rounds,
                diverged=True,
                best_round=best_round,
                total_rounds=len(rounds),
                initial_score=initial_score,
                final_score=best_score,
                total_elapsed=time.time() - total_start,
            )

        # ── Step 5: If no longer hallucinating, stop early ───────────
        if not new_detected or new_score < HALLUCINATION_THRESHOLD:
            logger.info(
                "ICR | Hallucination resolved at round %d (score=%.4f < threshold=%.2f).",
                round_num, new_score, HALLUCINATION_THRESHOLD,
            )
            return IterativeResult(
                final_answer=best_answer,
                rounds=rounds,
                converged=True,
                best_round=best_round,
                total_rounds=len(rounds),
                initial_score=initial_score,
                final_score=best_score,
                total_elapsed=time.time() - total_start,
            )

        # Prepare for next round
        current_text = corrected
        current_verdicts = new_verdicts
        prev_score = new_score

    # ── Max rounds reached ───────────────────────────────────────────
    logger.info(
        "ICR | Max rounds reached. Best round=%d, score=%.4f→%.4f",
        best_round, initial_score, best_score,
    )

    return IterativeResult(
        final_answer=best_answer,
        rounds=rounds,
        converged=False,
        best_round=best_round,
        total_rounds=len(rounds),
        initial_score=initial_score,
        final_score=best_score,
        total_elapsed=time.time() - total_start,
    )


def _one_shot_refine(
    query: str,
    rag_output: str,
    claim_verdicts: list,
    route: str,
) -> str:
    """Fallback to the existing one-shot refiner if surgical correction is disabled."""
    try:
        from nodes.refiner import refine_with_evidence  # type: ignore[import-not-found]
        claim_report = {
            "claim_verdicts": claim_verdicts,
            "original_output": "",
        }
        return refine_with_evidence(
            query=query,
            rag_output=rag_output,
            claim_report=claim_report,
            route=route,
        )
    except Exception as e:
        logger.warning("ICR | One-shot refine failed: %s", e)
        return ""
