from __future__ import annotations

import logging
import re
from typing import List, Optional

from nodes.local_llm import chat_completion  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.surgical_corrector")


def _local_generate_short(prompt: str) -> str:
    """Generate a short response using the local LLM (Ollama/HF)."""
    messages = [
        {"role": "system", "content": "You are a concise fact corrector. Output ONLY the corrected text — no preamble, no explanation."},
        {"role": "user", "content": prompt},
    ]
    try:
        result = chat_completion(messages, max_tokens=512, temperature=0.2)
        return result.strip() if result else ""
    except Exception as e:
        logger.warning("Surgical | Local LLM call failed: %s", e)
        return ""


def _find_claim_in_text(claim: str, text: str) -> Optional[str]:
    # Try exact substring match first
    if claim in text:
        return claim

    # Fuzzy match: find the sentence with the highest word overlap
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    if not sentences:
        return None

    claim_words = set(claim.lower().split())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "and", "or", "but", "not", "it", "this", "that",
    }
    claim_content = claim_words - stop_words
    if not claim_content:
        return None

    best_match = None
    best_overlap = 0.0
    for sent in sentences:
        sent_words = set(sent.lower().split()) - stop_words
        if not sent_words:
            continue
        overlap = len(claim_content & sent_words) / len(claim_content)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = sent

    # Require at least 50% overlap to count as a match
    if best_overlap >= 0.5:
        return best_match
    return None


def _generate_replacement(
    claim: str,
    evidence: str,
    query: str,
) -> str:
    prompt = (
        "You are a surgical fact corrector. Replace the INCORRECT claim below "
        "with a CORRECTED version using ONLY the evidence provided.\n\n"
        "RULES:\n"
        "• Output ONLY the replacement sentence — no preamble, no explanation.\n"
        "• Use ONLY facts from the Evidence. Do NOT add your own knowledge.\n"
        "• Keep the same tone and style as the original claim.\n"
        "• If the evidence doesn't contain enough info to correct the claim, "
        "output: '[INSUFFICIENT EVIDENCE]'\n\n"
        f"Original query (context): {query}\n\n"
        f"INCORRECT claim: {claim}\n\n"
        f"Evidence: {evidence[:3000]}\n\n"
        "Corrected sentence:"
    )

    replacement = _local_generate_short(prompt)

    # Validate: reject if the replacement is empty, too long, or a refusal
    if not replacement:
        return ""
    if "[INSUFFICIENT EVIDENCE]" in replacement:
        return ""
    if len(replacement) > len(claim) * 3:
        logger.warning(
            "Surgical | Replacement too long (%d chars vs %d original), rejecting.",
            len(replacement), len(claim),
        )
        return ""

    return replacement


def surgical_correct(
    original_output: str,
    claim_verdicts: list,
    rag_output: str,
    query: str,
) -> str:
    logger.info(
        "Surgical | Starting claim-level correction (%d verdicts)…",
        len(claim_verdicts),
    )

    corrected = original_output
    corrections_made = 0
    corrections_failed = 0

    for verdict in claim_verdicts:
        # Extract fields — handle both dict and dataclass
        if isinstance(verdict, dict):
            v_verdict = verdict.get("verdict", "")
            v_claim = verdict.get("claim", "")
            v_evidence = verdict.get("evidence", "")
        else:
            v_verdict = getattr(verdict, "verdict", "")
            v_claim = getattr(verdict, "claim", "")
            v_evidence = getattr(verdict, "evidence", "")

        if v_verdict == "SUPPORTED" or v_verdict == "NO_CLAIM":
            continue  # Don't touch correct claims

        if v_verdict == "HONEST_UNCERTAINTY":
            continue  # Honest — not a hallucination

        # Find the claim's location in the text
        matched_text = _find_claim_in_text(v_claim, corrected)
        if not matched_text:
            logger.debug(
                "Surgical | Could not locate claim in text: '%s'",
                v_claim[:60],
            )
            corrections_failed += 1
            continue

        if v_verdict == "UNVERIFIABLE":
            # NLI often marks implicit contradictions as UNVERIFIABLE
            # (e.g., claim names person X for a role, RAG names person Y —
            # high P(neutral) because RAG doesn't say "X is NOT the role").
            # Try a RAG-grounded replacement first; _generate_replacement
            # returns "" when evidence genuinely lacks the answer, in which
            # case we fall back to removal.
            replacement = _generate_replacement(
                v_claim,
                v_evidence or rag_output[:3000],
                query,
            )
            if replacement:
                corrected = corrected.replace(matched_text, replacement, 1)
                corrections_made += 1
                logger.info(
                    "Surgical | Replaced UNVERIFIABLE claim with RAG-grounded fact:\n"
                    "  OLD: '%s'\n  NEW: '%s'",
                    v_claim[:80], replacement[:80],
                )
                continue

            # No supporting evidence — remove the claim
            corrected = corrected.replace(matched_text, "", 1)
            corrected = re.sub(r"\n\s*\n\s*\n", "\n\n", corrected)
            corrected = re.sub(r"  +", " ", corrected)
            corrections_made += 1
            logger.info(
                "Surgical | Removed UNVERIFIABLE claim (no RAG support): '%s'",
                v_claim[:60],
            )
            continue

        if v_verdict == "CONTRADICTED":
            # Generate a focused replacement
            replacement = _generate_replacement(
                v_claim,
                v_evidence or rag_output[:3000],
                query,
            )
            if replacement:
                corrected = corrected.replace(matched_text, replacement, 1)
                corrections_made += 1
                logger.info(
                    "Surgical | Replaced CONTRADICTED claim:\n"
                    "  OLD: '%s'\n  NEW: '%s'",
                    v_claim[:80], replacement[:80],
                )
            else:
                corrections_failed += 1
                logger.warning(
                    "Surgical | Failed to generate replacement for: '%s'",
                    v_claim[:60],
                )

    # Clean up the corrected text
    corrected = corrected.strip()
    if not corrected or len(corrected) < 10:
        logger.warning("Surgical | Correction produced empty result, falling back.")
        return original_output

    logger.info(
        "Surgical | Done: %d corrections made, %d failed.",
        corrections_made, corrections_failed,
    )
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Single-Claim Surgical Correction (standalone entry point)
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_CLAIM_PROMPT = """\
You are given an answer, one specific wrong claim within it, and evidence.
Replace ONLY that claim with a corrected version grounded in the evidence.
Do not rewrite, restructure, or change any other part of the answer.
Return the full answer with only that one claim changed.

Original answer:
{original_answer}

Wrong claim:
{wrong_claim}

Evidence:
{evidence}

Corrected answer:"""


def surgical_correct_single(
    original_answer: str,
    wrong_claim: str,
    evidence: str,
) -> str:
    if not original_answer or not wrong_claim:
        return original_answer or ""

    prompt = _SINGLE_CLAIM_PROMPT.format(
        original_answer=original_answer,
        wrong_claim=wrong_claim,
        evidence=evidence[:4000] if evidence else "(no evidence provided)",
    )

    corrected = _local_generate_short(prompt)

    # ── Validation ────────────────────────────────────────────────────
    if not corrected or len(corrected.strip()) < 10:
        logger.warning("Surgical | Single-claim correction returned empty, keeping original.")
        return original_answer

    # Strip markdown fences if the LLM wrapped the output
    if corrected.startswith("```") and corrected.endswith("```"):
        corrected = re.sub(r"^```\w*\n?", "", corrected)
        corrected = re.sub(r"\n?```$", "", corrected)
        corrected = corrected.strip()

    # Sanity check: the corrected answer should still contain most of
    # the original content (LLM shouldn't have rewritten everything)
    original_words = set(original_answer.lower().split())
    corrected_words = set(corrected.lower().split())
    if original_words:
        preservation = len(original_words & corrected_words) / len(original_words)
        if preservation < 0.5:
            logger.warning(
                "Surgical | LLM rewrote too much (%.0f%% words preserved), "
                "keeping original.",
                preservation * 100,
            )
            return original_answer

    # Sanity check: the wrong claim should NOT appear verbatim in the
    # corrected output (it should have been replaced)
    if wrong_claim in corrected:
        logger.warning("Surgical | Wrong claim still present after correction, retrying locally.")
        # Fall back to local replacement using _generate_replacement
        replacement = _generate_replacement(wrong_claim, evidence, "")
        if replacement:
            local_match = _find_claim_in_text(wrong_claim, original_answer)
            if local_match:
                return original_answer.replace(local_match, replacement, 1)

    logger.info(
        "Surgical | Single-claim correction succeeded (%.0f%% preserved).",
        preservation * 100 if original_words else 100,
    )
    return corrected
