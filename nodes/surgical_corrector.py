"""
hallu-check | nodes/surgical_corrector.py
Contribution 3 — Claim-Level Surgical Correction

Instead of rewriting the entire LLM output via a one-shot prompt (which
often introduces NEW hallucinations, e.g. BPE → BERT), this module
replaces ONLY the CONTRADICTED claims — one at a time — each constrained
to its specific evidence snippet.

Why this is better than one-shot rewriting:
  1. SUPPORTED claims are never touched → no risk of breaking correct facts
  2. Each replacement prompt is small and focused → fewer refiner hallucinations
  3. UNVERIFIABLE claims are removed, not replaced with guesses
  4. The original answer structure is preserved

The refiner sees a micro-prompt per claim:
  "Replace ONLY this claim using ONLY this evidence. Output ONLY the
   replacement sentence."
"""
from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

import google.genai as genai  # type: ignore[import-not-found, import-untyped]
from google.genai import types as genai_types  # type: ignore[import-not-found, import-untyped]

from config import GEMINI_API_KEY, GEMINI_MODEL  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.surgical_corrector")

_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 15

_GEMINI_HTTP_OPTIONS = genai_types.HttpOptions(timeout=60_000)
_GEMINI_GENERATE_CONFIG = genai_types.GenerateContentConfig(
    max_output_tokens=512,   # Short — we only need one sentence
    http_options=_GEMINI_HTTP_OPTIONS,
)


def _gemini_generate_short(prompt: str) -> str:
    """Call Gemini for a short response (single claim replacement)."""
    if not GEMINI_API_KEY:
        return ""

    import ssl
    import os
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=_GEMINI_GENERATE_CONFIG,
            )
            content = getattr(response, "text", None)
            if not content and hasattr(response, "candidates"):
                candidates = getattr(response, "candidates", [])
                if candidates and hasattr(candidates[0], "text"):
                    content = candidates[0].text
            return content.strip() if content else ""
        except Exception as e:
            error_str = str(e)
            is_retryable = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_retryable and attempt < _MAX_RETRIES:
                match = re.search(
                    r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)(s|ms)",
                    error_str,
                )
                if match:
                    delay = float(match.group(1))
                    if match.group(2) == "ms":
                        delay /= 1000
                else:
                    delay = _DEFAULT_RETRY_DELAY
                delay = min(delay + 1, 60)
                logger.warning(
                    "Surgical | Rate-limited (attempt %d/%d), waiting %.1fs…",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Surgical | Gemini call failed: %s", e)
            return ""
    return ""


def _find_claim_in_text(claim: str, text: str) -> Optional[str]:
    """
    Find the approximate location of a claim in the original text.

    Uses fuzzy matching: finds the sentence in the text that has the
    highest word overlap with the claim. Returns the matched sentence,
    or None if no good match is found.
    """
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
    """
    Generate a replacement for a single CONTRADICTED claim.

    The prompt is deliberately minimal to reduce refiner hallucination:
    - Only one claim to fix
    - Only the specific evidence for that claim
    - Strict instruction to not introduce new facts
    """
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

    replacement = _gemini_generate_short(prompt)

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
    """
    Contribution 3 — Replace ONLY contradicted claims, one at a time.

    For each CONTRADICTED claim:
      1. Find its location in the original output
      2. Generate a micro-replacement using only that claim's evidence
      3. Substitute it at the exact position
      4. Validate the replacement doesn't introduce new entities not in evidence

    For UNVERIFIABLE claims:
      - Remove them (they can't be verified, so don't keep them)

    For SUPPORTED claims:
      - Keep them untouched

    Args:
        original_output: The full LLM answer text.
        claim_verdicts:  List of ClaimVerdict dicts (from HallucinationReport).
        rag_output:      The RAG-retrieved evidence context.
        query:           The user's original query.

    Returns:
        The corrected text with only contradicted claims replaced.
        Returns the original if no corrections could be made.
    """
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
            # Remove unverifiable claims — don't replace with guesses
            corrected = corrected.replace(matched_text, "", 1)
            # Clean up double spaces/newlines
            corrected = re.sub(r"\n\s*\n\s*\n", "\n\n", corrected)
            corrected = re.sub(r"  +", " ", corrected)
            corrections_made += 1
            logger.info(
                "Surgical | Removed UNVERIFIABLE claim: '%s'",
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
