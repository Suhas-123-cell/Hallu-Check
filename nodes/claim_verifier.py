"""
hallu-check | nodes/claim_verifier.py
Node 5 — Claim-Level Hallucination Detection

Extracts atomic claims from the LLM output, verifies each one against
RAG-retrieved context using NLI (Natural Language Inference), and produces
a detailed hallucination report with per-claim verdicts.

Design constraints:
  - Uses exactly ONE Gemini call (extraction + verification batched)
  - Detects honest uncertainty locally (no LLM needed)
  - Falls back gracefully if Gemini is unavailable
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import google.genai as genai  # type: ignore[import-not-found, import-untyped]
from google.genai import types as genai_types  # type: ignore[import-not-found, import-untyped]

from config import (  # type: ignore[import-not-found]
    GEMINI_API_KEY,
    GEMINI_MODEL,
    HALLUCINATION_THRESHOLD,
)

logger = logging.getLogger("hallu-check.claim_verifier")

# Maximum retries for rate-limited Gemini calls
_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 15  # seconds

# ── Gemini generation config (scaled for large inputs/outputs) ───────────────
_GEMINI_HTTP_OPTIONS = genai_types.HttpOptions(timeout=120_000)  # 120s (in ms) for massive prompts
_GEMINI_GENERATE_CONFIG = genai_types.GenerateContentConfig(
    max_output_tokens=8192,
    http_options=_GEMINI_HTTP_OPTIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClaimVerdict:
    """Verdict for a single atomic claim."""
    claim: str
    verdict: str        # SUPPORTED, CONTRADICTED, UNVERIFIABLE, NO_CLAIM, HONEST_UNCERTAINTY
    evidence: str       # RAG snippet backing the verdict (empty if none)
    confidence: float   # 0.0–1.0 confidence in this verdict
    reasoning: str = "" # Brief reasoning chain explaining how the verdict was reached

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HallucinationReport:
    """Full hallucination analysis report."""
    claim_verdicts: List[ClaimVerdict] = field(default_factory=list)
    hallucination_score: float = 0.0       # 0.0 (clean) → 1.0 (fully hallucinated)
    hallucination_detected: bool = False
    summary: str = ""                      # Human-readable summary

    def to_dict(self) -> dict:
        return {
            "claim_verdicts": [cv.to_dict() for cv in self.claim_verdicts],
            "hallucination_score": self.hallucination_score,
            "hallucination_detected": self.hallucination_detected,
            "summary": self.summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Honest Uncertainty Detection (LOCAL — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────
_UNCERTAINTY_PATTERNS = [
    r"i (?:don['\u2019]?t|do not) (?:have|know)",
    r"i couldn['\u2019]?t find",
    r"i (?:am |['\u2019]m )not (?:sure|certain|aware)",
    r"(?:no|not enough) (?:specific |reliable )?information",
    r"(?:cannot|can['\u2019]?t) (?:find|provide|confirm|verify)",
    r"there (?:is|are) no (?:specific |reliable )?(?:information|data|records?)",
    r"i (?:am |['\u2019]m )unable to (?:find|provide|verify)",
    r"as of my (?:last |knowledge )?(?:update|cut.?off)",
    r"my (?:training )?(?:data|knowledge) (?:does not|doesn['\u2019]?t)",
    r"beyond my (?:current )?knowledge",
    r"not (?:a )?(?:widely |publicly )?(?:known|recognized|notable)",
    r"i['\u2019]m not sure",
]
_UNCERTAINTY_RE = re.compile(
    "|".join(_UNCERTAINTY_PATTERNS), re.IGNORECASE
)


def is_honest_uncertainty(text: str) -> bool:
    """
    Detect if the LLM output is an honest admission of not knowing.
    This is NOT a hallucination — it's the model being truthful.

    Uses regex patterns only (zero API calls).
    """
    return bool(_UNCERTAINTY_RE.search(text))


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Check if RAG output has substantive content (LOCAL — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────
_RAG_EMPTY_MARKERS = [
    "no relevant context found",
    "no web results found",
    "failed to scrape",
    "using fallback context",
    "no relevant information",
    "no specific information",
    # Social media / login-wall boilerplate (scraped but empty)
    "title: facebook",
    "title: instagram",
    "title: linkedin",
    "sign up to see",
    "log in to continue",
    "content isn't available",
    "javascript is required",
    "page not found",
]


def _rag_has_substantive_content(rag_output: str, query: str) -> bool:
    """
    Determine if the RAG output contains real, substantive content about
    the query subject — not just empty/fallback messages.

    Checks:
      1. RAG output is not a known fallback/empty message
      2. RAG output has meaningful length (>150 chars after stripping boilerplate)
      3. At least one word from the query appears in the RAG output

    Returns True if the RAG output has real content worth using for refinement.
    """
    if not rag_output or len(rag_output.strip()) < 50:
        return False

    rag_lower = rag_output.lower()

    # Check for known empty/fallback markers
    if any(marker in rag_lower for marker in _RAG_EMPTY_MARKERS):
        return False

    # Check that at least one query keyword appears in the RAG output
    # (ensures the content is actually relevant to the question)
    query_words = set(query.lower().split())
    stop_words = {
        "who", "what", "where", "when", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "do", "does", "tell", "me", "about", "please",
    }
    meaningful_query_words = query_words - stop_words
    if not meaningful_query_words:
        meaningful_query_words = query_words  # fallback: use all words

    query_word_in_rag = any(word in rag_lower for word in meaningful_query_words)
    if not query_word_in_rag:
        return False

    # Has enough content (not just a title or one-liner)
    content_length = len(rag_output.strip())
    if content_length < 100:
        return False

    # Check for actual informational sentences — not just page titles or URLs.
    # Social media pages often contain the query keyword in the URL/title
    # but have zero real biographical/factual content.
    sentences = [s.strip() for s in re.split(r'[.!?]+', rag_output) if len(s.strip()) > 30]
    informational_sentences = [
        s for s in sentences
        if not s.lower().startswith(("title:", "url", "source:", "**source", "http"))
        and "---" not in s
    ]
    if len(informational_sentences) < 2:
        logger.debug(
            "Node 5 | RAG has keywords but no informational sentences (%d found), "
            "treating as non-substantive",
            len(informational_sentences),
        )
        return False

    logger.debug(
        "Node 5 | RAG has substantive content (%d chars, %d informational sentences)",
        content_length,
        len(informational_sentences),
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gemini API Helper (with rate-limit retry)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_retry_delay(error_msg: str) -> float:
    """Extract retryDelay seconds from a Gemini 429 error message."""
    match = re.search(r"retryDelay['\"]:\s*['\"]([\d.]+)(s|ms)", str(error_msg))
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value / 1000 if unit == "ms" else value
    return _DEFAULT_RETRY_DELAY


def _gemini_generate(prompt: str) -> str:
    """Call Gemini with rate-limit-aware retries. Returns raw text."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file.\n"
            "Get one free at https://aistudio.google.com/app/apikey"
        )

    # SSL workaround for macOS
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
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            is_timeout = "timed out" in error_str or "ReadTimeout" in error_str or "TimeoutError" in error_str
            is_retryable = is_rate_limit or is_timeout

            if is_retryable and attempt < _MAX_RETRIES:
                if is_rate_limit:
                    delay = min(_parse_retry_delay(error_str) + 1, 60)
                else:
                    delay = 5  # short delay before timeout retry
                logger.warning(
                    "Node 5 | Gemini %s (attempt %d/%d), waiting %.1fs…",
                    "rate-limited" if is_rate_limit else "timed out",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            logger.warning(
                "Node 5 | Gemini call failed (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, e,
            )
            return ""

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Claim Extraction + Verification (SINGLE Gemini call)
# ─────────────────────────────────────────────────────────────────────────────
_VERIFY_PROMPT = """\
You are a hallucination detection system. You will be given a user's question, an LLM's answer, and retrieved factual context.

Your task:
1. Extract every atomic factual claim from the LLM answer.
   - An "atomic claim" is the smallest standalone factual statement.
   - Skip filler phrases, greetings, and meta-statements like "Sure, here's the answer".
   - If the answer contains NO factual claims (e.g., "I don't know"), return an empty claims list.

2. For each claim, verify it against the provided context and assign a verdict.
   IMPORTANT — follow these reasoning rules carefully:

   a) USE LOGICAL/TRANSITIVE INFERENCE:
      Do NOT evaluate claims in isolation. Use logical reasoning to connect facts.
      Example: If the context says "X is the current President" and the LLM claims
      "Y was the previous President", these are CONSISTENT — do NOT mark Y's claim
      as CONTRADICTED just because the context names X. Only mark CONTRADICTED if
      the context explicitly names a DIFFERENT person as the previous President.

   b) CONSIDER THE QUESTION'S INTENT:
      If the user asks about the "previous" holder of a role, and the LLM correctly
      names someone who held the role before the current holder mentioned in the
      context, that is logically SUPPORTED, not contradicted.

   c) DISTINGUISH "OUTDATED" FROM "WRONG":
      If a claim was historically true but is no longer current (e.g., "X has been
      serving as PM since 2020" but the context shows X was replaced in 2025), the
      IDENTITY part (X was PM) may still be correct even if the TENURE part (still
      serving) is outdated. Evaluate each atomic fact separately.

   d) STRICT CONTRADICTED THRESHOLD:
      Only use CONTRADICTED when the context DIRECTLY and EXPLICITLY conflicts with
      the claim — not when the context merely provides different (but compatible)
      information. When in doubt between CONTRADICTED and UNVERIFIABLE, prefer
      UNVERIFIABLE.

   Verdict options:
   - SUPPORTED: The context confirms this claim (directly or through logical inference).
   - CONTRADICTED: The context DIRECTLY and EXPLICITLY conflicts with this specific claim.
   - UNVERIFIABLE: The context has no relevant information about this claim, or the
     evidence is ambiguous.
   - NO_CLAIM: This is not a factual claim (opinion, hedge, question, etc.)

3. For each verdict, provide:
   - The exact evidence quote from the context (or "No relevant evidence found" if UNVERIFIABLE).
   - A brief reasoning chain showing how you arrived at the verdict.
   - A confidence score (0.0 to 1.0) for your verdict.

Question: {query}

LLM Answer:
{llm_output}

Retrieved Context:
{rag_context}

Respond ONLY with this JSON (no markdown fences, no extra text):
{{
  "claims": [
    {{
      "claim": "<atomic claim text>",
      "verdict": "<SUPPORTED|CONTRADICTED|UNVERIFIABLE|NO_CLAIM>",
      "evidence": "<exact quote from context or 'No relevant evidence found'>",
      "reasoning": "<brief reasoning chain explaining how you reached this verdict>",
      "confidence": <0.0-1.0>
    }}
  ]
}}
"""


def _extract_and_verify_claims(
    llm_output: str,
    rag_output: str,
    query: str,
) -> List[ClaimVerdict]:
    """
    Extract atomic claims from LLM output and verify each against RAG context.

    Uses exactly ONE Gemini API call (extraction + verification batched).
    Falls back to keyword-based heuristic if Gemini is unavailable.
    """
    # Truncate context to avoid token limits (Gemini free tier)
    rag_truncated = rag_output[:12000] if len(rag_output) > 12000 else rag_output

    prompt = _VERIFY_PROMPT.format(
        query=query,
        llm_output=llm_output,
        rag_context=rag_truncated,
    )

    raw = _gemini_generate(prompt)
    if not raw:
        logger.warning("Node 5 | Gemini unavailable, using fallback heuristic.")
        return _fallback_verify(llm_output, rag_output)

    # Parse JSON response
    try:
        # Try extracting from code block first
        json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1).strip())
        else:
            parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Try to find any JSON object in the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Node 5 | Failed to parse Gemini response, using fallback.")
                return _fallback_verify(llm_output, rag_output)
        else:
            logger.warning("Node 5 | No JSON in Gemini response, using fallback.")
            return _fallback_verify(llm_output, rag_output)

    # Convert parsed JSON to ClaimVerdict objects
    verdicts: List[ClaimVerdict] = []
    claims = parsed.get("claims", [])

    if not isinstance(claims, list):
        logger.warning("Node 5 | 'claims' is not a list, using fallback.")
        return _fallback_verify(llm_output, rag_output)

    for item in claims:
        if not isinstance(item, dict):
            continue
        verdict_str = item.get("verdict", "UNVERIFIABLE").upper()
        # Normalize verdict
        valid_verdicts = {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NO_CLAIM"}
        if verdict_str not in valid_verdicts:
            verdict_str = "UNVERIFIABLE"

        verdicts.append(ClaimVerdict(
            claim=item.get("claim", ""),
            verdict=verdict_str,
            evidence=item.get("evidence", ""),
            confidence=float(item.get("confidence", 0.5)),
            reasoning=item.get("reasoning", ""),
        ))

    logger.info(
        "Node 5 | Extracted %d claims from LLM output via Gemini.",
        len(verdicts),
    )
    return verdicts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fallback Heuristic (when Gemini is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_verify(llm_output: str, rag_output: str) -> List[ClaimVerdict]:
    """
    Simple keyword-overlap heuristic for claim verification when Gemini
    is unavailable. Treats the entire LLM output as one claim and checks
    word overlap with RAG context.
    """
    llm_words = set(llm_output.lower().split())
    rag_words = set(rag_output.lower().split())

    # Remove stop words
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
            "at", "to", "for", "and", "or", "but", "not", "it", "this", "that",
            "with", "by", "from", "as", "be", "has", "have", "had", "do", "does"}
    llm_content = llm_words - stop
    rag_content = rag_words - stop

    if not llm_content or not rag_content:
        return [ClaimVerdict(
            claim=llm_output[:200],
            verdict="UNVERIFIABLE",
            evidence="(Gemini unavailable; no context overlap analysis possible)",
            confidence=0.3,
        )]

    overlap = llm_content & rag_content
    overlap_ratio = len(overlap) / len(llm_content) if llm_content else 0

    if overlap_ratio > 0.4:
        verdict = "SUPPORTED"
    elif overlap_ratio < 0.1:
        verdict = "UNVERIFIABLE"
    else:
        verdict = "UNVERIFIABLE"

    return [ClaimVerdict(
        claim=llm_output[:200],
        verdict=verdict,
        evidence=f"(Fallback: {len(overlap)}/{len(llm_content)} content words overlap with RAG context)",
        confidence=0.3,
    )]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hallucination Scoring
# ─────────────────────────────────────────────────────────────────────────────
def _compute_hallucination_score(
    verdicts: List[ClaimVerdict],
    bertscore_f1: float,
) -> float:
    """
    Compute a composite hallucination score (0.0 = clean, 1.0 = fully hallucinated).

    Weights:
      - CONTRADICTED claims are the strongest hallucination signal (weight=1.0)
      - UNVERIFIABLE claims are a moderate signal (weight=0.3)
      - SUPPORTED / NO_CLAIM claims reduce the score (weight=0.0)
      - BERTScore is used as a secondary signal (inverted: low BERTScore → higher score)

    Formula:
      claim_score = weighted_sum(verdicts) / total_claims
      hallucination_score = 0.7 * claim_score + 0.3 * (1 - bertscore_f1)
    """
    if not verdicts:
        # No claims extracted — use BERTScore alone
        return max(0.0, min(1.0, 1.0 - bertscore_f1))

    weights = {
        "CONTRADICTED": 1.0,
        "UNVERIFIABLE": 0.3,
        "SUPPORTED": 0.0,
        "NO_CLAIM": 0.0,
        "HONEST_UNCERTAINTY": 0.0,
    }

    weighted_sum = sum(
        weights.get(v.verdict, 0.3) * v.confidence
        for v in verdicts
    )
    total = len(verdicts)
    claim_score = weighted_sum / total if total > 0 else 0.0

    # Composite: 70% claim-level, 30% BERTScore
    composite = 0.7 * claim_score + 0.3 * (1.0 - bertscore_f1)
    return max(0.0, min(1.0, round(composite, 4)))


def _generate_summary(verdicts: List[ClaimVerdict], score: float) -> str:
    """Generate a human-readable hallucination summary."""
    counts = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    total = len(verdicts)
    parts = []

    if counts.get("HONEST_UNCERTAINTY"):
        return (
            f"The LLM honestly admitted uncertainty. "
            f"This is NOT a hallucination — no correction needed."
        )

    if counts.get("CONTRADICTED", 0) > 0:
        n = counts["CONTRADICTED"]
        parts.append(
            f"{n} of {total} claim(s) CONTRADICTED by evidence"
        )
    if counts.get("UNVERIFIABLE", 0) > 0:
        n = counts["UNVERIFIABLE"]
        parts.append(
            f"{n} of {total} claim(s) could not be verified"
        )
    if counts.get("SUPPORTED", 0) > 0:
        n = counts["SUPPORTED"]
        parts.append(
            f"{n} of {total} claim(s) supported by evidence"
        )

    if not parts:
        return "No factual claims detected in the LLM output."

    summary = "; ".join(parts) + f". Hallucination score: {score:.2f}."
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public API — Full Claim Verification Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def verify_claims(
    llm_output: str,
    rag_output: str,
    query: str,
    bertscore_f1: float = 0.0,
) -> HallucinationReport:
    """
    Node 5 — Full claim-level hallucination detection.

    1. Check for honest uncertainty (local, no API call)
    2. Extract & verify claims via Gemini (single API call)
    3. Compute hallucination score (BERTScore + claim verdicts)
    4. Return structured report

    Args:
        llm_output: The LLM's preliminary answer (Node 1).
        rag_output: RAG-retrieved context (Node 4).
        query:      The user's original query.
        bertscore_f1: BERTScore F1 from Node 4+5 (0.0 if not computed).

    Returns:
        HallucinationReport with per-claim verdicts and overall score.
    """
    logger.info("Node 5 | Starting claim-level verification…")

    # ── Step 1: Check for honest uncertainty (FREE — no API call) ────
    if is_honest_uncertainty(llm_output):
        # Before accepting honest uncertainty, check if RAG actually found
        # substantive content. If the web search found real information but
        # the LLM said "I don't know", that's the LLM being wrong — not honest.
        rag_has_content = _rag_has_substantive_content(rag_output, query)

        if rag_has_content:
            logger.info(
                "Node 5 | LLM expressed uncertainty, but RAG found real content. "
                "Treating as CONTRADICTED — will trigger refinement."
            )
            verdict = ClaimVerdict(
                claim="LLM claimed it could not find information about the topic.",
                verdict="CONTRADICTED",
                evidence=(
                    "RAG context contains substantive information about the query subject. "
                    "The LLM's claim of 'no information found' is incorrect."
                ),
                confidence=0.9,
            )
            score = max(0.7, 0.7 * 1.0 + 0.3 * (1.0 - bertscore_f1))
            return HallucinationReport(
                claim_verdicts=[verdict],
                hallucination_score=min(1.0, score),
                hallucination_detected=True,
                summary=(
                    "The LLM said it couldn't find information, but the web search "
                    "found relevant content. Refinement will use the retrieved evidence."
                ),
            )

        # RAG also has no useful content — genuine honest uncertainty
        logger.info("Node 5 | Detected HONEST UNCERTAINTY — not a hallucination.")
        verdict = ClaimVerdict(
            claim=llm_output[:300],
            verdict="HONEST_UNCERTAINTY",
            evidence="LLM expressed honest uncertainty about the topic.",
            confidence=0.95,
        )
        return HallucinationReport(
            claim_verdicts=[verdict],
            hallucination_score=0.0,
            hallucination_detected=False,
            summary=(
                "The LLM honestly admitted it doesn't have information about "
                "this topic. This is NOT a hallucination — no correction needed."
            ),
        )

    # ── Step 2: Extract & verify claims (SINGLE Gemini call) ─────────
    verdicts = _extract_and_verify_claims(llm_output, rag_output, query)

    if not verdicts:
        logger.info("Node 5 | No claims extracted from LLM output.")
        return HallucinationReport(
            claim_verdicts=[],
            hallucination_score=0.0,
            hallucination_detected=False,
            summary="No factual claims detected in the LLM output.",
        )

    # ── Step 3: Compute hallucination score ──────────────────────────
    score = _compute_hallucination_score(verdicts, bertscore_f1)
    detected = score >= HALLUCINATION_THRESHOLD

    # ── Step 4: Generate summary ─────────────────────────────────────
    summary = _generate_summary(verdicts, score)

    report = HallucinationReport(
        claim_verdicts=verdicts,
        hallucination_score=score,
        hallucination_detected=detected,
        summary=summary,
    )

    logger.info(
        "Node 5 | Verification complete: score=%.4f, detected=%s, claims=%d "
        "(SUPPORTED=%d, CONTRADICTED=%d, UNVERIFIABLE=%d)",
        score,
        detected,
        len(verdicts),
        sum(1 for v in verdicts if v.verdict == "SUPPORTED"),
        sum(1 for v in verdicts if v.verdict == "CONTRADICTED"),
        sum(1 for v in verdicts if v.verdict == "UNVERIFIABLE"),
    )

    return report
