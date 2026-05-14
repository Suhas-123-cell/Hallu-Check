from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import (  # type: ignore[import-not-found]
    HALLUCINATION_THRESHOLD,
    USE_NLI_MODEL,
    NLI_BATCH_SIZE,
    ENABLE_EGV,
    ENABLE_SURGICAL_CORRECTION,
)

logger = logging.getLogger("hallu-check.claim_verifier")


# ─────────────────────────────────────────────────────────────────────────────
# NLI Model (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────────────
_nli_loaded = False


def _ensure_nli_model() -> bool:
    global _nli_loaded
    if _nli_loaded:
        return True

    if not USE_NLI_MODEL:
        logger.info("Node 5 | NLI model disabled (USE_NLI_MODEL=false).")
        return False

    try:
        from nodes.nli_model import load_model, is_loaded  # type: ignore[import-not-found]
        if is_loaded():
            _nli_loaded = True
            return True
        success = load_model()
        _nli_loaded = success
        if success:
            logger.info("Node 5 | NLI model loaded successfully.")
        else:
            logger.warning("Node 5 | NLI model not available — falling back to Gemini.")
        return success
    except Exception as e:
        logger.warning("Node 5 | Failed to load NLI model (%s) — falling back to Gemini.", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClaimVerdict:
    claim: str
    verdict: str        # SUPPORTED, CONTRADICTED, UNVERIFIABLE, NO_CLAIM, HONEST_UNCERTAINTY
    evidence: str       # RAG snippet backing the verdict (empty if none)
    confidence: float   # 0.0–1.0 (real softmax probability from NLI model)
    reasoning: str = "" # Brief reasoning chain explaining how the verdict was reached
    source_url: str = ""        # URL of the evidence source
    source_paragraph: str = ""  # Specific paragraph that provided the strongest NLI signal
    nli_probabilities: Dict[str, float] = field(default_factory=dict)  # Full probability distribution
    verification_method: str = "NLI"  # "NLI", "EGV_CODE", "EGV_MATH" — per-claim verifier used

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HallucinationReport:
    claim_verdicts: List[ClaimVerdict] = field(default_factory=list)
    hallucination_score: float = 0.0       # 0.0 (clean) → 1.0 (fully hallucinated)
    hallucination_detected: bool = False
    summary: str = ""                      # Human-readable summary
    verification_method: str = "gemini"    # "nli" or "gemini" — tracks which method was used

    def to_dict(self) -> dict:
        return {
            "claim_verdicts": [cv.to_dict() for cv in self.claim_verdicts],
            "hallucination_score": self.hallucination_score,
            "hallucination_detected": self.hallucination_detected,
            "summary": self.summary,
            "verification_method": self.verification_method,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Honest Uncertainty Detection (LOCAL — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────
_UNCERTAINTY_PATTERNS = [
    r"i (?:don['\u2019]?t|do not) know\b",
    r"i couldn['\u2019]?t find (?:any )?(?:information|data)",
    r"i (?:am |['\u2019]m )not sure\b",
    r"(?:cannot|can['\u2019]?t) (?:find|provide|confirm|verify) (?:any )?(?:information|data)",
    r"there (?:is|are) no (?:specific |reliable )?(?:information|data|records?) available",
    r"i (?:am |['\u2019]m )unable to (?:find|provide|verify) (?:the )?(?:answer|information)",
    r"beyond my (?:current )?knowledge",
]
_UNCERTAINTY_RE = re.compile(
    "|".join(_UNCERTAINTY_PATTERNS), re.IGNORECASE
)


def is_honest_uncertainty(text: str) -> bool:
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
# 3. Claim Extraction (Gemini) — ONLY extracts claims, no verification
# ─────────────────────────────────────────────────────────────────────────────
_EXTRACT_PROMPT = """\
You are a claim extraction system. Extract every atomic factual claim from the LLM answer below.

CRITICAL RULES — you MUST follow these:
- COPY claims using the SAME words as the original text. Do NOT rephrase or paraphrase.
- Do NOT add any information that is not EXPLICITLY stated in the LLM answer.
- Do NOT confuse similar-sounding terms (e.g., BPE ≠ BERT, BP ≠ BPE).
- Every claim you output MUST be directly traceable to a sentence in the LLM answer.
- An "atomic claim" is the smallest standalone factual statement.
- Skip filler phrases, greetings, and meta-statements like "Sure, here's the answer".
- If the answer contains NO factual claims (e.g., "I don't know"), return an empty list.
- Do NOT verify or judge the claims — only extract them.

User's original question (for context only — do NOT extract claims from this):
{query}

LLM Answer (extract claims ONLY from this):
{llm_output}

Respond ONLY with this JSON (no markdown fences, no extra text):
{{
  "claims": [
    "claim 1 text",
    "claim 2 text",
    "claim 3 text"
  ]
}}
"""


def _extract_claims(llm_output: str, query: str = "") -> List[str]:
    if not llm_output or len(llm_output.strip()) < 10:
        return []

    # Strip fenced code blocks before tokenization — code fragments are
    # not verifiable claims and confuse sentence splitting.
    prose_only = re.sub(r"```[\s\S]*?```", "", llm_output)

    # Also strip inline code spans (`...`)
    prose_only = re.sub(r"`[^`]+`", "", prose_only)

    # If stripping removed everything, there are no prose claims
    if len(prose_only.strip()) < 20:
        logger.info("Node 5 | LLM output is code-only — no prose claims to extract.")
        return []

    # Use NLTK sent_tokenize for robust sentence boundary detection.
    # Falls back to regex if NLTK data is not available.
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
        sentences = sent_tokenize(prose_only)
    except Exception:
        # Fallback: split on sentence-ending punctuation followed by space+uppercase
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', prose_only)

    # Filter: keep only sentences with meaningful content (≥5 words)
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20 and len(s.strip().split()) >= 5
    ]

    # Filter out meta-statements (not factual claims).
    # Match at sentence START — but first strip leading transition words
    # so "However, please note…" / "But I recommend…" are caught too.
    meta_starts = [
        "sure,", "here is ", "here's ", "i can help", "let me ",
        "in summary", "i hope ", "feel free", "let me know",
        "you can test", "you can use", "you can run",
        "this code ", "this solution ", "this should ",
        "i don't have", "i do not have", "please note", "note that",
        "as an ai", "i am an ai", "i cannot", "i can't",
        "it's important to note", "keep in mind", "i am a large",
        "i recommend", "i would recommend", "i suggest",
        "for the most ", "for more ", "for up-to-date ",
        "please check", "please verify", "please consult",
        "this information may", "this may have changed",
        "the information provided",
    ]
    _LEADING_CONNECTIVES = re.compile(
        r"^(?:however|but|also|additionally|furthermore|moreover|"
        r"nonetheless|nevertheless|though|although|still|yet|so|"
        r"therefore|thus|hence)[,:]?\s+",
        re.IGNORECASE,
    )

    def _is_meta(sentence: str) -> bool:
        stripped = _LEADING_CONNECTIVES.sub("", sentence.lower().strip())
        return any(stripped.startswith(meta) for meta in meta_starts)

    claims = [s for s in sentences if not _is_meta(s)]

    # Cap at 10 claims
    claims = claims[:10]

    # Validate: discard any claims not grounded in the original LLM output
    claims = _validate_claims(claims, llm_output)

    logger.info("Node 5 | Extracted %d validated claims via NLTK sentence split.", len(claims))
    return claims


def _fallback_extract_claims(llm_output: str) -> List[str]:
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", llm_output)
        if len(s.strip()) > 20
    ]
    # Filter out meta-statements
    claims = [
        s for s in sentences
        if not any(meta in s.lower() for meta in [
            "sure,", "here's", "i can help", "let me",
            "based on", "according to", "in summary",
        ])
    ]
    logger.info("Node 5 | Extracted %d claims via fallback sentence split.", len(claims))
    return claims[:10]  # cap at 10


def _validate_claims(claims: List[str], llm_output: str) -> List[str]:
    if not claims:
        return []

    llm_words = set(llm_output.lower().split())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "and", "or", "but", "not", "it", "this", "that",
        "with", "by", "from", "as", "be", "has", "have", "had", "do", "does",
        "its", "their", "which", "who", "whom", "what", "where", "when",
    }

    validated: List[str] = []
    for claim in claims:
        claim_words = set(claim.lower().split())
        claim_content = claim_words - stop_words
        if not claim_content:
            validated.append(claim)  # all stop words — keep it
            continue

        overlap = len(claim_content & llm_words) / len(claim_content)
        if overlap >= 0.6:
            validated.append(claim)
        else:
            logger.warning(
                "Node 5 | Discarding fabricated claim (overlap=%.0f%%): '%s'",
                overlap * 100,
                claim[:80],
            )

    logger.info(
        "Node 5 | Claim validation: %d/%d claims passed (grounded in LLM output).",
        len(validated),
        len(claims),
    )
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# 4. NLI-Based Claim Verification (DeBERTa)
# ─────────────────────────────────────────────────────────────────────────────
def _verify_claims_nli(
    claims: List[str],
    rag_output: str,
    query: str,
) -> List[ClaimVerdict]:
    from nodes.nli_model import classify_nli_batch  # type: ignore[import-not-found]

    if not claims:
        return []

    # Truncate RAG context for NLI model (256 token max in training)
    rag_truncated = rag_output[:2000] if len(rag_output) > 2000 else rag_output

    # Build NLI pairs: (premise=evidence, hypothesis=claim)
    pairs = [(rag_truncated, claim) for claim in claims]

    # Batch classify all claims at once
    t0 = time.time()
    nli_results = classify_nli_batch(pairs, batch_size=NLI_BATCH_SIZE)
    elapsed_ms = (time.time() - t0) * 1000

    logger.info(
        "Node 5 | NLI verification: %d claims in %.0fms (%.1fms/claim)",
        len(claims),
        elapsed_ms,
        elapsed_ms / len(claims) if claims else 0,
    )

    # Extract source URL from RAG context (for attribution)
    source_url = ""
    url_match = re.search(r"\*\*Source:\*\*\s*(https?://\S+)", rag_output)
    if url_match:
        source_url = url_match.group(1)

    # Find best-matching evidence paragraph per claim
    rag_paragraphs = [
        p.strip() for p in rag_output.split("\n\n") if len(p.strip()) > 30
    ]

    verdicts: List[ClaimVerdict] = []
    for claim, nli_result in zip(claims, nli_results):
        # Find the most relevant evidence paragraph
        best_evidence = ""
        if rag_paragraphs:
            # Simple overlap scoring to find best paragraph
            claim_words = set(claim.lower().split())
            best_score = -1
            for para in rag_paragraphs[:20]:  # limit to first 20 paragraphs
                para_words = set(para.lower().split())
                overlap = len(claim_words & para_words)
                if overlap > best_score:
                    best_score = overlap
                    best_evidence = para[:300]  # truncate evidence

        verdict = ClaimVerdict(
            claim=claim,
            verdict=nli_result["verdict"],
            evidence=best_evidence or "No specific evidence paragraph identified.",
            confidence=nli_result["confidence"],
            reasoning=(
                f"NLI model classified with "
                f"P(entailment)={nli_result['probabilities']['entailment']:.3f}, "
                f"P(neutral)={nli_result['probabilities']['neutral']:.3f}, "
                f"P(contradiction)={nli_result['probabilities']['contradiction']:.3f}"
            ),
            source_url=source_url,
            source_paragraph=best_evidence[:200] if best_evidence else "",
            nli_probabilities=nli_result["probabilities"],
        )
        verdicts.append(verdict)

    return verdicts





# ─────────────────────────────────────────────────────────────────────────────
# 6. Keyword-Based Fallback (last resort)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_verify(llm_output: str, rag_output: str) -> List[ClaimVerdict]:
    llm_words = set(llm_output.lower().split())
    rag_words = set(rag_output.lower().split())

    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
            "at", "to", "for", "and", "or", "but", "not", "it", "this", "that",
            "with", "by", "from", "as", "be", "has", "have", "had", "do", "does"}
    llm_content = llm_words - stop
    rag_content = rag_words - stop

    if not llm_content or not rag_content:
        return [ClaimVerdict(
            claim=llm_output[:200],
            verdict="UNVERIFIABLE",
            evidence="(Both NLI model and Gemini unavailable; no verification possible)",
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
# 7. Hallucination Scoring (NLI-based)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_hallucination_score(
    verdicts: List[ClaimVerdict],
    nli_alignment_score: float,
) -> float:
    if not verdicts:
        return max(0.0, min(1.0, 1.0 - nli_alignment_score))

    claim_risks: List[float] = []
    for v in verdicts:
        if v.nli_probabilities:
            # Use real NLI probabilities
            p_contra = v.nli_probabilities.get("contradiction", 0.0)
            p_neutral = v.nli_probabilities.get("neutral", 0.0)
            risk = p_contra + 0.3 * p_neutral
        else:
            # Fallback: use verdict string (for Gemini fallback path)
            weights = {
                "CONTRADICTED": 1.0,
                "UNVERIFIABLE": 0.3,
                "SUPPORTED": 0.0,
                "NO_CLAIM": 0.0,
                "HONEST_UNCERTAINTY": 0.0,
            }
            risk = weights.get(v.verdict, 0.3) * v.confidence
        claim_risks.append(risk)

    claim_score = sum(claim_risks) / len(claim_risks) if claim_risks else 0.0

    # Composite: 70% claim-level risk, 30% (1 - alignment)
    composite = 0.7 * claim_score + 0.3 * (1.0 - nli_alignment_score)
    return max(0.0, min(1.0, round(composite, 4)))


def _generate_summary(verdicts: List[ClaimVerdict], score: float, method: str) -> str:
    counts: Dict[str, int] = {}
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

    method_labels = {
        "nli": "NLI model (DeBERTa)",
        "gemini": "Gemini LLM",
        "mixed": "NLI + Execution-Grounded Verification",
        "EGV_CODE": "Code execution verification",
        "EGV_MATH": "SymPy math verification",
    }
    method_label = method_labels.get(method, method)
    summary = "; ".join(parts) + f". Hallucination score: {score:.2f} (verified by {method_label})."
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 8. Public API — Full Claim Verification Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def verify_claims(
    llm_output: str,
    rag_output: str,
    query: str,
    bertscore_f1: float = 0.0,
    nli_alignment_score: float | None = None,
) -> HallucinationReport:
    logger.info("Node 5 | Starting claim-level verification…")

    # Use NLI alignment if available, otherwise fall back to BERTScore
    alignment_score = nli_alignment_score if nli_alignment_score is not None else bertscore_f1

    # ── Step 1: Check for honest uncertainty (FREE — no API call) ────
    if is_honest_uncertainty(llm_output):
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
            score = max(0.7, 0.7 * 1.0 + 0.3 * (1.0 - alignment_score))
            return HallucinationReport(
                claim_verdicts=[verdict],
                hallucination_score=min(1.0, score),
                hallucination_detected=True,
                summary=(
                    "The LLM said it couldn't find information, but the web search "
                    "found relevant content. Refinement will use the retrieved evidence."
                ),
                verification_method="heuristic",
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
            verification_method="heuristic",
        )

    # ── Step 2: Determine verification method ────────────────────────
    use_nli = _ensure_nli_model()
    method = "nli" if use_nli else "keyword"

    if use_nli:
        # ── NLI PATH: Extract claims → Classify → Route ─────────────
        logger.info("Node 5 | Using NLI model (DeBERTa) + EGV routing.")

        # Step 2a: Extract atomic claims (local sentence splitting + validation)
        claims = _extract_claims(llm_output, query)
        if not claims:
            logger.info("Node 5 | No claims extracted from LLM output.")
            return HallucinationReport(
                claim_verdicts=[],
                hallucination_score=0.0,
                hallucination_detected=False,
                summary="No factual claims detected in the LLM output.",
                verification_method=method,
            )

        # Step 2b: Classify each claim and route to the appropriate verifier
        verdicts = _classify_and_verify_claims(
            claims, llm_output, rag_output, query,
        )

        # Determine the dominant verification method for the report
        methods_used = {v.verification_method for v in verdicts}
        if len(methods_used) > 1:
            method = "mixed"
        elif methods_used:
            method = methods_used.pop()
        # else: stays as "nli"
    else:
        # ── KEYWORD FALLBACK: No NLI model and no Gemini API ─────────
        logger.info("Node 5 | NLI model unavailable — using keyword-overlap fallback.")
        verdicts = _fallback_verify(llm_output, rag_output)

    if not verdicts:
        logger.info("Node 5 | No claims extracted from LLM output.")
        return HallucinationReport(
            claim_verdicts=[],
            hallucination_score=0.0,
            hallucination_detected=False,
            summary="No factual claims detected in the LLM output.",
            verification_method=method,
        )

    # ── Step 3: Surgical correction (DISABLED — handled by ICR loop) ──
    # The ICR loop in iterative_refiner.py handles correction with
    # route-awareness: Gemini for FACTUAL, surgical for REASONING.
    # Running it here with the local LLM corrupts factual answers.

    # ── Step 4: Compute hallucination score ──────────────────────────
    score = _compute_hallucination_score(verdicts, alignment_score)
    detected = score >= HALLUCINATION_THRESHOLD

    # ── Step 5: Generate summary ─────────────────────────────────────
    summary = _generate_summary(verdicts, score, method)

    report = HallucinationReport(
        claim_verdicts=verdicts,
        hallucination_score=score,
        hallucination_detected=detected,
        summary=summary,
        verification_method=method,
    )

    logger.info(
        "Node 5 | Verification complete [%s]: score=%.4f, detected=%s, claims=%d "
        "(SUPPORTED=%d, CONTRADICTED=%d, UNVERIFIABLE=%d)",
        method.upper(),
        score,
        detected,
        len(verdicts),
        sum(1 for v in verdicts if v.verdict == "SUPPORTED"),
        sum(1 for v in verdicts if v.verdict == "CONTRADICTED"),
        sum(1 for v in verdicts if v.verdict == "UNVERIFIABLE"),
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 9. Claim Classification + EGV Routing
# ─────────────────────────────────────────────────────────────────────────────

def _classify_and_verify_claims(
    claims: List[str],
    llm_output: str,
    rag_output: str,
    query: str,
) -> List[ClaimVerdict]:
    from nodes.claim_classifier import classify_claim  # type: ignore[import-not-found]

    # ── Classify all claims ───────────────────────────────────────────
    factual_claims: List[str] = []
    code_claims: List[str] = []
    math_claims: List[str] = []

    # Preserve original order: (index, claim, type)
    claim_routing: List[tuple] = []

    for i, claim in enumerate(claims):
        if ENABLE_EGV:
            claim_type = classify_claim(claim)
        else:
            claim_type = "factual"  # EGV disabled → everything goes to NLI

        claim_routing.append((i, claim, claim_type))

        if claim_type == "code":
            code_claims.append(claim)
        elif claim_type == "math":
            math_claims.append(claim)
        else:
            factual_claims.append(claim)

    logger.info(
        "Node 5 | Claim routing: %d factual, %d code, %d math (EGV=%s)",
        len(factual_claims), len(code_claims), len(math_claims),
        "enabled" if ENABLE_EGV else "disabled",
    )

    # ── Verify factual claims via NLI (batched) ───────────────────────
    factual_verdicts: Dict[str, ClaimVerdict] = {}
    if factual_claims:
        nli_verdicts = _verify_claims_nli(factual_claims, rag_output, query)
        for claim_str, verdict in zip(factual_claims, nli_verdicts):
            verdict.verification_method = "NLI"
            factual_verdicts[claim_str] = verdict

    # ── Verify code claims via EGV ────────────────────────────────────
    code_verdicts: Dict[str, ClaimVerdict] = {}
    if code_claims:
        try:
            from nodes.code_claim_verifier import verify_code_claim  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Node 5 | code_claim_verifier not available, falling back to NLI.")
            # Fall back to NLI for code claims
            nli_fallback = _verify_claims_nli(code_claims, rag_output, query)
            for claim_str, verdict in zip(code_claims, nli_fallback):
                verdict.verification_method = "NLI"
                code_verdicts[claim_str] = verdict
        else:
            for claim_str in code_claims:
                code_verdicts[claim_str] = _verify_single_code_claim(
                    claim_str, llm_output, rag_output, verify_code_claim,
                )

    # ── Verify math claims via EGV ────────────────────────────────────
    math_verdicts: Dict[str, ClaimVerdict] = {}
    if math_claims:
        try:
            from nodes.math_claim_verifier import verify_math_claim  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Node 5 | math_claim_verifier not available, falling back to NLI.")
            nli_fallback = _verify_claims_nli(math_claims, rag_output, query)
            for claim_str, verdict in zip(math_claims, nli_fallback):
                verdict.verification_method = "NLI"
                math_verdicts[claim_str] = verdict
        else:
            for claim_str in math_claims:
                math_verdicts[claim_str] = _verify_single_math_claim(
                    claim_str, verify_math_claim,
                )

    # ── Reassemble in original order ──────────────────────────────────
    all_verdicts: List[ClaimVerdict] = []
    for _i, claim_str, claim_type in claim_routing:
        if claim_type == "code" and claim_str in code_verdicts:
            all_verdicts.append(code_verdicts[claim_str])
        elif claim_type == "math" and claim_str in math_verdicts:
            all_verdicts.append(math_verdicts[claim_str])
        elif claim_str in factual_verdicts:
            all_verdicts.append(factual_verdicts[claim_str])
        else:
            # Safety fallback — should not happen
            all_verdicts.append(ClaimVerdict(
                claim=claim_str,
                verdict="UNVERIFIABLE",
                evidence="",
                confidence=0.3,
                verification_method="NLI",
            ))

    return all_verdicts


def _verify_single_code_claim(
    claim: str,
    llm_output: str,
    rag_output: str,
    verify_fn: Any,
) -> ClaimVerdict:
    import re as _re
    import textwrap as _tw

    # Extract code snippet from LLM output for verification
    # Strategy 1: Fenced ```python blocks
    code_block_re = _re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", _re.IGNORECASE)
    blocks = code_block_re.findall(llm_output)

    if blocks:
        code_snippet = blocks[0].strip()
    else:
        # Strategy 2: Extract bare function definitions (handles unfenced code)
        func_re = _re.compile(
            r"(def\s+\w+\s*\([^\)]*\)[^\n]*:\n(?:[ \t]+[^\n]*\n?)*)",
            _re.MULTILINE,
        )
        funcs = func_re.findall(llm_output)
        if funcs:
            # Take the longest match (likely the main function with helpers)
            code_snippet = max(funcs, key=len).strip()
        else:
            code_snippet = llm_output

    # Always dedent so code starts at column 0 (handles class-wrapped code)
    code_snippet = _tw.dedent(code_snippet)

    try:
        result = verify_fn(claim, code_snippet)
    except Exception as e:
        logger.warning("Node 5 | EGV code verification failed for claim: %s", e)
        return ClaimVerdict(
            claim=claim,
            verdict="UNVERIFIABLE",
            evidence=f"Code verification error: {e}",
            confidence=0.3,
            verification_method="EGV_CODE",
        )

    verdict_str = result.get("verdict", "UNKNOWN")
    failed_tests = result.get("failed_tests", [])

    # Map EGV verdicts to claim verdicts
    verdict_map = {
        "SUPPORTED": "SUPPORTED",
        "CONTRADICTED": "CONTRADICTED",
        "UNKNOWN": "UNVERIFIABLE",
    }

    evidence_parts = []
    for ft in failed_tests:
        evidence_parts.append(
            f"Test '{ft.get('description', '?')}': "
            f"input={ft.get('input', '?')}, "
            f"expected={ft.get('expected', '?')}, "
            f"actual={ft.get('actual', '?')}"
        )
    evidence = "; ".join(evidence_parts) if evidence_parts else "All tests passed."

    return ClaimVerdict(
        claim=claim,
        verdict=verdict_map.get(verdict_str, "UNVERIFIABLE"),
        evidence=evidence,
        confidence=1.0 if verdict_str == "SUPPORTED" else (0.9 if verdict_str == "CONTRADICTED" else 0.3),
        reasoning=f"EGV code verification: {verdict_str} ({len(failed_tests)} test(s) failed)",
        verification_method="EGV_CODE",
    )


def _verify_single_math_claim(
    claim: str,
    verify_fn: Any,
) -> ClaimVerdict:
    try:
        result = verify_fn(claim)
    except Exception as e:
        logger.warning("Node 5 | EGV math verification failed for claim: %s", e)
        return ClaimVerdict(
            claim=claim,
            verdict="UNVERIFIABLE",
            evidence=f"Math verification error: {e}",
            confidence=0.3,
            verification_method="EGV_MATH",
        )

    verdict_str = result.get("verdict", "UNKNOWN")
    computed = result.get("computed", None)

    verdict_map = {
        "SUPPORTED": "SUPPORTED",
        "CONTRADICTED": "CONTRADICTED",
        "UNKNOWN": "UNVERIFIABLE",
    }

    evidence = f"SymPy computed: {computed}" if computed else "Could not extract verifiable math."

    return ClaimVerdict(
        claim=claim,
        verdict=verdict_map.get(verdict_str, "UNVERIFIABLE"),
        evidence=evidence,
        confidence=1.0 if verdict_str == "SUPPORTED" else (0.95 if verdict_str == "CONTRADICTED" else 0.3),
        reasoning=f"EGV math verification: {verdict_str} (computed={computed})",
        verification_method="EGV_MATH",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Surgical Correction (post-processing)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_surgical_corrections(
    verdicts: List[ClaimVerdict],
    llm_output: str,
    rag_output: str,
) -> tuple:
    contradicted = [v for v in verdicts if v.verdict == "CONTRADICTED"]
    if not contradicted:
        return verdicts, llm_output

    try:
        from nodes.surgical_corrector import surgical_correct_single  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Node 5 | surgical_corrector not available, skipping corrections.")
        return verdicts, llm_output

    corrected = llm_output
    corrections = 0

    for v in contradicted:
        evidence = v.evidence if v.evidence else rag_output[:3000]
        try:
            corrected = surgical_correct_single(corrected, v.claim, evidence)
            corrections += 1
            logger.info(
                "Node 5 | Surgically corrected [%s]: '%s'",
                v.verification_method, v.claim[:60],
            )
        except Exception as e:
            logger.warning(
                "Node 5 | Surgical correction failed for '%s': %s",
                v.claim[:60], e,
            )

    if corrections:
        logger.info(
            "Node 5 | Applied %d/%d surgical corrections.",
            corrections, len(contradicted),
        )

    return verdicts, corrected
