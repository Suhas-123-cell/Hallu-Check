"""
hallu-check | nodes/refiner.py
Node 6 — Prompt Refiner & Reprompt LLM (Gemini)

If hallucination is detected (Node 5), this module creates a Refined
Prompt that includes the RAG-retrieved ground truth and sends it to
Gemini to generate a corrected, factual response.

Supports route-aware prompting:
  • FACTUAL  → strict factual editor prompt (fix claims with evidence)
  • REASONING → Senior Staff SWE / technical tutor prompt (evaluate logic & code)

Includes rate-limit-aware retry logic for the Gemini free tier.
"""
from __future__ import annotations

import logging
import re
import time

import google.genai as genai  
from google.genai import types as genai_types  
from config import GEMINI_API_KEY, GEMINI_MODEL  
from nodes.generator import generate_llm_output_with_context  

logger = logging.getLogger("hallu-check.refiner")

# Maximum retries for rate-limited Gemini calls
_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 15  # seconds

# ── Gemini generation config (scaled for large inputs/outputs) ───────────────
_GEMINI_HTTP_OPTIONS = genai_types.HttpOptions(timeout=120_000)  # 120s (in ms) for massive prompts
_GEMINI_GENERATE_CONFIG = genai_types.GenerateContentConfig(
    max_output_tokens=8192,
    http_options=_GEMINI_HTTP_OPTIONS,
)


def _parse_retry_delay(error_msg: str) -> float:
    """Extract retryDelay seconds from a Gemini 429 error message."""
    # Match patterns like "retryDelay': '31s'" or "retryDelay': '537.373052ms'"
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)(s|ms)", str(error_msg))
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value / 1000 if unit == "ms" else value
    return _DEFAULT_RETRY_DELAY


def _gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file.\n"
            "Get one free at https://aistudio.google.com/app/apikey"
        )
    # Workaround for macOS Python SSL Certificate / gRPC DNS resolution issues
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
                    delay = _parse_retry_delay(error_str)
                    delay = min(delay + 1, 60)
                else:
                    delay = 5  # short delay before timeout retry
                logger.warning(
                    "Node 6 | Gemini %s (attempt %d/%d), "
                    "waiting %.1fs before retry…",
                    "rate-limited" if is_rate_limit else "timed out",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            # Non-rate-limit error, or final attempt exhausted
            logger.warning(
                "Node 6 | Gemini call failed (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, e,
            )
            return ""

    return ""


def refine_response(query: str, rag_output: str) -> str:
    """
    Node 6 — Use Gemini to produce a corrected, factual answer based on
    RAG-retrieved context.

    Previously used a two-step approach (Gemini rewrites prompt → Llama answers)
    but the small Llama model almost always outputs "Not found in context."
    Now Gemini directly synthesizes the answer from the RAG context.

    Args:
        query:      The user's original question.
        rag_output: Factual context from PageIndex RAG (Node 4).

    Returns:
        The refined, corrected answer — or empty string if Gemini is
        unavailable (caller should fall back to original LLM output).
    """
    logger.info("Node 6 | Refining answer with Gemini…")

    prompt = (
        "You are a factual question-answering assistant.\n"
        "Answer the user's question using ONLY the context provided below.\n"
        "Be concise and direct. If the context contains relevant information,\n"
        "synthesize a clear answer from it.\n"
        "CRITICAL: Do NOT introduce any facts, names, dates, or claims that are\n"
        "not explicitly stated in the context below. Do NOT use your own knowledge.\n"
        "If the original LLM answer was about a COMPLETELY DIFFERENT topic than the\n"
        "question (e.g., the question asks about BPE but the answer discusses\n"
        "Backpropagation), IGNORE the original answer entirely and write a fresh\n"
        "answer using ONLY the context below.\n"
        "If the context truly has NO relevant information at all, summarize\n"
        "what the context does contain and note the specific gap.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{rag_output[:12000]}\n\n"
        "Answer:"
    )

    answer = _gemini_generate(prompt)
    if not answer:
        logger.warning(
            "Node 6 | Gemini unavailable (rate limit or error). "
            "Skipping refinement — will return original LLM output."
        )
        return ""

    logger.info("Node 6 | Refined answer received (%d chars).", len(answer))
    return answer


# ── Route-aware system prompts ────────────────────────────────────────────────

def _build_factual_prompt(
    query: str,
    rag_output: str,
    claim_report: dict,
) -> str:
    """Build the strict factual editor prompt for FACTUAL-routed queries."""
    return (
        "You are a strict factual editor. Your ONLY job is to produce a corrected "
        "answer using EXCLUSIVELY the evidence provided below.\n\n"
        "CRITICAL RULES — VIOLATIONS ARE UNACCEPTABLE:\n"
        "• You MUST NOT introduce ANY facts, names, dates, numbers, or claims "
        "that are not EXPLICITLY stated in the Evidence below.\n"
        "• You MUST NOT use your own training knowledge to fill gaps.\n"
        "• If the Evidence does not contain the answer to the query, say: "
        "'Based on the available evidence: [summarize what IS known from evidence]. "
        "The specific answer to [aspect] was not found in the retrieved sources.'\n"
        "• Remove all CONTRADICTED claims from the original answer.\n"
        "• Keep all SUPPORTED claims.\n"
        "• For UNVERIFIABLE claims: remove them — do NOT replace with guesses.\n"
        "• If the original answer is about a COMPLETELY DIFFERENT topic than the query\n"
        "  (e.g., query asks about BPE but the answer discusses Backpropagation),\n"
        "  IGNORE the original answer entirely and write a FRESH answer using\n"
        "  ONLY the Evidence below.\n\n"
        f"Original Query: {query}\n\n"
        f"Evidence (USE ONLY THIS — nothing else):\n{rag_output[:12000]}\n\n"
        f"Verification Report (which claims are true/false/unknown):\n{claim_report}\n\n"
        "Output format:\n"
        "1. Write a corrected, factual answer using ONLY the evidence above.\n"
        "2. List corrections under a '📝 Corrections:' header.\n"
        "3. End with one engaging follow-up question related to the topic.\n\n"
        "REMEMBER: If the evidence doesn't contain a specific fact, DO NOT invent it."
    )


def _build_reasoning_prompt(
    query: str,
    llm_output: str,
    claim_report: dict,
) -> str:
    """Build the Senior Staff SWE / technical tutor prompt for REASONING-routed queries."""
    return (
        "You are a Senior Staff Software Engineer and an engaging technical tutor.\n\n"
        "Your task is to evaluate the provided code or logic solution, improve it, "
        "and then challenge the user to deepen their understanding.\n\n"
        f"Here is the Original Query: {query}\n"
        f"Here is the User's Generated Answer / Code:\n{llm_output}\n"
        f"Here is the Verification Report: {claim_report}\n\n"
        "Follow these strict rules:\n"
        "1. Evaluate the provided code or logic for:\n"
        "   a. **Correctness**: Are there logical bugs, off-by-one errors, or wrong outputs?\n"
        "   b. **Edge Cases**: Does it handle empty inputs, negative numbers, large inputs, etc.?\n"
        "   c. **Time & Space Complexity**: State the Big-O complexity and whether it can be improved.\n"
        "2. Provide the **optimized, correct solution** with clear inline comments.\n"
        "3. If the original solution was already correct and optimal, acknowledge that "
        "and explain *why* it works well.\n"
        "4. List any corrections or improvements you made under a '📝 Corrections:' header.\n"
        "5. End your entire response with a single, engaging follow-up question "
        "to test the user's understanding of the algorithm or logic.\n"
        "   For example: ask about DP state transitions, edge cases they might have missed, "
        "or how they would adapt the solution to a harder variant.\n\n"
        "Make the follow-up question thought-provoking and directly related to the query."
    )


def refine_with_evidence(
    query: str,
    rag_output: str,
    claim_report: dict,
    route: str = "FACTUAL",
) -> str:
    """
    Node 6 (Enhanced) — Use Gemini to produce a corrected answer by fixing
    only the claims that are CONTRADICTED or UNVERIFIABLE.

    Dynamically selects the Gemini system prompt based on the gatekeeper route:
      • FACTUAL  → strict factual editor (fix claims using RAG evidence)
      • REASONING → Senior Staff SWE tutor (evaluate code/logic, optimize)

    Args:
        query:        The user's original question.
        rag_output:   Factual context from PageIndex RAG (Node 4), or a
                      synthetic placeholder for REASONING queries.
        claim_report: The HallucinationReport.to_dict() output.
        route:        The gatekeeper classification — "FACTUAL" or "REASONING".

    Returns:
        The refined, corrected answer — or empty string if Gemini
        is unavailable (caller should fall back to original LLM output).
    """
    logger.info("Node 6 | Evidence-based refinement (route=%s)…", route)

    # ── Select prompt based on gatekeeper route ──────────────────────
    if route == "REASONING":
        # For reasoning queries, rag_output is typically a placeholder;
        # the llm_output is embedded in the claim_report's original text.
        llm_output = claim_report.get("original_output", rag_output)
        prompt = _build_reasoning_prompt(query, llm_output, claim_report)
    else:
        # FACTUAL (default — also catches any unknown routes)
        prompt = _build_factual_prompt(query, rag_output, claim_report)

    answer = _gemini_generate(prompt)
    if not answer:
        logger.warning(
            "Node 6 | Gemini unavailable. Falling back to basic refinement."
        )
        # Fall back to basic refine_response
        return refine_response(query, rag_output)

    logger.info("Node 6 | Evidence-based refinement received (%d chars).", len(answer))
    return answer
