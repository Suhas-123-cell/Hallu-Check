"""
hallu-check | nodes/gatekeeper.py
Node 0 — Semantic Query Router (Gatekeeper)

Classifies incoming queries into one of three categories using a two-stage
approach:
  Stage 1: Fast heuristic pre-classifier (keyword + pattern matching)
  Stage 2: LLM-based classification via Llama 3.2-1B (only if heuristic is unsure)

Categories:
  • FACTUAL:   Requires real-world data, web search, and factual verification.
  • REASONING: Requires logic, coding, math, or text transformation.
  • CHITCHAT:  Conversational greetings or simple interactions.

The classification determines which pipeline path the query takes:
  - FACTUAL  → full pipeline (Nodes 1-7)
  - REASONING → Node 1 → Node 5 → Node 6 (skip web search)
  - CHITCHAT  → Node 1 → return immediately
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict

from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]

from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.gatekeeper")

# Valid categories — FACTUAL is the safe fallback
_VALID_CATEGORIES = {"FACTUAL", "REASONING", "CHITCHAT"}

# ── Stage 1: Heuristic pre-classifier patterns ──────────────────────────────

# Code snippets: if the query contains actual code, it's almost certainly REASONING
_CODE_PATTERNS = [
    r"#include\s*<",           # C/C++ includes
    r"\busing\s+namespace\b",  # C++ namespace
    r"\bclass\s+\w+\s*[({]",  # Class definitions
    r"\bdef\s+\w+\s*\(",      # Python function defs
    r"\bfunction\s+\w+\s*\(", # JS function defs
    r"\bvoid\s+\w+\s*\(",     # C/Java void functions
    r"\bint\s+\w+\s*\(",      # C/Java int functions
    r"\breturn\s+\w",         # return statements
    r"vector<",               # C++ STL
    r"public\s*:",            # C++ access specifiers
    r"```\w*\n",              # Markdown code blocks
    r"//\s*Write your code",  # Code placeholder comments
    r"//\s*TODO",             # TODO comments
    r"\bfor\s*\(.+;.+;",     # C-style for loops
    r"\bwhile\s*\(.+\)\s*\{", # While loops
    r"\bimport\s+\w+",       # Python/Java imports
    r"\bfrom\s+\w+\s+import", # Python imports
]

# Algorithm / competitive programming keywords
_REASONING_KEYWORDS = [
    # Problem structure
    "write a function", "write a program", "write code", "implement",
    "complete the", "your task", "write your code here",
    "sample input", "sample output", "input format", "output format",
    "constraints", "time complexity", "space complexity",
    "expected output", "test case",
    # Algorithm terms
    "algorithm", "dynamic programming", "recursion", "backtracking",
    "binary search", "breadth first", "depth first", "greedy",
    "sorting", "traversal", "maximize", "minimize", "optimal",
    "subarr", "subsequence", "permutation", "combination",
    # Math terms
    "calculate", "compute", "evaluate", "prove", "derive",
    "equation", "formula", "theorem", "factorial", "fibonacci",
    # Coding terms
    "refactor", "debug", "fix this code", "what does this code",
    "explain this code", "optimize this", "big o", "o(n)",
    "runtime", "leetcode", "hackerrank", "codeforces",
    # Logic
    "logic", "logical", "reasoning", "deduce", "infer",
    "translate this", "convert this", "transform this",
    "write a regex", "parse this",
]

# Chitchat patterns
_CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|night)|thanks|thank\s*you|bye|goodbye|ok|okay)\s*[!.?]*$",
    r"^how\s+are\s+you",
    r"^what'?s?\s+up",
    r"^yo\b",
]

# Factual query patterns — "who is X?", "what is X?", etc.
_FACTUAL_PATTERNS = [
    r"^who\s+(is|was|are|were)\b",
    r"^what\s+(is|was|are|were)\s+the\b",
    r"^where\s+(is|was|are|were)\b",
    r"^when\s+(did|was|is|were)\b",
    r"^how\s+(many|much|old|tall|far|long)\b",
    r"^(tell\s+me\s+about|describe)\s+",
]

# Factual keywords — presence of any strongly suggests FACTUAL
_FACTUAL_KEYWORDS = [
    "capital of", "president of", "prime minister", "chief minister",
    "population of", "founded in", "born in", "died in",
    "latest news", "current", "who won", "who lost",
]


def _heuristic_classify(query: str) -> Dict[str, object] | None:
    """
    Stage 1: Fast heuristic pre-classifier.

    Returns a classification dict if confident, or None to fall through
    to the LLM-based classifier.
    """
    query_lower = query.lower().strip()

    # ── Check CHITCHAT first (short, simple greetings) ────────────────
    if len(query_lower) < 50:
        for pattern in _CHITCHAT_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                logger.info("Node 0 | Heuristic → CHITCHAT (greeting pattern)")
                return {"category": "CHITCHAT", "confidence": 0.95}

    # ── Check FACTUAL patterns (common factual question structures) ───
    for pattern in _FACTUAL_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            logger.info("Node 0 | Heuristic → FACTUAL (question pattern: %s)", pattern[:30])
            return {"category": "FACTUAL", "confidence": 0.95}

    # Check for factual keywords
    if any(kw in query_lower for kw in _FACTUAL_KEYWORDS):
        logger.info("Node 0 | Heuristic → FACTUAL (keyword match)")
        return {"category": "FACTUAL", "confidence": 0.90}

    # ── Check for code snippets (strong signal for REASONING) ─────────
    for pattern in _CODE_PATTERNS:
        if re.search(pattern, query, re.MULTILINE):
            logger.info("Node 0 | Heuristic → REASONING (code pattern: %s)", pattern[:30])
            return {"category": "REASONING", "confidence": 0.95}

    # ── Check for algorithm/coding keywords ───────────────────────────
    keyword_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in query_lower)
    if keyword_hits >= 3:
        logger.info("Node 0 | Heuristic → REASONING (%d keyword hits)", keyword_hits)
        return {"category": "REASONING", "confidence": 0.90}

    # ── Long queries with structured problem format ───────────────────
    # (e.g., "Input:", "Output:", "Example:", "Constraints:" sections)
    structure_markers = ["input:", "output:", "example", "constraints", "sample"]
    structure_hits = sum(1 for m in structure_markers if m in query_lower)
    if structure_hits >= 3:
        logger.info("Node 0 | Heuristic → REASONING (%d structure markers)", structure_hits)
        return {"category": "REASONING", "confidence": 0.90}

    # Not confident enough — fall through to LLM
    return None


_CLASSIFY_SYSTEM_PROMPT = """\
You are a query classifier. Your ONLY job is to classify the user's query into exactly one category.

Categories:
- FACTUAL: The query asks about real-world facts, people, places, events, dates, statistics, current news, or anything that requires external data to answer accurately.
  Examples: "Who is the President of India?", "What is the population of Tokyo?", "Latest news about AI regulation"

- REASONING: The query asks for logic, math, coding, text transformation, comparisons, explanations of concepts, or creative writing that does NOT require real-world data lookup.
  Examples: "Write a Python function to sort a list", "Explain how recursion works", "What is 2^10?", "Translate this to French"

- CHITCHAT: The query is a greeting, small talk, or simple conversational interaction that needs no factual or logical processing.
  Examples: "Hello!", "How are you?", "Thanks!", "Good morning"

Respond with ONLY this JSON (no other text):
{"category": "<FACTUAL|REASONING|CHITCHAT>", "confidence": <0.0-1.0>}"""


def classify_query(query: str) -> Dict[str, object]:
    """
    Node 0 — Classify the user's query for intelligent pipeline routing.

    Uses a two-stage approach:
      1. Fast heuristic pre-classifier (patterns + keywords)
      2. LLM-based fallback via Llama 3.2-1B (only if heuristic is unsure)

    Falls back to FACTUAL (safest path) if all classification fails.

    Args:
        query: The user's raw query string.

    Returns:
        Dict with "category" (str) and "confidence" (float).
        Example: {"category": "FACTUAL", "confidence": 0.92}
    """
    logger.info("Node 0 | Classifying query: %r", query[:100])

    # ── Stage 1: Heuristic pre-classifier ─────────────────────────────
    heuristic_result = _heuristic_classify(query)
    if heuristic_result is not None:
        logger.info(
            "Node 0 | Classification (heuristic): %s (confidence: %.2f)",
            heuristic_result["category"],
            heuristic_result["confidence"],
        )
        return heuristic_result

    # ── Stage 2: LLM-based classification ─────────────────────────────
    if not HF_API_TOKEN:
        logger.warning("Node 0 | HF_API_TOKEN not set, defaulting to FACTUAL.")
        return {"category": "FACTUAL", "confidence": 0.0}

    # Truncate query for the small LLM — it doesn't need the full
    # problem statement, just enough context to classify.
    query_for_llm = query[:500]
    if len(query) > 500:
        query_for_llm += "\n[... truncated ...]"

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": query_for_llm},
            ],
            max_tokens=60,
            temperature=0.1,
        )

        raw = ""
        if response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content.strip()

        if not raw:
            logger.warning("Node 0 | Empty response from LLM, defaulting to FACTUAL.")
            return {"category": "FACTUAL", "confidence": 0.0}

        # ── Parse the JSON response ──────────────────────────────────────
        parsed = _parse_classification(raw)
        logger.info(
            "Node 0 | Classification (LLM): %s (confidence: %.2f)",
            parsed["category"],
            parsed["confidence"],
        )
        return parsed

    except Exception as e:
        logger.warning("Node 0 | Classification failed (%s), defaulting to FACTUAL.", e)
        return {"category": "FACTUAL", "confidence": 0.0}


def _parse_classification(raw: str) -> Dict[str, object]:
    """
    Parse the LLM's classification response. Tries strict JSON first,
    then falls back to regex extraction. Always returns a valid dict.
    """
    # Attempt 1: Direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "category" in parsed:
            category = str(parsed["category"]).upper()
            confidence = float(parsed.get("confidence", 0.5))
            if category in _VALID_CATEGORIES:
                return {"category": category, "confidence": confidence}
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: Extract from markdown code block
    json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            if isinstance(parsed, dict) and "category" in parsed:
                category = str(parsed["category"]).upper()
                confidence = float(parsed.get("confidence", 0.5))
                if category in _VALID_CATEGORIES:
                    return {"category": category, "confidence": confidence}
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: Regex fallback — look for category keyword in raw text
    raw_upper = raw.upper()
    for cat in ("CHITCHAT", "REASONING", "FACTUAL"):
        if cat in raw_upper:
            logger.debug("Node 0 | Extracted category via regex fallback: %s", cat)
            return {"category": cat, "confidence": 0.4}

    # Final fallback: FACTUAL is safest (triggers full verification)
    logger.warning("Node 0 | Could not parse classification, defaulting to FACTUAL.")
    return {"category": "FACTUAL", "confidence": 0.0}

