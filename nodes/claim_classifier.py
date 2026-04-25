"""
hallu-check | nodes/claim_classifier.py
Claim-Level Type Classifier

Classifies an individual atomic claim into one of three types so the
correct verification pathway is used:

  • factual  →  NLI-based verification against RAG evidence
  • math     →  Symbolic / numeric execution verification
  • code     →  Sandbox execution verification (EGV)

Two-stage approach (mirrors gatekeeper.py's pattern):
  Stage 1:  Fast heuristic (regex + keyword matching) — ~0 ms, no API cost.
  Stage 2:  One-shot Gemini call — only fired when heuristics are ambiguous.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger("hallu-check.claim_classifier")

ClaimType = Literal["factual", "math", "code"]

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Heuristic patterns
# ─────────────────────────────────────────────────────────────────────────────

# ── Code indicators ──────────────────────────────────────────────────────────
# Patterns that, if present in a claim, almost certainly make it a code claim.
_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdef\s+\w+\s*\("),              # Python function def
    re.compile(r"\breturn\s+\w"),                  # return statement
    re.compile(r"\bclass\s+\w+\s*[(:{]"),          # class definition
    re.compile(r"\bimport\s+\w+"),                 # import statement
    re.compile(r"\bfrom\s+\w+\s+import\b"),        # from … import
    re.compile(r"\bfor\s+\w+\s+in\b"),             # Python for-in loop
    re.compile(r"\bfor\s*\(.+;.+;"),               # C-style for loop
    re.compile(r"\bwhile\s*\(.+\)\s*\{"),          # C/Java while loop
    re.compile(r"\bif\s*\(.+\)\s*\{"),             # C/Java if block
    re.compile(r"```\w*\n"),                       # Markdown code fence
    re.compile(r"`[^`]+`"),                        # Inline code (backtick)
    re.compile(r"#include\s*<"),                   # C/C++ include
    re.compile(r"\bvoid\s+\w+\s*\("),              # C/Java void function
    re.compile(r"\bint\s+\w+\s*\("),               # C/Java int function
    re.compile(r"\bfunction\s+\w+\s*\("),          # JS function
    re.compile(r"\bconst\s+\w+\s*=\s*\("),         # JS arrow function
    re.compile(r"=>"),                             # Arrow operator
    re.compile(r"\bprint\s*\("),                   # print call
    re.compile(r"\bself\.\w+"),                    # Python self reference
    re.compile(r"\bnew\s+\w+\s*\("),               # new Object()
    re.compile(r"\b[a-z]+_[a-z]+\("),              # snake_case function call
    re.compile(r"\b\w+\.append\("),                # .append() method call
    re.compile(r"\b\w+\.sort\("),                  # .sort() method call
    re.compile(r"\b\w+\.pop\("),                   # .pop() method call
    re.compile(r"\b\w+\.keys\("),                  # .keys() method call
    re.compile(r"\blen\s*\("),                     # len() call
    re.compile(r"\brange\s*\("),                   # range() call
    re.compile(r"\b(?:True|False|None)\b"),         # Python literals
    re.compile(r"\braise\s+\w+"),                  # raise Exception
    re.compile(r"\btry\s*:"),                      # try block
    re.compile(r"\bexcept\s+\w+"),                 # except clause
    re.compile(r"\bassert\s+\w"),                  # assert statement
    re.compile(r"\blambda\s+\w"),                  # lambda expression
    re.compile(r"\[\s*\w+\s+for\s+\w+\s+in"),      # list comprehension
    re.compile(r"\bdict\s*\("),                    # dict() call
    re.compile(r"\blist\s*\("),                    # list() call
    re.compile(r"\bset\s*\("),                     # set() call
    re.compile(r"\btuple\s*\("),                   # tuple() call
    re.compile(r"\b\w+\[\d+\]"),                   # array indexing: arr[0]
    re.compile(r"\b\w+\[\s*:\s*\]"),                # slice: arr[:]
    re.compile(r"\bTypeError\b|\bValueError\b|\bIndexError\b"),  # exception types
    re.compile(r"\b(?:str|int|float|bool)\s*\("),  # type casting
]

_CODE_KEYWORDS: list[str] = [
    "algorithm", "function", "binary search", "linked list",
    "hash map", "hash table", "stack", "queue", "tree traversal",
    "depth-first", "breadth-first", "recursion", "recursive",
    "dynamic programming", "time complexity", "space complexity",
    "O(1)", "O(n)", "O(n^2)", "O(log n)", "O(n log n)",
    "big-o", "big o", "runtime complexity",
    "array", "pointer", "iterator", "loop invariant",
    "sorting algorithm", "merge sort", "quick sort", "bubble sort",
    "heap sort", "insertion sort", "selection sort",
    "null pointer", "segfault", "syntax error", "compiler",
    "API endpoint", "REST API", "HTTP request",
    # Additional HumanEval-relevant patterns
    "returns", "input", "output", "parameter", "argument",
    "string", "substring", "character", "index",
    "list", "dictionary", "tuple", "boolean",
    "empty list", "empty string", "edge case",
    "helper function", "utility function", "wrapper",
    "data structure", "implementation", "method",
    "exception", "error handling", "validation",
    "variable", "constant", "global", "local",
    "iterate", "traversal", "loop", "nested loop",
    "base case", "recursive call", "memoization",
    "palindrome", "anagram", "fibonacci", "prime",
    "binary", "decimal", "hexadecimal", "bitwise",
    "slice", "concatenate", "reverse", "flatten",
    "filter", "map", "reduce", "lambda",
    "type hint", "annotation", "docstring",
]

# ── Math indicators ──────────────────────────────────────────────────────────
_MATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d+\s*[+\-*/^%]\s*\d+"),          # 5 + 3, 2^10
    re.compile(r"\d+\s*\*\*\s*\d+"),                # 2 ** 10
    re.compile(r"\d+!"),                             # 5!
    re.compile(r"(?:sqrt|log|sin|cos|tan)\s*\("),    # math functions
    re.compile(r"∫|∑|∏|∂|∇|√|±|≈|≠|≤|≥|∞"),        # math symbols (unicode)
    re.compile(r"\\(?:int|sum|prod|frac|sqrt)\b"),   # LaTeX math commands
    re.compile(r"\bx\s*[=<>]\s*[-+]?\d"),            # x = 5, x > 3
    re.compile(r"\b\d+\s*=\s*\d+\s*[+\-*/]"),       # 10 = 5 + 5
    re.compile(r"\blim\s*_"),                        # lim_
    re.compile(r"\bdy/dx\b|\bdx\b"),                 # derivatives
    re.compile(r"\d+\s*mod\s*\d+", re.IGNORECASE),   # modular arithmetic
]

_MATH_KEYWORDS: list[str] = [
    "equals", "equal to", "sum of", "product of", "difference of",
    "equation", "formula", "theorem", "proof", "corollary", "lemma",
    "derivative", "integral", "integration", "differentiation",
    "factorial", "fibonacci", "prime number", "prime factorization",
    "greatest common divisor", "gcd", "lcm", "least common multiple",
    "logarithm", "exponential", "exponent", "power of",
    "quadratic", "polynomial", "linear equation", "matrix",
    "determinant", "eigenvalue", "eigenvector",
    "probability", "permutation", "combination",
    "arithmetic", "geometric", "sequence", "series",
    "modular arithmetic", "congruent", "modulo",
    "calculate", "compute", "evaluate", "solve for",
    "square root", "cube root", "nth root",
    "numerator", "denominator", "fraction",
    "pi", "euler", "infinity",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 entry point
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_classify(claim: str) -> ClaimType | None:
    """
    Fast, zero-cost heuristic classification of a single claim.

    Returns the label if confident, or ``None`` to fall through to Gemini.
    """
    # ── Code check (highest priority — code often embeds math) ────────
    for pat in _CODE_PATTERNS:
        if pat.search(claim):
            logger.debug("claim_classifier | heuristic → code (pattern: %s)", pat.pattern[:30])
            return "code"

    claim_lower = claim.lower()

    code_hits = sum(1 for kw in _CODE_KEYWORDS if kw in claim_lower)
    if code_hits >= 2:
        logger.debug("claim_classifier | heuristic → code (%d keyword hits)", code_hits)
        return "code"

    # ── Math check ────────────────────────────────────────────────────
    for pat in _MATH_PATTERNS:
        if pat.search(claim):
            logger.debug("claim_classifier | heuristic → math (pattern: %s)", pat.pattern[:30])
            return "math"

    math_hits = sum(
        1 for kw in _MATH_KEYWORDS
        if re.search(r'\b' + re.escape(kw) + r'\b', claim_lower)
    )
    if math_hits >= 2:
        logger.debug("claim_classifier | heuristic → math (%d keyword hits)", math_hits)
        return "math"

    # ── Single-keyword tiebreaker (weaker signal, but still useful) ───
    if code_hits == 1 and math_hits == 0:
        return "code"
    if math_hits == 1 and code_hits == 0:
        return "math"

    # ── Factual shortcut: if the claim looks like a plain factual
    #    statement with no code/math signal at all, skip the LLM call.
    #    Heuristic: contains a proper noun (capitalized word not at start)
    #    or a date/number used in a factual context.
    _factual_indicators = [
        re.search(r"\b(?:is|was|are|were|has|had|born|founded|located)\b", claim_lower),
        re.search(r"\b\d{4}\b", claim),          # year-like number
        re.search(r"(?<=\s)[A-Z][a-z]{2,}", claim),  # proper noun (mid-sentence cap)
    ]
    if any(_factual_indicators):
        logger.debug("claim_classifier | heuristic → factual (factual indicator)")
        return "factual"

    # Not confident — fall through to Gemini
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Gemini fallback (one-shot, ~0.3 s)
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_PROMPT = """\
Classify the following atomic claim into exactly one category.

Categories:
- **factual**: A real-world factual statement about people, places, events, dates, \
statistics, definitions, or any assertion that can be verified against external knowledge.
- **math**: A mathematical expression, equation, computation, numeric result, or \
statement about mathematical properties (e.g., "2^10 = 1024", "the derivative of x² is 2x").
- **code**: A programming-related statement, code snippet, algorithmic claim, or \
assertion about software behavior (e.g., "binary search runs in O(log n)", \
"the function returns -1 when not found").

Example:
Claim: "The time complexity of merge sort is O(n log n)."
Label: code

Claim: "{claim}"
Label:"""


def _gemini_classify(claim: str) -> ClaimType:
    """
    Stage 2: Classify via a single Gemini call.

    Re-uses the existing Gemini helper from claim_verifier to honour
    rate-limit retries and SSL workarounds already in place.
    """
    try:
        from nodes.claim_verifier import _gemini_generate  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("claim_classifier | Cannot import _gemini_generate, defaulting to 'factual'.")
        return "factual"

    prompt = _GEMINI_PROMPT.format(claim=claim.replace('"', '\\"'))
    raw = _gemini_generate(prompt).strip().lower()

    # Parse — accept the first valid label found in the response
    for label in ("code", "math", "factual"):
        if label in raw:
            logger.info("claim_classifier | Gemini → %s (raw=%r)", label, raw[:40])
            return label  # type: ignore[return-value]

    # If Gemini returned garbage, default to factual (safest path — triggers NLI)
    logger.warning("claim_classifier | Gemini returned unparseable %r, defaulting to 'factual'.", raw[:60])
    return "factual"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_claim(claim: str) -> ClaimType:
    """
    Classify a single atomic claim into ``"factual"``, ``"math"``, or ``"code"``.

    Uses a two-stage approach:
      1. **Heuristic** — regex + keyword matching (~0 ms, zero API cost).
         Covers the vast majority of claims.
      2. **Gemini fallback** — one-shot prompt, only when heuristics are
         ambiguous.  Re-uses the rate-limit-aware helper from claim_verifier.

    Args:
        claim: A single atomic claim string (as returned by the claim extractor).

    Returns:
        One of ``"factual"``, ``"math"``, ``"code"``.
    """
    if not claim or not claim.strip():
        return "factual"

    # Stage 1: heuristic
    result = _heuristic_classify(claim)
    if result is not None:
        logger.info("claim_classifier | '%s…' → %s (heuristic)", claim[:50], result)
        return result

    # Stage 2: Gemini fallback
    logger.info("claim_classifier | '%s…' → ambiguous, falling back to Gemini.", claim[:50])
    result = _gemini_classify(claim)
    logger.info("claim_classifier | '%s…' → %s (gemini)", claim[:50], result)
    return result
