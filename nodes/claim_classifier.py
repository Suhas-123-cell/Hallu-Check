from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger("hallu-check.claim_classifier")

ClaimType = Literal["factual", "math", "code"]

_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdef\s+\w+\s*\("),              
    re.compile(r"\breturn\s+\w"),                  
    re.compile(r"\bclass\s+\w+\s*[(:{]"),          
    re.compile(r"\bimport\s+\w+"),                 
    re.compile(r"\bfrom\s+\w+\s+import\b"),        
    re.compile(r"\bfor\s+\w+\s+in\b"),             
    re.compile(r"\bfor\s*\(.+;.+;"),               
    re.compile(r"\bwhile\s*\(.+\)\s*\{"),           
    re.compile(r"\bif\s*\(.+\)\s*\{"),             
    re.compile(r"```\w*\n"),                       
    re.compile(r"`[^`]+`"),                        
    re.compile(r"#include\s*<"),                   
    re.compile(r"\bvoid\s+\w+\s*\("),              
    re.compile(r"\bint\s+\w+\s*\("),               
    re.compile(r"\bfunction\s+\w+\s*\("),          
    re.compile(r"\bconst\s+\w+\s*=\s*\("),         
    re.compile(r"=>"),                             
    re.compile(r"\bprint\s*\("),                   
    re.compile(r"\bself\.\w+"),                    
    re.compile(r"\bnew\s+\w+\s*\("),               
    re.compile(r"\b[a-z]+_[a-z]+\("),              
    re.compile(r"\b\w+\.append\("),                
    re.compile(r"\b\w+\.sort\("),                  
    re.compile(r"\b\w+\.pop\("),                   
    re.compile(r"\b\w+\.keys\("),                  
    re.compile(r"\blen\s*\("),                     
    re.compile(r"\brange\s*\("),                   
    re.compile(r"\b(?:True|False|None)\b"),         
    re.compile(r"\braise\s+\w+"),                  
    re.compile(r"\btry\s*:"),                      
    re.compile(r"\bexcept\s+\w+"),                 
    re.compile(r"\bassert\s+\w"),                  
    re.compile(r"\blambda\s+\w"),                  
    re.compile(r"\[\s*\w+\s+for\s+\w+\s+in"),      
    re.compile(r"\bdict\s*\("),                    
    re.compile(r"\blist\s*\("),                    
    re.compile(r"\bset\s*\("),                     
    re.compile(r"\btuple\s*\("),                   
    re.compile(r"\b\w+\[\d+\]"),                   
    re.compile(r"\b\w+\[\s*:\s*\]"),                
    re.compile(r"\bTypeError\b|\bValueError\b|\bIndexError\b"),  
    re.compile(r"\b(?:str|int|float|bool)\s*\("),  
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




def _heuristic_classify(claim: str) -> ClaimType | None:
    claim_lower = claim.lower()

 
    factual_override_phrases = (
        "was invented",
        "was created",
        "was developed",
        "was introduced",
        "was born",
        "was founded",
    )
    if any(phrase in claim_lower for phrase in factual_override_phrases):
        logger.debug("claim_classifier | heuristic → factual (historical override)")
        return "factual"

   
    for pat in _CODE_PATTERNS:
        if pat.search(claim):
            logger.debug("claim_classifier | heuristic → code (pattern: %s)", pat.pattern[:30])
            return "code"

    code_hits = sum(1 for kw in _CODE_KEYWORDS if kw in claim_lower)
    if code_hits >= 2:
        logger.debug("claim_classifier | heuristic → code (%d keyword hits)", code_hits)
        return "code"

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

   
    if code_hits == 1 and math_hits == 0:
        return "code"
    if math_hits == 1 and code_hits == 0:
        return "math"

    
    _factual_indicators = [
        re.search(r"\b(?:is|was|are|were|has|had|born|founded|located)\b", claim_lower),
        re.search(r"\b\d{4}\b", claim),          
        re.search(r"(?<=\s)[A-Z][a-z]{2,}", claim),  
    ]
    if any(_factual_indicators):
        logger.debug("claim_classifier | heuristic → factual (factual indicator)")
        return "factual"

    # Not confident — fall through to Gemini
    return None




_GEMINI_PROMPT = """\
Classify the following atomic claim into exactly one category.
Respond with EXACTLY ONE WORD: factual, math, or code. Do not include any other text.

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
    prompt = _GEMINI_PROMPT.format(claim=claim.replace('"', '\\"'))

    raw = ""


    try:
        from nodes.local_llm import chat_completion, is_available  # type: ignore[import-not-found]
        if is_available():
            raw = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            ).strip().lower()
    except Exception as e:
        logger.warning("claim_classifier | Local LLM failed: %s", str(e)[:100])

    # (No Gemini fallback — API reserved for refiner only)

    if not raw:
        logger.warning("claim_classifier | All LLMs failed, defaulting to 'factual'.")
        return "factual"

    
    for label in ("code", "math", "factual"):
        if label in raw:
            logger.info("claim_classifier | LLM → %s (raw=%r)", label, raw[:40])
            return label  # type: ignore[return-value]

 
    logger.warning("claim_classifier | LLM returned unparseable %r, defaulting to 'factual'.", raw[:60])
    return "factual"



def classify_claim(claim: str) -> ClaimType:
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
