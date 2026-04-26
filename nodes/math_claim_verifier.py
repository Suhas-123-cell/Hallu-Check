"""
hallu-check | nodes/math_claim_verifier.py
Claim-Level Math Verification via SymPy

Verifies a math-related claim by:
  1. Asking Gemini to extract the equation / numerical assertion as a
     SymPy-parseable string pair (lhs, rhs) or a single expression.
  2. Evaluating it symbolically with ``sympy.simplify`` and, when
     applicable, ``sympy.solve``.
  3. Returning a structured verdict dict.

This is a *claim*-level verifier (operates on a single extracted claim),
complementing the existing ``execution_verifier.verify_math`` which
operates on full LLM output with regex extraction.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("hallu-check.math_claim_verifier")

# ── Subprocess timeout for SymPy evaluation ──────────────────────────────────
_EVAL_TIMEOUT = 10  # seconds — symbolic simplification can be slow


# ─────────────────────────────────────────────────────────────────────────────
# LLM — extract equation / assertion (local Ollama, Gemini fallback)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are a math extraction system. Given a natural-language math claim, extract \
the mathematical content as SymPy-parseable Python expressions.

Rules:
- Use Python/SymPy syntax: ** for exponents, sqrt() for square root, \
factorial() for factorials, pi for π, E for Euler's number, oo for infinity.
- For equations/equalities (e.g. "2^10 = 1024"), return BOTH sides.
- For derivative/integral assertions, express them as SymPy calls: \
diff(x**2, x) for "the derivative of x² is 2x".
- For numeric computations (e.g. "5! = 120"), return both the expression and \
the claimed value.
- Use Symbol('x'), Symbol('y') etc. for variables. You may assume \
"from sympy import *" is available.
- If the claim contains no verifiable math, set "extractable" to false.

Claim: "{claim}"

Respond with ONLY this JSON (no markdown fences, no extra text):
{{
  "extractable": true,
  "expression": "<SymPy expression for the left-hand side or the computation>",
  "claimed_value": "<SymPy expression for the right-hand side or claimed result>",
  "verification_type": "<one of: equality, simplification, derivative, integral, solve, numeric>"
}}

If the claim has no verifiable math:
{{
  "extractable": false,
  "expression": "",
  "claimed_value": "",
  "verification_type": ""
}}"""


def _extract_math_from_claim(claim: str) -> Optional[Dict[str, str]]:
    """
    Extract the mathematical content from a claim using local LLM (Ollama).

    Falls back to Gemini if Ollama is unavailable.
    Returns a dict with keys: extractable, expression, claimed_value,
    verification_type.  Returns None if all LLMs fail.
    """
    prompt = _EXTRACT_PROMPT.format(claim=claim[:500])

    raw = ""

    # ── Primary: Local LLM via Ollama ────────────────────────────────────
    try:
        from nodes.local_llm import chat_completion, is_available  # type: ignore[import-not-found]
        if is_available():
            raw = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise math extraction system. "
                            "Always respond with ONLY valid JSON. "
                            "No explanations, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.0,
            )
    except Exception as e:
        logger.warning("math_claim_verifier | Local LLM failed: %s", str(e)[:100])

    # ── Fallback: Gemini ─────────────────────────────────────────────────
    if not raw:
        try:
            from nodes.claim_verifier import _gemini_generate  # type: ignore[import-not-found]
            raw = _gemini_generate(prompt)
        except Exception as e:
            logger.warning("math_claim_verifier | Gemini fallback failed: %s", str(e)[:100])

    if not raw:
        logger.warning("math_claim_verifier | All LLMs returned empty response.")
        return None

    # ── Parse JSON ───────────────────────────────────────────────────
    for candidate in [raw.strip(), _extract_json_block(raw)]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "extractable" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    logger.warning("math_claim_verifier | Failed to parse LLM extraction response.")
    return None


def _extract_json_block(text: str) -> str:
    """Extract JSON from a markdown fence or find a bare JSON object."""
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else ""


# ─────────────────────────────────────────────────────────────────────────────
# SymPy evaluation via subprocess (never eval/exec in-process)
# ─────────────────────────────────────────────────────────────────────────────

_SYMPY_VERIFY_TEMPLATE = """\
import json
from sympy import *

x, y, z, n, k, m, t = symbols('x y z n k m t')

try:
    expr = {expression}
    claimed = {claimed_value}
    vtype = {verification_type!r}

    result = {{"ok": True}}

    if vtype == "derivative":
        # expr is the derivative call, claimed is the expected result
        computed = simplify(expr)
        claimed_simplified = simplify(claimed)
        matches = simplify(computed - claimed_simplified) == 0
        result["computed"] = str(computed)
        result["matches"] = matches

    elif vtype == "integral":
        computed = simplify(expr)
        claimed_simplified = simplify(claimed)
        # Integrals may differ by a constant — check the derivative
        diff_check = simplify(diff(computed - claimed_simplified, x))
        matches = diff_check == 0
        result["computed"] = str(computed)
        result["matches"] = matches

    elif vtype == "solve":
        computed = solve(expr)
        claimed_val = claimed
        if isinstance(claimed_val, (list, tuple)):
            matches = set(computed) == set(claimed_val)
        else:
            matches = claimed_val in computed
        result["computed"] = str(computed)
        result["matches"] = matches

    elif vtype == "numeric":
        computed = simplify(expr)
        claimed_simplified = simplify(claimed)
        # Try numeric evaluation
        try:
            computed_n = float(computed.evalf())
            claimed_n = float(claimed_simplified.evalf())
            matches = abs(computed_n - claimed_n) < 1e-9
            result["computed"] = str(computed_n)
        except (TypeError, ValueError):
            matches = simplify(computed - claimed_simplified) == 0
            result["computed"] = str(computed)
        result["matches"] = matches

    else:
        # equality / simplification — default path
        computed = simplify(expr)
        claimed_simplified = simplify(claimed)
        try:
            # Try numeric comparison first (faster, handles floats)
            computed_n = float(computed.evalf())
            claimed_n = float(claimed_simplified.evalf())
            matches = abs(computed_n - claimed_n) < 1e-9
            result["computed"] = str(computed_n)
        except (TypeError, ValueError, AttributeError):
            # Fall back to symbolic comparison
            matches = simplify(computed - claimed_simplified) == 0
            result["computed"] = str(computed)
        result["matches"] = matches

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)[:300]}}))
"""


def _run_sympy_verification(
    expression: str,
    claimed_value: str,
    verification_type: str,
) -> Dict[str, Any]:
    """
    Run SymPy verification in an isolated subprocess.

    Returns a dict with keys: ok, computed (str), matches (bool), error (str).
    """
    script = _SYMPY_VERIFY_TEMPLATE.format(
        expression=expression,
        claimed_value=claimed_value,
        verification_type=verification_type,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="hallu_mathverify_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=_EVAL_TIMEOUT,
                cwd=tmpdir,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"SymPy evaluation timed out after {_EVAL_TIMEOUT}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": stderr[:300] if stderr else f"exit code {proc.returncode}",
        }

    # Parse JSON output
    try:
        result = json.loads(stdout)
        return result
    except json.JSONDecodeError:
        return {"ok": False, "error": f"Unparseable SymPy output: {stdout[:200]}"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def verify_math_claim(claim: str) -> Dict[str, Any]:
    """
    Verify a math-related claim using LLM extraction + SymPy evaluation.

    Pipeline:
      1. Local LLM (Ollama) extracts the equation / assertion as SymPy strings.
      2. SymPy evaluates and simplifies in an isolated subprocess.
      3. The computed result is compared against the claimed value.

    Args:
        claim: A single atomic math claim (e.g., "2^10 = 1024",
               "the derivative of x² is 2x", "5! = 120").

    Returns:
        Dict with:
          - ``"verdict"``: ``"SUPPORTED"``, ``"CONTRADICTED"``, or ``"UNKNOWN"``
          - ``"computed"``: The SymPy-computed result (str), or ``None``
    """
    if not claim or not claim.strip():
        return {"verdict": "UNKNOWN", "computed": None}

    # ── Step 1: Extract math via local LLM ────────────────────────────
    extraction = _extract_math_from_claim(claim)
    if extraction is None:
        logger.warning("math_claim_verifier | Gemini extraction failed.")
        return {"verdict": "UNKNOWN", "computed": None}

    if not extraction.get("extractable", False):
        logger.info("math_claim_verifier | No verifiable math in claim.")
        return {"verdict": "UNKNOWN", "computed": None}

    expression = extraction.get("expression", "").strip()
    claimed_value = extraction.get("claimed_value", "").strip()
    verification_type = extraction.get("verification_type", "equality").strip()

    if not expression or not claimed_value:
        logger.warning(
            "math_claim_verifier | Incomplete extraction: expr=%r, claimed=%r",
            expression, claimed_value,
        )
        return {"verdict": "UNKNOWN", "computed": None}

    logger.info(
        "math_claim_verifier | Extracted: expr=%r, claimed=%r, type=%s",
        expression[:80], claimed_value[:80], verification_type,
    )

    # ── Step 2: Evaluate with SymPy ───────────────────────────────────
    result = _run_sympy_verification(expression, claimed_value, verification_type)

    if not result.get("ok", False):
        error = result.get("error", "unknown error")
        logger.warning("math_claim_verifier | SymPy evaluation failed: %s", error[:200])
        return {"verdict": "UNKNOWN", "computed": None}

    computed = result.get("computed", "?")
    matches = result.get("matches", False)

    # ── Step 3: Determine verdict ─────────────────────────────────────
    if matches:
        verdict = "SUPPORTED"
        logger.info(
            "math_claim_verifier | SUPPORTED — computed=%s matches claimed=%s",
            computed, claimed_value,
        )
    else:
        verdict = "CONTRADICTED"
        logger.info(
            "math_claim_verifier | CONTRADICTED — computed=%s ≠ claimed=%s",
            computed, claimed_value,
        )

    return {"verdict": verdict, "computed": computed}
