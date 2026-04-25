"""
hallu-check | nodes/code_claim_verifier.py
Claim-Level Code Verification via Subprocess Execution

Verifies a code-related claim by:
  1. Asking Gemini to generate 3 test cases for the code snippet
  2. Running code + each test in an isolated subprocess (subprocess.run,
     never eval/exec) with a 5-second wall-clock timeout
  3. Returning a structured verdict dict

Safety:
  • Dangerous imports (os, sys, subprocess, shutil, signal, socket, ctypes)
    are stripped from the executed code before running.
  • Each test runs in a fresh tempdir with Python's isolated mode (-I).
  • Hard 5-second timeout per test case.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

logger = logging.getLogger("hallu-check.code_claim_verifier")

# ── Timeout for each subprocess execution ────────────────────────────────────
_EXEC_TIMEOUT = 5  # seconds

# ── Maximum output captured from each subprocess ─────────────────────────────
_MAX_OUTPUT_CHARS = 4000

# ── Forbidden imports — stripped before execution ────────────────────────────
_FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "signal", "socket",
    "ctypes", "importlib", "pathlib", "glob", "tempfile",
})

_IMPORT_LINE_RE = re.compile(
    r"^(\s*)"                         # leading whitespace
    r"(?:import\s+(\S+)"              # import X
    r"|from\s+(\S+)\s+import\b)",     # from X import …
    re.MULTILINE,
)


def _sanitize_code(code: str) -> str:
    """
    Remove lines that import any forbidden module.

    Handles:
      • ``import os``
      • ``import os, sys``  (entire line removed if ANY module is forbidden)
      • ``from os import path``
      • ``from os.path import join``
    """
    clean_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()

        # Check `import X` or `import X, Y, Z`
        if stripped.startswith("import "):
            modules = [m.strip().split(".")[0] for m in stripped[7:].split(",")]
            if any(m in _FORBIDDEN_MODULES for m in modules):
                logger.debug("code_claim_verifier | stripped forbidden import: %s", stripped)
                continue

        # Check `from X import …` or `from X.sub import …`
        if stripped.startswith("from "):
            match = re.match(r"from\s+(\S+)\s+import", stripped)
            if match:
                root_module = match.group(1).split(".")[0]
                if root_module in _FORBIDDEN_MODULES:
                    logger.debug("code_claim_verifier | stripped forbidden import: %s", stripped)
                    continue

        clean_lines.append(line)

    return "\n".join(clean_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini — generate test cases
# ─────────────────────────────────────────────────────────────────────────────

_TEST_GEN_PROMPT = """\
You are a test-case generator. Given a code snippet and a CLAIM about it, \
produce exactly 3 test cases to verify whether the code satisfies the claim.

CRITICAL: The "expected" value in each test case must be derived from what the \
CLAIM says the code should do — NOT from reading the code's actual implementation. \
You are testing whether the code matches the claim, so the expected values must \
reflect the claim's assertions.

Code snippet (for understanding the function signature and parameter types ONLY):
```
{code_snippet}
```

Claim to verify: "{claim}"

For each test case, provide:
- "input": the arguments to pass (as a Python expression string, e.g. "[1, 2, 3], 2")
- "expected": the expected return value AS STATED BY THE CLAIM, as a Python repr \
string (e.g. "-1" if the claim says "returns -1")
- "description": a one-line description of what aspect of the claim this test checks

Respond with ONLY this JSON array (no markdown fences, no extra text):
[
  {{"input": "...", "expected": "...", "description": "..."}},
  {{"input": "...", "expected": "...", "description": "..."}},
  {{"input": "...", "expected": "...", "description": "..."}}
]"""


def _generate_test_cases(claim: str, code_snippet: str) -> List[Dict[str, str]]:
    """
    Ask Gemini to produce 3 test cases for the code snippet.

    Returns a list of dicts with keys: input, expected, description.
    Returns an empty list if Gemini is unavailable or returns garbage.
    """
    try:
        from nodes.claim_verifier import _gemini_generate  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("code_claim_verifier | Cannot import _gemini_generate.")
        return []

    prompt = _TEST_GEN_PROMPT.format(
        code_snippet=code_snippet[:3000],
        claim=claim[:500],
    )

    raw = _gemini_generate(prompt)
    if not raw:
        logger.warning("code_claim_verifier | Gemini returned empty response.")
        return []

    # ── Parse JSON ───────────────────────────────────────────────────────
    # Try direct parse first, then extract from markdown fences
    for candidate in [
        raw.strip(),
        _extract_json_block(raw),
    ]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Validate structure
                valid = [
                    tc for tc in parsed
                    if isinstance(tc, dict) and "input" in tc and "expected" in tc
                ]
                if valid:
                    logger.info(
                        "code_claim_verifier | Gemini generated %d test cases.", len(valid)
                    )
                    return valid[:3]
        except json.JSONDecodeError:
            continue

    logger.warning("code_claim_verifier | Failed to parse Gemini test-case response.")
    return []


def _extract_json_block(text: str) -> str:
    """Extract JSON from a markdown ```json … ``` fence if present."""
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find a bare JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else ""


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess execution
# ─────────────────────────────────────────────────────────────────────────────

def _extract_function_name(code: str) -> str | None:
    """Extract the first function name from code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None


def _run_single_test(
    code: str,
    func_name: str,
    test_input: str,
    timeout: int = _EXEC_TIMEOUT,
) -> Dict[str, Any]:
    """
    Run one test case in an isolated subprocess via subprocess.run().

    The script:
      1. Defines the code (with forbidden imports stripped)
      2. Calls the function with the test input
      3. Prints the repr of the return value

    Returns a dict with keys: stdout, stderr, ok, error.
    """
    # Build the full script that the subprocess will execute
    script = (
        "import math, fractions, decimal, statistics, itertools, functools\n"
        "try:\n"
        "    import sympy\n"
        "except ImportError:\n"
        "    sympy = None\n"
        "\n"
        f"{code}\n"
        "\n"
        f"_result_ = {func_name}({test_input})\n"
        "print(repr(_result_))\n"
    )

    try:
        with tempfile.TemporaryDirectory(prefix="hallu_codeverify_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"stdout": "", "stderr": "", "ok": False, "error": str(e)[:300]}

    stdout = (proc.stdout or "")[:_MAX_OUTPUT_CHARS]
    stderr = (proc.stderr or "")[:_MAX_OUTPUT_CHARS]

    return {
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "ok": proc.returncode == 0,
        "error": None if proc.returncode == 0 else f"exit code {proc.returncode}",
    }


def _outputs_match(actual: str, expected: str) -> bool:
    """
    Compare actual subprocess output with expected value.

    Handles common repr differences:
      • whitespace / trailing newlines
      • string quoting (single vs double quotes)
      • numeric equivalence (1.0 == 1)
    """
    actual = actual.strip()
    expected = expected.strip()

    # Direct match
    if actual == expected:
        return True

    # Strip surrounding quotes for string comparison
    for q in ('"', "'"):
        if actual.startswith(q) and actual.endswith(q):
            if actual[1:-1] == expected:
                return True
        if expected.startswith(q) and expected.endswith(q):
            if expected[1:-1] == actual:
                return True

    # Numeric equivalence
    try:
        if abs(float(actual) - float(expected)) < 1e-9:
            return True
    except (ValueError, TypeError):
        pass

    # repr vs str: e.g., actual="[1, 2, 3]" expected="[1,2,3]"
    if actual.replace(" ", "") == expected.replace(" ", ""):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def verify_code_claim(claim: str, code_snippet: str) -> Dict[str, Any]:
    """
    Verify a code-related claim by generating and executing test cases.

    Pipeline:
      1. Gemini generates 3 test cases for the code snippet.
      2. Forbidden imports (os, sys, subprocess, etc.) are stripped.
      3. Each test runs in an isolated subprocess (``subprocess.run``,
         never ``eval``/``exec``) with a 5-second timeout.
      4. Returns a verdict based on pass/fail ratio.

    Args:
        claim:        The atomic claim about the code (e.g., "this function
                      returns -1 when the target is not found").
        code_snippet: The code being verified (should contain at least one
                      function definition).

    Returns:
        Dict with:
          - ``"verdict"``: ``"SUPPORTED"`` (all pass), ``"CONTRADICTED"``
            (any fail), or ``"UNKNOWN"`` (no tests / couldn't run).
          - ``"failed_tests"``: List of dicts describing each failed test,
            with keys ``description``, ``input``, ``expected``, ``actual``,
            ``error``.
    """
    if not code_snippet or not code_snippet.strip():
        logger.warning("code_claim_verifier | Empty code snippet.")
        return {"verdict": "UNKNOWN", "failed_tests": []}

    # ── Extract function name ─────────────────────────────────────────
    func_name = _extract_function_name(code_snippet)
    if not func_name:
        logger.warning("code_claim_verifier | No function definition found in code snippet.")
        return {"verdict": "UNKNOWN", "failed_tests": []}

    # ── Sanitize: strip forbidden imports ─────────────────────────────
    safe_code = _sanitize_code(code_snippet)

    # ── Generate test cases via Gemini ────────────────────────────────
    test_cases = _generate_test_cases(claim, code_snippet)
    if not test_cases:
        logger.warning("code_claim_verifier | No test cases generated.")
        return {"verdict": "UNKNOWN", "failed_tests": []}

    # ── Run each test case ────────────────────────────────────────────
    failed_tests: List[Dict[str, str]] = []
    passed = 0

    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        test_expected = tc.get("expected", "")
        test_desc = tc.get("description", f"test_{i + 1}")

        result = _run_single_test(safe_code, func_name, test_input)

        if not result["ok"]:
            # Execution error (syntax error, runtime error, timeout)
            failed_tests.append({
                "description": test_desc,
                "input": test_input,
                "expected": test_expected,
                "actual": result["stderr"][:200] if result["stderr"] else "",
                "error": result["error"] or "execution failed",
            })
            logger.info(
                "code_claim_verifier | Test %d FAIL (error): %s", i + 1, result["error"]
            )
        elif not _outputs_match(result["stdout"], test_expected):
            # Wrong output
            failed_tests.append({
                "description": test_desc,
                "input": test_input,
                "expected": test_expected,
                "actual": result["stdout"][:200],
                "error": "",
            })
            logger.info(
                "code_claim_verifier | Test %d FAIL (wrong output): expected=%r, actual=%r",
                i + 1, test_expected, result["stdout"][:60],
            )
        else:
            passed += 1
            logger.info("code_claim_verifier | Test %d PASS", i + 1)

    # ── Determine verdict ─────────────────────────────────────────────
    total = len(test_cases)
    if passed == total:
        verdict = "SUPPORTED"
    elif failed_tests:
        verdict = "CONTRADICTED"
    else:
        verdict = "UNKNOWN"

    logger.info(
        "code_claim_verifier | Verdict: %s (%d/%d passed, %d failed)",
        verdict, passed, total, len(failed_tests),
    )

    return {"verdict": verdict, "failed_tests": failed_tests}
