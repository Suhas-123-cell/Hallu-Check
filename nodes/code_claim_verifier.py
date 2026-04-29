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
# Test Case Generation (Local LLM via Ollama — no API dependency)
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
    Generate 3 test cases for the code snippet using the local LLM.

    Falls back to Gemini if available. Always returns at least one fallback
    "crash test" so EGV never silently skips verification.

    Returns a list of dicts with keys: input, expected, description.
    """
    prompt = _TEST_GEN_PROMPT.format(
        code_snippet=code_snippet[:3000],
        claim=claim[:500],
    )

    raw = _call_llm_for_test_cases(prompt)

    # ── Parse LLM response ───────────────────────────────────────────
    if raw:
        parsed = _robust_parse_test_cases(raw)
        if parsed:
            logger.info("code_claim_verifier | Generated %d test cases.", len(parsed))
            return parsed[:3]
        logger.warning("code_claim_verifier | Failed to parse LLM test-case response.")

    # ── Fallback: generate a trivial crash test without any LLM ──────
    func_name = _extract_function_name(code_snippet)
    if func_name:
        fallback = _generate_crash_test(func_name, code_snippet)
        logger.info("code_claim_verifier | Using fallback crash test for %s.", func_name)
        return [fallback]

    logger.warning("code_claim_verifier | No test cases and no function found — UNKNOWN.")
    return []


def _robust_parse_test_cases(raw: str) -> List[Dict[str, str]]:
    """
    Robustly parse test cases from LLM output.

    Handles Llama 3B's common output patterns:
      - Wrapped in markdown ```json ... ``` fences
      - Explanatory text before/after the JSON
      - Individual JSON objects instead of an array
      - Extra whitespace and newlines
    """
    # Strategy 1: Direct parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list) and parsed:
            return _validate_test_cases(parsed)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences (```json ... ``` or ``` ... ```)
    fence_patterns = [
        re.compile(r"```(?:json)?\s*\n([\s\S]*?)```"),
        re.compile(r"```([\s\S]*?)```"),
    ]
    for pat in fence_patterns:
        match = pat.search(raw)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if isinstance(parsed, list) and parsed:
                    return _validate_test_cases(parsed)
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find the first [...] array in the response
    array_match = re.search(r"\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]", raw)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list) and parsed:
                return _validate_test_cases(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Greedy [...] match (less precise but catches more)
    greedy_match = re.search(r"\[[\s\S]*\]", raw)
    if greedy_match:
        try:
            parsed = json.loads(greedy_match.group(0))
            if isinstance(parsed, list) and parsed:
                return _validate_test_cases(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 5: Extract individual {...} objects and build an array
    obj_matches = re.findall(r"\{[^{}]*\}", raw)
    if obj_matches:
        test_cases = []
        for obj_str in obj_matches:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "input" in obj:
                    test_cases.append(obj)
            except json.JSONDecodeError:
                continue
        if test_cases:
            return _validate_test_cases(test_cases)

    return []


def _validate_test_cases(parsed: list) -> List[Dict[str, str]]:
    """Filter a parsed list to only valid test case dicts."""
    valid = [
        tc for tc in parsed
        if isinstance(tc, dict) and "input" in tc and "expected" in tc
    ]
    return valid


def _generate_crash_test(func_name: str, code_snippet: str) -> Dict[str, str]:
    """
    Generate a trivial fallback test: call the function with a simple input
    and check it doesn't crash. This ensures EGV always has at least one
    test case even when the LLM fails to produce parseable output.
    """
    # Try to infer a sensible trivial input from the function signature
    sig_match = re.search(r"def\s+" + re.escape(func_name) + r"\s*\(([^)]*)", code_snippet)
    if sig_match:
        params = [p.strip().split(":")[0].strip().split("=")[0].strip()
                  for p in sig_match.group(1).split(",") if p.strip()]
        n_params = len(params)
    else:
        n_params = 1

    # Generate trivial inputs based on parameter count
    trivial_inputs = {
        0: "",
        1: "[]",
        2: "[], 0",
        3: "[], 0, 0",
    }
    test_input = trivial_inputs.get(n_params, ", ".join(["0"] * n_params))

    return {
        "input": test_input,
        "expected": "__CRASH_TEST__",  # Special sentinel — only checks no crash
        "description": f"Fallback crash test: {func_name}({test_input}) should not raise",
    }


def _call_llm_for_test_cases(prompt: str) -> str:
    """
    Call the local LLM (Ollama) for test case generation.
    Falls back to Gemini if local is unavailable.

    Uses the project's local_llm module — Ollama on localhost,
    zero cost, no API quotas.
    """
    # ── Primary: Local LLM via Ollama (free, no quota) ───────────────────
    try:
        from nodes.local_llm import chat_completion, is_available  # type: ignore[import-not-found]
        if is_available():
            result = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise test-case generator. "
                            "Always respond with ONLY a valid JSON array. "
                            "No explanations, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.2,
            )
            if result:
                logger.info("code_claim_verifier | Test cases generated via local LLM.")
                return result
    except Exception as e:
        logger.warning("code_claim_verifier | Local LLM failed: %s", str(e)[:120])

    # ── Fallback: Gemini (if available and has quota) ────────────────────
    try:
        from nodes.claim_verifier import _gemini_generate  # type: ignore[import-not-found]
        result = _gemini_generate(prompt)
        if result:
            logger.info("code_claim_verifier | Test cases generated via Gemini (fallback).")
            return result
    except Exception as e:
        logger.warning("code_claim_verifier | Gemini fallback also failed: %s", str(e)[:120])

    return ""


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
    # Dedent so class-wrapped or indented code starts at column 0
    import textwrap as _tw
    clean_code = _tw.dedent(code)
    script = (
        "import math, fractions, decimal, statistics, itertools, functools\n"
        "try:\n"
        "    import sympy\n"
        "except ImportError:\n"
        "    sympy = None\n"
        "\n"
        f"{clean_code}\n"
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

def verify_code_claim(
    claim: str,
    code_snippet: str,
    ground_truth_code: str = "",
) -> Dict[str, Any]:
    """
    Verify a code-related claim by generating and executing test cases.

    Pipeline:
      1. Local LLM (Ollama) generates 3 test cases for the code snippet.
         Falls back to a trivial crash test if LLM response is unparseable.
      2. Forbidden imports (os, sys, subprocess, etc.) are stripped.
      3. Each test runs in an isolated subprocess (``subprocess.run``,
         never ``eval``/``exec``) with a 5-second timeout.
      4. Returns a verdict based on pass/fail ratio.

        Differential testing (optional):
            - If ``ground_truth_code`` is provided, each failing claim-test is re-run
                on the ground-truth implementation.
            - If the ground truth also fails, that test is discarded as invalid.
            - A test is counted as CONTRADICTED only when submitted code fails and
                ground truth passes.

        Args:
        claim:        The atomic claim about the code (e.g., "this function
                      returns -1 when the target is not found").
        code_snippet: The code being verified (should contain at least one
                      function definition).
                ground_truth_code:
                                            Optional HumanEval reference implementation used for
                                            differential testing.

    Returns:
        Dict with:
          - ``"verdict"``: ``"SUPPORTED"`` (all pass), ``"CONTRADICTED"``
            (any fail), or ``"UNKNOWN"`` (no tests / couldn't run).
          - ``"failed_tests"``: List of dicts describing each failed test,
            with keys ``description``, ``input``, ``expected``, ``actual``,
            ``error``.
                    - ``"tests_generated"``: number of generated test cases.
                    - ``"discarded_invalid_tests"``: number of tests discarded because
                        both submitted and ground-truth failed.
                    - ``"tests_kept"``: number of tests retained for verdicting.
                    - ``"final_verdict"``: same as ``"verdict"`` (explicit summary key).
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

    # ── Optional differential-testing setup ───────────────────────────
    use_differential = bool(ground_truth_code and ground_truth_code.strip())
    safe_ground_truth = ""
    gt_func_name: str | None = None
    if use_differential:
        safe_ground_truth = _sanitize_code(ground_truth_code)

        # Prefer matching the submitted function name for apples-to-apples calls.
        if re.search(r"def\s+" + re.escape(func_name) + r"\s*\(", safe_ground_truth):
            gt_func_name = func_name
        else:
            gt_func_name = _extract_function_name(safe_ground_truth)

        if not gt_func_name:
            logger.warning(
                "code_claim_verifier | Differential testing disabled: "
                "no function definition found in ground_truth_code."
            )
            use_differential = False

    # ── Generate test cases via local LLM ─────────────────────────────
    test_cases = _generate_test_cases(claim, code_snippet)
    if not test_cases:
        logger.warning("code_claim_verifier | No test cases generated.")
        return {"verdict": "UNKNOWN", "failed_tests": []}

    # ── Run each test case ────────────────────────────────────────────
    failed_tests: List[Dict[str, str]] = []
    passed = 0
    discarded_invalid_tests = 0

    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        test_expected = tc.get("expected", "")
        test_desc = tc.get("description", f"test_{i + 1}")
        is_crash_test = (test_expected == "__CRASH_TEST__")

        result = _run_single_test(safe_code, func_name, test_input)

        # Evaluate submitted-code outcome first
        submitted_pass = False
        submitted_actual = ""
        submitted_error = ""
        gt_actual = "N/A"
        kept = True

        if not result["ok"]:
            submitted_pass = False
            submitted_actual = result["stderr"][:200] if result["stderr"] else ""
            submitted_error = result["error"] or "execution failed"
        elif is_crash_test:
            submitted_pass = True
        elif _outputs_match(result["stdout"], test_expected):
            submitted_pass = True
        else:
            submitted_pass = False
            submitted_actual = result["stdout"][:200]

        # Differential testing: keep failure only if GT passes the same test
        if not submitted_pass and use_differential and gt_func_name:
            gt_result = _run_single_test(safe_ground_truth, gt_func_name, test_input)
            gt_actual = (
                gt_result["stdout"][:200]
                if gt_result["ok"]
                else (gt_result["stderr"][:200] if gt_result["stderr"] else str(gt_result.get("error", "")))
            )

            gt_pass = False
            if gt_result["ok"]:
                if is_crash_test:
                    gt_pass = True
                else:
                    gt_pass = _outputs_match(gt_result["stdout"], test_expected)

            if not gt_pass:
                discarded_invalid_tests += 1
                kept = False
                logger.debug(
                    "Test: input=%s | submitted=%s | ground_truth=%s | kept=%s",
                    test_input,
                    submitted_actual or ("PASS" if submitted_pass else submitted_error),
                    gt_actual,
                    kept,
                )
                logger.info(
                    "code_claim_verifier | Test %d discarded as invalid "
                    "(submitted and ground truth both fail).",
                    i + 1,
                )
                continue

        logger.debug(
            "Test: input=%s | submitted=%s | ground_truth=%s | kept=%s",
            test_input,
            submitted_actual or ("PASS" if submitted_pass else submitted_error),
            gt_actual,
            kept,
        )

        if not submitted_pass and submitted_error:
            # Execution error (syntax error, runtime error, timeout)
            failed_tests.append({
                "description": test_desc,
                "input": test_input,
                "expected": test_expected if not is_crash_test else "(no crash)",
                "actual": submitted_actual,
                "error": submitted_error,
            })
            logger.info(
                "code_claim_verifier | Test %d FAIL (error): %s | stderr: %s",
                i + 1, submitted_error, submitted_actual[:120],
            )
        elif submitted_pass and is_crash_test:
            # Crash test passed — function ran without error
            passed += 1
            logger.info("code_claim_verifier | Test %d PASS (crash test — no error)", i + 1)
        elif not submitted_pass:
            # Wrong output
            failed_tests.append({
                "description": test_desc,
                "input": test_input,
                "expected": test_expected,
                "actual": submitted_actual,
                "error": "",
            })
            logger.info(
                "code_claim_verifier | Test %d FAIL (wrong output): expected=%r, actual=%r",
                i + 1, test_expected, submitted_actual[:60],
            )
        else:
            passed += 1
            logger.info("code_claim_verifier | Test %d PASS", i + 1)

    # ── Determine verdict ─────────────────────────────────────────────
    effective_total = passed + len(failed_tests)
    if effective_total == 0:
        verdict = "UNKNOWN"
    elif passed == effective_total:
        verdict = "SUPPORTED"
    elif failed_tests:
        verdict = "CONTRADICTED"
    else:
        verdict = "UNKNOWN"

    logger.info(
        "code_claim_verifier | Verdict: %s (%d/%d passed, %d failed, %d discarded_invalid)",
        verdict, passed, effective_total, len(failed_tests), discarded_invalid_tests,
    )

    return {
        "verdict": verdict,
        "failed_tests": failed_tests,
        "tests_generated": len(test_cases),
        "discarded_invalid_tests": discarded_invalid_tests,
        "tests_kept": effective_total,
        "final_verdict": verdict,
    }
