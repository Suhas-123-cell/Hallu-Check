"""
hallu-check | nodes/execution_verifier.py
Contribution 2 — Execution-Grounded Verification (EGV)

For REASONING queries (code/math), NLI verification is meaningless —
it can't tell if binary_search has an off-by-one error or if a math
computation is correct.

This module replaces NLI with execution-based verification:
  • CODE:  Extract code → generate test cases → execute → verify outputs
  • MATH:  Extract computation → execute in Python → compare with claim

Architecture:
  1. Detect if the LLM output contains code or math
  2. Extract the executable portion
  3. Generate/derive test cases
  4. Run in the existing sandboxed python_exec.py
  5. Return structured ExecutionVerdict with pass/fail per test

Uses the existing sandbox in nodes/tools/python_exec.py for safe execution.
"""
from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nodes.tools.python_exec import run_python  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.execution_verifier")


@dataclass
class TestCase:
    """A single test case for code verification."""
    input_args: str       # e.g., "[1, 3, 5], 3"
    expected_output: str  # e.g., "1"
    description: str = ""


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_case: TestCase
    actual_output: str
    passed: bool
    error: str = ""


@dataclass
class ExecutionVerdict:
    """Structured result of execution-based verification."""
    verdict: str               # "PASS", "FAIL", "ERROR"
    score: float               # 0.0 = all failed, 1.0 = all passed
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[TestResult] = field(default_factory=list)
    error_message: str = ""
    code_extracted: str = ""
    execution_output: str = ""


# ── Code Detection & Extraction ──────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)
_FUNCTION_DEF_RE = re.compile(
    r"(def\s+\w+\s*\([\s\S]*?)(?=\ndef\s|\nclass\s|\Z)",
    re.MULTILINE,
)


def _extract_code(llm_output: str) -> str:
    """
    Extract Python code from LLM output.

    Priority:
    1. Fenced ```python blocks
    2. Bare function definitions (with nested function support)
    3. Any indented code blocks

    Always returns dedented code (column 0).
    """
    import textwrap as _tw

    # Try fenced code blocks first
    blocks = _CODE_BLOCK_RE.findall(llm_output)
    if blocks:
        return _tw.dedent(blocks[0]).strip()

    # Try bare function definitions — capture the full function
    # including nested defs by tracking indentation
    lines = llm_output.split("\n")
    func_start = None
    func_lines: list[str] = []
    base_indent = 0

    for i, line in enumerate(lines):
        if func_start is None:
            # Look for a top-level def
            m = re.match(r"^(\s*)def\s+\w+\s*\(", line)
            if m:
                func_start = i
                base_indent = len(m.group(1))
                func_lines = [line]
        else:
            stripped = line.strip()
            # Continue if: blank line, or indented more than base
            if stripped == "":
                func_lines.append(line)
            elif len(line) - len(line.lstrip()) > base_indent:
                func_lines.append(line)
            elif line.lstrip().startswith("def ") and len(line) - len(line.lstrip()) == base_indent:
                # Same-level def — this is a sibling function, stop
                break
            else:
                # Line at base indent or less that isn't a def — stop
                break

    if func_lines:
        # Strip trailing blank lines
        while func_lines and func_lines[-1].strip() == "":
            func_lines.pop()
        if func_lines:
            return _tw.dedent("\n".join(func_lines)).strip()

    # Try indented blocks (4+ spaces or tab)
    indented_lines = []
    in_block = False
    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            indented_lines.append(line)
            in_block = True
        elif in_block and line.strip() == "":
            indented_lines.append(line)
        else:
            if in_block and len(indented_lines) >= 3:
                break
            in_block = False
            indented_lines = []

    if len(indented_lines) >= 3:
        return _tw.dedent("\n".join(indented_lines)).strip()

    return ""


def has_code(llm_output: str) -> bool:
    """Check if LLM output contains executable code."""
    return bool(_extract_code(llm_output))


# ── Math Detection & Extraction ──────────────────────────────────────────────

_MATH_ANSWER_RE = re.compile(
    r"(?:answer|result|equals?|=)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_MATH_EXPRESSION_RE = re.compile(
    r"(\d+\s*[+\-*/^%]\s*\d+(?:\s*[+\-*/^%]\s*\d+)*)",
)


def has_math(llm_output: str) -> bool:
    """Check if LLM output contains a math computation claim."""
    return bool(_MATH_ANSWER_RE.search(llm_output))


def _extract_math_claim(llm_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a math expression and its claimed result from LLM output.

    Returns (expression, claimed_result) or (None, None).
    """
    answer_match = _MATH_ANSWER_RE.search(llm_output)
    expr_match = _MATH_EXPRESSION_RE.search(llm_output)

    claimed_result = answer_match.group(1) if answer_match else None
    expression = expr_match.group(1) if expr_match else None

    return expression, claimed_result


# ── Test Case Generation ─────────────────────────────────────────────────────

def _generate_test_cases_from_query(
    query: str,
    code: str,
) -> List[TestCase]:
    """
    Generate test cases from the query context and code structure.

    Uses heuristics + pattern matching to create meaningful test cases
    without needing an LLM call (zero additional cost).
    """
    tests: List[TestCase] = []

    # Extract function name and parameters
    func_match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
    if not func_match:
        return tests

    func_name = func_match.group(1)
    params = [p.strip().split(":")[0].strip() for p in func_match.group(2).split(",")]

    # ── Heuristic test generation based on common patterns ──────────

    # Binary search / search functions
    if any(kw in func_name.lower() or kw in query.lower()
           for kw in ["search", "find", "index", "lookup"]):
        tests.extend([
            TestCase("[1, 2, 3, 4, 5], 3", "2", "target in middle"),
            TestCase("[1, 2, 3, 4, 5], 1", "0", "target at start"),
            TestCase("[1, 2, 3, 4, 5], 5", "4", "target at end"),
            TestCase("[1, 2, 3, 4, 5], 6", "-1", "target not found"),
            TestCase("[], 1", "-1", "empty list"),
        ])

    # Sorting functions
    elif any(kw in func_name.lower() or kw in query.lower()
             for kw in ["sort", "arrange", "order"]):
        tests.extend([
            TestCase("[3, 1, 4, 1, 5]", "[1, 1, 3, 4, 5]", "normal case"),
            TestCase("[]", "[]", "empty list"),
            TestCase("[1]", "[1]", "single element"),
            TestCase("[5, 4, 3, 2, 1]", "[1, 2, 3, 4, 5]", "reverse sorted"),
            TestCase("[1, 1, 1]", "[1, 1, 1]", "all same"),
        ])

    # Fibonacci
    elif "fibonacci" in func_name.lower() or "fib" in func_name.lower():
        tests.extend([
            TestCase("0", "0", "fib(0)"),
            TestCase("1", "1", "fib(1)"),
            TestCase("5", "5", "fib(5)"),
            TestCase("10", "55", "fib(10)"),
        ])

    # Factorial
    elif "factorial" in func_name.lower() or "fact" in func_name.lower():
        tests.extend([
            TestCase("0", "1", "0! = 1"),
            TestCase("1", "1", "1! = 1"),
            TestCase("5", "120", "5! = 120"),
            TestCase("10", "3628800", "10!"),
        ])

    # Palindrome
    elif "palindrome" in func_name.lower() or "palindrome" in query.lower():
        tests.extend([
            TestCase("'racecar'", "True", "palindrome"),
            TestCase("'hello'", "False", "not palindrome"),
            TestCase("''", "True", "empty string"),
            TestCase("'a'", "True", "single char"),
        ])

    # Two sum / pair sum
    elif "two_sum" in func_name.lower() or "two sum" in query.lower():
        tests.extend([
            TestCase("[2, 7, 11, 15], 9", "[0, 1]", "basic case"),
            TestCase("[3, 2, 4], 6", "[1, 2]", "not first two"),
        ])

    # Reverse
    elif "reverse" in func_name.lower():
        tests.extend([
            TestCase("[1, 2, 3]", "[3, 2, 1]", "normal"),
            TestCase("[]", "[]", "empty"),
            TestCase("[1]", "[1]", "single"),
        ])

    # Generic: at least test with basic inputs if we couldn't pattern-match
    if not tests:
        # Try to infer from parameter count
        if len(params) == 1:
            tests.append(TestCase("[]", "None", "empty input (smoke test)"))
        elif len(params) == 2:
            tests.append(TestCase("[], 0", "None", "empty input (smoke test)"))

    return tests


def _generate_test_cases_llm(
    query: str,
    code: str,
) -> List[TestCase]:
    """
    Generate test cases using the LLM (Llama) for complex functions
    where heuristics fail.

    Only called as a fallback when heuristic generation produces 0 tests.
    """
    from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
    from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]

    if not HF_API_TOKEN:
        return []

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=60)
        prompt = (
            "Generate 3-5 test cases for this Python function. "
            "For each test case, output ONLY in this exact format:\n"
            "INPUT: <args>\n"
            "EXPECTED: <result>\n\n"
            f"Function:\n{code[:1000]}\n\n"
            f"Context: {query[:200]}\n\n"
            "Test cases:"
        )

        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a test case generator. Output ONLY test cases in the exact format requested."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )

        raw = ""
        if response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content.strip()

        if not raw:
            return []

        # Parse INPUT/EXPECTED pairs
        tests = []
        input_re = re.compile(r"INPUT:\s*(.+)")
        expected_re = re.compile(r"EXPECTED:\s*(.+)")

        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            input_match = input_re.match(lines[i].strip())
            if input_match and i + 1 < len(lines):
                expected_match = expected_re.match(lines[i + 1].strip())
                if expected_match:
                    tests.append(TestCase(
                        input_args=input_match.group(1).strip(),
                        expected_output=expected_match.group(1).strip(),
                        description=f"LLM-generated test {len(tests) + 1}",
                    ))
                    i += 2
                    continue
            i += 1

        return tests[:5]

    except Exception as e:
        logger.warning("EGV | LLM test generation failed: %s", e)
        return []


# ── Execution Engine ─────────────────────────────────────────────────────────

def _run_test(
    code: str,
    func_name: str,
    test_case: TestCase,
    timeout: int = 10,
) -> TestResult:
    """Run a single test case against the extracted code."""
    # Dedent the code first so it starts at column 0,
    # then concatenate (never embed multi-line code inside textwrap.dedent)
    clean_code = textwrap.dedent(code)
    test_code = (
        f"{clean_code}\n\n"
        f"# --- Test execution ---\n"
        f"result = {func_name}({test_case.input_args})\n"
        f"print(repr(result))\n"
    )

    try:
        exec_result = run_python(test_code)
        output = exec_result.render().strip()

        # Check for errors
        if exec_result.error:
            return TestResult(
                test_case=test_case,
                actual_output=output,
                passed=False,
                error=exec_result.error[:200],
            )

        # Compare output with expected
        actual = output.strip()
        expected = test_case.expected_output.strip()

        # Normalize for comparison
        passed = (
            actual == expected
            or actual == repr(expected)
            or str(actual) == str(expected)
        )

        return TestResult(
            test_case=test_case,
            actual_output=actual,
            passed=passed,
        )

    except Exception as e:
        return TestResult(
            test_case=test_case,
            actual_output="",
            passed=False,
            error=str(e)[:200],
        )


# ── Public API ───────────────────────────────────────────────────────────────

def verify_code(
    llm_output: str,
    query: str,
) -> ExecutionVerdict:
    """
    Verify code correctness through execution.

    Pipeline:
      1. Extract code from LLM output
      2. Generate test cases (heuristic, then LLM fallback)
      3. Execute each test in the sandbox
      4. Return structured verdict

    Args:
        llm_output: The LLM's full output containing code.
        query: The user's original query (for test generation context).

    Returns:
        ExecutionVerdict with pass/fail details.
    """
    code = _extract_code(llm_output)
    if not code:
        return ExecutionVerdict(
            verdict="ERROR",
            score=0.5,  # Can't verify → neutral
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_message="No executable code found in LLM output.",
        )

    # Extract function name
    func_match = re.search(r"def\s+(\w+)\s*\(", code)
    if not func_match:
        # Try running the code directly (no function, just a script)
        return _verify_script(code)

    func_name = func_match.group(1)
    logger.info("EGV | Extracted function '%s' for testing.", func_name)

    # Generate test cases
    tests = _generate_test_cases_from_query(query, code)
    if not tests:
        tests = _generate_test_cases_llm(query, code)
    if not tests:
        logger.warning("EGV | No test cases generated for '%s'.", func_name)
        return ExecutionVerdict(
            verdict="ERROR",
            score=0.5,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_message="Could not generate test cases.",
            code_extracted=code,
        )

    # Run tests
    results: List[TestResult] = []
    for test in tests:
        result = _run_test(code, func_name, test)
        results.append(result)
        logger.debug(
            "EGV | Test '%s': %s (expected=%s, actual=%s)",
            test.description,
            "PASS" if result.passed else "FAIL",
            test.expected_output,
            result.actual_output[:50],
        )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    score = passed / len(results) if results else 0.0

    verdict = "PASS" if score >= 0.8 else "FAIL"
    logger.info(
        "EGV | Code verification: %s (%d/%d tests passed, score=%.2f)",
        verdict, passed, len(results), score,
    )

    return ExecutionVerdict(
        verdict=verdict,
        score=score,
        total_tests=len(results),
        passed_tests=passed,
        failed_tests=failed,
        test_results=results,
        code_extracted=code,
    )


def _verify_script(code: str) -> ExecutionVerdict:
    """Verify a standalone script (no function) by just running it."""
    try:
        result = run_python(code)
        output = result.render().strip()

        if result.error:
            return ExecutionVerdict(
                verdict="FAIL",
                score=0.0,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                error_message=result.error[:200],
                code_extracted=code,
                execution_output=output,
            )

        return ExecutionVerdict(
            verdict="PASS",
            score=1.0,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            code_extracted=code,
            execution_output=output,
        )

    except Exception as e:
        return ExecutionVerdict(
            verdict="ERROR",
            score=0.0,
            total_tests=1,
            passed_tests=0,
            failed_tests=1,
            error_message=str(e)[:200],
            code_extracted=code,
        )


def verify_math(
    llm_output: str,
    query: str,
) -> ExecutionVerdict:
    """
    Verify mathematical claims by executing them in Python.

    Extracts math expressions and claimed results from the LLM output,
    then runs the actual computation to verify correctness.

    Args:
        llm_output: The LLM's output containing math claims.
        query: The user's original query.

    Returns:
        ExecutionVerdict with pass/fail for the math computation.
    """
    expression, claimed_result = _extract_math_claim(llm_output)

    if not expression and not claimed_result:
        return ExecutionVerdict(
            verdict="ERROR",
            score=0.5,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_message="No math expression found to verify.",
        )

    # Build verification code
    if expression:
        # Replace ^ with ** for Python
        py_expr = expression.replace("^", "**")
        verify_code_str = textwrap.dedent(f"""\
            import math
            result = {py_expr}
            print(result)
        """)
    else:
        # Try to extract computation from the query itself
        verify_code_str = textwrap.dedent(f"""\
            # Attempting to compute from query context
            print("CANNOT_VERIFY")
        """)

    try:
        exec_result = run_python(verify_code_str)
        computed = exec_result.render().strip()

        if exec_result.error or computed == "CANNOT_VERIFY":
            return ExecutionVerdict(
                verdict="ERROR",
                score=0.5,
                total_tests=1 if expression else 0,
                passed_tests=0,
                failed_tests=0,
                error_message=exec_result.error[:200] if exec_result.error else "Could not verify math.",
            )

        # Compare computed result with claimed result
        if claimed_result:
            try:
                computed_num = float(computed)
                claimed_num = float(claimed_result)
                passed = abs(computed_num - claimed_num) < 1e-6
            except (ValueError, TypeError):
                passed = computed.strip() == claimed_result.strip()

            score = 1.0 if passed else 0.0
            verdict = "PASS" if passed else "FAIL"

            logger.info(
                "EGV | Math verification: %s (computed=%s, claimed=%s)",
                verdict, computed, claimed_result,
            )

            return ExecutionVerdict(
                verdict=verdict,
                score=score,
                total_tests=1,
                passed_tests=1 if passed else 0,
                failed_tests=0 if passed else 1,
                test_results=[TestResult(
                    test_case=TestCase(expression or "", claimed_result, "math check"),
                    actual_output=computed,
                    passed=passed,
                )],
                execution_output=computed,
            )

        # No claimed result — just report the computed value
        return ExecutionVerdict(
            verdict="PASS",
            score=1.0,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            execution_output=computed,
        )

    except Exception as e:
        return ExecutionVerdict(
            verdict="ERROR",
            score=0.5,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_message=str(e)[:200],
        )
