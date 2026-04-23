"""
hallu-check | nodes/tools/python_exec.py
Sandboxed Python executor for the Recursive Language Model reasoner.

Executes short Python snippets emitted by the LLM inside a subprocess with:
  • hard wall-clock timeout
  • isolated cwd (a fresh temp dir, discarded after)
  • Python's isolated mode (-I) to ignore user site-packages / env vars
  • output truncation
  • a small preamble that imports common math libs (math, fractions, decimal,
    statistics, sympy if available) so the model doesn't need to remember them

This is the "REPL tool" from the MIT RLM paper — the model writes Python
for any sub-step that is computational (arithmetic, algebra, combinatorics)
instead of trying to compute it token-by-token, which is where small models
hallucinate numbers.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("hallu-check.python_exec")

_DEFAULT_TIMEOUT = 5
_MAX_OUTPUT_CHARS = 4000

# Preamble run before the model's snippet. Keep imports cheap.
_PREAMBLE = (
    "import math, fractions, decimal, statistics, itertools, functools\n"
    "try:\n"
    "    import sympy\n"
    "    from sympy import symbols, solve, simplify, sympify, Rational, Eq\n"
    "except ImportError:\n"
    "    sympy = None\n"
)


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    error: str | None  # populated on timeout / launch failure

    def render(self) -> str:
        """Compact textual form for injection back into the leaf answer."""
        if self.error:
            return f"[python error: {self.error}]"
        if self.ok:
            out = self.stdout.strip()
            return f"[python result]\n{out}" if out else "[python result: (no stdout)]"
        return f"[python error]\n{self.stderr.strip()[:500]}"


def run_python(code: str, timeout: int = _DEFAULT_TIMEOUT) -> ExecResult:
    """
    Execute a Python snippet in an isolated subprocess.

    Args:
        code:    The snippet to run. Will be prefixed with a safe preamble.
        timeout: Wall-clock timeout in seconds.

    Returns:
        ExecResult with stdout/stderr and an ok flag.
    """
    if not code or not code.strip():
        return ExecResult(ok=False, stdout="", stderr="", error="empty code")

    full_script = _PREAMBLE + "\n# --- model code below ---\n" + code

    try:
        with tempfile.TemporaryDirectory(prefix="hallu_pyexec_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", full_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                check=False,
            )
    except subprocess.TimeoutExpired:
        logger.warning("python_exec | snippet timed out after %ds", timeout)
        return ExecResult(ok=False, stdout="", stderr="", error=f"timeout after {timeout}s")
    except Exception as e:
        logger.warning("python_exec | launch failed: %s", e)
        return ExecResult(ok=False, stdout="", stderr="", error=str(e))

    stdout = (proc.stdout or "")[:_MAX_OUTPUT_CHARS]
    stderr = (proc.stderr or "")[:_MAX_OUTPUT_CHARS]
    ok = proc.returncode == 0

    if not ok:
        logger.info("python_exec | non-zero exit (%d). stderr: %s", proc.returncode, stderr[:200])

    return ExecResult(ok=ok, stdout=stdout, stderr=stderr, error=None)
