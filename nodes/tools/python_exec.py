from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("hallu-check.python_exec")

_UNSAFE_IMPORTS = {
    "os", "subprocess", "sys", "socket", "shutil", "pathlib", "ctypes",
    "importlib", "builtins", "signal", "multiprocessing", "threading",
    "asyncio", "urllib", "http", "ftplib", "smtplib", "telnetlib",
    "xmlrpc", "pickle", "shelve", "dbm", "pty", "fcntl", "resource",
}
_UNSAFE_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}


def _check_code_safety(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            root = module.split(".")[0]
            if root in _UNSAFE_IMPORTS:
                raise ValueError(f"Unsafe import: {module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _UNSAFE_BUILTINS:
                raise ValueError(f"Unsafe call: {node.func.id}")

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
        if self.error:
            return f"[python error: {self.error}]"
        if self.ok:
            out = self.stdout.strip()
            return f"[python result]\n{out}" if out else "[python result: (no stdout)]"
        return f"[python error]\n{self.stderr.strip()[:500]}"


def run_python(code: str, timeout: int = _DEFAULT_TIMEOUT) -> ExecResult:
    if not code or not code.strip():
        return ExecResult(ok=False, stdout="", stderr="", error="empty code")

    try:
        _check_code_safety(code)
    except (ValueError, SyntaxError) as e:
        return ExecResult(ok=False, stdout="", stderr="", error=str(e))

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
