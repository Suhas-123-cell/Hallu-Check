# ─────────────────────────────────────────────
# hallu-check  |  config.py
# Loads all secrets from environment / .env
# ─────────────────────────────────────────────
from __future__ import annotations

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv  # type: ignore[import-untyped, import-not-found]

load_dotenv(override=True)

# ── HuggingFace ─────────────────────────────
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
LOCAL_MODEL_ID: str = os.getenv(
    "LOCAL_MODEL_ID",
    "meta-llama/Llama-3.2-1B-Instruct",
)

# ── Google Gemini ─────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Web Search ────────────────────────────────
# Using googlesearch-python (no API key needed)
SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

# ── PageIndex / RAG ───────────────────────────
# Directory for UUID-based temp markdown files (concurrent-safe)
SCRAPED_MD_DIR: str = os.getenv("SCRAPED_MD_DIR", "/tmp/hallu-check")


def generate_md_path() -> str:
    md_dir = Path(SCRAPED_MD_DIR)
    md_dir.mkdir(parents=True, exist_ok=True)
    return str(md_dir / f"hallu_{uuid.uuid4().hex[:12]}.md")


# ── NLI Model (DeBERTa-v3 trained on MNLI) ───
NLI_MODEL_PATH: str = os.getenv(
    "NLI_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "nli-deberta-v3-mnli" / "final"),
)
NLI_DEVICE: str = os.getenv("NLI_DEVICE", "auto")  # auto, mps, cuda, cpu
NLI_BATCH_SIZE: int = int(os.getenv("NLI_BATCH_SIZE", "16"))
USE_NLI_MODEL: bool = os.getenv("USE_NLI_MODEL", "true").lower() in ("true", "1", "yes")

# ── Self-Consistency Checking ─────────────────
ENABLE_SELF_CONSISTENCY: bool = os.getenv("ENABLE_SELF_CONSISTENCY", "true").lower() in ("true", "1", "yes")
N_CONSISTENCY_SAMPLES: int = int(os.getenv("N_CONSISTENCY_SAMPLES", "3"))

# ── Recursive Language Model (RLM) Reasoner ──
# Llama self-recursion for REASONING queries — zero extra paid API cost.
ENABLE_RLM_REASONING: bool = os.getenv("ENABLE_RLM_REASONING", "true").lower() in ("true", "1", "yes")

# ── Hallucination Detection ──────────────────
# Hallucination score threshold (0.0 = clean, 1.0 = fully hallucinated)
# Scores ABOVE this trigger refinement
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.3"))

# ── Iterative Convergent Refinement (ICR) ─────
# Multi-round verify→refine loop until hallucination score converges.
ENABLE_ICR: bool = os.getenv("ENABLE_ICR", "true").lower() in ("true", "1", "yes")
ICR_MAX_ROUNDS: int = int(os.getenv("ICR_MAX_ROUNDS", "3"))
ICR_CONVERGENCE_EPSILON: float = float(os.getenv("ICR_CONVERGENCE_EPSILON", "0.05"))

# ── Execution-Grounded Verification (EGV) ─────
# For REASONING queries: verify code/math via execution instead of NLI.
ENABLE_EGV: bool = os.getenv("ENABLE_EGV", "true").lower() in ("true", "1", "yes")
EGV_TIMEOUT: int = int(os.getenv("EGV_TIMEOUT", "10"))  # seconds per execution

# ── Surgical Correction ──────────────────────
# Replace only CONTRADICTED claims instead of rewriting the full answer.
ENABLE_SURGICAL_CORRECTION: bool = os.getenv("ENABLE_SURGICAL_CORRECTION", "true").lower() in ("true", "1", "yes")

