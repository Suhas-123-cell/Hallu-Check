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
SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "3"))

# ── PageIndex / RAG ───────────────────────────
# Directory for UUID-based temp markdown files (concurrent-safe)
SCRAPED_MD_DIR: str = os.getenv("SCRAPED_MD_DIR", "/tmp/hallu-check")


def generate_md_path() -> str:
    """Generate a unique, concurrent-safe markdown file path."""
    md_dir = Path(SCRAPED_MD_DIR)
    md_dir.mkdir(parents=True, exist_ok=True)
    return str(md_dir / f"hallu_{uuid.uuid4().hex[:12]}.md")


# ── Hallucination Detection ──────────────────
# Hallucination score threshold (0.0 = clean, 1.0 = fully hallucinated)
# Scores ABOVE this trigger refinement
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.3"))

