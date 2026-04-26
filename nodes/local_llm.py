"""
hallu-check | nodes/local_llm.py
Local LLM inference via Ollama (OpenAI-compatible API)

Provides a unified interface for all LLM calls in the pipeline:
  - Code generation (Node 1)
  - Test case generation (EGV)
  - Claim classification (Stage 2 fallback)

Stack:  Ollama (localhost:11434) → Llama 3.2 3B → Apple Metal acceleration
Cost:   $0.00 — everything runs on your MacBook.

Falls back to HuggingFace Inference API if Ollama is not running.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("hallu-check.local_llm")

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ── Singleton client ─────────────────────────────────────────────────────────
_client = None
_ollama_available: Optional[bool] = None


def _get_client():
    """Lazy-init an OpenAI client pointing at Ollama."""
    global _client
    if _client is None:
        from openai import OpenAI  # type: ignore[import-untyped]
        _client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't need a real key
        )
    return _client


def is_available() -> bool:
    """Check if Ollama is running and the model is loaded."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available

    try:
        client = _get_client()
        # Quick health check — tiny completion
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        _ollama_available = bool(response.choices)
        if _ollama_available:
            logger.info("local_llm | Ollama available (%s @ %s)", OLLAMA_MODEL, OLLAMA_BASE_URL)
        return _ollama_available
    except Exception as e:
        logger.warning("local_llm | Ollama not available: %s", str(e)[:100])
        _ollama_available = False
        return False


def chat_completion(
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Run a chat completion via Ollama (local) or HuggingFace (fallback).

    Args:
        messages:    OpenAI-format message list.
        max_tokens:  Maximum output tokens.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling.

    Returns:
        The model's response text, or empty string on failure.
    """
    # ── Primary: Ollama (local, free) ────────────────────────────────────
    if is_available():
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("local_llm | Ollama call failed: %s", str(e)[:120])

    # ── Fallback: HuggingFace Inference API ──────────────────────────────
    try:
        from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
        from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]

        if HF_API_TOKEN:
            client = InferenceClient(api_key=HF_API_TOKEN, timeout=60)
            response = client.chat_completion(
                model=LOCAL_MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("local_llm | HuggingFace fallback failed: %s", str(e)[:120])

    return ""
