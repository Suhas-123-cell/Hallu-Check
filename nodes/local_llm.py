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
    global _client
    if _client is None:
        from openai import OpenAI  # type: ignore[import-untyped]
        _client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't need a real key
        )
    return _client


def is_available() -> bool:
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
