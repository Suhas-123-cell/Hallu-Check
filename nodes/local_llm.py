from __future__ import annotations

import logging

import httpx

from config import LLM_BACKEND, OLLAMA_MODEL, OLLAMA_URL  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.local_llm")

_ollama_available: bool | None = None


def is_available() -> bool:
    """True when the Ollama server is up AND OLLAMA_MODEL is pulled."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        _ollama_available = OLLAMA_MODEL in names
        if _ollama_available:
            logger.info("local_llm | Ollama available (%s @ %s)", OLLAMA_MODEL, OLLAMA_URL)
        else:
            logger.warning("local_llm | Ollama is up but %s is not pulled (run: ollama pull %s)",
                           OLLAMA_MODEL, OLLAMA_MODEL)
    except Exception as e:
        logger.warning("local_llm | Ollama not available: %s", str(e)[:100])
        _ollama_available = False
    if LLM_BACKEND == "ollama" and not _ollama_available:
        # Never silently switch models mid-experiment.
        raise RuntimeError(
            f"LLM_BACKEND=ollama but {OLLAMA_MODEL} is not available at {OLLAMA_URL}. "
            f"Start it with `ollama serve` and `ollama pull {OLLAMA_MODEL}`."
        )
    return _ollama_available


def chat_completion(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    from nodes.llm_client import OllamaChatClient, make_client  # type: ignore[import-not-found]

    from config import LOCAL_MODEL_ID  # type: ignore[import-not-found]

    client = OllamaChatClient(timeout=180) if is_available() else make_client(timeout=60)
    try:
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
        if LLM_BACKEND == "ollama":
            raise
        logger.warning("local_llm | chat completion failed: %s", str(e)[:120])
    return ""
