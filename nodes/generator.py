"""
hallu-check | nodes/generator.py
Node 1 — Original LLM Generator

Calls Llama 3.2 via Ollama (local, free) or HuggingFace Inference API
(fallback) and returns a preliminary answer.

Priority:
  1. Ollama (localhost:11434) — free, no quota, Metal-accelerated
  2. HuggingFace Inference API — free tier, may hit quota limits
"""
from __future__ import annotations

import logging
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore[import-untyped, import-not-found]

from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.generator")

# ── Node 1 ────────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_llm_output(query: str) -> str:
    """
    Node 1 — generate a preliminary LLM answer.

    Uses Ollama (local) as primary backend, falls back to HuggingFace API.

    Args:
        query: The user's natural-language question.

    Returns:
        The model's raw text response (the LLM Output).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, accurate assistant. "
                "Answer the user's question concisely and factually."
            ),
        },
        {"role": "user", "content": query},
    ]

    # ── Primary: Ollama (local, free) ────────────────────────────────
    try:
        from nodes.local_llm import chat_completion, is_available  # type: ignore[import-not-found]
        if is_available():
            llm_output = chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
            )
            if llm_output:
                logger.info("Node 1 | LLM output received via Ollama (%d chars).", len(llm_output))
                return llm_output
    except Exception as e:
        logger.warning("Node 1 | Ollama failed: %s", str(e)[:120])

    # ── Fallback: HuggingFace Inference API ──────────────────────────
    logger.info("Node 1 | Falling back to HuggingFace API (%s)…", LOCAL_MODEL_ID)

    if not HF_API_TOKEN:
        raise EnvironmentError(
            "Neither Ollama nor HF_API_TOKEN available. "
            "Start Ollama with: brew services start ollama\n"
            "Or set HF_API_TOKEN in your .env file."
        )

    from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
    client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)

    try:
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
        )
    except Exception as e:
        error_str = str(e)
        # Gracefully handle payload-too-large / context-window-exceeded errors
        if any(code in error_str for code in ("413", "422", "payload too large", "context window", "token limit")):
            logger.error(
                "Node 1 | Input too large for model context window: %s", error_str[:200]
            )
            raise ValueError(
                "Your query is too large for the model's context window. "
                "Please shorten your input and try again."
            ) from e
        logger.error("HuggingFace API Error: %s", error_str)
        raise

    llm_output = ""
    if response.choices and response.choices[0].message.content:
        llm_output = response.choices[0].message.content.strip()

    if not llm_output:
        raise ValueError("Node 1 | Received empty response from HuggingFace model.")

    logger.info("Node 1 | LLM output received (%d chars).", len(llm_output))
    return llm_output


def generate_llm_output_with_context(
    query: str,
    context: str,
    system_prompt: str | None = None,
) -> str:
    """
    Node 1 (contextual) — generate an answer grounded in RAG context.

    Args:
        query: The refined user question/prompt.
        context: Retrieved factual context to ground the answer.
        system_prompt: Optional system prompt override.

    Returns:
        The model's grounded text response.
    """
    base_system_prompt = (
        system_prompt
        or "You are a factual question-answering assistant. "
        "You MUST answer ONLY using the facts in the context provided below. "
        "NEVER use your own training knowledge. "
        "If the context names a person, use EXACTLY that name in your answer. "
        "If the context does not contain the answer, say 'Not found in context.'"
    )

    user_prompt = (
        "=========== CONTEXT (use ONLY this) ===========\n"
        f"{context}\n"
        "=========== END CONTEXT ===========\n\n"
        f"Question: {query}\n\n"
        "IMPORTANT: Your answer MUST come from the context above. "
        "Do NOT use prior knowledge. Extract the answer from the context and respond concisely."
    )

    messages = [
        {"role": "system", "content": base_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── Primary: Ollama (local, free) ────────────────────────────────
    try:
        from nodes.local_llm import chat_completion, is_available  # type: ignore[import-not-found]
        if is_available():
            llm_output = chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.2,
                top_p=0.9,
            )
            if llm_output:
                logger.info("Node 6 | Refined LLM output via Ollama (%d chars).", len(llm_output))
                return llm_output
    except Exception as e:
        logger.warning("Node 6 | Ollama failed: %s", str(e)[:120])

    # ── Fallback: HuggingFace Inference API ──────────────────────────
    logger.info("Node 6 | Falling back to HuggingFace API (%s)…", LOCAL_MODEL_ID)

    if not HF_API_TOKEN:
        raise EnvironmentError(
            "Neither Ollama nor HF_API_TOKEN available."
        )

    from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
    client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)

    try:
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.9,
        )
    except Exception as e:
        error_str = str(e)
        if any(code in error_str for code in ("413", "422", "payload too large", "context window", "token limit")):
            logger.error(
                "Node 6 | Input too large for model context window: %s", error_str[:200]
            )
            raise ValueError(
                "The refined prompt is too large for the model's context window. "
                "Please shorten your input and try again."
            ) from e
        logger.error("HuggingFace API Error (refined): %s", error_str)
        raise

    llm_output = ""
    if response.choices and response.choices[0].message.content:
        llm_output = response.choices[0].message.content.strip()

    if not llm_output:
        raise ValueError("Node 6 | Received empty response from HuggingFace model.")

    logger.info("Node 6 | Refined LLM output received (%d chars).", len(llm_output))
    return llm_output