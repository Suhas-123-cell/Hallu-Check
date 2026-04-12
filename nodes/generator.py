"""
hallu-check | nodes/generator.py
Node 1 — Original LLM Generator

Calls Qwen2.5-1.5B-Instruct via the HuggingFace Inference API
(serverless — no local GPU required) and returns a preliminary answer.

Compatible with huggingface_hub >= 1.0.0 (installed: 1.5.0).
The model is passed at call time (not in the constructor) per the
current InferenceClient API.
"""
from __future__ import annotations

import logging
from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
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
    Node 1 — call HuggingFace API to produce a fast preliminary answer.

    Args:
        query: The user's natural-language question.

    Returns:
        The model's raw text response (the LLM Output).
    """
    logger.info("Node 1 | Calling %s via HuggingFace API…", LOCAL_MODEL_ID)

    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HF_API_TOKEN is not set. Add it to your .env file.\n"
            "Get one at https://huggingface.co/settings/tokens"
        )

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
    Node 1 (contextual) — call HuggingFace API with grounded context.

    Args:
        query: The refined user question/prompt.
        context: Retrieved factual context to ground the answer.
        system_prompt: Optional system prompt override.

    Returns:
        The model's grounded text response.
    """
    logger.info("Node 6 | Calling %s via HuggingFace API (refined)…", LOCAL_MODEL_ID)

    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HF_API_TOKEN is not set. Add it to your .env file.\n"
            "Get one at https://huggingface.co/settings/tokens"
        )

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
        # Gracefully handle payload-too-large / context-window-exceeded errors
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