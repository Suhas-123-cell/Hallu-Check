"""
Single backend switch for every small-model call in the pipeline
(generator, gatekeeper, keyword extraction, C-RAG relevance, PageIndex tree
search, self-consistency, RLM, EGV).

    LLM_BACKEND=ollama  every call goes to the local Ollama model OLLAMA_MODEL.
                        The `model=` argument callers pass is ignored, so one
                        fixed set of weights is used everywhere (reproducible).
    LLM_BACKEND=hf      HuggingFace Inference API with the caller's model id.

`make_client(timeout)` returns an object with the same
`.chat_completion(model=, messages=, max_tokens=, temperature=, top_p=)`
interface and `.choices[0].message.content` response shape as
huggingface_hub.InferenceClient, so call sites only change one line.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx

from config import (  # type: ignore[import-not-found]
    HF_API_TOKEN,
    LLM_BACKEND,
    LOCAL_MODEL_ID,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_URL,
)


class OllamaChatClient:
    """Native /api/chat (not the OpenAI-compatible endpoint) so that num_ctx can
    be set; Ollama's default context silently truncates long RAG prompts."""

    def __init__(self, timeout: float = 120, model: Optional[str] = None) -> None:
        self.timeout = timeout
        self.model = model or OLLAMA_MODEL  # override only for graders, never pipeline nodes

    def chat_completion(
        self,
        model: Optional[str] = None,  # ignored on purpose: one model everywhere
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        options: Dict[str, Any] = {"num_ctx": OLLAMA_NUM_CTX}
        for key, value in (("num_predict", max_tokens), ("temperature", temperature),
                           ("top_p", top_p), ("stop", stop), ("seed", seed)):
            if value is not None:
                options[key] = value
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages or [], "stream": False,
                  "options": options},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_client(timeout: float = 120) -> Any:
    if LLM_BACKEND == "ollama":
        return OllamaChatClient(timeout)
    if LLM_BACKEND != "hf":
        raise ValueError(f"LLM_BACKEND must be 'ollama' or 'hf', got {LLM_BACKEND!r}")
    from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
    return InferenceClient(api_key=HF_API_TOKEN, timeout=timeout)


def backend_label() -> str:
    return f"ollama:{OLLAMA_MODEL}" if LLM_BACKEND == "ollama" else f"hf:{LOCAL_MODEL_ID}"
