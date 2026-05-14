from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable, List, Optional, Tuple

from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore[import-untyped, import-not-found]

from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]
from nodes.tools.python_exec import run_python  # type: ignore[import-not-found]

logger = logging.getLogger("hallu-check.recursive_reasoner")

_MIN_SUB_QUESTIONS = 2
_MAX_SUB_QUESTIONS = 4
_LEAF_TIMEOUT = 90
_ROOT_TIMEOUT = 120

# Regex for <python>...</python> blocks emitted by the leaf model.
_PYTHON_BLOCK_RE = re.compile(r"<python>\s*([\s\S]*?)\s*</python>", re.IGNORECASE)
# Regex for <rag>...</rag> blocks — queries the shared PageIndex tree.
_RAG_BLOCK_RE = re.compile(r"<rag>\s*([\s\S]*?)\s*</rag>", re.IGNORECASE)
# Cap the number of tool calls per leaf to bound cost of a runaway response.
_MAX_EXEC_PER_LEAF = 3
_MAX_RAG_PER_LEAF = 3

# A callable that answers a RAG sub-query against a pre-built tree.
# Returns the retrieved text, or an empty string on miss.
TreeQueryFn = Callable[[str], str]


def _client() -> InferenceClient:
    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HF_API_TOKEN is not set. Add it to your .env file.\n"
            "Get one at https://huggingface.co/settings/tokens"
        )
    return InferenceClient(api_key=HF_API_TOKEN, timeout=_ROOT_TIMEOUT)


def _extract_json_array(text: str) -> List[str] | None:
    # Prefer a fenced ```json block if present
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        bracket = re.search(r"\[[\s\S]*?\]", text)
        candidate = bracket.group(0) if bracket else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    subs = [str(x).strip() for x in parsed if isinstance(x, (str, int, float)) and str(x).strip()]
    return subs or None


# ── Step 1: DECOMPOSE ────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
def _decompose(query: str) -> List[str]:
    system = (
        "You are a reasoning planner. Break the user's problem into the smallest "
        "number of independent sub-questions that, when answered, let you solve "
        "the original problem. Each sub-question must be self-contained — do NOT "
        "reference other sub-questions by number or assume their answers are known. "
        f"Produce between {_MIN_SUB_QUESTIONS} and {_MAX_SUB_QUESTIONS} sub-questions."
    )
    user = (
        f"Original problem:\n{query}\n\n"
        "Output ONLY a JSON array of strings — no prose, no keys, no comments.\n"
        'Example: ["What is X?", "How does X relate to Y?", "Given X and Y, what is Z?"]'
    )

    client = _client()
    response = client.chat_completion(
        model=LOCAL_MODEL_ID,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
    )
    raw = ""
    if response.choices and response.choices[0].message.content:
        raw = response.choices[0].message.content.strip()

    subs = _extract_json_array(raw)
    if not subs:
        logger.warning("RLM | Decomposition returned no parseable sub-questions. Raw: %s", raw[:200])
        return []

    # Clamp to allowed range
    subs = subs[:_MAX_SUB_QUESTIONS]
    logger.info("RLM | Decomposed into %d sub-questions.", len(subs))
    for i, sq in enumerate(subs, 1):
        logger.info("RLM |   [%d] %s", i, sq[:120])
    return subs


# ── Step 2: SOLVE LEAVES (parallel) ──────────────────────────────────────────
_PYTHON_TOOL_DOC = (
    "TOOL <python>: For ANY arithmetic, algebra, combinatorics, or numeric "
    "computation, you MUST emit a Python block instead of computing by hand. "
    "Small models make numeric errors; Python does not. Format:\n"
    "<python>\n"
    "# your code — use print() to show results\n"
    "print(17 * 23)\n"
    "</python>\n"
    "Available: math, fractions, decimal, statistics, itertools, functools, "
    "and sympy (symbols, solve, simplify, Rational, Eq).\n"
    "Do NOT guess what the Python will output — wait for the result.\n"
)

_RAG_TOOL_DOC = (
    "TOOL <rag>: For any factual question (names, dates, events, definitions) "
    "you are NOT certain about, issue a retrieval query against the document "
    "index. Format:\n"
    "<rag>\n"
    "your focused sub-query here\n"
    "</rag>\n"
    "The retrieved text will be inserted in place of the block. Do NOT "
    "answer factual sub-questions from your own memory if this tool is "
    "available — prefer retrieval.\n"
)


def _build_leaf_system(has_python: bool, has_rag: bool) -> str:
    parts = [
        "You are a precise reasoner. Answer the single question below "
        "concisely and correctly.\n"
    ]
    if has_python:
        parts.append(_PYTHON_TOOL_DOC)
    if has_rag:
        parts.append(_RAG_TOOL_DOC)
    parts.append(
        "For pure conceptual questions requiring no tool, answer directly in "
        "prose. If the question asks for a number, end with 'Answer: <value>'."
    )
    return "\n".join(parts)


def _exec_python_blocks(draft: str) -> tuple[str, int]:
    blocks = _PYTHON_BLOCK_RE.findall(draft)
    if not blocks:
        return draft, 0
    result = draft
    count = 0
    for code in blocks[:_MAX_EXEC_PER_LEAF]:
        res = run_python(code)
        rendered = res.render()
        result = _PYTHON_BLOCK_RE.sub(lambda _m, r=rendered: r, result, count=1)
        count += 1
    return result, count


def _exec_rag_blocks(draft: str, tree_query: TreeQueryFn) -> tuple[str, int]:
    blocks = _RAG_BLOCK_RE.findall(draft)
    if not blocks:
        return draft, 0
    result = draft
    count = 0
    for q in blocks[:_MAX_RAG_PER_LEAF]:
        q_clean = q.strip()
        try:
            retrieved = tree_query(q_clean) if q_clean else ""
        except Exception as e:
            logger.warning("RLM | tree_query failed (%s): %s", q_clean[:60], e)
            retrieved = ""
        # Truncate to keep the finalize-pass prompt manageable
        retrieved = (retrieved or "").strip()[:2000]
        rendered = (
            f"[rag result for query: {q_clean!r}]\n{retrieved}"
            if retrieved
            else f"[rag result: no relevant context found for {q_clean!r}]"
        )
        result = _RAG_BLOCK_RE.sub(lambda _m, r=rendered: r, result, count=1)
        count += 1
    return result, count


def _solve_leaf(sub_question: str, tree_query: Optional[TreeQueryFn] = None) -> str:
    has_python = True
    has_rag = tree_query is not None
    leaf_system = _build_leaf_system(has_python, has_rag)

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=_LEAF_TIMEOUT)

        draft_resp = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": leaf_system},
                {"role": "user", "content": sub_question},
            ],
            max_tokens=1024,
            temperature=0.2,
            top_p=0.9,
        )
        draft = ""
        if draft_resp.choices and draft_resp.choices[0].message.content:
            draft = draft_resp.choices[0].message.content.strip()
        if not draft:
            return ""

        substituted, py_count = _exec_python_blocks(draft)
        rag_count = 0
        if tree_query is not None:
            substituted, rag_count = _exec_rag_blocks(substituted, tree_query)

        total_tool_calls = py_count + rag_count
        if total_tool_calls == 0:
            return draft

        logger.info(
            "RLM | Leaf used %d tool(s) [py=%d, rag=%d] for: %s",
            total_tool_calls, py_count, rag_count, sub_question[:80],
        )

        finalize_system = (
            "You previously drafted a solution that used tools (Python and/or "
            "document retrieval). The real tool results are shown below. Using "
            "ONLY those results plus your own reasoning, produce a concise final "
            "answer. Do NOT recompute or second-guess tool output. Do NOT invent "
            "facts that are not in the retrieved text. If the question asks for "
            "a number, end with 'Answer: <value>'."
        )
        finalize_user = (
            f"Question: {sub_question}\n\n"
            f"Your draft with real tool results substituted:\n{substituted}\n\n"
            "Write the final answer now."
        )
        final_resp = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": finalize_system},
                {"role": "user", "content": finalize_user},
            ],
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )
        if final_resp.choices and final_resp.choices[0].message.content:
            return final_resp.choices[0].message.content.strip()
        return substituted

    except Exception as e:
        logger.warning("RLM | Leaf call failed for sub-question (%s): %s", sub_question[:60], e)
    return ""


async def _solve_leaves_parallel(
    sub_questions: List[str],
    tree_query: Optional[TreeQueryFn] = None,
) -> List[str]:
    tasks = [asyncio.to_thread(_solve_leaf, sq, tree_query) for sq in sub_questions]
    return await asyncio.gather(*tasks)


# ── Step 3: COMPOSE ──────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
def _compose(query: str, pairs: List[Tuple[str, str]], original_answer: str) -> str:
    evidence_block = "\n\n".join(
        f"Sub-question {i}: {q}\nSub-answer {i}: {a if a else '[no answer produced]'}"
        for i, (q, a) in enumerate(pairs, 1)
    )

    system = (
        "You are a careful reasoner composing a final answer from verified "
        "sub-answers. Rules:\n"
        "1. Build the final answer using ONLY the sub-answers below.\n"
        "2. If two sub-answers contradict each other, state the contradiction "
        "explicitly and choose the one with clearer supporting work.\n"
        "3. If the sub-answers are insufficient, say so — do NOT invent facts.\n"
        "4. Be concise. No preamble."
    )
    user = (
        f"Original problem:\n{query}\n\n"
        f"Sub-questions and sub-answers (your only evidence):\n{evidence_block}\n\n"
        f"(For reference only — an earlier single-shot attempt produced:\n{original_answer[:800]}\n"
        "Do not trust it; use it only to notice what might be missing.)\n\n"
        "Write the final answer now."
    )

    client = _client()
    response = client.chat_completion(
        model=LOCAL_MODEL_ID,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
    )
    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    return ""


# ── Public entry point ───────────────────────────────────────────────────────
async def recursive_reason(
    query: str,
    original_answer: str,
    tree_query: Optional[TreeQueryFn] = None,
) -> str:
    logger.info(
        "Node 1.5 | Recursive reasoning start (rag_tool=%s).",
        "on" if tree_query else "off",
    )

    try:
        sub_questions = await asyncio.to_thread(_decompose, query)
    except Exception as e:
        logger.warning("Node 1.5 | Decomposition failed: %s — falling back.", e)
        return original_answer

    if len(sub_questions) < _MIN_SUB_QUESTIONS:
        logger.info("Node 1.5 | Too few sub-questions (%d) — falling back.", len(sub_questions))
        return original_answer

    sub_answers = await _solve_leaves_parallel(sub_questions, tree_query)
    pairs = list(zip(sub_questions, sub_answers))

    # If every leaf failed, bail
    if not any(a for a in sub_answers):
        logger.warning("Node 1.5 | All leaf calls failed — falling back.")
        return original_answer

    try:
        composed = await asyncio.to_thread(_compose, query, pairs, original_answer)
    except Exception as e:
        logger.warning("Node 1.5 | Composition failed: %s — falling back.", e)
        return original_answer

    if not composed:
        logger.warning("Node 1.5 | Empty composition — falling back.")
        return original_answer

    logger.info("Node 1.5 | Recursive reasoning complete (%d chars).", len(composed))
    return composed
