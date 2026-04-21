"""
hallu-check | nodes/pageindex_rag.py
Node 4 — PageIndex Vectorless RAG

Builds a hierarchical tree-structure index from scraped Markdown using
VectifyAI/PageIndex, then performs reasoning-based retrieval through
tree search using Gemini to extract the most relevant factual context.

CRUCIAL CONSTRAINT: No FAISS, Chroma, embeddings, or text chunking.

Integration approach:
  - PageIndex is vendored at <project>/PageIndex/
  - Its LLM layer (utils.py) calls OpenAI via ChatGPT_API_async()
  - We monkey-patch those functions at import time to use Gemini instead
  - We also patch count_tokens() to handle non-OpenAI model names

Retrieval follows the official PageIndex pattern (from their RAG cookbook):
  1. Build the tree with md_to_tree(if_add_node_summary='yes', if_add_node_text='yes')
  2. Strip 'text' from a copy of the tree → pass the lightweight tree to Gemini
  3. Gemini reasons over the tree structure + query → returns relevant node_ids
  4. Use a node_id → node mapping to collect the original text from those nodes
  5. Concatenated node texts = RAG Output
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── NLI-based Alignment (replaces BERTScore) ────────────────────────────────
# BERTScore measured word similarity, not factual correctness.
# NLI alignment measures actual entailment — semantically correct.

import tiktoken  # type: ignore[import-untyped]
from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]

from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-untyped]

logger = logging.getLogger("hallu-check.pageindex_rag")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Add PageIndex to sys.path so we can import it
# ─────────────────────────────────────────────────────────────────────────────
_PAGEINDEX_DIR = str(Path(__file__).resolve().parent.parent / "PageIndex")
if _PAGEINDEX_DIR not in sys.path:
    sys.path.insert(0, _PAGEINDEX_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monkey-patch PageIndex's LLM calls BEFORE importing pageindex modules
# ─────────────────────────────────────────────────────────────────────────────
def _setup_llm():
    """Validate HuggingFace API token."""
    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HF_API_TOKEN is not set. Add it to your .env file.\n"
            "Get one at https://huggingface.co/settings/tokens"
        )
    return


def _hf_chat(model_name: str, prompt: str, **_kwargs) -> str:
    """Synchronous HuggingFace call — replaces ChatGPT_API()."""
    _setup_llm()
    client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
    response = client.chat_completion(
        model=LOCAL_MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Follow instructions precisely."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.1,
    )
    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    return ""


async def _hf_chat_async(model_name: str, prompt: str, **_kwargs) -> str:
    """Async HuggingFace call — replaces ChatGPT_API_async().
    huggingface_hub has no native async, so we run in a thread pool.
    """
    _setup_llm()
    loop = asyncio.get_running_loop()

    def _call() -> str:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Follow instructions precisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    return await loop.run_in_executor(None, _call)


def _patched_count_tokens(text: str, model: str | None = None) -> int:
    """
    Drop-in for pageindex.utils.count_tokens().
    Handles non-OpenAI model names by falling back to cl100k_base.
    """
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model or "gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── Apply the monkey-patches ─────────────────────────────────────────────────
import pageindex.utils as _pi_utils  # type: ignore[import-not-found]  # noqa: E402

# The vendored PageIndex now uses litellm-based function names:
#   llm_completion (sync), llm_acompletion (async), count_tokens
_pi_utils.llm_completion = _hf_chat
_pi_utils.llm_acompletion = _hf_chat_async
_pi_utils.count_tokens = _patched_count_tokens

# Also patch old names in case any code path still references them
if hasattr(_pi_utils, "ChatGPT_API"):
    _pi_utils.ChatGPT_API = _hf_chat
if hasattr(_pi_utils, "ChatGPT_API_async"):
    _pi_utils.ChatGPT_API_async = _hf_chat_async

logger.info("PageIndex LLM layer monkey-patched → HuggingFace (%s)", LOCAL_MODEL_ID)

# ── Now safe to import PageIndex modules ─────────────────────────────────────
from pageindex.page_index_md import md_to_tree  # type: ignore[import-not-found]  # noqa: E402
from pageindex.utils import (  # type: ignore[import-not-found]  # noqa: E402
    remove_fields,
    structure_to_list,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: create_node_mapping (not in vendored utils — only in paid client)
# ─────────────────────────────────────────────────────────────────────────────
def create_node_mapping(tree_structure: Any) -> Dict[str, Dict]:
    """
    Flatten the tree into a dict keyed by node_id → node dict.
    This replicates the function available in the PageIndex API client
    but missing from the open-source vendored code.
    """
    mapping: Dict[str, Dict] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            nid = node.get("node_id")
            if nid:
                mapping[nid] = node
            for child in node.get("nodes", []):
                _walk(child)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(tree_structure)
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build the PageIndex tree
# ─────────────────────────────────────────────────────────────────────────────
async def build_tree_index(md_path: str) -> Dict[str, Any]:
    """
    Call PageIndex's md_to_tree() to build a hierarchical tree index
    with node summaries AND full text over the scraped Markdown file.
    """
    logger.info("Node 4 | Building PageIndex tree from: %s", md_path)
    
    # Verify the file exists and has content
    try:
        from pathlib import Path
        md_file = Path(md_path)
        if not md_file.exists():
            logger.error("Node 4 | Markdown file not found at %s", md_path)
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        
        file_size = md_file.stat().st_size
        if file_size == 0:
            logger.error("Node 4 | Markdown file is empty at %s", md_path)
            raise ValueError(f"Markdown file is empty: {md_path}")
        
        logger.info("Node 4 | Markdown file verified: %d bytes", file_size)
    except Exception as e:
        logger.error("Node 4 | File verification failed: %s", e)
        raise

    try:
        tree = await md_to_tree(
            md_path=md_path,
            if_thinning=False,
            if_add_node_summary="no",
            summary_token_threshold=200,
            model="gpt-4o-2024-11-20",  # Use open-source default
            if_add_doc_description="no",
            if_add_node_text="yes",       # keep text — needed for retrieval
            if_add_node_id="yes",
        )
    except Exception as e:
        logger.error("Node 4 | Tree building failed: %s", e)
        raise

    logger.info("Node 4 | Tree index built.  doc_name=%s", tree.get("doc_name"))
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reasoning-based Tree Search Retrieval (official PageIndex pattern)
#
#    From the PageIndex RAG cookbook (pageindex_RAG_simple.ipynb):
#      1. Strip 'text' from a copy of the tree (keep titles + summaries)
#      2. Send the lightweight tree + query to Gemini in ONE call
#      3. Gemini reasons and returns relevant node_ids as JSON
#      4. Look up the original text from a node_id → node mapping
# ─────────────────────────────────────────────────────────────────────────────
async def tree_search_retrieve(
    tree_structure: Dict[str, Any],
    query: str,
) -> str:
    """
    Perform reasoning-based retrieval via tree search.

    Follows the official PageIndex pattern: one LLM call with the
    full tree structure (sans text) to identify relevant nodes,
    then look up the original text from the selected node_ids.
    """
    structure = tree_structure.get("structure", tree_structure)

    # Build node_id → node mapping (with full text)
    node_map = create_node_mapping(structure)
    if not node_map:
        logger.warning("Node 4 | No nodes found in tree.")
        return "No relevant context found."

    # Strip 'text' from a copy → lightweight tree for the LLM prompt
    tree_without_text = remove_fields(copy.deepcopy(structure), fields=["text"])

    # Ask Gemini to reason over the tree and pick relevant nodes
    search_prompt = (
        "You are given a question and a tree structure of a document.\n"
        "Each node contains a node_id, node title, and possibly a summary.\n"
        "Your task is to find ALL nodes that are likely to contain the answer.\n\n"
        f"Question: {query}\n\n"
        f"Document tree structure:\n{json.dumps(tree_without_text, indent=2)}\n\n"
        "Please reply in the following JSON format:\n"
        '{\n'
        '    "thinking": "<Your reasoning about which nodes are relevant>",\n'
        '    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]\n'
        '}\n'
        "Directly return the final JSON structure. Do not output anything else."
    )

    try:
        raw_response = await _hf_chat_async("", search_prompt)
        logger.debug("Node 4 | Tree search raw response: %s", raw_response[:500])

        # Parse the JSON response
        parsed = None
        # Try extracting JSON from markdown code block first
        json_match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        if not parsed:
            try:
                parsed = json.loads(raw_response.strip())
            except json.JSONDecodeError:
                pass

        if parsed and isinstance(parsed, dict) and "node_list" in parsed:
            node_ids = parsed["node_list"]
            logger.info(
                "Node 4 | Gemini tree search selected %d node(s): %s",
                len(node_ids), node_ids,
            )

            # Collect text from the selected nodes
            selected_texts: List[str] = []
            for nid in node_ids:
                node = node_map.get(nid)
                if node and node.get("text", "").strip():
                    selected_texts.append(node["text"].strip())

            if selected_texts:
                rag_output = "\n\n---\n\n".join(selected_texts)
                logger.info(
                    "Node 4 | RAG output assembled (%d chars from %d selected nodes).",
                    len(rag_output), len(selected_texts),
                )
                return rag_output
            else:
                logger.warning(
                    "Node 4 | Selected node_ids had no text, falling back."
                )
        else:
            logger.warning(
                "Node 4 | Could not parse tree search response, falling back."
            )

    except Exception as e:
        logger.warning("Node 4 | Gemini tree search failed: %s, falling back.", e)

    # ── Fallback: return all node texts sorted by relevance to query ──
    logger.info("Node 4 | Using fallback: returning top nodes by keyword overlap.")
    all_nodes = structure_to_list(structure)
    query_words = set(query.lower().split())

    scored: List[tuple[float, str]] = []
    for n in all_nodes:
        text = (n.get("text") or "").strip()
        if not text:
            continue
        title = (n.get("title") or "").lower()
        text_lower = text.lower()
        # Score by keyword overlap in title + text
        score = sum(1 for w in query_words if w in title) * 3
        score += sum(1 for w in query_words if w in text_lower)
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_texts = [t for _, t in scored[:10] if _]  # take top 10 with score > 0

    if not top_texts:
        # Last resort: just take first few nodes
        top_texts = [
            n.get("text", "").strip()
            for n in all_nodes
            if n.get("text", "").strip()
        ][:3]

    if not top_texts:
        return "No relevant context found."

    rag_output = "\n\n---\n\n".join(top_texts)
    logger.info(
        "Node 4 | Fallback RAG output assembled (%d chars from %d nodes).",
        len(rag_output), len(top_texts),
    )
    return rag_output


# ─────────────────────────────────────────────────────────────────────────────
# 7. Alignment Scoring (NLI-based, replaces BERTScore)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_alignment(candidate: str, reference: str) -> dict:
    """
    Compute NLI-based alignment between candidate (LLM output) and
    reference (RAG context). Replaces BERTScore.

    Why NLI instead of BERTScore:
      - BERTScore measures word/embedding similarity
      - "Modi is President" scores HIGH against "Modi is PM" on BERTScore
      - NLI correctly classifies this as CONTRADICTION

    Returns:
        Dict with 'alignment_score', 'precision', 'recall', 'f1' (for
        backward compatibility), and 'method'.
    """
    try:
        from nodes.nli_model import compute_nli_alignment, is_loaded, load_model  # type: ignore

        if not is_loaded():
            load_model()

        if is_loaded():
            result = compute_nli_alignment(candidate, reference)
            score = result["alignment_score"]
            return {
                "precision": score,
                "recall": score,
                "f1": score,
                "alignment_score": score,
                "method": "nli",
                "details": result.get("details", []),
            }
    except Exception as e:
        logger.warning("NLI alignment failed (%s), using word-overlap fallback.", e)

    # Fallback: simple word overlap (Jaccard-like)
    candidate_words = set(candidate.lower().split())
    reference_words = set(reference.lower().split())
    if not candidate_words or not reference_words:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "alignment_score": 0.0, "method": "fallback"}
    intersection = candidate_words & reference_words
    union = candidate_words | reference_words
    jaccard = len(intersection) / len(union)
    return {
        "precision": jaccard,
        "recall": jaccard,
        "f1": jaccard,
        "alignment_score": jaccard,
        "method": "fallback",
    }


# Backward compatibility alias
def evaluate_bertscore(candidate: str, reference: str, **kwargs) -> dict:
    """DEPRECATED: Use evaluate_alignment() instead. Kept for backward compatibility."""
    return evaluate_alignment(candidate, reference)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public API — build + retrieve in one call
# ─────────────────────────────────────────────────────────────────────────────
async def run_pageindex_rag(md_path: str, query: str) -> str:
    """
    Node 4 — Full PageIndex RAG pipeline:
      1. Build hierarchical tree index from scraped Markdown
      2. Reasoning-based tree-search retrieval using Gemini
      3. Return the most relevant factual context (RAG Output)

    Args:
        md_path: Path to the scraped Markdown file (output of Node 3).
        query:   The user's original query.

    Returns:
        RAG Output string — factual context for hallucination comparison.
    """
    tree = await build_tree_index(md_path)
    rag_output = await tree_search_retrieve(tree, query)
    return rag_output

# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Pipeline: RAG + BERTScore Evaluation
# ─────────────────────────────────────────────────────────────────────────────
async def run_pageindex_rag_with_bertscore(
    md_path: str,
    query: str,
    llm_output: str,
    lang: str = "en",
    model_type: str = "distilbert-base-uncased",
) -> dict:
    """
    Run the full PageIndex RAG pipeline and evaluate the output with
    NLI-based alignment scoring (replaces BERTScore).

    The function name is preserved for backward compatibility but internally
    uses NLI alignment when the model is available.

    Args:
        md_path: Path to the scraped Markdown file (output of Node 3).
        query:   The user's original query.
        llm_output: LLM answer to compare against retrieved context.
        lang: (DEPRECATED) Ignored — kept for backward compatibility.
        model_type: (DEPRECATED) Ignored — kept for backward compatibility.
    Returns:
        Dict with 'rag_output' and 'bertscore' (precision, recall, f1,
        alignment_score, method).
    """
    rag_output = await run_pageindex_rag(md_path, query)
    alignment = evaluate_alignment(llm_output, rag_output)
    return {"rag_output": rag_output, "bertscore": alignment}
