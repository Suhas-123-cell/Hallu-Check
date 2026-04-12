"""
hallu-check  |  main.py
FastAPI application — Agentic Multi-Hop Hallucination Detection Pipeline.

Node 0 (Gatekeeper) classifies each query, then routes it:

  FACTUAL  → Node 1 → Node 2 → Node 3 (C-RAG) → Node 4 → Node 5 → Node 6 → Node 7
  REASONING → Node 1 → Node 5 → Node 6 → Node 7 (skip web search)
  CHITCHAT  → Node 1 → Node 7 (return immediately)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional
import warnings

# Suppress annoying third-party deprecation warnings (from langchain/google)
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

import nltk  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException  # type: ignore[import-untyped, import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-untyped, import-not-found]
from pydantic import BaseModel, Field  # type: ignore[import-untyped, import-not-found]

# ── Config ────────────────────────────────────────────────────────────────────
from config import generate_md_path  # type: ignore[import-not-found]

# ── Node imports ──────────────────────────────────────────────────────────────
from nodes.gatekeeper import classify_query                   # type: ignore[import-not-found] # Node 0
from nodes.generator import generate_llm_output               # type: ignore[import-not-found] # Node 1
from nodes.web_search import extract_keywords, search_and_scrape  # type: ignore[import-not-found] # Nodes 2-3
from nodes.pageindex_rag import run_pageindex_rag_with_bertscore  # type: ignore[import-not-found] # Node 4+BERTScore
from nodes.claim_verifier import verify_claims                 # type: ignore[import-not-found] # Node 5
from nodes.refiner import refine_with_evidence, refine_response  # type: ignore[import-not-found] # Node 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("hallu-check")


# ── NLTK resource bootstrap ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Download NLTK data once at start-up."""
    logger.info("Downloading NLTK punkt tokenizer…")

    # ── Workaround for macOS Python SSL Certificate issues during NLTK download ──
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    logger.info("NLTK ready. Hallu-Check Agentic Pipeline is live.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hallu-Check — Agentic LLM Hallucination Detection API",
    description=(
        "Agentic multi-hop hallucination detection pipeline with semantic routing "
        "(Gatekeeper), Corrective RAG (C-RAG), claim-level NLI verification, "
        "and BERTScore."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        json_schema_extra={"examples": ["What is the capital of France?"]},
    )


class ClaimVerdictSchema(BaseModel):
    claim: str
    verdict: str           # SUPPORTED, CONTRADICTED, UNVERIFIABLE, NO_CLAIM, HONEST_UNCERTAINTY
    evidence: str          # RAG snippet backing the verdict
    confidence: float      # 0.0–1.0
    reasoning: str = ""    # Brief reasoning chain for the verdict


class GenerateResponse(BaseModel):
    query: str
    query_category: str                       # FACTUAL, REASONING, CHITCHAT
    llm_output: str
    rag_output: str
    bertscore: dict
    claim_verdicts: List[ClaimVerdictSchema]   # Per-claim breakdown
    hallucination_score: float                 # 0.0 (clean) → 1.0 (fully hallucinated)
    hallucination_detected: bool              # Clear boolean flag
    hallucination_summary: str                # Human-readable summary
    final_answer: str


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "service": "hallu-check", "version": "3.0.0"}


# ── CHITCHAT handler ─────────────────────────────────────────────────────────
async def _handle_chitchat(query: str) -> GenerateResponse:
    """Handle CHITCHAT queries: Node 1 → return immediately."""
    logger.info("── Route: CHITCHAT → Node 1 only")

    llm_output = await asyncio.to_thread(generate_llm_output, query)

    return GenerateResponse(
        query=query,
        query_category="CHITCHAT",
        llm_output=llm_output,
        rag_output="",
        bertscore={"precision": 1.0, "recall": 1.0, "f1": 1.0},
        claim_verdicts=[],
        hallucination_score=0.0,
        hallucination_detected=False,
        hallucination_summary="Conversational query — no fact-checking required.",
        final_answer=llm_output,
    )


# ── REASONING handler ────────────────────────────────────────────────────────
async def _handle_reasoning(query: str) -> GenerateResponse:
    """
    Handle REASONING queries: Node 1 → Node 5 → Node 6.
    Skips web search and PageIndex entirely.
    """
    logger.info("── Route: REASONING → Node 1 → Node 5 → Node 6")

    # ── Node 1: LLM Generator ────────────────────────────────────────
    logger.info("── Node 1: Generating answer with LLM…")
    llm_output = await asyncio.to_thread(generate_llm_output, query)

    # ── Synthetic context for reasoning queries ──────────────────────
    rag_output = "No external context required for reasoning/logic queries."
    bertscore = {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    # ── Node 5: Claim Verification (internal logic check) ────────────
    logger.info("── Node 5: Internal claim verification (no RAG context)…")
    hallu_report = await asyncio.to_thread(
        verify_claims,
        llm_output,
        rag_output,
        query,
        bertscore["f1"],
    )

    logger.info(
        "── Node 5: hallucination_score=%.4f  detected=%s  claims=%d",
        hallu_report.hallucination_score,
        hallu_report.hallucination_detected,
        len(hallu_report.claim_verdicts),
    )

    # ── Node 6: Refinement (only if hallucination detected) ──────────
    final_answer = llm_output
    report_dict = hallu_report.to_dict()

    if hallu_report.hallucination_detected:
        logger.info("── Node 6: Hallucination detected — refining…")
        try:
            refined = await asyncio.to_thread(
                refine_with_evidence,
                query,
                rag_output,
                report_dict,
                "REASONING",
            )
            if refined:
                final_answer = refined
        except Exception as refine_exc:
            logger.warning("── Node 6: Refinement failed (%s).", refine_exc)
    else:
        logger.info("── Node 6: No hallucination — keeping original answer.")

    return GenerateResponse(
        query=query,
        query_category="REASONING",
        llm_output=llm_output,
        rag_output=rag_output,
        bertscore=bertscore,
        claim_verdicts=[
            ClaimVerdictSchema(
                claim=cv.claim,
                verdict=cv.verdict,
                evidence=cv.evidence,
                confidence=cv.confidence,
                reasoning=cv.reasoning,
            )
            for cv in hallu_report.claim_verdicts
        ],
        hallucination_score=hallu_report.hallucination_score,
        hallucination_detected=hallu_report.hallucination_detected,
        hallucination_summary=hallu_report.summary,
        final_answer=final_answer,
    )


# ── FACTUAL handler ──────────────────────────────────────────────────────────
async def _handle_factual(query: str) -> GenerateResponse:
    """
    Handle FACTUAL queries: full pipeline.
    Node 1 → Node 2 → Node 3 (C-RAG) → Node 4 → Node 5 → Node 6 → Node 7.
    Uses UUID-based temp files for concurrent safety.
    """
    logger.info("── Route: FACTUAL → full pipeline")

    # Generate a unique temp markdown path for this request
    md_path = generate_md_path()

    try:
        # ── Node 1: LLM Generator ────────────────────────────────────
        logger.info("── Node 1: Generating preliminary answer with LLM…")
        llm_output = await asyncio.to_thread(generate_llm_output, query)

        # ── Node 2: Keyword Extraction ────────────────────────────────
        logger.info("── Node 2: Extracting search keywords…")
        keywords = await asyncio.to_thread(extract_keywords, query)

        # ── Node 3: Web Search + Scrape + C-RAG ──────────────────────
        logger.info("── Node 3: Searching & scraping (depth-2 + C-RAG)…")
        md_path, _ = await asyncio.to_thread(
            search_and_scrape,
            keywords,
            True,       # enable_depth2
            query,
            md_path,    # UUID-based temp file
        )

        # ── Node 4: PageIndex RAG + BERTScore ────────────────────────
        logger.info("── Node 4: Building tree index, retrieving context…")
        eval_result = await run_pageindex_rag_with_bertscore(md_path, query, llm_output)
        rag_output = eval_result["rag_output"]
        bertscore = eval_result["bertscore"]

        # ── Node 5: Claim-Level Verification ─────────────────────────
        logger.info("── Node 5: Claim extraction + NLI verification…")
        hallu_report = await asyncio.to_thread(
            verify_claims,
            llm_output,
            rag_output,
            query,
            bertscore["f1"],
        )
        report_dict = hallu_report.to_dict()

        logger.info(
            "── Node 5: hallucination_score=%.4f  detected=%s  claims=%d",
            hallu_report.hallucination_score,
            hallu_report.hallucination_detected,
            len(hallu_report.claim_verdicts),
        )

        # ── Node 6: Evidence-Based Refinement ────────────────────────
        final_answer = llm_output

        if hallu_report.hallucination_detected:
            logger.info("── Node 6: Hallucination detected — refining with evidence…")
            try:
                refined = await asyncio.to_thread(
                    refine_with_evidence,
                    query,
                    rag_output[:15000],
                    report_dict,
                    "FACTUAL",
                )
                if refined:
                    final_answer = refined
                else:
                    logger.warning(
                        "── Node 6: Evidence-based refinement empty. Trying basic…"
                    )
                    basic_refined = await asyncio.to_thread(
                        refine_response,
                        query,
                        rag_output[:15000],
                    )
                    if basic_refined:
                        final_answer = basic_refined
            except Exception as refine_exc:
                logger.warning(
                    "── Node 6: Refinement failed (%s). Using original.", refine_exc,
                )
        else:
            logger.info("── Node 6: No hallucination — keeping original answer.")

        # ── Node 7: Final Output ─────────────────────────────────────
        logger.info(
            "═══════ Pipeline complete | hallu_score=%.4f | BERTScore F1=%.4f ═══════",
            hallu_report.hallucination_score,
            bertscore["f1"],
        )

        return GenerateResponse(
            query=query,
            query_category="FACTUAL",
            llm_output=llm_output,
            rag_output=rag_output,
            bertscore=bertscore,
            claim_verdicts=[
                ClaimVerdictSchema(
                    claim=cv.claim,
                    verdict=cv.verdict,
                    evidence=cv.evidence,
                    confidence=cv.confidence,
                    reasoning=cv.reasoning,
                )
                for cv in hallu_report.claim_verdicts
            ],
            hallucination_score=hallu_report.hallucination_score,
            hallucination_detected=hallu_report.hallucination_detected,
            hallucination_summary=hallu_report.summary,
            final_answer=final_answer,
        )

    finally:
        # ── Cleanup: Remove UUID-based temp markdown file ─────────────
        try:
            if os.path.exists(md_path):
                os.remove(md_path)
                logger.debug("Cleaned up temp markdown: %s", md_path)
        except OSError as e:
            logger.warning("Failed to cleanup temp file %s: %s", md_path, e)


# ── Pipeline-level timeout (seconds) ─────────────────────────────────────────
_PIPELINE_TIMEOUT = 300  # 5 minutes — enough for massive inputs through all nodes

# ── Main pipeline endpoint ───────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse, tags=["pipeline"])
async def generate(payload: GenerateRequest) -> GenerateResponse:
    """
    Agentic hallucination-detection pipeline with semantic routing.

    **Node 0 (Gatekeeper)** classifies the query, then routes to:
    - **FACTUAL**: Full pipeline (Nodes 1→7) with web search, PageIndex RAG,
      claim verification, and refinement.
    - **REASONING**: Skip web search (Nodes 1→5→6) for logic/coding queries.
    - **CHITCHAT**: Immediate LLM response (Node 1 only).
    """
    query = payload.query
    logger.info("═══════ Pipeline start: %r ═══════", query)

    try:
        return await asyncio.wait_for(
            _run_pipeline(query), timeout=_PIPELINE_TIMEOUT
        )

    except asyncio.TimeoutError:
        logger.error("Pipeline timed out after %ds for query: %r", _PIPELINE_TIMEOUT, query[:100])
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {_PIPELINE_TIMEOUT}s. Try a shorter query.",
        )
    except ValueError as exc:
        # Payload-too-large errors raised by generator.py
        logger.warning("Pipeline payload error: %s", exc)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}",
        ) from exc


async def _run_pipeline(query: str) -> GenerateResponse:
    """Inner pipeline logic, separated for asyncio.wait_for timeout wrapping."""
    # ── Node 0: Gatekeeper — classify query ──────────────────────
    logger.info("── Node 0: Classifying query…")
    classification = await asyncio.to_thread(classify_query, query)
    category = classification["category"]
    confidence = classification["confidence"]
    logger.info(
        "── Node 0: Category=%s (confidence=%.2f)", category, confidence,
    )

    # ── Route based on classification ─────────────────────────────
    if category == "CHITCHAT":
        return await _handle_chitchat(query)
    elif category == "REASONING":
        return await _handle_reasoning(query)
    else:
        # FACTUAL (default — also catches unknown categories)
        return await _handle_factual(query)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn  # type: ignore[import-untyped, import-not-found]

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
