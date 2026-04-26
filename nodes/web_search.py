"""
hallu-check | nodes/web_search.py
Nodes 2 & 3 — Keyword Extractor → Web Search → Scraper → Markdown

Node 2: Extract 3-5 concise search keywords from the query using Gemini.
Node 3: Search the web for those keywords using googlesearch-python,
        fetch the top URLs, scrape their text, and write everything 
        to a single Markdown file (needed by PageIndex which reads .md files).

Uses googlesearch-python for web search (no API key needed).
Includes Name Entity Recognition for better handling of person/place queries.
"""

from __future__ import annotations

import logging
import re
import textwrap
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import httpx  # type: ignore[import-untyped]
from bs4 import BeautifulSoup  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from config import (  # type: ignore
    SEARCH_MAX_RESULTS,
    generate_md_path,
)
import random
import time
from ddgs import DDGS  # type: ignore[import-untyped]
import nltk  # type: ignore[import-untyped]
from nltk.chunk import ne_chunk  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]
from nltk import pos_tag  # type: ignore[import-untyped]
# ─────────────────────────────────────────────────────────────────────────────
# Name Entity Recognition (NER) for better query handling
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(text: str) -> dict:
    """
    Extract named entities (PERSON, LOCATION, ORGANIZATION) from text.
    Useful for identifying names and places in queries.
    """
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ne_tree = ne_chunk(pos_tags)
        
        entities = {
            "PERSON": [],
            "LOCATION": [],
            "ORGANIZATION": [],
            "GPE": [],
        }
        
        for subtree in ne_tree:
            if hasattr(subtree, "label"):
                entity_type = subtree.label()
                entity_name = " ".join([word for word, tag in subtree.leaves()])
                if entity_type in entities:
                    entities[entity_type].append(entity_name)
        
        return entities
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
        return {"PERSON": [], "LOCATION": [], "ORGANIZATION": [], "GPE": []}


def has_person_name(keywords: List[str]) -> bool:
    """
    Check if any keyword is a person name (using NER).
    """
    combined = " ".join(keywords)
    entities = extract_entities(combined)
    has_person = len(entities.get("PERSON", [])) > 0
    if has_person:
        logger.info(f"Node 2 | Detected person name(s): {entities['PERSON']}")
    return has_person



def _wants_recent_info(query: str | None, keywords: List[str]) -> bool:
    """Heuristic to detect if user likely needs up-to-date/current information."""
    hay = " ".join([(query or ""), *keywords]).lower()
    freshness_terms = [
        "latest",
        "current",
        "today",
        "now",
        "recent",
        "as of",
        "present",
        "incumbent",
    ]
    # "who is" queries about people are usually factoid lookups, not news
    # Only treat political/positional queries as needing recency
    positional_terms = ["cm ", "chief minister", "president", "prime minister", "governor", "ceo"]
    if any(t in hay for t in positional_terms):
        return True
    return any(term in hay for term in freshness_terms)


def _wants_factual_origin(query: str | None, keywords: List[str]) -> bool:
    """
    Heuristic to detect if the user is asking about origins, inventions,
    definitions, or historical facts — queries where Wikipedia excels.
    """
    hay = " ".join([(query or ""), *keywords]).lower()
    origin_terms = [
        "invented", "inventor", "invent",
        "created", "creator", "create",
        "founded", "founder", "found",
        "discovered", "discoverer", "discover",
        "origin", "history of", "etymology",
        "what is", "what are", "what was",
        "who invented", "who created", "who discovered",
        "who founded", "who developed", "who proposed",
        "when was", "where was",
        "first proposed", "first used", "first introduced",
        "algorithm", "technique", "method",
        "definition", "concept",
    ]
    return any(term in hay for term in origin_terms)


def _score_result_recency(
    title: str, url: str, snippet: str,
    prefer_recent: bool = False,
    prefer_factual: bool = False,
) -> int:
    """Context-aware domain relevance score for ranking web results.

    Scoring adapts based on query type:
      - Recency queries: boost news/gov, penalize Wikipedia
      - Factual/origin queries: boost Wikipedia + encyclopedic sources
      - Neutral queries: no Wikipedia penalty or boost
    """
    score = 0
    text = f"{title} {url} {snippet}".lower()
    year_now = datetime.now().year
    for yr in (year_now, year_now - 1, year_now - 2):
        if str(yr) in text:
            score += max(1, 5 - (year_now - yr) * 2)

    url_lower = url.lower()

    # Favor news/government domains for current-affairs style queries
    if any(
        dom in url_lower
        for dom in (
            "gov.in",
            "nic.in",
            "thehindu.com",
            "indianexpress.com",
            "hindustantimes.com",
            "timesofindia.com",
            "bbc.com",
            "reuters.com",
            "economictimes.com",
        )
    ):
        score += 2

    # Wikipedia scoring: context-dependent
    if "wikipedia.org" in url_lower:
        if prefer_factual:
            score += 3   # Strong boost for factual/origin queries
        elif prefer_recent:
            score -= 1   # Slight penalty for recency queries
        # else: neutral — no change

    # Encyclopedic / academic sources also get a boost for factual queries
    if prefer_factual and any(
        dom in url_lower
        for dom in (
            "britannica.com",
            "scholarpedia.org",
            "stanford.edu",
            "arxiv.org",
        )
    ):
        score += 2

    return score



logger = logging.getLogger("hallu-check.web_search")

# ── Login-walled domains that never yield useful scraped content ───────────
_UNSCRAPABLE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
}


def _is_unscrapable_domain(url: str) -> bool:
    """Check if a URL belongs to a login-walled domain that can't be scraped."""
    try:
        netloc = urlparse(url).netloc.lower()
        return any(d in netloc for d in _UNSCRAPABLE_DOMAINS)
    except Exception:
        return False


def _search_fallback(
    search_query: str, max_results: int
) -> List[Tuple[str, str]]:
    """
    Fallback method: Try again with simplified query.

    Used only when the primary search call fails or returns no results.
    """
    logger.info("Node 3 | Retrying with fallback (simplified query)…")
    results: List[Tuple[str, str]] = []
    seen_urls: set[str] = set()

    try:
        short_query = " ".join(search_query.split()[0:3])
        hits = list(DDGS().text(short_query, max_results=min(max_results, 10)))
        for r in hits:
            title = r.get("title", "Untitled")
            href = r.get("href", "")
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)
            results.append((title, href))
            if len(results) >= max_results:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Node 3 | Fallback search failed: %s", exc)

    return results


# ── LLM for keyword extraction (uses HuggingFace / Qwen) ─────────────────────
from huggingface_hub import InferenceClient  # type: ignore[import-untyped, import-not-found]
from config import HF_API_TOKEN, LOCAL_MODEL_ID  # type: ignore[import-not-found]


def _hf_extract_keywords(prompt: str) -> str:
    """Use the HuggingFace / Qwen model for keyword extraction (no Gemini needed)."""
    if not HF_API_TOKEN:
        logger.warning("Node 2 | HF_API_TOKEN not set, cannot extract keywords.")
        return ""

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a keyword extraction assistant. "
                        "Extract search keywords and return ONLY a comma-separated list. "
                        "No explanation, no numbering, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""
    except Exception as e:
        logger.warning("Node 2 | HuggingFace keyword extraction failed: %s", e)
        return ""


# ─{70}
# Node 2 — Keyword Extractor (uses HuggingFace / Qwen)
# ─{70}
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=1, max=5),
    reraise=True,
)
def _strip_filler_words(words: List[str]) -> List[str]:
    """Remove common question/filler words, keeping only meaningful terms."""
    filler = {
        "who", "what", "where", "when", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "do", "does", "did", "can", "could", "would", "should",
        "tell", "me", "about", "please", "find", "search", "look", "up",
    }
    return [w for w in words if w.lower() not in filler]


def extract_keywords(query: str) -> List[str]:
    """
    Node 2 — extract search-friendly keywords from the query.

    For short/simple queries (≤4 meaningful words), uses the query directly
    to avoid the LLM mangling names or injecting garbage. For longer queries,
    uses HuggingFace/Qwen to distill keywords.

    Returns:
        A list of keyword strings (1-5 items).
    """
    logger.info("Node 2 | Extracting search keywords from query…")

    # Clean the query
    clean_query = query.strip().rstrip("?!.")
    query_words = clean_query.split()
    meaningful_words = _strip_filler_words(query_words)

    # ── Short query fast path ─────────────────────────────────────────
    # For short queries (like "sam leteps" or "who is sam leteps"),
    # just use the meaningful words directly — no LLM needed.
    if len(meaningful_words) <= 4:
        # Use the meaningful words as a single keyword phrase
        # (preserves exact spelling of names)
        keywords = [" ".join(meaningful_words)] if meaningful_words else query_words[:5]
        logger.info("Node 2 | Keywords (direct): %s", keywords)
        return keywords

    # ── Longer queries → use LLM to distill ───────────────────────────
    prompt = (
        "Extract 3 to 5 search keywords from this query. "
        "Return ONLY a comma-separated list. "
        "Keep all names exactly as spelled.\n\n"
        "Query: " + query
    )

    raw: str | None = None
    try:
        import concurrent.futures

        def call_llm() -> str:
            return _hf_extract_keywords(prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_llm)
            try:
                raw = future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Node 2 | HuggingFace keyword extraction timed out, "
                    "using fallback"
                )
                raw = None
    except Exception as e:
        logger.warning(
            "Node 2 | Keyword extraction failed: %s, using fallback", e
        )
        raw = None

    raw_str = (raw or "").strip()
    if not raw_str:
        # Fallback: use meaningful words from the query
        keywords = [" ".join(meaningful_words)] if meaningful_words else query_words[:5]
        logger.info("Node 2 | Keywords (fallback): %s", keywords)
        return keywords

    # Parse comma-separated keywords; strip extra whitespace / quotes
    keywords = [
        kw.strip().strip('"').strip("'")
        for kw in raw_str.split(",")
        if kw.strip()
    ]

    # Filter out garbage: remove any keyword that looks like a prompt instruction
    # (the small LLM sometimes extracts words from the prompt itself)
    instruction_words = {
        "extract", "keywords", "query", "return", "only", "comma",
        "separated", "list", "never", "alter", "correct", "re-spell",
        "copy", "exactly", "duplicate", "critical", "rules", "keep",
        "names", "spelled", "search",
    }
    keywords = [
        kw for kw in keywords
        if kw.lower() not in instruction_words
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            deduped.append(kw)
    keywords = deduped[:5]
    if not keywords:
        keywords = [" ".join(meaningful_words)] if meaningful_words else query_words[:5]

    # ── Name preservation ─────────────────────────────────────────────
    # Detect named entities in the query and ensure they survive
    # the LLM keyword extraction with exact spelling.
    query_lower = clean_query.lower()

    # Find the "subject" of the query (the name or entity being asked about)
    subject: str | None = None
    # Pattern 1: "who/what is X" → X is the subject
    for trigger in ["who is", "who's", "what is", "what's", "about", "tell me about"]:
        idx = query_lower.find(trigger)
        if idx != -1:
            after = clean_query[idx + len(trigger):].strip()
            if after:
                subject = after
                break
    # Pattern 2: Short query is likely just a name/entity itself
    if not subject and len(meaningful_words) <= 3:
        subject = " ".join(meaningful_words)

    if subject and len(subject) > 1:
        subject_lower = subject.lower()
        # Check if the subject appears (exactly) in at least one keyword
        subject_in_keywords = any(subject_lower in kw.lower() for kw in keywords)
        if not subject_in_keywords:
            logger.warning(
                "Node 2 | LLM altered subject '%s' in keywords %s, injecting original",
                subject, keywords,
            )
            # Put the original subject first, keep other useful keywords
            keywords = [subject] + [
                kw for kw in keywords if subject_lower not in kw.lower()
            ][:4]

    logger.info("Node 2 | Keywords extracted: %s", keywords)
    return keywords


# ─{70}
# Node 3 helpers — Web Search + Scraper
# ─{70}
def _web_search(
    keywords: List[str], max_results: int, query: str | None = None
) -> List[Tuple[str, str]]:
    """
    Search using ddgs (DuckDuckGo Search, no API key required).
    Returns (title, url) pairs.

    Strategy:
      1. Search with the original query first (most reliable for names/entities)
      2. If that fails, search with extracted keywords
      3. Final fallback: simplified query (first 3 words)
    """
    prefer_recent = _wants_recent_info(query, keywords)
    prefer_factual = _wants_factual_origin(query, keywords)
    year_hint = str(datetime.now().year) if prefer_recent else ""

    # Build search queries to try in order of priority
    search_queries: List[str] = []

    # Priority 1: The original user query (preserves exact spelling)
    if query:
        q1 = query.strip()
        if prefer_recent:
            q1 = f"{q1} {year_hint}".strip()
        search_queries.append(q1)

    # Priority 2: Keywords-based query
    kw_query = " ".join([*keywords, "latest" if prefer_recent else "", year_hint]).strip()
    if kw_query and kw_query not in search_queries:
        search_queries.append(kw_query)

    # Priority 3: For person names, also search LinkedIn/profile sites
    is_person = has_person_name(keywords)
    if is_person or (query and any(t in query.lower() for t in ["who is", "who's"])):
        name = " ".join(keywords)
        search_queries.append(f"{name} linkedin")
        search_queries.append(f"{name} site:linkedin.com")

    # Priority 4: For factual/origin queries, inject Wikipedia as a source
    if prefer_factual and not prefer_recent:
        kw_str = " ".join(keywords)
        wiki_query = f"{kw_str} site:wikipedia.org"
        if wiki_query not in search_queries:
            search_queries.append(wiki_query)
            logger.info("Node 3 | Injecting Wikipedia search for factual query.")

    scored_results: List[Tuple[int, str, str]] = []
    seen_urls: set[str] = set()

    for search_query in search_queries:
        if len(scored_results) >= max_results * 2:
            break  # collect more candidates, rank later

        logger.info("Node 3 | Web search: %r", search_query)
        try:
            hits = list(DDGS().text(
                search_query,
                max_results=min(max_results * 2, 20),
            ))
            for r in hits:
                title = r.get("title", "Untitled")
                item_url = r.get("href", "")
                snippet = r.get("body", "") or ""
                if not item_url or item_url in seen_urls:
                    continue
                if _is_unscrapable_domain(item_url):
                    logger.debug("Node 3 | Skipping unscrapable domain: %s", item_url)
                    continue
                seen_urls.add(item_url)
                score = _score_result_recency(
                    title, item_url, snippet,
                    prefer_recent=prefer_recent,
                    prefer_factual=prefer_factual,
                )
                scored_results.append((score, title, item_url))

            if scored_results:
                logger.info(
                    "Node 3 | Web search succeeded with %d result(s).",
                    len(scored_results),
                )
        except Exception as e:
            logger.warning("Node 3 | Web search failed for %r: %s", search_query, e)

    # Fallback with simplified query
    if not scored_results:
        fallback_q = query or " ".join(keywords)
        fallback_results = _search_fallback(fallback_q, max_results=max_results * 2)
        for title, url in fallback_results:
            if _is_unscrapable_domain(url):
                continue
            scored_results.append((
                _score_result_recency(
                    title, url, "",
                    prefer_recent=prefer_recent,
                    prefer_factual=prefer_factual,
                ),
                title, url,
            ))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    results = [(title, url) for _, title, url in scored_results[:max_results]]

    logger.info("Node 3 | Got %d result(s) from web search.", len(results))
    return results


def _is_unwanted_secondary_link(url: str, anchor_text: str) -> bool:
    url_l = url.lower().strip()
    anchor_l = anchor_text.lower().strip()

    if not url_l.startswith("http"):
        return True

    blocked_markers = [
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "youtube.com",
        "whatsapp",
        "mailto:",
        "/share",
        "?share=",
    ]
    # Note: linkedin.com is intentionally NOT blocked — useful for person lookups
    if any(marker in url_l for marker in blocked_markers):
        return True

    if any(anchor_l.startswith(prefix) for prefix in ("share", "tweet", "follow")):
        return True

    if any(url_l.endswith(ext) for ext in (".jpg", ".png", ".gif", ".svg", ".pdf", ".zip")):
        return True

    return False


# ── Content quality gate (login-wall / boilerplate detection) ─────────────
_LOGIN_WALL_INDICATORS = [
    "log in", "login to continue", "sign up", "sign in",
    "create an account", "you must log in", "please sign in",
    "accept cookies", "cookie policy", "this content isn't available",
    "content not available", "page not found", "access denied",
    "javascript is required", "enable javascript",
    "we value your privacy", "privacy settings",
]


def _is_boilerplate_content(text: str, min_sentences: int = 3) -> bool:
    """
    Detect login-wall boilerplate, cookie walls, or empty content that
    shouldn't be indexed.  Returns True if the text is garbage.
    """
    if not text or len(text.strip()) < 200:
        return True

    text_lower = text.lower()

    # Check for login-wall / cookie-wall indicators (2+ = likely a wall)
    login_hits = sum(1 for p in _LOGIN_WALL_INDICATORS if p in text_lower)
    if login_hits >= 2:
        return True

    # Real content has multiple informational sentences (>20 chars each)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    if len(sentences) < min_sentences:
        return True

    return False


def _extract_clean_text(html_text: str, max_chars: int = 8000) -> str:
    """Extract clean readable text from HTML."""
    soup = BeautifulSoup(html_text or "", "lxml")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Prefer <article> or <main>; fall back to <body>
    main = soup.find("article") or soup.find("main") or soup.body
    if main is None:
        return ""

    text = main.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars] if text else ""


def _fetch_html(
    url: str,
    timeout: float = 120.0,
    mirror_timeout: float = 30.0,
    use_mirror: bool = True,
) -> str:
    """Fetch HTML quickly with optional mirror fallback for blocked pages."""
    headers = {
        "User-Agent": random.choice(
            [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
            ]
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    request_timeout = httpx.Timeout(
        timeout,
        connect=min(3.0, timeout),
        read=timeout,
        write=min(3.0, timeout),
        pool=min(2.0, timeout),
    )

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=request_timeout,
            headers=headers,
            verify=False,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as exc:
        logger.debug("Node 3 | Primary fetch failed for %s: %s", url, exc)

    if not use_mirror:
        return ""

    # Mirror fallback is useful for anti-bot blocks; keep it strictly time-bounded.
    mirror_target = url if url.startswith(("http://", "https://")) else f"http://{url}"
    fallback_url = f"https://r.jina.ai/{mirror_target}"
    logger.info("Node 3 | Retrying via mirror: %s", fallback_url)

    try:
        mirror_req_timeout = httpx.Timeout(
            mirror_timeout,
            connect=min(1.5, mirror_timeout),
            read=mirror_timeout,
            write=min(1.5, mirror_timeout),
            pool=min(1.0, mirror_timeout),
        )
        with httpx.Client(
            follow_redirects=True,
            timeout=mirror_req_timeout,
            headers=headers,
            verify=False,
        ) as client:
            mirror_resp = client.get(fallback_url)
            mirror_resp.raise_for_status()
            return mirror_resp.text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Node 3 | Failed to scrape %s: %s", url, exc)
        return ""


def _scrape_url(url: str, timeout: float = 120.0, use_mirror: bool = True) -> str:
    """
    Fetch a URL and extract clean readable text using BeautifulSoup.
    Returns an empty string on any error.
    """
    try:
        html_text = _fetch_html(url, timeout=timeout, use_mirror=use_mirror)
        if not html_text:
            return ""

        text = _extract_clean_text(html_text, max_chars=8000)
        if not text:
            logger.debug("Node 3 | Empty text extracted from %s", url)
            return ""

        if _is_boilerplate_content(text):
            logger.debug("Node 3 | Boilerplate/login-wall content at %s, discarding", url)
            return ""

        logger.debug("Node 3 | Scraped %d chars from %s", len(text), url)
        return text[:8000]  # type: ignore[index]

    except Exception as exc:  # noqa: BLE001
        logger.warning("Node 3 | Failed to scrape %s: %s", url, exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Depth-2 Crawling Functions (for enriched context retrieval)
# ─────────────────────────────────────────────────────────────────────────────


def extract_links(html: str, base_url: str) -> List[Tuple[str, str]]:
    """
    Extract all <a> tags from HTML and return (href, anchor_text) pairs.
    Handles relative URLs by converting them to absolute URLs.

    Args:
        html: Raw HTML content of a page.
        base_url: Base URL for resolving relative links.

    Returns:
        List of tuples: (absolute_url, anchor_text_lowercase).
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = str(a_tag.get("href", "")).strip()
            if not href or href.startswith("#"):
                continue
            # Convert relative URLs to absolute
            if href.startswith("/"):
                href = urljoin(base_url, href)
            elif not href.startswith("http"):
                href = urljoin(base_url, href)
            anchor_text = a_tag.get_text(strip=True).lower()
            links.append((href, anchor_text))
        return links
    except Exception as e:
        logger.debug("Node 3 | Failed to extract links from %s: %s", base_url, e)
        return []


def filter_links_by_keywords(
    links: List[Tuple[str, str]], keywords: List[str], max_links: int = 5
) -> List[str]:
    """
    STRICT SEMANTIC FILTER: Only keep links where the URL or anchor text
    contains at least ONE of the original search keywords.

    This prevents context bloat by:
      1. Checking both URL and anchor text against ALL keywords
      2. Requiring at least ONE keyword match (AND logic)
      3. Limiting to max_links to prevent excessive secondary links
      4. Avoiding duplicates

    Args:
        links: List of (url, anchor_text) tuples.
        keywords: Original search keywords.
        max_links: Maximum number of links to return (default: 5).

    Returns:
        Filtered list of URLs (up to max_links).
    """
    core_keywords = set(kw.lower() for kw in keywords)
    filtered = []

    for url, anchor_text in links:
        if _is_unwanted_secondary_link(url, anchor_text):
            continue
        url_lower = url.lower()
        # Check if ANY keyword appears in URL or anchor text
        if any(kw in url_lower or kw in anchor_text for kw in core_keywords):
            if url not in filtered:  # Deduplication
                filtered.append(url)
                if len(filtered) >= max_links:
                    break

    return filtered


def crawl_secondary_content(
    primary_url: str,
    keywords: List[str],
    timeout: float = 3.5,
    max_secondary_per_page: int = 2,
    primary_html: str | None = None,
) -> List[str]:
    """
    Visit secondary (linked) pages from a primary page and extract their text.
    Implements strict semantic filtering and rate limiting.

    Args:
        primary_url: The primary page URL to extract links from.
        keywords: Original search keywords for semantic filtering.
        timeout: HTTP timeout per request (seconds).
        max_secondary_per_page: Max secondary links to crawl per primary page.

    Returns:
        List of secondary page texts (empty list if no valid secondary links).
    """
    secondary_texts = []

    try:
        if not primary_html:
            primary_html = _fetch_html(
                primary_url,
                timeout=timeout,
                mirror_timeout=min(2.0, timeout),
                use_mirror=False,
            )
            if not primary_html:
                return []

        # Extract all links from primary page
        links = extract_links(primary_html, primary_url)
        logger.debug("Node 3 | Extracted %d links from %s", len(links), primary_url)

        # Apply strict semantic filter: keep only keyword-relevant links
        filtered_urls = filter_links_by_keywords(links, keywords, max_links=5)
        logger.debug("Node 3 | Filtered to %d relevant secondary links", len(filtered_urls))

        # Limit to max_secondary_per_page to prevent context bloat
        filtered_urls = filtered_urls[:max_secondary_per_page]  # type: ignore

        def scrape_secondary(secondary_url: str) -> str:
            try:
                return _scrape_url(
                    secondary_url,
                    timeout=min(timeout, 3.0),
                    use_mirror=False,
                )[:4000]
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Node 3 | Failed to crawl secondary link %s: %s",
                    secondary_url,
                    exc,
                )
                return ""

        if filtered_urls:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(3, len(filtered_urls))
            ) as executor:
                futures = [executor.submit(scrape_secondary, u) for u in filtered_urls]
                for future in concurrent.futures.as_completed(futures):
                    text = future.result()
                    if text:
                        secondary_texts.append(text)

    except Exception as e:
        logger.debug("Node 3 | Depth-2 crawl failed for %s: %s", primary_url, e)

    return secondary_texts


def _build_markdown_with_depth2(results: List[dict]) -> str:
    """
    Build markdown from consolidated primary + secondary content.

    Each result dict contains:
      - title: Page title
      - primary_url: Primary page URL
      - primary_text: Main text from primary page
      - secondary_context: List of texts from secondary (linked) pages

    Args:
        results: List of result dictionaries.

    Returns:
        Consolidated markdown string ready for PageIndex ingestion.
    """
    parts: List[str] = ["# Retrieved Web Sources (with Depth-2 Context)\n\n"]

    for result in results:
        title = result["title"]
        primary_url = result["primary_url"]
        primary_text = result["primary_text"]
        secondary_texts = result.get("secondary_context", [])

        if not primary_text:
            continue

        # Primary section
        parts.append(f"## {title}\n\n")
        parts.append(f"**Source:** {primary_url}\n\n")
        parts.append(textwrap.indent(primary_text, "> "))

        # Secondary section (if depth-2 content exists)
        if secondary_texts:
            parts.append("\n\n### Related Content (Depth-2):\n\n")
            for i, sec_text in enumerate(secondary_texts, 1):
                parts.append(f"**Related Link {i}:**\n\n")
                # Cap secondary content to 1000 chars each to prevent bloat
                parts.append(textwrap.indent(sec_text[:1000], "> "))
                parts.append("\n\n")

        parts.append("\n\n")  # blank line between sections

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Corrective RAG (C-RAG) — Chunk Relevance Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_chunk_relevance(text: str, query: str) -> bool:
    """
    C-RAG: Evaluate whether a scraped text chunk is relevant to the query.

    Uses Llama 3.2-1B via HuggingFace Inference API to score relevance.
    A chunk is deemed relevant if the LLM says YES.

    Args:
        text: The scraped text chunk (truncated for token limits).
        query: The user's original query.

    Returns:
        True if the chunk is relevant, False otherwise.
    """
    if not text or len(text.strip()) < 50:
        return False

    # Truncate chunk to avoid token limits on small model
    chunk_preview = text[:1500]

    prompt = (
        "You are a relevance evaluator. Does the following text contain "
        "information that is useful for answering the given question?\n\n"
        f"Question: {query}\n\n"
        f"Text:\n{chunk_preview}\n\n"
        "Answer with ONLY 'YES' or 'NO'. Nothing else."
    )

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a strict relevance judge. Answer ONLY 'YES' or 'NO'."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.1,
        )

        raw = ""
        if response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content.strip().upper()

        is_relevant = "YES" in raw
        logger.debug(
            "C-RAG | Chunk relevance for %r: %s (raw: %r)",
            query[:40], "RELEVANT" if is_relevant else "IRRELEVANT", raw,
        )
        return is_relevant

    except Exception as e:
        logger.warning("C-RAG | Relevance evaluation failed (%s), keeping chunk.", e)
        # On failure, keep the chunk (conservative — avoid losing data)
        return True


def _rewrite_query(query: str, keywords: List[str]) -> List[str]:
    """
    C-RAG: Rewrite the search keywords when initial scrape yields zero
    relevant chunks.

    Uses Llama 3.2-1B to generate an alternative set of search terms
    that might retrieve better results.

    Args:
        query: The user's original query.
        keywords: The original keywords that failed.

    Returns:
        A new list of keywords (falls back to original if rewrite fails).
    """
    logger.info("C-RAG | All chunks irrelevant. Rewriting query…")

    prompt = (
        "The following search keywords did not find relevant results:\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"Original question: {query}\n\n"
        "Generate 3-5 alternative search keywords that might find better results. "
        "Return ONLY a comma-separated list. No explanation."
    )

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, timeout=120)
        response = client.chat_completion(
            model=LOCAL_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a search query optimizer. Return ONLY comma-separated keywords."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
            temperature=0.3,
        )

        raw = ""
        if response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content.strip()

        if not raw:
            return keywords

        new_keywords = [kw.strip().strip('"').strip("'") for kw in raw.split(",") if kw.strip()]
        if not new_keywords:
            return keywords

        logger.info("C-RAG | Rewritten keywords: %s", new_keywords)
        return new_keywords[:5]

    except Exception as e:
        logger.warning("C-RAG | Query rewrite failed (%s), using original keywords.", e)
        return keywords


def _build_markdown(
    results: List[Tuple[str, str]],
    texts: List[str],
    query: str | None = None,
) -> Tuple[str, bool]:
    """
    Combine scraped pages into a single Markdown document with C-RAG filtering.
    PageIndex reads this file to build its hierarchical tree index.

    When a query is provided, each chunk is evaluated for relevance using
    the C-RAG evaluator. Irrelevant chunks are discarded.

    Args:
        results: List of (title, url) tuples.
        texts: Corresponding scraped text for each result.
        query: The user's original query (for C-RAG evaluation).

    Returns:
        Tuple of (markdown_string, all_irrelevant_flag).
        all_irrelevant_flag is True if every chunk was filtered out.
    """
    parts: List[str] = ["# Retrieved Web Sources\n\n"]
    kept_count = 0

    for (title, url), text in zip(results, texts):
        if not text:
            continue

        # C-RAG: evaluate chunk relevance before including
        if query:
            if not _evaluate_chunk_relevance(text, query):
                logger.info("C-RAG | Discarded irrelevant chunk: %s", title[:60])
                continue

        # Each page becomes a level-2 section with scraped text as a blockquote
        parts.append(f"## {title}\n\n")
        parts.append(f"**Source:** {url}\n\n")
        # textwrap.indent prefixes every line — works for multiline text
        parts.append(textwrap.indent(text, "> "))
        parts.append("\n\n")  # blank line between sections
        kept_count += 1

    all_irrelevant = (kept_count == 0)
    if all_irrelevant:
        logger.warning("C-RAG | All %d chunks were irrelevant to query.", len(texts))

    return "".join(parts), all_irrelevant



# ─{70}
# Node 3 — Depth-2 Enhanced Search & Scrape
# ─{70}
def search_and_scrape_with_depth2(
    keywords: List[str], query: str | None = None, md_path: str | None = None,
) -> Tuple[str, str]:
    """
    Node 3 — Enhanced search with depth-2 crawling.

    Fetches top 20 results from Google (primary pages), then for each:
      1. Extracts all linked pages
      2. Filters by semantic relevance (keyword presence)
      3. Crawls 3-5 filtered secondary links per primary page
      4. Consolidates primary + secondary content in markdown

    Args:
        keywords: Output of Node 2's extract_keywords().

    Returns:
        Tuple of:
          - md_path (str): absolute path to the saved Markdown file.
          - markdown_text (str): the full markdown content with depth-2 context.
    """
    if not keywords:
        logger.error("Node 3 | No keywords provided")
        raise ValueError("Node 3 requires keywords from Node 2")

    if not md_path:
        md_path = generate_md_path()

    results = _web_search(
        keywords,
        max_results=max(SEARCH_MAX_RESULTS, 6),
        query=query,
    )

    if not results:
        logger.warning("Node 3 | No search results found.")
        fallback = (
            "# No web results found\n\n"
            "The web search returned no results. Using fallback context."
        )
        try:
            md_file = Path(md_path)
            md_file.parent.mkdir(parents=True, exist_ok=True)
            md_file.write_text(fallback, encoding="utf-8")
            logger.info("Node 3 | Fallback markdown saved to %s", md_path)
        except Exception as e:
            logger.error("Node 3 | Failed to save fallback markdown: %s", e)
        return md_path, fallback

    logger.info(
        "Node 3 | Depth-2 Crawling: Scraping %d primary URLs with secondary context…",
        len(results),
    )

    def process_primary(item: Tuple[str, str]) -> dict | None:
        title, url = item
        try:
            primary_html = _fetch_html(
                url,
                timeout=120.0,
                mirror_timeout=30.0,
                use_mirror=True,
            )
            if not primary_html:
                return None

            primary_text = _extract_clean_text(primary_html, max_chars=8000)
            if not primary_text:
                return None

            if _is_boilerplate_content(primary_text):
                logger.debug("Node 3 | Boilerplate content from %s, skipping", url)
                return None

            secondary_texts = crawl_secondary_content(
                primary_url=url,
                keywords=keywords,
                timeout=30.0,
                max_secondary_per_page=2,
                primary_html=primary_html,
            )

            return {
                "title": title,
                "primary_url": url,
                "primary_text": primary_text,
                "secondary_context": secondary_texts,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Node 3 | Error processing %s: %s", url, exc)
            return None

    enriched_results: List[dict] = []
    max_workers = min(4, len(results)) if results else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_primary, item) for item in results]
        for future in concurrent.futures.as_completed(futures):
            processed = future.result()
            if processed:
                enriched_results.append(processed)

    if not enriched_results:
        logger.warning("Node 3 | All URLs failed to scrape, using fallback")
        fallback = f"# Failed to scrape web results\n\nTried {len(results)} URLs but all scraping failed."
        try:
            md_file = Path(md_path)
            md_file.parent.mkdir(parents=True, exist_ok=True)
            md_file.write_text(fallback, encoding="utf-8")
        except Exception as e:
            logger.error("Node 3 | Failed to save error markdown: %s", e)
        return md_path, fallback

    # Build consolidated markdown with depth-2 context
    markdown = _build_markdown_with_depth2(enriched_results)

    # Save to disk — PageIndex will read from this path
    try:
        md_file = Path(md_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(markdown, encoding="utf-8")
        logger.info(
            "Node 3 | Depth-2 markdown saved → %s (%d chars from %d primary URLs).",
            md_path,
            len(markdown),
            len(enriched_results),
        )
    except Exception as e:
        logger.error("Node 3 | Failed to save markdown to %s: %s", md_path, e)
        raise

    return md_path, markdown


# ─{70}
# Node 3 — Main entry point
# ─{70}
def search_and_scrape(
    keywords: List[str],
    enable_depth2: bool = False,
    query: str | None = None,
    md_path: str | None = None,
) -> Tuple[str, str]:
    """
    Node 3 — search the web, scrape results, and write a Markdown file.

    Supports optional depth-2 crawling for enriched context retrieval.

    Args:
        keywords: Output of Node 2's extract_keywords().
        enable_depth2: If True, use depth-2 crawling with secondary link extraction.
                      If False, use basic depth-1 crawling (faster, less context).

    Returns:
        Tuple of:
          - md_path (str): absolute path to the saved Markdown file.
          - markdown_text (str): the full markdown content.
    """
    if enable_depth2:
        return search_and_scrape_with_depth2(keywords, query=query, md_path=md_path)

    # Original depth-1 logic
    if not keywords:
        logger.error("Node 3 | No keywords provided")
        raise ValueError("Node 3 requires keywords from Node 2")

    if not md_path:
        md_path = generate_md_path()

    results = _web_search(
        keywords,
        max_results=max(SEARCH_MAX_RESULTS, 6),
        query=query,
    )

    if not results:
        logger.warning("Node 3 | No search results found.")
        fallback = (
            "# No web results found\n\n"
            "The web search returned no results. Using fallback context."
        )
        try:
            md_file = Path(md_path)
            md_file.parent.mkdir(parents=True, exist_ok=True)
            md_file.write_text(fallback, encoding="utf-8")
            logger.info(
                "Node 3 | Fallback markdown saved to %s", md_path
            )
        except Exception as e:
            logger.error("Node 3 | Failed to save fallback markdown: %s", e)
        return md_path, fallback

    logger.info("Node 3 | Scraping %d URL(s)…", len(results))
    texts = [_scrape_url(url) for _, url in results]

    # Filter out empty texts
    valid_results = [
        (title, url) for (title, url), text in zip(results, texts) if text
    ]
    valid_texts = [text for text in texts if text]

    if not valid_texts:
        logger.warning("Node 3 | All URLs failed to scrape, using fallback")
        fallback = (
            "# Failed to scrape web results\n\n"
            f"Tried {len(results)} URLs but all scraping failed."
        )
        try:
            md_file = Path(md_path)
            md_file.parent.mkdir(parents=True, exist_ok=True)
            md_file.write_text(fallback, encoding="utf-8")
        except Exception as e:
            logger.error("Node 3 | Failed to save error markdown: %s", e)
        return md_path, fallback

    # C-RAG: Build markdown with chunk relevance evaluation
    markdown, all_irrelevant = _build_markdown(valid_results, valid_texts, query=query)

    # C-RAG: If all chunks were irrelevant, rewrite query and retry ONCE
    if all_irrelevant and query:
        logger.info("C-RAG | Triggering query rewrite and re-search…")
        new_keywords = _rewrite_query(query, keywords)

        retry_results = _web_search(
            new_keywords,
            max_results=max(SEARCH_MAX_RESULTS, 6),
            query=query,
        )

        if retry_results:
            retry_texts = [_scrape_url(url) for _, url in retry_results]
            retry_valid_results = [
                (title, url) for (title, url), text in zip(retry_results, retry_texts) if text
            ]
            retry_valid_texts = [text for text in retry_texts if text]

            if retry_valid_texts:
                # Rebuild markdown with C-RAG filtering on the retry results
                markdown, _ = _build_markdown(
                    retry_valid_results, retry_valid_texts, query=query
                )
                logger.info(
                    "C-RAG | Re-search produced %d chunks after filtering.",
                    len(retry_valid_texts),
                )

    # Save to disk — PageIndex will read from this path
    try:
        md_file = Path(md_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(markdown, encoding="utf-8")
        logger.info(
            "Node 3 | Scraped markdown saved → %s (%d chars from %d URLs).",
            md_path,
            len(markdown),
            len(valid_results),
        )
    except Exception as e:
        logger.error(
            "Node 3 | Failed to save markdown to %s: %s", md_path, e
        )
        raise

    return md_path, markdown


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Gap Recovery — Targeted Search for Unverifiable Claims
# ─────────────────────────────────────────────────────────────────────────────
def targeted_gap_search(
    unverifiable_claims: List[str],
    query: str,
    max_results_per_claim: int = 3,
) -> str:
    """
    Lightweight targeted search to fill knowledge gaps for unverifiable claims.

    When Node 5 (claim verification) finds UNVERIFIABLE claims, this function
    performs focused searches specifically for the missing information, then
    returns supplementary context as a markdown string.

    IMPORTANT: We search using the ORIGINAL USER QUERY, not the hallucinated
    claims. The claims are from the LLM's wrong answer — using them as search
    queries would just find more wrong information.

    This is much lighter than a full search_and_scrape():
      - No depth-2 crawling
      - No C-RAG evaluation
      - No PageIndex tree building
      - Just quick DuckDuckGo search + scrape for the specific gaps

    Args:
        unverifiable_claims: List of claim strings that couldn't be verified.
        query: The user's original query (for context).
        max_results_per_claim: Max search results to fetch per search query.

    Returns:
        Supplementary context as a markdown string (may be empty if nothing found).
    """
    if not unverifiable_claims:
        return ""

    logger.info(
        "Gap Recovery | Searching for %d unverifiable claim(s)…",
        len(unverifiable_claims),
    )

    # Build smart search queries from the ORIGINAL QUERY, not the hallucinated claims
    search_queries: List[str] = []

    # Priority 1: The original user query (most likely to find the right answer)
    clean_query = query.strip().rstrip("?!.")
    search_queries.append(clean_query)

    # Priority 2: Wikipedia-targeted search
    search_queries.append(f"{clean_query} site:wikipedia.org")

    # Priority 3: Add "origin" / "history" / "inventor" context if not already present
    query_lower = clean_query.lower()
    if any(t in query_lower for t in ["invented", "invent", "who", "creator", "origin"]):
        # Query already has these terms — try an encyclopedic version
        search_queries.append(f"{clean_query} encyclopedia")
    else:
        search_queries.append(f"{clean_query} origin history")

    supplementary_parts: List[str] = [
        "# Supplementary Context (Gap Recovery)\n\n"
    ]
    seen_urls: set[str] = set()
    found_count = 0

    for search_query in search_queries:
        if found_count >= max_results_per_claim:
            break

        logger.info("Gap Recovery | Searching: %r", search_query[:100])

        try:
            hits = list(DDGS().text(
                search_query,
                max_results=max_results_per_claim * 2,
            ))

            for r in hits:
                if found_count >= max_results_per_claim:
                    break
                title = r.get("title", "Untitled")
                item_url = r.get("href", "")
                if not item_url or item_url in seen_urls:
                    continue
                if _is_unscrapable_domain(item_url):
                    continue
                seen_urls.add(item_url)

                # Quick scrape — short timeout, no mirror fallback
                text = _scrape_url(item_url, timeout=10.0, use_mirror=False)
                if not text or len(text.strip()) < 100:
                    continue

                # Truncate to keep context manageable
                text = text[:4000]

                supplementary_parts.append(f"## {title}\n\n")
                supplementary_parts.append(f"**Source:** {item_url}\n\n")
                supplementary_parts.append(textwrap.indent(text, "> "))
                supplementary_parts.append("\n\n")
                found_count += 1

        except Exception as e:
            logger.warning("Gap Recovery | Search failed for %r: %s", search_query[:60], e)

        # Small delay between searches
        time.sleep(random.uniform(0.3, 0.8))

    if found_count == 0:
        logger.info("Gap Recovery | No supplementary context found.")
        return ""

    result = "".join(supplementary_parts)
    logger.info(
        "Gap Recovery | Found %d supplementary source(s) (%d chars).",
        found_count, len(result),
    )
    return result

