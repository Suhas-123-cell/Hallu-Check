from __future__ import annotations

import difflib
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hallu-check.query_cache")

_DB_PATH = Path(__file__).resolve().parent.parent / "cache" / "query_cache.db"
_SIMILARITY_THRESHOLD = 0.82   # SequenceMatcher ratio to count as "same question"
_MAX_ENTRIES = 1000             # evict oldest entries when exceeded
_CACHE_TTL_SECONDS = 7 * 86400 # 7 days — keeps time-sensitive facts fresh

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "and", "or", "but", "if", "so", "yet", "nor",
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "this", "that", "these", "those",
    "tell", "please", "explain", "describe", "give",
}


def _normalize(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    words = [w for w in q.split() if w not in _STOP_WORDS]
    return " ".join(words)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class QueryCache:
    def __init__(self, db_path: str | Path = _DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    final_answer TEXT NOT NULL,
                    hallucination_score REAL NOT NULL,
                    verification_method TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_hit_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_normalized ON cache(normalized_query)"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[dict]:
        """Return a cached entry if a sufficiently similar query exists, else None."""
        norm = _normalize(query)
        if not norm:
            return None

        cutoff = time.time() - _CACHE_TTL_SECONDS

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cache WHERE created_at > ? ORDER BY last_hit_at DESC",
                (cutoff,),
            ).fetchall()

        best_score = 0.0
        best_row = None
        for row in rows:
            score = _similarity(norm, row["normalized_query"])
            if score > best_score:
                best_score = score
                best_row = row

        if best_row and best_score >= _SIMILARITY_THRESHOLD:
            logger.info(
                "Cache HIT (similarity=%.3f): %r → %d-char cached answer",
                best_score, query[:60], len(best_row["final_answer"]),
            )
            self._record_hit(best_row["id"])
            return dict(best_row)

        logger.debug("Cache MISS (best_similarity=%.3f): %r", best_score, query[:60])
        return None

    def store(self, query: str, final_answer: str, hallucination_score: float,
              verification_method: str = "NLI") -> None:
        """Store a verified answer. Only store when hallucination_score is low."""
        norm = _normalize(query)
        if not norm or not final_answer:
            return

        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO cache
                    (original_query, normalized_query, final_answer,
                     hallucination_score, verification_method, created_at, last_hit_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (query, norm, final_answer, hallucination_score,
                  verification_method, now, now))

        logger.info(
            "Cache STORE: %r (hallu_score=%.3f, method=%s)",
            query[:60], hallucination_score, verification_method,
        )
        self._evict_if_needed()

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            hits  = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
        return {"total_entries": total, "total_hits": hits, "db_path": str(self._db_path)}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _record_hit(self, row_id: int) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
                (now, row_id),
            )

    def _evict_if_needed(self) -> None:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count > _MAX_ENTRIES:
                excess = count - _MAX_ENTRIES
                conn.execute("""
                    DELETE FROM cache WHERE id IN (
                        SELECT id FROM cache ORDER BY last_hit_at ASC LIMIT ?
                    )
                """, (excess,))
                logger.info("Cache evicted %d oldest entries.", excess)


# Module-level singleton
_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache
