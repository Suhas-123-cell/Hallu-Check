"""
Shared evaluation infrastructure.

Everything an experiment needs to be leakage-free, controlled and reproducible:

* Dataset loading with a deterministic seeded sample (TruthfulQA is ordered by
  category, so "the first N" is a biased sample).
* ``patched_globals`` — the ONLY reliable way to toggle pipeline features.
  Modules do ``from config import X`` (copy by value) and ``config.py`` calls
  ``load_dotenv(override=True)``, so setting ``os.environ`` and reloading
  ``config`` never reaches the pipeline.
* ``WebGuard`` — blocks benchmark-hosting domains and any page that mentions
  the benchmark, so the web-retrieval step cannot read the answer key.
* ``RetrievalSnapshot`` — freezes the scraped web corpus per question, so every
  system / ablation sees identical evidence and the corpus can be released.
* ``Tracer`` — per-sample counters (Gemini calls, blocked URLs, consistency
  score, gap-recovery text).
* Statistics: bootstrap CIs, paired bootstrap, exact McNemar, AUROC,
  dev-tuned thresholds, Cohen's kappa.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"


def _rows_via_server(repo: str, config: str, split: str, cache_dir: Path) -> List[dict]:
    """Fallback loader that needs neither `datasets` nor `pyarrow`."""
    import httpx

    cache = cache_dir / f"{repo.replace('/', '__')}__{config}__{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    rows: List[dict] = []
    offset, page = 0, 100
    with httpx.Client(timeout=60) as client:
        while True:
            r = client.get(_DATASETS_SERVER, params={
                "dataset": repo, "config": config, "split": split,
                "offset": offset, "length": page,
            })
            r.raise_for_status()
            data = r.json()
            batch = [x["row"] for x in data.get("rows", [])]
            rows.extend(batch)
            total = data.get("num_rows_total", len(rows))
            offset += len(batch)
            if not batch or offset >= total:
                break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def load_rows(repo: str, config: str, split: str, cache_dir: Path) -> List[dict]:
    try:
        from datasets import load_dataset  # type: ignore
        return [dict(r) for r in load_dataset(repo, config, split=split)]
    except ImportError:
        return _rows_via_server(repo, config, split, cache_dir)


def seeded_indices(n_total: int, n: int, seed: int) -> List[int]:
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return idx[: min(n, n_total)]


# ─────────────────────────────────────────────────────────────────────────────
# JSONL I/O (resume-safe: every row is flushed as soon as it exists)
# ─────────────────────────────────────────────────────────────────────────────

_io_lock = threading.Lock()


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # truncated last line from a killed run
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock, path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_by_key(rows: Iterable[dict], key: Callable[[dict], Any]) -> Dict[Any, dict]:
    """Last write wins (a re-run of a failed row supersedes the failure)."""
    out: Dict[Any, dict] = {}
    for r in rows:
        out[key(r)] = r
    return out


def run_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return "unknown"

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
    }
    meta.update(extra)
    return meta


def pipeline_config_snapshot() -> Dict[str, Any]:
    import config  # type: ignore

    keys = [
        "LLM_BACKEND", "OLLAMA_MODEL", "OLLAMA_NUM_CTX",
        "LOCAL_MODEL_ID", "GEMINI_MODEL", "NLI_MODEL_PATH", "HALLUCINATION_THRESHOLD",
        "ENABLE_ICR", "ICR_MAX_ROUNDS", "ENABLE_SELF_CONSISTENCY", "N_CONSISTENCY_SAMPLES",
        "ENABLE_EGV", "ENABLE_SURGICAL_CORRECTION", "ENABLE_RLM_REASONING", "SEARCH_MAX_RESULTS",
    ]
    snap = {k: getattr(config, k, None) for k in keys}
    try:
        from nodes import local_llm  # type: ignore
        from nodes.llm_client import backend_label  # type: ignore
        local_llm.is_available()  # raises if LLM_BACKEND=ollama and the model is missing
        snap["generator_backend"] = backend_label()
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001
        snap["generator_backend"] = f"unknown ({e})"
    calib = Path(str(snap["NLI_MODEL_PATH"])) / "calibration_params.json"
    snap["nli_calibration_active"] = calib.exists()
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# Feature-flag patching
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_modules() -> List[Any]:
    return [
        m for n, m in list(sys.modules.items())
        if m is not None and (n in ("config", "main") or n.startswith("nodes."))
    ]


@contextlib.contextmanager
def patched_globals(overrides: Dict[str, Any]):
    """Temporarily set a module-level name in config, main and every nodes.* module
    that defines it. Raises if no loaded module defines a requested name, so a
    typo in an ablation config fails loudly instead of silently doing nothing."""
    saved: List[Tuple[Any, str, Any]] = []
    for mod in _pipeline_modules():
        for k, v in overrides.items():
            if k in vars(mod):
                saved.append((mod, k, vars(mod)[k]))
                setattr(mod, k, v)
    missing = sorted(set(overrides) - {k for _, k, _ in saved})
    if missing:
        for mod, k, v in reversed(saved):
            setattr(mod, k, v)
        raise KeyError(f"No loaded pipeline module defines: {missing}")
    try:
        yield
    finally:
        for mod, k, v in reversed(saved):
            setattr(mod, k, v)


class _NoCache:
    def lookup(self, *_a, **_k):
        return None

    def store(self, *_a, **_k):
        return None

    def stats(self):
        return {}


def disable_query_cache(main_mod: Any) -> None:
    """The pipeline's semantic query cache returns stored final answers for
    similar questions — across systems and runs. It must be off during eval."""
    main_mod.get_cache = lambda: _NoCache()


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample tracing
# ─────────────────────────────────────────────────────────────────────────────

class Tracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.gemini_calls = 0
            self.gemini_failures = 0
            self.fetched_urls: List[str] = []
            self.blocked_urls: List[str] = []
            self.blocked_pages: List[str] = []
            self.consistency_score: Optional[float] = None
            self.gap_text: str = ""
            self.snapshot_hit = False

    def add(self, field: str, value: Any) -> None:
        with self._lock:
            getattr(self, field).append(value)

    def incr_gemini(self) -> None:
        with self._lock:
            self.gemini_calls += 1

    def incr_gemini_failure(self) -> None:
        with self._lock:
            self.gemini_failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "gemini_calls": self.gemini_calls,
                "gemini_failures": self.gemini_failures,
                "n_fetched_urls": len(self.fetched_urls),
                "blocked_urls": list(self.blocked_urls),
                "blocked_pages": list(self.blocked_pages),
                "consistency_score": self.consistency_score,
                "gap_recovery_chars": len(self.gap_text),
                "retrieval_snapshot_hit": self.snapshot_hit,
            }


TRACE = Tracer()


def install_gemini_counter() -> None:
    from nodes import refiner, claim_classifier  # type: ignore

    for mod, name in ((refiner, "_gemini_generate"), (claim_classifier, "_gemini_classify")):
        orig = getattr(mod, name)
        if getattr(orig, "_counted", False):
            continue

        def wrapped(*a, _orig=orig, _name=name, **k):
            TRACE.incr_gemini()
            try:
                out = _orig(*a, **k)
            except Exception:
                TRACE.incr_gemini_failure()
                raise
            # refiner._gemini_generate swallows quota/API errors and returns ""
            if _name == "_gemini_generate" and not out:
                TRACE.incr_gemini_failure()
            return out

        wrapped._counted = True  # type: ignore[attr-defined]
        setattr(mod, name, wrapped)


def install_consistency_probe() -> None:
    """main.py imports check_self_consistency lazily, so patching the module
    attribute is enough. We record the score to evaluate it as a detector
    (the pipeline itself computes it but never uses it)."""
    from nodes import self_consistency  # type: ignore

    orig = self_consistency.check_self_consistency
    if getattr(orig, "_probed", False):
        return

    def probed(*a, **k):
        res = orig(*a, **k)
        try:
            TRACE.consistency_score = float(res["consistency_score"])
        except Exception:
            pass
        return res

    probed._probed = True  # type: ignore[attr-defined]
    self_consistency.check_self_consistency = probed


# ─────────────────────────────────────────────────────────────────────────────
# Leakage control for web retrieval
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_DOMAINS: Tuple[str, ...] = (
    "huggingface.co", "hf.co", "github.com", "githubusercontent.com", "gitlab.com",
    "arxiv.org", "openreview.net", "aclanthology.org", "paperswithcode.com",
    "semanticscholar.org", "researchgate.net", "kaggle.com", "papers.nips.cc",
    "proceedings.mlr.press", "proceedings.neurips.cc", "ar5iv.labs.arxiv.org",
    "alphaxiv.org", "scholar.archive.org",
)

BENCHMARK_MARKERS = re.compile(
    r"truthful[\s_\-]?qa|halu[\s_\-]?eval|imitative falsehoods|"
    r"lin,? hilton,? (and|&) evans",
    re.IGNORECASE,
)


def _domain_blocked(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in BENCHMARK_DOMAINS)


class WebGuard:
    """Installed once per process. Blocks at the search-result level and at
    the HTML-fetch level (which every scrape path — primary, depth-2 secondary,
    C-RAG retry and gap recovery — goes through)."""

    installed = False

    @classmethod
    def install(cls) -> None:
        if cls.installed:
            return
        from nodes import web_search as ws  # type: ignore

        orig_unscrapable = ws._is_unscrapable_domain

        def is_unscrapable(url: str) -> bool:
            if _domain_blocked(url):
                TRACE.add("blocked_urls", url)
                return True
            return orig_unscrapable(url)

        orig_fetch = ws._fetch_html

        def fetch_html(url: str, *a, **k) -> str:
            if _domain_blocked(url):
                TRACE.add("blocked_urls", url)
                return ""
            html = orig_fetch(url, *a, **k)
            if html and BENCHMARK_MARKERS.search(html):
                TRACE.add("blocked_pages", url)
                return ""
            if html:
                TRACE.add("fetched_urls", url)
            return html

        ws._is_unscrapable_domain = is_unscrapable
        ws._fetch_html = fetch_html
        cls.installed = True


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:20]


class RetrievalSnapshot:
    """Freeze the scraped corpus per question (and gap-recovery text per
    question+claims). First call scrapes and stores; later calls replay."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def install(self, main_mod: Any) -> None:
        orig_ss = main_mod.search_and_scrape
        orig_gap = main_mod.targeted_gap_search
        snap_dir = self.dir

        def search_and_scrape(keywords, enable_depth2=False, query=None, md_path=None):
            f = snap_dir / f"web_{_key(query or ' '.join(keywords))}.md"
            if f.exists():
                text = f.read_text(encoding="utf-8")
                if md_path:
                    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(md_path).write_text(text, encoding="utf-8")
                TRACE.snapshot_hit = True
                return md_path, text
            path, text = orig_ss(keywords, enable_depth2, query, md_path)
            try:
                text = Path(path).read_text(encoding="utf-8")
            except Exception:
                pass
            f.write_text(text or "", encoding="utf-8")
            return path, text

        def targeted_gap_search(claims, query, *a, **k):
            f = snap_dir / f"gap_{_key(query, *sorted(claims))}.md"
            if f.exists():
                text = f.read_text(encoding="utf-8")
            else:
                text = orig_gap(claims, query, *a, **k) or ""
                f.write_text(text, encoding="utf-8")
            TRACE.gap_text = text
            return text

        main_mod.search_and_scrape = search_and_scrape
        main_mod.targeted_gap_search = targeted_gap_search


# ─────────────────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────────────────

_ABSTAIN_RE = re.compile(
    r"\b(i (do not|don't) know|i have no comment|no comment|not found in (the )?context|"
    r"i(?:'m| am) not (sure|certain|able to)|cannot (be )?(determined|answered|answer)|"
    r"can't (determine|answer|say)|unable to (determine|find|answer|verify)|"
    r"no (reliable |verifiable )?information|insufficient (information|evidence)|"
    r"there is no (scientific )?(consensus|evidence|definitive answer))\b",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    """Heuristic: short answer that is mostly a refusal / uncertainty.
    Reported separately so a system cannot win TruthfulQA by refusing."""
    t = (text or "").strip()
    return bool(t) and len(t) < 400 and bool(_ABSTAIN_RE.search(t))


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    vals = list(values)
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(_mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return _mean(vals), lo, hi


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], **kw) -> Tuple[float, float, float]:
    """Mean of (a - b) with a bootstrap CI over paired items."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], **kw)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary outcomes."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mann–Whitney AUROC with tie handling; labels: 1 = positive."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def auroc_ci(labels: Sequence[int], scores: Sequence[float], n_boot: int = 1000,
             seed: int = 0) -> Tuple[float, float, float]:
    labels, scores = list(labels), list(scores)
    point = auroc(labels, scores)
    if math.isnan(point):
        return (point,) * 3
    rng = random.Random(seed)
    n = len(labels)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = auroc([labels[i] for i in idx], [scores[i] for i in idx])
        if not math.isnan(v):
            boots.append(v)
    boots.sort()
    return point, boots[int(0.025 * len(boots))], boots[min(len(boots) - 1, int(0.975 * len(boots)))]


def binary_metrics(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def tune_threshold(labels: Sequence[int], scores: Sequence[float], metric: str = "accuracy") -> float:
    """Pick the threshold (pred = score >= t) maximising `metric` on a dev split."""
    cands = sorted(set(scores)) or [0.5]
    best_t, best_v = cands[0], -1.0
    for t in cands:
        v = binary_metrics(labels, [int(s >= t) for s in scores])[metric]
        if v > best_v:
            best_t, best_v = t, v
    return best_t


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fmt_ci(t: Tuple[float, float, float], pct: bool = True) -> str:
    m, lo, hi = t
    if any(math.isnan(x) for x in t):
        return "n/a"
    if pct:
        return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
