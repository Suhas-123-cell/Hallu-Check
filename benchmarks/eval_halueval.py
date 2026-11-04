"""
HaluEval-QA hallucination *detection* benchmark (verifier only, standard protocol:
the knowledge passage is given; no web retrieval).

Usage:
    python -m benchmarks.eval_halueval run    --run runs/halueval --questions 500
    python -m benchmarks.eval_halueval run    --run runs/halueval_lm --questions 500 --length-matched
    python -m benchmarks.eval_halueval report --run runs/halueval

Data: the `qa` config, which has BOTH a right and a hallucinated answer per
question → perfectly balanced, paired items. (The previous e2e harness read the
`qa_samples` config and passed its `hallucination` field — the literal label
"yes"/"no" — to the verifier as if it were an answer, and treated `answer` as
correct even when it was the hallucinated one.)

Detectors (score: higher = more likely hallucinated):
    pipeline  Hallu-Check Node 5 (claim extraction + NLI + composite score)
    nli       the same NLI checkpoint on (knowledge, answer) — no claim logic
    gemini    Gemini as a judge (same model the pipeline uses), 0–100 score
    length    answer length in characters — a shortcut baseline.

WARNING — length shortcut: on HaluEval-QA, answer length ALONE reaches
AUROC ≈ 0.976 (measured on 2,000 questions: median 12 chars for right
answers vs 56 for hallucinated ones). A detector can look strong here simply
by reacting to longer answers. Always report the `length` row, and use
`--length-matched` (pairs whose lengths differ by < 1.5×, ~6.5% of the
dataset) as the controlled evaluation.

Protocol: questions are split 30/70 into dev/test (seeded, pairs kept
together). AUROC is threshold-free; P/R/F1 use a threshold tuned on dev only.
Errors are counted and reported, never scored as correct or incorrect; metrics
use the item set on which every detector produced a score.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from benchmarks import harness as H

logger = logging.getLogger("benchmark.halueval")

DATASET = ("pminervini/HaluEval", "qa", "data")
DETECTORS = ("pipeline", "nli", "gemini", "length")

_GEMINI_PROMPT = """You are checking a question-answering system for hallucinations.

Knowledge:
{knowledge}

Question: {question}
Answer: {answer}

Does the answer contain information that is unsupported by or contradicts the knowledge, or fail to answer the question correctly? Think briefly, then on the final line output exactly:
SCORE: <integer 0-100>
where 100 means certainly hallucinated and 0 means certainly correct."""


def _det_pipeline(item: dict) -> Dict[str, Any]:
    from nodes.claim_verifier import verify_claims  # type: ignore
    from nodes.pageindex_rag import evaluate_alignment  # type: ignore

    align = evaluate_alignment(item["answer"], item["knowledge"])
    rep = verify_claims(item["answer"], item["knowledge"], item["question"],
                        align["f1"], align["alignment_score"])
    return {"score": float(rep.hallucination_score), "detected": bool(rep.hallucination_detected),
            "n_claims": len(rep.claim_verdicts), "method": rep.verification_method}


def _det_nli(item: dict) -> Dict[str, Any]:
    from nodes.nli_model import classify_nli, is_loaded, load_model  # type: ignore

    if not is_loaded() and not load_model():
        raise RuntimeError("pipeline NLI model not available")
    r = classify_nli(item["knowledge"], item["answer"])
    return {"score": 1.0 - float(r["probabilities"]["entailment"])}


def _det_gemini(item: dict) -> Dict[str, Any]:
    from nodes import refiner  # type: ignore

    out = refiner._gemini_generate(_GEMINI_PROMPT.format(**item))
    m = re.findall(r"SCORE:\s*(\d{1,3})", out or "")
    if not m:
        raise ValueError(f"unparseable Gemini judge output: {(out or '')[-120:]!r}")
    return {"score": min(100, int(m[-1])) / 100.0}


def _det_length(item: dict) -> Dict[str, Any]:
    return {"score": float(len(item["answer"]))}


DETECTOR_FNS: Dict[str, Callable[[dict], Dict[str, Any]]] = {
    "pipeline": _det_pipeline, "nli": _det_nli, "gemini": _det_gemini, "length": _det_length,
}


def _length_matched(r: dict, lo: float = 0.67, hi: float = 1.5) -> bool:
    ratio = len(r["hallucinated_answer"]) / max(1, len(r["right_answer"]))
    return lo <= ratio <= hi


def _items(run_root: Path, n_questions: int, seed: int, length_matched: bool = False) -> List[dict]:
    path = run_root / "items.jsonl"
    existing = H.read_jsonl(path)
    if existing:
        return existing
    rows = H.load_rows(*DATASET, cache_dir=run_root / "data_cache")
    pool = [i for i, r in enumerate(rows) if not length_matched or _length_matched(r)]
    qids = [pool[j] for j in H.seeded_indices(len(pool), n_questions, seed)]
    split_rng = random.Random(seed + 1)
    shuffled = qids[:]
    split_rng.shuffle(shuffled)
    dev = set(shuffled[: int(0.3 * len(shuffled))])
    items = []
    for q in qids:
        r = rows[q]
        for label, ans in ((0, r["right_answer"]), (1, r["hallucinated_answer"])):
            items.append({"uid": f"{q}-{label}", "qid": q, "label": label,
                          "split": "dev" if q in dev else "test",
                          "knowledge": r["knowledge"], "question": r["question"], "answer": ans})
    for it in items:
        H.append_jsonl(path, it)
    return items


def cmd_run(args: argparse.Namespace) -> None:
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=True)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    bad = [d for d in detectors if d not in DETECTOR_FNS]
    if bad:
        raise SystemExit(f"unknown detectors {bad}")

    import nltk  # type: ignore
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    items = _items(root, args.questions, args.seed, args.length_matched)
    meta = H.run_metadata({"command": "halueval-run", "dataset": DATASET, "n_items": len(items),
                           "seed": args.seed, "detectors": detectors,
                           "length_matched": args.length_matched,
                           "pipeline": H.pipeline_config_snapshot()})
    (root / f"meta_run_{int(time.time())}.json").write_text(json.dumps(meta, indent=2, default=str))

    out = root / "scores.jsonl"
    done = {(r["uid"], r["detector"]) for r in H.read_jsonl(out) if not r.get("error")}
    for det in detectors:
        todo = [it for it in items if (it["uid"], det) not in done]
        logger.info("Detector %s: %d items to score", det, len(todo))
        for i, it in enumerate(todo, 1):
            t0 = time.time()
            row: Dict[str, Any] = {"uid": it["uid"], "detector": det}
            try:
                row.update(DETECTOR_FNS[det](it))
                row["error"] = None
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            row["latency_s"] = round(time.time() - t0, 3)
            H.append_jsonl(out, row)
            if i % 50 == 0:
                logger.info("  %s %d/%d", det, i, len(todo))


def cmd_report(args: argparse.Namespace) -> None:
    root = Path(args.run)
    items = {it["uid"]: it for it in H.read_jsonl(root / "items.jsonl")}
    all_rows = H.read_jsonl(root / "scores.jsonl")
    scores = H.latest_by_key([r for r in all_rows if not r.get("error")],
                             lambda r: (r["uid"], r["detector"]))
    errors = H.latest_by_key(all_rows, lambda r: (r["uid"], r["detector"]))
    dets = sorted({d for _, d in scores}, key=lambda d: DETECTORS.index(d) if d in DETECTORS else 99)
    common = [u for u in items if all((u, d) in scores for d in dets)]

    L = ["# HaluEval-QA detection results", "",
         f"Items: {len(items)} ({len(items) // 2} questions × right/hallucinated). "
         f"Scored by every detector: **{len(common)}**. Threshold tuned on dev, metrics on test.", "",
         "| Detector | Test AUROC [95% CI] | Acc | Prec | Rec | F1 | FPR | Errors |",
         "|---|---|---|---|---|---|---|---|"]
    result: Dict[str, Any] = {"n_items": len(items), "n_common": len(common), "detectors": {}}
    dev = [u for u in common if items[u]["split"] == "dev"]
    test = [u for u in common if items[u]["split"] == "test"]
    for d in dets:
        n_err = sum(1 for (u, dd), r in errors.items() if dd == d and r.get("error"))
        yd = [items[u]["label"] for u in dev]
        sd = [scores[(u, d)]["score"] for u in dev]
        yt = [items[u]["label"] for u in test]
        st = [scores[(u, d)]["score"] for u in test]
        thr = H.tune_threshold(yd, sd, "accuracy") if dev else 0.5
        m = H.binary_metrics(yt, [int(s >= thr) for s in st])
        auc = H.auroc_ci(yt, st)
        result["detectors"][d] = {"auroc": auc, "threshold": thr, "test": m, "errors": n_err}
        L.append(f"| {d} | {H.fmt_ci(auc, pct=False)} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                 f"{m['recall']:.3f} | {m['f1']:.3f} | {m['fpr']:.3f} | {n_err} |")

    if "pipeline" in dets:
        yt = [items[u]["label"] for u in test]
        m = H.binary_metrics(yt, [int(scores[(u, "pipeline")]["detected"]) for u in test])
        result["pipeline_deployed_threshold"] = m
        L += ["", f"Pipeline at its deployed threshold (no tuning): acc {m['accuracy']:.3f}, "
                  f"precision {m['precision']:.3f}, recall {m['recall']:.3f}, FPR {m['fpr']:.3f}"]
    if "length" in result["detectors"]:
        L += ["", "_`length` is a shortcut baseline (answer length only). On the unmatched "
                  "HaluEval-QA pool it reaches AUROC ≈ 0.98; a detector is only credible where it "
                  "clearly beats `length`, i.e. on a `--length-matched` run._"]

    md = "\n".join(L)
    (root / "report.md").write_text(md)
    (root / "report.json").write_text(json.dumps(result, indent=2, default=float))
    print(md)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for noisy in ("httpx", "hallu-check", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--questions", type=int, default=500)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--detectors", default=",".join(DETECTORS))
    r.add_argument("--length-matched", action="store_true",
                   help="only questions whose two answers differ in length by < 1.5x")
    r.set_defaults(fn=cmd_run)
    rp = sub.add_parser("report")
    rp.add_argument("--run", required=True)
    rp.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
