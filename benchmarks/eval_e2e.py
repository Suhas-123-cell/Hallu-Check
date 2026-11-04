"""
End-to-end TruthfulQA evaluation — leakage-free, with strong baselines.

Usage (run from the repo root):
    python -m benchmarks.eval_e2e generate --run runs/tqa --samples 200
    python -m benchmarks.eval_e2e grade    --run runs/tqa --judge nli
    python -m benchmarks.eval_e2e grade    --run runs/tqa --judge llm
    python -m benchmarks.eval_e2e export-human --run runs/tqa --n 100
    python -m benchmarks.eval_e2e report   --run runs/tqa [--human runs/tqa/human_sheet.csv]

Systems (all receive ONLY the question):
    llama        raw generator output — the exact text the pipeline then verifies
    gemini       Gemini closed-book (same model the pipeline uses as refiner)
    gemini_rag   Gemini + the exact evidence the pipeline retrieved for that question
    pipeline     full FACTUAL route: web retrieval → claim verification → ICR
    pipeline-*   ablations, see VARIANTS (run via benchmarks.ablation)

What changed vs. the previous version, and why:
    * The previous harness passed TruthfulQA's `best_answer` to the verifier and
      to Gemini as "RAG context", then measured whether the output entailed
      `best_answer`. That is answer-key leakage. Here, evidence comes only from
      live web retrieval, with benchmark-hosting domains and any page mentioning
      the benchmark blocked (harness.WebGuard).
    * The grader is no longer the pipeline's own NLI verifier (circular); see
      benchmarks/graders.py.
    * The semantic query cache is disabled (it returns stored final answers).
    * Samples are a seeded random draw, not the first N (TruthfulQA is ordered).
    * Failures are recorded as failures, never silently scored.
    * The scraped corpus is frozen per question, so every system and ablation
      sees the same evidence and the corpus can be released with the paper.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from benchmarks import harness as H

logger = logging.getLogger("benchmark.e2e")

DATASET = ("truthfulqa/truthful_qa", "generation", "validation")
PIPELINE_TIMEOUT_S = 600
EVIDENCE_CAP = 15000  # same cap the pipeline applies before refinement


def _no_gap_search(*_a, **_k) -> str:
    return ""


# name -> overrides applied with harness.patched_globals for that run only
VARIANTS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "pipeline": lambda: {},
    # one-shot refinement instead of the iterative verify→refine loop
    "pipeline-no_icr": lambda: {"ENABLE_ICR": False},
    # refine every answer regardless of the detector: tests whether detection
    # (gating) adds anything over "always let Gemini rewrite with evidence"
    "pipeline-always_refine": lambda: {"HALLUCINATION_THRESHOLD": -1.0},
    # disable Node 5.5 targeted search for UNVERIFIABLE claims
    "pipeline-no_gap_recovery": lambda: {"targeted_gap_search": _no_gap_search},
}

BASELINES = ("llama", "gemini", "gemini_rag")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

class Run:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.samples = self.root / "samples.jsonl"
        self.generations = self.root / "generations.jsonl"
        self.evidence = self.root / "evidence"
        self.web = self.evidence / "web"

    def grades(self, judge_name: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", judge_name).strip("_")
        return self.root / f"grades__{slug}.jsonl"

    def evidence_file(self, sid: int) -> Path:
        return self.evidence / f"{sid}.json"


# ─────────────────────────────────────────────────────────────────────────────
# generate
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_samples(run: Run, n: int, seed: int) -> List[dict]:
    existing = H.read_jsonl(run.samples)
    if existing:
        if len(existing) < n:
            logger.warning("Run already fixed at %d samples (asked for %d); keeping the "
                           "existing sample. Start a new --run to change it.", len(existing), n)
        return existing
    rows = H.load_rows(*DATASET, cache_dir=run.root / "data_cache")
    out = []
    for i in H.seeded_indices(len(rows), n, seed):
        r = rows[i]
        out.append({
            "id": i, "question": r["question"], "category": r.get("category"),
            "type": r.get("type"), "best_answer": r["best_answer"],
            "correct_answers": list(r["correct_answers"]),
            "incorrect_answers": list(r["incorrect_answers"]),
        })
    for r in out:
        H.append_jsonl(run.samples, r)
    return out


def _gemini_closed_book_prompt(q: str) -> str:
    return f"Answer the following question concisely and factually.\n\nQuestion: {q}\nAnswer:"


def _gemini_rag_prompt(q: str, evidence: str) -> str:
    return (
        "Answer the question concisely and factually. Retrieved web evidence is "
        "provided below; it may be incomplete or contain irrelevant material.\n\n"
        f"Evidence:\n{evidence[:EVIDENCE_CAP]}\n\nQuestion: {q}\nAnswer:"
    )


def _pipeline_summary(resp: Any) -> Dict[str, Any]:
    verdicts = Counter(cv.verdict for cv in resp.claim_verdicts)
    return {
        "hallucination_score": resp.hallucination_score,
        "hallucination_detected": resp.hallucination_detected,
        "refinement_failed": getattr(resp, "refinement_failed", None),
        "refined": resp.final_answer.strip() != resp.llm_output.strip(),
        "verification_method": resp.verification_method,
        "n_claims": len(resp.claim_verdicts),
        "verdicts": dict(verdicts),
        "llm_output": resp.llm_output,
        "rag_chars": len(resp.rag_output or ""),
    }


async def _generate(run: Run, samples: List[dict], systems: List[str]) -> None:
    import main as main_mod  # type: ignore
    from nodes import refiner  # type: ignore
    from nodes.generator import generate_llm_output  # type: ignore

    done = {
        (r["id"], r["system"])
        for r in H.read_jsonl(run.generations) if not r.get("error")
    }

    def record(sample: dict, system: str, **fields: Any) -> None:
        row = {"id": sample["id"], "system": system, "question": sample["question"],
               "output": None, "error": None, **fields}
        H.append_jsonl(run.generations, row)
        status = "ERR " + str(row["error"])[:80] if row["error"] else "ok"
        logger.info("  [%s] id=%s %s", system, sample["id"], status)

    variants = [s for s in systems if s in VARIANTS]
    # full pipeline first: gemini_rag reuses its evidence, llama reuses its draft
    variants.sort(key=lambda s: s != "pipeline")

    for k, sample in enumerate(samples, 1):
        sid, q = sample["id"], sample["question"]
        logger.info("── sample %d/%d (id=%s)", k, len(samples), sid)

        for system in variants:
            if (sid, system) in done:
                continue
            H.TRACE.reset()
            t0 = time.time()
            try:
                with H.patched_globals(VARIANTS[system]()):
                    resp = await asyncio.wait_for(main_mod._handle_factual(q, None),
                                                  PIPELINE_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                record(sample, system, error=f"{type(e).__name__}: {e}",
                       latency_s=round(time.time() - t0, 1), trace=H.TRACE.snapshot())
                continue
            trace = H.TRACE.snapshot()
            if trace["gemini_failures"]:
                # The pipeline falls back to the unrefined draft when Gemini is
                # unavailable. That is an infrastructure failure (quota/API), not
                # the method's behaviour: record as an error so resume retries it.
                record(sample, system, error=f"Gemini unavailable during pipeline run "
                                             f"({trace['gemini_failures']} failed call(s)); retry",
                       latency_s=round(time.time() - t0, 1), trace=trace,
                       pipeline=_pipeline_summary(resp))
                continue
            record(sample, system, output=resp.final_answer, latency_s=round(time.time() - t0, 1),
                   trace=trace, pipeline=_pipeline_summary(resp))
            done.add((sid, system))

            if system == "pipeline":
                run.evidence.mkdir(parents=True, exist_ok=True)
                run.evidence_file(sid).write_text(json.dumps({
                    "rag_output": resp.rag_output,
                    "gap_recovery": H.TRACE.gap_text,
                    "blocked_urls": trace["blocked_urls"],
                    "blocked_pages": trace["blocked_pages"],
                }, ensure_ascii=False))
                if "llama" in systems and (sid, "llama") not in done:
                    record(sample, "llama", output=resp.llm_output, latency_s=None,
                           trace={"gemini_calls": 0}, source="pipeline.llm_output")
                    done.add((sid, "llama"))

        if "llama" in systems and (sid, "llama") not in done:
            t0 = time.time()
            try:
                out = await asyncio.to_thread(generate_llm_output, q)
                record(sample, "llama", output=out, latency_s=round(time.time() - t0, 1),
                       trace={"gemini_calls": 0}, source="generator")
                done.add((sid, "llama"))
            except Exception as e:  # noqa: BLE001
                record(sample, "llama", error=f"{type(e).__name__}: {e}")

        for system in ("gemini", "gemini_rag"):
            if system not in systems or (sid, system) in done:
                continue
            if system == "gemini":
                prompt = _gemini_closed_book_prompt(q)
            else:
                ev_file = run.evidence_file(sid)
                if not ev_file.exists():
                    record(sample, system, error="no pipeline evidence for this id "
                                                 "(run the 'pipeline' system first)")
                    continue
                ev = json.loads(ev_file.read_text())
                evidence = ev["rag_output"] + ("\n\n" + ev["gap_recovery"] if ev["gap_recovery"] else "")
                prompt = _gemini_rag_prompt(q, evidence)
            H.TRACE.reset()
            t0 = time.time()
            out = await asyncio.to_thread(refiner._gemini_generate, prompt)
            if out:
                record(sample, system, output=out, latency_s=round(time.time() - t0, 1),
                       trace={"gemini_calls": H.TRACE.gemini_calls})
                done.add((sid, system))
            else:
                record(sample, system, error="empty Gemini response (rate limit / API error)")


def cmd_generate(args: argparse.Namespace) -> None:
    run = Run(args.run)
    run.root.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in VARIANTS and s not in BASELINES]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; choose from {list(BASELINES) + list(VARIANTS)}")
    if "gemini_rag" in systems and "pipeline" not in systems and not run.evidence.exists():
        raise SystemExit("gemini_rag needs the pipeline's evidence: include 'pipeline' in --systems")

    import nltk  # type: ignore
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    import main as main_mod  # type: ignore  (imports every pipeline node)
    H.disable_query_cache(main_mod)
    H.WebGuard.install()
    H.install_gemini_counter()
    H.install_consistency_probe()
    H.RetrievalSnapshot(run.web).install(main_mod)

    samples = _prepare_samples(run, args.samples, args.seed)
    meta = H.run_metadata({
        "command": "generate", "dataset": DATASET, "n_samples": len(samples), "seed": args.seed,
        "systems": systems, "blocked_domains": H.BENCHMARK_DOMAINS,
        "blocked_markers": H.BENCHMARK_MARKERS.pattern, "pipeline": H.pipeline_config_snapshot(),
    })
    (run.root / f"meta_generate_{int(time.time())}.json").write_text(json.dumps(meta, indent=2, default=str))
    if not meta["pipeline"]["nli_calibration_active"]:
        logger.warning("NLI calibration file missing — Platt scaling is NOT active in this run.")
    logger.info("Generator backend: %s", meta["pipeline"]["generator_backend"])

    asyncio.run(_generate(run, samples, systems))


# ─────────────────────────────────────────────────────────────────────────────
# grade
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grade(args: argparse.Namespace) -> None:
    from benchmarks.graders import make_grader

    run = Run(args.run)
    samples = {s["id"]: s for s in H.read_jsonl(run.samples)}
    gens = H.latest_by_key(H.read_jsonl(run.generations), lambda r: (r["id"], r["system"]))
    grader = make_grader(args.judge, args.judge_model)
    out_path = run.grades(grader.name)
    done = {(g["id"], g["system"]) for g in H.read_jsonl(out_path) if not g.get("error")}

    todo = [g for k, g in gens.items() if g.get("output") and k not in done]
    logger.info("Grading %d generations with %s → %s", len(todo), grader.name, out_path)
    for i, g in enumerate(todo, 1):
        s = samples[g["id"]]
        row = {"id": g["id"], "system": g["system"], "judge": grader.name,
               "abstained": H.is_abstention(g["output"])}
        try:
            row.update(grader.grade(s["question"], g["output"], s["correct_answers"],
                                    s["incorrect_answers"]))
            row["error"] = None
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        H.append_jsonl(out_path, row)
        if i % 25 == 0:
            logger.info("  graded %d/%d", i, len(todo))


# ─────────────────────────────────────────────────────────────────────────────
# human annotation
# ─────────────────────────────────────────────────────────────────────────────

def cmd_export_human(args: argparse.Namespace) -> None:
    run = Run(args.run)
    samples = {s["id"]: s for s in H.read_jsonl(run.samples)}
    gens = H.latest_by_key(H.read_jsonl(run.generations), lambda r: (r["id"], r["system"]))
    systems = sorted({s for _, s in gens} if not args.systems else args.systems.split(","))
    ids = sorted(i for i in samples if all(gens.get((i, s), {}).get("output") for s in systems))
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]

    items, key = [], {}
    for sid in ids:
        for system in systems:
            uid = f"{sid}-{len(items):05d}"
            key[uid] = {"id": sid, "system": system}
            s = samples[sid]
            items.append({
                "item_uid": uid, "question": s["question"],
                "true_references": " | ".join(s["correct_answers"]),
                "false_references": " | ".join(s["incorrect_answers"]),
                "answer": gens[(sid, system)]["output"], "truthful (1/0)": "", "notes": "",
            })
    random.Random(args.seed + 1).shuffle(items)  # blind: no system order, no grouping

    sheet = run.root / "human_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)
    (run.root / "human_key.json").write_text(json.dumps(key, indent=1))
    print(f"Wrote {len(items)} blinded items ({len(ids)} questions × {len(systems)} systems) → {sheet}")
    print("Give annotators the CSV only; human_key.json maps items back to systems.")


def _load_human(run: Run, path: str) -> Dict[tuple, int]:
    key = json.loads((run.root / "human_key.json").read_text())
    out: Dict[tuple, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("truthful (1/0)") or "").strip()
            if v in ("0", "1") and row["item_uid"] in key:
                k = key[row["item_uid"]]
                out[(k["id"], k["system"])] = int(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────

def _report_for_judge(run: Run, grades_path: Path, ref_system: str,
                      human: Optional[Dict[tuple, int]]) -> Dict[str, Any]:
    samples = {s["id"]: s for s in H.read_jsonl(run.samples)}
    gens = H.latest_by_key(H.read_jsonl(run.generations), lambda r: (r["id"], r["system"]))
    grades = H.latest_by_key(
        [g for g in H.read_jsonl(grades_path) if not g.get("error")],
        lambda r: (r["id"], r["system"]),
    )
    systems = sorted({s for _, s in gens}, key=lambda s: (s not in BASELINES, s))
    n_total = len(samples)

    # paired set: every system produced and was graded on these ids
    common = sorted(i for i in samples if all((i, s) in grades for s in systems))
    rep: Dict[str, Any] = {"judge": grades_path.name, "n_sampled": n_total,
                           "n_common": len(common), "systems": {}}

    for s in systems:
        g_rows = [gens[(i, s)] for i in samples if (i, s) in gens]
        n_err = sum(1 for g in g_rows if g.get("error"))
        tr = [float(grades[(i, s)]["truthful"]) for i in common]
        ab = [float(grades[(i, s)]["abstained"]) for i in common]
        ti = [float(grades[(i, s)]["truthful"] and not grades[(i, s)]["abstained"]
                    and grades[(i, s)].get("informative", True)) for i in common]
        calls = [g.get("trace", {}).get("gemini_calls") for g in g_rows if not g.get("error")]
        calls = [c for c in calls if c is not None]
        lat = [g["latency_s"] for g in g_rows if g.get("latency_s") is not None and not g.get("error")]
        entry = {
            "errors": n_err, "missing": n_total - len(g_rows),
            "truthful": H.bootstrap_ci(tr), "abstain": H.bootstrap_ci(ab),
            "truthful_informative": H.bootstrap_ci(ti),
            "mean_gemini_calls": sum(calls) / len(calls) if calls else None,
            "mean_latency_s": sum(lat) / len(lat) if lat else None,
        }
        if s in VARIANTS:
            refail = [g for g in g_rows if g.get("pipeline", {}).get("refinement_failed")]
            entry["refinement_failed"] = len(refail)
            entry["refine_rate"] = (sum(1 for g in g_rows if g.get("pipeline", {}).get("refined"))
                                    / max(1, len(g_rows) - n_err))
        rep["systems"][s] = entry

    # paired comparisons against the reference system
    rep["paired"] = {}
    if ref_system in systems:
        a = [bool(grades[(i, ref_system)]["truthful"]) for i in common]
        for s in systems:
            if s == ref_system:
                continue
            b = [bool(grades[(i, s)]["truthful"]) for i in common]
            rep["paired"][s] = {
                "delta_truthful": H.paired_bootstrap_diff([float(x) for x in a], [float(y) for y in b]),
                "mcnemar": H.mcnemar_exact(a, b),
            }

    # detection quality on the raw generator's answers (the pipeline's actual job)
    det: Dict[str, Any] = {}
    ids = [i for i in samples if (i, "llama") in grades and (i, "pipeline") in gens
           and not gens[(i, "pipeline")].get("error")]
    if ids:
        labels = [int(not grades[(i, "llama")]["truthful"]) for i in ids]
        pl = [gens[(i, "pipeline")]["pipeline"] for i in ids]
        det["n"] = len(ids)
        det["base_rate_untruthful"] = sum(labels) / len(labels)
        det["pipeline_score_auroc"] = H.auroc_ci(labels, [p["hallucination_score"] for p in pl])
        det["pipeline_at_deployed_threshold"] = H.binary_metrics(
            labels, [int(p["hallucination_detected"]) for p in pl])
        cs = [(lab, gens[(i, "pipeline")]["trace"].get("consistency_score"))
              for lab, i in zip(labels, ids)]
        cs = [(lab, c) for lab, c in cs if c is not None]
        if cs:
            det["self_consistency_auroc"] = H.auroc_ci([l for l, _ in cs], [1 - c for _, c in cs])
            det["self_consistency_n"] = len(cs)
    rep["detection"] = det

    # did refinement fix or break answers?
    trans: Counter = Counter()
    for i in samples:
        if (i, "llama") in grades and (i, "pipeline") in grades:
            before = grades[(i, "llama")]["truthful"]
            after = grades[(i, "pipeline")]["truthful"]
            refined = gens[(i, "pipeline")].get("pipeline", {}).get("refined", False)
            tag = {(False, True): "fixed", (True, False): "broken",
                   (True, True): "kept_right", (False, False): "still_wrong"}[(before, after)]
            trans[f"{'refined' if refined else 'untouched'}:{tag}"] += 1
    rep["refinement_transitions"] = dict(trans)

    # leakage audit
    prow = [g for k, g in gens.items() if k[1] in VARIANTS and not g.get("error")]
    rep["leakage_audit"] = {
        "pipeline_rows": len(prow),
        "rows_with_blocked_domain": sum(1 for g in prow if g["trace"].get("blocked_urls")),
        "rows_with_blocked_page": sum(1 for g in prow if g["trace"].get("blocked_pages")),
        "total_blocked_pages": sum(len(g["trace"].get("blocked_pages", [])) for g in prow),
    }

    if human:
        hk = [k for k in human if k in grades]
        rep["human"] = {
            "n": len(hk),
            "judge_agreement": sum(int(grades[k]["truthful"]) == human[k] for k in hk) / max(1, len(hk)),
            "cohen_kappa": H.cohen_kappa([int(grades[k]["truthful"]) for k in hk], [human[k] for k in hk]),
            "human_truthful_by_system": {
                s: H.bootstrap_ci([float(human[k]) for k in hk if k[1] == s]) for s in systems
            },
        }
    return rep


def _render_md(rep: Dict[str, Any], ref_system: str) -> str:
    L = [f"## Judge: `{rep['judge']}`", "",
         f"Sampled questions: {rep['n_sampled']} · paired set (all systems succeeded): "
         f"**{rep['n_common']}**", "",
         "| System | Truthful % [95% CI] | Abstain % | Truthful∧informative % | Errors | Gemini calls/q | Latency s/q |",
         "|---|---|---|---|---|---|---|"]
    for s, e in rep["systems"].items():
        gc = f"{e['mean_gemini_calls']:.2f}" if e["mean_gemini_calls"] is not None else "—"
        lt = f"{e['mean_latency_s']:.1f}" if e["mean_latency_s"] is not None else "—"
        L.append(f"| {s} | {H.fmt_ci(e['truthful'])} | {100 * e['abstain'][0]:.1f} | "
                 f"{H.fmt_ci(e['truthful_informative'])} | {e['errors']} | {gc} | {lt} |")
    if rep["paired"]:
        L += ["", f"**Paired vs `{ref_system}`** (positive Δ = {ref_system} better)", "",
              "| Other system | Δ truthful (pp) [95% CI] | ref-only right | other-only right | McNemar p |",
              "|---|---|---|---|---|"]
        for s, p in rep["paired"].items():
            m, lo, hi = p["delta_truthful"]
            mc = p["mcnemar"]
            L.append(f"| {s} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {mc['a_only']} | "
                     f"{mc['b_only']} | {mc['p_value']:.3g} |")
    d = rep.get("detection") or {}
    if d:
        L += ["", "**Detection** — does the pipeline's hallucination score identify untruthful raw generator answers?", "",
              f"n={d['n']}, untruthful base rate {100 * d['base_rate_untruthful']:.1f}%", "",
              f"- Pipeline score AUROC: {H.fmt_ci(d['pipeline_score_auroc'], pct=False)}"]
        if "self_consistency_auroc" in d:
            L.append(f"- Self-consistency (1−score) AUROC: {H.fmt_ci(d['self_consistency_auroc'], pct=False)} "
                     f"(n={d['self_consistency_n']})")
        b = d["pipeline_at_deployed_threshold"]
        L.append(f"- At deployed threshold: precision {b['precision']:.3f}, recall {b['recall']:.3f}, "
                 f"F1 {b['f1']:.3f}, FPR {b['fpr']:.3f}")
    if rep["refinement_transitions"]:
        L += ["", "**Refinement transitions (llama → pipeline)**: " +
              ", ".join(f"{k}={v}" for k, v in sorted(rep["refinement_transitions"].items()))]
    la = rep["leakage_audit"]
    L += ["", f"**Leakage audit**: {la['pipeline_rows']} pipeline rows (ablation variants replay the "
              f"frozen corpus, so only first-time retrievals fetch); "
              f"{la['rows_with_blocked_domain']} hit a blocked domain, "
              f"{la['rows_with_blocked_page']} fetched a page mentioning the benchmark "
              f"({la['total_blocked_pages']} pages dropped before reaching the pipeline)."]
    if "human" in rep:
        h = rep["human"]
        L += ["", f"**Human validation** (n={h['n']}): judge agreement {100 * h['judge_agreement']:.1f}%, "
                  f"Cohen's κ = {h['cohen_kappa']:.3f}", ""]
        for s, ci in h["human_truthful_by_system"].items():
            L.append(f"- {s}: human-truthful {H.fmt_ci(ci)}")
    return "\n".join(L)


def cmd_report(args: argparse.Namespace) -> None:
    run = Run(args.run)
    paths = sorted(run.root.glob("grades__*.jsonl"))
    if args.judge_file:
        paths = [Path(args.judge_file)]
    if not paths:
        raise SystemExit("No grades found; run `grade` first.")
    human = _load_human(run, args.human) if args.human else None

    reports = [_report_for_judge(run, p, args.ref, human) for p in paths]
    md = "# TruthfulQA end-to-end results\n\n" + "\n\n".join(_render_md(r, args.ref) for r in reports)
    (run.root / "report.md").write_text(md)
    (run.root / "report.json").write_text(json.dumps(reports, indent=2, default=float))
    print(md)


# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--samples", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--systems", default="pipeline,llama,gemini,gemini_rag")
    g.set_defaults(fn=cmd_generate)

    gr = sub.add_parser("grade")
    gr.add_argument("--run", required=True)
    gr.add_argument("--judge", choices=["nli", "llm"], default="nli")
    gr.add_argument("--judge-model", default=None)
    gr.set_defaults(fn=cmd_grade)

    eh = sub.add_parser("export-human")
    eh.add_argument("--run", required=True)
    eh.add_argument("--n", type=int, default=100, help="number of questions")
    eh.add_argument("--systems", default=None)
    eh.add_argument("--seed", type=int, default=0)
    eh.set_defaults(fn=cmd_export_human)

    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--ref", default="pipeline")
    r.add_argument("--judge-file", default=None)
    r.add_argument("--human", default=None, help="annotated human_sheet.csv")
    r.set_defaults(fn=cmd_report)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for noisy in ("httpx", "openai", "hallu-check", "rag_pipeline", "huggingface_hub", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
