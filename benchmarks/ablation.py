"""
Ablation study on TruthfulQA (FACTUAL route).

    python -m benchmarks.ablation --run runs/tqa --samples 200
    python -m benchmarks.eval_e2e grade  --run runs/tqa --judge nli
    python -m benchmarks.eval_e2e report --run runs/tqa

Runs every pipeline variant in eval_e2e.VARIANTS inside the same run directory
as the main experiment, so all variants share:
  * the same seeded questions,
  * the same frozen web corpus (harness.RetrievalSnapshot) — retrieval noise
    does not masquerade as an ablation effect,
  * the same grader and report (paired tests against `pipeline`).

Why the old ablation was invalid: it set os.environ and reloaded `config`, but
config.py calls load_dotenv(override=True) (so .env wins) and every module
copies flags with `from config import X`. The toggles never reached the
pipeline; every "ablation" ran the same configuration. Flags are now patched
in-process with harness.patched_globals, which fails loudly on unknown names.

Flags that do NOT matter on the FACTUAL route and are therefore not ablated
here: ENABLE_RLM_REASONING (disabled on FACTUAL in main.py), ENABLE_EGV and
ENABLE_SURGICAL_CORRECTION (REASONING route). ENABLE_SELF_CONSISTENCY only
affects latency: main.py computes the score but never uses it — the report
evaluates it as a stand-alone detector instead.
"""
from __future__ import annotations

import argparse
import logging

from benchmarks import eval_e2e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", default=",".join(eval_e2e.VARIANTS))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for noisy in ("httpx", "openai", "hallu-check", "rag_pipeline", "huggingface_hub", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if "pipeline" not in variants:
        variants.insert(0, "pipeline")  # reference for paired comparisons
    eval_e2e.cmd_generate(argparse.Namespace(
        run=args.run, samples=args.samples, seed=args.seed, systems=",".join(variants),
    ))


if __name__ == "__main__":
    main()
