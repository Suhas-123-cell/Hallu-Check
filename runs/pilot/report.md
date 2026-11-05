# TruthfulQA end-to-end results

## Judge: `grades__llm_ollama_qwen2_5_7b.jsonl`

Sampled questions: 5 · paired set (all systems succeeded): **4**

| System | Truthful % [95% CI] | Abstain % | Truthful∧informative % | Errors | Gemini calls/q | Latency s/q |
|---|---|---|---|---|---|---|
| gemini | 100.0 [100.0, 100.0] | 0.0 | 100.0 [100.0, 100.0] | 1 | 1.00 | 4.2 |
| gemini_rag | 100.0 [100.0, 100.0] | 0.0 | 100.0 [100.0, 100.0] | 1 | 1.00 | 5.2 |
| llama | 50.0 [0.0, 100.0] | 0.0 | 50.0 [0.0, 100.0] | 0 | 0.00 | — |
| pipeline | 50.0 [0.0, 100.0] | 0.0 | 50.0 [0.0, 100.0] | 1 | 3.25 | 58.6 |

**Paired vs `pipeline`** (positive Δ = pipeline better)

| Other system | Δ truthful (pp) [95% CI] | ref-only right | other-only right | McNemar p |
|---|---|---|---|---|
| gemini | -50.0 [-100.0, +0.0] | 0 | 2 | 0.5 |
| gemini_rag | -50.0 [-100.0, +0.0] | 0 | 2 | 0.5 |
| llama | +0.0 [-75.0, +75.0] | 1 | 1 | 1 |

**Detection** — does the pipeline's hallucination score identify untruthful raw generator answers?

n=4, untruthful base rate 50.0%

- Pipeline score AUROC: 0.000 [0.000, 0.000]
- Self-consistency (1−score) AUROC: 0.000 [0.000, 0.000] (n=4)
- At deployed threshold: precision 0.500, recall 1.000, F1 0.667, FPR 1.000

**Refinement transitions (llama → pipeline)**: refined:broken=1, refined:fixed=1, refined:kept_right=1, untouched:still_wrong=1

**Leakage audit**: 4 pipeline rows (ablation variants replay the frozen corpus, so only first-time retrievals fetch); 0 hit a blocked domain, 0 fetched a page mentioning the benchmark (0 pages dropped before reaching the pipeline).

## Judge: `grades__nli_MoritzLaurer_DeBERTa_v3_large_mnli_fever_anli_ling_wanli.jsonl`

Sampled questions: 5 · paired set (all systems succeeded): **4**

| System | Truthful % [95% CI] | Abstain % | Truthful∧informative % | Errors | Gemini calls/q | Latency s/q |
|---|---|---|---|---|---|---|
| gemini | 100.0 [100.0, 100.0] | 0.0 | 75.0 [25.0, 100.0] | 1 | 1.00 | 4.2 |
| gemini_rag | 100.0 [100.0, 100.0] | 0.0 | 75.0 [25.0, 100.0] | 1 | 1.00 | 5.2 |
| llama | 75.0 [25.0, 100.0] | 0.0 | 25.0 [0.0, 75.0] | 0 | 0.00 | — |
| pipeline | 75.0 [25.0, 100.0] | 0.0 | 25.0 [0.0, 75.0] | 1 | 3.25 | 58.6 |

**Paired vs `pipeline`** (positive Δ = pipeline better)

| Other system | Δ truthful (pp) [95% CI] | ref-only right | other-only right | McNemar p |
|---|---|---|---|---|
| gemini | -25.0 [-75.0, +0.0] | 0 | 1 | 1 |
| gemini_rag | -25.0 [-75.0, +0.0] | 0 | 1 | 1 |
| llama | +0.0 [+0.0, +0.0] | 0 | 0 | 1 |

**Detection** — does the pipeline's hallucination score identify untruthful raw generator answers?

n=4, untruthful base rate 25.0%

- Pipeline score AUROC: 0.333 [0.000, 1.000]
- Self-consistency (1−score) AUROC: 0.333 [0.000, 1.000] (n=4)
- At deployed threshold: precision 0.250, recall 1.000, F1 0.400, FPR 1.000

**Refinement transitions (llama → pipeline)**: refined:kept_right=3, untouched:still_wrong=1

**Leakage audit**: 4 pipeline rows (ablation variants replay the frozen corpus, so only first-time retrievals fetch); 0 hit a blocked domain, 0 fetched a page mentioning the benchmark (0 pages dropped before reaching the pipeline).