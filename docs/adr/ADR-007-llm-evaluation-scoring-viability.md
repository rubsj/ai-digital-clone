# ADR-007: LLM-Based Evaluation Scoring Viability

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation
**Status:** Accepted
**Date:** 2026-04-27

---

## Context

The production evaluator (`src/evaluation/evaluator.py`) does two things. It computes a groundedness score in [0,1] that needs to agree with the production embedding-cosine scorer (`score_groundedness()`, which is sentence-level OpenAI embedding cosine similarity, not keyword overlap), because that score feeds the `0.4×style + 0.4×groundedness + 0.2×confidence` formula and decides deliver vs fallback. It also generates a one-sentence explanation of the result, which gets surfaced to the caller but doesn't affect routing.

Both pieces currently call GPT-4o-mini through `instructor.from_litellm`. The question for Day 6 was whether a local model (Ollama qwen3:8b) could replace either of them and cut the API cost.

Phase 6 Run 2 (`scripts/experiment_6e_run2_groundedness_agreement.py`) had both models score groundedness independently from (query, top-5 chunk texts) on 10 queries from `data/eval/queries_v1.json`. The production embedding-cosine scorer was the calibration baseline. I computed three Pearson correlations: GPT vs baseline, Ollama vs baseline, and GPT vs Ollama.

Two things about the experiment matter for reading the numbers. First, Run 1 (`scripts/experiment_6e_local_vs_api.py`) was degenerate: both models received pre-computed component scores and applied the same deterministic formula, so they agreed at Pearson=1.0 by construction. Run 1's latency numbers (Ollama 2.1x faster) only measure the explanation-generation task, not scoring. Run 2 fixed the design. Both runs are kept in `docs/iteration-log.md` for audit.

Second, the Run 2 prompt described the score scale with explicit 0.0/0.5/1.0 anchors. Both models quantized to those anchors instead of producing continuous scores (GPT used {0.0, 0.5}, Ollama used {0.0, 0.5, 0.9}). Rank-order is preserved so the Pearson values are real signal, but the absolute correlations carry roughly ±0.05 quantization noise. The directional findings hold; precise threshold claims about the exact correlations don't.

---

## Decision

Keep GPT-4o-mini in production for both scoring and explanation generation. Pearson(GPT, baseline) = 0.8172 (p=0.0039) at latency parity with Ollama (1504ms vs 1465ms, 0.97x). That's a strong enough match to the production embedding scorer to treat the LLM scoring path as validated on this query set, not just defaulted into. No code change to `src/evaluation/evaluator.py`.

Ollama qwen3:8b is fine for explanation generation but not for production scoring. Pearson(Ollama, baseline) = 0.6796 (p=0.0306), which is 0.14 below GPT against the same baseline, and the gap shows up as directional disagreement on individual queries. Clearest example: q03 (binary search) where Ollama scored 0.9, GPT scored 0.5, and baseline was 0.60. The 0.9 reads like the model knows binary search well, not like a grounded assessment of whether the retrieved chunks answered the query. For explanation generation alone, where calibration to a numeric baseline doesn't matter, Ollama is a viable cheaper option (Run 1: 739ms mean, 59ms std, 100% structured-output success via `instructor.Mode.JSON`).

The production config doesn't change. The optional split (Ollama for explanations in dev, GPT for scoring in prod) is recorded as available, not mandated.

---

## Alternatives Considered

**Use Ollama qwen3:8b for both sub-tasks.** Drops API cost on evaluation to near zero. The 0.14 Pearson gap kills it for production scoring: at the 0.75 deliver/fallback threshold and 0.4 groundedness weight, the q03-style 0.4-point disagreement (Ollama=0.9 vs baseline=0.60) shifts the final score by 0.16, which is enough to flip a fallback to a deliver. Fine for dev iteration; not fine where the routing decision actually matters.

**Keep GPT-4o-mini for both (status quo).** What I picked. Worth flagging that the Phase 6 experiments turned this from an inherited default into something with measured calibration (Pearson=0.82 vs baseline) on at least these 10 CS queries.

**Few-shot calibration examples instead of abstract anchor descriptions.** Run 2's 0.0/0.5/1.0 anchor prose is what caused the score quantization. Replacing the anchors with (query, chunks, target score) example triples would probably produce continuous scores and cleaner Pearson values. Not running this in Day 6, but it's the right prompt-design fix for the next round of evaluation work.

**Drop LLM scoring entirely and use only the embedding-cosine scorer.** The production `score_groundedness()` already gives a calibrated, deterministic score, so the LLM scoring path is only earning its keep through the explanation generation and the multi-criteria reasoning that the cosine scorer can't do. If the explanation gets dropped, the LLM call becomes redundant. I'm keeping it because the explanation is genuinely useful for interpretability when reviewing failures.

---

## Quantified Validation

Numbers from `docs/iteration-log.md` 6e Run 2 entry. Pearson values carry ±0.05 quantization noise per the Context section.

The three correlations:

| Metric | Value | p-value |
|---|---|---|
| Pearson(GPT, baseline) | +0.8172 | 0.0039 |
| Pearson(Ollama, baseline) | +0.6796 | 0.0306 |
| Pearson(GPT, Ollama) | +0.7982 | 0.0056 |

The first one is what drove the decision: GPT lines up with the production scorer. The second is the 0.14 gap that disqualifies Ollama for scoring. The third (inter-model agreement) is interesting but doesn't speak to calibration.

Latency on the scoring task (Run 2):

| Model | Mean latency | Ratio |
|---|---|---|
| GPT-4o-mini | 1504 ms | 1.00x |
| Ollama qwen3:8b | 1465 ms | 0.97x |

Latency on the explanation-generation task (Run 1, kept for reference):

| Model | Mean latency | Std | Ratio |
|---|---|---|---|
| GPT-4o-mini | 1570 ms | 640 ms | 1.00x |
| Ollama qwen3:8b | 739 ms | 59 ms | 0.47x (2.1x faster) |

The 2.1x Ollama advantage from Run 1 doesn't transfer to Run 2's reasoning-over-chunks task. Run 1's speed is specific to short text generation.

Structured-output success was 100% for both models on both runs, using `instructor.from_litellm` with `mode=instructor.Mode.JSON` for Ollama.

Score distributions under the anchor-based prompt:

| Model | Observed values | Largest divergence from baseline |
|---|---|---|
| GPT-4o-mini | {0.0, 0.5} | q03: GPT=0.5 vs baseline=0.60 (−0.10) |
| Ollama qwen3:8b | {0.0, 0.5, 0.9} | q03: Ollama=0.9 vs baseline=0.60 (+0.30) |

---

## Consequences

The GPT-4o-mini scoring path now has a measured calibration number against the production embedding scorer (Pearson=0.82 on the 10 CS queries) instead of being trusted by default. The validation only covers `queries_v1.json` CS queries under the query-as-proxy regime; it doesn't extend to non-CS domains or to generated responses. ADR-006's proxy-regime limitation applies here too.

For dev iteration where API cost matters more than score calibration, the recorded option is Ollama qwen3:8b for explanations, GPT-4o-mini for scoring. That keeps production-relevant calibration on the path that affects routing while cutting the call volume that doesn't.

Run 1's "Ollama for dev, GPT for prod" framing was based on a 2.1x latency win that was real but only for the easy task (short text generation). Run 2 found parity (0.97x) on the harder reasoning-over-chunks task. The Run 1 latency numbers stay in the iteration log labeled as explanation-generation only.

For the next round of evaluation prompt design: use few-shot (query, chunks, target score) triples instead of abstract anchor descriptions. Abstract anchors collapsed the score range to 2 or 3 values and compressed the Pearson signal in Run 2.

(Python analogue: validating that a lightweight mock returns the same routing decision as the live dependency. The LLM scoring path used to be the mock that nobody had checked.)
