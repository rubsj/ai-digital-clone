# ADR-007: LLM-Based Evaluation Scoring Viability

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation / Model Selection
**Status:** Accepted
**Date:** 2026-04-27

---

## Context

The production evaluator (`src/evaluation/evaluator.py`) has two distinct sub-tasks:

1. **Scoring** — computing a groundedness score in [0,1] that calibrates with the production embedding-cosine scorer (`score_groundedness()`, which uses sentence-level OpenAI embedding cosine similarity, not keyword overlap). This value feeds the `0.4×style + 0.4×groundedness + 0.2×confidence` weighted formula and determines deliver/fallback routing.

2. **Explanation generation** — producing a one-sentence natural-language explanation of the evaluation result. This is surfaced to the caller but does not affect routing decisions.

Both sub-tasks currently use GPT-4o-mini via `instructor.from_litellm`. The question is whether a local model (Ollama qwen3:8b) can substitute for either sub-task, reducing API cost at acceptable quality.

**Experiment.** Phase 6 Run 2 (`scripts/experiment_6e_run2_groundedness_agreement.py`) had both models independently score groundedness from (query, top-5 chunk texts) for 10 queries from `data/eval/queries_v1.json`. The production embedding-cosine scorer provided a calibration baseline. Three Pearson correlations were computed: GPT-vs-baseline, Ollama-vs-baseline, and GPT-vs-Ollama inter-model.

**Audit note.** Phase 6 Run 1 (`scripts/experiment_6e_local_vs_api.py`) was a prior attempt; both models received pre-computed component scores and applied the same deterministic formula, producing a degenerate Pearson=1.0. Run 1's latency measurement (Ollama 2.1x faster) reflects the explanation-generation sub-task only, not the scoring sub-task. Run 2 corrects the experimental design; both runs are preserved in `docs/iteration-log.md` for audit.

**Quantization caveat.** The Run 2 prompt included explicit 0.0/0.5/1.0 calibration anchor descriptions. Both models quantized scores to those anchor points rather than scoring on a continuous scale (GPT: {0.0, 0.5}; Ollama: {0.0, 0.5, 0.9}). The Pearson values in this ADR are real signal — rank-order is preserved — but the absolute correlation values carry approximately ±0.05 quantization noise. Directional findings are robust; precise threshold claims about the exact correlation values are not.

---

## Decision

**GPT-4o-mini: retain for both scoring and explanation generation in production.** Pearson(GPT, baseline) = 0.8172 (p=0.0039) at latency parity (GPT mean=1504ms, Ollama mean=1465ms, ratio=0.97x). Calibration to the production embedding scorer is sufficiently strong to support LLM-based groundedness scoring as a viable alternative where explanation generation and multi-criteria reasoning are also needed. No change to `src/evaluation/evaluator.py`.

**Ollama qwen3:8b: approved for explanation generation; not approved for evaluation scoring.** Pearson(Ollama, baseline) = 0.6796 (p=0.0306), a 0.14 deficit relative to GPT on the same baseline. The calibration gap produces directional disagreements on individual queries: q03 (binary search) Ollama=0.9 vs GPT=0.5 vs baseline=0.60 — Ollama's high score likely reflects model knowledge rather than a grounded assessment of whether the retrieved chunks answer the query. For explanation generation (Run 1: mean=739ms, std=59ms, 100% structured-output success via `instructor.Mode.JSON`), Ollama is a viable lower-cost option where score calibration is not required.

The production evaluator's configuration does not change as a result of this ADR. An optional explanation-generation split (Ollama for dev, GPT for prod scoring) is recorded here as a documented option for teams where API cost reduction is a priority; it is not mandated.

---

## Alternatives Considered

**Ollama qwen3:8b for both scoring and explanation generation.** Reduces API costs to near-zero for all evaluation. Rejected for production scoring: the 0.14 Pearson gap (0.68 vs 0.82) produces directional disagreements on individual queries that would alter routing decisions. At a 0.75 deliver/fallback threshold with a 0.4 groundedness weight, a 0.4-point absolute disagreement (q03: Ollama=0.9 vs baseline=0.60) shifts final score by 0.16 — enough to flip a fallback to a deliver. Acceptable for dev iteration; not acceptable for production where the routing decision is consequential.

**GPT-4o-mini for both sub-tasks (status quo).** No change. The Phase 6 experiments produced positive evidence for this choice (Pearson=0.82 vs baseline), upgrading the prior from "default" to "validated on 10 CS queries." The status quo is now a confirmed decision, not inherited inertia.

**Few-shot calibration examples instead of abstract scale anchors.** Run 2 used abstract 0.0/0.5/1.0 anchor descriptions, which caused score quantization. Using (query, chunks, score) example triples instead would likely produce continuous scores, higher Pearson discriminability, and a cleaner comparison. Not implemented in Day 6 (do not re-run); flagged as a prompt-design improvement for future evaluation experiments.

**Embedding-cosine scorer only (no LLM scoring).** The production `score_groundedness()` already provides a calibrated, deterministic groundedness score. The LLM scoring path adds explanation generation and multi-criteria reasoning that the embedding-cosine scorer cannot produce. If explanation generation is dropped, the embedding-cosine scorer alone is sufficient and no LLM call is needed for scoring. Kept as a known option; the current design retains the LLM explanation path for its interpretability value.

---

## Quantified Validation

All numbers from `docs/iteration-log.md` 6e (Run 2) entry; no fresh measurement in this ADR. Pearson values carry ±0.05 quantization noise (see Context section).

**Three-Pearson framework (most to least informative):**

| Metric | Value | p-value | Interpretation |
|---|---|---|---|
| Pearson(GPT, baseline) | +0.8172 | 0.0039 | Core actionable finding — GPT calibrated to prod scorer |
| Pearson(Ollama, baseline) | +0.6796 | 0.0306 | Secondary — Ollama materially less calibrated (0.14 gap) |
| Pearson(GPT, Ollama) | +0.7982 | 0.0056 | Tertiary — inter-model agreement, not calibration signal |

**Latency — scoring task (Run 2):**

| Model | Mean latency | Ratio |
|---|---|---|
| GPT-4o-mini | 1504 ms | 1.00x (reference) |
| Ollama qwen3:8b | 1465 ms | 0.97x (parity) |

**Latency — explanation generation task (Run 1, for reference):**

| Model | Mean latency | Std | Ratio |
|---|---|---|---|
| GPT-4o-mini | 1570 ms | 640 ms | 1.00x (reference) |
| Ollama qwen3:8b | 739 ms | 59 ms | 0.47x (2.1x faster) |

Run 1's 2.1x Ollama latency advantage does not transfer to Run 2's reasoning-over-chunk-texts task (0.97x parity). Run 1 speed is task-specific to simple text generation.

**Structured-output success:**

Both runs: 100% / 100% (GPT and Ollama) using `instructor.from_litellm` with `mode=instructor.Mode.JSON` for Ollama.

**Score distribution under quantization:**

| Model | Observed values | Largest divergence from baseline |
|---|---|---|
| GPT-4o-mini | {0.0, 0.5} | q03: GPT=0.5 vs baseline=0.60 (−0.10) |
| Ollama qwen3:8b | {0.0, 0.5, 0.9} | q03: Ollama=0.9 vs baseline=0.60 (+0.30) |

---

## Consequences

The production evaluator's GPT-4o-mini scoring path is now validated (not just defaulted) against the embedding-cosine baseline at Pearson=0.82 on 10 CS queries. This validation is scoped to the `queries_v1.json` CS query set and the query-as-proxy input regime; it does not extend to non-CS domains or generated responses (the proxy-regime limitation documented in ADR-006 applies here too).

An optional explanation-generation split is documented: for dev iteration where reducing API cost is the priority, Ollama qwen3:8b generates explanations while GPT-4o-mini retains scoring. This preserves production-relevant calibration while reducing API call volume in the dev loop.

Run 1's "Ollama for dev, GPT for prod" recommendation was based on a 2.1x latency advantage that was task-specific to Run 1's degenerate experiment. Run 2 corrected the experimental design and found parity (0.97x) on the harder scoring task. Any citation of Run 1's latency numbers should specify that they reflect the explanation-generation sub-task, not the scoring sub-task.

Future evaluation prompt design should use few-shot calibration examples (query + chunks + target score triples) rather than abstract scale anchor descriptions. Abstract anchors cause score quantization that reduces Pearson discriminability, as observed in Run 2.

(In Python service terms, this is analogous to validating that a lightweight mock returns the same routing decision as the live dependency — GPT-4o-mini's LLM scoring now has measured calibration against the production embedding scorer rather than inherited trust from the original design choice.)
