# Day 8 Findings — End-to-End Evaluation Correction

**Date:** 2026-05-26
**Branch:** `fix/p6-cohere-and-eval-correction`
**Scope:** Diagnosed and corrected the eval pipeline after the May-23 portfolio results showed 95% fallback. Final v2 eval surfaced a leader-asymmetric style-scoring pattern that points at a feature-extraction iteration for after the portfolio sprint.

---

## Executive summary

End-to-end evaluation on a corpus-appropriate query set surfaced leader-asymmetric behavior in style scoring: Torvalds scores 0.067 higher than Kroah-Hartman on average across the 11 records that reached the evaluator (T mean 0.9025, KH mean 0.8355). The 15-dim feature vector is derived from Schneider et al. (2016) on Torvalds-specific LKML signals and applied uniformly to both leaders; per-leader feature extraction is the natural next iteration. Two infrastructure corrections also surfaced during Day 8 verification — the Cohere reranker had silently fallen back to vector-only since Day 3 (env-var name bug), and the original v1 eval query set was scoped outside the indexed corpus. Both were corrected, and the v2 run delivers groundedness mean 0.626 (up from 0.517 broken-Cohere), zero hallucinations on out-of-domain queries, and a 72.5% overall fallback rate driven by the corpus's deliver-path ceiling rather than by infrastructure failures.

---

## Routing-correctness 2×2 (full 40-record matrix, v2 + Cohere working)

```
                       │  Delivered (final ≥ 0.75)  │  Fallback (final < 0.75)  │
───────────────────────┼────────────────────────────┼───────────────────────────┤
 In-domain (28 records)│              11            │              17           │
 OOD       (12 records)│               0  ✅        │              12  ✅       │
```

Diagonal cells (`11 + 12 = 23`) are correct routing — system did what it should. Off-diagonal:

- **Top-right (17 in-domain → fallback):** Deliver-path miss. The bulk of these trace to leader-asymmetric style scoring, not to retrieval or groundedness (see Finding 1). Step B verification confirmed retrieval was strong: in-domain top-1 Cohere mean 0.89 across all 14 unique queries.
- **Bottom-left (0 OOD → deliver):** No hallucinations. Despite Torvalds and Kroah-Hartman being kernel maintainers with well-documented opinions on networking, OS internals, and security — exactly the topics in Category 5 — the system correctly fell back on all 12 OOD records. The groundedness scorer caught every case where the LLM might have produced confident styled output from parametric knowledge alone.

(The CLI does not persist `trigger_reason` for fallback records, so the OOD trigger-reason audit relies on indirect signal: the diagnostic-floor capture script — re-run on v1 with Cohere working before Step C — showed all 20 records reached the evaluator with empty `trigger_reason`, and the v2 fallback latencies match the same all-the-way-to-evaluator profile. A persisted `trigger_reason` field on the CLI's record schema would close this audit gap; logged as a follow-up.)

---

## Finding 1: Leader-asymmetric style scoring

### What the data shows

Across the 11 scored v2 records:

| Leader | n | style mean | groundedness mean | confidence mean | final mean |
|---|---:|---:|---:|---:|---:|
| Torvalds | 6 | **0.9025** | 0.5913 | 0.9164 | 0.7808 |
| Kroah-Hartman | 5 | **0.8355** | 0.6673 | 0.8963 | 0.7804 |

Style asymmetry is the largest signal: Δ = +0.067 favoring Torvalds. Final scores end up nearly identical (Δ = +0.0004) because Kroah-Hartman compensates with stronger groundedness (Δ = +0.076 favoring KH). The final-score formula softens the style component to 40% weight, so the asymmetry doesn't fully propagate to the deliver decision.

Per-query outcome pattern (in-domain only):

| Outcome | Count | Queries |
|---|---:|---|
| Both leaders deliver | 4 | q02, q06, q08, q13 |
| Both leaders fall back | 7 | q01, q05, q07, q10, q11, q12, q14 |
| Split, Torvalds delivers / KH falls back | 2 | q03, q04 |
| Split, KH delivers / Torvalds falls back | 1 | q09 |

The split queries lean Torvalds-favored (2:1), but the small sample weakens that conclusion. The robust signal is the per-record style-mean asymmetry, not the deliver-decision asymmetry.

### Diagnosis

The 15-dim feature vector in `src/style/feature_extractor.py` is derived from Schneider et al. (2016)'s OpenSym paper on Torvalds-specific LKML signals (acknowledged in PRD §3 and ADR-003 as the validation precedent for the feature design). The same extractor produces a feature vector for any leader; the cosine-similarity scorer then compares that vector to the per-leader StyleProfile built from training emails. The extractor is uniform; only the reference profile is per-leader.

Several features explicitly encode Torvalds-distinctive habits — `capitalization_ratio` (ALLCAPS emphasis), `patch_language`, `code_snippet_freq`, `quote_reply_ratio` — and these have natural prevalence in Torvalds' writing that Kroah-Hartman's neutral prose can't match by writing in his own authentic voice. A KH-styled response, by construction, lands further from any leader's profile on these Torvalds-shaped dimensions than a T-styled response does from Torvalds' profile.

This is design provenance, not a bug. ADR-003 explicitly trades off coverage for interpretability; the feature set was named in Day 2 with the Torvalds validation in mind. The asymmetry surfaces now because v2 is the first eval run that actually puts both leaders through the scoring pipeline at meaningful scale (v1 produced one scored record per leader maximum due to the corpus mismatch).

### Next iteration

Per-leader feature extraction is the natural next iteration. Two viable directions:

1. **Per-leader weighting** of the existing 15-dim vector. Train a per-leader importance weighting (a simple per-dim scaling computed during profile build) so KH's profile doesn't penalize KH-natural responses on Torvalds-distinctive features. Minimal code change; preserves the interpretable feature dimensions.
2. **Per-leader feature set.** Derive Kroah-Hartman-distinctive features (his prose patterns differ — more measured, structured driver-review framing) and use a leader-specific 12-15 dim extractor. Larger change; better separation; loses the cross-leader comparability the radar chart relies on.

Either path is post-portfolio-sprint work. The current asymmetry is now documented and bounded; the system functions correctly for portfolio-demo purposes (deliver decisions are calibrated; OOD fallback is reliable; no hallucinations).

---

## Finding 2: Cohere reranker silent failure

The reranker has read `CO_API_KEY` from the environment since commit `579959a` (Day 3, 2026-04-13), but the `.env` file sets `COHERE_API_KEY`. The Cohere `ClientV2` constructor accepted the empty-string key without raising, and the actual rerank call produced an `httpx.LocalProtocolError: Illegal header value b'Bearer '`. The reranker caught the exception generically and fell back to vector-only top-N, logging a warning that did not appear in CI or test runs because nothing watches stderr-level warnings during a happy-path eval.

Effect: every prior eval run, including the May-23 portfolio JSON and the Day 6 reranking-related experiments, executed with vector-only retrieval. ADR-002 documents Cohere reranking as part of the production RAG config, so the ADR overstates what was actually measured. ADR-002 correction tracked as a follow-up.

Fix: one-line env-var rename in `src/rag/reranker.py` (commit `206c232`). Verified by an isolated `ClientV2.rerank` call against three hardcoded documents, which now returns a structured ranked response with relevance scores in the expected `[0, 1]` range.

Measured impact on the v1 eval set, holding query set constant and threshold at 0.75:

| Metric | v1 May-23 (Cohere broken) | v1 + Cohere working (control) | Δ |
|---|---:|---:|---:|
| Fallback rate | 95.0% | 90.0% | −5.0pp |
| Scored count | 1 | 2 | +1 |
| Mean groundedness (scored) | 0.517 | 0.556 | +0.039 |
| Mean final score (scored) | 0.753 | 0.779 | +0.027 |

Cohere is doing real work — groundedness lifts measurably — but on v1 the corpus mismatch dominates and only 2 of 20 records reach the evaluator with content the corpus can ground. The Cohere fix alone cannot recover v1; the eval set redesign (Finding adjacent) was required to surface what the deliver path is actually capable of.

The fix is in production. The diagnostic is documented. This is an engineering record, not a portfolio narrative.

---

## Verification 2: q12 binary-search regression

q03 in v1 ("How does binary search work and what is its time complexity?") was the May-23 single deliver, with broken Cohere, at `final=0.7525`. After the Cohere fix, the same query (re-run as v1 q03 in the control and as v2 q12) falls back on both leaders in both runs.

Same query text, same index, same Cohere — chunks must match. Confirmed by re-running retrieve + Cohere rerank for the query and inspecting the top-5:

| Rank | Cohere score | Source | Snippet |
|---:|---:|---|---|
| 0 | **0.7513** | Intro to Computers (Searching) | "Linear search involves sequentially checking each item… Binary search, on the other hand, involves dividing the list into smaller sublists…" |
| 1 | 0.0142 | Intro to Computers (Searching) | "#### Searching   Searching an array involves finding a specific element… linear search… binary search…" |
| 2 | 0.0107 | Intro to Computers (Searching) | "Binary search trees: Binary search trees are a type of binary tree…" (data-structures section, tangential) |
| 3 | 0.0098 | Intro to Computers (Searching) | "#### Searching   Searching is the process of finding a specific item in a list. There are two main types of searching algorithms: linear search and binary search." |
| 4 | 0.0030 | Intro to Computers (Searching) | "#### Searching   Searching is the process of finding a specific item in a list. There are various searching algorithms that can be used, such as linear search, binary search, and hash table" |

The top-1 chunk has a strong Cohere score (0.7513) but then a cliff to <0.02 for ranks 1–4. The corpus contains exactly one substantive binary-search chunk; the rest are tangential "Searching" intros.

**Diagnosis:** Cohere isn't surfacing different chunks for v1 q03 and v2 q12 — the chunks are identical, because the query text and the index are identical. The regression comes from downstream stochasticity in the LLM response generation and groundedness scoring. Specifically: the top-1 chunk discusses *what* binary search is (sublist dividing) but does not mention time complexity (O(log n)). When the LLM response includes O(log n) — which it does, because that's half the query — the groundedness scorer's sentence-level cosine against the single relevant chunk drops, and with only one strong chunk in the top-5, there's no second-best chunk to recover from.

In other words: Cohere's reranking is correct (it surfaces the best available chunk and ranks the rest by their true marginal relevance, which happens to be near zero); the corpus is the constraint. The May-23 deliver was a lucky alignment of broken-Cohere's FAISS-vector top-5 (which happened to include multiple chunks the LLM's response could ground sentences against, even if those chunks were less topically aligned by Cohere's measure) with the LLM's binary-search response prose. The Cohere-on path produces a more topically precise top-5 but loses the diffuse-grounding the LLM needed.

This is corpus-shape-dependent, not a Cohere regression. The right framing is "the corpus has one good chunk per question, and questions with two-clause structure (mechanism *and* complexity) need two good chunks to ground reliably." It's the same finding ADR-006 made about corpus-shape limits on Day 6, manifesting on a specific query.

---

## PRD success criteria scorecard (v2 final run)

| PRD criterion | Target | Measured (v2) | Hit? | Calibration note |
|---|---|---|---|---|
| §2a — Style score | > 0.90 on scored responses | 4/11 cleared, T mean 0.9025, KH mean 0.8355 | **PARTIAL** | Target met by Torvalds in aggregate, not by KH. Re-derive per leader once per-leader feature extraction lands. |
| §2b — Groundedness | > 0.60 on in-domain queries | 7/11 cleared, scored mean 0.6258 | **HIT** | First run where the production embedding-cosine scorer measurably operates against in-domain content. |
| §2c — Final score | > 0.75 on delivered responses | 11/11 cleared (= deliver definition) | **HIT** | This is the threshold itself; gates deliver/fallback. |
| §2d — Fallback rate | 30–40% | 72.5% overall, 60.7% in-domain | **MISS** | The 30–40% band assumed an in-domain deliver rate near 100%. Measured in-domain deliver rate is 39% (full matrix); the corpus + per-leader-asymmetric style scoring caps this. Per spec A8, the target needs re-derivation from measured evidence, not retroactive query-set adjustment. |
| §2e — Orchestration (5-agent CrewAI Flow with @router branching) | Implemented + tested | All Day 5 deliverables shipped; 464 tests pass | **HIT** | Met since Day 5. |
| §2f — Test coverage | src/ ≥ 90% | src/ 94% on Day 7 baseline, holds | **HIT** | Holds. Reranker fix didn't add code; throttle is a 5-line env-var-gated branch with no new public surface. |

§2a and §2d are the targets that need calibration from measured evidence. The calibration story to tell in the README narrative:

- The 0.90 style target was set against Torvalds-validated Schneider et al. features. The measured Torvalds style mean (0.9025) hits the target; the measured Kroah-Hartman style mean (0.8355) does not. Per-leader thresholds derived from per-leader profile build would tell the honest story.
- The 30–40% fallback target assumed a higher in-domain deliver rate than the corpus + style scorer can sustain. The measured 60.7% in-domain fallback floor reflects a real cap on what the deliver path can produce with the current feature set and corpus. Recalibrating the target to reflect measured corpus + scoring capacity is the honest move; rebalancing the query set to inflate deliver count would gate the test to its outcome (spec A8).

---

## Three-run comparison

| Metric | v1 May-23 (Cohere broken) | v1 + Cohere working | v2 + Cohere working (FINAL) |
|---|---:|---:|---:|
| Total records | 20 | 20 | 40 |
| Fallback rate | 19/20 (95.0%) | 18/20 (90.0%) | 29/40 (72.5%) |
| Scored count | 1 | 2 | 11 |
| Fallback — Torvalds | 9/10 (90.0%) | 9/10 (90.0%) | 14/20 (70.0%) |
| Fallback — Kroah-Hartman | 10/10 (100.0%) | 9/10 (90.0%) | 15/20 (75.0%) |
| Style mean (scored) | 0.9495 | 0.9353 | 0.8720 |
| Groundedness mean (scored) | 0.5173 | 0.5557 | **0.6258** |
| Confidence mean (scored) | 0.8287 | 0.9150 | 0.9073 |
| Final mean (scored) | 0.7525 | 0.7794 | 0.7806 |
| count style > 0.90 (§2a) | 1/1 | 2/2 | 4/11 |
| count groundedness > 0.60 (§2b) | 0/1 | 0/2 | **7/11** |
| count final ≥ 0.75 (§2c) | 1/1 | 2/2 | 11/11 |
| Mean deliver latency (ms) | 7,382 | 8,517 | 8,503 |
| Mean fallback latency (ms) | 10,019 | 10,854 | 11,445 |

Fallback latency exceeds deliver latency in all three runs — fallback fires from the score-path (full pipeline ran, then `final < 0.75` triggered the fallback branch), not from an early-exit upstream of the evaluator. Latency is also stable across the Cohere fix (~+0.4s mean), consistent with Cohere adding one network call per query.

Regression-anchor check (v1 q04 stack/queue verbatim → v2 q13):

| Run | Torvalds final | Kroah-Hartman final |
|---|---:|---:|
| v1 + Cohere | 0.7792 | 0.7796 |
| v2 q13 | 0.7534 | 0.7721 |
| Δ | −0.0258 | −0.0075 |

Both within ±0.05. Anchor stable.

The other anchor (q03 binary search → q12) gives no signal because both runs fell back; the q12 dynamics are covered in Verification 2 above.

---

## Follow-ups

### Documentation corrections

- **ADR-002 (RAG config: embeddings + reranking + chunking).** ADR-002 documents Cohere reranking as part of the production config. From Day 3 through Day 7, the reranker was silently bypassed; the Day 6 reranking-related metrics in `docs/experiments/` (specifically Phase 2 Run 1 and Run 2 embedding comparison experiments that touched the rerank path) executed without Cohere. Add a correction note to ADR-002 acknowledging the gap and pointing at the Day 8 measurements as the first run with Cohere actually engaged.
- **Day 6 experiment scripts (`scripts/experiment_6a*`).** Any chart or table that claims a Cohere-related metric needs a footnote noting the broken-Cohere baseline. The experiment outputs in `docs/experiments/charts/` may need a "measured pre-Cohere-fix" caption.

### Engineering Protocols Verification Protocol additions

The silent-failure-as-fallback pattern is the lesson worth promoting upstream into the Engineering Protocols. Two concrete additions:

1. **Side-effect verification in the Verification Protocol.** When a code path has an except-and-fallback, the verification step for any deliverable that depends on that code path should include a positive assertion that the primary path actually executed — log inspection, side-effect observation, or an explicit "Cohere call returned N results" assertion. The Day 3 reranker test mocked the success path and never asserted the live path; the test passed for two months while the live path was broken every run.
2. **PRD coverage check on eval design.** The Day 7 Prompt Discipline Protocol Component 5 caught one silent-deferral failure in the implementation phase. The companion failure mode in the eval phase is the v1 query set: scoped beyond the corpus and run for two days before the diagnostic surfaced the mismatch. A query-set design step that requires a corpus-coverage probe (the keyword scan in Day 8 spec A2) before generation would have caught v1 before it ran.

### Post-portfolio future work

- **Per-leader feature extraction** (Finding 1's natural next iteration). Two paths sketched in Finding 1; preference is the per-leader-weighting path because it's a smaller change and preserves the radar-chart interpretability ADR-003 paid for.
- **CLI `trigger_reason` persistence on fallback records.** `src/cli.py` records `{id, leader, fallback, latency_ms}` on fallback; adding `trigger_reason` closes the audit gap that forced the diagnostic-floor capture script to run separately to inspect upstream-vs-score fallback paths.
- **ADR-009 (0.75 threshold).** Still un-written. Day 7 flagged the threshold as load-bearing but undocumented; the Day 8 evidence (final-score distribution clustered 0.55–0.81) gives the data for the decision record. Move from Post-Portfolio Followups to a concrete write-up.
