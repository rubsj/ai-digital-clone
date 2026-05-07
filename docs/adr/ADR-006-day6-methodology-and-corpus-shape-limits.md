# ADR-006: Day 6 Methodology and Corpus-Shape Limits

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation / Measurement Design
**Status:** Accepted
**Date:** 2026-04-27

---

## Context

Three independent Day 6 experiments produced results that look like absence-of-effect but reflect measurement-design choices interacting with corpus characteristics and input proxy decisions.

**Phase 2 (6a Run 2) — Cohere reranker bimodal behavior on this corpus.** ADR-002 carried forward the P5 finding that Cohere reranking provides a ~20% Recall@5 lift. Day 6 experiments on the `open-phi/textbooks` CS corpus show this lift is corpus-shape sensitive. Reranker relevance scores cluster bimodally: 4 queries with Cohere max > 0.20 (q03 binary search max=0.751, q04 stacks/queues max=0.999, q06 routing protocols max=0.372–0.421), and 6 queries with Cohere max < 0.05 (q01 TCP, q05 isolation levels, q07 page replacement, q08 DP/greedy, q09 buffer overflow, q10 cache coherence). There is no smooth middle ground. On the 6 near-zero queries, both embedding configurations (OpenAI and MiniLM) collapse to near-identical post-rerank groundedness regardless of which 20 candidate chunks were retrieved. P6's prose-heavy programming-textbook corpus with broad CS queries elicits binary verdicts from the reranker, not a percentage lift.

**Phase 4 (6c) — Query-as-proxy pins style score, making weight sensitivity unmeasurable.** Short CS queries (10–15 words) do not resemble Torvalds' verbose kernel emails, so the style component scores structurally at ≈0.50 under the proxy regime (mean=0.5023, std=0.0101 across all 10 queries) rather than the production range of 0.80–0.95. With style structurally constant, sweeping three weight configurations produced Δ(style_heavy − default) = +0.0039 (+0.7%) and Δ(ground_heavy − default) = −0.0038 (−0.7%) — both below any meaningful threshold. The confidence scorer also shows proxy artifacts: `completeness` and `uncertainty_penalty` sub-components are mathematically pinned at 1.0 under proxy, biasing confidence_mean upward (0.7070). The production weights 0.4/0.4/0.2 are retained by inertia, not validated by Day 6 evidence.

**Phase 5 (6d) — Within-email variance dominates between-period mean shifts for Torvalds style evolution.** The 2018-09 behavioral shift (Torvalds' public apology and temporary leave) was expected to produce a detectable tone change at individual-email feature resolution. Using the criterion |pre_mean − post_mean| > 2 × std(feature on larger partition), no feature cleared the threshold across 11,052 partitioned emails: sentiment Δ = −0.00437 (2σ = 0.197), capitalization Δ = −0.00150 (2σ = 0.034), exclamations Δ = +0.00019 (2σ = 0.050), formality Δ = +0.01680 (2σ = 0.212). The largest signal, formality at +0.017, represents 8% of its 2σ band. High within-email variance (std ≈ 0.10–0.21 for sentiment and formality) swamps inter-period mean shifts (|Δ| = 0.0002–0.017). The null result reflects a measurement resolution limit, not confirmation that no behavioral change occurred.

All three findings share a common structure: an experiment designed to measure X produced a result that looks like "no effect" but actually reflects measurement-design limit Y. None are anomalies in isolation; together they indicate that corpus shape and input proxy are first-class variables for this evaluation setup, not background assumptions.

---

## Decision

Accept these three findings as documented measurement-design limits with no production configuration changes. The production pipeline (weights 0.4/0.4/0.2, OpenAI embeddings, Cohere reranker) retains its Day 5 configuration. Three future-work items are flagged:

1. **Re-measure weight sensitivity against generated responses** (not query-as-proxy). When the StyleCrew is invoked in production, style scores are in the 0.80–0.95 range; weight sweeps on that range would produce production-relevant guidance. Out of scope for Day 6.

2. **Re-measure Torvalds style evolution at population level.** Monthly rolling mean aggregation (shown in `docs/images/6d-style-evolution.png`) may reveal trends not visible at individual-email resolution. A mixed-effects model partitioning within-email vs between-period variance is the statistically appropriate test. Out of scope for Day 6.

3. **Qualify the ADR-002 Cohere lift claim as corpus-shape sensitive.** The ~20% Recall@5 figure from P5 should be amended to note that on P6's programming-textbook corpus with broad CS queries, Cohere provides bimodal verdicts rather than a smooth percentage lift. The decision (keep Cohere reranker) is unchanged; the precision of the claim is not.

---

## Alternatives Considered

**Dismiss each finding individually as underpowered.** Each null result could be attributed to the specific experiment being too small (10 queries, one corpus). Technically defensible but misses the common cause. All three findings trace to measurement-design choices suppressing the signal the experiment was designed to detect — treating them as isolated underpowered experiments would obscure the pattern and lose the actionable re-measurement guidance.

**Re-run with corrected experimental designs within Day 6.** Generate actual responses, aggregate at population level, or expand the corpus before closing the day. Rejected: Day 6 is measurement, not construction (per day6-plan.md §Context). Re-running would extend scope with no guarantee of production-relevant findings within the day's envelope. The value of documenting the limits cleanly exceeds the value of re-running within Day 6.

**Write a targeted ADR-002 amendment.** Produce a formal update to ADR-002 covering the Cohere corpus-shape finding. Deferred: ADR-002 covers the broader RAG configuration choice (embeddings + reranking + chunking), not just Cohere reranking behavior in isolation. A targeted ADR-002 amendment is the cleaner path and is flagged as a follow-on task for the next iteration day; writing it here would conflate two decisions.

---

## Quantified Validation

All numbers from `docs/iteration-log.md` Day 6 entries; no fresh measurement in this ADR.

**Phase 2 (6a Run 2) — Cohere bimodal behavior:**

| Subset | Cohere max range | Δ groundedness (OA − MiniLM) |
|---|---|---|
| High-signal (q03, q04, q06) | 0.372–0.999 | Differentiated |
| Near-zero (q01, q05, q07, q08, q09, q10) | < 0.012 | ≤ 0.001 (collapsed) |
| Full 10-query mean post-rerank | — | +0.0114 (+2.5%) |

P5 prior: Cohere provides ~20% Recall@5 lift. P6 observed: 2.5% mean Δ groundedness; 6/10 queries collapsed to near-identical scores regardless of embedding model.

**Phase 4 (6c) — Proxy regime pins style:**

| Config | mean_final | Δ vs default |
|---|---|---|
| Default 0.4/0.4/0.2 | 0.5278 ± 0.0548 | — |
| Style-heavy 0.5/0.3/0.2 | 0.5317 ± 0.0438 | +0.0039 (+0.7%) |
| Ground-heavy 0.3/0.5/0.2 | 0.5240 ± 0.0658 | −0.0038 (−0.7%) |

style_mean = 0.5023, std = 0.0101 (effectively constant across all 10 queries under proxy).

**Phase 5 (6d) — Per-email resolution insufficient:**

| Feature | Pre-2018 | Post-2018 | Δ | 2σ band | Signal ratio |
|---|---|---|---|---|---|
| sentiment | 0.0747 | 0.0704 | −0.00437 | 0.197 | 2.2% |
| capitalization | 0.0218 | 0.0203 | −0.00150 | 0.034 | 4.4% |
| exclamations | 0.0047 | 0.0049 | +0.00019 | 0.050 | 0.4% |
| formality | 0.4884 | 0.5052 | +0.01680 | 0.212 | 7.9% |

No feature clears the 2σ threshold. Largest signal (formality +0.017) = 8% of its 2σ band.

---

## Consequences

ADR-002's Cohere lift claim is directionally supported (reranker helps on high-signal queries) but its magnitude does not generalize across corpus shapes. Any future corpus that differs structurally from P5's financial reports should validate Cohere lift independently before relying on the P5 percentage.

The production weights 0.4/0.4/0.2 are accepted as is. There is no data supporting or refuting them under production conditions (generated responses with style ≥ 0.80). The weight-sensitivity question is deferred until responses can be used as the scoring input rather than query proxies.

The PRD §8 exit criterion ("style evolution chart shows measurable shift") is not met at individual-email feature resolution. The handover note documents this as a null result with the measurement-resolution explanation, not as "no behavioral change occurred." Monthly aggregation shows directional trends in the chart; the individual-email significance test is the appropriate statistical bar, and it did not clear.

These three limits are specific to this corpus, this input proxy, and this email feature resolution. They do not invalidate the P6 pipeline design; they qualify the conditions under which the pipeline's measurement layer produces reliable signal.

(In Spring the equivalent would be a test suite that passes because all external dependencies are mocked — the tests are valid artifacts, but their signal about production behavior is limited by what the mocks can replicate.)
