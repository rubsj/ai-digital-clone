# ADR-006: Day 6 Methodology and Corpus-Shape Limits

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation
**Status:** Accepted
**Date:** 2026-04-27

---

## Context

Three Day 6 experiments returned what looks like absence-of-effect. After working through the numbers I think all three are measurement artifacts, not real null results. Worth writing down before they get cited as evidence the underlying effects don't exist.

The first is the Cohere reranker. ADR-002 carries forward a P5 finding that Cohere gives roughly a 20% Recall@5 lift, and I assumed that would transfer to P6. On the `open-phi/textbooks` CS corpus it doesn't, or at least not as a smooth percentage. The reranker scores split bimodally: 4 queries land high (q03 binary search max=0.751, q04 stacks/queues max=0.999, q06 routing protocols max=0.372–0.421) and 6 queries land near zero (q01 TCP, q05 isolation levels, q07 page replacement, q08 DP/greedy, q09 buffer overflow, q10 cache coherence). Nothing in the middle. On the 6 near-zero queries, both OpenAI and MiniLM embeddings collapse to nearly identical post-rerank groundedness regardless of which 20 candidate chunks came in. The reranker is giving binary verdicts on this corpus, not a percentage lift. Programming textbooks plus broad CS queries seems to be the wrong shape for the lift behavior P5 saw on financial reports.

The second is the weight-sensitivity sweep. The production scorer combines style, groundedness, and confidence with weights 0.4/0.4/0.2, and I wanted to see whether those weights actually matter. The problem is that I used queries as the input proxy for the scoring, and CS queries are 10–15 words of textbook prose, nothing like Torvalds' verbose kernel emails. The style component scored at ≈0.50 across all 10 queries (mean=0.5023, std=0.0101) instead of the production range of 0.80–0.95. With style structurally pinned, sweeping the weights produced Δ(style_heavy − default) = +0.0039 and Δ(ground_heavy − default) = −0.0038. Both well below noise. The confidence scorer has the same proxy artifact: `completeness` and `uncertainty_penalty` are mathematically pinned at 1.0 under the proxy regime, biasing confidence_mean to 0.7070. The production weights are still in place because nothing has actually been measured against them, not because Day 6 confirmed them.

The third is Torvalds' style evolution around the 2018-09 apology and leave. I expected a detectable tone shift at individual-email resolution. Using |pre_mean − post_mean| > 2 × std on the larger partition as the threshold, no feature clears it across 11,052 emails: sentiment Δ = −0.00437 (2σ = 0.197), capitalization Δ = −0.00150 (2σ = 0.034), exclamations Δ = +0.00019 (2σ = 0.050), formality Δ = +0.01680 (2σ = 0.212). The biggest signal, formality, is 8% of its 2σ band. Within-email variance (std ≈ 0.10–0.21 for sentiment and formality) swamps the inter-period mean shifts (|Δ| = 0.0002–0.017). That's a measurement-resolution problem, not evidence the behavioral shift didn't happen.

All three are the same shape: the experiment got swamped by something about how I set it up rather than by the absence of the effect I was trying to detect. Worth treating corpus shape and input proxy as variables that need to be picked deliberately for this kind of evaluation, not assumptions that hold by default.

---

## Decision

Accept the three findings as documented limits and ship nothing new. Production stays on weights 0.4/0.4/0.2, OpenAI embeddings, and the Cohere reranker, which is the Day 5 configuration.

Three things go on the followup list. The weight-sensitivity sweep needs to be re-run against actual generated responses rather than queries. When the StyleCrew runs in production, style scores sit in 0.80–0.95, and a sweep on that range would actually tell me something. The Torvalds style evolution needs a population-level test. Monthly rolling means (the chart in `results/charts/07-style-evolution.png` is suggestive) plus a mixed-effects model partitioning within-email vs between-period variance is the right statistical bar; the per-email significance test was the wrong instrument. And the ADR-002 Cohere claim needs an amendment noting that the 20% Recall@5 figure was on P5's financial reports and doesn't transfer cleanly to programming-textbook content with broad queries. The decision to keep Cohere isn't changing; the precision of the claim is.

None of these are inside Day 6 scope.

---

## Alternatives Considered

**Treat each finding as an underpowered one-off.** Every null here could be blamed on a small experiment: 10 queries, one corpus, one author. Technically defensible per result, but it misses that all three trace to the same class of cause. Treating them as isolated would lose the pattern and the actionable re-measurement guidance.

**Re-run the experiments inside Day 6 with corrected setups.** Generate actual responses for the weight sweep, aggregate Torvalds emails monthly, expand the corpus for the reranker test. Day 6 is scoped as measurement, not construction (`day6-plan.md §Context`), and re-running blows the day's envelope with no guarantee of production-relevant numbers. Documenting the limits cleanly is more useful than a partial re-run.

**Write the ADR-002 amendment now.** Tempting, but ADR-002 covers the full RAG configuration choice (embeddings, reranking, chunking), not Cohere behavior in isolation. A targeted amendment is cleaner as its own followup. Folding it in here would conflate two decisions.

---

## Quantified Validation

Numbers from `docs/iteration-log.md` Day 6 entries.

Cohere bimodal behavior (Phase 2, 6a Run 2):

| Subset | Cohere max range | Δ groundedness (OA − MiniLM) |
|---|---|---|
| High-signal (q03, q04, q06) | 0.372–0.999 | Differentiated |
| Near-zero (q01, q05, q07, q08, q09, q10) | < 0.012 | ≤ 0.001 (collapsed) |
| Full 10-query mean post-rerank |   | +0.0114 (+2.5%) |

P5 prior: ~20% Recall@5 lift. P6 observed: 2.5% mean Δ groundedness; 6 of 10 queries collapsed to near-identical scores regardless of embedding model.

Weight sweep under proxy regime (Phase 4, 6c):

| Config | mean_final | Δ vs default |
|---|---|---|
| Default 0.4/0.4/0.2 | 0.5278 ± 0.0548 |   |
| Style-heavy 0.5/0.3/0.2 | 0.5317 ± 0.0438 | +0.0039 (+0.7%) |
| Ground-heavy 0.3/0.5/0.2 | 0.5240 ± 0.0658 | −0.0038 (−0.7%) |

style_mean = 0.5023, std = 0.0101, effectively constant across all 10 queries under proxy.

Per-email feature resolution for Torvalds 2018-09 shift (Phase 5, 6d):

| Feature | Pre-2018 | Post-2018 | Δ | 2σ band | Signal ratio |
|---|---|---|---|---|---|
| sentiment | 0.0747 | 0.0704 | −0.00437 | 0.197 | 2.2% |
| capitalization | 0.0218 | 0.0203 | −0.00150 | 0.034 | 4.4% |
| exclamations | 0.0047 | 0.0049 | +0.00019 | 0.050 | 0.4% |
| formality | 0.4884 | 0.5052 | +0.01680 | 0.212 | 7.9% |

Largest signal (formality, +0.017) is 8% of its 2σ band. Nothing clears the threshold.

---

## Consequences

The ADR-002 Cohere claim is directionally right (the reranker helps when it has signal to work with) but the magnitude doesn't generalize. Future corpora that look unlike P5's financial reports need their own reranker validation before borrowing the percentage.

Production weights stay at 0.4/0.4/0.2. There is no Day 6 evidence supporting them and no evidence refuting them, because the experiment that was supposed to test them couldn't actually test them. The sweep gets re-run when generated responses are available as the input.

The PRD §8 exit criterion ("style evolution chart shows measurable shift") is not met at individual-email resolution. The chart shows directional movement at monthly aggregation, but the per-email significance test is the right statistical bar and it didn't clear. That's logged as a null with the resolution explanation, not as evidence the shift didn't happen.

These three limits are about this corpus, this proxy, and this feature resolution. They don't change the pipeline design; they tell me where its measurement layer can produce reliable signal and where it can't.

(Spring analogue: a green test suite where every external dependency is mocked out. Tests pass, but they're not telling you what you wanted to know.)
