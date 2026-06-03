# Day 13 Session Notes — Torvalds Floor Investigation

**Date:** 2026-06-02  
**Data:** results/evaluation_day12_reeval2.json  
**Status at close:** Investigation reframed, not closed. Scorer-validity test is the next stage. No code changed, no fix started, floor untouched.

---

## What we set out to do

The investigation opened with a defined fork: H2 (off-source persona tails, localized, removable by prompt fix) vs H3 (terse voice grounds less throughout, a legitimate structural deficit requiring a floor or persona decision). Torvalds sits at 42.9% in-domain delivery against KH shipping above the floor. The goal was to adjudicate between H2 and H3 using span-level containment markup on the 84 responses in reeval2, then run excision arithmetic to quantify what fraction of the Torvalds-KH gap is excision-recoverable.

The method: blind an Opus pass over markup_input.json (42 records, 691 spans), label each span grounded/inferable/free, compute length-weighted containment fractions per clone and query, run the within-query discriminator (same chunks, two clones), classify each query H2/H3/blend, emit audit cards for pivotal spans. Phase C would un-blind and deliver a verdict.

The QC step changed the question before Phase C ran.

---

## What the markup showed

The blind Opus markup completed. 691 spans labeled across 84 responses (42 records, 2 clones each). Label distribution: 495 grounded, 140 inferable, 56 free. No contamination found in the blinded artifact; the A/B assignment held. Span counts were consistent with 7-10 sentence responses per record per clone.

---

## Scorer-vs-containment disagreement

The QC compared Opus containment fractions against the production groundedness scorer (ADR-004: sentence-level max cosine similarity against top-5 chunks) across all 14 queries.

**Production scorer, per query (mean across 3 passes):**

| Query | Category | Scorer T | Scorer KH | Gap | Scorer direction |
|-------|----------|----------|-----------|-----|-----------------|
| q01 | statistical_learning_ml | 0.522 | 0.583 | +0.061 | KH > T |
| q02 | statistical_learning_ml | 0.660 | 0.724 | +0.065 | KH > T |
| q03 | statistical_learning_ml | 0.555 | 0.597 | +0.042 | KH > T |
| q04 | statistical_learning_ml | 0.581 | 0.612 | +0.031 | KH > T |
| q05 | data_mining | 0.518 | 0.560 | +0.042 | KH > T |
| q06 | data_mining | 0.613 | 0.697 | +0.084 | KH > T |
| q07 | data_mining | 0.553 | 0.575 | +0.022 | KH > T |
| q08 | data_mining | 0.656 | 0.663 | +0.007 | KH > T |
| q09 | numerical_methods | 0.570 | 0.637 | +0.068 | KH > T |
| q10 | numerical_methods | 0.544 | 0.599 | +0.055 | KH > T |
| q11 | numerical_methods | 0.643 | 0.659 | +0.016 | KH > T |
| q12 | programming_fundamentals | 0.553 | 0.566 | +0.013 | KH > T |
| q13 | programming_fundamentals | 0.575 | 0.607 | +0.032 | KH > T |
| q14 | programming_fundamentals | 0.579 | 0.623 | +0.044 | KH > T |
| **Aggregate** | | **0.580** | **0.622** | **+0.041** | **KH > T** |

Scorer gives KH > T on all 14 queries without exception.

**Opus containment fractions, aggregate:** T=0.688, KH=0.672, gap -0.016 (reversed, toward Torvalds). Direction agreement with the scorer: 5 of 14 queries. On 9 of 14 queries the containment markup and the scorer point in opposite directions.

**Interpretation (provisional):** the scorer's per-leader gap does not reproduce as a containment deficit. Torvalds responses contain at least as much chunk-grounded material as KH responses by Opus markup, yet the scorer consistently ranks KH higher. What the scorer is measuring is not yet established. The disagreement is large enough and consistent enough across 14 queries that it is not noise. Whether the scorer is picking up voice or style differences rather than containment is an open question (see below).

---

## Retrieval-duplication bug

Separate from the scorer question, the markup identified a retrieval defect affecting 6 of 14 in-domain queries: q03, q07, q09, q10, q11, q14 each retrieved duplicate chunks (same passage appearing 2-3 times in the top-5). This reduces effective context to 3-4 distinct passages for those queries. The defect is leader-agnostic (chunks are shared across both clones per record, confirmed by byte-identity check in Phase A). It does not explain the per-leader scorer gap, but it depresses absolute grounding scores on roughly 43% of in-domain queries.

This is a distinct bug to triage separately. It is not part of the H2/H3 adjudication and was not the cause of the investigation reframe.

---

## Why H2/H3 is parked

The excision arithmetic in Phase B is designed to act on a real containment deficit: it compares grounded-core lengths between clones, classifies localized vs distributed free spans, and asks whether the lower clone's deficit is excision-recoverable. That test is meaningful when the lower clone actually has less grounded content.

The markup shows Torvalds is not the lower clone on containment. On 9 of 14 queries, Opus marked Torvalds responses as containing more grounded material than KH responses, reversing the scorer's direction. There is no containment deficit to excise. Running the H2/H3 pivot-card audit on these spans would be classifying noise. Parked without running.

The H2/H3 fork was constructed on the premise that the scorer gap reflects a containment gap. That premise did not survive the QC check.

---

## Open question for next stage

Is the scorer style-confounded? The scorer (sentence-level max cosine similarity against chunks) is intended to measure containment, but voice and sentence-level phrasing affect cosine similarity too. A response written in Torvalds' characteristically terse style may use fewer of the chunk's exact vocabulary items and produce lower cosine scores even when it is fully grounded.

Seed case from the markup: q02, both clones marked fully grounded by Opus, yet the scorer gaps them +0.065 (T=0.660, KH=0.724). If both responses are fully grounded but the scorer separates them by 0.065, the scorer is responding to something other than containment on that query.

The next stage is a scorer-validity test: hold containment constant (fully-grounded query pairs from the markup) and measure whether the scorer gap persists. If it does, the scorer is confounded by voice and the floor decision needs to account for that. If it does not, the gap requires another explanation.

---

## Scorer probe and verdict

The scorer-validity test was run on the same reeval2 data. No new generation. All numbers are given; none were recomputed here.

**Part 1 — Scorer identity confirmed.**

Function: `score_groundedness()` at `src/evaluation/groundedness_scorer.py`. Algorithm: regex sentence split → `text-embedding-3-small` via LiteLLM → per-sentence max cosine against top-k chunk embeddings → mean of per-sentence maxima. Wiring: `harness.run_leader_pair()` → `DigitalCloneFlow` → `ScoringEngine.score()` → `score_groundedness()` → `groundedness_score` field in reeval2. One path, no lookalike.

**Part 2 — Three probes.**

Probe A (verbosity): chars vs groundedness score, Pearson r=+0.394 pooled, p<0.001. Real, but a symptom, not the mechanism: in 3 of 7 near-equal containment queries (q02, q06, q13), Torvalds responses are longer in characters yet score lower.

Probe B (containment held equal): 7 of 14 queries had Opus grounded-fraction gap ≤0.05 between clones. The scorer separated all 7, with KH > T on every one, gaps +0.013 to +0.084. Char-length direction was inconsistent across the 7 cases, so raw response length is not the mechanism.

Probe C (mechanism, direct): among sentences with identical grounded status by Opus markup, Torvalds' characteristic meta and transition sentences score 0.17–0.27 per-sentence cosine (examples: "So, choose wisely based on your specific needs." cos=0.17; "Now, when it comes to which method beats the other..." cos=0.27). Equivalent KH sentences that echo chunk terminology directly score 0.71–0.73. Same grounded label, cosine gap up to 0.56. Driver is lexical alignment, not length or containment.

**Verdict.**

The groundedness floor measures lexical echo, not containment. Torvalds is not less grounded; he is less source-aligned. The scorer rewards sentences that restate chunk vocabulary and penalizes sentences that synthesize or reframe in the author's own voice, even when Opus marks both as grounded. Neither H2 nor H3 describes the actual problem; both were constructed on the premise that the scorer gap reflects a containment gap. That premise did not survive the probe. The metric is the defect.

---

**Investigation CLOSED with verdict: groundedness scorer is lexically confounded (cosine rewards source-echo over synthesis). Confirmed three ways. No fix started, floor untouched, no ADR written. Forward plan to be authored next session.**
