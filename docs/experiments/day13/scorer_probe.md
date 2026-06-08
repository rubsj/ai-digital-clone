# Scorer Probe — Day 13

**Date:** 2026-06-02  
**Data:** results/archive/evaluation_day12_reeval2.json + docs/experiments/day13/markup_output.json  
**Scope:** read and measure only. No code changed.

---

## Part 1 — Scorer Identity

**Function:** `score_groundedness(response, chunks, top_k=5)` at `src/evaluation/groundedness_scorer.py:41`

**Algorithm:** split response into sentences via `_split_sentences()` (regex `(?<=[.!?])\s+`, min 10 chars, line 27-30); batch-embed all sentences in one `embed_openai()` call (text-embedding-3-small via LiteLLM, 1536-d, L2-normalized, MD5-keyed npz cache at `data/cache/embeddings_openai.npz`); reuse chunk embeddings already on `rr.chunk.embedding`, re-embed any missing in a second batch call; per sentence, take max cosine against top-k chunk vectors; return mean of per-sentence maxima.

**Wiring to reeval2:**  
`src/eval/harness.py::run_leader_pair()` → `DigitalCloneFlow.kickoff()` → `ScoringEngine.score()` at `src/components/scoring_engine.py:65` → `score_groundedness(response, chunks)` → result stored as `ev.groundedness_score` → extracted by `_extract_leader_record()` at `src/eval/harness.py:114` → written to the `groundedness_score` field in each leader record.

This is the single wired-in function. No lookalike or alternate path produces `groundedness_score` in reeval2.

**Scorer identity confirmed: yes.**

---

## Part 2 — Probes

### Probe A: Verbosity Correlation

Hypothesis under test: longer responses score higher independent of content.

Per-response character count (chars) and sentence count (sents, same splitter as scorer) correlated against `groundedness_score`. n=42 per leader, n=84 pooled.

**Descriptive stats:**

| Leader | mean chars | min | max | mean sents | mean gs |
|--------|-----------|-----|-----|-----------|---------|
| Torvalds | 1059 | 794 | 1419 | 8.5 | 0.580 |
| KH | 1075 | 675 | 1418 | 8.0 | 0.622 |

**Correlations (Pearson r):**

| | chars vs gs | | sents vs gs | |
|---|---|---|---|---|
| | r | p | r | p |
| Torvalds (n=42) | +0.493 | 0.0009 | -0.182 | 0.248 |
| KH (n=42) | +0.330 | 0.033 | -0.346 | 0.025 |
| Pooled (n=84) | +0.394 | 0.0002 | -0.330 | 0.002 |

Chars vs gs is positive and significant at both the per-leader and pooled level. Sents vs gs is negative and significant pooled: more sentences without more total chars lowers the mean (each additional sentence contributes another per-sentence-max draw to the average, which can dilute it if the sentence is less vocabulary-aligned with the chunks).

**Hypothesis: CONFIRMED.** Longer character counts predict higher groundedness scores, p<0.001 pooled.

---

### Probe B: Containment-Held-Equal

Hypothesis under test: when both clones have near-equal grounded fraction (within 0.05 by Opus markup), the scorer still separates them and favors the more verbose or vocabulary-aligned one.

7 of 14 queries meet the near-equal threshold. In all 7, the scorer direction is KH > T.

| QID | GF_T | GF_KH | GF_gap | scorer_T | scorer_KH | scorer_gap | T_chars | KH_chars | chars_gap |
|-----|------|-------|--------|----------|-----------|------------|---------|---------|-----------|
| q01 | 0.812 | 0.806 | -0.007 | 0.522 | 0.583 | +0.061 | 1036 | 1170 | +134 |
| q02 | 1.000 | 1.000 | 0.000 | 0.660 | 0.724 | +0.065 | 1173 | 1120 | **-53** |
| q04 | 0.815 | 0.857 | +0.042 | 0.581 | 0.612 | +0.031 | 971 | 999 | +28 |
| q05 | 0.352 | 0.333 | -0.019 | 0.518 | 0.560 | +0.042 | 1147 | 1261 | +114 |
| q06 | 0.963 | 0.944 | -0.019 | 0.613 | 0.697 | +0.084 | 1234 | 1185 | **-49** |
| q12 | 0.607 | 0.653 | +0.046 | 0.553 | 0.566 | +0.013 | 855 | 860 | +4 |
| q13 | 0.863 | 0.861 | -0.002 | 0.575 | 0.607 | +0.032 | 1023 | 946 | **-77** |

GF = grounded fraction by Opus span markup (mean across 3 passes, count-weighted).  
Chars = mean response chars across 3 passes.

Scorer gap when containment is near-equal: min +0.013, max +0.084, direction KH > T on 7 of 7.

In 3 of the 7 (q02, q06, q13), Torvalds responses are actually longer in characters yet score lower. The char-gap direction is inconsistent across the 7 cases, so raw response length does not explain the direction of the scorer gap.

**Hypothesis: CONFIRMED.** The scorer separates clones on all 7 near-equal-containment queries. The confound is not raw length alone; it is present even when T responses are longer than KH responses.

---

### Probe C: Per-Sentence Cosine, Grounded Spans

Hypothesis under test: among equally-supported sentences, longer or source-echoing sentences score higher than terse ones.

3 queries used: q02 (both clones fully grounded by markup), q06 and q13 (near-equal containment). Pass 1 per query. Embeddings retrieved from cache; all 5 chunk embeddings found for all three queries. Per-sentence max-cosine computed against all 5 chunks.

**q02** — "What is actually happening when an SVM uses a kernel function..."

| Clone | Leader | scorer_gs | span_chars | per-sent-cos | span text (truncated) |
|-------|--------|-----------|-----------|-------------|----------------------|
| A | torvalds | 0.654 | 118c | 0.674 | "This transformation allows the SVM to find a linear decision boundary..." |
| A | torvalds | 0.654 | 170c | **0.544** | "The beauty of this approach is that it computes the dot products in this..." |
| B | kroah_hartman | 0.740 | 163c | 0.845 | "This flexibility means that the SVM can handle non-linearly separable da..." |
| B | kroah_hartman | 0.740 | 290c | 0.816 | "When an SVM uses a kernel function instead of explicitly mapping data in..." |

Both clones are fully grounded. Torvalds' 170c explanatory sentence ("The beauty of this approach...") scores 0.544. KH's 163c and 290c sentences that echo chunk vocabulary directly score 0.845 and 0.816. Scorer gap here: +0.086 to KH.

**q06** — "What is the difference between forward selection and backward selection..."

| Clone | Leader | scorer_gs | span_chars | per-sent-cos | span text (truncated) |
|-------|--------|-----------|-----------|-------------|----------------------|
| B | torvalds | 0.585 | 85c | **0.272** | "Now, when it comes to which method beats the other, it really depends on..." |
| B | torvalds | 0.585 | 172c | 0.611 | "This way, you can start with a full model and then refine it by adding o..." |
| A | kroah_hartman | 0.678 | 84c | 0.710 | "In practice, combining both forward and backward selection can yield bet..." |
| A | kroah_hartman | 0.678 | 120c | 0.710 | "Forward selection begins with no variables in the model and adds them on..." |

Torvalds' 85c transition sentence ("Now, when it comes to which method beats the other...") scores 0.272 — a meta/editorial sentence that doesn't use chunk vocabulary. KH's 84c sentence directly names the concept and scores 0.710.

**q13** — "What is the difference between a stack and a queue..."

| Clone | Leader | scorer_gs | span_chars | per-sent-cos | span text (truncated) |
|-------|--------|-----------|-----------|-------------|----------------------|
| A | torvalds | 0.537 | 47c | **0.166** | "So, choose wisely based on your specific needs." |
| A | torvalds | 0.537 | 152c | 0.838 | "In terms of applications, both stacks and queues are fundamental in vari..." |
| B | kroah_hartman | 0.611 | 88c | 0.731 | "Stacks and queues are fundamental data structures with distinct behavior..." |
| B | kroah_hartman | 0.611 | 155c | 0.681 | "The choice between using a stack or a queue typically depends on the spe..." |

Torvalds' 47c closing sentence ("So, choose wisely based on your specific needs.") — marked grounded by Opus — scores 0.166. It pulls the mean down substantially. KH has no such sentences in its grounded span set.

**Hypothesis: CONFIRMED.** Among grounded spans, Torvalds' style produces editorial and meta sentences (transition phrases, closing remarks, "it depends" framings) that are marked grounded by Opus but have very low chunk-vocabulary cosine scores (0.17–0.27 observed). KH responses lack this pattern; their short sentences tend to use direct domain terminology. The mechanism is vocabulary alignment, not raw sentence length — in q02 and q06, Torvalds writes longer sentences that still score lower because they explain in different vocabulary rather than echoing chunk text.

---

## Summary

| Probe | Hypothesis | Result |
|-------|-----------|--------|
| A: verbosity (chars vs gs) | longer responses score higher | CONFIRMED (r=+0.394, p<0.001 pooled) |
| A: verbosity (sents vs gs) | more sentences score lower | CONFIRMED (r=-0.330, p=0.002 pooled) |
| B: containment held equal | scorer gaps clones when containment is equal | CONFIRMED (7/7 near-equal queries, scorer gap +0.013 to +0.084, all KH > T) |
| B: verbosity direction | scorer gap follows char length | NOT CONFIRMED (3 of 7 near-equal cases T is longer yet scores lower) |
| C: per-sentence mechanism | terse/meta sentences score lower than vocabulary-echoing ones | CONFIRMED (grounded Torvalds meta-sentences: cos 0.17–0.27; equivalent KH sentences: 0.71–0.79) |

**The scorer is style-confounded.** The confound is not raw verbosity but chunk-vocabulary alignment. Torvalds' writing style produces grounded sentences that explain in different vocabulary (transition phrases, "the beauty of this approach," "now, when it comes to which method") while KH sentences tend to use chunk terminology directly ("stacks and queues are fundamental data structures," "forward selection begins with no variables"). Both Opus-marked as grounded; the scorer separates them by 0.17–0.54 per sentence.

**Scorer identity confirmed: yes. Confound evidence: chars-vs-gs r=+0.394 pooled (p<0.001); scorer gaps near-equal-containment queries on 7/7 (max gap +0.084); Torvalds style-characteristic meta sentences score 0.17–0.27 cosine despite grounded markup. No code changed.**
