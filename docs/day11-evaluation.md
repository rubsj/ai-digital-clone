# Day 11 Evaluation Report

**Authored:** Day 12 (per PRD §7.5.1 naming convention)
**Branch:** refactor/p6-multi-agent-rework
**Measurement harness:** `src/eval/harness.py`, `run_measurement()`
**Raw results:** `results/archive/evaluation_day12.json` (48 pair-records, 96 leader-records)
**Run design:** Pass 1 all 20 queries × 2 leaders = 40 pair-records. Passes 2-3 14 in-domain queries × 2 leaders = 28 pair-records each. No reactive OOD recheck triggered (0 OOD delivers in pass 1).

---

## 2x2 Routing Grid

In-domain axis: 14 queries from `statistical_learning_ml`, `data_mining`, `numerical_methods`, `programming_fundamentals`. OOD axis: 6 queries from `systems_absent_from_corpus` and `off_topic_technical`. Grid derived from `category` field via `classify_category()`.

### Torvalds (pass 1)

|                    | Predicted deliver | Predicted fallback |
|--------------------|:-----------------:|:-----------------:|
| **In-domain** (14) | 0                 | 14                |
| **OOD** (6)        | 0                 | 6                 |

### Kroah-Hartman (pass 1)

|                    | Predicted deliver | Predicted fallback |
|--------------------|:-----------------:|:-----------------:|
| **In-domain** (14) | 0                 | 14                |
| **OOD** (6)        | 0                 | 6                 |

---

## Per-Leader Deliver and Fallback Rates

### In-domain deliver rate (three runs)

| Pass | Torvalds | Kroah-Hartman |
|------|----------|---------------|
| 1    | 0/14 = 0.0% | 0/14 = 0.0% |
| 2    | 0/14 = 0.0% | 0/14 = 0.0% |
| 3    | 0/14 = 0.0% | 0/14 = 0.0% |

Three-run variance: zero (rate is 0% in every pass).

ADR-015 floors: Torvalds 42.9% (6/14), Kroah-Hartman 35.7% (5/14). Both leaders miss the floor by the full margin. ADR-015 E2 target (55%) is also not reached.

### OOD fallback rate

| Leader        | Pass 1  |
|---------------|---------|
| Torvalds      | 6/6 = 100% |
| Kroah-Hartman | 6/6 = 100% |

OOD fallback is 100% for both leaders. Gate clause (b) passes.

### Category-5 hallucination check

OOD queries span both `systems_absent_from_corpus` and `off_topic_technical`. Zero OOD delivers across all three passes. Hallucination count on category-5 OOD: 0. Gate clause (c) passes.

### ADR-013 per-leader delta

Torvalds 0.0%, Kroah-Hartman 0.0%, delta = 0.0pp. ADR-013 contingency (fires if delta > 20pp) does not apply.

---

## Score Summary (pass 1, in-domain)

| Metric               | Torvalds mean | KH mean |
|----------------------|:-------------:|:-------:|
| style_score          | 0.877         | 0.818   |
| groundedness_score   | 0.575         | 0.621   |

Day-8 v2 baseline (pre-rework, ADR-010): style mean Torvalds 0.9025 / KH 0.8355, groundedness 0.6258 pooled. Day-12 v2 (post-rework) style is similar; groundedness is slightly below baseline for Torvalds, above for KH.

---

## Regression Anchor Check

q12 (binary search, `programming_fundamentals`) and q13 (stack vs queue, `programming_fundamentals`) are designated regression anchors (`regression_anchor: true` in queries.json).

| Query | Leader | Pass 1 | Pass 2 | Pass 3 | gs range |
|-------|--------|--------|--------|--------|----------|
| q12   | Torvalds | fallback | fallback | fallback | 0.470-0.558 |
| q12   | KH | fallback | fallback | fallback | 0.515-0.571 |
| q13   | Torvalds | fallback | fallback | fallback | 0.524-0.576 |
| q13   | KH | fallback | fallback | fallback | 0.614-0.670 |

Both anchors fallback consistently across all three passes. q13 KH groundedness (0.614-0.670) is above the 0.60 target but the EvaluatorAgent still flags `low_groundedness`.

---

## trigger_category Distribution

All 40 in-domain fallback records in pass 1 carry trigger_category `low_groundedness`. No diversity across categories.

Integrity assertion (non-null iff fallback, valid literal): one soft violation. q14 Torvalds was assigned trigger_category `low_groundedness` by the GatekeeperAgent but the EvaluatorAgent flags were `['low_confidence', 'low_style']` with no `low_groundedness` flag raised. The structural rule (non-null iff fallback) passes; the content accuracy of the label does not for this one record.

---

## In-Domain Fallback Analysis

All 14 in-domain queries are classified by the cause of their fallback. Analysis is based on `clone_response_text` (the CloneAgent output scored by the EvaluatorAgent) compared against `chunk_contents` from `results/archive/evaluation_day12.json`. No additional LLM calls used.

### Failure mode distribution (pass 1, Torvalds)

**q01-pattern (grounded core + generalization tail): 7 queries**

The CloneAgent response covers the core concept accurately and cites chunk content, then adds 1-2 sentences of contextual guidance ("when would you use this") that go beyond the retrieved text. Retrieval is strong (rank-0 score > 0.75 in all cases).

- q01 (L2 vs L1): core is grounded (L1/L2 mechanism, sparsity). Tail adds "high-dimensional data where features are irrelevant" and "interpretability" — neither in chunks.
- q02 (SVM kernel): core grounded (mapping to higher-dim, linear separability). Response adds "kernel trick" and "dot product between pairs without explicit mapping" — the mechanism is not in the chunks, only the effect.
- q03 (bias-variance + hidden layers): chunks cover general bias-variance seesaw. Response maps this specifically to adding hidden layers — not in chunks.
- q04 (cross-validation): chunks cover train-test split and k-fold. Response adds "not memorizing training data" framing — beyond what chunks state.
- q06 (forward vs backward selection): chunks cover backward selection well. Response describes forward selection, which is not in the retrieved chunks.
- q08 (collaborative filtering): chunks cover item-based CF. Response describes user-based CF from model knowledge. gs=0.690 for Torvalds — the highest in-domain score; response is largely grounded.
- q12 (binary search): core grounded in chunks. Response adds O(log n) time complexity — not stated in chunks.

**Thin or absent retrieval: 6 queries**

The rank-0 chunk either scores low or does not contain explanatory content. The CloneAgent response generates from model knowledge.

- q05 (curse of dimensionality): rank-0 chunk is an exercise question asking to explain the concept, not an explanation. Ranks 1-2 score 0.002. Response generates entirely from training knowledge.
- q07 (confusion matrix): all three retrieved chunks are identical ("a table that summarizes the performance..."). No TP/TN/FP/FN content in chunks. Response generates from training knowledge.
- q09 (trapezoidal vs Simpson's rule): all three chunks are identical ("both rules can be used with high accuracy, require many evaluations"). No "when to use each" content in chunks.
- q10 (fixed-point iteration): all three chunks are identical and show only the formula. No convergence analysis in chunks.
- q11 (sparse matrix + Gaussian elimination): most relevant chunk (sparse matrix section) reranked last at score 0.044. Gauss-Seidel chunk ranked first at 0.691 but addresses a different method.
- q14 (recursion vs iteration): all retrieval scores below 0.33. Response generates from training knowledge. gs_T=0.658 because the limited chunk content is directionally consistent with the response.

**EvaluatorAgent over-flagging: 1 query**

- q13 (stack vs queue): response closely follows chunk content. Chunks cover LIFO/FIFO definitions, memory management use case, real-time systems — and the response mirrors all three. gs_T=0.524 is lower than the content similarity warrants. This is the clearest case of the scorer assigning a low groundedness score to a well-grounded response.

---

## Root Cause Analysis

### Root cause 1 — EvaluatorAgent flag threshold calibration

The EvaluatorAgent prompt states the groundedness target as 0.60 and instructs the LLM to flag any dimension "below its target." However, the LLM is raising `low_groundedness` for scores well above 0.60: q08 Torvalds gs=0.690, q08 KH gs=0.706, q02 Torvalds gs=0.674. The effective flag threshold is approximately 0.70-0.75 based on the observed outputs, not 0.60 as specified. This is an LLM calibration error in the flag-raising step.

### Root cause 2 — GatekeeperAgent routes on flag presence, not score

The GatekeeperAgent prompt says "Default: DELIVER. Route to FALLBACK only when a specific score or flag justifies it." In practice the Gatekeeper routes to fallback whenever `low_groundedness` appears in the flags, regardless of the numerical score. Three data points confirm this:

- q08 KH: gs=0.706, flag=`low_groundedness` → fallback. Gatekeeper reasoning: "groundedness score is 0.706, which indicates that the response lacks sufficient citations" — the Gatekeeper agrees the score is not below threshold but still routes to fallback.
- q11 KH: gs=0.651, flag=`low_groundedness` → fallback. Gatekeeper reasoning: "groundedness score of 0.651, which is below the acceptable threshold of 0.60" — factual arithmetic error (0.651 > 0.60).
- q06 Torvalds: gs=0.609, flag=`low_groundedness` → fallback. Gatekeeper notes "below the acceptable threshold" even though 0.609 > 0.60.

The Gatekeeper is not reading the actual score against the threshold; it is following the presence of the flag. When the EvaluatorAgent incorrectly raises the flag, the Gatekeeper faithfully routes to fallback.

### Interaction

The two root causes compound. For strong-retrieval queries where the CloneAgent mostly stays within the chunks (q02 gs=0.674, q08 gs=0.706), the correct decision would be deliver. Instead: EvaluatorAgent over-flags → Gatekeeper follows the flag → fallback. For thin-retrieval queries (q05, q07, q09, q10, q11, q14), fallback may be the correct routing regardless of the scoring issue.

---

## Comparison to Day-8 Baseline

Day-8 v2 pipeline: in-domain deliver 11/28 pooled (39.3%), Torvalds 6/14 (42.9%), KH 5/14 (35.7%). OOD 12/12 fallback. The Day-8 pipeline used a scoring formula with a 0.75 threshold; GatekeeperAgent (ADR-010) replaced that formula. Day-12 deliver rate is 0/14 for both leaders, representing a significant regression from Day-8. This is not attributable to a worse corpus or worse CloneAgent output — it is attributable to the GatekeeperAgent being more conservative than the formula it replaced, compounded by EvaluatorAgent flag miscalibration.

---

## Gate Decision

Gate is evaluated against the amended ADR-015 geometry (see ADR-015 amendment, Day 12).

Gate clause (a) — in-domain deliver rate at or above floors: **FAIL.** Torvalds 0.0% vs floor 42.9%. KH 0.0% vs floor 35.7%.

Gate clause (b) — OOD fallback 100% both leaders: **PASS.**

Gate clause (c) — zero hallucinations on category-5 OOD: **PASS.**

**Gate decision: NO-SHIP.** Phase 2 (cli.py/visualization.py refactor, v1 retirement) does not proceed until in-domain deliver rate is investigated and a fix is measured. The investigation target is the two-root-cause chain: EvaluatorAgent flag threshold calibration and GatekeeperAgent routing logic.
