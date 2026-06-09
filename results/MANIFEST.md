# Results Manifest

Every file in `results/` (top level and `results/archive/`) is accounted for here.
The ADRs cite evidence in prose, not by file path — this manifest is the sole traceability
anchor between the shipped-ADR evidence trail and the result files on disk.

---

## Current-truth files (`results/` top level)

These files remain at the top level alongside the forthcoming `evaluation_day15.json`.
None are superseded by the P2 canonical run; each answers a distinct question or isolates
a distinct variable.

### `bakeoff_hhem_isolated_day14.json`

**Question answered:** What are HHEM-2.1-Open's per-query entailment scores for the 14
in-domain queries, and where does it rank on oracle alignment vs. the other candidate models?

**Cited by:** ADR-020 §HHEM column (the selected model); W1b.2 threshold derivation uses
these scores as input; W3a (`w3a_metric_effect_day14.json`) reuses these per-query HHEM
means directly — no re-scoring was needed.

---

### `w1b2_threshold_day14.json`

**Question answered:** Where does `GROUNDEDNESS_MIN = 0.40` sit on HHEM-2.1-Open's
entailment scale — what train/heldout sweep analysis yields this threshold?

**Cited by:** ADR-020 threshold derivation; ADR-021 regime-dependence discussion.

---

### `w3a_metric_effect_day14.json`

**Question answered:** What is the metric effect alone — how do deliver/fallback verdicts
change when HHEM@0.40 replaces cosine@0.60 on the *same frozen (response, chunks)* records
from `results/archive/evaluation_day12_reeval2.json`?

**Beside, not over:** This file ran the scorer(s) on frozen inputs to isolate one variable;
it answers a different question than the live P2 run (`evaluation_day15.json`). The P2 run
does not supersede it.

**Coherence note:** The README quotes the fresh P2 distribution; the ADRs keep W3a's numbers
(64.3% in-domain deliver / 78.6% weighted-oracle match); the divergence between them is
expected temp=0.3 variance, not a contradiction.

**Cited by:** ADR-015 amendment (64.3% / 78.6%); ADR-021.

---

### `w3b_retrieval_effect_day14.json`

**Question answered:** What is the retrieval (dedup) effect alone — how do deliver/fallback
verdicts change when the W2 dedup fix is live in the Retriever, with HHEM@0.40 held fixed
throughout?

**Beside, not over:** This file ran the fixed Retriever on fresh passes while holding the
metric constant; it answers a different question than the live P2 run (`evaluation_day15.json`).
The P2 run does not supersede it.

**Coherence note:** The README quotes the fresh P2 distribution; the ADRs keep W3b's
per-query numbers; the divergence is expected temp=0.3 variance, not a contradiction.

**Cited by:** ADR-002 amendment; ADR-015 amendment. Note on q07 Torvalds: W3b records
HHEM 0.335, which is **below** the 0.40 gate — q07 T falls back after the dedup fix
despite being grounded (oracle 53.6%). This is the accepted paraphrase-misroute limitation:
a grounded response that routes to fallback due to paraphrase mismatch, not a confirmed
deliver. Documented as a known limitation in ADR-015.

---

## Audit-trail files (`results/archive/`)

The ADRs cite numbers in prose, not by file path — moving these files to `archive/` breaks
no ADR markdown link. Removing them would orphan the evidence behind shipped ADR claims.
None are deleted.

### `archive/bakeoff_w1b0_day14.json`

**Phase / ADR:** Day 14 W1b.0; ADR-020 G1–G4 bake-off table; ADR-021 alternatives.

**Evidence it is:** Head-to-head bake-off grid comparing DeBERTa-v3-NLI and MiniCheck
flan-t5-large against the ANLI oracle across 8 in-domain queries — the evidence that
neither matched the oracle as well as HHEM-2.1-Open.

---

### `archive/bakeoff_hhem_probe_day14.json`

**Phase / ADR:** Day 14 W1b.1 Probe A; ADR-020 Probe A paragraph; ADR-021 Door A.

**Evidence it is:** Aggregation-variant experiment (min/max/mean/k-th chunk) testing whether
changing the aggregation strategy removed the Door C bias — the evidence that no variant
cleared both G1 and G2, routing the decision to Door C.

---

### `archive/evaluation_day12_reeval2.json`

**Phase / ADR:** Day 12 Phase 1.6.5; W3a frozen input.

**Evidence it is:** The 84 frozen in-domain records (14 queries × 2 leaders × 3 passes)
whose stored cosine scores W3a re-applied HHEM@0.40 to. W3a's per-query verdicts are
uninterpretable without this frozen-input baseline.

---

### `archive/evaluation_day12.json`

**Phase / ADR:** Day 12 Phase 1 (NO-SHIP baseline); cited in `docs/evaluation-methodology.md`
and `docs/day11-evaluation.md`.

**Evidence it is:** The 48-pair-record Phase 1 run under cosine@0.60 (42 in-domain + 6 OOD
across three passes) — the foundational measurement establishing RC-1 (deterministic flag
threshold) and RC-2 (GatekeeperAgent routing). Source for the root-cause analysis in
`docs/day11-evaluation.md`.

---

### `archive/evaluation_day12_reeval.json`

**Phase / ADR:** Day 12 Phase 1.5b (after STYLE_MIN 0.90→0.70, before RC-1 fix); cited in
`docs/session-notes/day12.md`.

**Evidence it is:** 42-pair-record intermediate re-eval (0/14 Torvalds, 1/14 KH) after the
STYLE_MIN recalibration — the evidence that RC-2 remained the blocking issue independent of
STYLE_MIN.

---

## Worthless files (`results/archive/`)

These files carry no current question and have no live citations. Archived per plan default
(archive, not delete). Deletion requires explicit approval.

### `archive/evaluation_20260523_121048.json`

**Why worthless:** Day 7 snapshot (committed 2026-05-25) with schema `id/leader/fallback/
latency_ms` only — no groundedness, style, or confidence scores. 19/20 fallback. Predates
the scoring schema entirely; answers no current question. Zero live citations outside
`docs/plans/day15-plan.md` (which names it only as a classification candidate).

Zero-citation grep: `grep -r "evaluation_20260523" docs/ src/ tests/` → one hit,
`docs/plans/day15-plan.md` candidate list only.

---

## P3a — Chart set (Day 15, complete)

9 charts in `results/charts/`, all §7.6-named. 8 match the §7.6 inventory; 1 is a §7.6 addition.

| File | §7.6 slot | Source | Notes |
|------|-----------|--------|-------|
| `charts/01-style-radar-dual-leader.png` | #1 | Style profiles | Both leaders overlaid, 15 features |
| `charts/02-routing-correctness-grid.png` | #2 | eval_day15 pass 1 | New chart (was missing); 32/40 correct |
| `charts/03-style-score-distribution.png` | #3 | eval_day15 all passes | Per-leader overlaid histograms |
| `charts/04-groundedness-score-distribution.png` | #4 | eval_day15 all passes | HHEM entailment, 0.40 gate marked |
| `charts/05-deliver-rate-distribution.png` | #5 | eval_day15 passes 1–3 | §7.6 slot renamed from "score-component-breakdown" |
| `charts/06-fallback-trigger-distribution.png` | #6 | eval_day15 all passes | trigger_reason categories |
| `charts/07-latency-distribution.png` | #7 | eval_day15 all passes | Deliver vs fallback path separated |
| `charts/08-torvalds-style-evolution-pre-post-2018.png` | #8 | torvalds.mbox | 4-panel monthly time-series, Sept 2018 marker |
| `charts/09-retrieval-relevance-contrast.png` | — | eval_day15 pass 1 | §7.6 addition; in-domain vs OOD top-chunk score |

Stale-named PNGs (01-07 with old names) removed. §7.6 #5 slot renamed (flagged in session notes).
