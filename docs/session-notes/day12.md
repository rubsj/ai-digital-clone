# Day 12 Session Notes

Plan: `docs/plans/day12-plan.md`. Scope: end-to-end evaluation gate (Phase 1), deterministic flag-raising fix (Phase 1.5), v1 retirement (Phase 2 — gated, not started). Branch: `feat/evaluator-deterministic-flags` off `main` (previously `refactor/p6-multi-agent-rework`, merged as PR #15 Day 11). Two commits on the branch; Phase 2 remains blocked.

## Plan amendments made during execution

One pre-execution amendment and one mid-phase amendment:

1. **1.5.2 stop-branch clarified.** The original 1.5.2 prose said "STOP and surface" if `_parse_review` was meaningfully rewording. Corrected before execution: both branches (parse removed OR parse retained) continue to 1.5.3. The re-eval always runs; what varies is only whether the parse call was removed. The flag fix (1.5.1) is the isolated variable; the parse removal is conditional and only shifts latency.

2. **STYLE_MIN recalibration (ADR-017 Amendment 1).** After STOP GATE 1.5a, a pre-re-eval diagnostic showed that STYLE_MIN=0.90 would block 68% of in-domain records regardless of the groundedness fix, because 0.90 is an undocumented synthetic-data calibration target (ADR-003), not a validated per-response delivery threshold. STYLE_MIN was corrected to 0.70 (ADR-003 full-corpus self-similarity benchmark) before the re-eval. The amendment is recorded in ADR-017 Amendment 1 and changes the STYLE_MIN constant only; CONFIDENCE_MIN and GROUNDEDNESS_MIN are unchanged.

---

## Phase 1 — Measurement

### Harness (`src/eval/harness.py`)

Created `src/eval/harness.py` with two public functions:

`run_leader_pair(query) -> dict`: mirrors the ADR-005 shared-retrieval pattern from `compare_leaders()` but holds both `DigitalCloneFlow` instances to capture `routing_decision`, evaluation scores, output type, `clone_response_text`, `delivered_text`, `chunk_contents`, and every timing key. Asserts exactly one `Retriever.run()` call across the pair using a `patch.object` counter — the same gate as `tests/integration/test_compare_leaders.py`. Torvalds flow retrieves; Kroah-Hartman flow receives `chunks=shared_chunks` and early-exits the retrieve step.

`run_measurement(path, output)`: implements the C4 run design — one full pass over all 20 queries (40 records), then two in-domain-only re-runs (28 records each), then a reactive OOD recheck if any OOD record delivered in pass 1. Writes incrementally after every pair.

Captured per record: `decision`, `trigger_category`, `trigger_reason`, `routing_reasoning`, `clone_response_text`, `delivered_text`, `output_type`, `style_score`, `groundedness_score`, `confidence_score`, `flags`, `chunk_contents` (rank, score, source_topic, content for each chunk), and all timing keys (`retrieve_ms`, `clone_generate_ms`/`clone_parse_ms`, `evaluate_score_ms`/`evaluate_generate_ms`/`evaluate_parse_ms`, `route_generate_ms`/`route_parse_ms`, `deliver_ms` or `fallback_generate_ms`/`fallback_parse_ms`, `pair_elapsed_s`).

### Phase 1 run results

Full run (20 queries × 2 leaders) plus 2 in-domain re-runs = 96 records written to `results/archive/evaluation_day12.json`.

**In-domain 2x2 grid (pass-1 full run, per leader):**

| | Torvalds | Kroah-Hartman |
|---|---|---|
| In-domain deliver | 0/14 | 0/14 |
| In-domain fallback | 14/14 | 14/14 |
| OOD fallback | 6/6 | 6/6 |
| OOD deliver | 0/6 | 0/6 |

**Three-run in-domain deliver rate:** Torvalds 0–0/14 (0%), Kroah-Hartman 0–0/14 (0%). Zero variance across all three passes — deterministic behavior, not a sampling artifact.

**OOD gate:** 6/6 fallback both leaders all passes. Zero hallucinations. This clause passes; the reactive OOD recheck did not fire.

**ADR-013 style trigger:** Both leaders at 0/14; the 20-point gap criterion between leaders did not apply.

**`trigger_category` integrity assertion:** PASS. All fallback records have a non-null `trigger_category` from the valid five-literal set. All deliver records have `trigger_category=None`. No sixth literal appeared.

**`trigger_category` distribution (pass-1, 28 fallback records):**

| trigger_category | count |
|---|---|
| low_groundedness | 27 |
| low_confidence | 1 |

**Latency (per-pair):** ~54s average. Dominant costs: CloneAgent generate (~6s), EvaluatorAgent generate (~8s) and score (embed_openai, ~2s), GatekeeperAgent generate (~4s), FallbackAgent generate (~6s), EvaluatorAgent `_parse_review` (~3s sequential second LLM call). All pairs exceeded the PRD §2.7 8s SLA. Logged as Day-13 latency item; not a gate clause.

### Phase 1 documents

- `docs/day11-evaluation.md` — 2x2 grid, per-leader deliver/fallback, three-run variance, Day-8 regression-anchor check (q12 and q13 confirmed in-domain, both fell back — not regression from Day 8, the pipeline mechanism changed), PRD §2 scorecard, gate decision citing ADR-015 and the Amendment 1 floor correction.
- `docs/evaluation-methodology.md` — three-layer architecture (Layer 1 unit continuous, Layer 2 per-agent recorded-LLM contract tests, Layer 3 system via query set), run design, category classification, one-retrieval invariant, ungrounded span analysis, groundedness scoring architecture.
- `docs/adr/ADR-015-post-rework-eval-acceptance-criteria.md` — Amendment appended (Day 12): corrects E1 floor from pooled 39% to honest per-leader baselines (Torvalds ≥42.9% = 6/14, Kroah-Hartman ≥35.7% = 5/14); sub-floor handling documents the two root causes.

### STOP GATE 1 result

**NO-SHIP.** In-domain deliver rate 0/14 both leaders, three-pass zero variance (deterministic, not a sampling artifact). OOD 6/6 fallback both leaders, 0 hallucinations. Conjunctive gate clause (a) fails below both per-leader floors (Torvalds 42.9%, Kroah-Hartman 35.7%). Investigation branch taken. Root cause and fix: ADR-017 (Deterministic Flag-Raising).

---

## Phase 1.5 — Investigation and fix (ADR-017)

### Investigation step 1 — read-only code inspection

Read `src/agents/evaluator_agent.py` and `src/agents/gatekeeper_agent.py` to locate where flags are raised.

**Finding (EvaluatorAgent — RC-1):** No numerical threshold comparison anywhere in the code. The three thresholds (style > 0.90, groundedness > 0.60, confidence > 0.80) exist only as f-string literals inside `_build_task_description()`. The CrewAI kickoff LLM writes a prose verdict. `_parse_review()` (a second `instructor.from_litellm` call) extracts flag labels from that prose without re-reading any number. The EvaluatorAgent backstory ("be direct, name concrete problems rather than hedging") biases the reviewer toward raising flags. Measured consequence: 13 of 28 in-domain records scored at or above the 0.60 groundedness target but still carried `low_groundedness`, highest at 0.706 (q08 Kroah-Hartman).

**Finding (GatekeeperAgent — RC-2 hypothesis):** No numerical threshold comparison. Backstory says "default to deliver when no flags are raised," whose contrapositive governs in practice. The Gatekeeper sees the flags list and scores but is never instructed to compare numerically against 0.60. Reasoning strings in the Phase-1 records confirm this, including a false arithmetic claim on q11 Kroah-Hartman: "gs=0.651 is below the acceptable threshold of 0.60."

### ADR-017

Authored `docs/adr/ADR-017-deterministic-flag-raising.md` (Status: Accepted, 2026-06-01). Decision: move flag-raising from LLM judgment into deterministic code in EvaluatorAgent. Each flag raised if and only if the corresponding ScoringEngine score is below its named-constant threshold. GatekeeperAgent deliberately not changed; RC-2 reads off the re-evaluation.

### Step 1.5.1 — Deterministic flag-raising

Changes to `src/agents/evaluator_agent.py`:

- Added three named constants: `GROUNDEDNESS_MIN: float = 0.60` (line 37), `STYLE_MIN: float = 0.90` (line 42, later corrected to 0.70 per Amendment 1), `CONFIDENCE_MIN: float = 0.80` (line 43).
- Added `_compute_flags(scores: Scores) -> list[str]`: three arithmetic comparisons, one per threshold.
- Removed `flags: list[str]` field from `_ReviewDraft` (dead code once flags come from code).
- Removed `Field` from pydantic import.
- Updated `_build_goal()`, `_build_task_description()`, and `expected_output` to remove the instruction for the LLM to produce flag labels.
- Updated `_parse_review()` prompt: "Return only the explanation text. Do not include any flag labels or appended content."
- Changed `run()`: `flags=_compute_flags(scores)` instead of `flags=draft.flags`.

**Unit assertion (96 Phase-1 stored records):** PASS — no record with groundedness_score ≥ 0.60 carries `low_groundedness`.

### Step 1.5.2 — Equivalence guard / `_parse_review` disposition

Ran the equivalence guard on q01, q08, q13 (pass-1 Torvalds) using Phase-1 stored `clone_response_text` and `chunk_contents`, calling `_build_crew()` + `kickoff()` + `_parse_review()` directly without full flow execution.

Finding: `_parse_review` is extraction-only (verbatim lift of the quality assessment paragraph from `.raw`), not meaningful rewording. BUT `.raw` is not equivalent to `draft.explanation`: `.raw` includes the "Flags:" section that the first LLM was previously instructed to produce, and on two of three records also appends the full clone response text after a `---` separator. Taking `.raw` directly as the explanation would degrade it.

**Branch taken: parse retained, deferral surfaced.** Removing `_parse_review` tonight would require either text-splitting `.raw` or changing the CrewAI task expected_output to suppress the "Flags:" section — both change more than one variable. `_parse_review` retained for Phase 1.5; removal deferred as a noted Day-13 latency follow-up (one sequential LLM round-trip eliminated, ~3s saved per pair).

### STOP GATE 1.5a

Evidence pasted: `evaluator_agent.py` diff, three named constants with file:line, equivalence-guard comparison (3 records), unit assertion (PASS), GatekeeperAgent `git diff --stat` (no output). Gate approved. Proceeded to pre-re-eval diagnostic before running 84 records.

### Pre-re-eval diagnostic — threshold provenance and score distributions

Before authorizing the 84-record re-eval spend, investigated: did the three thresholds have calibrated bases, and what are the actual score distributions?

**GROUNDEDNESS_MIN = 0.60 — documented and calibrated.** ADR-004 records a 5-sample LLM-judge comparison at the 0.60 agreement level, 4/5 agreements, LKML-specific.

**STYLE_MIN = 0.90 — undocumented literal as a flag threshold.** ADR-003 records that 0.90 was a synthetic-data calibration target explicitly corrected to 0.70 by full-corpus validation. The 0.70 is the profile-validation self-similarity benchmark. The 0.90 survived only as a distribution-mean reference (PRD §122: "style mean 0.80–0.90") and a docstring characterization ("a score ≥ 0.90 indicates strong style match"). PRD §195 marks ">0.90 pass/fail" as informational only. No ADR, no calibration run against generated responses ever validated 0.90 as a per-response delivery threshold. The Day-6 sub-weight sweep (ADR-006) was intended to calibrate weights but failed as a proxy-regime artifact.

**CONFIDENCE_MIN = 0.80 — partially documented literal.** `confidence_scorer.py` docstring cites the PRD quality table ("Target: > 0.80") but that table gives a distribution mean range (0.75–0.90), not a per-response floor. No calibration run exists. The Day-6 sweep failed in the proxy regime (confidence pinned at 1.0 on CS-query inputs instead of the production range).

**Style score distribution (in-domain, pass-1, n=28):** min 0.671, max 0.959, mean 0.848, median 0.835. At or above 0.90: 9 records (32%). Below 0.90: 19 records (68%). The 0.90 threshold would block most of the in-domain set regardless of the groundedness fix.

**Confidence score distribution (in-domain, pass-1, n=28):** min 0.472, max 0.995, mean 0.809, median 0.821. At or above 0.80: 15 records (54%). Below 0.80: 13 records (46%).

### ADR-017 Amendment 1 — STYLE_MIN recalibrated 0.90 → 0.70

Appended to `docs/adr/ADR-017-deterministic-flag-raising.md`. STYLE_MIN changed from 0.90 to 0.70. Basis: 0.70 is ADR-003's validated full-corpus self-similarity benchmark — the cosine proximity the leaders' own LKML emails clear against the style profile. Applied as the response delivery threshold: a generated response should clear at least the style proximity the leader's own corpus clears.

CONFIDENCE_MIN stays at 0.80 (no validated alternative value exists the way ADR-003 hands 0.70 to style; eye-picking a confidence number would repeat the same mistake). GROUNDEDNESS_MIN stays at 0.60 (calibrated, ADR-004).

**Extended unit assertion (96 Phase-1 stored records):** PASS on both assertions — no record with groundedness_score ≥ 0.60 carries `low_groundedness`; no record with style_score ≥ 0.70 carries `low_style`.

### Step 1.5.3 — Re-evaluation (in-domain only)

Wrote `scripts/reeval_indomain.py` (throwaway, not production code) calling `run_leader_pair()` from the existing harness, loading only in-domain queries, 3 passes. OOD excluded per ADR-017 Quantified Validation (OOD groundedness 0.33–0.48, cannot flip on a threshold change; OOD already passed at 6/6 fallback).

84 records written to `results/archive/evaluation_day12_reeval.json` (does not overwrite `evaluation_day12.json`).

**Call count estimate (pre-run):** 42 pairs × ~14 chat completions per pair ≈ 588 completions + ~84 batched embed_openai calls. Within the pre-approved Phase 1.5.3 envelope (plan explicitly waives the cost-guard pre-flight for this scope).

**Run time:** ~165 minutes (42 pairs × ~60–90s each, some pairs with LLM latency spikes reaching ~110s and one outlier at ~688s — network variability, not a code defect).

### Step 1.5.4 — GatekeeperAgent disposition read

Analysis script `scripts/analyze_reeval.py` (throwaway). Results:

**In-domain deliver rate (re-eval, all 3 passes):**

| Leader | Pass 1 | Pass 2 | Pass 3 | Floor |
|---|---|---|---|---|
| Torvalds | 0/14 (0.0%) | 0/14 (0.0%) | 0/14 (0.0%) | 42.9% |
| Kroah-Hartman | 1/14 (7.1%) | 1/14 (7.1%) | 1/14 (7.1%) | 35.7% |

Three-run variance: zero. Identical results all three passes.

**Flag-clean deliver check (RC-2 read):**

Fully flag-clean records (gs ≥ 0.60, ss ≥ 0.70, cs ≥ 0.80, no flags): 25 across all 42 pairs.

- DELIVER: 3 (q02 KH pass-1 gs=0.694, q08 KH pass-2 gs=0.675, q02 KH pass-3 gs=0.716)
- FALLBACK: 22 — **RC-2 confirmed independent**

All 22 flag-clean fallbacks carry `trigger_category=low_groundedness` with Gatekeeper reasoning strings characterizing the score as insufficient. Sample:

- q02 Torvalds gs=0.678, flags=[], tc=low_groundedness: "The groundedness score of 0.678 indicates that the response lacks sufficient direct references to the provided source material..."
- q08 Torvalds gs=0.674, flags=[], tc=low_groundedness: "...a lack of sufficient references to the provided source material..."
- q08 KH gs=0.715, flags=[], tc=low_groundedness: "0.715, which indicates that the response lacks sufficient grounding in the provided source material..."
- q09 KH gs=0.653, flags=[], tc=low_groundedness: "...which is below the target of 0.60..." [arithmetically false: 0.653 > 0.60]

**RC-2 refined characterization (more precise than the ADR-017 hypothesis):** RC-2 is not simply "routes on flag presence." The GatekeeperAgent independently re-evaluates groundedness score with its own LLM judgment and applies an effective internal threshold of approximately 0.70–0.75. It falls back when it judges gs too low regardless of what the EvaluatorAgent flagged. The three delivers (gs 0.694, 0.675, 0.716) suggest the Gatekeeper's effective threshold sits around 0.68–0.72. This is the same class of defect as RC-1 (a number meaningful in one context, producing inconsistent behavior in another), one layer up.

**Low-confidence-only-blocked count:** 10 records across 3 passes (5 unique query-leader pairs: q06 Torvalds, q11 Torvalds, q11 Kroah-Hartman, q14 Kroah-Hartman, q14 Torvalds pass-3). CONFIDENCE_MIN=0.80 is the next binding threshold in code. However, because RC-2 governs routing above the flag check, most of these records would still fall back even if the `low_confidence` flag were removed — the Gatekeeper would judge their groundedness independently.

**RC-3 recurrence:** 34 instances across the re-eval. RC-3 (Gatekeeper assigns `trigger_category=low_groundedness` when no `low_groundedness` flag was raised) recurs on: all 22 flag-clean fallbacks, all 10 low-confidence-only-blocked records, and 2 low-style-only records. RC-3 is not a separate defect from RC-2; it is the same mechanism expressed as a labeling artifact. The Gatekeeper performs its own groundedness evaluation and labels the result `low_groundedness` regardless of what flag list it received. RC-3 is expected to resolve when RC-2 is fixed.

**Trigger category distribution (re-eval, 81 fallback records):** low_groundedness 81 (100%). Only one trigger_category assigned in the entire re-eval.

**Trigger category integrity assertion:** PASS — all fallback records non-null, all deliver records null, all values in valid set.

### STOP GATE 1.5b result

**NO-SHIP. RC-2 confirmed independent.**

Floors not cleared: Torvalds 0/14 (0%), Kroah-Hartman 1/14 (7.1%), both below Day-8 per-leader baselines. Three-run variance: zero. RC-2 is a confirmed independent defect requiring its own GatekeeperAgent fix (numerical score comparison against the stated 0.60 threshold, not independent LLM groundedness judgment) and its own re-eval. Phase 2 remains blocked behind STOP GATE 1.5b + floors clear + RC-2 resolved.

---

## Phase 1.6 — RC-2 fix: deterministic routing + enriched fallback (ADR-018)

STOP GATE 1.5b closed NO-SHIP with RC-2 confirmed independent. Executing 1.6.1–1.6.4 per ADR-018. Stopped at STOP GATE 1.6a; re-eval not yet started.

### ADR-018

Authored `docs/adr/ADR-018-deterministic-routing.md` (Status: Accepted, 2026-06-01). Decision: replace the GatekeeperAgent LLM decision with deterministic arithmetic routing, move the explanation role to FallbackAgent. Three-step decision: compute flags from scores, label trigger_category in code (empty_retrieval checked before low_groundedness; zero-chunk also fails the gs floor so chunk count is the only discriminant), fallback iff a blocking category was set. Supersedes ADR-010; ADR-010 retained as the Day-10 record.

### Steps 1.6.1–1.6.4 (implemented, not yet re-evaluated)

**1.6.1 — Deterministic router (`src/agents/gatekeeper_agent.py`)**

Full rewrite. Same `run()` signature (`query, response_text, chunks, evaluation, leader`) and `RoutingDecision` return type; the flow contract is unchanged. All LLM infrastructure removed (`crewai`, `instructor`, `litellm` imports gone). Threshold constants imported from `evaluator_agent.py` (`GROUNDEDNESS_MIN`, `STYLE_MIN`, `CONFIDENCE_MIN`) so any future recalibration propagates automatically. `_compute_flags()` recomputes the deterministic flag set from scores independently. Labeling tree: `len(chunks)==0 → empty_retrieval` (checked first); `low_groundedness in flags → low_groundedness`. `trigger_reason` is a factual code-templated string with the actual score value. `last_run_timings` set to `{"generate_ms": 0.0, "parse_ms": 0.0}`.

`quality_flags` semantics (corrected at STOP GATE 1.6a review): carries **non-blocking flags only** (`low_style`, `low_confidence`). The blocking flag (`low_groundedness`) promotes to `trigger_category` and is **not** duplicated in `quality_flags`. `_BLOCKING_FLAGS = frozenset({"low_groundedness"})` is defined at module level; `quality_flags = [f for f in flags if f not in _BLOCKING_FLAGS]`. Zero-chunk fallback (`trigger_category=empty_retrieval`): the scorer returns `gs=0.0` which also fires `low_groundedness`, but `low_groundedness` is a blocking flag and therefore excluded from `quality_flags` regardless of whether it is the trigger — `quality_flags=[]` on a pure empty-retrieval fallback.

**1.6.2 — Schema additions (`src/schemas.py`), additive only**

`RoutingDecision`: added `quality_flags: list[str] = Field(default_factory=list)`, added `"evaluation_error"` as the sixth literal in `trigger_category`. `FallbackResponse`: added `trigger_category: Optional[Literal[...six-literal set...]] = None`. Both Literal sets are identical: `{low_groundedness, off_domain, hallucination_risk, chunk_mismatch, empty_retrieval, evaluation_error}`. No existing fields changed.

**1.6.3 — Enriched FallbackAgent (`src/agents/fallback_agent.py`)**

`run()` receives four new Optional kwargs: `trigger_category`, `groundedness_score`, `style_score`, `confidence_score` (all default `None` for backward compatibility). `_build_task_description` now receives all four plus `style_profile` (which was previously accepted by `run()` but silently unused — the live dead parameter from ADR-018 Precondition Check). `_format_style_examples()` added to extract up to two sample emails (truncated to 400 chars each) from the style profile as in-voice grounding examples. Task description now includes failure category, actual quality scores, and style examples so the redirect is specific to the trigger and in the leader's voice. `trigger_category` propagated to `FallbackResponse.trigger_category` on both the success and failsafe paths.

**1.6.4 — Flow wiring (`src/flow.py`)**

`handle_fallback`: added `trigger_category` extraction from `state.routing_decision.trigger_category`. Passes `trigger_category`, `groundedness_score`, `style_score`, `confidence_score` as kwargs to `FallbackAgent.run()`.

`route()` — emergency guard: corrected at STOP GATE 1.6a review. The evaluate-is-None path now emits `trigger_category="evaluation_error"` and `trigger_reason="evaluation_error: evaluate step returned None"` (previously left null, which violated the ADR-018 contract). The guard diff:

```python
# before
RoutingDecision(
    decision="fallback",
    reasoning="evaluate step produced no result — emergency fallback",
)

# after
RoutingDecision(
    decision="fallback",
    reasoning="evaluate step produced no result — emergency fallback",
    trigger_category="evaluation_error",
    trigger_reason="evaluation_error: evaluate step returned None",
)
```

### STOP GATE 1.6a (pending Ruby's gate decision)

Initial assertion submission had two issues caught at gate review: (a) `quality_flags` was incorrectly carrying the blocking flag `low_groundedness`, and (b) `evaluation_error` was missing from both Literal sets and the emergency guard was not emitting `trigger_category`. Both corrected; see 1.6.1 and 1.6.2 notes above. Corrected assertion results (four checks, no API calls):

```
Checked 84 records from results/evaluation_day12_reeval.json
PASS: delivers iff groundedness >= 0.6 — all 84 records correct
PASS: no blocking flags in quality_flags across all 84 records

── Zero-chunk unit case (empty_retrieval branch) ──
  decision:         fallback
  trigger_category: empty_retrieval
  trigger_reason:   empty_retrieval: 0 chunks retrieved
  quality_flags:    []
PASS: zero-chunk unit case → trigger_category='empty_retrieval', quality_flags=[]

── Deliver-low-style unit case (non-blocking flag on deliver path) ──
  decision:         deliver
  trigger_category: None
  quality_flags:    ['low_style']
PASS: deliver-low-style → quality_flags=['low_style'], no blocking flag

── Evaluation-is-None unit case (evaluation_error guard in flow.py) ──
  decision:         fallback
  trigger_category: evaluation_error
  trigger_reason:   evaluation_error: evaluate step returned None
  quality_flags:    []
PASS: evaluation-is-None → trigger_category='evaluation_error'
```

Assertion script: `scripts/assert_router_16a.py` (throwaway). All changes committed at `a395c40` on `feat/evaluator-deterministic-flags`.

---

## Phase 1.6.5 — Re-eval (84 records, ADR-018 deterministic router live)

STOP GATE 1.6a cleared. Ran the 14 in-domain queries × 2 leaders × 3 passes = 84 records through the new deterministic router.

### Prerequisite

`scripts/analyze_reeval.py` `VALID_TRIGGER_CATEGORIES` updated from five to six literals (adding `"evaluation_error"`) to match the Phase 1.6.2 schema change before running the integrity assertion.

### Re-eval run

Script: `scripts/reeval2_indoman.py` (throwaway). Output: `results/archive/evaluation_day12_reeval2.json`. OOD excluded (OOD groundedness 0.33–0.48; cannot flip on a threshold change; already 6/6 fallback in Phase 1).

### STOP GATE 1.6b — Surface A: deliver-rate re-gate

**Per-pass in-domain 2×2 grid:**

| Pass | Torvalds | Floor | KH | Floor | E2 target (55%) |
|---|---|---|---|---|---|
| 1 | 3/14 (21.4%) | 42.9% BELOW | 10/14 (71.4%) | 35.7% CLEARS | CLEARS |
| 2 | 4/14 (28.6%) | 42.9% BELOW | 9/14 (64.3%) | 35.7% CLEARS | CLEARS |
| 3 | 4/14 (28.6%) | 42.9% BELOW | 9/14 (64.3%) | 35.7% CLEARS | CLEARS |

Three-run variance: Torvalds 3–4/14, KH 9–10/14 (non-zero variance, as expected from CloneAgent's temp=0.3).

**ADR-013 per-leader gap:** 35.7–50.0pp across all three passes. Trigger fired all three passes (threshold: >20pp).

**Torvalds fallback groundedness breakdown (31 total fallbacks across 3 passes):**
- Near-miss (0.55–0.60, gs range 0.554–0.599): 18 — correctly below the 0.60 floor
- Clearly low (<0.55, gs range 0.497–0.546): 13 — clear groundedness deficit

Both categories are legitimate groundedness failures: the deterministic router is correctly applying the 0.60 floor. Torvalds below floor is the honest per-leader operating point. Floor does not move (ADR-004 calibrated; pre-committed discipline).

KH: honest partial success (clears floor and E2 target). ADR-013 contingency fires on all three passes.

### STOP GATE 1.6b — Surface B: fallback-quality prose read

Four samples read. Voice differentiation visible:

- **Torvalds (direct/terse):** "I can't provide a solid answer on why L2 regularization spreads weight magnitudes more evenly than L1, or when to prefer L1. It's outside my expertise right now."
- **KH (formal/deferential):** "Thank you for your question regarding the differences between L1 and L2 regularization. While I appreciate the interest in the nuances of these techniques, I must admit that I can't provide a well-grounded answer…"

All four fallbacks name the exact query topic (not a generic redirect). Old v1 boilerplate ("I'm not able to answer off-topic questions") absent. Voice differentiation present but not strongly leader-distinctive — acknowledged as a Phase 1.7+ investigation item, not a routing failure.

**Harness gap identified:** `_extract_leader_record()` in `src/eval/harness.py` does not serialize `routing_decision.quality_flags` to JSON. Stored records show `quality_flags=None` even for delivered records where non-blocking flags apply. The 1.6a pure-function assertions confirmed the router computes `quality_flags` correctly; this is a recording gap only. Fix deferred to Phase 2 alongside the `cli.py` refactor.

### Three additional confirmations

- **RC-3 gone:** 0 instances of `trigger_category=low_groundedness` where no `low_groundedness` flag was computed. Not recurred.
- **trigger_category integrity:** PASS — non-null iff fallback, all values from the valid six-literal set, across all 84 leader-records.
- **quality_flags blocking-flag check:** No blocking flags in `quality_flags` on any delivered record. Confirmed via the 1.6a assertions; stored JSON shows `quality_flags=None` (harness gap noted above).

### STOP GATE 1.6b result

**PASSED for the fix.** Routing is deterministic (RC-2 resolved), RC-1 and RC-3 resolved, fallback quality meets the bar (trigger-specific, voice-present, no old generic redirect). Torvalds below floor is an honest per-leader result; per pre-committed discipline the floor does not move and Phase 2 remains blocked until both leaders clear their floors.

---

## Phase 1.7 — Gatekeeper rename (agent → component, ADR-018 Consequences)

Executed after STOP GATE 1.6b passed. Behavior-preserving rename sequenced AFTER the logic change so that logic and rename never mix in one diff.

### 1.7.0 — Convention finding (read-only)

Existing deterministic components: `Retriever` (`src/components/retriever.py`), `ScoringEngine` (`src/components/scoring_engine.py`), `StyleProfileBuilder` (`src/components/style_profile_builder.py`). Convention: `src/components/<concept_name>.py`, `PascalCase` noun, no suffix. `Retriever` is the unsuffixed single-concept precedent.

Rename target: `src/components/gatekeeper.py`, class `Gatekeeper`. Convention match confirmed before touching anything.

### 1.7.1 — Rename and move

Files changed:

| File | Action |
|---|---|
| `src/agents/gatekeeper_agent.py` | Deleted (git rm) |
| `src/components/gatekeeper.py` | Created — same body, class `GatekeeperAgent` → `Gatekeeper`, module docstring updated |
| `src/flow.py` | Import path + class name + route() docstring updated |
| `src/components/scoring_engine.py` | Docstring: `GatekeeperAgent` → `Gatekeeper` |
| `src/agents/evaluator_agent.py` | Docstring: `GatekeeperAgent` → `Gatekeeper` |
| `src/agents/fallback_agent.py` | Docstring: `GatekeeperAgent` → `Gatekeeper` |
| `src/schemas.py` | Docstring: `GatekeeperAgent` → `Gatekeeper` |
| `tests/integration/test_gatekeeper_agent.py` | Deleted — old LLM-plumbing tests for nonexistent API (`_build_backstory`, `_build_crew`, etc.); was already causing a collection error |
| `tests/integration/test_gatekeeper.py` | Created — 17 deterministic tests for `_compute_flags` and `Gatekeeper.run()` (no mocks; pure-function checks) |
| `tests/test_flow.py` | Patch strings: `src.flow.GatekeeperAgent.run` → `src.flow.Gatekeeper.run` (3 occurrences); docstring updated |
| `tests/integration/test_compare_leaders.py` | Patch strings: `src.flow.GatekeeperAgent.run` → `src.flow.Gatekeeper.run` (4 occurrences) |

`src/components/__init__.py` not modified: adding `Gatekeeper` there creates a circular import (`src.components.__init__` → `gatekeeper` → `evaluator_agent` → `src.components.scoring_engine` → partially-initialized `src.components`). `flow.py` imports directly from `src.components.gatekeeper`; no re-export needed.

Zero logic changed. `run()` signature, `RoutingDecision` return type, `_compute_flags`, `_BLOCKING_FLAGS`, `last_run_timings`, and all routing arithmetic are byte-for-byte identical.

### 1.7.2 — Verification

**No-remaining-reference grep:** `grep -rn "GatekeeperAgent\|gatekeeper_agent" src/ tests/` — empty (zero results).

**Test suite after rename:**
```
8 failed, 492 passed, 37 skipped, 28 warnings in 4.41s
```

All 8 failures are pre-existing, none caused by the rename:

| Failure | Root cause | Phase when introduced |
|---|---|---|
| `test_load_queries_canonical_file` | Missing `queries_v1.json` | Pre-existing (documented in plan) |
| 6 × `test_fallback_agent` | `_build_task_description()` API enriched in 1.6.3 (new required params); test file not yet updated | Phase 1.6.3 |
| `test_run_propagates_flags` | Deterministic flags now also emit `low_confidence` at cs<0.80; test expected only `[low_style, low_groundedness]` | Phase 1.5.1 |

The old `test_gatekeeper_agent.py` was already failing with a collection error before this phase (imported `_build_backstory` and other functions removed in 1.6.1). The rename eliminated that collection error and replaced it with 17 passing deterministic tests.

New `tests/integration/test_gatekeeper.py` — all 17 passed: `_compute_flags` arithmetic (6 tests), deliver path with quality_flags (5 tests), fallback `low_groundedness` path (3 tests), `empty_retrieval` ordering (2 tests), `last_run_timings` (1 test).

### 1.7.3 — ADR-014 inventory correction

`docs/adr/ADR-014-agent-component-inventory.md` updated:

- Dated correction block (2026-06-01) citing ADR-018 added above the Decision counts.
- Agent list 4 → 3: GatekeeperAgent entry removed.
- Component list 3 → 4: Gatekeeper entry added with file path `src/components/gatekeeper.py` and one-line breadcrumb for readers following ADR-018 or ADR-010 file references.
- Consequences section counts corrected from "four files in `src/agents/`, three in `src/components/`" to three and four respectively, with a parenthetical dated note.

### STOP GATE 1.7 result

**CLEARED (2026-06-02).** Rename verified: zero logic changed, no remaining `GatekeeperAgent`/`gatekeeper_agent` references in `src/` or `tests/`, 17 new deterministic tests pass, ADR-014 inventory corrected. Phase 2 unblocked from the structural side; remains blocked on Torvalds floor not cleared.

---

## Artifacts produced

| Artifact | Type | Status |
|---|---|---|
| `src/eval/harness.py` | New — measurement harness | Complete |
| `results/archive/evaluation_day12.json` | Data — 96-record full-run results | Complete |
| `results/archive/evaluation_day12_reeval.json` | Data — 84-record re-eval results | Complete |
| `docs/day11-evaluation.md` | Doc — gate evaluation report | Complete |
| `docs/evaluation-methodology.md` | Doc — three-layer methodology | Complete |
| `docs/adr/ADR-015-...md` | Amendment appended | Complete |
| `docs/adr/ADR-017-deterministic-flag-raising.md` | New ADR + Amendment 1 | Complete |
| `docs/adr/ADR-018-deterministic-routing.md` | New ADR | Complete |
| `src/agents/evaluator_agent.py` | Modified — deterministic flags + STYLE_MIN | Complete |
| `src/agents/gatekeeper_agent.py` | Full rewrite — deterministic router (ADR-018) | Complete |
| `src/agents/fallback_agent.py` | Modified — trigger_category + scores + style_profile wiring | Complete |
| `src/flow.py` | Modified — handle_fallback wiring + evaluation_error guard | Complete |
| `src/schemas.py` | Modified — quality_flags + six-literal trigger_category sets | Complete |
| `scripts/reeval_indomain.py` | Throwaway — re-eval runner (Phase 1.5.3) | Complete |
| `scripts/analyze_reeval.py` | Throwaway — gate analysis (updated to six-literal set in 1.6.5) | Complete |
| `scripts/assert_router_16a.py` | Throwaway — STOP GATE 1.6a assertion (4 checks) | Complete |
| `scripts/reeval2_indoman.py` | Throwaway — Phase 1.6.5 re-eval runner | Complete |
| `scripts/equivalence_guard.py` | Throwaway — ADR-017 Phase 1.5.2 equivalence check | Complete |
| `results/archive/evaluation_day12_reeval2.json` | Data — 84-record Phase 1.6.5 re-eval results | Complete |
| `src/components/gatekeeper.py` | New — deterministic Gatekeeper component (renamed from gatekeeper_agent.py) | Complete |
| `src/agents/gatekeeper_agent.py` | Deleted (renamed to Gatekeeper in src/components/) | Complete |
| `tests/integration/test_gatekeeper.py` | New — 17 deterministic tests for Gatekeeper | Complete |
| `tests/integration/test_gatekeeper_agent.py` | Deleted (LLM plumbing tests for removed API) | Complete |
| `docs/adr/ADR-014-agent-component-inventory.md` | Amendment — agent count 4→3, component count 3→4, breadcrumb | Complete |

## Pending (not started this session)

| Item | Blocked on |
|---|---|
| CONFIDENCE_MIN calibration | Post-1.6 re-eval data |
| `_parse_review` removal | Day-13 latency item; needs CrewAI task `expected_output` update to suppress "Flags:" section in `.raw` |
| Phase 2 (cli.py / visualization.py refactor, v1 retirement, Notion sync) | STOP GATE 1.6b + STOP GATE 1.7 cleared; also Torvalds floor not cleared |
| Notion ADR sync: ADR-010 + ADR-015 + ADR-017 + ADR-018 + ADR-014 correction | Phase 2 |
| 6 × test_fallback_agent failures (API mismatch from 1.6.3 enrichment) | Phase 2 test-file update |
| `test_run_propagates_flags` (evaluator test expects pre-1.5.1 flag set) | Phase 2 test-file update |
| Harness gap: `quality_flags` not serialized to JSON by `_extract_leader_record()` | Phase 2 alongside cli.py refactor |

## Dead code ledger (unchanged from Day 11)

All Day-11 ledger items carry forward. No new retirements this session (Phase 2 not started).

## Architecture-honesty check

Three Agents (`src/agents/`): CloneAgent, EvaluatorAgent, FallbackAgent. Four Components (`src/components/`): Retriever, ScoringEngine, StyleProfileBuilder, Gatekeeper. One Flow orchestrator. Matches ADR-014 inventory as corrected by the 2026-06-01 amendment. No LLM call in the Gatekeeper, no LLM call in any other Component. The EvaluatorAgent's flag-raising is deterministic code; the LLM produces only the explanation text. Routing from scores to decision is arithmetic. The flow contract (route() returns a string, downstream reads `.decision`) is unchanged from Day 11.
