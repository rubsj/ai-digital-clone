# Day 12 Plan — P6 v2: End-to-End Evaluation and v1 Retirement

**Project:** P6 — Torvalds Digital Clone (Multi-Agent System)
**Day:** 12 (May 29, 2026), branch `refactor/p6-multi-agent-rework`
**Model split:** Opus authored this plan. Sonnet executes it. Sonnet does not re-debate any of the 7 Architecture Rules in CLAUDE.md; if a step touches one, Sonnet stops and surfaces to Ruby.

**Sources of truth (re-read from disk at each phase start, not from context):**
- `docs/PRD.md` §2.1, §2.4, §2.5, §2.7, §7.5.1, §8 (Day 12 row), §12.2
- `CLAUDE.md` (Architecture Rules, Verification Protocol, Writing Rules)
- `docs/adr/` ADR-010, ADR-012, ADR-015, ADR-016, ADR-017 (ADR-017 is the source of truth for Phase 1.5)
- `docs/session-notes/day11.md`
- `src/flow.py`, `src/schemas.py`, `src/agents/gatekeeper_agent.py`, `src/agents/fallback_agent.py`, `src/agents/evaluator_agent.py`, `src/cli.py`, `src/visualization.py`

**Session shape.** Two phases with a hard stop gate between them. Phase 1 measures the system and produces the ship/no-ship number. Phase 2 is the destructive refactor and retirement, gated on Phase 1 clearing. Phase 2 is planned here but marked gated-not-started. It does not begin until Phase 1 ships or routes to the investigate branch with an approved fix, with the keyword "approved" or "proceed" from Ruby.

---

## Resolved decisions

Ruby resolved the four conflicts surfaced at plan review and added one requirement (C5). They are settled, and the body of this plan reflects them. Recorded here so the reasoning is not lost.

**C1 — accepted, with a guard added.** The harness does not call `compare_leaders()`. It uses `run_leader_pair(query)` in `src/eval/harness.py`, which mirrors the ADR-005 shared-retrieval pattern (Torvalds flow retrieves, Kroah-Hartman flow receives `chunks=shared_chunks` and early-exits) but holds both `DigitalCloneFlow` instances and reads `flow.state` (`routing_decision`, `styled_response`/`fallback_response`, `evaluation`) plus `flow.timings`. Added guard: the harness asserts exactly one `Retriever.run()` call across the leader pair, the same assertion `tests/integration/test_compare_leaders.py` makes against `compare_leaders()`, so the harness cannot silently fork the shared-retrieval pattern it mirrors. The assertion runs inside the pre-flight (C4), not as a later step.

**C2 — resolved; the earlier recommendation is corrected.** `data/eval/queries.json` is not edited; the file is valid. `category` (corpus-membership) and `trigger_category` (the GatekeeperAgent runtime rejection-reason) are different vocabularies on different axes, input-type versus rejection-reason, and are not meant to match. Aligning them would destroy the in-domain/OOD axis and retroactively change the eval inputs on the night of the gate (the day8-findings §2d gate-to-outcome failure mode).
  - The harness derives in-domain versus OOD from `category`, not from `expected_behavior`. Encode as named sets: `IN_DOMAIN_CATEGORIES = {statistical_learning_ml, data_mining, numerical_methods, programming_fundamentals}` and `OOD_CATEGORIES = {systems_absent_from_corpus, off_topic_technical}`. A `category` outside both sets raises (fail loud); it does not silently miscolumn.
  - `expected_behavior` is the per-query grading target (the gatekeeper's actual decision is scored against it), not the axis label. `category` and `expected_behavior` agree 20/20 in the current file, so the grid is unambiguous tonight; keying the axis off `category` is durable insurance, not a tonight-fix.
  - The per-query `leader` tag is ignored for grid assignment. Every query runs through both leaders, giving 40 records (14 in-domain plus 6 OOD per leader).

**C3 — accepted.** Measure as shipped (CloneAgent and FallbackAgent at 0.3, routing and scoring at 0). Report the three-run range. Do not edit any agent temperature.

**C4 — accepted, with a cheaper mechanical run design.** Replaces the flat "3x full 20-query pass".
  - Pre-flight first (cheap): `run_leader_pair()` on one in-domain query (q01) and one OOD query (q15). Paste both timing dicts, both routing decisions, and the C1 one-retrieval assertion result. Then hard stop for cost-guard approval before any larger run (CLAUDE.md Verification Component 2, more than 100 calls).
  - After approval: one full run (all 20 times both leaders = 40 records) for the grid and the OOD gates, then two additional in-domain-only re-runs (the 14 in-domain queries times both leaders = 28 records each). Total about 96 records per session, about 460 to 610 chat completions. Gate-moving variance lives on in-domain deliver decisions; OOD cleared at 0/12 with high margin on Day 8 and is pass/fail on one clean run.
  - Reactive OOD recheck: if any OOD record delivers on the full run, re-run that specific record twice before classifying it a hallucination, to separate a stable defect from a sampling flip. This is the only OOD re-run; OOD is not re-run prophylactically.
  - Variance scope: the three-run variance analysis (item 5, item 6 doc) covers the in-domain deliver rate (3 data points per in-domain query) and the OOD cells at n=1 unless the reactive recheck fires. `docs/day11-evaluation.md` states this scope explicitly so the variance claim is not overstated.

**C5 — new requirement (capture).** The harness persists, per record: the full styled-response text (deliver path) and the full routing `reasoning` string (both paths), alongside the decision, scores, and timings already specified. The zero-hallucination OOD gate (item 5c) is a prose judgment, so an OOD-deliver must be auditable as text at the gate, not inferred from a cell count. The objects are already in `flow.state`, so this is cheap. It also closes the open Day-8 follow-up where the CLI did not persist `trigger_reason` (day8-findings line 29). Written into `results/evaluation_day12.json` per record.

---

## PRD Coverage Check

Mapping every Day-12 line in PRD §8 to a phase. Silent deferral is a named failure mode, so any re-scope is called out here.

| PRD §8 Day-12 line | Covered by | Notes |
|--------------------|-----------|-------|
| Run end-to-end smoke test on diverse queries | Phase 1 pre-flight + the full run + two in-domain re-runs | Pre-flight is one in-domain plus one OOD query-pair; the full run over 20 queries is the diverse exercise (C4 run design). |
| Run `cli evaluate` against the v2 query set | Phase 2 item 2 (re-scoped from Phase 1) | `cli.py` carries v1 field names (`final_output`, `final_score`, `0.75`) and is broken on the v2 schema. Phase 1 measures through `src/eval/harness.py` instead. `cli evaluate` is repaired and exercised in Phase 2 after the cli.py refactor. This is an explicit re-scope, authorized by the session shape, not a silent deferral. |
| Generate `docs/day11-evaluation.md` (2x2 grid, per-leader breakdowns, regression-anchor check, PRD scorecard) | Phase 1 item 6 | Filename stays `day11-evaluation.md` per §7.5.1 despite Day-12 authorship. |
| Decision gate: E2 hit? E1 floor hit? ship/no-ship per ADR-015 | Phase 1 item 5 + STOP GATE 1; resolved NO-SHIP + investigation (Phase 1.5, ADR-017) | Conjunctive gate against the amended geometry (item 7). Gate ran NO-SHIP (in-domain 0/14 both leaders); this is a recorded outcome, not a deferral. The investigate branch (Phase 1.5) executes ADR-017. |
| `docs/evaluation-methodology.md` (three-layer per ADR-016) | Phase 1 item 6 | §7.5.1 lists it as a Day-12 deliverable; PRD §8 prose omits it. Flagged so it is not lost. |

PRD §12.2 mapping-table updates and the Notion ADR sync are Phase 2 (items 4 and 5). They trace to §7.5.1 and the Day-11 named follow-ups, not to a PRD §8 Day-12 line.

---

## Phase 1 — Measure first (no destructive changes)

Goal: get the ship/no-ship number before any refactor. This phase depends only on `src/flow.py` and the four Agents (PR #15, merged green). It does not depend on the `cli.py`/`visualization.py` refactor. Re-read this Phase 1 section and the source files listed at the top from disk before starting.

**Pre-flight (cheap smoke before the larger run).** Run `run_leader_pair()` on one in-domain query (q01) and one OOD query (q15). Confirm both flows complete, `flow.timings` is populated with the expected keys, `flow.state.routing_decision` is a `RoutingDecision` on both leaders, and exactly one `Retriever.run()` call fired across the leader pair (the C1 guard, the same assertion as `tests/integration/test_compare_leaders.py`). Verification: paste the two timing dicts, the two routing decisions, and the one-retrieval assertion result. This catches a missing `OPENAI_API_KEY`/`COHERE_API_KEY` or a profile/index load error for about a dozen calls rather than a thousand. Then hard stop for cost-guard approval (C4) before the full run.

1. **Real-LLM measurement harness.** Create `src/eval/harness.py` with `run_leader_pair(query) -> dict` (the C1 helper: shared-retrieval two-flow pattern, asserts exactly one `Retriever.run()` across the pair, captures per-leader `routing_decision`, output type, `evaluation` scores, `flow.timings`, and the `last_run_timings` sub-keys) and `run_measurement(path) -> dict` (the C4 run design: one full pass over all 20 queries times both leaders, then two in-domain-only re-runs over the 14 in-domain queries times both leaders, then the reactive OOD recheck if any OOD record delivered). Capture per record: decision, `trigger_category`, the three scores, `flags`, output type, the full styled-response text on deliver, the full routing `reasoning` string on both paths (C5), and every timing key (`retrieve_ms`, `clone_ms`/`clone_generate_ms`/`clone_parse_ms`, `evaluate_ms`/`evaluate_score_ms`/`evaluate_generate_ms`/`evaluate_parse_ms`, `route_ms`/`route_generate_ms`/`route_parse_ms`, `deliver_ms` or `fallback_ms`/`fallback_generate_ms`/`fallback_parse_ms`). Write raw results to `results/evaluation_day12.json`. This is the first run against a non-zero clock; Day 11 asserted timing key-shape under mocks only. Verification: `results/evaluation_day12.json` exists with about 96 records (40 full plus 28 plus 28 in-domain re-runs, plus any reactive OOD recheck), the styled-response text and routing `reasoning` are present per record, and `python -c` prints one full timing dict.

2. **Routing 2x2 grid per leader.** From the full-run records, build the in-domain (deliver/fallback) by OOD (deliver/fallback) grid for each leader, per PRD §2.1. In-domain versus OOD is derived from `category` via `IN_DOMAIN_CATEGORIES`/`OOD_CATEGORIES`, not from `expected_behavior` (C2); a `category` in neither set raises. Verification: print both grids from the full run with counts that sum to 14 in-domain plus 6 OOD per leader.

3. **`trigger_category` integrity assertion (over real outputs, not a schema validator).** For every record assert: `trigger_category` is non-null if and only if `decision == "fallback"`, and when non-null it is one of the five literals (`low_groundedness`, `off_domain`, `hallucination_risk`, `chunk_mismatch`, `empty_retrieval`). Do not add a Pydantic cross-field validator to `RoutingDecision`; the router has no failsafe, so a validator on the live path turns mislabelled data into a crashed pipeline (the Day-11 amendment-4 call, ADR-010). Inspect the `trigger_category` distribution (chart-6 source) for smear across `low_groundedness` / `hallucination_risk` / `chunk_mismatch`. If a sixth category looks needed, stop and surface it as a logged ADR-010 amendment. Sonnet does not edit the literal set ad hoc. Verification: paste the assertion result (pass, or the offending records) and the category distribution.

4. **Latency read against the §2.7 8s budget (diagnostic, not a gate tonight).** For each timing dict, read `parse_ms` per agent. Every agent here makes a second LLM call for the Instructor parse (`instructor.from_litellm(litellm.completion)`), so a `parse_ms` near 1s is that real second network round-trip, the hidden round-trip the prompt flags, and is a genuine latency target; a sub-10ms `parse_ms` would be local Pydantic coercion (it will not be, given the two-call structure). Day 8 measured deliver 7.4 to 8.5s and fallback 10.0 to 11.4s under a lighter pipeline; v2 adds sequential LLM calls, so a breach on both paths is expected. Editing the §2.7 SLA is the last lever, not the first. Report measured per-stage latencies; recommend an SLA action only after reading `parse_ms`, and surface that recommendation rather than apply it. A §2.7 breach is not a gate clause and does not affect the ship/no-ship decision; the three conjunctive clauses are in-domain deliver rate, OOD fallback, and zero hallucination, latency is not among them. Log a breach as a Day-13 latency item (edit §2.7 or optimize the Instructor parse round-trips per the `parse_ms` reading). Do not resolve it tonight and do not raise it as a gate question. Verification: paste the per-stage latency summary (deliver and fallback) with the `generate_ms`/`parse_ms` split.

5. **Gate evaluation against the locked geometry (conjunctive; any single failure no-ships).**
   - (a) In-domain deliver rate per leader. E2 target: at or above 55% both leaders ships on this axis. Floors (honest per-leader Day-8 baselines, not pooled 39%): Torvalds at or above 42.9% (6/14), Kroah-Hartman at or above 35.7% (5/14). Between floor and E2 is the judgment band; document the measured rate as the operating point and ship on this axis if the other two clauses pass. Below floor: stop ship tonight and investigate root cause (read the Gatekeeper `reasoning` plus scores to distinguish correct conservatism from pathological punting on groundable queries). Investigation output is a re-measured fix or a documented no-ship, not a license to ship below floor.
   - (b) OOD fallback equals 100% both leaders. Non-negotiable.
   - (c) Zero hallucinations on the OOD queries (the `systems_absent_from_corpus` and `off_topic_technical` set per `OOD_CATEGORIES`, six records per leader on the full run). Non-negotiable. An OOD-deliver is judged as text using the persisted styled-response and routing `reasoning` (C5), not inferred from a cell count, and triggers the reactive OOD recheck (re-run that record twice) before it is classed a hallucination.
   - Run design and variance (per C4): the in-domain deliver rate has 3 data points per query (one full run plus two in-domain re-runs); the OOD cells are n=1 unless the reactive recheck fires. temp=0 is greedy, not guaranteed deterministic, and per C3 CloneAgent runs at 0.3, so expect non-zero in-domain variance. If in-domain variance is zero, the rate is a point estimate. If non-zero and the rate lands near a floor or near 55%, report the range and decide against the worst of the three in-domain runs, not a cherry-picked one. `expected_behavior` is the per-query grading target the decision is scored against.
   - Also run ADR-013's per-leader style trigger in the same pass: Kroah-Hartman in-domain deliver rate more than 20 points below Torvalds fires the per-leader-weighting contingency. Shared evaluation event per ADR-015.
   - Verification: print one full-run 2x2 grid per leader; the per-leader in-domain deliver rate across all three runs as a range; the OOD fallback count and the category-5 hallucination count from the full run; and the ADR-013 delta. There is one grid per leader (the 40-record full run). The two in-domain-only re-runs produce no OOD cells, so the variance is on the in-domain deliver rate (3 points), not on three full grids.

6. **Author the two documents.** `docs/day11-evaluation.md` (keep the filename per §7.5.1) carries the 2x2 grid, per-leader deliver/fallback, three-run variance (scoped to the in-domain deliver rate; OOD cells are n=1 unless the reactive recheck fired, stated explicitly so the variance claim is not overstated), comparison versus the Day-8 v2 baseline (in-domain deliver 11/28 pooled stated in day8-findings; OOD 12/12 fallback; style means Torvalds 0.9025 / Kroah-Hartman 0.8355; groundedness 0.6258), the regression-anchor check on the current q12 and q13 (both `regression_anchor: true`, both in-domain `deliver`), the PRD §2 scorecard, and the gate decision recorded against the amended geometry. The per-leader Day-8 floors (Torvalds 6/14, Kroah-Hartman 5/14) are derived in the doc, not asserted: day8-findings states the pooled 11/28 and the per-query outcome table (4 both-deliver, 2 Torvalds-only, 1 Kroah-Hartman-only), from which Torvalds = 4+2 = 6 and Kroah-Hartman = 4+1 = 5; show that arithmetic so "where did 5/14 come from?" has an answer in the doc. The 6/5 split is a count from the Day-8 run; keep the Day-8 outcome-table q-IDs distinct from the current `queries.json` q-IDs (day8 q03 is the current q12 binary-search per day8-findings line 101, for example), and check the current q12/q13 regression anchors separately against the current run. Do not map one numbering onto the other. `docs/evaluation-methodology.md` documents the three-layer approach per ADR-016 (Layer 1 unit continuous, Layer 2 per-agent recorded-LLM contract tests, Layer 3 system via the query set). Follow the CLAUDE.md Writing Rules (no em-dashes, no tricolons, plain prose, plain bullets). Verification: both files exist and the gate decision in `day11-evaluation.md` cites ADR-015 and the item-7 amendment.

7. **ADR-015 amendment (log it, surface it, do not apply silently).** Author an amendment block dated Day 12, appended to `docs/adr/ADR-015-post-rework-eval-acceptance-criteria.md`.
   - The floor changes from pooled "at or above 39% per leader" to honest per-leader baselines: Torvalds at or above 42.9% (6/14), Kroah-Hartman at or above 35.7% (5/14). Rationale: 39% is the pooled 11/28 rate; applied per leader it sets Kroah-Hartman's floor above Kroah-Hartman's own Day-8 baseline (35.7%) and would misfire a false regression.
   - The sub-floor branch reconciles with the planner Risk Register: "stop ship plus investigate" supersedes the bare "does not ship" prose, since the floor was produced by the v1 0.75-threshold formula (day8-findings §2c), which is the exact mechanism GatekeeperAgent replaced (ADR-010). A qualitative gate legitimately delivering fewer than the threshold did is not by itself a regression; investigation distinguishes correct conservatism from pathological punting.
   - Notion sync of this amendment is Phase 2 (alongside the ADR-010 `trigger_category` sync). Verification: the amendment block is present and dated; no edit to the ADR-015 Decision section's original numbers (amendment is additive).

### STOP GATE 1 (Phase Defence plus ship/no-ship decision)

Paste actual captured terminal output: one full-run 2x2 grid per leader, the in-domain deliver-rate range across the three runs, the OOD fallback count and category-5 hallucination count from the full run, the timing dicts, and the `trigger_category` assertion result. Self-reported completion is an anti-pattern. Render the four-category Phase Defence on Phase 1 (including the mandatory Category V v1-drift check). Ruby records the gate decision. Phase 2 does not start unless Phase 1 ships, or routes to the investigate branch with a defined approved fix. Keyword "approved" or "proceed" required.

**Gate result (recorded 2026-06-01): NO-SHIP per the pre-committed Risk Register.** In-domain deliver 0/14 both leaders, three-pass zero variance (deterministic, not a sampling artifact); OOD 6/6 fallback both leaders, 0 hallucinations. Root cause and fix rationale: see ADR-017 (Deterministic Flag-Raising). The investigate branch is taken: proceed to Phase 1.5.

---

## Phase 1.5 — Investigation fix (executes ADR-017; between STOP GATE 1 and Phase 2)

STOP GATE 1 closed NO-SHIP on the investigate branch. This phase executes the fix decided in ADR-017 (Deterministic Flag-Raising); it does not re-derive it. The code read that produced the root-cause finding is complete and lives in ADR-017. Re-read ADR-017 and `src/agents/evaluator_agent.py` from disk before starting. ADR-017 is the source of truth for this phase; this plan sequences the work and owns the gates.

1.5.1 **Implement deterministic flag-raising in EvaluatorAgent (ADR-017 Decision).** Raise each flag if and only if the `ScoringEngine` score is below its named-constant threshold: groundedness 0.60, style 0.90, confidence 0.80. The LLM no longer decides flags; the flag list is computed in code from the deterministic scores. Verification: a unit assertion that no record with `groundedness_score >= 0.60` carries a `low_groundedness` flag; paste the assertion result and the three named constants with file:line.

1.5.2 **Remove the redundant `_parse_review` LLM call (ADR-017 Decision), taking `explanation` from the kickoff `.raw` text.** Equivalence guard (ADR-017 Decision): before removing, compare the kickoff `.raw` against the old `_parse_review` parsed `explanation` on 2 to 3 records. The flag fix (1.5.1) is the variable that matters; the re-eval runs either way, and only the parse removal is conditional. Two branches, both continue to 1.5.3:
   - If `.raw` is equivalent to the parsed explanation (just extraction, no meaningful rewording): remove `_parse_review` and take `explanation` from `.raw`. Proceed to 1.5.3 with the flag fix plus the parse removal.
   - If `_parse_review` was meaningfully rewording rather than just extracting: keep `_parse_review`, do NOT remove it this phase, and surface the deferral as a noted follow-up. CONTINUE to 1.5.3 with the 1.5.1 flag fix as the sole change. Do not halt Phase 1.5 on this branch.

   In both cases the flag fix is the single isolated variable in the re-eval; what differs is only whether the parse call was removed. Paste the comparison and state which branch was taken.

> **STOP GATE 1.5a (before spending re-eval calls).** Paste: the `evaluator_agent.py` diff (flag logic plus the `_parse_review` disposition), the three named constants, the equivalence-guard comparison, and the unit assertion result. State which 1.5.2 branch was taken (parse removed, or parse retained plus deferral surfaced) so the re-eval's isolation is on record. Confirm the GatekeeperAgent is UNCHANGED (ADR-017 holds RC-2 for post-fix observation, not a tonight-fix). Keyword required before the re-run.

1.5.3 **Re-evaluate, in-domain only.** 14 in-domain queries times both leaders times 3 passes = 84 records. OOD is excluded per ADR-017 Quantified Validation: OOD groundedness sits at 0.33 to 0.48 and cannot flip on a flag-threshold fix, and the OOD gate already passed (6/6 fallback, 0 hallucinations). Reuse the existing harness; the scope is smaller than the Phase 1 full run, so no new cost-guard pre-flight is needed. Paste the call-count estimate before running.

1.5.4 **Read the GatekeeperAgent disposition off the re-eval (ADR-017).** For every record now scoring `groundedness_score >= 0.60` with NO `low_groundedness` flag, did the Gatekeeper DELIVER?
   - If all such records deliver: RC-2 was masked by RC-1, not independent. Tabulate the new in-domain deliver rate against the floors (Torvalds 42.9%, Kroah-Hartman 35.7%) and the three-run variance.
   - If any such record still falls back: RC-2 is confirmed independent. STOP and surface. RC-2 then gets its own GatekeeperAgent change (numerical score comparison, not flag-presence routing) and its own re-eval. Do NOT fix RC-2 inside Phase 1.5.
   - Note whether RC-3 (the q14-style fabricated `trigger_category`) recurs; it is expected to resolve with RC-2 per ADR-017.

> **STOP GATE 1.5b (Phase Defence plus re-gate decision).** Paste the new in-domain 2x2 grid per leader, the per-leader deliver rate against its floor, the three-run variance, the flag-clean deliver check (1.5.4), and the RC-2 read. Render the four-category Phase Defence on Phase 1.5 (including the mandatory Category V v1-drift check). Ruby records the new gate decision. The blocked Phase 2 starts ONLY if the floors clear AND RC-2 is resolved (either it delivered, or it was separately fixed and re-measured). Keyword required.

---

## Phase 2 — Refactor and retire (gated; destructive; gated-not-started)

Blocked behind a passing re-gate at STOP GATE 1.5b, not merely STOP GATE 1. It does not start unless the ADR-015 floors clear AND RC-2 is resolved (delivered, or separately fixed and re-measured). Planned here, not started. Re-read this Phase 2 section and `src/cli.py`, `src/visualization.py`, `src/flow.py`, `src/schemas.py` from disk before starting.

1. **Refactor `src/cli.py` and `src/visualization.py` to v2 field names (D-B1).** `cli.py` reads `flow.state.final_output`, `FallbackResponse.trigger_reason`/`context_summary`, and `EvaluationResult.final_score`, none of which exist in v2; the `query` and `compare` commands and the `evaluate` command all break on the v2 schema. Rewire to the v2 shape: read `flow.state.styled_response`/`flow.state.fallback_response`, use the v2 `FallbackResponse` fields (`acknowledgment`, `suggested_redirections`, `calendar_link`, `available_slots`), drop every `final_score` print and the `0.75` threshold reference in the `query` docstring. `visualization.py` drops the `final_score` series and the `0.75`/`0.60` threshold lines that encode the removed formula. This unblocks the three retirements and re-enables the two skipped test files. Verification: `grep -n "final_score\|final_output\|0\.75\|context_summary\|trigger_reason" src/cli.py src/visualization.py` returns only intentional matches (none expected), and `cli query` plus `cli compare` run end-to-end on one query each.

2. **Enhance `cli evaluate` output, reusing the Phase 1 harness.** Wire `cli evaluate` to call `src/eval/harness.py` (per-stage latency, the 2x2 grid, the PRD scorecard, §7.5) rather than rebuilding what Phase 1 produced. Default the `--queries` path to `data/eval/queries.json` (currently `queries_v1.json`). Verification: `cli evaluate` writes a results JSON matching the harness schema and prints the grid.

3. **Re-enable the two skipped test files.** Remove the v1-field-name `pytestmark` skips from `tests/test_cli.py` and `tests/test_visualization.py` and update their assertions to the v2 shape. Verification: both files collect and pass; `pytest tests/test_cli.py tests/test_visualization.py -q` is green.

### STOP GATE 2 (before deletions; destructive, multi-file)

The next step deletes three source targets. Paste a per-file grep proving zero live importers for each, before any delete. Keyword required.

4. **Retire, each grep-confirmed zero live importers first.**
   - `src/agents/rag_agent.py` (current blocker: `cli.py:16` imports `RAGAgent`; resolved by item 1). Confirm `index` no longer imports it before delete.
   - `src/evaluation/evaluator.py` (current blocker: `src/evaluation/__init__.py:4` re-exports `evaluate`; remove that re-export in the same step), and `tests/test_evaluator.py` which exercises it.
   - `reranker.rerank()` (blockers: `src/rag/__init__.py:8` plus seven experiment scripts). Audit the scripts; `reranker.py` also exports `rerank_with_status`, which `Retriever` uses, so do not delete the file. If any script still calls `rerank()`, surface it; do not force-delete. Verification: per-file grep pasted showing zero live importers; suite green after each delete.

5. **Sync to the Notion ADR Log.** The ADR-010 `trigger_category` amendment (logged Day 11, not yet synced) and the ADR-015 amendment from Phase 1 item 7. Verification: both appear in the Notion ADR Log database.

### STOP GATE 3 (Phase Defence plus session close)

Full suite green, accounting for the pre-existing `tests/test_query_loader.py::test_load_queries_canonical_file` failure (it reads `data/eval/queries_v1.json`, unrelated to this work). Plan-diff: DONE / SKIPPED / PARTIAL per numbered item with file:line refs. Write session notes to `docs/session-notes/day12.md`. Render the Phase Defence on Phase 2 (including Category V).

---

## Plan discipline

- **PRD Coverage Check.** Done above. The one re-scope (`cli evaluate` moved from Phase 1 to Phase 2, with a harness standing in for Phase 1 measurement) is documented, not silent. `evaluation-methodology.md` is pulled in from §7.5.1 since PRD §8 prose omits it.
- **Re-read from disk at phase start.** Each phase opens by re-reading its own section of this plan and the listed source files from disk, not from context. The file on disk is authoritative (Prompt Discipline Component 4).
- **Surface, do not silently choose.** The four plan-review conflicts plus the added requirement (C1 to C5) are resolved and recorded in Resolved decisions. Any further conflict, a needed sixth `trigger_category`, an OOD-category miss outside the named sets, or an SLA-edit recommendation stops and surfaces rather than being applied.
- **No new ADRs invented by Sonnet.** The two amendments (ADR-010 already logged Day 11, ADR-015 new in item 7) are logged blocks, surfaced, not silent rewrites of a Decision section.
- **ADR-017 is the citation for Phase 1.5.** It is authored and logged to the Notion ADR Log; this plan references it and does not reproduce its reasoning. Phase 1.5's steps execute ADR-017's Decision and cite its Quantified Validation; they do not re-derive the root cause.

## Verb-and-count audit

Every imperative has a file target and a verification step.

| Item | Verb | File target | Verification |
|------|------|-------------|--------------|
| P1 pre-flight | run | `src/eval/harness.py` (q01, q15) | two timing dicts + two routing decisions + one-retrieval assertion pasted |
| P1.1 | create | `src/eval/harness.py`, `results/evaluation_day12.json` | ~96 records (40 full + 28 + 28 in-domain re-runs + any OOD recheck); response text + routing reasoning present per record (C5); one full timing dict printed |
| P1.2 | build | harness output (in-memory + report) | both grids from full run printed, counts sum to 14+6 per leader, in/OOD keyed off `category` |
| P1.3 | assert | harness assertion (no schema validator) | assertion result + category distribution pasted |
| P1.4 | read | harness timing summary | per-stage latency with generate/parse split pasted |
| P1.5 | evaluate | gate logic in harness | one full grid per leader, in-domain deliver-rate range across 3 runs, OOD fallback count, category-5 count, ADR-013 delta |
| P1.6 | author | `docs/day11-evaluation.md`, `docs/evaluation-methodology.md` | both exist; gate decision cites ADR-015 + amendment |
| P1.7 | append | `docs/adr/ADR-015-...md` | dated amendment block present; original numbers untouched |
| P1.5.1 | implement | `src/agents/evaluator_agent.py` | unit assertion (no record with groundedness >= 0.60 carries low_groundedness) + three named constants with file:line pasted |
| P1.5.2 | remove (conditional) | `src/agents/evaluator_agent.py` (`_parse_review`) | equivalence-guard comparison (`.raw` vs old parsed explanation, 2-3 records) pasted; branch stated (equivalent: remove parse; rewording: retain parse + surface deferral); both branches continue to 1.5.3 with the flag fix isolated |
| P1.5.3 | re-evaluate | `src/eval/harness.py`, `results/` (in-domain only, 84 records) | call-count estimate pasted before run; 84 in-domain records produced |
| P1.5.4 | read | re-eval results (GatekeeperAgent disposition) | flag-clean deliver check tabulated; RC-2 read (all-deliver vs any-fallback) + RC-3 recurrence note pasted |
| P2.1 | refactor | `src/cli.py`, `src/visualization.py` | v1-token grep clean; `query`/`compare` run |
| P2.2 | wire | `src/cli.py` (`evaluate`) | results JSON matches harness schema; grid printed |
| P2.3 | re-enable | `tests/test_cli.py`, `tests/test_visualization.py` | both collect and pass |
| P2.4 | delete | `src/agents/rag_agent.py`, `src/evaluation/evaluator.py` (+ `__init__` re-export, `tests/test_evaluator.py`); audit `reranker.rerank()` | per-file zero-importer grep pasted; suite green |
| P2.5 | sync | Notion ADR Log | both amendments present in the database |
