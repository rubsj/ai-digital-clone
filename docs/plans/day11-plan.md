# Day 11 — Remaining Agents and Flow Integration

> **Plan-file authority (Rule 7):** Re-read `docs/plans/day11-plan.md` from disk at the start of every phase. The file on disk is authoritative over any version in context.

## Context

Day 10 built 3 Components (Retriever, StyleProfileBuilder, ScoringEngine) and 2 Agents (CloneAgent, EvaluatorAgent), reshaped `EvaluationResult`, and left v1 `flow.py` plus its dead code on disk (Dead Code Ledger in `docs/session-notes/day10.md`). Day 11 finishes the agent roster and turns the v1 Flow shell into the v2 pipeline: the `@router` step stops returning `evaluation.decision` and starts calling a real GatekeeperAgent; the fallback branch stops calling `build_fallback_response()` and starts calling a real FallbackAgent (ADR-010, ADR-012). Retrieval stays shared across leaders (ADR-005).

The day is split into two phases with a **hard stop gate between them**. Phase A builds the two new Agents and their schemas in isolation, with no Flow wiring — every Day-10 unit test stays green and the v1 Flow keeps running unchanged. Phase B does the rewire, the latency instrumentation, and the end-to-end smoke, then retires v1 dead code **last**, only after the smoke is green.

**Scope (PRD §8 Day 11):** GatekeeperAgent, FallbackAgent (+ templated failsafe), `RoutingDecision` model + `CloneState.routing_decision` field, `src/flow.py` rewire, `compare_leaders()` shared-retrieval preservation, per-Agent integration tests, end-of-day smoke (one query end-to-end).

**Out of scope (Day 12+), must not appear:** `cli evaluate` run against the 20-query set (Day 12); `docs/day11-evaluation.md` (Day 12); the 2×2 routing-correctness grid and 8 charts (Day 12/13); Streamlit work (Day 13); per-leader feature weighting (ADR-013 contingency, Day 12+); any StyleProfileBuilder / §4.8 / 15-feature edit (frozen, ADR-013); any `final_score` field or threshold comparison (ADR-010, Architecture Rule 3).

**Estimated duration:** ~6–8 working hours across 2 phases.
**Tooling:** Python 3.12+, `uv run`, `ruff` (line 100, double quotes). `mypy` is still **not** a declared dependency (`pyproject.toml` has `pytest`, `pytest-cov`, `ruff` only) — aspired-to, not run; installing it trips Stop Gate 2. Same documented deviation as Day 10.

### Confirmed decisions (derived from PRD/ADR; not open)

1. **Routing is a real Agent.** The Flow `@router` step calls `GatekeeperAgent().run(...)` and returns `routing_decision.decision` (`"deliver"` | `"fallback"`). No weighted formula, no threshold, no `final_score` (ADR-010, Rule 3).
2. **Fallback is a real Agent with a failsafe.** FallbackAgent is a CrewAI Agent generating leader-voice acknowledgment + 2–3 redirections + calendar mock; a try/except around the LLM call returns a templated `FallbackResponse` on any LLM failure (ADR-012, PRD §5.1.4).
3. **Both new Agents follow the Day-10 canonical pattern** — class wrapping a CrewAI `Agent` (role/goal/backstory) + single `Task` + single-agent `Crew`, with one Instructor parse for structured output (the kickoff → Instructor-parse pattern in `clone_agent.py` / `evaluator_agent.py`). GatekeeperAgent at `temperature=0` (deterministic routing, PRD §2.5); FallbackAgent at `temperature=0.3` (voice variation, PRD §5.1.4).
4. **Retrieve-once is preserved** — the `@start` retrieve step early-exits when `state.chunks` is already populated; `compare_leaders()` snapshots the first leader's chunks and injects them into the second run (ADR-005). The integration test asserts the second leader's Retriever is **never called**.
5. **Latency is instrumented, not asserted** (Day-11 delta #3). Per-stage timing is captured at the dict/orchestration layer (the Day-10 decision: latency is observability, never a result-model field) and emitted in the `cli evaluate` output shape. The §2.7 `<8s` budget is **not** asserted on Day 11 — Day 12 is the first real-LLM timing pass.

### Decisions to confirm before Sonnet executes (surfaced, not chosen)

- **D-A1 — FallbackResponse reshape timing.** PRD §5.1.4 / §5.4 define `FallbackResponse` as `acknowledgment`, `suggested_redirections`, `calendar_link`, `available_slots`, `unstyled_response`. The current schema is `trigger_reason`, `context_summary`, `calendar_link`, `available_slots`, `unstyled_response`. FallbackAgent (built in Phase A) must emit the v2 shape, so the reshape lands in Phase A. Reshaping breaks v1 `flow.py::handle_fallback` and `fallback_steps.py` **at runtime** (not import) — same blast pattern as Day 10's `EvaluationResult` change. Proposed handling: reshape in A, leave v1 callers on disk (already in the Ledger / `test_flow.py` already skipped), retire in B. **Confirm:** reshape `FallbackResponse` in place in Phase A (vs. introduce a parallel model and rename later).
- **D-A2 — `CloneState` reshape split across phases.** Phase A adds only the **additive** `routing_decision: RoutingDecision | None` field (safe, no v1 breakage). The fuller §5.4 reshape — rename `retrieved_chunks → chunks`, `styled_response: str → response_text: str`, add `citations`, `style_profile`, `styled_response: StyledResponse | None`, `fallback_response` — is destructive to v1 `flow.py` and belongs in Phase B's rewire (ADR-005 explicitly names the v2 field `CloneState.chunks`). **Confirm:** additive-only in A, full reshape in B.
- **D-B1 — `cli.py` / `visualization.py` deferral.** Both read v1 `final_output` / `final_score` and will break at runtime once `CloneState` and the Flow output reshape (Ledger rows already flag them). PRD §8 puts `cli evaluate` in **Day 12**, not Day 11, so the Day-11 smoke runs through the Flow's public API (`DigitalCloneFlow().kickoff(...)` / `compare_leaders(...)`) directly, **not** through `cli.py`. Proposed: skip `test_cli.py` / `test_visualization.py` with a Day-12 reason and add the cli/visualization refactor to Day 12 (before `cli evaluate` is first used) rather than silently deferring. **Confirm:** Day 11 does not touch `cli.py` / `visualization.py`; their refactor is explicit Day-12 scope.

---

## Phase A — New Agents + schemas, isolated (no Flow wiring)

Re-read this plan from disk before starting. **Nothing in Phase A imports or edits `src/flow.py`.** Every Day-10 unit test must stay green; the v1 Flow keeps running on the old shapes until Phase B.

### A1 — Schemas (`src/schemas.py`)

- **`RoutingDecision`** — NEW model (PRD §5.4): `decision: Literal["deliver", "fallback"]`, `reasoning: str` (the human-readable routing rationale, PRD §2.5), `trigger_reason: str | None = None` (free-text rationale on fallback, consumed by the shipped FallbackAgent prompt — **unchanged**), and **NEW** `trigger_category: Literal["low_groundedness", "off_domain", "hallucination_risk", "chunk_mismatch", "empty_retrieval"] | None = None` (bounded routing taxonomy, set iff `decision == "fallback"`). GatekeeperAgent's Instructor `response_model`.
  - **Why `trigger_category` exists (PRD-internal tension surfaced by the Gatekeeper output contract):** §5.4 specifies `trigger_reason` as free text, but §2.10 chart 6 (required) is the fallback-trigger **distribution**, which must group by a bounded set — free text will not group and post-hoc Day-13 bucketing is fragile. The fix is additive: keep free-text `trigger_reason` for the FallbackAgent prompt and human rationale; add bounded `trigger_category` for distribution analysis. Chart 6 (Day 13) groups by `trigger_category`.
  - **Vocabulary** (the 5 literals above) finalized against the A3 trade-offs (`low_groundedness`, `hallucination_risk`, `chunk_mismatch` map directly to the role/goal/backstory trade-offs; `off_domain` / `empty_retrieval` cover the no-usable-evidence cases) and the §6.5 error table. Kept deliberately small — extend only with a logged amendment, not ad hoc.
  - **"Set iff fallback" is enforced at the prompt + A3 contract-test layer, not as a schema cross-field validator** — symmetric with the existing `trigger_reason` handling. `RoutingDecision` stays permissive (`trigger_category` is `Literal[...] | None`); the A3 contract test asserts the iff-relationship.
  - **ADR-010 amendment (Day 11):** routing output carries a bounded `trigger_category` (fixed 5-literal set) for the §2.10 fallback-trigger distribution; free-text `trigger_reason` is retained for the FallbackAgent prompt and human-readable rationale. No change to the no-formula / no-`final_score` core of ADR-010. *(Sync to Notion ADR Log per `reference_notion_adr_log.md`.)*
- **`FallbackResponse`** — reshape to PRD §5.1.4 (see D-A1): `acknowledgment: str`, `suggested_redirections: list[str] = []`, `calendar_link: str`, `available_slots: list[str] = []`, `unstyled_response: str` (failsafe backup; keep populated so the system always returns usable prose). This reshape breaks v1 `flow.py` / `fallback_steps.py` construction at runtime — confirm v1 callers stay on disk and any v1 test exercising the old shape is skipped, not edited.
- **`CloneState`** — add only `routing_decision: RoutingDecision | None = None` (additive, see D-A2). Do **not** rename `retrieved_chunks` or change `styled_response` here — that is Phase B.
- Do **not** add `final_score` anywhere; do **not** add a latency field to any model (latency is captured at the orchestration layer, Day-10 decision).

**Test focus:** `RoutingDecision` constructs and rejects an invalid `decision` literal; `trigger_reason` optional; `trigger_category` defaults `None`, accepts each of the 5 valid literals, and rejects an out-of-set value (e.g. `"weird_reason"` raises `ValidationError`). `FallbackResponse` constructs with the 5 v2 fields. `CloneState` carries `routing_decision` (and therefore `trigger_category` rides along automatically). Update `tests/test_schemas.py` for both new/reshaped models (this file **churns** — see Phase B test-churn ledger).

### A2 — FallbackAgent (`src/agents/fallback_agent.py`)

- *Rewrite* as a real CrewAI Agent (PRD §5.1.4, ADR-012). Inputs: `query`, `leader`, `trigger_reason`, `style_profile`, `chunks`. Output: `FallbackResponse`. Follow the canonical pattern (role/goal/backstory + Task + single-agent Crew + one Instructor parse). The role/goal/backstory encode leader-appropriate fallback voice (terse Torvalds vs. measured Kroah-Hartman) and that honesty about the limitation is the point. Suggested redirections are inferred from the retrieved `chunks` when adjacent topics exist (PRD §2.6). Calendar mock (`calendar_link` + 3 `available_slots`) is always generated — the existing `src/fallback/calendar_mock.py` may be reused (it is LLM-free helper code, allowed from an Agent).
- **Templated failsafe (ADR-012):** wrap the LLM call in try/except. On any LLM failure (timeout, rate limit, Instructor validation failure), return a templated `FallbackResponse` with the leader name substituted and the same calendar mock data, so the system always returns a usable response. The "5-line try/except" framing in ADR-012 is **descriptive of the shape, not a literal line-count target** — what matters is the behavior, not the line count (see verb-and-count audit below).
- `src/agents/fallback_steps.py` (v1 function) is **not** edited — it goes to the Ledger and is retired in Phase B.

**Test focus (`tests/integration/test_fallback_agent.py`) — contract test, not behavior.** With the LLM mocked, any assertion on the *content* of the acknowledgment or redirections only re-asserts the fixture the mock supplied, not the Agent's behavior. Frame the assertions as plumbing:
- **Inputs reach the LLM:** assert the Task/prompt the Agent builds contains the `query`, `trigger_reason`, and the chunk text (so redirection context and trigger context actually flow into the prompt).
- **Output parses to a valid `FallbackResponse`:** the mocked LLM output structures into the v2 shape — non-empty `acknowledgment`, `suggested_redirections` is a list, `calendar_link` + 3 `available_slots` present (the calendar mock is the deterministic `calendar_mock.py` helper, so its presence **is** real, not mock-supplied), `unstyled_response` populated.
- **Failsafe test (Day-11 delta #2, real behavior — leave as-is):** mock the LLM call to raise, assert a templated `FallbackResponse` is returned (calendar mock intact, leader name substituted), no exception escapes `run()`.

"Recorded responses" means fixture strings + monkeypatched kickoff/Instructor (no VCR library in `pyproject.toml`, see verb-and-count audit). **Leader-voice quality, redirection relevance, and trigger-appropriate tone are real-LLM behavior — measured Day 12 (PRD §2.6), not provable under a mock.** A green A2 suite proves plumbing, not fallback quality.

### >> INTRA-PHASE-A STOP GATE (before GatekeeperAgent) <<

Stop and wait for Ruby's explicit "continue." GatekeeperAgent is the load-bearing routing decision (ADR-010) and **Day 12's evaluation depends entirely on it** — Ruby reviews the schema, the role/goal/backstory, and the input contract before it is built. Report: A1 schema diff, A2 FallbackAgent file:line + failsafe test result, Day-10 suite still green.

### A3 — GatekeeperAgent (`src/agents/gatekeeper_agent.py`)

- *Create fresh* as a real CrewAI Agent (PRD §5.1.3, ADR-010). Inputs: `query`, `response_text`, `chunks`, `evaluation` (the `EvaluationResult` — three scores + explanation + flags), `leader`. Output: `RoutingDecision`. Canonical pattern, `temperature=0`. The role/goal/backstory encode the trade-offs to watch (high style + low groundedness = confident-hallucination risk → fallback; flag indicating chunk mismatch → fallback; permissive by default, conservative on specific flags). The prompt **must demand the `reasoning` reference specific scores and flags** (PRD §2.5, ADR-010 risk mitigation). When `decision == "fallback"`, the prompt **must also emit a `trigger_category` from the fixed 5-literal set** (`low_groundedness` / `off_domain` / `hallucination_risk` / `chunk_mismatch` / `empty_retrieval`) — the bounded label powers the §2.10 chart-6 distribution; `trigger_reason` stays the free-text rationale referencing the specific scores/flags. On `deliver`, both `trigger_reason` and `trigger_category` are `None`. No threshold comparison, no `final_score` read (Rule 3).
- Single-query routing only; dual-leader cross-comparison is deferred (PRD §5.1.3).

**Test focus (`tests/integration/test_gatekeeper_agent.py`) — contract test, not routing behavior.** With the LLM mocked, "deterministic at `temperature=0`", "OOD routes to fallback", and "reasoning references the scores" are vacuous or just assert the fixture — none of them exercise the Agent's actual judgement. Frame the assertions as plumbing:
- **Inputs reach the LLM:** assert the Task/prompt the Agent builds contains the three `EvaluationResult` scores and the `flags` it was given (so routing evidence actually flows into the prompt, ADR-010).
- **Output parses to a valid `RoutingDecision`:** the mocked LLM output structures into a valid `decision` literal (`"deliver"`/`"fallback"`), a non-empty `reasoning`, and `trigger_reason` set when the decision is `"fallback"` / `None` when `"deliver"`.
- **`trigger_category` contract:** when the mocked output is `fallback`, `trigger_category` is one of the 5 literals (parses to the `Literal` type); when `deliver`, `trigger_category is None`. This asserts the iff-relationship at the contract layer (the schema does not enforce it). Whether the *chosen* category is the correct one for the inputs is real-LLM behavior (Day 12).
- `temperature=0` is set on the LLM config (assert the config value, not behavioral determinism).

**Moved to Day 12 as real-LLM measurements (PRD §2.1/§2.5):** determinism at `temperature=0` (same inputs → same decision across runs), in-domain deliver vs. OOD fallback routing correctness, category-5 hallucination detection (100% OOD → fallback), and **`trigger_category` correctness** (whether the Gatekeeper labels the *right* category for given scores/flags — the contract test only proves it picks one of the legal literals). These are routing *behavior* and cannot be proven under a mock. **A green A3 suite proves plumbing, not routing behavior** — do not read it as evidence the Gatekeeper routes correctly.

### Phase A exit gate (STOP gate + Phase Defence)

- `RoutingDecision` and reshaped `FallbackResponse` exist; `CloneState.routing_decision` added; no `final_score` anywhere new; `tests/test_schemas.py` updated and green.
- `src/agents/fallback_agent.py` and `src/agents/gatekeeper_agent.py` exist; both are real CrewAI Agents (`from crewai import Agent`, role/goal/backstory present).
- Per-Agent integration tests pass (LLM mocked); FallbackAgent failsafe test passes; GatekeeperAgent OOD-fallback test passes.
- **Architecture-honesty greps pass** (CLAUDE.md set): 4 Agent files now carry `from crewai import Agent` + role/goal/backstory; Components still import no `litellm|openai|cohere|instructor`; no Agent-suffixed functions outside `src/agents/`; `final_score` only in the `schemas.py` deprecation docstring + expected v1-leak files (Ledger).
- **No `src/flow.py` edit in Phase A** — confirm `git diff --stat` shows `flow.py` untouched.
- **Phase Defence (Teach-Back Component 2 + Category V).** Sonnet's status report must prompt Ruby for the Phase A defence and **stop for Ruby's answer** before Phase B. Category V (v1-drift, mandatory for Agent phases): did any new code introduce a weighted formula, threshold, `final_score`, or a Python function named like an Agent? If clean, say so with the grep that proves it.

### >> HARD STOP GATE: PHASE A → PHASE B <<

Do not begin the Flow rewire until Ruby says "continue." Phase B edits `flow.py` and reshapes `CloneState` (multi-file, breaks v1 construction) — both are stop-gate triggers (destructive / multi-file / architecture-touching).

---

## Phase B — Flow integration, latency, smoke, retire v1 last

Re-read this plan from disk before starting. Order: pin the groundedness gate → reshape `CloneState` + rewire `flow.py` → instrument latency → smoke (both arms) → **retire v1 dead code only after smoke is green.**

### B0 — Pin the groundedness embedding path from source (gate, Day-11 delta #2)

Before wiring `evaluate`, confirm from source whether `ScoringEngine.score()` groundedness **reads chunk-carried embeddings or re-embeds chunk text**, because Day-10 tests used synthetic chunks and may have masked the live path. The plan's pinned reading (verify it still holds, surface if it has changed):

- `score_groundedness` (`src/evaluation/groundedness_scorer.py`) reuses `rr.chunk.embedding` **if present**, otherwise batch re-embeds the missing chunks' **text** via `embed_openai`.
- `Retriever.run()` returns chunks reconstructed by `src/rag/retriever.py::retrieve()` as `KnowledgeChunk(**metadata[idx])`, and `src/rag/indexer.py::build_index()` builds metadata with `model_dump(exclude={"embedding"})` — **embeddings are excluded from `metadata.json` by design** (the vectors live in the FAISS index).
- **Conclusion to confirm:** against the real `Retriever.run()`, every `RetrievalResult.chunk.embedding is None`, so groundedness **always re-embeds the 5 chunk texts** (MD5-cached, warm after first call). It is not a correctness bug (same text → same vector), but it means: (a) a repeated OpenAI embed call lands **inside the `evaluate` stage** and will show up in that stage's latency, and (b) the groundedness score is computed against freshly-embedded chunk text, not the index vectors. If source no longer matches this, **STOP and surface** rather than wiring on a stale assumption.
- **Coverage gap to close (do not leave the live branch untested):** if `tests/test_groundedness_scorer.py` builds synthetic chunks that carry `.embedding`, it covers the **reuse** branch the live pipeline never hits and leaves the **always-taken re-embed branch uncovered**. Add one test that passes chunks with `chunk.embedding is None` and asserts the re-embed path runs (`embed_openai` invoked on the chunk texts, score still in `[0,1]`). The file stays on the B4 stay-green list; this is one added test, not a rewrite of the frozen scorer. If adding it now risks the freeze or runs long, **record it as a named Day-12 coverage follow-up** in the session notes — but do not silently ship the live branch untested.

### B1 — `CloneState` reshape + `flow.py` rewire

- **`CloneState`** (the §5.4 reshape deferred from A2): `chunks` (rename from `retrieved_chunks`, per ADR-005), `style_profile: StyleProfile | None`, `response_text: str | None`, `citations: list[Citation]`, `evaluation`, `routing_decision` (from A1), `styled_response: StyledResponse | None`, `fallback_response: FallbackResponse | None`. This breaks v1 `flow.py` construction — that is expected; `flow.py` is rewired in the same phase.
- **`DigitalCloneFlow`** — rewire each step to the v2 pipeline (PRD §5.5, §6.1). "Rewire" here means **edit the existing Flow class in place** (keep the `Flow[CloneState]` shell, the `@start/@listen/@router/@listen` decorator topology, and the dual-leader early-exit), and **swap what each step calls** — not a from-scratch rewrite:
  - `@start retrieve`: early-exit if `state.chunks` populated (ADR-005); else `state.chunks = Retriever().run(state.query)`.
  - `@listen clone`: `CloneAgent(...).run(query, leader, style_profile, chunks)` → set `state.response_text`, `state.citations`.
  - `@listen evaluate`: `EvaluatorAgent().run(query, response_text, profile, chunks)` → `state.evaluation`.
  - `@router route`: `state.routing_decision = GatekeeperAgent().run(query, response_text, chunks, evaluation, leader)`; `return state.routing_decision.decision`. This **replaces** the v1 `return self.state.evaluation.decision` (the field no longer exists).
  - `@listen("deliver")`: assemble `StyledResponse` into `state.styled_response`.
  - `@listen("fallback")`: `FallbackAgent().run(query, leader, trigger_reason, style_profile, chunks)` → `state.fallback_response`; failsafe is inside the Agent (B-A2).
- **Profile loading (one mechanism for Day 11):** the **caller injects `style_profile` via `kickoff(inputs=...)`** for both single-query and `compare_leaders()`. **The Flow has no profile-load step** — `retrieve` reads `state.style_profile` as already populated. `compare_leaders()` loads each leader's profile and injects it (alongside the shared chunks for leader 2); the Day-11 smoke (B3) cannot use `cli.py` per D-B1 and injects directly anyway, so a single inject path covers every Day-11 entry point. An in-Flow convenience load (so `cli query` need not pre-load) is **deferred to Day 12 with the `cli.py` refactor** — do not add it now.
- **`compare_leaders()`** — keep the ADR-005 shape (sequential runs, snapshot first leader's chunks, inject into the second via `kickoff(inputs={"chunks": ...})`). Update field name `retrieved_chunks → chunks`. Surface both leaders' outcomes faithfully (asymmetric deliver/fallback is expected, ADR-005).

### B2 — Latency instrumentation (Day-11 delta #3)

- Instrument **per-stage** wall-clock timing (retrieve / clone / evaluate / route / deliver|fallback) at the orchestration/dict layer — **not** as a Pydantic field (Day-10 decision). Emit it in the structure that `cli evaluate` will serialize (PRD §6.1 per-stage budgets, ADR-016). **Do not assert the §2.7 `<8s` budget** — Day 12 is the first real-LLM timing pass. Note that the `evaluate` stage timing includes the groundedness chunk re-embed (B0).

### B3 — Smoke (both @router arms)

- Drive through the Flow's public API directly (`DigitalCloneFlow().kickoff(...)` / `compare_leaders(...)`), not `cli.py` (D-B1). **Run at least one deliver-path query and one fallback-path query** so both `@router` arms execute (Day-11 delta #2). Confirm each step's output is the expected Pydantic type and one query traces through all 5 Flow steps (PRD §8 end-of-day check).
- **`compare_leaders` integration test (pass/fail gate, Day-11 delta #2): assert the second leader's Retriever is called exactly 0 times** (spy/mock on the Retriever Component). Note for the defence: this asserts the *Retriever Component* runs 0 times for leader 2 — it does **not** mean zero embedding calls, because the per-leader `evaluate` stage still re-embeds chunk text for groundedness (B0). Keep the two distinct in the test name and assertion message.

### B4 — Test churn vs. must-stay-green (Day-11 delta #2: name them)

- **Churn (rewritten / re-enabled this phase):**
  - `tests/test_flow.py` — re-enable (remove the Day-10 module-level skip) and rewrite to the v2 Flow: 5 steps in order, `@router` branches on `deliver`/`fallback`, `CloneState` typed throughout, `compare_leaders` shares chunks (Retriever-call-count = 0 for leader 2).
  - `tests/test_schemas.py` — already updated in A1 for `RoutingDecision` / reshaped `FallbackResponse` / `CloneState.routing_decision`; the `CloneState` field rename in B1 churns it again.
  - **New:** `tests/integration/` (create the dir — it does not exist) holding the per-Agent contract tests (`test_fallback_agent.py`, `test_gatekeeper_agent.py` from Phase A) and `test_compare_leaders.py`.
- **Removed with retired v1 code (B5, after smoke green):** `tests/test_style_crew.py` (with `style_crew.py`), `tests/test_evaluator.py` (with `evaluation/evaluator.py`).
- **Skipped, refactor deferred to Day 12 (D-B1):** `tests/test_cli.py`, `tests/test_visualization.py` — both read v1 `final_output`/`final_score` and break after the reshape; skip with a Day-12 reason, do not edit logic.
- **Must stay green (do not touch):** all Day-10 unit tests (`test_clone_agent.py`, `test_evaluator_agent.py`, `test_components_*.py`) and the frozen-logic suite (`test_email_parser.py`, `test_feature_extractor.py`, `test_profile_builder.py`, `test_groundedness_scorer.py`, `test_style_scorer.py`, `test_confidence_scorer.py`, `test_reranker.py`, `test_retriever.py`, `test_embedder.py`, `test_indexer.py`, `test_chunker.py`, `test_corpus_loader.py`, `test_citation_extractor.py`, `test_calendar_mock.py`, `test_context_summarizer.py`, `test_unstyled_responder.py`, `test_config.py`). Pre-existing unrelated failure `test_query_loader.py::test_load_queries_canonical_file` stays as-is (out of scope, missing data file).

### B5 — Retire v1 dead code (LAST, only after smoke is green)

After B3 smoke passes and B4 churn is green, retire the Ledger rows now superseded by the live v2 Flow. **"Retire" means physically delete the file** (and its dedicated test), not just stop importing it — confirm via grep that nothing live imports each file before deleting. This is a destructive, multi-file step → it sits behind its own stop gate (Stop Gate 1).

Ledger rows triggered by the Day-11 Flow refactor (delete now that the live Flow no longer references them — verify per row):
- `src/agents/rag_agent.py` (Flow now imports `Retriever`) + any test exercising it.
- `src/agents/style_crew.py` + `tests/test_style_crew.py` (Flow now calls `CloneAgent`).
- `src/agents/evaluator_steps.py` + `src/evaluation/evaluator.py` + `tests/test_evaluator.py` (Flow now calls `EvaluatorAgent`; weighted formula gone).
- `src/agents/fallback_steps.py` (Flow now calls `FallbackAgent`).
- `src/rag/reranker.py::rerank()` thin wrapper — delete only if no live caller remains; otherwise keep and re-flag.
- Update PRD §12.2 mapping rows from "retire when…" to "retired (Day 11)."

### Phase B exit gate (STOP gate + Phase Defence)

- One query traces through all 5 Flow steps; deliver-arm and fallback-arm smokes both pass; each Agent's output is the expected Pydantic type (PRD §8 end-of-day check).
- `compare_leaders` integration test green, **second-leader Retriever-call-count = 0** asserted.
- Per-stage latency emitted in the `cli evaluate` output shape; §2.7 budget **not** asserted (recorded for Day 12).
- Full suite: Day-10 unit + frozen-logic green; `test_flow.py` re-enabled and green; `test_cli.py`/`test_visualization.py` skipped with Day-12 reason; v1 Ledger files + their tests deleted.
- **Architecture-honesty greps pass** (full CLAUDE.md set): 4 real Agents with role/goal/backstory, 3 Components with `run()`/`score()` and no LLM imports, no Agent-suffixed functions outside `src/agents/`, no `final_score` field, no threshold/weighted-formula in routing, adapter boundary intact.
- **Phase Defence (Teach-Back Component 2 + Category V).** Sonnet's status report must prompt Ruby for the Phase B defence and **stop for Ruby's answer**. Category V: confirm the rewire introduced no threshold/formula/`final_score`/Agent-named-function, with the grep that proves it; confirm the groundedness re-embed (B0) is the known, documented behavior, not a silent regression.

---

## Gates as pass/fail plan steps (Day-11 delta #2, collected)

1. **`compare_leaders` Retriever-call-count = 0** for the second leader (B3 integration test). Pass = mock asserts 0 calls.
2. **Groundedness embedding path pinned from source** (B0) before wiring `evaluate`. Pass = source matches the documented re-embed conclusion, or the deviation is surfaced to Ruby.
3. **Smoke covers both @router arms** (B3) — one deliver query, one fallback query. Pass = both arms execute and each step output is the expected type.
4. **FallbackAgent failsafe** (A2 test) — mock the LLM to raise, assert a templated `FallbackResponse`. Pass = no exception escapes, calendar mock intact, leader substituted.
5. **Test-file churn vs. stay-green named** (B4) — the named churn/stay-green/skip/delete lists hold after the rewire. Pass = suite matches the B4 ledger (no unexpected file in any bucket).

---

## Verb-and-count audit (seeded with Day-10 archetypes)

Applied to this plan's own language, per the three Day-10 archetypes. Each entry pins the intended reading so Sonnet cannot take the wrong one.

**Archetype 1 — aspirational tooling** (Day 10: `mypy` listed but not in `pyproject.toml`).
- `mypy` — aspired-to, **not run** on Day 11 (not a declared dependency; running it trips Stop Gate 2). Active toolchain is `ruff` only.
- "integration tests on **recorded LLM responses**" — there is **no VCR/cassette library** in `pyproject.toml`. "Recorded responses" means fixture strings + monkeypatched `Crew.kickoff()`/Instructor, the Day-10 mock pattern. Do not add a recording dependency.
- `tests/integration/` and `tests/e2e/` — named in PRD §7.2 but **do not exist on disk** (all tests are currently flat in `tests/`). Phase B **creates** `tests/integration/`; `tests/e2e/` is Day-12 (`cli evaluate`), not Day 11.
- **"contract test" (A2/A3, amended)** — means a **plumbing-assertion test** (inputs reach the built prompt; mocked output parses to the right Pydantic type). It does **not** mean a consumer-contract / Pact-style framework — no such dependency exists or is added.

**Archetype 2 — two-reading verbs** (Day 10: "rename" = `git mv` vs. "create fresh, leave on disk"; "ONE LLM call" = single Instructor call vs. kickoff + parse).
- **"rewire `flow.py`"** → edit the existing Flow class **in place** (keep shell + decorator topology), swap what each step calls. **Not** a from-scratch rewrite.
- **"retire v1 dead code"** → **physically delete** the file + its dedicated test (after grep-confirming no live import), **B5 only, after smoke green**. Not "stop importing but leave on disk."
- **"create fresh" (GatekeeperAgent)** vs. **"rewrite" (FallbackAgent)** — GatekeeperAgent has no v1 predecessor file, so it is net-new; FallbackAgent replaces v1 `fallback_steps.py` (left on disk until B5). Neither is `git mv`.
- **"reshape `FallbackResponse` / `CloneState`"** → change fields in place; this **breaks v1 construction at runtime** (not import), handled by skip + Ledger, same as Day-10's `EvaluationResult`.
- **"inject `style_profile`" (B1, amended)** → pass it via `kickoff(inputs=...)`; the Flow has **no profile-load step** on Day 11. Not "load inside the Flow" — that convenience is Day-12 with `cli.py`.
- **"shared retrieval" / "retrieve-once"** → the *Retriever Component* runs once; per-leader `evaluate` **still re-embeds chunk text** for groundedness (B0). The two are not the same "embedding" — kept distinct in the gate-1 assertion.

**Archetype 3 — count-as-literal-implementation** (Day 10: "ONE LLM call" read as a literal single-call directive).
- **"5-line try/except" (failsafe)** — descriptive of shape (ADR-012 wording), **not** a literal line-count target. The test asserts behavior (templated `FallbackResponse` on raise), not line count.
- **"Retriever-call-count = 0"** — this count **is literal**: the integration test asserts exactly 0 Retriever-Component invocations for leader 2.
- **"temperature=0" (Gatekeeper) / "0.3" (Fallback)** — literal config asserts (PRD §2.5 / §5.1.4).
- **"all 5 Flow steps" / "both @router arms"** — literal coverage asserts in the smoke.
- **"4 Agents + 3 Components"** — descriptive architecture invariant enforced by the honesty greps, not a thing to construct.
- **"2–3 redirections", "3 time slots", "top-5 chunks"** — domain targets (PRD §2.6 / §5.1.4), satisfied by the Agent prompt + helpers; tests assert presence/range, not an exact literal where the PRD gives a band ("2–3").
- **"`trigger_category` 5-literal set" (A1/A3 amendment)** — the **count and the members are literal**: a closed `Literal[...]` of exactly those 5 strings, asserted by the schema test (each accepted, an out-of-set value rejected). It is **not** a Pact-style enum table, a DB-backed taxonomy, or a separate model — just an inline Pydantic `Literal` on `RoutingDecision`. **"set iff fallback"** reads as a *contract-test* guarantee (A3), **not** a schema cross-field validator — `RoutingDecision` stays permissive, symmetric with the shipped `trigger_reason`. **`trigger_category` is consumed only Day 13** (chart 6); Day 11 ships the field + its plumbing assertion, nothing reads it yet.

---

## PRD coverage check (PRD §8 Day 11)

| PRD §8 Day-11 deliverable | Phase | Covered |
|---|---|---|
| Build GatekeeperAgent | A3 | ✓ |
| Build FallbackAgent (+ templated failsafe) | A2 | ✓ |
| Update `CloneState` with `routing_decision` field | A1 (additive) + B1 (full §5.4 reshape) | ✓ |
| Refactor `src/flow.py` to call Agents at each step | B1 | ✓ |
| Update `compare_leaders()` for shared retrieval (ADR-005) | B1 + B3 test | ✓ |
| Integration tests per Agent | A2/A3 (+ B3 `compare_leaders`) | ✓ |
| End-of-day smoke (one query end-to-end) | B3 | ✓ |
| End-of-day check: 5 steps trace, expected Pydantic types, honesty check passes | B exit gate | ✓ |

**Brought forward beyond PRD §8 (Day-11 deltas, flagged):** per-stage latency instrumentation (delta #3 — PRD §2.7/§6.1 observability, emitted not asserted). **Surfaced as deferred, not silently dropped:** `cli.py`/`visualization.py` refactor (D-B1 → Day 12, before `cli evaluate`); `docs/day11-evaluation.md`, the 2×2 grid, the 8 charts, and the `cli evaluate` run (all PRD-assigned to Day 12/13, not Day 11). No PRD §8 Day-11 deliverable is left without a covering phase.

---

## End-of-day exit criteria (checklist)
- [ ] `RoutingDecision` added; `FallbackResponse` reshaped to §5.1.4; `CloneState` has `routing_decision` (A) and the full §5.4 shape (B); no `final_score` anywhere new.
- [ ] `gatekeeper_agent.py` + `fallback_agent.py` are real CrewAI Agents (role/goal/backstory); failsafe + OOD-fallback tests pass.
- [ ] `flow.py` rewired: `@router` calls GatekeeperAgent and returns `.decision`; fallback branch calls FallbackAgent; retrieve early-exits on injected chunks.
- [ ] `compare_leaders()` preserves ADR-005; integration test asserts second-leader Retriever-call-count = 0.
- [ ] Groundedness embedding path pinned from source (B0); deviation surfaced if any.
- [ ] Smoke: one deliver query + one fallback query trace all 5 steps; per-stage latency emitted (not asserted).
- [ ] Test churn matches the B4 ledger; `test_flow.py` re-enabled and green; v1 Ledger files deleted (after smoke green).
- [ ] Architecture-honesty greps pass (full set).
- [ ] Phase Defence rendered and answered for both phases (Sonnet stops for Ruby).
- [ ] No new ADR needed (else STOP and surface).

## Stop gates
1. Before deleting any v1 code under `src/` (B5 — default: leave in place until smoke green).
2. Before adding any `pyproject.toml` dependency (all needed deps already present; do not add VCR or mypy).
3. **Intra-Phase-A, before GatekeeperAgent** (ADR-010 load-bearing; Day 12 depends on it).
4. **Hard gate Phase A → Phase B** (Flow rewire + `CloneState` reshape: destructive + multi-file).
5. Before any architecture-rule conflict, any `final_score`/threshold/weighted-formula reintroduction, or any decision that would change a documented ADR → STOP and surface (do not silently choose).
