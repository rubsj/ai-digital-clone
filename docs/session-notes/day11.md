# Day 11 Session Notes

Plan: `docs/plans/day11-plan.md`. Scope: GatekeeperAgent + FallbackAgent (new), `RoutingDecision` schema, `FallbackResponse` reshape, `CloneState.routing_decision`, per-Agent integration tests. Two phases with a hard stop gate. **Phase A complete; Phase B not yet started — awaiting Phase A defence gate clearance.**

## Plan amendments (before and during execution)

Four amendments were made to `docs/plans/day11-plan.md` before or during Phase A execution. None touched the passed A1/A2 gates:

1. **A3 tests reframed as contract tests.** The original A3 test claims implied behavioral verification ("deterministic at temperature=0", "OOD routes to fallback", "reasoning references the scores"). These are all vacuous under a mocked LLM — the fixture proves itself. Reframed as plumbing: inputs (scores + flags) reach the Task description; output parses to a valid RoutingDecision; temperature=0 is set on the LLM config. Behavioral correctness (routing accuracy, determinism, category accuracy) moved explicitly to Day 12 as real-LLM measurements. This matches the A2 test contract already established.

2. **Profile-loading pinned to one mechanism.** The plan's B1 section had ambiguity about whether `style_profile` would be loaded inside the Flow or injected by the caller. Resolved to: the caller injects via `kickoff(inputs=...)` for both single-query and `compare_leaders()` paths on Day 11; the Flow has no profile-load step. An in-Flow convenience loader (needed for `cli query`) is deferred to Day 12 with the `cli.py` refactor.

3. **B0 groundedness coverage gap named.** The live `Retriever.run()` pipeline always returns chunks with `embedding=None` (the FAISS indexer stores vectors in the index and excludes them from `metadata.json`). This means `score_groundedness()` always takes the re-embed-chunk-text branch, never the reuse-embedding branch. Day 10 tests built synthetic chunks *with* embeddings, masking the live path. The always-taken branch is uncovered. Resolution: named as a Day-12 coverage follow-up (`test_groundedness_scorer: chunk.embedding=None re-embed path`). Recorded here rather than silently left; not added to Phase B to avoid scope creep.

4. **`trigger_category` added to `RoutingDecision` (PRD-internal tension).** The v1 `trigger_reason` field is free text, which cannot group into a distribution for the required fallback-trigger analysis chart. Adding a bounded five-literal `trigger_category` alongside free-text `trigger_reason` resolves the tension: `trigger_reason` feeds the FallbackAgent prompt and human rationale (unchanged), `trigger_category` powers chart-6 grouping (Day 13). The five literals (`low_groundedness`, `off_domain`, `hallucination_risk`, `chunk_mismatch`, `empty_retrieval`) were finalized against the GatekeeperAgent role/goal/backstory trade-offs. Enforced at the Gatekeeper-prompt + A3 contract-test layer (not as a schema cross-field validator), symmetric with `trigger_reason`. Logged as an ADR-010 amendment; the no-formula / no-`final_score` core is unchanged. The schema change landed in the A3 execution session (GatekeeperAgent is its first consumer).

## Phase A — New Agents and schemas, isolated

### A1 — `src/schemas.py`

- `RoutingDecision` — new model: `decision: Literal["deliver", "fallback"]`, `reasoning: str` (min_length=1), `trigger_reason: Optional[str]`, `trigger_category: Optional[Literal["low_groundedness", "off_domain", "hallucination_risk", "chunk_mismatch", "empty_retrieval"]]`. GatekeeperAgent's Instructor `response_model`. `trigger_category` physically added in A3 (see amendment 4 above).
- `FallbackResponse` — reshaped v1→v2: replaced `trigger_reason: str` + `context_summary: str` with `acknowledgment: str` + `suggested_redirections: list[str]`; `unstyled_response` changed from `Optional[str]` to `str` (always populated). The v2 shape is what FallbackAgent emits; v1 callers (`flow.py`, `fallback_steps.py`) break at construction-time, not import-time — left on disk per the Ledger and the same blast-pattern as the Day-10 `EvaluationResult` reshape.
- `CloneState` — additive only: `routing_decision: Optional[RoutingDecision] = None`. Full reshape (`retrieved_chunks → chunks`, `styled_response` type change, `citations`, `style_profile`, `fallback_response`) deferred to Phase B.
- `tests/test_schemas.py` updated: new `_make_routing_decision()` helper, 7 RoutingDecision tests (deliver, fallback, invalid literal, empty reasoning, trigger_category defaults None, all 5 literals accepted, invalid literal rejected), 3 FallbackResponse v2 tests (valid shape, empty redirections, v1 fields rejected), 3 CloneState.routing_decision tests.
- **Surprising:** `tests/test_flow.py` has a module-level `_MOCK_FALLBACK = FallbackResponse(trigger_reason=..., context_summary=...)` that executes at import time — before `pytestmark = pytest.mark.skip(...)` applies. The v1 fields no longer exist on the v2 model, so the suite failed to collect. Fixed by updating the constructor to v2 fields while keeping the module-level skip intact. This is the same blast pattern as Day 10 but at collection time instead of runtime.
- ADR candidate: no. Executes ADR-010/012.

### A2 — `src/agents/fallback_agent.py`

- Created fresh (v1 `fallback_steps.py` left on disk). Real CrewAI Agent: role/goal/backstory per leader (terse Torvalds vs. measured Kroah-Hartman), `temperature=0.3` generation, `temperature=0` Instructor parse. Inputs: `query`, `leader`, `trigger_reason`, `style_profile`, `chunks`. Output: `FallbackResponse`. Calendar mock (`generate_available_slots`, 3 slots) called outside the try/except — always real slots, never LLM-generated. Instructor parses a `_FallbackDraft(acknowledgment, suggested_redirections)` from the raw kickoff output; the run method assembles the full `FallbackResponse` (adding `calendar_link`, `available_slots`, `unstyled_response=raw`).
- Templated failsafe: `try/except` wraps the full LLM path. On any exception, `_make_failsafe_response(leader)` returns a leader-named `FallbackResponse` with `seed=42` deterministic slots, then `model_copy(update={"available_slots": slots})` replaces them with the already-generated real slots.
- Created `tests/integration/` directory + `__init__.py`. `tests/integration/test_fallback_agent.py` — 13 tests. Contract tests: query/trigger_reason/chunk text reach the Task description; crew has one agent + one task; agent has role/goal/backstory; run() returns valid `FallbackResponse` (non-empty acknowledgment, list redirections, calendar_link, 3 slots from real helper, unstyled_response); parse uses temperature=0. Two real-behavior tests: `test_run_failsafe_on_llm_raise` (Crew.kickoff raises → FallbackResponse returned, "torvalds" in acknowledgment, 3 slots); `test_run_failsafe_on_instructor_raise` (instructor parse raises → failsafe for "kroah_hartman").
- Suite after A2: 501 passed, 40 skipped, 1 pre-existing failure.
- ADR candidate: no. Executes ADR-012.

### >> INTRA-PHASE-A STOP GATE (held) <<

Stopped before A3. Reported: A1 schema diff, A2 FallbackAgent file:line + failsafe test result, Day-10 suite green, `flow.py` untouched, architecture-honesty greps clean. Waited for Ruby's explicit "continue." Gate held as required by the plan — GatekeeperAgent is the load-bearing routing decision, and Day 12's evaluation depends entirely on it.

### A3 — `src/agents/gatekeeper_agent.py`

- Created fresh (no v1 predecessor). Real CrewAI Agent, `temperature=0`. Inputs: `query`, `response_text`, `chunks`, `evaluation` (EvaluationResult — 3 scores + explanation + flags), `leader`. Output: `RoutingDecision`. Role/goal/backstory encode the key routing trade-offs: confident-sounding but poorly-grounded = hallucination risk → fallback; flag indicating chunk mismatch or off-domain content → fallback; default to deliver when scores are reasonable and no flags raised. No threshold, no `final_score` read (Rule 3).
- `_build_task_description()` includes all three scores formatted to `.3f` (so tests can assert the exact values reach the prompt), the flags list, the evaluator explanation, and explicit routing rules: reasoning must cite specific score values and flags; on fallback, emit `trigger_category` from the 5-literal set + free-text `trigger_reason`; on deliver, both null.
- `_parse_decision(raw)` uses Instructor with `response_model=RoutingDecision` directly (no intermediate draft type — GatekeeperAgent's Instructor output IS the final model, unlike CloneAgent/EvaluatorAgent which add computed fields after parse).
- `tests/integration/test_gatekeeper_agent.py` — 16 tests. Prompt builders: role/goal/backstory non-empty; style_score/groundedness_score/confidence_score each appear in the task description (`.3f` format); flags appear; no-flags case renders "(none)". Crew shape: one agent, one task, has role/goal/backstory. run() contract: deliver returns valid RoutingDecision with trigger_category=None/trigger_reason=None; fallback returns valid RoutingDecision with reasoning set. trigger_category iff: deliver → None; fallback → one of 5 literals (asserted via `_TRIGGER_CATEGORIES` tuple); each of the 5 literals parses without raising. temperature=0: Instructor parse uses `response_model=RoutingDecision` and `temperature=0`.
- Suite after A3: 520 passed, 40 skipped, 1 pre-existing failure. ruff clean. `flow.py` untouched (confirmed `git diff --stat`).
- ADR candidate: no — ADR-010 amended (see plan amendments, amendment 4); this executes it.

## Dead Code Ledger additions (Day 11)

| Item | Status | Safe to delete |
| --- | --- | --- |
| `src/agents/fallback_steps.py` | Dead — superseded by `src/agents/fallback_agent.py` (same role, now a real CrewAI Agent). Left on disk; v1 `flow.py::handle_fallback` still calls it. | After Phase B smoke green. |

All Day-10 Ledger items carry forward unchanged.

## Phase A exit gate — architecture-honesty greps

All four greps pass:
- **4 Agent files carry `from crewai import LLM, Agent, Crew, Task` + role/goal/backstory:** `clone_agent.py`, `evaluator_agent.py`, `fallback_agent.py`, `gatekeeper_agent.py` (plus dead `style_crew.py`, expected).
- **Components import no `litellm|openai|cohere|instructor`:** all matches in `src/components/` are in docstrings and string literals, not import lines. Actual imports in `src/rag/embedder.py` (LiteLLM embed), `src/rag/reranker.py` (Cohere), and dead `src/evaluation/evaluator.py` / `src/fallback/unstyled_responder.py` are pre-existing Ledger entries.
- **No Agent-suffixed functions outside `src/agents/`:** zero matches.
- **`final_score` only in v1 dead-code files and deprecation docstrings:** zero matches in any Day-11 file. All field-level matches are in existing Ledger entries (`flow.py`, `cli.py`, `visualization.py`, `evaluation/evaluator.py`) and their already-skipped tests.

## Phase A defence (rendered, awaiting Ruby's answer)

Four-category defence rendered at the Phase A exit gate:

- **Category I (what was built):** 3 schema changes (RoutingDecision, FallbackResponse v2, CloneState.routing_decision + trigger_category), 2 new Agent files (FallbackAgent, GatekeeperAgent), 2 new integration test files (29 tests total), 1 test-collection fix (test_flow.py).
- **Category II (decided and why):** trigger_category not a schema cross-field validator (prompt + contract-test layer, symmetric with trigger_reason per plan); RoutingDecision used directly as Instructor response_model in GatekeeperAgent (no intermediate draft — output has no computed additions); trigger_category schema change landed in A3 session (GatekeeperAgent is its first consumer, co-located with its consumer).
- **Category III (alternatives considered):** separate `TriggerCategory` Enum rejected (a 5-value inline `Literal` is sufficient, no multi-model reference needed); single-call Instructor approach for GatekeeperAgent rejected (departs from the kickoff→parse canonical pattern enforced by architecture-honesty greps).
- **Category V — v1-drift (mandatory):** no weighted formula, no threshold comparison, no `final_score` field, no Agent-named functions in any Day-11 file. Proved by grep.

## Phase B — `flow.py` rewire, latency, smoke, v1 retire

Phase B cleared the Phase A defence gate and executed B0→B5 as specified.

### B0 — groundedness source audit (pre-condition)

Source-confirmed three facts before touching any code:

1. **`indexer.py:74` excludes embeddings from `metadata.json`:** `metadata = [c.model_dump(exclude={"embedding"}) for c in chunks]`. The FAISS index holds the vectors; `metadata.json` does not.
2. **`retriever.py:49` reconstructs chunks without an embedding:** `chunk = KnowledgeChunk(**metadata[idx])`. The `embedding` key is absent from the dict, so `chunk.embedding is None` on every `Retriever.run()` call in production.
3. **`embed_openai` raises on API failure — no silent MiniLM fallback.** `embed_openai` at `embedder.py:110` calls `litellm.embedding()` directly; any exception propagates. `embed_minilm` is a completely separate function; the two are not wired together. Consequence: `score_groundedness()` always batch-re-embeds chunk text via `embed_openai` (the `embedding is None` branch is always taken live), and any API failure hard-fails rather than degrading.
4. **Coverage gap pre-closed.** `tests/test_groundedness_scorer.py:161` already has `test_score_groundedness_missing_chunk_embedding_triggers_batch` (embedding=None → re-embed path, score ∈ [0,1]). No new test needed.

B0 produced no code changes. The coverage gap named in amendment 3 (Phase A) was already closed before Phase B started.

### B1 — CloneState v2 reshape + `flow.py` rewrite

**CloneState** (`src/schemas.py`): full v2 reshape. Renamed `retrieved_chunks` → `chunks`. Replaced `styled_response: str` and `trigger_reason` and `final_output` with typed optional fields: `style_profile: Optional[StyleProfile]`, `response_text: Optional[str]`, `citations: list[Citation]`, `styled_response: Optional[StyledResponse]`, `fallback_response: Optional[FallbackResponse]`. Added `routing_decision: Optional[RoutingDecision]` (additive in Phase A, now part of the canonical v2 shape). `model_config = ConfigDict(arbitrary_types_allowed=True)` required for `StyleProfile` (numpy array inside). Ten fields total; `trigger_reason` and `final_output` gone — v1 callers (`cli.py`, `visualization.py`) break at attribute-access time, not import time, consistent with the Day-10 blast-radius pattern.

**`flow.py`** (`src/flow.py`): full rewrite. Five-step pipeline:

| Step | Decorator | Method | Agent called |
|------|-----------|--------|--------------|
| 1 | `@start()` | `retrieve` | `Retriever()` (Component) |
| 2 | `@listen(retrieve)` | `clone` | `CloneAgent()` |
| 3 | `@listen(clone)` | `evaluate` | `EvaluatorAgent()` |
| 4 | `@router(evaluate)` | `route` | `GatekeeperAgent()` |
| 5a | `@listen("deliver")` | `finalize` | — (state assembly) |
| 5b | `@listen("fallback")` | `handle_fallback` | `FallbackAgent()` |

`retrieve` early-exits when `state.chunks` is non-empty (ADR-005 shared-retrieval optimization). `route` has an emergency guard: if `state.evaluation is None`, returns `"fallback"` immediately with a `RoutingDecision` describing the skip rather than calling GatekeeperAgent on a None input.

`compare_leaders()` updated: loads profiles externally, injects via `kickoff(inputs={"style_profile": ...})`, uses `state.chunks` (renamed). Torvalds flow runs first and retrieves; Kroah-Hartman flow receives `chunks=shared_chunks` so its retrieve step early-exits — one Retriever call total (ADR-005 gate).

**Pydantic PrivateAttr** (surprising): The v1 `flow.py` set `self._config = ...` in `__init__` after `super().__init__()` and this was never verified at runtime because all v1 flow tests were skipped. CrewAI `Flow` inherits `BaseModel`; Pydantic's `__getattribute__` intercepts access to unknown underscore names set outside the Pydantic init path. Setting `self._timings = {}` in `__init__` does not persist across subsequent step-method calls — attribute reads return `AttributeError: 'DigitalCloneFlow' object has no attribute '_timings'. Did you mean: 'timings'?`. Fix: `_timings: dict = PrivateAttr(default_factory=dict)` declared at class level. The `__init__` override was removed entirely. This is the correct Pydantic v2 pattern for per-instance private state on a BaseModel subclass.

### B2 — per-stage latency instrumentation

Two-layer timing:

- **Step-level wall-clock** on `DigitalCloneFlow` via `_timings: dict = PrivateAttr(default_factory=dict)` (same PrivateAttr from B1). `perf_counter()` wraps each agent call; result stored as `retrieve_ms`, `clone_ms`, `evaluate_ms`, `route_ms`, `deliver_ms`, `fallback_ms`. Exposed via `flow.timings` property (returns a copy).
- **Generate/parse split** per agent via a `last_run_timings: dict` instance attribute set at the end of each agent's `run()` method. Flow reads it with `getattr(agent, "last_run_timings", {})` — graceful fallback to `{}` when the agent is mocked. Stored as `clone_generate_ms`/`clone_parse_ms`, `evaluate_score_ms`/`evaluate_generate_ms`/`evaluate_parse_ms`, `route_generate_ms`/`route_parse_ms`, `fallback_generate_ms`/`fallback_parse_ms`.

Timings are observability only — not asserted in any test, not stored on `CloneState`. Day 12 is the first real-LLM timing measurement pass. The key structure is locked in now so Day 12 can assert on specific keys.

**Why `last_run_timings` instead of returning `(result, timing_dict)`.** Changing any agent's return type from `CloneResponse`/`EvaluationResult`/etc. to a tuple would break all Phase A contract tests that assert on typed return values. The instance-attribute side-channel adds zero coupling; mocked agents simply don't set the attribute and `getattr` falls back to `{}`.

### B3 / B4 — test suite rewire

**`tests/test_flow.py`** (full rewrite, 25 tests): removed the module-level `pytest.mark.skip` that had protected the collection-time `FallbackResponse(trigger_reason=...)` crash. Updated to v2 constructors. Two central helpers: `_run_deliver()` patches `Retriever`, `CloneAgent.run`, `EvaluatorAgent.run`, `GatekeeperAgent.run`; `_run_fallback()` adds `FallbackAgent.run`. Tests cover: deliver 5-step trace (all five state fields populated), fallback 5-step trace, typed field assertions on both arms, `retrieve` early-exit (`Retriever.run.assert_not_called`), `retrieve_ms` absent when early-exiting, Kroah-Hartman path, all 7 timing-key groups.

**`tests/integration/test_compare_leaders.py`** (new, 4 tests): gate test asserts `mock_retriever_instance.run.assert_called_once()` — exactly one Retriever-Component call across both flow runs (Torvalds retrieves; Kroah-Hartman early-exits). Assertion message explicitly notes this counts Retriever-Component calls, not embedding calls (evaluate still re-embeds for groundedness — expected, not a regression). Three shape tests: `compare_leaders` returns a `LeaderComparison`, `result.query` matches input, both arms are `StyledResponse`.

**`tests/test_schemas.py`**: `CloneState` block updated to v2 field names (`chunks`, `style_profile`, `response_text`, `citations`, `styled_response`, `fallback_response`). `test_clone_state_incremental_population` uses v2 fields in pipeline order.

**`tests/test_cli.py` and `tests/test_visualization.py`**: `pytestmark = pytest.mark.skip(reason="... Day 12 (D-B1)")`. No module-level v1-shaped constructors in either file (all v1 field access is inside function bodies), so a simple `pytestmark` is safe — verified by checking collection completes with zero errors. Pre-existing `F841` in `test_cli.py` (`as mock_cls` unused variable) fixed while the file was open.

**Suite after B3/B4:** 519 passed, 37 skipped, 1 pre-existing failure (`test_load_queries_canonical_file`).

### Stop Gate 1 — Phase B defence (rendered and cleared)

Four-category defence rendered before B5.

**Category I (what was built):** CloneState v2 (10 fields; `retrieved_chunks`/`trigger_reason`/`final_output` gone); `flow.py` v2 (5-step pipeline calling real agents); per-stage latency via `PrivateAttr(_timings)` + `last_run_timings` side-channel on all four agents; `test_flow.py` rewritten (25 tests); `test_compare_leaders.py` created (4 tests, Retriever-count gate); collection-safe skip markers on `test_cli.py` + `test_visualization.py`.

**Category II (decided and why):** `PrivateAttr` for `_timings` — `BaseModel.__getattribute__` intercepts `__init__`-set underscore names; `PrivateAttr(default_factory=dict)` is the correct Pydantic v2 pattern. `getattr(agent, "last_run_timings", {})` — preserves agent return-type contracts unchanged; mocked agents return `{}` without special handling. Both step-level and generate/parse split recorded — they are complementary (outer vs inner measure) and Day 12 needs both.

**Category III (alternatives considered):** Timings on `CloneState` rejected — latency is observability, not a result-model field; CloneState is serialized/validated by Pydantic. Tuple return from agents rejected — breaks Phase A contract tests. Combined `evaluate_and_route` step rejected — plan explicitly splits them; the split attributes the groundedness re-embed clearly to `evaluate_ms`.

**Category V — v1-drift:** No weighted formula, threshold comparison, `final_score` field, or Agent-named function in any Day-11 B file. `final_score` in the repo is confined to: `schemas.py` deprecation docstring; the three blocked/deferred Day-12 files (`cli.py`, `visualization.py`, `evaluation/evaluator.py`); their skipped tests; and two v2 assertions that confirm the field is absent or rejected.

Gate cleared by Ruby. B5 proceeded.

### B5 — v1 dead code retirement

Per-file grep before every delete. Zero live importers required; any unexpected caller would stop the retirement of that file.

**Retired (grep zero, deleted):**

| File | Grep result | Live callers | Action |
|------|-------------|--------------|--------|
| `src/agents/style_crew.py` | `tests/test_style_crew.py` only | Co-retired | Deleted |
| `tests/test_style_crew.py` | imports `style_crew` | Co-retired | Deleted |
| `src/agents/evaluator_steps.py` | zero | — | Deleted |
| `src/agents/fallback_steps.py` | zero | — | Deleted |
| `scripts/timing_dual_leader.py` | zero | — | Deleted |

**Blocked (unexpected live callers — not deleted):**

| File | Blocking caller | Resolution |
|------|-----------------|------------|
| `src/agents/rag_agent.py` | `src/cli.py:16` imports `RAGAgent` | Blocked. `cli.py` is deferred Day-12 (D-B1); `rag_agent.py` must survive until then. |
| `src/evaluation/evaluator.py` | `src/evaluation/__init__.py:4` re-exports `evaluate` | Blocked. `__init__.py` is a live package init not in the retire list. Retire Day 12 alongside cli.py refactor. |
| `src/rag/reranker.py::rerank()` | `src/rag/__init__.py:8`, `tests/test_reranker.py:12`, 7 experiment scripts | Blocked. `rerank()` has many callers; file also exports `rerank_with_status` used by `Retriever`. Keep. |

**Tooling note (rm -i alias):** first delete attempt used `rm` without path prefix; the shell alias `rm='rm -i'` rendered the interactive prompt but the files were not deleted in the non-interactive bash context (the prompt output was captured but no `y` was provided). Second attempt used `\rm` to bypass the alias — deletions confirmed by `ls` verification before and after.

**Suite after B5:** 498 passed, 37 skipped, 1 pre-existing failure. 21 tests removed (the `test_style_crew.py` suite, which had all been live-passing). ruff clean on `src/` and `tests/`.

**PRD §12.2** mapping rows updated: four retired files marked "retired (Day 11)"; three blocked files annotated with their blocking reason and Day-12 trigger.

## Dead Code Ledger — final state (Day 11)

| Item | Status |
|------|--------|
| `src/agents/rag_agent.py` | Blocked — `src/cli.py` imports `RAGAgent`. Retire Day 12 with cli.py refactor. |
| `src/evaluation/evaluator.py` | Blocked — `src/evaluation/__init__.py` re-exports `evaluate`. Retire Day 12. |
| `tests/test_evaluator.py` | Blocked — exercises `evaluation/evaluator.py`. Remove with it Day 12. |
| `src/rag/reranker.py::rerank()` | Live thin wrapper. Many callers; not retiring. |
| `src/cli.py` / `src/visualization.py` | Use v1 `final_score`/`final_output`. Refactor Day 12 (D-B1). |
| `tests/test_cli.py` / `tests/test_visualization.py` | Skip-marked (collection-safe). Re-enable Day 12. |
| `src/agents/style_crew.py` | **Retired Day 11.** |
| `tests/test_style_crew.py` | **Retired Day 11.** |
| `src/agents/evaluator_steps.py` | **Retired Day 11.** |
| `src/agents/fallback_steps.py` | **Retired Day 11.** |
| `scripts/timing_dual_leader.py` | **Retired Day 11.** |

## Phase B exit gate — architecture-honesty greps

Run after B5 retirement. All four pass:

1. **All 4 agent files `from crewai import LLM, Agent, Crew, Task` + role/goal/backstory present:** `clone_agent.py`, `evaluator_agent.py`, `gatekeeper_agent.py`, `fallback_agent.py`. ✓
2. **`src/components/` imports no `litellm|openai|cohere|instructor`:** zero matches. ✓
3. **No Agent-suffixed function definitions outside `src/agents/`:** zero matches. ✓
4. **`final_score` confined to deprecation docstring + three blocked Day-12 files:** `schemas.py` (docstring prose); `cli.py`, `visualization.py`, `evaluation/evaluator.py` (live but blocked — known Day-12 items); their skipped tests; plus two v2 assertions in `test_schemas.py` and `test_evaluator_agent.py` that confirm the field is absent or rejected. Zero matches in any Day-11 file. ✓

## Day-12 named follow-ups

Carried forward from Phase A amendments plus B5 blockers:

- `test_groundedness_scorer: chunk.embedding=None re-embed path` — already closed (pre-existing test found in B0 audit). No action needed.
- `trigger_category` correctness verification with real LLM (contract test proves a legal literal is returned; behavioral accuracy requires live execution).
- Sync ADR-010 amendment (`trigger_category` addition) to Notion ADR Log.
- `cli.py` + `visualization.py` refactor to v2 field names — D-B1; unblocks `rag_agent.py` and `evaluation/evaluator.py` retirement.
- `src/evaluation/__init__.py` — remove `evaluate` re-export when `evaluation/evaluator.py` is retired.
- In-Flow convenience profile loader for `cli query` command (profile-loading is currently caller-side only).
- First real-LLM latency measurement pass using `flow.timings` + `last_run_timings` keys established in B2.
