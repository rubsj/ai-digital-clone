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

## Phase B — status

**Not started.** Phase B (CloneState reshape + `flow.py` rewire + latency + smoke + v1 retire) begins after Ruby clears the Phase A defence gate.

**Day-12 named follow-ups recorded from amendments:**
- `test_groundedness_scorer: chunk.embedding=None re-embed path` (amendment 3 — always-taken live branch uncovered).
- `trigger_category correctness` — whether the Gatekeeper labels the right category for given inputs (contract test only proves a legal literal is chosen; behavioral correctness requires real LLM).
- ADR-010 amendment — sync `trigger_category` addition to Notion ADR Log (`reference_notion_adr_log.md`).
