# Day 5 Plan — Flow Orchestration + Integration

> Execution will be performed on a `feat/day5-flow-orchestration` branch — never direct to main, PR-only.

---

## Context

Days 1–4 are merged to main. The repo now contains:

- **RAG retrieval** — `src/agents/rag_agent.py` exposes `RAGAgent.retrieve(query) → list[RetrievalResult]` (FAISS top-20 → Cohere rerank → top-5).
- **Evaluator** — `src/agents/evaluator_steps.py` (`EvaluatorAgent.evaluate(...)`) returns `EvaluationResult` with weighted final_score and `decision ∈ {"deliver","fallback"}` at the 0.75 threshold.
- **Fallback** — `src/agents/fallback_steps.py` (`build_fallback_response(...)`) returns `FallbackResponse` (3 calendar slots + context summary + unstyled answer).
- **CloneState** — already defined in `src/schemas.py` with `query, leader, retrieved_chunks, styled_response, evaluation, final_output`. No schema changes needed.
- **Crew/Agent abstraction** — currently zero usages of `crewai` in `src/`. Day 5 introduces it for exactly one place: `style_crew.py`.

What is **missing** is the orchestration layer that wires these pieces into a working query pipeline, plus the dual-leader optimization that makes "retrieve once, style twice" work for the comparison mode.

Day 5 delivers that orchestration. The PlannerAgent _is_ the `DigitalCloneFlow` class — there is no separate planner module. CrewAI Flows _are_ the planner.

**Hard exit criteria** (PRD §8 Day 5 row, verbatim):
> End-to-end Flow works. Dual-leader comparison produces two scored responses. @router correctly branches on threshold. Error recovery tested. ADR-005 documents shared RAG optimization.

**Carry-forward conventions** (do not deviate without an ADR):
- `instructor.from_litellm(litellm.completion)` for any structured output — never `from_openai`.
- Pydantic schemas live in `src/schemas.py`.
- Façade pattern from `src/agents/*_steps.py` (plain Python class with single public method, no Crew).
- ADR format: exactly 5 sections per CLAUDE.md (Context, Decision, Alternatives Considered, Quantified Validation, Consequences). Java/TS parallel as one parenthetical at the end of Consequences.
- Branch naming: `feat/day5-flow-orchestration`. One branch for the day. PR to main.

---

## Verification contract (applies to every phase)

Each phase has a **stop gate**. Before moving to the next phase, the executor must paste into the chat:

1. The exact pytest invocation used and its full terminal output (pass/fail counts visible).
2. For smoke scripts: the actual stdout of the script, not a paraphrased summary.
3. For files created: file path + line count (`wc -l <path>`).
4. For latency numbers used in ADR-005: the timing harness output that produced them.

Self-reported "I did it" claims are not acceptable. If a gate fails, stop and surface the failure — do not silently proceed.

---

## Phase breakdown

### Phase 0 — Branch + scaffolding (≤10 min)

**Orientation.** Cut `feat/day5-flow-orchestration` from current main. Confirm `src/flow.py` and `src/agents/style_crew.py` are still empty stubs (Day 4 baseline). Delete the empty `src/agents/rag_steps.py` stub — the Flow's retrieve step calls `RAGAgent.retrieve()` directly; no wrapper façade needed for symmetry's sake.

**Acceptance criteria.**
- Branch `feat/day5-flow-orchestration` exists and is checked out.
- `src/agents/rag_steps.py` is gone (`git status` shows the deletion).
- `pytest tests/ -q` reports **exactly 382 passing** — the Day 4 end-of-day baseline recorded in `CLAUDE.md` "Current State" (Last Updated 2026-04-14). Any deviation is a regression signal: do not proceed to Phase 1 until the count matches or the delta is explained.

**Stop gate output to paste.**
- `git branch --show-current`
- `git status`
- Full pytest summary line including the count (`382 passed in Xs`).

---

### Phase 1 — `src/agents/style_crew.py` (single-agent Crew)

**Orientation.** This is the **only** place in the codebase where the CrewAI `Agent`/`Crew` abstraction is used. The Crew has one Agent (ChatStyleAgent) whose role/goal/backstory must encode the leader's actual style features (short sentences, dashes, sign-off conventions, etc.) — not generic instructions like "write like Torvalds". Inputs: the leader's `StyleProfile`, the retrieved chunks, the user query. Output: a styled response string for `state.styled_response`.

Why a Crew here and nowhere else: style generation needs role/goal/backstory framing for the LLM to internalize voice; the other steps (retrieval, evaluation, fallback) are deterministic computation or single Instructor calls where Crew adds overhead with no benefit. ADR-001 already locks this in.

**Acceptance criteria.**
- A factory function builds a Crew bound to a specific leader's `StyleProfile`.
- `kickoff(...)` returns a non-empty string given mock chunks + a real loaded `StyleProfile` for one leader.
- The Agent's role/goal/backstory references concrete numerical features from the profile (sentence length, punctuation patterns, sign-off), not generic descriptors.
- Uses `litellm`-routed `gpt-4o-mini` (per [CLAUDE.md Architecture Rule #7](../../CLAUDE.md), line 137), not raw OpenAI SDK.
- **Differentiation test (replaces string-contains):** build a prompt for Torvalds and a prompt for Kroah-Hartman from their actual loaded profiles. Assert (a) `torvalds_prompt != kh_prompt` as a baseline sanity check, and (b) at least one numerical feature from the `StyleProfile` schema (use whichever fields the schema actually defines) appears in the Torvalds prompt with Torvalds' specific value, AND the same field appears in the Kroah-Hartman prompt with a different value. String-contains alone is gameable; per-leader numerical injection of an actual schema field is the contract.

**Stop gate output to paste.**
- `pytest tests/test_style_crew.py -v` full output.
- Smoke script stdout: load Torvalds profile → call `kickoff` with one mock chunk → first 200 chars of the returned string.
- `wc -l src/agents/style_crew.py tests/test_style_crew.py`.

---

### Phase 2 — `src/flow.py` happy path (no router, no fallback yet)

**Orientation.** Build the `DigitalCloneFlow(Flow[CloneState])` shell with three steps wired sequentially: retrieve → style → evaluate → (deliver only). The router/fallback comes in Phase 3 — splitting the work this way means Phase 2 can prove the Flow harness, state propagation, and step composition work end-to-end before adding branching complexity.

The retrieve step calls `RAGAgent.retrieve()` directly. The style step delegates to the Phase 1 Crew. The evaluate step calls `EvaluatorAgent.evaluate()`. The deliver step assembles a `StyledResponse` into `state.final_output`.

State propagation: every step reads from and writes to `self.state` (the existing `CloneState`). Do not introduce a parallel state object — `CloneState` already has every field needed.

**Acceptance criteria.**
- `DigitalCloneFlow().kickoff(inputs={"query": ..., "leader": ...})` runs to completion against mocked LLM calls and produces a `StyledResponse` in `state.final_output`.
- After the run, `state.retrieved_chunks` is populated, `state.styled_response` is non-empty, and `state.evaluation.final_score ∈ [0, 1]`.
- Integration test in `tests/test_flow.py` exercises the full happy path with mocks for the LLM-using steps.

**Stop gate output to paste.**
- `pytest tests/test_flow.py::test_happy_path -v` output.
- Smoke script stdout: run flow with one real query against mocked LLMs, print `state` field-by-field after completion.

**Out of scope for this phase.** Router branching, fallback wiring, error handling, dual-leader. Resist the urge to add them here — Phase 3 covers them.

---

### Phase 3 — Router + fallback + error recovery

**Orientation.** Convert `evaluate_response` from a plain `@listen` step into one that **also** carries `@router()`, returning the **string** `"deliver"` or `"fallback"` (not a bool — this is the failure mode called out in [ADR-001's Decision section](../adr/ADR-001-crewai-flow-pattern.md), line 39: "One wrong value (returning `True` instead of `\"deliver\"`, for example) and nothing routes."). Add the `fallback` branch step that calls `build_fallback_response(...)` and assembles a `FallbackResponse` into `state.final_output`. Wrap step bodies in **narrow** try/except — see exception discipline below — so that genuine degradation (LLM/network/parse failures) routes the flow into the fallback path with a typed `trigger_reason`, while bugs propagate.

The threshold (0.75) lives inside the evaluator already — the router just inspects `state.evaluation.decision` and returns the matching string. Do **not** re-implement the threshold check in the router; the evaluator is the single source of truth.

**Exception discipline (do not deviate).**
- Catch only specific exception classes that represent graceful-degradation paths: network/timeout errors (`httpx.HTTPError`, `litellm.exceptions.APIError`, `cohere.errors.*`), JSON/parse errors (`json.JSONDecodeError`), Instructor retry exhaustion. The exact list lives in `src/flow.py` near the catch site, with a one-line comment per class explaining why it's degradable.
- Never use bare `except:` or `except Exception:`. Both are banned.
- **`pydantic.ValidationError` and `AssertionError` MUST propagate** — they indicate bugs (a schema invariant broke, an internal assertion failed), not transient failure modes the user should be shielded from.
- `trigger_reason` format: `"<step_name> raised <ExceptionClassName>: <short message>"` (e.g., `"retrieve_knowledge raised APIError: cohere rerank timeout after 30s"`). Never just `str(exc)` — the class name is the load-bearing part for debugging.

**Acceptance criteria.**
- Router test with mocked evaluator returning `final_score = 0.7499` → routes to fallback. With `0.7500` → routes to deliver. (Boundary identical to the Day 4 evaluator boundary test.)
- Error-injection tests: forcing `RAGAgent.retrieve` to raise an `httpx.HTTPError` → flow ends with a `FallbackResponse` whose `trigger_reason` includes both `"retrieve_knowledge"` and `"HTTPError"`. Same shape for the style and evaluate steps.
- **Bug-propagation test:** force a step to raise `pydantic.ValidationError` (e.g., by feeding a malformed `EvaluationResult` into the evaluate step). The flow MUST raise — not route to fallback. Mirror the same with `AssertionError`. This test is the contract that exception discipline is being honored.
- `state.final_output` is populated whenever the flow exits via the deliver or fallback paths — never `None` on a successful exit.
- The `@router()` decorator is the only routing mechanism; no `if/else` branching outside it.

**Stop gate output to paste.**
- `pytest tests/test_flow.py -v` full output (happy path + boundary + 3 error-injection tests + 2 bug-propagation tests).
- Coverage report for `src/flow.py`: `pytest --cov=src/flow --cov-branch --cov-report=term-missing` — both line and branch coverage must show ≥ 90%. Branch coverage specifically catches untested `@router` paths, which line coverage hides.

---

### Phase 4 — Dual-leader comparison ("retrieve once, style twice")

**Orientation.** Add a wrapper (function or thin class) that runs `DigitalCloneFlow` twice — once for Torvalds, once for Kroah-Hartman — and shares the retrieved chunks across both runs. The mechanism is already implied by `CloneState`: pass a pre-populated `retrieved_chunks` list into the second run, and have the retrieve step early-exit when `len(self.state.retrieved_chunks) > 0`.

Output is a `LeaderComparison` (already in `src/schemas.py`) with both scored responses side-by-side.

The wrapper is the only addition; the Flow class itself just gets a one-line guard at the top of the retrieve step. Do not introduce a parallel `DualLeaderFlow` class — that duplicates code and breaks ADR-001's "single Flow class" stance.

**Two distinct timing numbers must be captured** — they feed two different parts of ADR-005:
1. **RAG retrieval cost avoided** — the time `RAGAgent.retrieve` takes on a warm FAISS index against the real RAG stack. This is the actual saving from the optimization (the work the second run skips). Measure it in isolation with a microbenchmark, not as a delta of two end-to-end runs. Expect sub-100ms on a warm index — that's the honest number.
2. **End-to-end dual-leader latency** — wall-clock from wrapper invocation to `LeaderComparison` returned, with mocked LLMs (fixed sleep representing realistic generation latency) and the real RAG call. This is the headline number that shows whether the dual-leader path stays inside the PRD's sub-1s envelope.

The PRD's "1s target" is the end-to-end number. The optimization saves the RAG cost specifically — framing it as "speedup achieved" in mocked tests overstates the win, because the LLM sleeps dominate. Quantified Validation in ADR-005 must say "RAG retrieval cost avoided" with the absolute ms number, not "X% speedup."

**Acceptance criteria.**
- A wrapper produces a `LeaderComparison` from a single query input.
- The second run does **not** call `RAGAgent.retrieve` (verify with a mock `assert_called_once`).
- **RAG cost number captured** in ms via a standalone microbenchmark on the real (warm) FAISS index — not derived from end-to-end timing.
- **End-to-end latency number captured** in ms via the wrapper-vs-mocked-LLM harness.
- A failure in leader A's pipeline does not block leader B from running (independent error paths).

**Stop gate output to paste.**
- `pytest tests/test_flow.py::test_dual_leader -v` full output.
- RAG retrieval microbenchmark output: warm-index `RAGAgent.retrieve` latency in ms (single number, plus the harness command).
- End-to-end timing harness output: dual-leader wrapper wall-clock in ms, with the mocked-LLM sleep value disclosed. Both numbers go into ADR-005 verbatim, with the framing "RAG retrieval cost avoided" — never "speedup."

---

### Phase 5 — ADR-005 + Mermaid diagrams (A2 + A3)

**Orientation.** Write `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` following the **5-section format** matching [ADR-004](../adr/ADR-004-groundedness-scoring-approach.md) exactly. ADR-004 is the most recent ADR on main and the structural reference: 5 H2 sections in the order Context → Decision → Alternatives Considered → Quantified Validation → Consequences, no sub-headers in Consequences, alternatives as `**bold name** — paragraph`, Java/TS parallel as one parenthetical at the end of Consequences. Open ADR-004 side by side while writing ADR-005.

The A2 (single-query) and A3 (dual-leader) Mermaid sequence diagrams live **inline** in ADR-005 — A2 in the Decision section as the baseline pipeline, A3 immediately after as the optimization the ADR is justifying.

Section-by-section guidance for ADR-005:

- **Context.** What problem motivates retrieve-once-style-twice: the dual-leader comparison mode needs sub-1s end-to-end latency, but two independent pipelines duplicate the RAG step (embed + FAISS + Cohere rerank). State the constraint and the cost shape; do not lead with the solution.
- **Decision.** State the chosen pattern: shared `CloneState.retrieved_chunks` across two Flow invocations, with the retrieve step early-exiting when chunks are already present. Embed A2 (single-query baseline) and A3 (dual-leader optimized). Both Mermaid blocks live here.
- **Alternatives Considered.** Two alternatives, each as `**Bold name** — paragraph` (never a table):
  - **Independent pipelines per leader** — run two full Flows in isolation, no shared state. Reject because it duplicates the RAG step on every dual-leader query; quote the Phase 4 RAG-cost-avoided number directly to ground the rejection in measurement, not estimate.
  - **Cached RAG with TTL keyed on query hash** — add a cross-request cache layer. Reject because dual-leader is one user-facing request, not two distinct requests separated in time; a TTL cache would be cleared between unrelated requests anyway, adding infrastructure for no latency win. State-based reuse is local to one request and needs zero infrastructure.
- **Quantified Validation.** Use the actual Phase 4 numbers verbatim. Frame as "RAG retrieval cost avoided in the dual-leader path" — not "speedup." Disclose the harness conditions (warm FAISS, real Cohere call, single query for the RAG number; mocked LLM with explicit sleep value for the end-to-end number). If the absolute RAG cost is small (sub-100ms on warm index), say so plainly — the optimization's value is "we don't pay for the RAG step twice," not "we're 50% faster end-to-end."
- **Consequences.** Single flowing prose section, no sub-headers. Cover state coupling between the two Flow runs (a failure in run 1 leaves run 2 with stale or empty chunks — mitigation is the Phase 4 independent-error-path requirement); the wrapper is the one place that knows about dual-leader, keeping the Flow class single-purpose; portability cost (a future "compare N leaders" mode would generalize this wrapper but not require Flow changes). End with a Java/TS parallel inline in one parenthetical. The mechanism here is passing a precomputed value as an explicit argument to skip recomputation on the second call — closer to memoization or constructor injection than to a framework-managed scope. Pick the analogy that maps to that specific behavior, and skip framework-flavored options (request-scoped beans, render-tree contexts) that don't fit. If no clean parallel exists, omit the parenthetical entirely — a forced analogy reads worse than none.

**Acceptance criteria.**
- File at `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` with exactly 5 H2 sections in the correct order. No Cross-References, Interview Signal, Java/TS Parallel, or Easier/Harder/Portability sub-headers. No bold emotional category labels in Consequences.
- Java/TS parallel appears as one parenthetical at end of Consequences only.
- Section structure matches ADR-004 exactly when diffed at the H2 level.
- A2 (single-query sequence) **must** show the `@router` decision point with both `deliver` and `fallback` as distinct outgoing arrows from the evaluator node. A diagram that hides the branch fails its purpose.
- A3 (dual-leader sequence) **must** show a single `retrieve_knowledge` node with arrows fanning out to both the leader-A and leader-B style+evaluate passes — the shared-chunks optimization must be visually obvious at a glance. If the diagram shows both leaders calling retrieve independently, redo it.
- Both diagrams validate without syntax errors at [mermaid.live](https://mermaid.live) (paste the shareable URL) or via `mmdc -i <file>` (paste exit code 0).
- All numerical claims trace to Phase 4 stop-gate output. No fabricated citations, no unverified quantitative claims.

**Stop gate output to paste.**
- Full ADR file content (`cat docs/adr/ADR-005-shared-rag-dual-leader-mode.md`).
- mermaid.live shareable URL **or** `mmdc` exit code for each of A2 and A3.
- A diff or visual confirmation that ADR-005's H2 section list matches ADR-004's H2 section list.

---

### Phase 6 — Final integration sweep + cleanup

**Orientation.** Run the full test suite, capture coverage, ensure CLAUDE.md's "Current State" block is updated to reflect Day 5 complete. No new code — this is the verification echo-back phase that closes the day.

**Acceptance criteria.**
- `pytest tests/ -v` — all prior tests still pass; new Day 5 tests pass; total count is `382 + N_new` where 382 is the locked Day 4 baseline and `N_new` is reported as a number, not "many".
- `pytest --cov=src/flow --cov=src/agents/style_crew --cov-branch --cov-report=term-missing` — both line and branch coverage on `src/flow.py` and `src/agents/style_crew.py` ≥ 90%. Branch coverage is required (not optional) because line coverage hides untested `@router` paths.
- CLAUDE.md "Current State" block updated: Last Updated date (2026-04-26 or actual), Current Day = Day 5 complete, Tests count refreshed to the new total, Branch = main once merged.

**PR description must contain (template, not optional):**
1. **Test count delta:** `Day 4 baseline: 382 → Day 5 final: <new_total> (+N_new)`. Both numbers visible.
2. **Branch coverage percentages** for `src/flow.py` and `src/agents/style_crew.py`, copied directly from the term-missing report. Line coverage alone is not sufficient.
3. **Phase 4 timing numbers** (both): RAG retrieval cost avoided in ms, and end-to-end dual-leader latency in ms (with the disclosed mocked-LLM sleep value). These are the same numbers that appear in ADR-005's Quantified Validation section — they must match.
4. **Direct link** to `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` (relative-path link from the PR body).
5. One-line summary of the four delivered pieces: flow, style_crew, dual-leader wrapper, ADR-005.

**Stop gate output to paste.**
- Full pytest summary line with the new total count.
- Coverage report (term-missing, with `--cov-branch` enabled).
- PR URL.
- The PR body content itself (so the reviewer can confirm all 5 required items are present).

---

## Files touched

| Path | Action | Phase |
|------|--------|-------|
| `src/flow.py` | Create | 2, 3, 4 |
| `src/agents/style_crew.py` | Create | 1 |
| `src/agents/rag_steps.py` | **Delete** (empty stub) | 0 |
| `tests/test_style_crew.py` | Create | 1 |
| `tests/test_flow.py` | Create | 2, 3, 4 |
| `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` | Create | 5 |
| `CLAUDE.md` | Update "Current State" | 6 |
| `src/schemas.py` | **No change** — `CloneState` already exists | — |

---

## Reuse map (do not duplicate)

- `src/schemas.py::CloneState` — Flow state model. Already complete.
- `src/schemas.py::StyledResponse, FallbackResponse, LeaderComparison, EvaluationResult` — output types. Already complete.
- `src/agents/rag_agent.py::RAGAgent.retrieve(query)` — call this in the retrieve step.
- `src/agents/evaluator_steps.py::EvaluatorAgent.evaluate(...)` — call this in the evaluate step.
- `src/agents/fallback_steps.py::build_fallback_response(query, chunks, trigger_reason)` — call this in the fallback step.
- `src/style/profile_builder.py::load_profile(leader)` — call this in the style step factory.
- Test fixture pattern: `_make_features()`, `_make_profile()`, `_make_result()`, `_mock_instructor_client()` from `tests/test_evaluator.py` — copy/adapt for Flow tests rather than reinventing.

---

## End-to-end verification (after all phases land)

A reviewer should be able to run, in order:

1. `pytest tests/ -v` → all green, count > Day 4 baseline of 382.
2. `pytest --cov=src/flow --cov=src/agents/style_crew --cov-branch --cov-report=term-missing` → both line and branch coverage ≥ 90%.
3. A scratch one-liner: `python -c "from src.flow import DigitalCloneFlow; ..."` running a real query against real LLM/RAG → produces a `StyledResponse` with `final_score ∈ [0,1]` in under ~1s.
4. The dual-leader wrapper on the same query → produces a `LeaderComparison` whose wall-clock matches the end-to-end dual-leader number captured in Phase 4 stop-gate output (with the mocked-LLM sleep value disclosed) and quoted verbatim in ADR-005's Quantified Validation section.
5. Open `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` and visually confirm A2 + A3 render in any Mermaid-aware viewer.

Each of those is what the user pastes into the PR description as proof Day 5 is done.
