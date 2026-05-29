# Day 10 — Components + First Two Agents

> **Plan-file authority (Rule 7):** Re-read `docs/plans/day10-plan.md` from disk at the start of every phase. The file on disk is authoritative over any version in context.

## Context

P6 v1 claimed "5 agents" but had 1 real LLM agent; v2 (ADR-009) splits work into 4 LLM-driven **Agents** (`src/agents/`) and 3 deterministic **Components** (`src/components/`). Day 10 builds the first 3 Components + first 2 Agents by **refactoring v1 code, not rewriting it** (§12.2). v2 also removes the v1 weighted-scoring formula (ADR-010/011); Day 10 begins that by reshaping `EvaluationResult`.

**Scope:** 3 Components (Retriever, StyleProfileBuilder, ScoringEngine) + 2 Agents (CloneAgent, EvaluatorAgent hybrid) + schema updates + unit tests for all five.

**Out of scope (Day 11+), must not appear:** GatekeeperAgent, FallbackAgent, Flow refactor, `CloneState.routing_decision`, `RoutingDecision` model, integration tests, e2e tests, CLI, Streamlit, §4.8 cleaning-pipeline edits, 15-feature edits.

**Estimated duration:** ~6–8 working hours across 4 phases.
**Tooling:** Python 3.12+, `uv run`, `ruff` (line 100, double quotes), `mypy` on `src/` + `tests/`.

### Confirmed decisions
1. **EvaluationResult**: strip `final_score` + `decision` + the formula `@model_validator`; add `flags: list[str]`. v1 `evaluator.py` becomes dead code left on disk until Day 11.
2. **CloneAgent output**: add a new `CloneResponse` model (`response_text: str`, `citations: list[Citation]`) to `schemas.py` as the Instructor `response_model`. `StyledResponse` untouched.
3. **Cohere**: keep ADR-002 graceful fallback, but loud WARNING on missing/empty `COHERE_API_KEY` (kill the empty-string silent default), and a unit test asserts the Cohere client was actually invoked.

---

## Phase 1 — Schemas (do first; Agents/Components depend on these types)

Re-read this plan from disk before starting.

**File:** `src/schemas.py` (edit only).

- **`EvaluationResult`** — remove `final_score`, remove `decision`, remove the `@model_validator` enforcing `0.4*style+0.4*ground+0.2*conf`. Final fields: `style_score: float`, `groundedness_score: float`, `confidence_score: float`, `explanation: str`, `flags: list[str]` (default `[]`). Set `model_config = ConfigDict(extra="forbid")` so any v1 caller still passing `final_score` (or `decision`) **fails loudly** rather than silently dropping the field — aligns with the Cohere fail-loudly principle in constraint #3.
- **`CloneResponse`** — NEW model: `response_text: str`, `citations: list[Citation]`. CloneAgent's Instructor `response_model`.
- **Audit against §5.1 contracts (note mismatches in session notes, do not over-engineer):**
  - `RetrievalResult` (chunk, score, rank) — matches CloneAgent `chunks` input. OK.
  - `Citation` (chunk_id, source_topic, text_snippet, relevance_score) — LLM cannot reliably emit `chunk_id`/`relevance_score`. Plan: the LLM emits citations keyed by **chunk index** (0-based position in the input `chunks` list); the agent code maps each index → the full `Citation` (`chunk_id`, `source_topic`, `text_snippet`, `relevance_score`) from the corresponding input chunk, and drops (with a log) any citation whose index is out of range. This is a reconciliation step, not a schema change.
  - `KnowledgeChunk` — confirm it satisfies CloneAgent/Evaluator inputs; expected OK as-is.
  - `StyleProfile` — **AUDITED (this session):** current fields are `leader_name`, `features`, `style_vector`, `email_count`, `last_updated`, `alpha` — **NO sample-emails field.** §5.1.1 requires CloneAgent to receive "3-5 sample emails for in-context style examples." **Decision (Ruby, this session):** add `sample_emails: list[str]` (default `[]`) to `StyleProfile` here in Phase 1; `StyleProfileBuilder` populates it in Phase 2 (see below). This carries forward already-cleaned text only — it does NOT change the §4.8 pipeline or the 15 features, so the ADR-013 freeze is not violated. This is the **one sanctioned modification to StyleProfileBuilder** under the freeze, explicitly approved.
- **Do NOT touch:** `CloneState` (no `routing_decision` — Day 11), no `RoutingDecision` model (Day 11), `FallbackResponse`/`StyledResponse` (leave v1 form).

**Test focus:** schema unit checks — `EvaluationResult` constructs with 5 fields and **rejects** `final_score`/`decision` (raises `ValidationError` under `extra="forbid"`); `CloneResponse` round-trips.

**Exit gate:**
- `grep final_score src/schemas.py` → empty (or deprecation comment only).
- `EvaluationResult` has exactly the 5 confirmed fields; no formula validator.
- `CloneResponse` exists. `CloneState` unchanged; no `RoutingDecision`.
- **Note:** this intentionally breaks v1 `evaluator.py` construction. Confirm v1 `evaluator.py` is left on disk untouched (dead until Day 11) and is not imported by any Phase 2–3 code. If a v1 test exercises it, skip/xfail rather than touch the logic.

---

## Phase 2 — Components (LLM-free; `src/components/`)

Re-read this plan from disk before starting. Create `src/components/` (does not yet exist) with `__init__.py`.

**Analysis-first (do before writing any Component):** read the v1 source you are about to wrap (`src/rag/`, `src/style/`, `src/evaluation/`) and produce a short per-function map — for each function: keep as-is / restructure / wrap behind the Component. Decide this from the actual code, then implement. Do not pre-commit to function-level decisions in this plan; they belong to execution time against the live source.

**What "refactor, not rewrite" means here:** the *behavior* of frozen logic must not change — the §4.8 cleaning pipeline and the 15 features (ADR-013) and the scoring math (ADR-003/004) must produce the same outputs. But you are NOT required to leave the surrounding code untouched: improving function decomposition, signatures, naming, type hints, or module structure to make a Component clean and testable is encouraged where it does not alter those frozen outputs. The test for an allowed change is "outputs identical, structure better," not "no lines moved." When unsure whether a change is behavioral, prefer wrapping and note it in the Dead Code Ledger.

Per file, the §12.2 disposition:

1. **`retriever.py`** — *wrap* `src/rag/` (`retriever.py` FAISS search, `reranker.py` Cohere, `embedder.py`). Reclassifies v1 `src/agents/rag_agent.py` façade as a Component (class with `run()`). Do NOT rewrite the FAISS/embedding logic.
   - **Cohere must actually execute (constraint #3 / Rule 6):** fix the empty-string silent default — read `COHERE_API_KEY`, log a loud WARNING if missing/empty, keep ADR-002 graceful fallback to FAISS top-5. Expose whether rerank ran (e.g. a flag/log the test can assert on).
   - Pipeline: FAISS top-20 → Cohere top-5 → `list[RetrievalResult]`.
2. **`style_profile_builder.py`** — *wrap* `src/style/` (`email_parser.py` §4.8 pipeline, `feature_extractor.py` 15 features, `profile_builder.py`). **Frozen per ADR-013** — no edits to the 8-step cleaning pipeline or the 15 features. Component exposes `run()` over the existing functions. **One sanctioned addition (approved by Ruby):** populate the new `StyleProfile.sample_emails` field by sampling 3-5 already-cleaned emails from the cleaned set during profile build. This carries cleaned text forward only — it does not touch the cleaning pipeline or the features, so the freeze holds. (Note in the Dead Code Ledger / session notes as the one freeze exception.)
3. **`scoring_engine.py`** — *wrap* the three v1 sub-scorers `src/evaluation/{style_scorer,groundedness_scorer,confidence_scorer}.py` into a `score()` method returning the three deterministic scores. **Do NOT wrap `evaluator.py`** (its weighted-formula combination + LLM explanation are gone / move to EvaluatorAgent). Don't rewrite the cosine/sentence math.

**LLM-free rule (ADR-007/009):** no `litellm`/`openai`/`cohere`/`instructor` imports in `src/components/`. (Note: `cohere` is used inside `src/rag/reranker.py`, called *by* the Retriever Component — the Component module itself must not import an LLM/reranker client directly; the grep target is the component file's own imports.)

**Adapter boundary (ADR-008):** no imports from `cli.py` or `streamlit_app.py`.

**Latency:** capture at the dict layer (the Flow/orchestration boundary), not as a Pydantic field. Latency is observability output, not a domain contract: per ADR-016 it is serialized per-query into the `cli evaluate` JSON, and PRD §6.1 defines it as per-stage budgets (retrieve <1s, clone <3s, evaluate <2s, route <2s) the Flow owns. Components/Agents return only their domain output (`RetrievalResult`, `CloneResponse`, `EvaluationResult`); the caller times them. Do NOT add a latency field to any result model — confirmed with Ruby this session; doing so would override constraint #9 and require an ADR.

**Test focus (§7.2):** Retriever — FAISS→Cohere, fallback path, **Cohere actually invoked (spy/mock assertion)**, citation coverage, **latency <1s cold (smoke)**. StyleProfileBuilder — mbox parse, `From:` filter, 8-step pipeline runs, 15 features per email, self-similarity > 0.70 on a small sample (no per-call latency budget — offline, runs once). ScoringEngine — style cosine, groundedness sentence math, confidence ∈ [0,1], edge cases (empty response, no chunks), **latency <500ms (smoke)**.

**Latency smoke checks (PRD §2.3/§2.4 — observe early, don't wait for Day 12):** each Component and Agent unit test wraps a single real (non-mocked) smoke call in `time.perf_counter()` and asserts under its budget. One call each — these are smoke checks, not benchmarks. Budgets: Retriever <1s cold (warm path not required Day 10), ScoringEngine <500ms, StyleProfileBuilder none (offline), CloneAgent <3s, EvaluatorAgent <2s. If a real LLM call is impractical in CI, run latency on a recorded-response replay and assert the deterministic portion is within budget, documenting the split in session notes. (Latency is still NOT a result-model field — see the Latency note above; these assertions live in the tests, timed by the test harness.)

**Exit gate (also a STOP gate before Phase 3):**
- 3 Components instantiate and `run()`/`score()` on a smoke input.
- Grep confirms no LLM imports in `src/components/`.
- Cohere reranking verified to run (test asserts client invoked).
- v1 `src/rag/`, `src/style/`, `src/evaluation/` left in place (no deletions — STOP gate before any v1 deletion; default leave until Day 11).
- **STOP between Components and Agents** — confirm with Ruby before proceeding.

---

## Phase 3 — Agents (real CrewAI Agents; `src/agents/`)

Re-read this plan from disk before starting. Each Agent = class wrapping a CrewAI `Agent` (role/goal/backstory) + single `Task` + single-agent `Crew` (§5.1). Structured output via Instructor + Pydantic v2 — no raw `json.loads` (Rule 4). Model: gpt-4o-mini via LiteLLM.

1. **`clone_agent.py`** — *rename* v1 `src/agents/style_crew.py` → `clone_agent.py`; stays an Agent. Adapt to §5.1: inputs `query`, `leader`, `style_profile`, `chunks`; output `CloneResponse` (response_text + citations) via Instructor. Reconcile LLM-emitted citations to full `Citation` objects from input chunks (see Phase 1 note). temperature=0.3. Adapt the existing prompt scaffolding — don't rewrite what already works.
   - **Test focus:** output is styled response + citations; prompt includes leader style features, the 3-5 `style_profile.sample_emails` (in-context style examples, §5.1.1), and chunks; Instructor parses `CloneResponse`; **latency <3s smoke** (real or recorded-replay, per the Phase 2 latency-smoke note).

2. **`evaluator_agent.py`** — *rewrite* (v1 had `evaluator.py` weighted formula + Python functions; no real Agent existed). **STOP gate before implementing the hybrid.** Hybrid call sequence (ADR-011), be explicit:
   - Step 1: call `ScoringEngine.score()` → `style_score`, `groundedness_score`, `confidence_score` (deterministic, no LLM).
   - Step 2: ONE LLM call (temperature=0, Instructor) reads response + chunks + the three scores → produces `explanation` (references the scores) + `flags: list[str]`.
   - Assemble `EvaluationResult` (5 fields, no `final_score`).
   - **Test focus:** calls ScoringEngine; LLM generates explanation referencing scores; `flags` populated correctly; NO `final_score` field present; **latency <2s smoke** (real or recorded-replay, per the Phase 2 latency-smoke note).

**Constraints:** no GatekeeperAgent/FallbackAgent (Day 11). Adapter boundary holds (no cli/streamlit imports).

**Exit gate:**
- CloneAgent generates a response from synthetic inputs.
- EvaluatorAgent produces `EvaluationResult` with three scores + explanation + flags.
- Unit tests pass for both Agents (LLM responses recorded/mocked for CI).

---

## Phase 4 — Wrap-up, verification, session notes

Re-read this plan from disk before starting.

- Run full architecture-honesty grep checks (from CLAUDE.md): Agents import `from crewai import Agent`; Agents have role/goal/backstory; Components have `run()` and no `litellm|openai|cohere|instructor` imports; no `final_score` in `schemas.py`; no Agent-suffixed functions outside `src/agents/`.
- **`final_score` repo-wide grep (pre-empts PRD §12.5 audit category 3, which greps `src/` at Day 14):** `grep -rn "final_score" src/ --include="*.py"` returns empty, except: (a) deprecation comments, which must name the file/Day they will be removed and be recorded in the Dead Code Ledger; (b) v1 files left on disk (e.g. `src/evaluation/evaluator.py`) — these are **expected leaks**, logged in the Dead Code Ledger with their trigger Day, and the grep result is annotated as expected rather than treated as a failure. A `final_score` match in any *new* Day-10 file (Components, Agents, schemas) IS a failure.
- Run unit tests for all five pieces; `ruff` + `mypy` clean on touched files.
- Write `docs/session-notes/day10.md`: what was wrapped/renamed/rewritten, the schema mismatches noted (Citation reconciliation), Cohere fix, any deferrals to Day 11.
- **Dead Code Ledger** — record in `docs/session-notes/day10.md` each v1 file/field that became dead during Day 10, plus the trigger Day on which it becomes safe to delete. Expected entries (confirm during execution):
  - `src/agents/rag_agent.py` — superseded by `src/components/retriever.py`; deletable once Flow imports the Component instead (Day 11).
  - `src/agents/style_crew.py` — renamed to `clone_agent.py`; if a copy is left behind rather than `git mv`, it is dead immediately.
  - `src/evaluation/evaluator.py` — weighted-formula combination + LLM explanation moved to EvaluatorAgent; deletable once Flow stops calling `evaluate()` (Day 11). `EvaluationResult` change breaks its construction in the meantime — skip/xfail any test that exercises it.
  - Any v1 test exercising the above — note for skip/xfail now, removal at the same trigger Day.
- **Promote the retirement schedule to the PRD:** annotate the existing §12.2 v1→v2 mapping table with a "retire when" note per affected row (e.g. "delete after Day 11 Flow refactor"), so the next day's prompt reads §12.2 and acts on it. This is a lightweight annotation, not a new section or a logic change. Deletions still pass through Stop Gate 1.
- **No new ADR.** If a decision surfaces that needs one → STOP and surface to Ruby (do not silently choose).

---

## End-of-day exit criteria (checklist)
- [ ] `schemas.py` updated; `EvaluationResult` has no `final_score` (and no `decision`/formula validator); `flags` added; `CloneResponse` added.
- [ ] 3 Components instantiate and `run()` on smoke input.
- [ ] CloneAgent generates a response from synthetic inputs.
- [ ] EvaluatorAgent produces `EvaluationResult` (3 scores + explanation + flags).
- [ ] Unit tests pass per §7.2 for all five pieces.
- [ ] Grep: no LLM imports in `src/components/`.
- [ ] Cohere reranking verified to run.
- [ ] `docs/session-notes/day10.md` written.
- [ ] Dead Code Ledger recorded in day10 notes; PRD §12.2 annotated with "retire when" triggers.
- [ ] No new ADR needed (else STOP).

## Stop gates
1. Before deleting any v1 code under `src/` (default: leave in place).
2. Before adding any `pyproject.toml` dependency (all needed deps — crewai, instructor, litellm, cohere, faiss-cpu, pydantic v2 — already present).
3. Between Components (Phase 2) and Agents (Phase 3).
4. Before EvaluatorAgent hybrid implementation (within Phase 3).
