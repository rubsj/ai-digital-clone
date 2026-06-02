# Evaluation Methodology

**Project:** P6 — Torvalds Digital Clone
**ADR reference:** ADR-016 (Three-Layer Testing Strategy)
**Authored:** Day 12

---

## Three-Layer Architecture

The evaluation stack has three layers. Each layer has a distinct role, failure mode, and evidence artifact.

### Layer 1 — Unit (continuous)

**What it covers.** The `ScoringEngine` component scores each response on three dimensions: style similarity (cosine similarity against the leader's `StyleProfile.style_vector`), groundedness (mean cosine similarity of response embeddings against retrieved chunk embeddings), and confidence (presence of hedging language detected via pattern matching).

**What it does not cover.** The LLM-driven agents — CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent. Those have behavior that depends on model temperature and prompt phrasing, not just deterministic computation.

**Evidence.** Unit tests in `tests/test_scoring_engine.py`. Run on every commit via CI. Failures here indicate a regression in the deterministic computation layer, not an agent behavior change.

### Layer 2 — Per-agent (recorded-LLM contract tests)

**What it covers.** Each agent's input-output contract: schema validation (Pydantic), field presence, and the structural rules that do not require a live model call (for example, `trigger_category` must be null on deliver decisions and non-null on fallback decisions).

**What it does not cover.** Whether the agent's reasoning is correct, whether groundedness flags are raised at the right threshold, or whether the GatekeeperAgent's routing decision is calibrated correctly. Those require real LLM calls.

**Evidence.** Tests in `tests/` for each agent. The `test_compare_leaders.py::test_compare_leaders_retriever_called_once` gate confirms the ADR-005 shared-retrieval guarantee at this layer.

### Layer 3 — System (query set evaluation)

**What it covers.** End-to-end behavior of the full pipeline — retrieval, clone, evaluation, routing, delivery or fallback — against a representative query set. This is the only layer that observes the interaction between agents and the real LLM responses they produce.

**What it does not cover.** Per-user personalization (P6 v2 is two fixed leaders). Latency under load (Day 12 measures single-threaded latency only). Latency SLA compliance is measured but not gated at Layer 3 tonight (§2.7 investigation is a Day-13 item).

**Evidence.** `src/eval/harness.py`, `run_measurement()`. Results in `results/evaluation_day12.json`. 48 pair-records across three passes. Deliver rates, OOD fallback rates, and per-leader score distributions are the primary outputs.

---

## Run Design

The C4 run design was adopted to separate cost concerns from measurement concerns.

**Pre-flight.** Two queries (one in-domain, one OOD) are run before the full pass. The pre-flight costs approximately 32 chat completions and confirms that API keys, FAISS index, and profile paths are valid before spending 460-610 completions on the full pass.

**Pass 1 (full).** All 20 queries against both leaders = 40 pair-records. OOD fallback rate, hallucination count, and per-leader deliver rate point estimates are read from pass 1.

**Passes 2-3 (in-domain only).** The 14 in-domain queries are re-run twice to estimate variance from CloneAgent temperature=0.3 stochasticity. Pass 2 and 3 each produce 14 pair-records.

**Reactive OOD recheck.** If any OOD record delivers in pass 1, the specific query is re-run twice before classifying the deliver as a hallucination. In Day-12 measurement, no OOD delivers occurred and the recheck was skipped.

---

## Category Classification

The in-domain versus OOD axis is derived from the query's `category` field via `classify_category()` in `src/eval/harness.py`. The two named sets are:

- `IN_DOMAIN_CATEGORIES`: `statistical_learning_ml`, `data_mining`, `numerical_methods`, `programming_fundamentals`
- `OOD_CATEGORIES`: `systems_absent_from_corpus`, `off_topic_technical`

A category in neither set raises `ValueError` — fail loud, not silent misclassification. The `expected_behavior` field in `data/eval/queries.json` is used only as the grading target (the human-intended answer) and is not used for axis assignment.

---

## One-Retrieval Invariant

The harness wraps `Retriever.run` with a call counter and asserts that exactly one retrieval call occurs across both flows. Torvalds retrieves; Kroah-Hartman receives pre-populated chunks via `kickoff(inputs={"chunks": shared_chunks})` and early-exits the retrieve step (ADR-005). The assertion mirrors `tests/integration/test_compare_leaders.py::test_compare_leaders_retriever_called_once`. If this assertion fails, the measurement is aborted.

---

## Ungrounded Span Analysis

For every in-domain fallback record, the `clone_response_text` (the CloneAgent's raw output that the EvaluatorAgent scored) and `chunk_contents` (the five retrieved chunks with rank, score, and full text) are persisted in the results JSON. Post-run analysis compares these to identify spans in the response that go beyond the retrieved text. This analysis requires no additional LLM calls.

Failure modes observed in Day-12 measurement:

- **q01-pattern.** The CloneAgent generates an accurate response that is mostly grounded in the chunks, then adds 1-2 sentences of contextual advice (typically "when would you use this method") that generalize beyond what the chunks state.
- **Thin or absent retrieval.** The rank-0 chunk either scores below 0.40 or contains a definition/formula without explanatory content. The response correctly describes the topic but draws from model training knowledge rather than the retrieved text.
- **EvaluatorAgent over-flagging.** The response closely follows chunk content but the EvaluatorAgent LLM assigns a groundedness score below the flag threshold and raises `low_groundedness`, causing the GatekeeperAgent to route to fallback.

---

## Groundedness Scoring Architecture

The groundedness score is computed deterministically by `ScoringEngine`: mean cosine similarity between the response sentence embeddings and the retrieved chunk embeddings. The target is 0.60. The EvaluatorAgent then receives this score and, in a separate LLM call, decides whether to raise a `low_groundedness` flag.

The GatekeeperAgent receives the three numerical scores and the flag list and decides deliver or fallback. The default prompt instruction is "Default: DELIVER. Route to FALLBACK only when a specific score or flag justifies it."

In Day-12 measurement, the LLM inside the EvaluatorAgent raised `low_groundedness` for scores up to 0.706, above the 0.60 target. The GatekeeperAgent followed the flag and routed to fallback. This is the primary failure mode identified at the gate.
