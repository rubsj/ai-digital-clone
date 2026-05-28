# ADR-005: Shared RAG Retrieval for Dual-Leader Comparison Mode

**Project:** P6: Torvalds Digital Clone
**Category:** Performance / Orchestration
**Status:** Accepted
**Date:** 2026-04-26

---

## Context

Comparison mode runs the full pipeline twice for one query: once styled as Torvalds, once as Kroah-Hartman. Each run does retrieve, then style, then evaluate.

The style and evaluate steps have to run per leader. They take the leader profile as input, so the outputs differ. Retrieval does not. Both leaders answer the same question against the same knowledge base, so the retrieved chunks are identical for both runs.

Retrieval is the most expensive step. It does an OpenAI embed + FAISS top-20 + Cohere rerank, and returns the top-5 chunks. The full round-trip costs roughly 600ms in production.

The naive implementation runs two independent `DigitalCloneFlow` instances back to back. Each one runs the Retriever Component for itself, paying the embed + FAISS + rerank cost twice for chunks that are guaranteed to be identical. The fix needs to share retrieved chunks between the two runs without coupling their error paths or polluting `DigitalCloneFlow` with knowledge of the comparison use case.

---

## Decision

The retrieve step in `DigitalCloneFlow` early-exits when `CloneState.chunks` is already populated:

```python
@start()
def retrieve(self) -> None:
    if self.state.chunks:   # dual-leader: chunks already injected
        return
    self.state.chunks = Retriever().run(self.state.query)
```

The field is `CloneState.chunks` in the v2 schema (PRD §5.4), renamed from the v1 `retrieved_chunks`; the snippets here reflect the planned v2 Flow (PRD §5.5), which lands during the Day 10 rework.

A thin wrapper, `compare_leaders(query)`, orchestrates two sequential flow runs. The first run (Torvalds) performs retrieval normally. After it completes, `compare_leaders` snapshots the chunks from the flow's state proxy and passes them as input to the second run (Kroah-Hartman) via `kickoff(inputs={"chunks": ...})`. The second run's retrieve step sees the pre-populated list and skips the embed + FAISS + rerank path.

```python
def compare_leaders(query: str) -> LeaderComparison:
    flow_t = DigitalCloneFlow()
    flow_t.kickoff(inputs={"query": query, "leader": _LEADERS[0]})
    shared_chunks = list(flow_t.state.chunks)   # snapshot from StateProxy
    flow_kh = DigitalCloneFlow()
    flow_kh.kickoff(inputs={
        "query": query, "leader": _LEADERS[1], "chunks": shared_chunks,
    })
    ...
    return LeaderComparison(query=query, torvalds=t_out, kroah_hartman=kh_out)
```

The two runs are sequential. If Torvalds' retrieval fails there are no chunks to share, and running Kroah-Hartman concurrently would just produce a second failure from empty context.

**A2: Single-query pipeline (baseline)**

```mermaid
sequenceDiagram
    participant U as User
    participant F as DigitalCloneFlow
    participant R as Retriever (Component)
    participant C as CloneAgent
    participant E as EvaluatorAgent
    participant G as GatekeeperAgent
    participant FB as FallbackAgent

    U->>F: kickoff(query, leader)
    F->>R: run(query)
    R-->>F: chunks [top-5]
    F->>C: generate styled response(profile, chunks, query)
    C-->>F: response_text + citations
    F->>E: evaluate(response, chunks, profile, query)
    E-->>F: EvaluationResult {scores, explanation, flags}
    F->>G: route(scores, chunks, response)
    G-->>F: RoutingDecision {decision}
    alt decision == "deliver"
        F-->>U: StyledResponse
    else decision == "fallback"
        F->>FB: generate fallback(query, leader, context)
        FB-->>F: FallbackResponse
        F-->>U: FallbackResponse
    end
```

**A3: Dual-leader comparison (retrieve-once optimization)**

```mermaid
sequenceDiagram
    participant U as User
    participant W as compare_leaders()
    participant F1 as DigitalCloneFlow (Torvalds)
    participant F2 as DigitalCloneFlow (Kroah-Hartman)
    participant R as Retriever (Component)
    participant C1 as CloneAgent (Torvalds)
    participant C2 as CloneAgent (Kroah-Hartman)
    participant E as EvaluatorAgent
    participant G as GatekeeperAgent
    participant FB as FallbackAgent

    U->>W: compare_leaders(query)
    W->>F1: kickoff(query, "torvalds")
    F1->>R: run(query)
    R-->>F1: chunks [top-5]
    F1->>C1: generate styled response(torvalds_profile, chunks, query)
    C1-->>F1: response_text + citations
    F1->>E: evaluate(torvalds_response, chunks, torvalds_profile, query)
    E-->>F1: EvaluationResult {scores, explanation, flags}
    F1->>G: route(scores, chunks, response)
    G-->>F1: RoutingDecision {decision}
    alt decision == "fallback"
        F1->>FB: generate fallback
        FB-->>F1: FallbackResponse
    end
    F1-->>W: state.chunks (snapshot)
    W->>F2: kickoff(query, "kroah_hartman", chunks=shared)
    Note over F2: retrieve step early-exits, chunks already present
    F2->>C2: generate styled response(kh_profile, shared chunks, query)
    C2-->>F2: response_text + citations
    F2->>E: evaluate(kh_response, shared chunks, kh_profile, query)
    E-->>F2: EvaluationResult {scores, explanation, flags}
    F2->>G: route(scores, chunks, response)
    G-->>F2: RoutingDecision {decision}
    alt decision == "fallback"
        F2->>FB: generate fallback
        FB-->>F2: FallbackResponse
    end
    W-->>U: LeaderComparison {torvalds, kroah_hartman}
```

---

## Alternatives Considered

**Independent pipelines per leader.** Run two separate `DigitalCloneFlow` instances with no chunk sharing, each running the Retriever Component for itself. This is what you get if you call the Flow twice. Simpler in that there's no shared state between runs, but it doubles the cost of the most expensive step in the pipeline. The Phase 4 timing harness measured the difference; numbers are below.

**Cached RAG with TTL keyed on query hash.** Introduce a cross-request cache layer (Redis or in-process LRU) that stores `retrieved_chunks` by `hash(query)` with a short TTL. A cache hit would skip retrieval on any future call with the same query, not just the second leader of the current comparison. I rejected this because the dual-leader comparison is a single synchronous request. Both leader pipelines run within the same Python process in the same ~500ms window, so a cross-request cache would be populated and then expire before any future request could hit it. The cache adds a client, TTL handling, and an invalidation path tied to knowledge-base updates, none of which earn anything for the actual use case. State threading through the wrapper does the same job and disappears when the request completes.

---

## Quantified Validation

Measurements from `scripts/timing_dual_leader.py`, run 2026-04-26 on Python 3.13.12 / macOS 25.4.0. All LLM calls mocked with a fixed 50ms `time.sleep`; `RAGAgent.retrieve` mocked with a fixed 100ms `time.sleep` to simulate embed + FAISS + Cohere rerank. Five-run average reported.

| Approach | Avg latency |
|---|---|
| `compare_leaders()` (shared retrieval) | 413.6 ms |
| Two independent `DigitalCloneFlow` runs | 460.9 ms |
| Savings | 47.3 ms (10.3%) |

The savings are smaller than the back-of-envelope prediction of 100ms (one avoided RAG mock). The gap comes from `DigitalCloneFlow.__init__` and `kickoff()` overhead. Each flow instance pays a per-run setup cost: state object creation, async event loop entry, CrewAI lifecycle hooks. Both paths instantiate two flows, so both pay that cost twice and the harness sees it in both totals. That overhead is independent of query complexity or knowledge-base size, so in production with real retrieval costs the savings move toward the cost of one full retrieval call.

These numbers were measured under the v1 step shape, where the harness mocked `RAGAgent.retrieve` and `StyleCrew`. The retrieve-once decision does not depend on what runs inside each step, so it carries into the v2 pipeline where the Retriever Component replaces the retrieve mock and CloneAgent replaces StyleCrew. The absolute latencies will shift when re-measured against the real v2 steps.

---

## Consequences

The shared-chunk pattern introduces one coupling point between the two runs: `compare_leaders` reads `flow_t.state.chunks` after the first run completes and passes it into the second. If the first run fails during retrieval (empty chunks, network error), the second run starts with an empty list and proceeds through style and evaluate with no context. An empty `chunks` list produces a low-quality response that GatekeeperAgent routes to fallback, so the second leader returns a `FallbackResponse` rather than a `StyledResponse`. `compare_leaders` then raises `ValueError` naming which leader's pipeline did not produce a styled response.

`compare_leaders` is the only code that knows about dual-leader orchestration. `DigitalCloneFlow` has no awareness of other flow instances; the comparison runs two independent Flow instances of the same orchestrator defined in ADR-001 (rewritten). The early-exit guard (`if self.state.chunks: return`) is a general optimization, not a comparison-mode hook. A future "compare N leaders" mode would loop the injection pattern in the wrapper without changing the Flow class.

The retrieve-once pattern is request-scoped. It does not persist beyond one `compare_leaders` call, does not interact with any caching infrastructure, and does not need invalidation when the knowledge base changes. (In Spring the equivalent would be passing a request-scoped bean through a chain of service calls rather than going to a shared cache.)
