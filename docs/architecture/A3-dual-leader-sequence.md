# A3 — Dual-Leader Sequence

> ADR-014 inventory: 3 Agents · 4 Components · 1 Flow. `compare_leaders()` pattern — retrieve once, evaluate per-leader (ADR-005).

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant CL  as compare_leaders()
    participant RT  as Retriever<br/>(Component)
    participant DCF_T as DigitalCloneFlow<br/>[Torvalds]
    participant DCF_K as DigitalCloneFlow<br/>[Kroah-Hartman]
    participant CA  as CloneAgent<br/>(Agent)
    participant EA  as EvaluatorAgent<br/>(Agent)
    participant SE  as ScoringEngine<br/>(Component)
    participant GK  as Gatekeeper<br/>(Component)
    participant FA  as FallbackAgent<br/>(Agent)

    Caller->>CL: compare_leaders(query)

    Note over CL,RT: Shared retrieval — once for both leaders (ADR-005)
    CL->>RT: run(query)
    RT-->>CL: list[RetrievalResult] (top-5 chunks, shared)

    par Per-leader evaluation — Torvalds
        Note over CL,DCF_T: Flow kickoff for Torvalds
        CL->>DCF_T: kickoff(query, "Linus Torvalds", chunks, style_profile_T)
        DCF_T->>CA: clone (Torvalds voice)
        CA-->>DCF_T: CloneResponse
        DCF_T->>SE: score
        SE-->>EA: scores
        DCF_T->>EA: evaluate
        EA-->>DCF_T: EvaluationResult
        DCF_T->>GK: route
        GK-->>DCF_T: RoutingDecision
        alt deliver
            DCF_T-->>CL: StyledResponse[Torvalds]
        else fallback
            DCF_T->>FA: handle_fallback
            FA-->>DCF_T: FallbackResponse
            DCF_T-->>CL: StyledResponse[Torvalds, fallback]
        end
    and Per-leader evaluation — Kroah-Hartman
        Note over CL,DCF_K: Flow kickoff for Kroah-Hartman
        CL->>DCF_K: kickoff(query, "Greg Kroah-Hartman", chunks, style_profile_KH)
        DCF_K->>CA: clone (KH voice)
        CA-->>DCF_K: CloneResponse
        DCF_K->>SE: score
        SE-->>EA: scores
        DCF_K->>EA: evaluate
        EA-->>DCF_K: EvaluationResult
        DCF_K->>GK: route
        GK-->>DCF_K: RoutingDecision
        alt deliver
            DCF_K-->>CL: StyledResponse[KH]
        else fallback
            DCF_K->>FA: handle_fallback
            FA-->>DCF_K: FallbackResponse
            DCF_K-->>CL: StyledResponse[KH, fallback]
        end
    end

    CL-->>Caller: LeaderComparison(query, torvalds=..., kroah_hartman=...)
```

## Notes

- **Retrieve-once optimization (ADR-005):** The top-5 reranked chunks are fetched once and shared between both leader flows. This halves the OpenAI embedding call + FAISS search + Cohere rerank cost for a dual-leader query.
- **Two independent Flow instances** run after the shared retrieval — each carries a per-leader `StyleProfile` from disk cache. The flows do not share state.
- **Gatekeeper** applies the same arithmetic thresholds for both leaders independently — the deliver/fallback decision is per-response, not per-leader.
- The `LeaderComparison` output carries either `StyledResponse` or `FallbackResponse` for each leader position, so the caller always gets both slots filled.

> **PRD §7.5.3 prose contradiction (deferred reconciliation):** The PRD §7.5.3 A3 description reflects the pre-ADR-018 per-leader path (Gatekeeper as an LLM Agent). This diagram follows ADR-014/ADR-018 — Gatekeeper is a deterministic Component.
