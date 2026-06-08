# A2 — Single-Query Sequence

> ADR-014 inventory: 3 Agents · 4 Components · 1 Flow. Single call to `DigitalCloneFlow.kickoff()`.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant DCF as DigitalCloneFlow
    participant RT  as Retriever<br/>(Component)
    participant CA  as CloneAgent<br/>(Agent)
    participant EA  as EvaluatorAgent<br/>(Agent)
    participant SE  as ScoringEngine<br/>(Component)
    participant GK  as Gatekeeper<br/>(Component)
    participant FA  as FallbackAgent<br/>(Agent)

    User->>DCF: kickoff(query, leader, style_profile)

    Note over DCF,RT: Step 1 — @start retrieve()
    DCF->>RT: run(query)
    RT-->>DCF: list[RetrievalResult] (top-5 reranked chunks)

    Note over DCF,CA: Step 2 — @listen(retrieve) clone()
    DCF->>CA: kickoff(query, chunks, style_profile)
    CA-->>DCF: CloneResponse (response_text + citations)

    Note over DCF,SE: Step 3 — @listen(clone) evaluate()
    DCF->>SE: run(response_text, chunks, style_profile)
    SE-->>EA: style_score, groundedness_score, confidence_score
    DCF->>EA: kickoff(scores, response_text, chunks)
    EA-->>DCF: EvaluationResult (scores + explanation + flags)

    Note over DCF,GK: Step 4 — @router(evaluate) route()
    DCF->>GK: run(evaluation, query)
    GK-->>DCF: RoutingDecision (deliver | fallback)

    alt decision == "deliver"
        Note over DCF: Step 5a — @listen("deliver") finalize()
        DCF-->>User: StyledResponse (response + evaluation + citations)
    else decision == "fallback"
        Note over DCF,FA: Step 5b — @listen("fallback") handle_fallback()
        DCF->>FA: kickoff(query, leader, routing_decision)
        FA-->>DCF: FallbackResponse (acknowledgment + redirections)
        DCF-->>User: StyledResponse (fallback=FallbackResponse)
    end
```

## Notes

- **Gatekeeper is deterministic** — arithmetic threshold comparison, no LLM (ADR-018). It reads `GROUNDEDNESS_MIN = 0.40` from EvaluatorAgent's constants and routes based on scores and flags.
- **ScoringEngine** is called inside the evaluate step before EvaluatorAgent; it provides the three numerical scores. EvaluatorAgent adds the LLM-generated explanation and flags.
- **FallbackAgent** carries a hardcoded template failsafe: if its LLM call fails, a short unstyled response returns so the user always gets a reply.
- **CloneState** is the typed Pydantic state object passed between all steps — each step reads prior outputs from state and writes its result back.

> **PRD §7.5.3 prose contradiction (deferred reconciliation):** The PRD §7.5.3 A2 description reflects the pre-ADR-018 routing step (Gatekeeper as an LLM Agent). This diagram follows ADR-014/ADR-018 — Gatekeeper is a deterministic Component.
