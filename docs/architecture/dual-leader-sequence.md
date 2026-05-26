# A3 — Dual-Leader Sequence

Runtime path for `compare_leaders(query)`. RAG retrieval runs once; the
shared chunk set is reused for both style+evaluate branches. This avoids
a second embedding + rerank round-trip for the same query text.

```mermaid
sequenceDiagram
    actor User
    participant CL as compare_leaders
    participant RA as RAGAgent
    participant SC1 as StyleCrew (Torvalds)
    participant EA1 as EvaluatorAgent (Torvalds)
    participant SC2 as StyleCrew (Kroah-Hartman)
    participant EA2 as EvaluatorAgent (Kroah-Hartman)

    User->>CL: compare_leaders(query)

    CL->>RA: retrieve(query)
    RA-->>CL: shared chunks + citations

    par Torvalds branch
        CL->>SC1: generate(query, torvalds, chunks)
        SC1-->>CL: StyledResponse
        CL->>EA1: evaluate(response, torvalds, chunks)
        EA1-->>CL: EvaluationResult
    and Kroah-Hartman branch
        CL->>SC2: generate(query, kroah_hartman, chunks)
        SC2-->>CL: StyledResponse
        CL->>EA2: evaluate(response, kroah_hartman, chunks)
        EA2-->>CL: EvaluationResult
    end

    CL-->>User: LeaderComparison(torvalds_result, kroah_hartman_result)
```
