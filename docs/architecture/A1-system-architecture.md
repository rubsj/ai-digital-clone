# A1 — System Architecture

> ADR-014 inventory: 3 Agents · 4 Components · 1 Flow. README hero diagram.

```mermaid
graph TB
    subgraph Adapters["Adapters"]
        CLI["CLI\n(Click, 5 commands)"]
        STA["Streamlit App\n(interactive demo)"]
    end

    subgraph DCF["DigitalCloneFlow  —  CrewAI Flow[CloneState]"]
        direction TB
        step1["@start  retrieve()"]
        step2["@listen clone()"]
        step3["@listen evaluate()"]
        step4["@router route()"]
        step5a["@listen finalize()"]
        step5b["@listen handle_fallback()"]

        step1 --> step2 --> step3 --> step4
        step4 -->|"deliver"| step5a
        step4 -->|"fallback"| step5b
    end

    subgraph LLM["LLM-Driven Agents  ·  src/agents/"]
        CA["CloneAgent\nGenerate response in leader voice"]
        EA["EvaluatorAgent\nScore response + produce explanation"]
        FA["FallbackAgent\nWrite graceful decline in leader voice"]
    end

    subgraph DET["Deterministic Components  ·  src/components/"]
        RT["Retriever\nEmbed → FAISS top-20 → Cohere rerank top-5"]
        SPB["StyleProfileBuilder\nParse mbox → 15 style features"]
        SE["ScoringEngine\nStyle / groundedness / confidence scores"]
        GK["Gatekeeper\nDeliver-or-fallback arithmetic router"]
    end

    subgraph EXT["External Services"]
        OAI["OpenAI\nEmbeddings + Chat completions"]
        COH["Cohere\nReranking"]
        FSS["FAISS\nVector index (disk)"]
    end

    CLI --> DCF
    STA --> DCF

    step1 --> RT
    step2 --> CA
    step3 --> EA
    step3 --> SE
    step4 --> GK
    step5b --> FA

    RT  --> OAI
    RT  --> FSS
    RT  --> COH
    CA  --> OAI
    EA  --> OAI
    FA  --> OAI
    SPB -.->|"offline build"| OAI
```

## Inventory (ADR-014, corrected 2026-06-01 per ADR-018)

| Kind | Count | Units |
|------|-------|-------|
| Agents (LLM-driven, `src/agents/`) | 3 | CloneAgent, EvaluatorAgent, FallbackAgent |
| Components (deterministic, `src/components/`) | 4 | Retriever, StyleProfileBuilder, ScoringEngine, Gatekeeper |
| Flow orchestrator | 1 | DigitalCloneFlow |

Gatekeeper is a deterministic Component (arithmetic router, no LLM) — reclassified from Agent per ADR-018.

> **PRD §7.5.3 prose contradiction (deferred reconciliation):** The PRD §7.5.3 A1 description reflects the pre-ADR-018 inventory (4 Agents + 3 Components, with Gatekeeper classified as an Agent). This diagram follows ADR-014 (the locked decision — 3 Agents + 4 Components); the PRD prose update is deferred to the full PRD reconciliation pass.
