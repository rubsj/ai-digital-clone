# A5 — Data Flow

> Two swim lanes: Offline (run once, cached to disk) and Online (per-query pipeline).

```mermaid
graph LR
    subgraph OFFLINE["OFFLINE  —  runs once, results cached to disk"]
        direction TB

        subgraph StyleLearning["Style Learning"]
            MBOX["mbox archives\ndata/emails/"]
            EP["EmailParser\n(parse + clean)"]
            FE["FeatureExtractor\n(15 features per email)"]
            SPB_off["StyleProfileBuilder\n(aggregate → StyleProfile)"]
            SP_cache["StyleProfile\n(JSON cache, per leader)"]

            MBOX --> EP --> FE --> SPB_off --> SP_cache
        end

        subgraph RAGIndex["RAG Indexing"]
            CORPUS["Textbook corpus\n(Linux Kernel Development)"]
            CHUNK["Chunker\n(split + clean)"]
            EMBED["Embedder\n(OpenAI text-embedding-3-small)"]
            FAISS_build["FAISS index\ndata/rag/faiss_index/"]
            NPZ["Embedding cache\ndata/cache/*.npz"]

            CORPUS --> CHUNK --> EMBED
            EMBED --> FAISS_build
            EMBED --> NPZ
        end
    end

    subgraph ONLINE["ONLINE  —  per-query, real-time"]
        direction TB

        QUERY["User query\n+ leader"]
        SP_load["StyleProfile\n(load from cache)"]
        RT_on["Retriever\nembed query → FAISS top-20\n→ Cohere rerank top-5"]
        CA_on["CloneAgent\ngenerate in leader voice\n(OpenAI chat)"]
        SE_on["ScoringEngine\nstyle cosine + HHEM ground\n+ confidence"]
        EA_on["EvaluatorAgent\nLLM explanation + flags\n(OpenAI chat)"]
        GK_on["Gatekeeper\narithmetic deliver/fallback\n(GROUNDEDNESS_MIN = 0.40)"]

        QUERY --> RT_on
        SP_load --> CA_on
        RT_on --> CA_on
        CA_on --> SE_on
        SE_on --> EA_on
        EA_on --> GK_on

        GK_on -->|"deliver"| STYLED["StyledResponse\n(response + eval + citations)"]
        GK_on -->|"fallback"| FA_on["FallbackAgent\nacknowledgment + redirections"]
        FA_on --> FALLBACK["StyledResponse\n(fallback=FallbackResponse)"]
    end

    SP_cache -.->|"load at query time"| SP_load
    FAISS_build -.->|"loaded at query time"| RT_on
    NPZ -.->|"loaded at query time"| RT_on
```

## Notes

- **Offline lane runs once per corpus / leader** — output is disk-cached. No offline step runs at query time.
- **StyleProfileBuilder** (offline only) is not called during a query — `StyleProfile` is loaded from the JSON cache. The distinction matters: StyleProfileBuilder is a Component, but it does not appear in the online path.
- **HHEM groundedness** in ScoringEngine uses the vendored HHEM-2.1-Open model (ADR-020) loaded from `data/models/` — not an API call. The model is loaded at process start, not per-query.
- **Gatekeeper** threshold: `GROUNDEDNESS_MIN = 0.40` (HHEM scale). The threshold and floors are locked (ADR-015, ADR-020, ADR-018).
- **Retriever** loads FAISS index from disk on each flow start. The `*.npz` embedding cache accelerates re-indexing; the live query path uses only the FAISS index.
