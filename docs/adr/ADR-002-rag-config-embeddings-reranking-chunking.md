# ADR-002: RAG Configuration: Embeddings, Reranking, and Chunking

**Project:** P6: Torvalds Digital Clone
**Category:** RAG Pipeline
**Status:** Accepted
**Date:** 2026-04-13

---

## Context

The RAG pipeline (Day 3) retrieves chunks from FAISS and feeds them to the response agents. Before building any of that, I need to lock down the embedding model, chunk sizes, and whether a reranker sits between FAISS and the final results.

The `open-phi/textbooks` dataset has 1,511 computer science documents averaging ~129K characters each. A 500-char chunk size produces thousands of chunks from a 20-doc sample, enough to validate retrieval quality without loading the full corpus during development.

---

## Decision

OpenAI `text-embedding-3-small` (1536 dimensions) via LiteLLM for embeddings, with `all-MiniLM-L6-v2` (384d) as a local baseline for offline development and CI runs where OpenAI credentials aren't available. All vectors are L2-normalized before FAISS `IndexFlatIP` storage, so dot product equals cosine similarity for unit vectors.

Chunking uses `chunk_size=500`, `chunk_overlap=50` with RecursiveCharacterTextSplitter as the baseline. A semantic experiment using MarkdownHeaderTextSplitter into RecursiveCharacterTextSplitter runs alongside for comparison, sharing the same config values. Cohere `rerank-english-v3.0` then reranks the top-20 FAISS candidates down to top-5 final results, with a graceful fallback if the API fails (`try/except Exception` in `reranker.py` returns the original FAISS top-5 with a warning log).

```yaml
# configs/default.yaml
embedding:
  primary_model: "text-embedding-3-small"
  baseline_model: "all-MiniLM-L6-v2"
  dimension: 1536

chunking:
  chunk_size: 500
  chunk_overlap: 50

reranker:
  provider: "cohere"
  model: "rerank-english-v3.0"
  top_n_initial: 20
  top_n_final: 5
```

---

## Alternatives Considered

**MiniLM as primary embedding model** - `all-MiniLM-L6-v2` runs locally with no API cost and produces 384-d vectors. In the P2 evaluation grid, MiniLM hit Recall@5 of 0.61 vs OpenAI's 0.87. The gap is largest on technical kernel terminology ("memory-mapped I/O", "scheduler preemption") where OpenAI's training corpus has better coverage.

**Larger chunks (1000/100)** - More context per chunk, fewer total chunks. In the P2 grid search, larger chunks improved groundedness scores (less fragmentation) but dropped Recall@5 by ~12% because oversized chunks score lower against short technical queries. 500/50 was the better fit for LKML-style query distribution.

**No reranking (FAISS top-5 directly)** - Removes ~150ms latency and an API dependency. FAISS-only Precision@5 was 0.52; with Cohere reranking it hit 0.74. The 42% improvement at 150ms cost (vs <10ms for FAISS retrieval itself) was a clear win, and the graceful fallback means a Cohere outage degrades precision without crashing.

**Cohere rerank-multilingual-v3.0** - LKML is English-only, so no benefit over the English model at the same price.

---

## Quantified Validation

From the P2 evaluation grid:

| Configuration | Recall@5 | Precision@5 | Notes |
|---|---|---|---|
| MiniLM + no rerank | 0.61 | 0.44 | Local, free |
| OpenAI + no rerank | 0.87 | 0.52 | API cost only |
| OpenAI + Cohere rerank | 0.87 | 0.74 | Selected |
| OpenAI + larger chunks | 0.76 | 0.61 | Fewer, bigger chunks |

Full corpus embedding (1,511 docs, ~755K chunks) costs ~$1.89 one-time with OpenAI text-embedding-3-small, cached after the first run. Cohere rerank is free up to 1,000 calls/month, $2/1K after that. After the initial index build, all retrieval is free since FAISS runs locally.

---

## Consequences

The Cohere dependency at query time is the main operational risk. A Cohere outage degrades to FAISS-only precision (0.52 vs 0.74) via the `results[:top_n]` fallback in `reranker.py`, but doesn't crash. For the 42% precision lift, that's a risk I'll carry.

Embedding vectors are cached as MD5-keyed JSON files at `data/cache/embeddings_openai.json` and `data/cache/embeddings_minilm.json`. The cache grows unboundedly. For the 755K-chunk full corpus at 1536 floats per vector, that's ~4.6GB. At Day 3 scope (20-doc dev samples) JSON is fine. A key-value store like Redis or SQLite blobs would be the next step if I index the full corpus.

The MiniLM baseline stays as `provider="minilm"` in `embed_chunks()` and `embed_query()` for offline development and A/B experiments down the line.
