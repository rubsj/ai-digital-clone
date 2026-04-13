# ADR-002: RAG Configuration — Embeddings, Reranking, and Chunking

**Project:** P6: Torvalds Digital Clone
**Category:** RAG Pipeline
**Status:** Accepted
**Date:** 2026-04-13

---

## Context

The RAG pipeline (Day 3) needs three configuration decisions before implementation:

1. **Embedding model:** What produces the chunk and query vectors stored in FAISS? The choice determines retrieval quality, latency, index size, and API cost.
2. **Chunking strategy:** How do we split textbook documents into KnowledgeChunks? Size and overlap affect retrieval granularity and context completeness.
3. **Reranking:** Should there be a second-stage reranker after FAISS retrieval? It adds latency and an API dependency but significantly improves precision.

The `open-phi/textbooks` dataset provides 1,511 computer science documents averaging ~129K characters each. A 500-char chunk size produces thousands of chunks from a 20-doc sample — enough to validate retrieval quality without loading the full corpus during development.

---

## Decision

**Embedding:** OpenAI `text-embedding-3-small` (1536 dimensions) via LiteLLM, with `all-MiniLM-L6-v2` (384d) as a local baseline for offline development. All vectors L2-normalized before FAISS `IndexFlatIP` storage — dot product equals cosine similarity for unit vectors.

**Chunking:** `chunk_size=500`, `chunk_overlap=50` as the baseline (RecursiveCharacterTextSplitter). A semantic experiment using MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter runs alongside for comparison. Both share the same config values.

**Reranking:** Cohere `rerank-english-v3.0`, top-20 FAISS candidates → top-5 final results. Graceful fallback: if the Cohere API fails, return original top-5 with a warning log.

**Cache:** MD5-keyed JSON files at `data/cache/embeddings_openai.json` and `data/cache/embeddings_minilm.json`. Normalized vectors cached before return — re-normalization on load is not needed.

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

**MiniLM as primary embedding model:** `all-MiniLM-L6-v2` runs locally with no API cost and produces 384-d vectors. In the P2 evaluation grid (10 queries, 20 docs, Recall@5 metric), MiniLM achieved 0.61 vs OpenAI's 0.87 — a 26% gap. The gap is largest on technical kernel terminology ("memory-mapped I/O", "scheduler preemption") where OpenAI's training corpus gives better coverage. MiniLM remains useful for offline/CI development where OpenAI is unavailable.

**Larger chunks (1000/100):** 1000-char chunks produce fewer chunks with more context per chunk. In P2 grid search, larger chunks improved groundedness scores (less fragmentation) but reduced Recall@5 by ~12% because oversized chunks score lower against short technical queries. The 500/50 baseline was the better trade-off for the query distribution in LKML topics.

**No reranking (FAISS top-5 directly):** Skipping Cohere adds no latency and removes an API dependency. In P2 evaluation, FAISS-only top-5 Precision@5 was 0.52; with Cohere reranking it was 0.74 — a 42% improvement. The reranker adds ~150ms on average (Cohere's median latency for 20 documents). Given that the FAISS retrieval step is <10ms, the reranking latency is the dominant cost, but the precision gain justifies it for a production system. The graceful fallback means one outage doesn't break the pipeline.

**Cohere rerank-multilingual-v3.0:** Marginally better for non-English content. LKML is English-only, so the English model is appropriate and costs the same.

---

## Quantified Validation

From P2 evaluation grid (10 representative LKML-style queries, 20-doc corpus sample):

| Configuration | Recall@5 | Precision@5 | Notes |
|---|---|---|---|
| MiniLM + no rerank | 0.61 | 0.44 | Local, free |
| OpenAI + no rerank | 0.87 | 0.52 | API cost only |
| OpenAI + Cohere rerank | 0.87 | 0.74 | **Selected** |
| OpenAI + larger chunks | 0.76 | 0.61 | Fewer, bigger chunks |

API cost estimate for full corpus (1,511 docs, ~500 chunks/doc ≈ 755,500 chunks):
- OpenAI text-embedding-3-small: $0.020/1M tokens, avg 125 tokens/chunk → ~$1.89 total, cached after first run
- Cohere rerank: free tier 1,000 calls/month, paid at $2/1K after that

The embedding cache eliminates repeat costs. A full index build from scratch costs ~$2 once; thereafter all retrieval is free (FAISS is local).

---

## Consequences

The two-stage pipeline (FAISS → Cohere) adds a hard dependency on the Cohere API at query time. The graceful fallback in `reranker.py` (`try/except Exception → return results[:top_n]`) means a Cohere outage degrades to FAISS-only precision (0.52 vs 0.74) without crashing the system. This is an acceptable trade-off given the 42% precision improvement during normal operation.

The JSON embedding cache grows unboundedly. For the 755K-chunk full corpus at 1536 floats per chunk, the cache file would be ~4.6GB. For production, the cache should be replaced with a key-value store (Redis or a SQLite blob store). For the Day 3 scope (20-doc dev samples), JSON is sufficient.

The `all-MiniLM-L6-v2` baseline remains in the codebase as `provider="minilm"` in `embed_chunks()` and `embed_query()`. This supports offline development, CI test runs without OpenAI credentials, and future A/B experiments.
