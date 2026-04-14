# ADR-004: Groundedness Scoring — Semantic Similarity Heuristic over LLM Judge

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation
**Status:** Accepted
**Date:** 2026-04-14

---

## Context

The evaluation pipeline (Day 4) needs to score whether a generated response is grounded in the retrieved knowledge chunks. This is 40% of the final quality score (`final = 0.4×style + 0.4×groundedness + 0.2×confidence`), which determines whether the system delivers a styled response or falls back to a calendar-booking message.

Groundedness is the hardest of the three signals to compute without an LLM: style is cosine similarity on numerical features (fast, free), confidence is rule-based heuristics (fast, free), but groundedness requires understanding whether factual claims in the response actually appear in the source chunks — which sounds like a reading comprehension task.

The question is: do I need an LLM to do that, or can a semantic similarity heuristic get close enough?

---

## Decision

Sentence-level max cosine similarity against the top-5 retrieved chunks, with all embeddings batched into one or two `embed_openai()` calls.

The algorithm:
1. Split the response into sentences via regex look-behind (`re.split(r"(?<=[.!?])\s+", ...)`). Skip fragments under 10 chars.
2. Batch-embed all sentences in one `embed_openai(sentences)` call — reuses the MD5 JSON cache.
3. Reuse `chunk.embedding` from the RAG pipeline (already set by `embed_chunks()` in Day 3). Batch-embed only chunks missing `.embedding`, again in one call.
4. For each sentence, compute cosine similarity against each chunk embedding and take the max.
5. Average the per-sentence maxima. That's the groundedness score.

The 0.60 threshold in the PRD quality table was validated by a 5-sample LLM judge comparison: three in-domain queries (memory management, scheduler, filesystem) and two out-of-domain queries (networking, compilers). The LLM judge agreed with the heuristic direction (grounded vs. not grounded) in 4 out of 5 cases. The one disagreement was a borderline case (LLM judge: 0.58, heuristic: 0.55) where both methods would round to "borderline passing."

```python
# src/evaluation/groundedness_scorer.py
def score_groundedness(response, chunks, top_k=5) -> float:
    sentences = _split_sentences(response)
    sentence_vecs = embed_openai(sentences)           # 1 API call
    chunk_vecs = [rr.chunk.embedding for rr in chunks[:top_k]]
    per_sentence_max = [max(_cosine(sv, cv) for cv in chunk_vecs) for sv in sentence_vecs]
    return float(np.mean(per_sentence_max))
```

---

## Alternatives Considered

| Approach | Why Not |
|---|---|
| Pure LLM judge per response | ~$0.002/call × 1,000 evals = $2 for eval alone. More importantly: 800ms latency added to every query. For a system that already calls LiteLLM for style generation and explanation strings, a third LLM call in the hot path would push end-to-end latency over 3s. The PRD doesn't require a judge for calibration — just for the one-time threshold validation. |
| Token overlap (BLEU/ROUGE) | Fast and free, but semantically blind. A response that says "the operating system allocates pages using a buddy system" would score near zero against a chunk containing "Linux uses zone-based page frame allocation" — same factual claim, no token overlap. The whole point of using embeddings in the RAG layer is that we've already traded token matching for semantic matching; reverting to token overlap for the groundedness check is inconsistent. |
| BERTScore (precision/recall/F1 on token embeddings) | Better than BLEU/ROUGE because it uses contextual embeddings, but requires loading a BERT model locally (adds 400MB dependency) and doesn't naturally handle the multi-chunk structure (response vs. one-of-five chunks). The sentence-level max-cosine approach is semantically equivalent but uses the same embedding model we already have cached. |
| Per-sentence LLM judge | Reduces per-call cost by batching, but the latency problem remains and the cost is now proportional to sentence count. A 5-sentence response at $0.002/call = $0.01/query at full evals. No obvious advantage over the heuristic for in-domain CS queries. |

---

## Consequences

The cache reuse is the main performance win. By the time a response is being scored, the top-5 chunks have already been embedded by the RAG pipeline. `score_groundedness` avoids paying for chunk embedding entirely in the common case — the embedding cost is amortized across all queries that hit those chunks, not repeated per evaluation. On a warm cache, the function runs a cosine similarity loop over pre-computed vectors: under 5ms.

The sentence embedding call is the one remaining API hit per evaluation. For a response with 4-6 sentences, this is one `embed_openai()` call returning 4-6 vectors. The cache will warm up quickly across dev runs on similar queries.

The scoring formula is interpretable: "sentence 3 had max cosine similarity 0.42 against the top 5 chunks" explains a low groundedness score in terms a developer can inspect. A black-box LLM judge returns "0.4, this response seems not fully grounded" with no way to pinpoint which sentence failed.

The failure mode is semantic mismatch rather than factual mismatch. A response that contradicts the source but uses similar technical vocabulary could score higher than it should. For LKML-domain queries where vocabulary is highly specific, this risk is lower than in general-domain QA — "slab allocator," "rcu_read_lock," and "CFS scheduler" aren't false friends. For broader topics (e.g., "explain networking"), the score may be overconfident. The Day 6 weight sensitivity sweep is designed to explore whether the groundedness threshold needs domain-specific calibration.

Porting groundedness scoring to a different domain (not kernel CS) is straightforward: the algorithm is domain-agnostic. What requires rethinking is the 0.60 threshold, which was calibrated on LKML-domain queries. A new domain needs its own 5-sample LLM judge calibration pass.

---

## Interview Signal

The key trade-off here is latency and cost vs. precision. A full LLM judge would be more precise on ambiguous cases — it can handle negation, coreference, and multi-hop reasoning that cosine similarity misses. But the 4/5 agreement rate on the calibration sample, combined with the 800ms × 3 LLM-calls-per-query cost, made the heuristic the right call for this scope. If the system were in production with quality SLAs and human feedback loops, I'd upgrade the judge on the groundedness dimension first — it's the one where the heuristic is most likely to fail on adversarial inputs.

The embedding cache architecture also matters here. Because `embed_openai()` uses MD5-keyed JSON cache, the sentence embeddings from the evaluation step can be re-used in future evaluations of the same response (e.g., A/B testing two style profiles on the same generated text). The cache isn't just a latency optimization — it's a consistency guarantee: the same text always produces the same score.

---

## Java/TS Parallel

In Java, this is equivalent to choosing between a regex-plus-cosine library (like Apache Lucene's BM25 scorer, fast but lexical) and a full NLP evaluation framework (like DeepEval's G-Eval, LLM-in-the-loop but expensive). The same trade-off applies: lexical scoring is fast and inspectable, but misses semantic equivalence; LLM scoring is accurate but adds latency and API cost to every evaluation call.

The batch embedding optimization — collect all sentences first, embed in one API call, then compute cosine similarities — mirrors the batching pattern in any Java ETL pipeline: accumulate records into a list, flush to the external service once, map results back to input indices. The `missing_indices` list in `groundedness_scorer.py` is exactly that pattern: identify which chunks need embedding, batch the request, fill the gaps. In a Java service you'd call it a "partial cache miss with batch backfill."

The key insight: embedding calls are the expensive I/O; cosine similarity is free CPU. Restructure the code so all the I/O happens in at most two batches (sentences + missing chunks), then do all the similarity computation locally. This is the same principle as N+1 query elimination in a JPA/Hibernate application — identify the batching opportunity and exploit it.

---

## Cross-References

- **ADR-002** — documents the embedding model choice (`text-embedding-3-small` via LiteLLM) and the chunk structure this scorer depends on. The `chunk.embedding` reuse pattern in `score_groundedness` is only possible because `embed_chunks()` stores normalized vectors on the `KnowledgeChunk` objects.
- **P2 empirical findings** — the RAG evaluation grid showing OpenAI `text-embedding-3-small` Recall@5 = 0.87 vs MiniLM 0.61. The same embedding quality advantage applies here: using `text-embedding-3-small` for sentence embeddings means the semantic space is consistent with the chunk embeddings built by the RAG indexer, so cosine similarity is meaningful across the boundary.
- **P5 production findings** — in P5, the evaluation pipeline used a lightweight heuristic for groundedness (BM25-based) and found it had ~15% false-positive rate on out-of-domain queries. The sentence-level max cosine approach in P6 avoids the lexical mismatch problem P5 encountered, at the cost of the embedding API call. The 4/5 calibration agreement in P6 compares favorably to P5's observed false-positive rate.
