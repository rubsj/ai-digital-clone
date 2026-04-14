# ADR-004: Groundedness Scoring — Semantic Similarity Heuristic over LLM Judge

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation
**Status:** Accepted
**Date:** 2026-04-14

---

## Context

The evaluation pipeline needs to score whether a generated response is grounded in the retrieved knowledge chunks. Groundedness is 40% of the final quality score (`final = 0.4×style + 0.4×groundedness + 0.2×confidence`), and it determines whether the system delivers a styled response or routes to fallback.

It's also the hardest of the three signals to compute without an LLM. Style is cosine similarity on numerical features — fast, no API call. Confidence is rule-based heuristics — same. Groundedness requires checking whether factual claims in the response actually appear in the source material, which sounds like a reading comprehension problem.

I had to decide: use an LLM judge per evaluation, or build a heuristic that's cheap enough to run on every query.

---

## Decision

Sentence-level max cosine similarity against the top-5 retrieved chunks, with all embeddings batched into at most two `embed_openai()` calls.

The algorithm:
1. Split the response into sentences via regex look-behind (`re.split(r"(?<=[.!?])\s+", ...)`). Skip fragments under 10 chars.
2. Batch-embed all sentences in one `embed_openai(sentences)` call — reuses the MD5 JSON cache.
3. Reuse `chunk.embedding` from the RAG pipeline. Batch-embed only chunks missing `.embedding` in a second call, if any.
4. For each sentence, compute cosine similarity against each chunk embedding and take the max.
5. Average the per-sentence maxima.

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

**Pure LLM judge per response** — ~$0.002/call × 1,000 evals = $2 for evaluation alone. The cost is manageable, but the latency isn't. The system already calls LiteLLM for style generation and the explanation string; a third LLM call in the hot path would push end-to-end latency over 3s. The PRD doesn't require a judge at inference time — just for the one-time threshold calibration.

**Token overlap (BLEU/ROUGE)** — Fast and free, but semantically blind. A response that says "the operating system allocates pages using a buddy system" would score near zero against a chunk containing "Linux uses zone-based page frame allocation" — same factual claim, no token overlap. We already traded token matching for semantic matching when we chose `text-embedding-3-small` for the RAG layer. Reverting to token overlap for the groundedness check is inconsistent with that decision.

**BERTScore (precision/recall/F1 on token embeddings)** — Better than BLEU/ROUGE because it uses contextual token embeddings, but requires loading a BERT model locally (~400MB dependency) and doesn't naturally handle the one-response-against-five-chunks structure. The sentence-level max-cosine approach is semantically equivalent but uses the same embedding model already cached, at no extra dependency cost.

**Per-sentence LLM judge** — Reduces per-call cost by batching sentences into one prompt, but the latency problem remains and cost becomes proportional to sentence count. A 5-sentence response at $0.002/call = $0.01/query at full evals. No measurable advantage over the heuristic for in-domain CS queries.

---

## Quantified Validation

The 0.60 threshold was validated by a 5-sample LLM judge comparison: three in-domain queries (memory management, scheduler, filesystem) and two out-of-domain (networking, compilers).

| Query | LLM judge | Heuristic | Agreement |
|---|---|---|---|
| Memory management | 0.81 | 0.78 | Yes |
| CFS scheduler | 0.73 | 0.70 | Yes |
| Filesystem inodes | 0.65 | 0.63 | Yes |
| Networking (OOD) | 0.58 | 0.55 | Yes (both borderline) |
| Compilers (OOD) | 0.44 | 0.61 | No |

4 out of 5 agreements. The disagreement was the compiler query, where the heuristic overestimated (0.61 vs 0.44) — the response used CS vocabulary that cosine-matched the chunks but made a subtly wrong claim. This is the known failure mode: semantic similarity can't detect contradiction.

---

## Consequences

The cache reuse is the main performance benefit. By the time a response is being scored, the top-5 chunks have already been embedded by the RAG pipeline and stored on `chunk.embedding`. `score_groundedness` skips chunk embedding entirely in the common case. On a warm cache, the function is a cosine similarity loop over pre-computed vectors: under 5ms. The sentence embedding call is the one remaining API hit — one `embed_openai()` call for 4-6 sentences.

The scoring is interpretable. "Sentence 3 had max cosine similarity 0.42 against the top 5 chunks" points at the specific claim that isn't grounded. An LLM judge returns "0.4, this response seems not fully grounded" with no way to attribute the score to a sentence.

The failure mode is semantic mismatch, not factual mismatch. A response that contradicts a source but uses similar vocabulary can score higher than it should. For LKML-domain queries where terminology is precise — "slab allocator," "rcu_read_lock," "CFS scheduler" — this risk is low. For broader topics, the score may be overconfident. The compiler disagreement in the calibration sample is the clearest example of this risk. The Day 6 weight sensitivity sweep will test whether the groundedness threshold needs domain-specific adjustment.

Porting to a different domain is straightforward at the algorithm level, but the 0.60 threshold is LKML-calibrated. A new domain needs its own 5-sample judge comparison before trusting the boundary. (Same as porting a LKML-tuned style feature extractor to Slack messages: the algorithm transfers, the calibration constants don't.)
