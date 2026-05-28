# ADR-004: Groundedness Scoring: Semantic Similarity Heuristic over LLM Judge

**Project:** P6: Torvalds Digital Clone
**Category:** Evaluation
**Status:** Accepted
**Date:** 2026-04-14

---

## Context

The evaluation pipeline needs to score whether a generated response is grounded in the retrieved knowledge chunks. Groundedness is one of the three scores the evaluator produces for a response, alongside style and confidence. v1 combined the three into a weighted `final = 0.4*style + 0.4*groundedness + 0.2*confidence` and routed against a 0.75 threshold; v2 drops the weighted formula and the threshold (ADR-010, Architecture Rule 3), and GatekeeperAgent reasons over the three individual scores instead. Either way groundedness has to be measured per response, and it is the signal for whether the response is actually supported by the retrieved chunks.

Style and confidence are cheap: cosine similarity on numerical features and rule-based heuristics, respectively. Groundedness is not. It requires checking whether factual claims in the response actually appear in the source material, which is a reading comprehension problem. I had to decide between an LLM judge per evaluation or a heuristic cheap enough to run on every query.

---

## Decision

Sentence-level max cosine similarity against the top-5 retrieved chunks, with all embeddings batched into at most two `embed_openai()` calls.

Split the response into sentences via regex look-behind (`re.split(r"(?<=[.!?])\s+", ...)`), skip fragments under 10 chars, batch-embed all sentences in one `embed_openai(sentences)` call through the MD5 JSON cache, then reuse `chunk.embedding` from the RAG pipeline (batch-embed only chunks missing `.embedding` in a second call if any). For each sentence, take the max cosine similarity against the chunk embeddings. Average the per-sentence maxima.

```python
# src/evaluation/groundedness_scorer.py
def score_groundedness(response, chunks, top_k=5) -> float:
    sentences = _split_sentences(response)
    sentence_vecs = embed_openai(sentences)           # 1 API call
    chunk_vecs = [rr.chunk.embedding for rr in chunks[:top_k]]
    per_sentence_max = [max(_cosine(sv, cv) for cv in chunk_vecs) for sv in sentence_vecs]
    return float(np.mean(per_sentence_max))
```

In v2 this is a method on the ScoringEngine Component (`src/components/scoring_engine.py`), which wraps the existing `src/evaluation/` helper rather than reimplementing the math (PRD §12.2). The computation stays deterministic and LLM-free, since v2 uses no LLM for any numerical score (ADR-007). Any LLM interpretation of groundedness happens later, in EvaluatorAgent's explanation generation (ADR-011), not inside the scorer.

---

## Alternatives Considered

**Pure LLM judge per response.** ~$0.002/call, so 1,000 evals costs $2. The cost is manageable but the latency isn't. The system already calls LiteLLM for style generation and the explanation string; a third LLM call in the hot path pushes end-to-end latency over 3s. The PRD doesn't require a judge at inference time, just for one-time calibration.

**Token overlap (BLEU/ROUGE).** Fast and free, but semantically blind. A response saying "the operating system allocates pages using a buddy system" scores near zero against a chunk containing "Linux uses zone-based page frame allocation": same factual claim, no token overlap. I already chose `text-embedding-3-small` for the RAG layer specifically to get semantic matching. Using token overlap for groundedness would undo that choice.

**BERTScore.** Better than BLEU/ROUGE because it uses contextual token embeddings, but requires loading a BERT model locally (~400MB dependency) and doesn't naturally handle one-response-against-five-chunks structure. The sentence-level max-cosine approach is semantically similar but reuses the embedding model already cached, with no extra dependency.

**Per-sentence LLM judge.** Reduces per-call cost by batching sentences into one prompt, but the latency problem remains and cost scales with sentence count. A 5-sentence response at $0.002/call is $0.01/query at full evals.

---

## Quantified Validation

The heuristic was calibrated against a 5-sample LLM judge comparison at the 0.60 agreement level: three in-domain queries (memory management, scheduler, filesystem) and two out-of-domain (networking, compilers).

| Query | LLM judge | Heuristic | Agreement |
|---|---|---|---|
| Memory management | 0.81 | 0.78 | Yes |
| CFS scheduler | 0.73 | 0.70 | Yes |
| Filesystem inodes | 0.65 | 0.63 | Yes |
| Networking (OOD) | 0.58 | 0.55 | Yes (both borderline) |
| Compilers (OOD) | 0.44 | 0.61 | No |

4 out of 5 agreements. The compiler query is the interesting one: the heuristic overestimated (0.61 vs 0.44) because the response used CS vocabulary that cosine-matched the chunks but made a subtly wrong claim. Semantic similarity can't detect contradiction. That's the known failure mode.

---

## Consequences

By the time a response is being scored, the top-5 chunks already have embeddings from the RAG pipeline stored on `chunk.embedding`. `score_groundedness` skips chunk embedding entirely in the common case. On a warm cache the function is a cosine similarity loop over pre-computed vectors: under 5ms. The one remaining API hit is `embed_openai()` for the 4-6 response sentences.

The per-sentence structure makes scores attributable. "Sentence 3 had max cosine similarity 0.42 against the top 5 chunks" points at the specific ungrounded claim. An LLM judge returns a single number with no way to trace it back to a sentence.

The failure mode is semantic similarity, not factual verification. A response that contradicts a source but uses similar vocabulary can score higher than it should. For LKML-domain queries where terminology is precise ("slab allocator," "rcu_read_lock," "CFS scheduler") this risk is low. The compiler disagreement in the calibration sample is the clearest example. The Day 6 weight sensitivity sweep tested whether the groundedness threshold needed domain-specific adjustment and produced null results as a proxy-regime artifact (ADR-006, PRD §10.3). In v2 there is no fixed threshold to tune; GatekeeperAgent reasons over groundedness qualitatively (ADR-010).

The 0.60 calibration is LKML-specific. A new domain would need its own judge comparison to validate the heuristic on that vocabulary.
