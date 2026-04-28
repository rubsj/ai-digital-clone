# Iteration Log

This file records experiment results for the P6 Digital Clone project, one entry per experiment. Each entry uses the six-field format from PRD §7g: Change, Reason, Metric Before, Metric After, Delta, Keep? Entries are newest-first within each day's H2 section. The query set driving experiments 6a/6b/6c/6e is versioned at `data/eval/queries_v1.json`.

**Query set provenance.** `queries_v1.json` was authored on 2026-04-27 from a 30-item random sample (seed=42, first 1511 CS-filtered items drawn from the `open-phi/textbooks` HuggingFace dataset). The sample is dominated by the "programming" subfield (≈98%), with a long tail of algorithms, systems, and networking textbooks. Core CS concepts — TCP, binary search, stacks/queues — appear in 9–28% of items, confirming HIGH bands; OS and DB synthesis topics (isolation levels, page replacement) appear in 1–2%, confirming MEDIUM bands; cross-cutting topics (cache coherence, buffer overflow) appear in under 1.5%, confirming LOW bands.

## Day 6 — Experiment Day (2026-04-27)

### 6a — Embedding comparison: OpenAI vs MiniLM

**Pre-run hypotheses (logged before running):**
- H1: q01 (TCP, 9% corpus coverage) and q03 (binary search, 28%) may top out near 1.0 groundedness for both configs — if both land ≥ 0.95 they are non-differentiating and are reported separately rather than included in the unweighted mean.
- H2: The 98% programming-subfield corpus concentration likely compresses the OpenAI-vs-MiniLM gap below P5 RAG-eval's 26% Recall@5 delta. Both embeddings can retrieve "some programming book" for almost any CS query. Expected Δmean_groundedness: 10–18%. Direction should hold (OpenAI > MiniLM); magnitude may shrink.

| Field | Value |
|---|---|
| **Change** | Swap `text-embedding-3-small` (OpenAI, 1536d) for `all-MiniLM-L6-v2` (384d) as the index and query embedding model. All other pipeline components held constant: chunking 500/50, Cohere rerank-english-v3.0 top-20→top-5, scoring weights 0.4/0.4/0.2. |
| **Reason** | Verify whether P5 RAG-eval's +26% Recall@5 lift for OpenAI over MiniLM replicates on P6's textbook corpus. Ground ADR-002's embedding-model claim in live P6 data. |
| **Metric Before** | OpenAI — mean groundedness: 0.4199, mean final: 0.6090, mean retrieval latency: 22539ms |
| **Metric After** | MiniLM — mean groundedness: 0.4121, mean final: 0.6057, mean retrieval latency: 584ms |
| **Delta** | Δmean_groundedness: +0.0077 (+1.9%); Δmean_final: +0.0033; Δlatency: −21955ms (MiniLM 38× faster on retrieve+rerank). H1: no queries topped out at ≥0.95 (max groundedness = 0.67 for q04); H2: actual +1.9% gap vs predicted 10–18% — corpus concentration effect is even more severe than hypothesised. Note: OpenAI retrieval latency (avg 22.5s) dominated by cold embed_query API calls per query; MiniLM latency (avg 0.6s) is local inference + Cohere rerank only. Corpus capped at max_docs=1 (~1476 chunks); 20-doc full corpus produced 30K chunks and a 921MB JSON embedding cache that caused a process crash at cleanup. |
| **Keep?** | Keep OpenAI. Direction holds (OpenAI > MiniLM groundedness), ADR-002's embedding claim confirmed on P6 data. However the gap (+1.9%) is far smaller than P5's +26% Recall@5 — the 98% programming-subfield corpus makes both models roughly equivalent at retrieving "some programming book." MiniLM's 38× latency advantage is compelling for dev loops; OpenAI's marginal groundedness edge justifies keeping it for prod evaluation runs. |

### 6a — Run 2 (corrected corpus + dual-rank measurement)

**Run 1 rejection findings corrected in this run:**
- F1/F2: Corpus expanded to 5 docs (6713 chunks, 4 subfields: programming_languages, human-computer_interfaces, data_mining×2, algorithms_and_data_structures). Candidate pool overlap dropped from 60% (q01, 1 doc) to diverse retrieval behavior across query topics.
- F3 (confidence scorer): Documented as limitation, not fixed (see below).
- F4 (npz cache): Switched embedding cache from JSON to numpy npz. 880MB JSON → 29MB npz for OpenAI (30× reduction). No crash at 6713 chunks.

**New metrics in Run 2:**
- `pre_rerank_groundedness`: top-5 by raw FAISS score, no Cohere. Isolates embedding quality.
- `post_rerank_groundedness`: top-5 by Cohere rerank. Production metric.
- `cohere_dist`: Cohere score mean/std/max across all top-20 candidates per query.

**Rate limiting note:** Cohere trial key (10 calls/min). Script makes 21 calls (1 pre-check + 20 query×embedding). Added 7s inter-query sleep to prevent 429 fallback. All 10 queries used Cohere successfully in this run.

| Field | Value |
|---|---|
| **Change** | Re-run with 5-doc corpus (6713 chunks), npz embedding cache, dual-rank groundedness metric (pre- and post-Cohere top-5), per-query Cohere score distribution logging. |
| **Reason** | Run 1 rejected: 1-doc corpus (1476 chunks) produced 60% candidate-pool overlap for q01 → reranker collapsed embedding differences. Bit-identical post-rerank groundedness for 7/10 queries in Run 1 was an experimental artifact, not a corpus finding. |
| **Metric Before (Run 1, invalid)** | OpenAI post-rerank groundedness: 0.4199 (1 doc, 1476 chunks) |
| **Metric After (Run 2)** | OpenAI post-rerank: 0.4653±0.1057 \| MiniLM post-rerank: 0.4539±0.1200 \| OpenAI pre-rerank: 0.4704±0.1035 \| MiniLM pre-rerank: 0.4574±0.1182 |
| **Delta** | Post-rerank Δ(OA−ML): +0.0114 (+2.5%); Pre-rerank Δ(OA−ML): +0.0130 (+2.8%). H1: no queries hit ≥0.95 ceiling. H2: actual delta +2.5-2.8% vs predicted 10-18% and P5 prior +26% — corpus concentration effect stronger than hypothesised even with 4 diverse subfields. |
| **Keep?** | Keep OpenAI. Direction confirmed (OA > MiniLM) on 5-doc corpus. Gap (+2.5% post-rerank) remains far below P5 prior. ADR-002 embedding claim is directionally supported but magnitude does not replicate. |

**Cohere reranker behavior on this corpus (Run 2 finding — separate from embedding comparison):**

Per-query Cohere max score range: [0.000, 0.999]. Bimodal distribution:
- HIGH signal (max > 0.20): q03 (binary search, max=0.751), q04 (stacks/queues, max=0.999), q05 OA (max=0.032 — marginal), q06 (routing protocols, max=0.372–0.421)
- NEAR-ZERO signal (max < 0.05): q01 (TCP), q02 (virtual memory), q05 ML (max=0.001), q07 (page replacement, max=0.000), q08 (DP vs greedy), q09 (buffer overflow), q10 (cache coherence)

Mean per-query Cohere std: 0.065 (both embeddings). Low std across the 20-candidate pool for most queries = reranker is not meaningfully differentiating candidates. q07 (page replacement): Cohere assigns 0.000 to all 20 chunks from both embeddings — the 5-doc corpus has zero OS content; reranker adds no signal for this query. **This finding challenges ADR-002's "20% reranker lift" carry-forward from P5** — P5's P5 corpus was domain-matched; P6's textbook corpus lacks coverage for networking, OS, security, and architecture queries.

**Confidence scorer limitation (documented per Phase 2 re-approval terms):**
score_confidence() with query-as-proxy makes completeness=1.0 and uncertainty_penalty=1.0 for all queries. Only retrieval_relevance (1/3 weight) varies. Confidence ≈ 0.667 + Cohere_mean/3. This is not a scorer bug — it is a design tradeoff from Day 4 that surfaces here: the query-as-proxy eliminates 2/3 of the confidence signal. ADR-006 candidate.

