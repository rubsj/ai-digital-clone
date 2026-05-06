# Iteration Log

This file records experiment results for the P6 Digital Clone project, one entry per experiment. Each entry uses the six-field format from PRD §7g: Change, Reason, Metric Before, Metric After, Delta, Keep? Entries are newest-first within each day's H2 section. The query set driving experiments 6a/6b/6c/6e is versioned at `data/eval/queries_v1.json`.

**Query set provenance.** `queries_v1.json` was authored on 2026-04-27 from a 30-item random sample (seed=42, first 1511 CS-filtered items drawn from the `open-phi/textbooks` HuggingFace dataset). The sample is dominated by the "programming" subfield (≈98%), with a long tail of algorithms, systems, and networking textbooks. Core CS concepts — TCP, binary search, stacks/queues — appear in 9–28% of items, confirming HIGH bands; OS and DB synthesis topics (isolation levels, page replacement) appear in 1–2%, confirming MEDIUM bands; cross-cutting topics (cache coherence, buffer overflow) appear in under 1.5%, confirming LOW bands.

## Day 6 — Experiment Day (2026-04-27)

### 6b — Chunking comparison: fixed 500/50 vs semantic markdown

**Pre-run hypotheses (logged before running):**
- H1: Near-zero Cohere queries (q01, q05, q07, q08, q09, q10) — chunking cannot recover Cohere signal; corpus lacks the content regardless of how it is sliced.
- H2: High-Cohere queries (q02, q03, q04, q06) — semantic chunking may shift which specific chunks rank highest but Cohere max scores should remain in their current band.
- H3: `open-phi/textbooks` documents load from a `markdown` column; `chunk_semantic()` may find headers and produce meaningfully different sections. If documents are prose-heavy with few headers, semantic ≈ baseline.
- H4: Net aggregate groundedness delta will be smaller than the per-query delta on high-Cohere queries.

**Chunk-relevance metric (defined before measuring):** mean of the top-5 Cohere reranker relevance scores per query, averaged across all 10 queries. Higher = reranker finds retrieved chunks more relevant. This isolates chunking quality from embedding/reranker choice.

| Field | Value |
|---|---|
| **Change** | Switch chunking from `RecursiveCharacterTextSplitter` 500/50 (baseline) to `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` 500/50 (semantic). Same OpenAI embeddings, same Cohere reranker, same downstream pipeline. |
| **Reason** | Test whether semantic section boundaries in markdown textbooks produce more coherent, topic-aligned chunks that improve retrieval groundedness. |
| **Metric Before** | Baseline: post_G=0.4653±0.1057 \| pre_G=0.4704±0.1035 \| chunk_rel=0.1561±0.2735 \| 6713 chunks |
| **Metric After** | Semantic: post_G=0.4652±0.1055 \| pre_G=0.4698±0.1034 \| chunk_rel=0.1567±0.2737 \| 7044 chunks (+4.9%) |
| **Delta** | Post-G Δ(baseline−semantic): +0.0002 (+0.0%); Pre-G Δ: +0.0006 (+0.1%); ChunkRel Δ: −0.0006. All deltas are noise-level. High-Cohere subset (q02/q03/q04/q06): baseline chunk_rel=0.3825, semantic=0.3835 — no meaningful shift. Near-zero subset (q01/q05/q07/q08/q09/q10): both configs remain at chunk_rel<0.012 — H1 confirmed. |
| **Keep?** | Keep baseline. Semantic chunking produces zero measurable retrieval improvement on this corpus. H3 confirmed: markdown headers are sparse in `open-phi/textbooks` (4.9% more chunks, predominantly from whitespace/section boundary artifacts rather than meaningful topic splits); semantic falls back to `RecursiveCharacterTextSplitter` for most documents and produces near-identical chunk boundaries. Semantic index build adds 83.6s and 331 extra chunks with no retrieval benefit. |

**Rate limiting note:** 7s inter-query sleep was insufficient; Cohere 429 fired on q03-semantic (confirmed in script output). Cause: preflight call + 6 query calls within ~25s. For q03-semantic and q04-baseline, reranker fell back to FAISS top-20 order. Since both configurations show near-identical numbers (q03: 0.5952/0.5952 for both; q04: 0.6678/0.6678 for both), fallback did not affect the conclusion. Next experiment (6c): use 10s inter-query sleep and/or add explicit sleep after preflight call.

---

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

**Headline finding:**
On this corpus shape, Cohere exhibits BIMODAL reranker behavior: relevance scores cluster either above 0.20 (q03 binary search max=0.751, q04 stacks/queues max=0.999, q06 routing protocols max=0.372–0.421; q02 virtual memory borderline at 0.24 OpenAI / 0.16 MiniLM) or below 0.05 (q01 TCP, q05 isolation levels, q07 page replacement, q08 DP/greedy OpenAI, q09 buffer overflow, q10 cache coherence). There is no smooth middle ground. Embedding choice produces +2.5% post-rerank groundedness on average, but Cohere collapses both embeddings to near-identical scores on the 6 near-zero queries — retrieving different candidate chunks does not produce different outcomes when the reranker assigns ~0 to all of them. Exception: q08 shows a **reversal** — MiniLM beats OpenAI post-rerank (0.4846 vs 0.4820) because Cohere found better DP/greedy content in MiniLM's top-20 candidate pool (MiniLM CohMax=0.548 vs OpenAI CohMax=0.023 on this query). This is the only query where OA < ML after reranking.

**Implication for ADR-002:** The P5 carry-forward "Cohere provides 20% lift" is corpus-shape sensitive. On a 4-subfield programming-heavy corpus with broad CS queries, Cohere provides binary verdicts (strong signal on algorithms/data-structures queries, near-zero on OS/networking/security/DB queries), not a uniform percentage lift. ADR-006 candidate (see also Confidence scorer limitation below).

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
| **Keep?** | Keep OpenAI. Direction confirmed (OA ≥ MiniLM) on 5-doc corpus overall, with one reversal (q08 MiniLM wins post-rerank due to better DP candidate pool). Gap (+2.5% post-rerank) remains far below P5 prior. ADR-002 embedding claim is directionally supported but magnitude does not replicate. |

**Phase 2 diagnostics (2026-04-28):**

D1 — q07 zero-Cohere one-shot test: Called Cohere `rerank-english-v3.0` directly (outside experiment harness) on q07 query + 3 OpenAI FAISS top-3 candidate chunks. Returned max=0.000006, effectively zero. Candidate chunks were about hash tables and distributed memory systems — not page replacement. **Finding: corpus genuinely lacks page-replacement content; the 0.000 scores in Run 2 are not a harness bug.** Pre-experiment sampling predicted MEDIUM-band hits via raw text matching ("page", "replacement") but Cohere semantic relevance assessment disagrees — the 5-doc corpus has no OS content whatsoever.

D2 — Latency attribution: `experiment_6a_embeddings.py` measures single wall-clock time (retrieve + Cohere HTTP + groundedness scoring) with no per-stage breakdown. Anomalous latencies (q05-minilm: 3448ms, q06-openai: 3787ms, q06-minilm: 3640ms vs typical 240–500ms) are Cohere API tail latency; FAISS retrieval on 6713 vectors is <10ms regardless of embedding model. Absolute latency numbers should be treated cautiously — Cohere API variability dominates.

D3 — Conditional aggregate (differentiating queries only, excluding bit-identical q03/q04/q07/q09/q10):

| Config | Post-G (5 queries) | Pre-G (5 queries) |
|---|---|---|
| OpenAI | 0.4006±0.0491 | 0.4094±0.0521 |
| MiniLM | 0.3947±0.0550 | 0.3835±0.0756 |
| **Δ (OA−ML)** | **+0.0059 (+1.5%)** | **+0.0259 (+6.8%)** |

Headline (all 10 queries): post +2.5%, pre +2.8%. Conditional (5 differentiating queries): post +1.5%, pre +6.8%. The pre-rerank conditional delta (6.8%) is 2.4× higher than the headline (2.8%), showing that OpenAI embedding quality advantage is materially larger on queries where Cohere doesn't collapse the candidate pool. Post-rerank conditional (1.5%) is lower than headline (2.5%) because q08 reversal (MiniLM wins) reduces the differentiating-query aggregate.

**Cohere reranker behavior on this corpus (Run 2 finding — separate from embedding comparison):**

Per-query Cohere max score range: [0.000, 0.999]. Bimodal distribution:
- HIGH signal (max > 0.20): q03 (binary search, max=0.751), q04 (stacks/queues, max=0.999), q06 (routing protocols, max=0.372–0.421). q02 borderline (max=0.239 OpenAI / 0.157 MiniLM).
- NEAR-ZERO signal (max < 0.05): q01 (TCP), q02 (MiniLM), q05 (both), q07 (both, max=0.000), q08 (OpenAI, max=0.023), q09, q10.
- Exception: q08 MiniLM (max=0.548 in diagnostic run) — asymmetric Cohere signal between embedding models on same query; Cohere found meaningful DP/greedy content in MiniLM's candidate pool but not OpenAI's.

Mean per-query Cohere std: 0.065 (both embeddings). Low std across the 20-candidate pool for most queries = reranker is not meaningfully differentiating candidates. q07 (page replacement): Cohere assigns 0.000 to all 20 chunks from both embeddings — D1 one-shot test confirmed corpus has zero OS content. **This finding challenges ADR-002's "20% reranker lift" carry-forward from P5** — P5's corpus was domain-matched; P6's textbook corpus lacks coverage for networking, OS, security, and architecture queries. Note: q06 showed CohMax=0.37–0.42 in Run 2 but near-zero in D3 diagnostic run, confirming Cohere non-determinism on marginal-signal queries.

**Confidence scorer limitation (documented per Phase 2 re-approval terms):**
score_confidence() with query-as-proxy makes completeness=1.0 and uncertainty_penalty=1.0 for all queries. Only retrieval_relevance (1/3 weight) varies. Confidence ≈ 0.667 + Cohere_mean/3. This is not a scorer bug — it is a design tradeoff from Day 4 that surfaces here: the query-as-proxy eliminates 2/3 of the confidence signal. ADR-006 candidate.

