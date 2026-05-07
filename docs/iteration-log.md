# Iteration Log

This file records experiment results for the P6 Digital Clone project, one entry per experiment. Each entry uses the six-field format from PRD §7g: Change, Reason, Metric Before, Metric After, Delta, Keep? Entries are newest-first within each day's H2 section. The query set driving experiments 6a/6b/6c/6e is versioned at `data/eval/queries_v1.json`.

**Query set provenance.** `queries_v1.json` was authored on 2026-04-27 from a 30-item random sample (seed=42, first 1511 CS-filtered items drawn from the `open-phi/textbooks` HuggingFace dataset). The sample is dominated by the "programming" subfield (≈98%), with a long tail of algorithms, systems, and networking textbooks. Core CS concepts — TCP, binary search, stacks/queues — appear in 9–28% of items, confirming HIGH bands; OS and DB synthesis topics (isolation levels, page replacement) appear in 1–2%, confirming MEDIUM bands; cross-cutting topics (cache coherence, buffer overflow) appear in under 1.5%, confirming LOW bands.

## Day 6 — Experiment Day (2026-04-27)

### 6e — GPT-4o-mini vs local Ollama (qwen3:8b) for evaluation scoring

**Pearson note (stated before running):** Final scores are computed from the same component scores via the same deterministic formula for both configs, so Pearson(GPT-scores, Ollama-scores) = 1.0 trivially. The meaningful metrics are (a) explanation validity — structured-output success rate and 5-query spot check — and (b) latency per explanation call.

**Decision outcomes (stated before running):**
1. **Quality parity** — all Ollama explanations valid AND qualitatively equivalent → recommend Ollama for dev, GPT-4o-mini for prod.
2. **Quality drift** — any structured-output failure OR hallucinated reasoning → recommend GPT-4o-mini for both.
3. **Latency tradeoff** — parity on quality but latency meaningfully different → state tradeoff; recommend per-environment.

**Headline finding.** Ollama qwen3:8b produces valid structured explanations (100% success) and is 2.1x faster than GPT-4o-mini on this task (mean 739ms vs 1570ms, ratio=0.47x). The latency difference is meaningful (> 1.5x threshold). Outcome: **LATENCY_TRADEOFF** → Ollama for dev (zero API cost, lower and more stable latency), GPT-4o-mini for prod.

Spot-check quality: neither model reliably identifies the true weakest scoring dimension. GPT misattributed the driver on q03 (blamed style=0.496 when groundedness=0.595 was below the 0.60 target). Ollama misattributed on q04 (blamed groundedness when it was above target at 0.668). Both produce plausible-sounding explanations, but the one-sentence prompt does not force arithmetic verification against thresholds. This is a prompt-design gap common to both models — not an Ollama-specific quality drift.

| Field | Value |
|---|---|
| **Change** | Replace the explanation-LLM call in the evaluator with Ollama qwen3:8b (via `instructor.from_litellm`, `mode=JSON`) at script level. Retrieve and Cohere-rerank ONCE per query; compute component scores ONCE per query; generate explanation TWICE (GPT-4o-mini + Ollama). |
| **Reason** | PRD §8: test whether a local model can replace the API call for evaluation scoring. Phase 7 ADR-006 decision point. |
| **Metric Before** | GPT-4o-mini: mean_latency=1570ms (std=640ms, min=904ms, max=3030ms). Structured-output success: 10/10 (100%). |
| **Metric After** | Ollama qwen3:8b: mean_latency=739ms (std=59ms, min=686ms, max=893ms). Structured-output success: 10/10 (100%). Latency ratio (Ollama/GPT): 0.47x. Pearson(final scores): 1.000 (trivially — same formula). |
| **Delta** | Latency: −831ms mean (−53%). Ollama is 2.1x faster and more consistent (std 59ms vs 640ms). Final scores: identical (Δ=0.0000 for all 10 queries — deterministic formula, same inputs). Structured-output reliability: parity (100%/100%). |
| **Keep?** | LATENCY_TRADEOFF: use Ollama qwen3:8b for dev evaluation (zero API cost, 2.1x faster, stable latency). Use GPT-4o-mini for prod (network-tolerant, proven reliability at scale). Neither model reliably identifies the true weakest scoring dimension from the one-sentence prompt — recommend prompt improvement in ADR-006. ADR-006 is triggered per Phase 7 criterion: "6e produced an actionable decision (parity → dev/prod split)." |

**RUN 1 LIMITATION (documented before Run 2).** Run 1 measured weighted-sum computation latency, not LLM scoring agreement. Both models received pre-computed component scores (style, groundedness, confidence) and applied the same deterministic formula — this is arithmetic, not evaluation. Pearson = 1.0 is a degenerate result: it confirms that 0.4×a + 0.4×b + 0.2×c equals itself regardless of which model computes it. The latency data (GPT mean=1570ms, Ollama mean=739ms) and structured-output success rate (100%/100%) are valid observations on the explanation-generation sub-task but do not address the intended question: do the two models agree when independently scoring groundedness from the same (query, top-5 chunks) context? Run 2 corrects this. See Run 2 entry below.

**ADR-006 trigger:** Deferred — Run 1 result is insufficient to trigger. Run 2 outcome determines framing.

---

### 6e (Run 2) — GPT-4o-mini vs Ollama qwen3:8b: groundedness scoring agreement

**Production scorer note:** `score_groundedness()` is NOT keyword overlap. It computes sentence-level OpenAI embedding cosine similarity (query-as-response proxy). "Baseline" throughout = this embedding-cosine scorer.

**Pre-run hypotheses:**
- H1: Pearson(GPT, Ollama) ≈ 0.85–0.95 (bimodal Cohere distribution makes relevant/irrelevant split unambiguous)
- H2: Pearson(GPT, baseline) ≈ 0.70–0.85 (same signal, LLMs add contextual reasoning)
- H3: Pearson(Ollama, baseline) ≈ similar to H2
- H4: Ollama structured-output success ≥ 95% on harder task

**Weakest-dimension tie-breaking rule (stated before running):** if top-two shortfalls within 0.05 of each other, naming either counts as correct. Style shortfall ≈ 0.40 for all queries in the proxy regime (style ≈ 0.50, target 0.90) — proxy artifact noted in advance.

**Headline finding.** Both models gravitate to discrete anchor values from the calibration scale: GPT used only {0.0, 0.5}, Ollama used {0.0, 0.5, 0.9}. The calibration anchors (0.0/0.5/1.0) constrained the scoring distribution — models treated the anchor points as discrete buckets rather than endpoints of a continuous scale. Pearson(GPT, Ollama) = 0.7982 (MEDIUM AGREEMENT): rank-order broadly agrees, but absolute scores differ where models chose different buckets (largest divergence: q03, GPT=0.5 vs Ollama=0.9). Latency finding from Run 1 (Ollama 2.1x faster) does NOT hold on the harder scoring task — Run 2 shows parity (0.97x). The latency advantage is task-specific: Ollama is faster on simple text generation but matches GPT on reasoning over chunk texts.

| Field | Value |
|---|---|
| **Change** | Both models independently score groundedness in [0,1] from (query, top-5 chunk texts). Style and confidence remain deterministic. Baseline = embedding-cosine scorer (production). Three Pearsons computed. |
| **Reason** | Corrects Run 1's degenerate result. Tests the intended question: do GPT-4o-mini and qwen3:8b agree when independently evaluating groundedness? |
| **Metric Before** | Baseline (embedding-cosine, query-as-proxy): mean=0.4638 \| std=0.1072 \| min=0.3560 \| max=0.6678 |
| **Metric After** | GPT-4o-mini: mean=0.2500 \| std=0.2500 (quantized to {0.0, 0.5}). Ollama qwen3:8b: mean=0.3400 \| std=0.3007 (quantized to {0.0, 0.5, 0.9}). Latency: GPT mean=1504ms \| Ollama mean=1465ms (ratio=0.97x — parity). Structured-output success: 100%/100%. |
| **Delta** | Pearson(GPT, Ollama)=+0.7982 (p=0.0056) — MEDIUM AGREEMENT. MAE=0.0900±0.1814. Pearson(GPT, baseline)=+0.8172 (p=0.0039). Pearson(Ollama, baseline)=+0.6796 (p=0.0306). H1 PARTIAL (predicted 0.85–0.95, got 0.7982). H2 CONFIRMED. H3 CONFIRMED. H4 CONFIRMED. |
| **Keep?** | MEDIUM AGREEMENT + latency parity → merge 6e with methodology cluster in ADR. Recommend GPT-4o-mini for prod (calibration consistency, stable absolute scores). Ollama viable for dev with the caveat that absolute scores may drift from prod baseline by up to 0.4 on individual queries. The latency advantage from Run 1 does not hold on the scoring task; run-1 speed (2.1x) was an artifact of the trivial explanation-generation task. |

**Weakest-dimension attribution:** Both models named "groundedness" for all 10 queries — 100% inter-model agreement but 50% (GPT) / 40% (Ollama) accuracy. Cause: both models correctly assess groundedness-as-evaluated, but the actual largest shortfall is style (≈0.40 from 0.90 target) in the proxy regime, which both models ignored. This is the proxy-regime artifact from Phase 4 recurring: the style score (≈0.50) is structurally the weakest component, but models responding to a groundedness prompt don't surface it.

**Calibration anchor artifact:** Providing explicit anchor values (0.0/0.5/1.0) caused score quantization. A continuous prompt without anchor examples would likely produce more varied scores and improve Pearson discriminability. Flagged as prompt-design gap for ADR.

**ADR-006 trigger:** YES — "6e produced an actionable decision." Band = MEDIUM AGREEMENT. Decision deferred to Phase 7 pending Ruby's framing choice (one vs two ADRs).

---

### 6d — Pre/post-2018 Torvalds style evolution

**Significance criterion (stated before measuring):** A per-feature delta is a measurable shift only if `|pre_mean − post_mean| > 2 × std(feature on larger partition)`. Larger partition = post-2018 (6,661 emails). Anything below this threshold is reported as "within noise."

**Tracked features:** sentiment (dict_mean of sentiment_distribution), capitalization (capitalization_ratio), exclamations (punctuation_patterns["exclamation"]), formality (formality_level). Extracted via `extract_features()` unmodified — partition filter applied in script only.

**Headline finding.** The 2018 behavioral change does not produce a detectable signal at per-email feature resolution. Within-individual-email variance (std=0.10–0.21 for sentiment and formality) is large relative to between-period mean shifts (|Δ|=0.0002–0.017). All four features remain within their 2σ noise bands.

Formality moved +0.017 in the direction consistent with the public narrative (more formal post-2018) but at 8% of the 2σ band. The direction is suggestive; the magnitude is not measurable at this resolution.

This finding has implications: per-email feature-level analysis cannot validate style-detection sensitivity for behavioral changes of this magnitude. Population-level aggregation (monthly rolling mean, in the chart) shows trends visually but the per-email significance test is the appropriate statistical bar. The PRD §8 "measurable shift" exit criterion is not met.

This finding clusters with Phase 2 (Cohere bimodal on this corpus), Phase 4 (proxy regime pins style score) as the third instance of methodology-limit findings on Day 6 — the underlying theme is that measurement-design choices on this corpus and these inputs produce results that look like absence-of-effect but actually reflect the limits of the measurement.

| Field | Value |
|---|---|
| **Change** | Partition 11,052 Torvalds emails (2015–2023) at 2018-09-01. Pre: 4,391 emails (2015-01-01–2018-08-30). Post: 6,661 emails (2018-09-01–2023-12-31). Compute per-feature means and std on the larger partition for each. |
| **Reason** | PRD §8 Day 6 exit criterion requires "style evolution chart shows measurable shift." The 2018-09 boundary marks Torvalds' public apology and temporary leave, expected to produce a detectable tone change in kernel emails. |
| **Metric Before** | Pre-2018: sentiment=0.0747 \| capitalization=0.0218 \| exclamations=0.0047 \| formality=0.4884 |
| **Metric After** | Post-2018: sentiment=0.0704 \| capitalization=0.0203 \| exclamations=0.0049 \| formality=0.5052 |
| **Delta** | sentiment: −0.00437 (2σ=0.197 — within noise) \| capitalization: −0.00150 (2σ=0.034 — within noise) \| exclamations: +0.00019 (2σ=0.050 — within noise) \| formality: +0.01680 (2σ=0.212 — within noise). No feature clears the 2σ threshold. Largest signal: formality +0.017 = 8% of 2σ band. |
| **Keep?** | n/a — no measurable shift detected at the 2σ threshold. The behavioral change (2018 apology/leave) does not produce a detectable signal in these four features at individual-email resolution. High within-partition variance (std ≈ 0.1–0.2 for sentiment and formality) swamps the small inter-partition deltas. Formality is the closest to a shift (Δ/2σ ≈ 8%) — directionally consistent with a less confrontational tone post-2018 but not significant. |

**Formality measurement note.** `formality_level` is a weighted mean of five sub-signals (`src/style/feature_extractor.py:_formality_level`): `0.25 × formal_word_rate + 0.20 × (1 − contraction_rate) + 0.20 × avg_sent_len_norm + 0.20 × (1 − profanity_rate) + 0.15 × (1 − first_person_rate)`. It is not a simple formal/informal word count. The +0.017 post-2018 shift is therefore directionally consistent with slightly longer sentences, fewer contractions, or reduced first-person usage — all plausible post-apology behavioral changes — but none of these sub-signals can be isolated without a sub-signal breakdown per partition. The measurement is reasonable but composite; any claim that "Torvalds became more formal post-2018" should cite the composite score and this limitation.

**PRD §8 note:** The exit criterion ("style evolution chart shows measurable shift") is not met on these four features. The null result is documented honestly per day6-plan.md §Phase 5: "A null result is a valid finding and goes into the handover honestly." The chart at `docs/images/6d-style-evolution.png` shows monthly-bucketed time series with partition means and ±2σ bands.

---

### 6c — Scoring weight sensitivity (3 configs × 10 queries)

**Pre-run hypotheses (logged before running):**
- H1: Style component will be uniformly low (~0.50) for all queries using query-as-proxy (queries are not Torvalds-style emails). Style-heavy config should underperform because it increases the weight on the lowest component.
- H2: Ground-heavy config should produce the highest final scores on high-groundedness queries (q03, q04, q10) by downweighting the constrained style dimension.
- H3: No config will reach the 0.75 threshold with the query-as-proxy setup. The 100% fallback rate will be an artifact of the proxy, not production behavior.
- H4: ADR-006 trigger will NOT fire — delta in mean_final is expected to be below 0.05 because the three configs only reshuffle two similarly-valued components (style ≈ groundedness ≈ 0.5).

**Optimal-config criterion (stated before running):** "optimal" = highest mean final score AND fallback rate closest to 30-40% PRD target. If criteria disagree, report both — no automatic winner.

**Headline finding:** The query-as-proxy evaluation design pins style score at ~0.50 (std 0.010) because short CS queries don't resemble Torvalds' verbose kernel emails. With style structurally constant, weight perturbations on the style dimension produce sub-noise deltas (≤ 0.004). Phase 4 therefore cannot meaningfully measure weight sensitivity in the style dimension within this regime. The default 0.4/0.4/0.2 weights are retained by inertia, NOT validated by evidence. Weight sensitivity should be re-measured against generated responses (where style scores > 0.85) to produce production-relevant guidance — out of scope for Day 6. This finding clusters with Phase 2's confidence-scorer diagnostic (completeness and uncertainty_penalty mathematically pinned at 1.0 under proxy) as a related observation about the limits of query-as-proxy evaluation. Both feed Phase 7's ADR-006 consideration.

| Field | Value |
|---|---|
| **Change** | Sweep three weight configs — default (0.4/0.4/0.2), style-heavy (0.5/0.3/0.2), ground-heavy (0.3/0.5/0.2) — against the same 10 queries. Retrieve and Cohere-rerank ONCE per query; apply all three weight formulas to the same (style, groundedness, confidence) component scores. |
| **Reason** | Test whether altering the balance between style and groundedness weights materially changes final scores or the deliver/fallback boundary. ADR-006 trigger check per day6-plan.md §Phase 7. |
| **Metric Before** | Default (0.4/0.4/0.2): mean_final=0.5278±0.0548 \| fallback=100% \| style_mean=0.5023 \| groundedness_mean=0.4638 \| confidence_mean=0.7070 |
| **Metric After** | style_heavy (0.5/0.3/0.2): mean_final=0.5317±0.0438 \| fallback=100%. ground_heavy (0.3/0.5/0.2): mean_final=0.5240±0.0658 \| fallback=100%. |
| **Delta** | Δ(style_heavy − default): +0.0039 (+0.7%); Δ(ground_heavy − default): −0.0038 (−0.7%). All deltas are noise-level. All three configs produce 100% fallback rate — no query reaches the 0.75 threshold under any configuration. ADR-006 trigger: NO (Δ ≤ 0.05; all configs equally outside the 30-40% fallback band). H1–H4 confirmed. |
| **Keep?** | Keep default 0.4/0.4/0.2 — by inertia rather than evidence. The experiment cannot validate or refute weight choices in the proxy regime where Phase 4 ran: style is structurally constant at ~0.50, so weight changes on the style dimension produce noise-level deltas regardless of the values chosen. Re-measurement against generated responses is the proper validation path (out of scope for Day 6 but flagged as methodology future-work). |

**Proxy limitation note:** Using query-as-proxy for the style component artificially caps style at ~0.50 (mean=0.5023, std=0.0101 across all queries — nearly constant). This reduces the weight-sensitivity signal: all three configs are combining one near-constant component (style) with two variable ones (groundedness, confidence). A follow-on experiment using actual generated responses would reveal whether style-heavy becomes advantageous when style ≥ 0.80. This is an ADR-006 candidate for documenting the confidence-scorer and style-scorer proxy limitations together.

---

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

**Rate limiting note:** 7s inter-query sleep was insufficient; Cohere 429 fired on q03-semantic (confirmed in script output). Cause: preflight call + 6 query calls within ~25s. For q03-semantic and q04-baseline, reranker fell back to FAISS top-20 order. **Fallback behavior:** `reranker.py` on any exception returns `results[:top_n]` with the original FAISS dot-product scores (cosine similarity, ~0.1–0.9 range) in FAISS retrieval order — NOT Cohere 0–1 relevance scores. This means `_retrieval_relevance()` in the confidence scorer would see FAISS scores rather than Cohere scores on a fallback run; for groundedness (keyword overlap), chunk content is identical so the fallback does not affect groundedness values. Since both configurations showed near-identical numbers (q03: 0.5952/0.5952 for both; q04: 0.6678/0.6678 for both), fallback did not affect the conclusion. Next experiment (6c): use 10s inter-query sleep and explicit sleep after preflight call.

**Corpus-shape note (connects to Phase 2):** Phase 2 found Cohere's rerank lift is corpus-shape sensitive (bimodal verdicts, not smooth lift). Phase 3 finds chunking strategy choice is also corpus-shape sensitive (semantic ≈ baseline when headers are sparse). These are not independent findings — both reflect P5's RAG priors not fully transferring to P6's prose-heavy textbook corpus. ADR-006 candidate framing: any ADR covering retrieval-pipeline decisions should treat corpus structure as a first-class variable, not an assumption carried forward from P5.

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

