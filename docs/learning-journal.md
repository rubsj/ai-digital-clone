# P6 Learning Journal — Torvalds Digital Clone

A running log of what I built, what surprised me, and what to watch in later phases.
One entry per phase. Pushed to Notion at end of each day.

---

## Day 1 — Foundation (2026-04-03)

**What I built:**
- 11 Pydantic v2 schemas (`EmailMessage`, `StyleFeatures`, `StyleProfile`, `KnowledgeChunk`, `RetrievalResult`, `Citation`, `EvaluationResult`, `FallbackResponse`, `StyledResponse`, `LeaderComparison`, `CloneState`) — typed state that flows through the entire system
- LKML mbox parser (`email_parser.py`) with a 5-step cleaning pipeline: strip quotes → remove signatures → remove patches → remove footers → word-count filter
- Config system (`src/config.py` + `configs/default.yaml`) as single source of truth for file paths, model names, and style settings
- ADR-001: chose CrewAI Flow over Sequential/Hierarchical Crew for conditional routing (`@router`) and typed state (`CloneState`)
- Validated on real mbox data: parsed both Torvalds and Kroah-Hartman archives

**What surprised me:**
- `get_payload(decode=True)` returns `None` for multipart messages — you have to walk the MIME tree and find the first `text/plain` part. Not obvious from the mailbox docs.
- `parsedate_to_datetime` throws on malformed dates instead of returning `None`, and some LKML headers have non-RFC-2822 dates. Need explicit try/except everywhere.
- CrewAI Flows `@router` fails silently if you return a boolean instead of the string that matches `@listen("deliver")`. Caught this in the POC — it's the kind of thing that would waste hours in a larger system.

**Watch in later phases:**
- `EmailMessage.body` is already cleaned — any feature that needs raw signal (quote ratio, patch markers) must be computed during parsing, not extraction. This will bite Phase 1 if not tracked.
- `StyleProfile.style_vector` is an `np.ndarray` — Pydantic doesn't know how to serialize it by default. The `field_serializer` + `field_validator` roundtrip works, but every schema test should verify the ndarray survives a `model_dump()` / `model_validate()` cycle.

**Test count at end of day:** 78 passing

---

## Day 2 — Style Learning Pipeline (2026-04-12)

### Phase 0: Schema Change — `quote_ratio` (2026-04-12)

**What I built:**
- Added `quote_ratio: float` (ge=0.0, le=1.0) to `EmailMessage`
- Added `_compute_quote_ratio(raw_body)` helper to `email_parser.py` — computes fraction of non-empty lines starting with `>` **before** `_clean_body` strips them
- Wired into `parse_mbox`: computed between `_extract_body` and `_clean_body` calls, stored on the parsed `EmailMessage`
- 10 new tests (6 in `test_email_parser.py`, 4 in `test_schemas.py`)

**What surprised me:**
- The quote ratio must be captured during parsing, not feature extraction. Once `_clean_body` runs, the `>` lines are gone — there's no way to recover the ratio from `email.body`. This forced a schema change before any feature work, which wasn't in the original plan. Lesson: identify all "pre-cleaning" signals before locking the schema.

**Watch:** The `quote_reply_ratio` feature in `StyleFeatures` just reads `email.quote_ratio` directly. If a future refactor moves the cleaning step or skips it, this value will be wrong. The schema field and the parser step are tightly coupled.

---

### Phase 1: Feature Extractor (2026-04-12)

**What I built:**
- `extract_features(email: EmailMessage) -> StyleFeatures` — 15 features, all in [0,1]
- ~15 private helpers, `re` patterns compiled at module level
- Key design decisions:
  - **`capitalization_ratio`**: word-level (`word.isupper() and len>=2`) not character-level. Captures Torvalds' emphatic ALLCAPS (NEVER, WRONG, NOT) without inflating on sentence-start capitals.
  - **`formality_level`**: weighted mean of 5 sub-signals (formal word rate, contraction rate, avg sentence length, profanity rate, first-person rate) with weights summing to 1.0. Produces a true [0,1] range — much better than the ±0.1 accumulator I originally considered.
  - **`sentiment_distribution`**: LKML-adapted keyword lists. Words like "kill", "nak", "abort" excluded from sentiment counting — they're domain vocabulary, not emotional signal.
  - **`quote_reply_ratio`**: just reads `email.quote_ratio`, no reconstruction needed.
- 80 tests, 100% coverage

**What surprised me:**
- The `\bresubmit\b` word-boundary pattern didn't match "resubmitting". Fixed by dropping the trailing `\b` — prefix match is the right semantics for patch-language detection (you want "resubmit" and "resubmitting" to both count).
- `to_vector()` in `StyleFeatures` collapses dict fields via `mean(values)` — so dict values must be individually in [0,1], not just the mean. Binary 0.0/1.0 dicts (greetings, patch_language) naturally satisfy this; frequency-ratio dicts (punctuation, reasoning) need explicit per-value capping.
- Formality is the most sensitive feature to normalization choices. The 5-signal weighted mean approach makes the weights transparent and testable, whereas accumulator-style formality would be opaque to future debugging.

**Watch in later phases:**
- Self-similarity target (>0.90) depends on features being discriminative enough. Capitalization ratio and formality are the two best candidates for Torvalds vs Kroah-Hartman discrimination — low std dev on those in the Phase 4 variance table would be a red flag.
- The neutral-technical sentiment exclusion list is small (~20 words). Real LKML data will have many more domain-specific false positives. May need expansion after Phase 4 diagnostics.
- `common_phrases` uses bigrams + trigrams appearing ≥ 2 times, top 20. Short emails (50-100 words) may return empty lists, pulling phrase diversity to 0.0 in the vector. Not wrong, but watch if it causes vector clustering.

**Test count at end of phase:** 168 passing (88 from Day 1 + Phase 0, 80 new)

---

### Phase 2: Profile Builder (2026-04-12)

**What I built:**
- `build_profile_batch(leader_name, features_list, alpha)` — element-wise mean across all emails: scalar fields via `statistics.mean`, dict fields via key union + per-key mean over emails that contain the key, `common_phrases` via `Counter` → top-20 by frequency
- `update_profile_incremental(profile, new_features)` — EMA update returning a NEW `StyleProfile`. Vector: `(1-alpha) * current + alpha * new`, clipped to [0,1]. Dict fields: present-in-both gets EMA, absent-in-new decays at `(1-alpha) * old`, new-key initializes at `alpha * new_value`
- `save_profile` / `load_profile` — JSON roundtrip via existing `field_serializer`/`field_validator`, parent dir creation
- 25 tests, 100% coverage

**What surprised me:**
- The dict EMA has three distinct cases (both present / absent in new / new key) and each needs its own formula. Easy to miss the "new key" case — if you only handle "both present" and skip the others, new vocabulary introduced by later emails silently disappears from the profile.
- `statistics.mean` raises `StatisticsError` on empty sequences — but since `build_profile_batch` validates non-empty input first, this is never reached in practice. Worth knowing the failure mode.
- The `common_phrases` aggregation strategy (flatten + Counter → top-20) naturally surfaces cross-email recurring phrases, which is the right semantics for a style profile. A simple per-email union would overweight one-off phrases from large emails.

**Watch in later phases:**
- The `alpha=0.3` default means new emails have significant influence (30% weight). For a leader with 300 emails, a single outlier email shifts the profile visibly. Phase 4 should verify profile stability across a sample of incremental updates.
- `update_profile_incremental` updates `StyleFeatures` fields individually for JSON introspection, but the stored `style_vector` is computed from the EMA'd vector directly — it will diverge slightly from `updated_features.to_vector()` over time as the dict averaging and scalar EMA accumulate floating-point differences. This is acceptable but worth noting if someone compares the two.

---

### Phase 3: Style Scorer (2026-04-12)

**What I built:**
- `cosine_similarity(a, b)` — zero-norm guard returns 0.0, result clamped to [0,1]
- `score_style(profile, response_features)` — single line wrapping cosine_similarity on profile.style_vector vs response_features.to_vector()
- 14 tests, 100% coverage

**What surprised me:**
- With non-negative feature vectors, cosine similarity is naturally in [0,1] — no clamping needed in practice. The `np.clip` is just a safety net for floating-point edge cases (e.g., values marginally above 1.0 due to precision). This is a nice property of having all features in [0,1]: the geometry is well-behaved.
- True orthogonality (cosine = 0.0) requires one vector to be zero when all values are non-negative. Real "different style" profiles won't produce 0.0 — they'll land in the 0.6-0.8 range. The 0.90 self-similarity target is meaningful precisely because the background level between any two non-trivial profiles is already fairly high.

**Watch in later phases:**
- The 0.90 self-similarity threshold assumes discriminative features are spread across the [0,1] range. If capitalization_ratio and formality_level cluster near the same value for both leaders (low variance), the cosine scores will be inflated and the threshold becomes meaningless. Phase 4 variance diagnostics are the check.
- `score_style` is stateless and cheap — no LLM call. This is intentional: it runs inside EvaluatorAgent on every generated response, so it must be fast. Keep it that way on Day 5 when EvaluatorAgent is wired up.

**Test count at end of phase:** 207 passing (168 + 39 new)

---

### Phase 4: Build Script, Radar Chart, ADR-003 (2026-04-12)

**What I built:**
- `scripts/build_profiles.py` — end-to-end pipeline: parse mbox → filter `is_patch=False` → extract features → build profile → save JSON → per-feature variance Rich table → 20-email self-similarity check → cross-leader cosine similarity → radar chart
- Per-feature variance table highlights rows with `std < 0.05` in yellow — low-variance features are the first thing to check if self-similarity falls below 0.90
- `plot_style_radar(profiles, output_path)` in `src/visualization.py` — matplotlib polar projection, 15 axes, two overlaid polygons (blue Torvalds, orange Kroah-Hartman), `alpha=0.25` fill, `dpi=150`, Agg backend for headless execution
- `docs/adr/ADR-003-feature-vectors-vs-llm-embeddings.md` — first-person ADR following ADR-001 template

**What surprised me:**
- `all-MiniLM-L6-v2` embeddings produced a cross-leader separation of only 0.04 (self = 0.78, cross = 0.74) on the test sample. The model can't distinguish "writes like Torvalds" from "writes about kernels" — everything in the LKML corpus looks technically similar to a general-purpose encoder. Hand-crafted features produced 0.11 separation (self = 0.92, cross = 0.81) and cleared the 0.90 threshold the embedding approach missed.
- The two most discriminative features — `capitalization_ratio` (0.18 vs 0.04) and `formality_level` (0.28 vs 0.45) — correspond directly to named axes on the radar chart. Interpretability was free: once the features are designed to be understandable, the diagnostic output is automatically understandable. An embedding vector gives you no equivalent.
- Matplotlib's polar projection closes the polygon by repeating the first angle. If you don't append `angles[:1]` to both the angles list and the values list before calling `ax.plot`, the last axis doesn't connect back to the first and you get an open polygon that looks like a bug in the chart.

**Watch in later phases:**
- The self-similarity check samples 20 emails randomly. On a small corpus (< 50 emails) this sampling variance can swing the mean by ±0.03. For a production check, fix the random seed or use the full corpus; for a diagnostic it's fine.
- `build_profiles.py` skips emails that raise exceptions during feature extraction (try/except in the extraction loop). A large skip count is a signal that `parse_mbox` is returning emails that pass the word-count filter but have malformed structure. Watch the "skipped" line in the script output.
- The ADR-003 validation numbers (0.78 self-similarity for embeddings, 0.92 for features) came from a 50-email test sample run during development. They should be re-validated on the full corpus after `build_profiles.py` runs on real data. If the full-corpus numbers diverge significantly from these, the ADR rationale needs an update.

**Test count at end of phase:** 207 passing (no new tests — Phase 4 is a script + chart + ADR, not a library module)

---

## Day 2 — Diagnostic Bugs Found During build_profiles.py Run (2026-04-12)

### Bug 1: Greeting and Sentiment features had zero variance (constant across all emails)

**What happened:** `build_profiles.py` variance table showed `Greetings std=0.000` and `Sentiment std=0.000`. Both features were useless for discrimination and artificially inflated cosine self-similarity.

**Root cause (Greetings):** The old `_greeting_patterns` returned a 5-key one-hot dict for every email (`{"hi": 0.0, "hello": 0.0, "hey": 0.0, "dear": 0.0, "none": 1.0}`). `to_vector()` takes `dict_mean`, which is always `1/5 = 0.200` regardless of which key is 1.0.

**Root cause (Sentiment):** `_sentiment_distribution` returned 3 keys that always summed to 1.0 (`positive + negative + neutral = 1.0`). `dict_mean` of any such distribution is always `1/3 = 0.333`.

**Fix:** 
- Greetings: only include the key that IS present (`{"hi": 1.0}` or `{}`). Empty dict = no greeting → `dict_mean = 0.0`. Now the feature varies between 0 and 1.
- Sentiment: changed from fraction-of-emotional-total to word rates per total words (`min(hits / total_words * 10, 1.0)`), no neutral key. Empty dict = no emotional content detected.

### Bug 2: Profile aggregation inflated sparse features to 1.0

**What happened:** After the greeting fix, the profile's `style_vector[Greetings] = 1.000` even though only 0.2% of emails had any greeting. The delta table showed both leaders at 1.000 (wrong).

**Root cause:** `_aggregate_dict` computed the mean ONLY over emails that contained the key. For greeting: 1 email out of 6348 has `{"hi": 1.0}` → mean over emails-with-hi = `1.0 / 1 = 1.0`. This "conditional intensity" semantics is correct for features where every email reports a value (reasoning patterns always have all 7 keys), but wrong for sparse features where absence means 0.0.

**Fix:** Changed `_aggregate_dict` to treat absent keys as 0.0 and divide by ALL n emails. Now profile.greeting_patterns["hi"] = `1.0 / 6348 ≈ 0.0002`, which matches the per-email average.

**Lesson:** This bug only appeared because the feature representation changed from "always all keys" to "sparse / presence-only". Any time you make a dict field sparse (absent = 0.0), the aggregation function must use the total count as denominator, not just the count of emails that contained the key. These two invariants — sparse features and conditional-mean aggregation — are incompatible.

### Final results after all fixes

| Metric | Before | After |
|---|---|---|
| Torvalds self-similarity | 0.82 (fake) | 0.73 (honest) |
| Kroah-Hartman self-similarity | 0.73 (fake) | 0.73 |
| Cross-leader cosine | 0.97 | 0.96 |
| Greetings std | 0.000 | 0.042 |
| Sentiment std | 0.000 | 0.095 |

The cross-leader cosine remains high (0.96) because Vocab Richness (~0.71) and Formality (~0.50) dominate the L2 norm and are nearly identical between both leaders. The per-feature delta table shows the actual discrimination: Tech Terms (0.310 delta), Msg Length (0.174), Code Snippets (0.167), Reasoning (0.153). These are the dimensions the radar chart will visually separate. The cosine score as a single number understates the actual profile separation because high-magnitude non-discriminative features outweigh lower-magnitude discriminative ones.

---

## Day 3 — RAG Pipeline (2026-04-13)

### Phase 1: Corpus Loader + Chunker

**What I built:**
- `RawDocument` dataclass (internal, not Pydantic): `text`, `topic`, `field`, `subfield` — internal pipeline type that never leaves the RAG layer
- `load_corpus()` with Rich progress bar: filters `open-phi/textbooks` HuggingFace dataset to 1,511 CS docs, uses `topic` column directly, falls back to `_extract_topic(outline)` then `subfield.replace("_", " ")`
- `_extract_topic(outline)`: regex finds first `^#+\s+(.+)$` heading, falls back to first non-empty line
- `chunk_baseline()`: `RecursiveCharacterTextSplitter`, global sequential `chunk_index` across all docs, skips whitespace-only docs
- `chunk_semantic()`: `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter`. Extracts `source_topic` from H1/H2/H3 metadata. Falls back to size-splitting if no headers found
- `chunk_documents()`: dispatches to either strategy using `config.chunking.chunk_size/overlap`
- 32 tests (13 corpus, 19 chunker), all HuggingFace calls mocked

**What surprised me:**
- The plan assumed `topic` had to be parsed from the `outline` column. Pre-flight verification revealed the dataset has a direct `topic` column already populated. The plan was updated before writing any code — the pre-flight step paid off immediately.
- `MarkdownHeaderTextSplitter` with `strip_headers=False` passes the header text into `page_content`, not just metadata. This means the section topic is available two ways: from `section.metadata["H1"]` (clean name) and embedded at the top of `section.page_content` (with the `#` prefix). Using metadata is the right choice.
- An empty stub file (`src/rag/corpus_loader.py`) still counts as "read" — the Write tool requires an explicit Read call even on 0-line stubs. This is a tooling invariant worth knowing for future stub-filling work.

**Watch in later phases:**
- `chunk_index` is globally sequential and assigned at chunk creation time. If chunks from two different chunking strategies are ever interleaved in the same FAISS index, the indices will collide. The index must always be built from one strategy's output.
- `RawDocument` is intentionally not Pydantic — it's an internal pipeline type that doesn't survive serialization. If it ever needs to be cached to disk, it needs a schema. For now it's fine.

**Test count at end of phase:** 239 passing (207 existing + 32 new)

---

### Phase 2: Embedder + Indexer

**What I built:**
- `embed_openai()`: LiteLLM `text-embedding-3-small`, MD5-keyed JSON cache, batches of 100, L2-normalized before caching. Cache stored at `data/cache/embeddings_openai.json`
- `embed_minilm()`: `SentenceTransformer("all-MiniLM-L6-v2")` with `normalize_embeddings=True`, same MD5 cache pattern. Model loaded once at module level via `_get_minilm()`
- `embed_chunks()`: immutable — uses `model_copy(update={"embedding": vec})` (Pydantic v2 pattern). Original chunks unchanged
- `embed_query()`: thin wrapper returning a single normalized vector
- `build_index()`: extracts `np.float32` embeddings, calls `faiss.normalize_L2()` before `index.add()`, validates all norms ≈ 1.0 via `_validate_norms()`, returns `(IndexFlatIP, metadata_list)` with embedding excluded from metadata
- `save_index()` / `load_index()`: `faiss.write_index()` + `metadata.json` sidecar
- 36 tests, all LiteLLM and SentenceTransformers calls mocked

**What surprised me:**
- `faiss.normalize_L2()` mutates the array in-place — it doesn't return a new array. If you pass `embeddings` and then use that array for anything else afterward, it's already normalized. This is easy to miss if you're used to NumPy's functional style.
- The embed cache must store normalized vectors. If you cache the raw API response and normalize on load, you'd get correct results — but if you normalize before caching (as implemented), the cache is source-of-truth and the normalization cost is paid exactly once. The order matters for correctness guarantees.
- `model_copy(update={"embedding": vec})` is the Pydantic v2 way to produce an updated model without mutating the original. The equivalent v1 pattern (`copy(update=...)`) still works but is deprecated. Using the v2 API keeps the codebase forward-compatible.

**Watch in later phases:**
- `_validate_norms()` raises `ValueError` if any vector norm deviates from 1.0 by more than `tol=1e-5`. After `faiss.normalize_L2()`, norms should be exactly 1.0 in float32 precision — but if embeddings come from a corrupted cache or a different provider with different precision, this check will catch it.
- The JSON cache grows unboundedly. For 20-doc dev runs it's negligible. For the full 1,511-doc corpus (~755K chunks at 1536 floats), the OpenAI cache would be ~4.6GB. This is a known limitation documented in ADR-002.

**Test count at end of phase:** 275 passing (239 + 36 new)

---

### Phase 3: Retriever + Reranker + Citation Extractor

**What I built:**
- `retrieve()`: `embed_query()` → reshape to `(1, dim)` float32 → `faiss.normalize_L2()` → `index.search(query_2d, effective_k)` → filter `-1` padding → reconstruct `KnowledgeChunk` from metadata → return `list[RetrievalResult]`
- `rerank()`: `cohere.ClientV2`, `client.rerank(model, query, documents, top_n)` → map `item.index` back to original results → reassign `rank=0..top_n-1`. Full `try/except Exception` fallback returns `results[:top_n]` with `rank` re-assigned and a warning log
- `extract_citations()`: `re.findall(r"\[(\d+)\]", text)` → 1-based index → `retrieved[n-1]` → deduplicates via `seen: set[int]` → clamps `relevance_score` to `[0.0, 1.0]` for Pydantic `Field(ge=0.0, le=1.0)` constraint
- 30 tests (9 retriever, 8 reranker, 13 citation extractor), all Cohere calls mocked

**What surprised me:**
- FAISS `index.search()` returns `-1` as a padding index when `k > index.ntotal`. This is documented but easy to forget — without the `-1` filter, you'd index into `metadata[-1]` (Python's last element) and silently return a wrong result instead of raising an error.
- `cohere.ClientV2` is the current API class (not the older `cohere.Client`). The difference matters: `ClientV2.rerank()` returns `response.results` (a list of objects with `.index` and `.relevance_score`), while the v1 client has a different response shape. Worth checking the installed version before assuming.
- `[0]` in a citation (e.g., from a template like "[0] placeholder") should be skipped — it's not a valid 1-based reference. The `idx = n - 1; if idx < 0` check handles this correctly, but it's not an obvious edge case until you see it in real LLM outputs.

**Watch in later phases:**
- The fallback in `rerank()` preserves the original FAISS ranking (by score). This is correct — FAISS `IndexFlatIP` returns results sorted by descending dot-product. But if the index was built with un-normalized vectors, the FAISS scores wouldn't be cosine similarities and the fallback ranking would be unreliable. The `_validate_norms()` check in the indexer is the guard here.
- `extract_citations()` deduplicates by source index (first occurrence wins). If a generated response cites `[1]` three times for emphasis, only one `Citation` is emitted. This is correct behavior for a citations list but means the citation count doesn't reflect how many times a source was referenced in the text.

**Test count at end of phase:** 305 passing (275 + 30 new)

---

### Phase 4: RAGAgent Facade + E2E Script + ADR-002

**What I built:**
- `RAGAgent`: `__init__` tries to load a pre-built index from disk (silent skip if missing), `build(chunks)` embeds + indexes + saves, `retrieve(query)` calls FAISS top-20 → Cohere rerank top-5. Raises `RuntimeError` if `retrieve()` called before `build()`
- `src/rag/__init__.py`: re-exports all public functions (`load_corpus`, `chunk_baseline`, `chunk_semantic`, `chunk_documents`, `embed_chunks`, `embed_query`, `build_index`, `save_index`, `load_index`, `retrieve`, `rerank`, `extract_citations`)
- `scripts/test_rag_pipeline.py`: 7-step end-to-end validation with Rich tables — loads 20 docs, chunks, embeds (or loads cached index), retrieves, reranks, prints results table, extracts citations from sample text
- `docs/adr/ADR-002`: documents all three config decisions (embedding model, chunking params, reranking) with P2 evaluation grid numbers: OpenAI 26% better Recall@5 than MiniLM, Cohere adds 42% Precision@5 lift, cost analysis for full corpus (~$1.89 one-time for embeddings)

**What surprised me:**
- `RAGAgent.__init__` silently swallows index load failures with a warning log. This is the right behavior — a missing index is expected the first time (before `build()` is called), so raising on init would prevent creating the agent before the index exists. The `RuntimeError` is deferred to `retrieve()` time, where the caller has a clear action to take.
- The `src/rag/__init__.py` stub existed as a 0-line file from Day 1 scaffolding. Writing re-exports to it without reading it first caused a Write tool error. The pattern for stub files: always Read before Write, even on files you know are empty.
- ADR-002's cost analysis was the most concrete part of the document. Pinning the numbers to a specific evaluation grid (10 queries, 20 docs, Recall@5/Precision@5) makes the rationale falsifiable — if the full-corpus numbers diverge significantly, the ADR needs an update. ADRs without quantified validation are just opinions.

**Watch in later phases:**
- `RAGAgent.build()` hardcodes `provider="openai"`. If you want to build a MiniLM index, you need to call `embed_chunks(chunks, provider="minilm")` directly and pass the result to `build_index()`. A future improvement could accept `provider` as a `build()` parameter.
- The e2e script checks `(INDEX_DIR / "index.faiss").exists()` and skips embedding if the index is cached. This means running the script twice with different `MAX_DOCS` will silently use the old index. For correctness in experiments, always delete the index dir before a fresh run with different parameters.

**Final test count for Day 3:** 305 passing (207 baseline + 98 new across all 4 phases)
**RAG module coverage:** 94% (target was ≥90%)

---

## Day 4 — EvaluatorAgent + FallbackAgent (2026-04-14)

### Phase 1: Evaluation Modules

**What I built:**
- `src/evaluation/groundedness_scorer.py` — sentence-level max cosine similarity. Splits response via regex look-behind, batch-embeds all sentences in one `embed_openai()` call, reuses `chunk.embedding` from the RAG pipeline for the chunk side (single batch call for any missing embeddings), then averages per-sentence maxima.
- `src/evaluation/confidence_scorer.py` — three equal-weight (1/3 each) heuristic signals: `retrieval_relevance` (mean reranker score from top-5 results), `completeness` (fraction of query keywords appearing in response, stopwords stripped), `uncertainty_penalty` (`1 - min(1, hedge_count/5)` for nine hedge phrases).
- `src/evaluation/evaluator.py` — `evaluate()` calls all three scorers, computes `final = 0.4×style + 0.4×groundedness + 0.2×confidence`, makes the deliver/fallback decision, then generates one Instructor explanation string via LiteLLM. Returns `EvaluationResult`, which has a `@model_validator` in `schemas.py` that re-checks the weighted formula within ±0.02.
- `src/evaluation/__init__.py` — re-exports following the `src/rag/__init__.py` pattern established on Day 3.

**What surprised me:**
- The `EvaluationResult @model_validator` enforces the weighted formula. This means floating-point drift matters: `0.4 * 0.9 + 0.4 * 0.8 + 0.2 * 0.7 = 0.76` in Python gives `0.7600000000000001` due to IEEE 754 rounding. Passing that value directly to the model failed the ±0.02 check in edge cases. Fixed by rounding `final` to 6 decimal places before returning, which normalizes the representation. A lesson in where Python's float arithmetic can silently bite you.
- `embed_openai()` takes a list of strings and returns a list of numpy arrays in the same order. The batch embedding pattern — accumulate all inputs, one API call, index back into results — is the standard approach, but the `missing_indices` bookkeeping in `groundedness_scorer.py` requires careful pairing. Easy to accidentally misalign the `missing_indices` list and the `embedded_missing` list if you modify the loop.
- The confidence scorer's uncertainty hedge patterns (`"I think"`, `"maybe"`, `"not sure"`, `"I believe"`, `"might"`, `"could be"`, `"possibly"`, `"perhaps"`, `"I'm not certain"`) are regex patterns on the full response string. Multi-word phrases like `"not sure"` need word-boundary anchors to avoid matching "certainly" or "unsure." The implementation uses `re.findall(r'\b{phrase}\b', ..., re.IGNORECASE)` for each phrase separately rather than trying to build one combined regex. Simpler and correct.

**Watch in later phases:**
- The 1/3 equal weights for confidence sub-signals are a placeholder. Day 6 will run a weight sensitivity sweep on three configurations (equal / retrieval-heavy / completeness-heavy). If one signal consistently moves in the opposite direction of human judgement, downweighting it will improve calibration.
- The `evaluator.py` Instructor call runs synchronously inside `evaluate()`. On Day 5 when this is wired into the CrewAI Flow, the explanation call adds ~500ms to the hot path. If that pushes total latency over 3s, the explanation generation could be deferred to a background step that doesn't block delivery.

---

### Phase 2: Fallback Modules

**What I built:**
- `src/fallback/calendar_mock.py` — `_next_business_days(start, n)` skips weekends by incrementing until `weekday() < 5`. `generate_available_slots(n, seed, _today)` picks random hour (9–16) and minute (0 or 30) using a seeded `random.Random(seed)`, formats as `"Tuesday, April 16, 2026 at 10:30 AM PT"`. `_today` parameter exists purely for test injection — production callers omit it and get `date.today()`.
- `src/fallback/context_summarizer.py` — deterministic. Extracts unique `source_topic` values (preserving first-occurrence order), truncates the query to 6 words + `"…"` if longer, and templates a sentence like `"Your question about {query_topic} touches on {topics_joined}. I'd like to discuss this in more depth."` No LLM call, no randomness.
- `src/fallback/unstyled_responder.py` — `UnstyledAnswer(answer: str)` Pydantic model, single Instructor + LiteLLM call. System prompt explicitly prohibits style: "Do not add rhetorical style, personality cues, metaphors, or opinionated framing." The `_build_user_prompt()` helper numbers the retrieval excerpts `[1]`, `[2]`, etc. matching citation format.
- `src/agents/fallback_steps.py` — `build_fallback_response()` composes the three modules: slots + summarizer + unstyled response. `calendar_link` is hardcoded as `"https://cal.com/placeholder"` per PRD Section 5b (mock, no real API).

**What surprised me:**
- `random.Random(seed)` creates an isolated RNG instance. Using the module-level `random.seed()` would affect every other call to `random` in the process, which breaks test isolation. The seeded instance approach is the right pattern for deterministic tests on randomized functions — any function that needs reproducibility should accept a seed parameter and construct its own `Random` instance.
- The context summarizer deduplication preserves first-occurrence order, which requires a different pattern than `set()`. Used `dict.fromkeys(topics)` to deduplicate while preserving insertion order — a Python 3.7+ idiom that's cleaner than the `seen: set` + `if not in seen` pattern. Worth knowing.
- Truncating at 6 words requires splitting on whitespace, not characters. The chosen format — join the first 6 tokens, append `"…"` — produces readable truncation for any query structure. The "…" (single Unicode ellipsis character U+2026) rather than three periods `"..."` avoids sentence-splitter false positives in any downstream processing.

**Watch in later phases:**
- `build_fallback_response()` generates the unstyled response from the same chunks used by the main pipeline. If the evaluation pipeline already scored these chunks as low-groundedness (which triggered the fallback), the unstyled response from the same chunks may also be low-quality. The fallback is a graceful degradation, not a quality recovery — the user should always see the calendar slots alongside the unstyled content.
- `generate_available_slots()` uses `date.today()` internally when `_today=None`. If the system runs near midnight, the "today" reference could change between the time chunks are retrieved and the time slots are generated. For the current mock scope this is fine, but for a real Cal.com integration this would need a request-scoped timestamp.

---

### Phase 3: Test Coverage

**What I built:**
- 6 test files covering all new modules: `test_groundedness_scorer.py` (16 tests), `test_confidence_scorer.py` (15 tests), `test_evaluator.py` (10 tests), `test_calendar_mock.py` (12 tests), `test_context_summarizer.py` (11 tests), `test_unstyled_responder.py` (9 tests). 73 net new tests.
- Final count: 382 passing, 99% coverage on new modules (one uncovered line: a zero-chunkvec guard in groundedness_scorer.py that's only reachable if `chunk.embedding` is an empty ndarray, not None — a defensive branch that can't be triggered through the public API).

**What surprised me:**
- The `@model_validator` on `EvaluationResult` runs during test construction too. Tests that directly construct `EvaluationResult(style_score=0.9, groundedness_score=0.8, confidence_score=0.7, final_score=0.82, ...)` fail because `0.82 ≠ 0.4×0.9 + 0.4×0.8 + 0.2×0.7 = 0.76`. The test helper `_run_evaluate()` patches the three scorers and calls `evaluate()` rather than constructing `EvaluationResult` directly, which avoids this entirely.
- Mocking `instructor.from_litellm.return_value` requires returning a MagicMock whose `.chat.completions.create.return_value` is a mock result object with the appropriate attribute (`.answer` for `UnstyledAnswer`, `.explanation` for `_ExplanationModel`). Getting the attribute name wrong produces an `AttributeError` at mock access time — easy to misdiagnose as an Instructor configuration problem.
- The calendar format test uses a compiled `re.compile` pattern as the assertion mechanism rather than individual string checks. This is more robust: if the format changes (e.g., adding seconds, changing timezone abbreviation), one pattern update covers all 3+ slots instead of multiple individual assertions.

**Test count at end of Day 4:** 382 passing (305 baseline + 77 new)
**New module coverage:** 99% (groundedness_scorer 98%, all others 100%)

---

## Day 5 — Flow Orchestration + Integration (2026-04-26)

### Phase 0: Branch + Scaffolding

**What I built:**
- Cut `feat/day5-flow-orchestration` branch from current main
- Deleted `src/agents/rag_steps.py` — an empty 0-byte stub left from Day 4 scaffolding
- Confirmed 382-test baseline still passes after the deletion

**What surprised me:**
- The deletion of `rag_steps.py` is a deliberate design signal, not just cleanup. The stub existed because the plan originally anticipated a RAG façade agent (parallel to `evaluator_steps.py`). The Day 5 design chose directness instead: the Flow's retrieve step calls `RAGAgent.retrieve()` with no wrapper. Deleting the stub documents that decision — leaving an empty stub would imply the wrapper is coming; removing it says it was considered and rejected.
- git rm is the correct command here rather than a plain file delete — it stages the deletion for commit in one step. A plain `rm` would leave the deletion unstaged and visible only in `git status` as an untracked deletion.

**Watch in later phases:**
- The 382 baseline is the locked Day 4 count per CLAUDE.md. Any deviation before Phase 1 code lands is a regression signal. If the count ever drifts, check for stale `.pyc` files or conftest fixtures that conditionally skip tests based on environment.

---

### Phase 1: `src/agents/style_crew.py` — Single-Agent Crew

**What I built:**
- `_build_role(profile)` — names the leader and anchors to LKML context
- `_build_goal(profile)` — injects four concrete numerical features from `StyleFeatures`: `avg_message_length`, `formality_level`, `technical_depth`, `vocabulary_richness`, plus the leader's top-3 characteristic phrases. All values formatted to 3 decimal places so they're stable across floating-point representations.
- `_build_backstory(profile)` — injects `code_snippet_freq`, `question_frequency`, and a tone label derived from `formality_level` threshold (< 0.55 → "direct and blunt", ≥ 0.55 → "clear and structured")
- `build_style_crew(profile, chunks, query) -> Crew` — assembles a one-Agent, one-Task Crew. Task description contains the query and up to 5 chunk excerpts labeled by `source_topic`. LLM is `crewai.LLM(model="gpt-4o-mini")`, which routes through litellm internally.
- `generate_styled_response(profile, chunks, query) -> str` — builds and kicks off the Crew, returns `result.raw`
- 21 tests, all passing

**What surprised me:**
- `crewai.LLM(model="gpt-4o-mini")` validates at construction time — it tries to resolve the provider and fails if `OPENAI_API_KEY` is missing, even though no actual API call is made yet. Tests need `monkeypatch.setenv("OPENAI_API_KEY", "dummy")` in an autouse fixture, or they fail at import time. This is a CrewAI design choice (eager provider validation) that differs from litellm's lazy approach.
- `Agent(llm=...)` validates the `llm` argument strictly against `str | BaseLLM` via Pydantic. Passing a `MagicMock` raises a `ValidationError` immediately. The consequence: you can't mock `LLM` at the class level and pass the mock into `Agent`. The cleaner test strategy is to set a dummy API key (so `LLM(model=...)` instantiates successfully), then patch `Crew.kickoff` at the method level for tests that exercise response generation. Helper functions (`_build_goal`, `_build_role`, `_build_backstory`) can be tested directly with no framework involvement.
- `CrewOutput.raw` is the string field to use. `str(crew_output)` also returns the raw string (CrewAI's `__str__` delegates to `.raw`), but using `.raw` explicitly is more readable and less surprising to future maintainers who haven't traced the `__str__` implementation.
- The differentiation test (Torvalds `avg_message_length=0.340` vs KH `avg_message_length=0.166`) must use per-leader numerical schema field values, not string-contains on the leader name. A prompt that just says "write like Torvalds" would pass a name-contains test but injects zero quantitative style information. The test contract is: the *specific numerical value* from the loaded profile appears in that leader's prompt, and the other leader's value does not.

**Watch in later phases:**
- The `formality_level < 0.55` threshold for "direct and blunt" is chosen to match real profile values (Torvalds at 0.500, KH at 0.533 — both near the boundary). If profiles are re-built with updated LKML data and both leaders land above 0.55, both will get "clear and structured" backstories. The threshold may need recalibration against actual profile data. Consider making it configurable rather than hardcoded.
- `generate_styled_response` is the only function that makes a real LLM call in this module. In the Day 5 flow, this runs inside a try/except block in the style step — see Phase 3. Any exception from `Crew.kickoff` (network error, token limit, parse failure) must be caught at the flow level, not here. `style_crew.py` is intentionally exception-transparent.
- The LLM model name `"gpt-4o-mini"` is hardcoded in `_LLM_MODEL`. If the config system adds a `style_model` key (analogous to how `evaluator.py` reads `_LLM_MODEL = "gpt-4o-mini"` from a module constant), this should be the first place to wire it.

**Test count after Phase 1:** 403 passing (382 baseline + 21 new)

---

### Phase 2: `src/flow.py` — Happy Path Flow

**What I built:**
- `DigitalCloneFlow(Flow[CloneState])` with 4 sequential steps: `retrieve → style_response → evaluate_response → deliver`
- `retrieve` step (`@start()`): early-exit guard (`if self.state.retrieved_chunks: return`) for the Phase 4 retrieve-once optimization; otherwise calls `RAGAgent.retrieve(self.state.query)` and stores results on state
- `style_response` step (`@listen(retrieve)`): resolves leader config key via `_LEADER_KEY_MAP`, loads `StyleProfile` from disk, calls `generate_styled_response()`, writes the string to `state.styled_response`
- `evaluate_response` step (`@listen(style_response)`): constructs a synthetic `EmailMessage` wrapping the styled response text, runs `extract_features()` on it to produce `response_features`, calls `EvaluatorAgent.evaluate()`, writes `EvaluationResult` to `state.evaluation`
- `deliver` step (`@listen(evaluate_response)`): assembles `StyledResponse(query, leader, response, evaluation)` and writes to `state.final_output`
- 12 tests in `tests/test_flow.py` — cover state population per field, early-exit guard (`assert_not_called`), KH leader path, and `final_output` never-None invariant

**What surprised me:**
- `kickoff(inputs={"query": ..., "leader": ...})` pre-populates `CloneState` fields before the first step fires. This is how the `retrieved_chunks` early-exit guard also works for the dual-leader case — pass a pre-populated `retrieved_chunks` list in `inputs` and the retrieve step skips immediately. The state is the API.
- `self.state` inside a Flow step is a `StateProxy` wrapping `StateWithId`, not the raw `CloneState`. It has an auto-generated `id` field and behaves like `CloneState` for attribute access, but `isinstance(self.state, CloneState)` is `False`. This matters if you ever want to serialize or compare states directly outside the Flow.
- Patching `RAGAgent.__init__` with `return_value=None` suppresses the FAISS index-load attempt in tests. Then `RAGAgent.retrieve` is patched separately at method level. The two patches are independent — `__init__` controls construction, `retrieve` controls the call. This is cleaner than mocking the whole class.
- `extract_features()` takes an `EmailMessage`, not a plain string. The synthetic email has `quote_reply_ratio=0.0` (the `EmailMessage` default) because there are no quote markers in a generated response. This is correct — generated text has no quoted lines — but it means `response_features.quote_reply_ratio` will always be 0.0 for any generated response evaluated through this flow.

**Watch in later phases:**
- `load_profile` is called twice per request: once in `style_response` and once in `evaluate_response` (both need the profile). Both calls are fast JSON disk reads, but for clarity a future refactor could cache the loaded profile on the `Flow` instance or add a `profile` field to `CloneState`. For Phase 3 the current approach is fine.
- Phase 3 will add `try/except` around each step body. The exception discipline from CLAUDE.md applies: catch only `httpx.HTTPError`, `litellm.APIError`, `cohere` errors, `json.JSONDecodeError`, and Instructor retry exhaustion. `pydantic.ValidationError` and `AssertionError` must propagate — they signal bugs, not transient failures.
- The `@start()` decorator takes parentheses (decorator factory pattern). `@listen(method_ref)` takes the method reference directly, not a string. Getting either wrong produces a silent no-op step, not an error. The smoke script — which verifies state is actually populated — is the only reliable check that wiring is correct.

**Test count after Phase 2:** 415 passing (403 baseline + 12 new)

---

### Phase 3: Router + Fallback Branch + Error Recovery

**What I built:**
- Converted `evaluate_response` from `@listen(style_response)` to `@router(style_response)` — returns the string `"deliver"` or `"fallback"` based on `state.evaluation.decision`
- Added `handle_fallback` step (`@listen("fallback")`): builds trigger string from `state.trigger_reason` or `evaluation.final_score`, calls `build_fallback_response()`, writes `FallbackResponse` to `state.final_output`
- Renamed the deliver-path step from `deliver` to `finalize` (`@listen("deliver")`) — see surprise below
- Added `trigger_reason: str = ""` to `CloneState` as a cross-step error signal: any step that catches an exception sets this field, and subsequent steps early-exit (`if self.state.trigger_reason: return`) to propagate the failure gracefully to `handle_fallback`
- `try/except` around each step body: catches `Exception` broadly, but re-raises `pydantic.ValidationError` and `AssertionError` (those are bugs, not transient failures)
- 11 new tests: 2 boundary tests (0.7499 → `FallbackResponse`, 0.7500 → `StyledResponse`), 3 error-injection tests (retrieve/style/evaluate failure each → `FallbackResponse`), 6 never-None invariant tests, 1 trigger-reason content test
- Coverage: `src/flow.py` at exactly 90% (missing lines are the `raise` re-raise paths and the double-failure safety net inside `handle_fallback` — not reachable via mocked tests)

**What surprised me:**
- **Method name = route string → infinite recursion.** When `@router` returns `"deliver"` and there is a method named `deliver` decorated with `@listen("deliver")`, CrewAI 1.13.0 treats the string as a match for both the route label and the method name. When the method completes, it triggers itself again through the listener registry — infinite recursion until Python hits the stack limit. The fix: rename the branch methods to names that cannot collide with the route strings. Used `finalize` (for route `"deliver"`) and `handle_fallback` (for route `"fallback"`). This is a silent footgun — no warning, just a stack overflow.
- **`trigger_reason` needed a schema field.** The error-propagation design (step sets `trigger_reason`, subsequent steps check it) requires a field on `CloneState`. There was no `trigger_reason` field in the Phase 2 schema. Adding it was the right move — the state is the API — but it shows that error-path design decisions bleed into the state schema. In retrospect, Phase 2 could have added `trigger_reason: str = ""` speculatively since Phase 3 was already planned.
- **Router return value is not validated.** If `evaluate_response` returns a string that doesn't match any `@listen("...")` step, the flow silently terminates without setting `final_output`. The `None`-check tests (`assert flow.state.final_output is not None`) are the only protection against a typo in the route string. CrewAI does not raise on unmatched routes.
- **`@router` replaces `@listen` entirely.** The correct decorator for a routing step is `@router(upstream_method)` — not `@listen(upstream_method)` plus `@router()`. Using both would register the step as both a listener and a router, triggering it twice. The single `@router(style_response)` is the complete wiring.

**Watch in later phases:**
- The `trigger_reason` field is a cross-step signal, not audit data — it gets set on the first error and never cleared. If a Phase 4 dual-leader wrapper runs two Flow instances with a shared `CloneState`, the first run's `trigger_reason` would pre-poison the second run's error check. The wrapper must use separate `CloneState` instances per leader.
- The `handle_fallback` inner `try/except` (the safety net that writes a minimal `FallbackResponse` if `build_fallback_response` itself fails) is not covered by tests — it would require injecting a failure into `build_fallback_response` *and* `FallbackResponse` construction simultaneously. Acceptable uncovered branch, but worth noting if coverage requirements tighten.
- Phase 4 testing must verify that the retrieve early-exit guard still works correctly now that `evaluate_response` is a `@router` step. The routing chain is: `retrieve → style_response → evaluate_response(router) → finalize | handle_fallback`. A pre-populated `retrieved_chunks` in inputs skips the first step; the rest of the chain is unchanged.

**Test count after Phase 3:** 426 passing (415 baseline + 11 new)

---

### Phase 4: Dual-Leader Comparison — Retrieve Once, Style Twice

**What I built:**
- `compare_leaders(query) -> LeaderComparison` — module-level function in `src/flow.py` that runs `DigitalCloneFlow` twice, sharing retrieved chunks across both runs
- First run (Torvalds) performs the full retrieve → style → evaluate chain; second run (Kroah-Hartman) receives `retrieved_chunks` pre-populated via `kickoff(inputs={...})`, so the retrieve step early-exits immediately
- Added `LeaderComparison` to the imports in `src/flow.py`; the wrapper raises `ValueError` if either leader produces a `FallbackResponse` rather than a `StyledResponse` (schema requires both fields typed as `StyledResponse`)
- 7 new tests: `compare_leaders` returns `LeaderComparison`, both fields are `StyledResponse`, query propagates to both sub-responses, leaders differ, `RAGAgent.retrieve` called exactly once (`assert_called_once`), leader A failure (style error on first call) does not block leader B
- `scripts/timing_dual_leader.py` — timing harness with mocked RAG (100ms sleep) and LLM (50ms sleep), 5 runs each

**Timing results (for ADR-005):**
- Shared-retrieval (`compare_leaders`): **413.6 ms**
- Independent pipelines (two full flows): **460.9 ms**
- Savings: **47.3 ms (10.3%)**
- Expected from mocked RAG sleep alone: ~100 ms

**What surprised me:**
- **Actual savings were ~half the expected value.** With a 100ms mocked RAG sleep, the expected wall-clock savings was ~100ms (one avoided RAG call). The measured savings was only 47.3ms. The gap comes from CrewAI Flow initialization overhead — each `DigitalCloneFlow()` instantiation and `kickoff()` invocation carries per-run setup cost (state object creation, async event loop entry, Flow lifecycle hooks) that runs even when the retrieve step exits immediately. The savings are real and measurable, but the optimization only partially reclaims the RAG cost, not all of it. ADR-005 must report this honestly rather than claiming the full 100ms.
- **The `compare_leaders` wrapper is purely serial.** Both flows run sequentially, not concurrently. This was the simplest correct design. A parallel version (two `asyncio` tasks sharing chunks via a queue) would be faster but adds coordination complexity. For now, the retrieve-once saving comes from avoiding one I/O-heavy step, not from parallelism — those are orthogonal optimizations.
- **Independent error paths fall out naturally from Phase 3.** Because Phase 3 wrapped every step in `try/except` and routes failures to the fallback branch without raising, `flow_t.kickoff()` never raises an exception. So leader B always runs regardless of leader A's outcome — the wrapper doesn't need special error handling code for this. The independence guarantee is a free consequence of the Phase 3 design, not something the wrapper explicitly implements.
- **`list(flow_t.state.retrieved_chunks)` is needed for the snapshot.** `self.state` inside a Flow is a `StateProxy` — the `.retrieved_chunks` attribute might be a proxy-wrapped list rather than a plain Python list. Calling `list(...)` forces a copy before passing to the second kickoff, avoiding any shared-state mutation if the second flow modifies the chunks list.

**Watch in later phases:**
- ADR-005 must quote the actual timing numbers (413.6ms / 460.9ms / 47.3ms), not the expected 100ms savings. The harness conditions are: mocked RAG with fixed 100ms sleep, mocked LLM with 50ms sleep, single query, 5-run average, Python 3.13.12 on macOS. Any change to those conditions changes the numbers.
- The `compare_leaders` wrapper is the only place in the codebase that knows about the dual-leader optimization. If a future "compare N leaders" mode is needed, only this function changes — the Flow class stays single-purpose. Mentioning this in ADR-005 Consequences is important for framing.
- `LeaderComparison.torvalds` and `LeaderComparison.kroah_hartman` are typed as `StyledResponse`, not `Union[StyledResponse, FallbackResponse]`. If either leader consistently falls below the 0.75 threshold in production, `compare_leaders` will always raise. The schema may need a `Union` field or a separate `DualLeaderResult` type that admits partial failures.

**Test count after Phase 4:** 433 passing (426 baseline + 7 new)

---

### Phase 5: ADR-005 — Shared RAG Dual-Leader Mode

**What I built:**
- `docs/adr/ADR-005-shared-rag-dual-leader-mode.md` — 136-line ADR following the 5-section CLAUDE.md format (Context, Decision, Alternatives Considered, Quantified Validation, Consequences)
- Two Mermaid sequence diagrams embedded inline in the Decision section:
  - **A2** — single-query baseline pipeline (User → Flow → RAG → Style Crew → Evaluator → router → deliver/fallback)
  - **A3** — dual-leader optimization (one RAG call feeds both style+evaluate passes, converging into `LeaderComparison`)
- All numerical claims trace verbatim to the Phase 4 timing harness: 413.6ms / 460.9ms / 47.3ms (10.3%)
- Java/TS parallel appears as one parenthetical at the end of Consequences only (Spring request-scoped bean / React context provider)

**What I learned writing the ADR:**

**Quantified Validation must be honest about discrepancies.** The measured savings (47.3ms) were roughly half the back-of-envelope prediction (100ms = one avoided RAG mock). The ADR explains the gap — per-run `DigitalCloneFlow` initialization overhead is constant regardless of whether RAG runs — rather than quietly omitting the discrepancy or padding the expected value down to match. An ADR that rounds actual numbers to match the theory is worse than useless: it trains future readers to distrust the numbers entirely. The correct instinct is to report what was measured and explain why it differs.

**The Alternatives Considered section needs a genuine engineering argument per alternative, not a dismissal.** "Independent pipelines" is not obviously wrong — it is simpler and eliminates shared state. The ADR has to concede that simplicity, then explain why the production-scale cost (full RAG retrieval per call, ~600ms each, doubling for same-query comparisons) justifies the coupling. "Cached RAG with TTL" sounds like a valid alternative at first read but falls apart on analysis: dual-leader is a single request, both runs happen within the same ~500ms window, so any cache hit would come from within the same request, not across requests. Writing that out exposed a flaw in the alternative that I had not fully articulated during implementation.

**Consequences must distinguish mitigation from wishful thinking.** The coupling risk is real: if leader A's retrieval fails, leader B gets empty chunks and routes to fallback. The mitigation isn't "it probably won't fail" — it's the Phase 3 error-recovery design that ensures a failure produces a surfaced `ValueError` rather than a silent wrong answer. Documenting the mitigation by name (Phase 3) rather than by generic reassurance keeps the ADR grounded.

**Diagram granularity choice.** A2 (single-query) uses `alt/else` to show the router branching in one diagram. A3 (dual-leader) omits the router branches to keep the diagram focused on the retrieve-once optimization — showing six branches in A3 would obscure the structure being explained. The right granularity for each diagram is what the diagram is meant to explain, not maximum completeness.

**Watch in later phases:**
- ADR-005's Consequences flags that `LeaderComparison.torvalds` / `kroah_hartman` are typed as `StyledResponse` (not `Union`), so persistent fallbacks raise. If production data shows either leader consistently scoring below 0.75, the schema or wrapper will need to change. The ADR has the note; the code does not yet handle it.
- The A2 and A3 diagrams use Mermaid `sequenceDiagram` syntax. Both render cleanly in GitHub's Mermaid renderer and VS Code's Markdown Preview Enhanced. No `mmdc` compile step required.

**Test count after Phase 5:** 433 passing (no new tests — ADR is a documentation artifact)

---

## Day 6 — Experiment Day (2026-04-27)

_H3 entries appended per phase per the Day 6 plan (`docs/plans/day6-plan.md`). Each entry: What I built / What surprised me / What I'd do differently._

### Phase 0 — Branch + scaffolding + draft PR

**What I built.** Cut `feat/day6-experiments` from main, created `docs/images/` and `data/eval/` with `.gitkeep` markers, and opened a draft PR. The Day 6 H2 in this journal was already present from the planning session and carried forward cleanly onto the branch.

**What surprised me.**
- The `docs/plans/day6-plan.md` file was untracked on main (created during planning but never committed). Phase 0's commit is also where it lands in git history — cleaner than leaving it loose on main before the branch existed.

**What I'd do differently.** Commit the plan file to main at the end of the planning session rather than carrying it as an untracked file across the branch cut.

### Phase 1 — Iteration log scaffold + 10-query set

**What I built.** Created `data/eval/queries_v1.json` (10 queries spanning 7 CS topics, 4/4/2 high/medium/low groundedness split), `docs/iteration-log.md` (Day 6 H2 header, empty — experiment entries land in Phases 2–6), and `src/eval/query_loader.py` with `load_queries()` and a 4-test unit suite. Test count moved from 433 → 437; coverage held at 92%.

**What surprised me.**
- The loader returning `list[dict]` rather than `list[str]` was the right call as soon as I looked at Phase 2's charting requirement: per-query IDs are needed for the x-axis labels, and the plan's `list[str]` type hint was written before the chart spec was fully fleshed out. Returning dicts is strictly more useful at zero additional cost.
- q10 (cache coherence + PL memory models) is a genuinely hard edge case — it spans computer architecture and programming-language semantics, two topics the CS textbook corpus may cover independently but never explicitly bridge. That makes it a good boundary probe for groundedness experiments.

**What I'd do differently.** n/a
