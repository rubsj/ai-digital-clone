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
