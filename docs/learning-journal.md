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
