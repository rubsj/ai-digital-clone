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
