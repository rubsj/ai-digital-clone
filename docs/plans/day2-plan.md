# Day 2 Plan: ChatStyleAgent — Style Learning Pipeline

## Context

Day 1 foundation is complete (PR #1 merged): schemas, email parser, config. Day 2 builds the style learning pipeline — extracting 15 numerical features from LKML emails, aggregating into per-leader profiles, and scoring style similarity via cosine distance. This is the core of the ChatStyleAgent: everything downstream (style-matched generation, evaluation, dual-leader comparison) depends on these profiles being distinct and self-consistent (score > 0.90).

**Pre-flight:** Pull main (PR #1 merged on GitHub, local main behind), create `feat/day2-style-pipeline` branch, verify 78 tests pass.

---

## Phase 0: Schema Change — `quote_ratio` on `EmailMessage`

The `quote_reply_ratio` feature needs the ratio of quoted lines to total lines in the **raw** body (before `_clean_body` strips them). Since `EmailMessage.body` is already cleaned, this data is lost by the time the feature extractor runs. We compute it during parsing and store it on the model.

### Changes

**`src/schemas.py`** — Add field to `EmailMessage`:
```python
class EmailMessage(BaseModel):
    # ... existing fields ...
    is_patch: bool = False
    quote_ratio: float = Field(default=0.0, ge=0.0, le=1.0)  # NEW: quoted lines / total lines (pre-cleaning)
```

**`src/style/email_parser.py`** — Compute `quote_ratio` between `_extract_body` and `_clean_body` in `parse_mbox`:
```python
body = _extract_body(msg)
if not body:
    skipped_encoding += 1
    continue

# Compute quote ratio BEFORE cleaning strips the quoted lines
quote_ratio = _compute_quote_ratio(body)

cleaned = _clean_body(body)
# ... rest of pipeline ...

results.append(
    EmailMessage(
        # ... existing fields ...
        is_patch=_detect_patch(subject, body),
        quote_ratio=quote_ratio,
    )
)
```

**New helper in `email_parser.py`:**
```python
def _compute_quote_ratio(raw_body: str) -> float:
    """Fraction of non-empty lines starting with '>' in the raw (pre-cleaned) body."""
    lines = [l for l in raw_body.splitlines() if l.strip()]
    if not lines:
        return 0.0
    quoted = sum(1 for l in lines if l.lstrip().startswith(">"))
    return quoted / len(lines)
```

**`tests/test_email_parser.py`** — Add tests:
- `test_parse_mbox_populates_quote_ratio`: mbox with quoted replies → `quote_ratio > 0`
- `test_compute_quote_ratio_no_quotes`: plain body → 0.0
- `test_compute_quote_ratio_all_quotes`: all `>` lines → 1.0
- `test_compute_quote_ratio_mixed`: 3 quoted + 7 original → 0.3

**`tests/test_schemas.py`** — Update `_make_email` helper if it exists, add:
- `test_email_message_quote_ratio_bounds`: validates ge=0.0, le=1.0

---

## Phase 1: Feature Extractor (`src/style/feature_extractor.py`)

**Stop gate:** All 15 features produce [0,1] values; `tests/test_feature_extractor.py` passes with >= 90% coverage.

### Public API

```python
def extract_features(email: EmailMessage) -> StyleFeatures:
    """Extract 15 style features from a single cleaned EmailMessage."""
```

Plus ~15 private helpers (one per feature), all `re` patterns compiled at module level (follow `email_parser.py` convention). No new dependencies — only `re`, `collections.Counter`, `numpy` from existing deps.

### Feature Specifications (maps to `to_vector()` order)

| Idx | Field | Type | Computation | Normalization |
|-----|-------|------|-------------|---------------|
| 0 | `avg_message_length` | float | `len(body.split())` | `min(word_count / 500, 1.0)` |
| 1 | `greeting_patterns` | dict | Check first non-empty line for hi/hello/hey/dear/none | Binary 0.0/1.0 per key |
| 2 | `punctuation_patterns` | dict | Keys: exclamation, question, ellipsis, dash, semicolon, colon. `count / len(body)` | `min(ratio * 50, 1.0)` |
| 3 | `capitalization_ratio` | float | ALLCAPS words (len>=2 and `word.isupper()`) / total words. Sentence-start caps don't count — this is the Torvalds discriminator (e.g. "NEVER", "WRONG", "NOT"). | Natural [0,1] |
| 4 | `question_frequency` | float | `question_sentences / total_sentences` | Natural [0,1] |
| 5 | `vocabulary_richness` | float | `len(set(words)) / len(words)` (type-token ratio) | Natural [0,1] |
| 6 | `common_phrases` | list | Bigrams + trigrams appearing >= 2x, top 20 | `to_vector`: `min(len/20, 1.0)` |
| 7 | `reasoning_patterns` | dict | Keys: because, therefore, however, but, so, if_then, the_thing_is. `count / sentences` | `min(ratio * 5, 1.0)` |
| 8 | `sentiment_distribution` | dict | Keys: positive, negative, neutral. LKML-adapted keyword lists (see below). `category_hits / total_sentiment_hits`, or all 0.0 if no hits. | Natural sums to ~1.0 |
| 9 | `formality_level` | float | Weighted mean of 5 continuous sub-signals (see below), each in [0,1]. `formality = 0.25*formal_word_rate + 0.20*(1-contraction_rate) + 0.20*avg_sent_len_norm + 0.20*(1-profanity_rate) + 0.15*(1-first_person_rate)` | Natural [0,1] — weighted mean of [0,1] signals |
| 10 | `technical_terminology` | float | ~50 kernel/systems terms. `matching_words / total_words` | `min(ratio * 20, 1.0)` |
| 11 | `code_snippet_freq` | float | Lines matching code patterns (indent, braces, includes, etc.) / total_lines | `min(ratio * 5, 1.0)` |
| 12 | `quote_reply_ratio` | float | Read `email.quote_ratio` directly (pre-computed during parsing in Phase 0, before quote stripping). No reconstruction needed. | Natural [0,1] — already a ratio |
| 13 | `patch_language` | dict | Keys: applied, nak, acked_by, reviewed_by, nack, looks_good, please_fix, resubmit. Binary presence | Natural 0.0/1.0 |
| 14 | `technical_depth` | float | Composite: tech term density + function refs `\b[a-z_]+\(\)` + file paths `\w+\.[ch]` + `CONFIG_\w+` + commit SHAs `[0-9a-f]{7,40}` | `min(weighted_sum, 1.0)` |

### Sentiment keyword lists (LKML-adapted)

Domain traps: "kill" means process termination, "NAK" is a code review term, Torvalds profanity is emphatic not necessarily negative. The neutral-technical list prevents common LKML vocabulary from inflating positive/negative counts.

**Positive** (~15 words): `"good", "great", "nice", "excellent", "works", "correct", "right", "perfect", "fine", "clean", "looks good", "applied", "proper", "well done", "reasonable"`

**Negative** (~15 words): `"broken", "wrong", "stupid", "crap", "clueless", "horrible", "buggy", "idiotic", "insane", "braindead", "garbage", "useless", "terrible", "disgusting", "nonsense"`

**Neutral-technical** (excluded from sentiment — these are LKML noise): `"patch", "commit", "merge", "config", "driver", "kernel", "module", "revert", "bisect", "rebase", "build", "test", "debug", "fix", "bug", "nak", "ack", "kill", "abort", "fatal"`

Computation: tokenize body (lowercase), count hits against positive and negative lists (skip any word also in neutral-technical). `positive_frac = pos_hits / max(pos_hits + neg_hits, 1)`, `negative_frac = neg_hits / max(pos_hits + neg_hits, 1)`, `neutral_frac = 1.0 - positive_frac - negative_frac` (clamped to 0). Multi-word phrases ("looks good") checked via substring match before tokenization.

### Formality sub-signals (weighted mean design)

Instead of fragile ±0.1 accumulator, each sub-signal is a continuous ratio in [0,1]:

| Sub-signal | Weight | Computation |
|------------|--------|-------------|
| `formal_word_rate` | 0.25 | Count of formal words (`"regarding", "furthermore", "consequently", "nevertheless", "therefore", "specifically", "respectively", "additionally", "subsequently", "hereby"`) / total words. `min(ratio * 50, 1.0)` |
| `1 - contraction_rate` | 0.20 | `contraction_count / total_words`. Contractions: `r"\b(don't|can't|won't|isn't|doesn't|it's|I'm|you're|they're|we're|that's|there's|what's|couldn't|shouldn't|wouldn't)\b"`. Invert so fewer contractions = more formal. |
| `avg_sent_len_norm` | 0.20 | `min(avg_words_per_sentence / 25, 1.0)`. Longer sentences = more formal. 25 words saturates at 1.0. |
| `1 - profanity_rate` | 0.20 | Count of profanity terms (`"crap", "stupid", "idiot", "damn", "hell", "bullshit", "shit", "ass", "clueless", "braindead"`) / total words. Invert so less profanity = more formal. `min(ratio * 100, 1.0)` before inversion. |
| `1 - first_person_rate` | 0.15 | Count of `r"\b(I|me|my|I'm|I've|I'd)\b"` / total words. Invert so less first-person = more formal. `min(ratio * 20, 1.0)` before inversion. |

`formality_level = 0.25 * formal_word_rate + 0.20 * (1 - contraction_rate) + 0.20 * avg_sent_len_norm + 0.20 * (1 - profanity_rate) + 0.15 * (1 - first_person_rate)`

This naturally spans [0, 1] since it's a weighted mean of [0,1] signals with weights summing to 1.0. Torvalds (heavy profanity, contractions, first-person) should land ~0.2-0.4. Kroah-Hartman (more formal, fewer expletives) should land ~0.5-0.7.

### Critical alignment with `to_vector()`

The `to_vector()` method in `src/schemas.py:65-93` reads fields in this exact order and collapses dicts via `mean(values)`. The feature extractor MUST:
- Populate all 5 dict fields with float values (not bools)
- Ensure dict values are individually in [0,1] since the mean inherits the range
- Use `common_phrases` as a `list[str]` — `to_vector()` converts via `min(len/20, 1.0)`

---

## Phase 2: Profile Builder (`src/style/profile_builder.py`)

**Stop gate:** Batch + incremental modes work; `tests/test_profile_builder.py` passes >= 90% coverage.

### Public API

```python
def build_profile_batch(
    leader_name: str,
    features_list: list[StyleFeatures],
    alpha: float = 0.3,
) -> StyleProfile:
    """Element-wise mean across all feature vectors. Dict keys unioned, values averaged."""

def update_profile_incremental(
    profile: StyleProfile,
    new_features: StyleFeatures,
) -> StyleProfile:
    """EMA: new_vec = (1-alpha) * current + alpha * new. Returns NEW StyleProfile."""

def save_profile(profile: StyleProfile, path: Path) -> None:
    """model_dump_json(indent=2) to file. Creates parent dirs."""

def load_profile(path: Path) -> StyleProfile:
    """model_validate_json from file."""
```

### Batch aggregation algorithm
1. Scalar fields: `statistics.mean(values)` across all emails
2. Dict fields: union all keys, mean values per key across emails that contain that key
3. `common_phrases`: flatten all lists, `Counter` → top 20 by frequency
4. `style_vector = aggregate_features.to_vector()`

### EMA incremental update
1. `new_vec = new_features.to_vector()`
2. `updated_vec = (1 - alpha) * profile.style_vector + alpha * new_vec`
3. `np.clip(updated_vec, 0.0, 1.0)`
4. Also EMA-update the `StyleFeatures` object (scalars and dicts individually) for JSON introspection
5. New keys in dict: initialize at `alpha * new_value`; absent keys: decay at `(1 - alpha) * old_value`

### Serialization
- Uses existing `StyleProfile.field_serializer/field_validator` for ndarray ↔ list roundtrip
- Save path: read from `config.leaders[name].profile_path` (set in `configs/default.yaml`). Expected default: `data/models/{leader}_profile.json`. Do NOT hardcode paths — the config is the single source of truth for file locations. `save_profile` and `load_profile` both accept a `Path` parameter; callers get the path from config.

---

## Phase 3: Style Scorer (`src/style/style_scorer.py`)

**Stop gate:** Self-similarity > 0.90 on synthetic; cross-leader < self-similarity; tests pass.

### Public API

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Returns 0.0 if either vector is zero-norm. Clamped to [0,1]."""

def score_style(profile: StyleProfile, response_features: StyleFeatures) -> float:
    """cosine_similarity(profile.style_vector, response_features.to_vector())."""
```

All features are non-negative → cosine similarity naturally in [0,1]. Only edge case: zero-norm vectors → return 0.0.

---

## Phase 4: Build Script, Radar Chart, ADR-003

**Stop gate:** End-to-end on real mbox data, self-similarity > 0.90, radar chart shows distinct profiles.

### `scripts/build_profiles.py`

1. `load_config()` → get leader paths, style settings
2. For each leader:
   - `parse_mbox(mbox_path, email_filter)` → `list[EmailMessage]`
   - Filter by `date_range` and exclude `is_patch=True`
   - `[extract_features(e) for e in emails]` with Rich progress bar
   - `build_profile_batch(name, features_list, alpha)`
   - `save_profile(profile, profile_path)` — profile path comes from `config.leaders[leader_key].profile_path`, not hardcoded. This ensures `load_profile` in later days (Flow orchestration, CLI) uses the same path.
   - **Per-feature variance diagnostics** (see below)
   - Self-similarity check: sample **20** random emails, mean `score_style` → print, warn if < 0.90
3. Cross-leader similarity: `cosine_similarity(t.style_vector, g.style_vector)` → print
4. Call radar chart function

### Per-feature variance diagnostics (new)

After building each profile, stack all per-email feature vectors into an `(N, 15)` ndarray. Compute and print a Rich table:

```
Leader: Linus Torvalds (327 emails)
┌─────────────────────┬──────┬─────────┬──────┬──────┐
│ Feature             │ Mean │ Std Dev │ Min  │ Max  │
├─────────────────────┼──────┼─────────┼────��─┼──────┤
│ avg_message_length  │ 0.31 │ 0.18    │ 0.04 │ 1.00 │
│ greeting_patterns   │ 0.12 │ 0.33    │ 0.00 │ 1.00 │
│ ...                 │ ...  │ ...     │ ...  │ ...  │
└─────────────────────┴──────┴─────────┴──────┴��─────┘
Mean self-similarity (20 samples): 0.93
```

Feature labels (15 items, matching `to_vector()` order):
`["msg_length", "greetings", "punctuation", "caps_ratio", "questions", "vocab_richness", "phrases", "reasoning", "sentiment", "formality", "tech_terms", "code_snippets", "quote_reply", "patch_lang", "tech_depth"]`

This diagnostic tells us on Day 2 whether the 0.90 target is realistic. High std dev on key discriminative features (caps_ratio, formality, punctuation) is good — it means the feature spreads across the range. Low std dev with mean near 0 or 1 means the feature is dead weight and we'd need to adjust normalization multipliers on Day 6.

### Radar Chart → `results/charts/style_radar.png`

Add `plot_style_radar(profiles, output_path)` to `src/visualization.py`:
- `matplotlib` polar projection, 15 axes
- Labels: short names for each feature dimension
- Two overlaid polygons (blue Torvalds, orange Kroah-Hartman), alpha=0.25 fill
- Legend + title, `dpi=150`

### ADR-003: Feature Vectors vs LLM Embeddings for Style

Following ADR-001 template (first-person voice, real reasoning):
- **Context:** Needed a comparable vector for writing style. Two options: 15-dim hand-crafted features vs 384/1536-dim LLM embeddings.
- **Decision:** Hand-crafted features.
- **Why:** Interpretability (radar chart axes have names), no model dependency for extraction, LKML-specific features (patch_language, code_snippet_freq), reproducibility (no model version drift), academic precedent (Schneider et al. 2016).
- **Trade-offs:** Limited to patterns we explicitly code for; adding dimensions requires code changes.

---

## Tests (~60 total)

### `tests/test_email_parser.py` + `tests/test_schemas.py` (Phase 0, ~5 new tests)
- `test_compute_quote_ratio_no_quotes`, `test_compute_quote_ratio_all_quotes`, `test_compute_quote_ratio_mixed`, `test_parse_mbox_populates_quote_ratio`
- `test_email_message_quote_ratio_bounds`

### `tests/test_feature_extractor.py` (~30 tests)
- Per-feature: at least one positive + one zero/near-zero test (15 x 2 = 30)
- Boundary: empty body, very long body (cap at 1.0)
- Integration: realistic email body → all 15 fields populated, `to_vector()` shape (15,), all in [0,1]

### `tests/test_profile_builder.py` (~15 tests)
- Batch: single email, multi-email averaging, dict key merging, phrase top-20, empty list raises, vector matches features
- EMA: vector moves by alpha, email_count increments, clips to [0,1]
- Serialization: save/load roundtrip, parent dir creation, file-not-found error

### `tests/test_style_scorer.py` (~10 tests)
- Identical vectors → 1.0, orthogonal → 0.0, zero-norm → 0.0
- Self-similarity: features → profile → score same features ≈ 1.0
- Cross-style: technical vs casual → lower score
- Always returns float in [0,1]

---

## Files Modified/Created

| File | Action |
|------|--------|
| `src/schemas.py` | Add `quote_ratio` field to `EmailMessage` (Phase 0) |
| `src/style/email_parser.py` | Add `_compute_quote_ratio`, wire into `parse_mbox` (Phase 0) |
| `src/style/feature_extractor.py` | Implement (currently stub) |
| `src/style/profile_builder.py` | Implement (currently stub) |
| `src/style/style_scorer.py` | Implement (currently stub) |
| `src/visualization.py` | Add `plot_style_radar` (currently stub) |
| `scripts/build_profiles.py` | Create new |
| `tests/test_schemas.py` | Add `test_email_message_quote_ratio_bounds` |
| `tests/test_email_parser.py` | Add `test_compute_quote_ratio_*` (4 tests) |
| `tests/test_feature_extractor.py` | Create new |
| `tests/test_profile_builder.py` | Create new |
| `tests/test_style_scorer.py` | Create new |
| `docs/adr/ADR-003-feature-vectors-vs-llm-embeddings.md` | Create new |
| `results/charts/style_radar.png` | Generated output |
| `data/models/torvalds_profile.json` | Generated output |
| `data/models/kroah_hartman_profile.json` | Generated output |

## Existing code to reuse
- `src/schemas.py` — `EmailMessage`, `StyleFeatures`, `StyleProfile` (all defined, including `to_vector()` and ndarray roundtrip)
- `src/style/email_parser.py` — `parse_mbox()` returns `list[EmailMessage]`
- `src/config.py` — `load_config()` returns `AppConfig` with leader paths and style settings
- `configs/default.yaml` — leader mbox/profile paths, `style.alpha=0.3`, date range
- `tests/test_schemas.py` — `_make_style_features()` and `_make_style_profile()` helpers

## Verification

1. `uv run pytest` — all existing 78 + new ~60 tests pass
2. `uv run pytest --cov=src/style --cov-report=term-missing` — >= 90% on new modules
3. `uv run python scripts/build_profiles.py` — prints per-feature variance table, self-similarity > 0.90 (20 samples) for both leaders, cross-leader similarity < 0.85
4. `results/charts/style_radar.png` — visually distinct polygons for the two leaders
5. `data/models/torvalds_profile.json` and `data/models/kroah_hartman_profile.json` — valid JSON, loadable

## Mitigation: If self-similarity < 0.90

1. Check normalization multipliers — if features cluster near 0 or 1, similarity inflates/deflates
2. Check that `is_patch=True` emails are excluded (patch emails look similar across leaders)
3. Verify dict fields have enough key variety (empty dicts → 0.0 → low discrimination)
4. Adjust multipliers per feature to spread the [0,1] range more evenly
5. Add feature weighting to cosine similarity (weight discriminative features higher) — last resort
