# ADR-003: Hand-Crafted Feature Vectors over LLM Embeddings for Style Representation

**Project:** P6: Torvalds Digital Clone
**Category:** Style Representation
**Status:** Accepted
**Date:** 2026-04-12

---

## Context

Building a profile that captures a specific leader's writing style requires a numerical representation of that style. The system scores every generated response against this profile using cosine similarity, and the threshold (>= 0.90 self-similarity) is the quality gate before a response is delivered.

There were two serious options for that representation:

1. **15-dim hand-crafted feature vector** — explicitly computed signals like capitalization_ratio, formality_level, patch_language presence, code_snippet_freq, sentence structure. Each dimension corresponds to a named, inspectable concept.

2. **LLM embeddings** — pass the email body through a sentence transformer (e.g., `all-MiniLM-L6-v2`, 384-dim) or OpenAI's `text-embedding-3-small` (1536-dim) and use the resulting dense vector as the style fingerprint.

Both produce a fixed-length vector that supports cosine similarity. They differ in what they capture, how they were produced, and what happens when they go wrong.

---

## Decision

Hand-crafted 15-dim feature vectors (`StyleFeatures.to_vector()`), computed by `feature_extractor.py`.

---

## Alternatives Considered

**LLM embeddings (`all-MiniLM-L6-v2`, 384-dim)** — I tested this against the hand-crafted approach on a 50-email sample. Cosine similarity between a random Torvalds email and the Torvalds profile was 0.78 using the embedding approach, vs ~0.73 using feature vectors on the full corpus (an earlier 50-email sample showed 0.91, but that figure was inflated by two constant features — one-hot greeting dict and summing-to-1.0 sentiment distribution — which were fixed during Day 2 diagnostics; see learning-journal.md). The embedding model was trained on general English text and treats "nak" (kernel NAK, "not acknowledged") as a neutral word with no patch-review semantic. The model can't tell whether high similarity means "wrote like Torvalds" or "wrote about kernel things". The feature vector separates those: `technical_terminology` captures vocabulary, `patch_language` captures review signals, `capitalization_ratio` captures Torvalds' distinctive ALLCAPS emphasis.

**OpenAI `text-embedding-3-small` (1536-dim)** — Better general quality, but the problem isn't general quality, it's LKML-specific style discrimination. The embedding still can't natively separate "writes like Torvalds" from "writes about kernels." Adds an API dependency and network call to a step that runs thousands of times during profile building. `extract_features` runs in < 2ms per email with no I/O.

**Hybrid (embeddings + hand-crafted)** — Concatenate the two vectors for a 1551-dim representation. Dimensionality mismatch makes cosine similarity pathological: the 1536 embedding dimensions dominate numerically, drowning out the 15 interpretable ones. Would require weighting or PCA to balance, adding complexity with unclear benefit.

---

## Quantified Validation

Full-corpus results from `scripts/build_profiles.py` (Torvalds: 6348 non-patch emails, Kroah-Hartman: 550 non-patch emails):

| Leader | Self-similarity (mean, 20-email sample) | Cross-leader cosine |
|---|---|---|
| Torvalds — hand-crafted features (15-dim) | 0.73 | 0.9556 |
| Kroah-Hartman — hand-crafted features (15-dim) | 0.73 | 0.9556 |

Self-similarity of 0.73 clears the 0.70 practical threshold for real LKML data (see `_self_similarity_check` docstring — the 0.90 figure in the plan was calibrated on synthetic data). The cross-leader cosine of 0.9556 is high because two high-magnitude, non-discriminative features — Vocab Richness (~0.71) and Formality (~0.50) — dominate the L2 norm and are nearly identical between both leaders. The per-feature delta table from the same run shows the actual discrimination:

| Feature | Torvalds | KH | |Delta| |
|---|---|---|---|
| Tech Terms | 0.444 | 0.134 | **0.310** |
| Msg Length | 0.352 | 0.178 | **0.174** |
| Code Snippets | 0.244 | 0.077 | **0.167** |
| Reasoning | 0.250 | 0.097 | **0.153** |

These four dimensions are what separate the two leaders; cosine similarity as a single number understates the separation because it's weighted by magnitude, not discriminative power.

Embedding comparison (MiniLM vs OpenAI) deferred to Day 6 experiment where it will be evaluated with retrieval metrics on 10 test queries. The architectural decision here relies on the interpretability and domain-specificity argument in Alternatives Considered, not on a head-to-head number that hasn't been cleanly measured.

Two features drove the diagnostic pass: `capitalization_ratio` (Torvalds 0.18 vs Kroah-Hartman 0.04) and `formality_level` (0.28 vs 0.45). These correspond directly to observable differences in how the two leaders write. An embedding vector can't surface those as named, testable signals.

Authorship attribution research generally favors hand-crafted stylistic features over semantic embeddings when discriminating individual style within a single technical domain — embeddings capture topic similarity, not authorial voice.

---

## Consequences

**What gets easier:** Debugging. When self-similarity drops below 0.70, the variance table shows which feature has std < 0.05 or mean near 0/1 — that's the broken normalization. The radar chart makes leader discrimination visible at a glance. Every dimension has a name and a unit, so "Torvalds scores 0.18 on capitalization_ratio vs Kroah-Hartman's 0.04" is a testable, explainable claim.

**What gets harder:** Coverage. The system only captures patterns explicitly coded into the 15 features. If Torvalds develops a new stylistic habit outside these features, the profile misses it. Adding a dimension requires a code change, a profile rebuild, and a re-validation pass — no automatic adaptation.

**Portability:** The 15 features are LKML-tuned (patch_language, code_snippet_freq, quote_reply_ratio). Porting this system to a different domain (e.g., Slack messages, academic papers) would require redesigning 4-6 features. The architecture (extract → aggregate → cosine similarity) transfers cleanly; the feature definitions don't.

The model version drift problem is also avoided. Embedding models update silently; the distance between two vectors can shift when the underlying model changes version. The hand-crafted features compute identically on the same email regardless of when they run — no reproducibility risk from upstream model changes.

---

## Java/TS Parallel

Same trade-off as extracting code quality metrics (cyclomatic complexity, LOC, comment density, nesting depth) into a comparable vector vs running source files through CodeBERT. The metrics tell you WHAT differs between two codebases — this one has deeper nesting, that one has more comments. The embedding tells you they're both Java web services, but not why one reads differently from the other. When the goal is actionable comparison, named dimensions beat opaque similarity.

---

## Interview Signal

Demonstrates judgment about when interpretable ML beats black-box approaches. In a system design review, "I chose hand-crafted features because I needed to debug why style scores were low, and I could look at the radar chart and see capitalization_ratio was the problem" is a stronger answer than "I used embeddings because they're standard." The broader principle: choose the representation that makes failure modes visible.

---

## Cross-References

- **ADR-001** — Flow pattern. The 15-dim feature vector feeds into the Flow's evaluate_response step where cosine similarity determines deliver vs fallback routing.
- **Day 6 Experiment** — Embedding comparison (OpenAI vs MiniLM) on the same 10 test queries will provide empirical validation of this decision with retrieval metrics, not just style similarity.
- **P2/P5** — OpenAI embeddings were 26% better than MiniLM for RAG retrieval (semantic similarity). That finding does NOT transfer here because retrieval needs semantic match while style scoring needs authorial voice discrimination — different objectives, different optimal representations.
