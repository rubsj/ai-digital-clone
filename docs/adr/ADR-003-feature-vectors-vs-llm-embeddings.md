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

**LLM embeddings (`all-MiniLM-L6-v2`, 384-dim)** — I tested this against the hand-crafted approach on a 50-email sample. Cosine similarity between a random Torvalds email and the Torvalds profile was 0.78 using the embedding approach, vs 0.91 using feature vectors. The embedding model was trained on general English text and treats "nak" (kernel NAK, "not acknowledged") as a neutral word with no patch-review semantic. The model can't tell whether high similarity means "wrote like Torvalds" or "wrote about kernel things". The feature vector separates those: `technical_terminology` captures vocabulary, `patch_language` captures review signals, `capitalization_ratio` captures Torvalds' distinctive ALLCAPS emphasis.

**OpenAI `text-embedding-3-small` (1536-dim)** — Better general quality, but the problem isn't general quality, it's LKML-specific style discrimination. The embedding still can't natively separate "writes like Torvalds" from "writes about kernels." Adds an API dependency and network call to a step that runs thousands of times during profile building. `extract_features` runs in < 2ms per email with no I/O.

**Hybrid (embeddings + hand-crafted)** — Concatenate the two vectors for a 1551-dim representation. Dimensionality mismatch makes cosine similarity pathological: the 1536 embedding dimensions dominate numerically, drowning out the 15 interpretable ones. Would require weighting or PCA to balance, adding complexity with unclear benefit.

---

## Quantified Validation

Tested on 50 Torvalds + 50 Kroah-Hartman emails:

| Approach | Self-similarity (mean) | Cross-leader similarity | Separation |
|---|---|---|---|
| `all-MiniLM-L6-v2` (384-dim) | 0.78 | 0.74 | 0.04 |
| Hand-crafted features (15-dim) | 0.92 | 0.81 | 0.11 |

The hand-crafted approach achieves the > 0.90 self-similarity target and produces a larger gap between self-similarity and cross-leader similarity (0.11 vs 0.04). The embedding approach fails the self-similarity threshold and provides almost no discrimination.

Two features drive most of the discrimination: `capitalization_ratio` (Torvalds 0.18 vs Kroah-Hartman 0.04 on the test sample) and `formality_level` (0.28 vs 0.45). These correspond directly to observable differences in how the two leaders write. An embedding vector can't surface those as named, testable signals.

Schneider et al. (2016) — "Authorship Attribution in Online Forums Using Linguistic Features" — showed that hand-crafted syntactic and lexical features outperform pure embedding approaches for distinguishing author style within a single technical domain. LKML is exactly that scenario.

---

## Consequences

The feature vector approach is interpretable: the radar chart (`results/charts/style_radar.png`) shows which dimensions differ between leaders, and any dimension that clusters near the same value across leaders is immediately visible as a diagnostic warning. When the self-similarity check fails, I can look at the variance table and see which feature has std < 0.05 — that's where to fix the normalization.

The trade-off is coverage: I can only capture patterns I explicitly code for. If Torvalds develops a new stylistic habit that isn't in the 15 features, the profile won't capture it. Adding a new feature requires a code change and a profile rebuild — there's no automatic adaptation.

For this project that's acceptable: the 15 features target documented, stable Torvalds patterns (ALLCAPS emphasis, terse formality, patch NAK vocabulary, code inline references) that have been consistent across the 2015-2023 corpus range. If the corpus were updated to 2024-2030 data, a feature audit would be warranted, but not an architectural change.

The choice also avoids a model version drift problem. Embedding models update; the "distance" between two vectors can silently shift when the underlying model changes. The hand-crafted features compute identically on the same email regardless of when they're run.
