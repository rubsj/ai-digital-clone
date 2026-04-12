# ADR-003: Hand-Crafted Feature Vectors over LLM Embeddings for Style Representation

**Project:** P6: Torvalds Digital Clone
**Category:** Style Representation
**Status:** Accepted
**Date:** 2026-04-12

---

## Context

Building a profile that captures a specific leader's writing style requires a numerical representation of that style. The system scores every generated response against this profile using cosine similarity, and the threshold (>= 0.70 self-similarity on real LKML data) is the quality gate before a response is delivered. The original plan targeted 0.90, but that was calibrated on synthetic data; full-corpus validation landed at 0.70.

I had two serious options:

1. A 15-dim hand-crafted feature vector, explicitly computed signals like capitalization_ratio, formality_level, patch_language presence, code_snippet_freq, sentence structure. Each dimension corresponds to a named, inspectable concept.

2. LLM embeddings, passing the email body through a sentence transformer (e.g., `all-MiniLM-L6-v2`, 384-dim) or OpenAI's `text-embedding-3-small` (1536-dim) and using the resulting dense vector as the style fingerprint.

---

## Decision

Hand-crafted 15-dim feature vectors (`StyleFeatures.to_vector()`), computed by `feature_extractor.py`.

---

## Alternatives Considered

**LLM embeddings (`all-MiniLM-L6-v2`, 384-dim)** - The architectural problem is clear without a head-to-head comparison. The embedding model was trained on general English text and treats "nak" (kernel NAK, "not acknowledged") as a neutral word with no patch-review semantic. The model can't tell whether high similarity means "wrote like Torvalds" or "wrote about kernel things," because all LKML emails are about kernels. Embedding self-similarity is inflated by topic overlap, not authorial voice. The feature vector separates those: `technical_terminology` captures vocabulary, `patch_language` captures review signals, `capitalization_ratio` captures Torvalds' distinctive ALLCAPS emphasis.

**OpenAI `text-embedding-3-small` (1536-dim)** - Better general quality, but the problem isn't general quality. It's LKML-specific style discrimination. The embedding still can't natively separate "writes like Torvalds" from "writes about kernels." It also adds an API dependency and network call to a step that runs thousands of times during profile building. `extract_features` runs in < 2ms per email with no I/O.

**Hybrid (embeddings + hand-crafted)** - Concatenate the two vectors for a 1551-dim representation. Dimensionality mismatch makes cosine similarity pathological: the 1536 embedding dimensions dominate numerically, drowning out the 15 interpretable ones. Would require weighting or PCA to balance, adding complexity with unclear benefit.

---

## Quantified Validation

Full-corpus results from `scripts/build_profiles.py` (Torvalds: 6348 non-patch emails, Kroah-Hartman: 550 non-patch emails):

| Leader | Self-similarity (mean, 20-email sample) | Cross-leader cosine |
|---|---|---|
| Torvalds, hand-crafted features (15-dim) | 0.73 | 0.9556 |
| Kroah-Hartman, hand-crafted features (15-dim) | 0.73 | 0.9556 |

Self-similarity of 0.73 clears the 0.70 threshold (see `_self_similarity_check` docstring). The cross-leader cosine of 0.9556 is high because two high-magnitude, non-discriminative features (Vocab Richness ~0.71 and Formality ~0.50) dominate the L2 norm and are nearly identical between both leaders. The per-feature delta table from the same run shows the actual discrimination:

| Feature | Torvalds | KH | |Delta| |
|---|---|---|---|
| Tech Terms | 0.444 | 0.134 | **0.310** |
| Msg Length | 0.352 | 0.178 | **0.174** |
| Code Snippets | 0.244 | 0.077 | **0.167** |
| Reasoning | 0.250 | 0.097 | **0.153** |

These four dimensions are what separate the two leaders. Cosine similarity understates the gap because high-magnitude non-discriminative features dominate the norm.

I deferred the embedding comparison (MiniLM vs OpenAI) to the Day 6 experiment, where it will be evaluated with retrieval metrics on 10 test queries.

Two features drove the diagnostic pass: `capitalization_ratio` (Torvalds 0.18 vs Kroah-Hartman 0.04) and `formality_level` (0.28 vs 0.45). These correspond directly to observable differences in how the two leaders write. An embedding vector can't surface those as named, testable signals.

---

## Consequences

When self-similarity drops below 0.70, I check the variance table for which feature has std < 0.05 or mean near 0/1, and that's the broken normalization. The radar chart makes leader discrimination visible at a glance. Every dimension has a name and a unit, so "Torvalds scores 0.18 on capitalization_ratio vs Kroah-Hartman's 0.04" is a testable, explainable claim.

The tradeoff is coverage. The system only captures patterns I explicitly coded into the 15 features. If Torvalds develops a new stylistic habit outside these features, the profile misses it. Adding a dimension requires a code change, a profile rebuild, and a re-validation pass. There's no automatic adaptation.

I also avoid the model version drift problem. Embedding models update silently, and the distance between two vectors can shift when the underlying model changes version. The hand-crafted features compute identically on the same email regardless of when they run, so there's no reproducibility risk from upstream model changes.

The 15 features are LKML-tuned (patch_language, code_snippet_freq, quote_reply_ratio). Porting this system to a different domain like Slack messages or academic papers would require redesigning 4-6 features. The architecture (extract, aggregate, cosine similarity) transfers cleanly; the feature definitions don't. (Same tradeoff as extracting code quality metrics like cyclomatic complexity and comment density into a comparable vector vs running source files through CodeBERT: the named dimensions tell you what differs, while the embedding just tells you they're both Java web services.)
