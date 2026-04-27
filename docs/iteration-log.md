# Iteration Log

This file records experiment results for the P6 Digital Clone project, one entry per experiment. Each entry uses the six-field format from PRD §7g: Change, Reason, Metric Before, Metric After, Delta, Keep? Entries are newest-first within each day's H2 section. The query set driving experiments 6a/6b/6c/6e is versioned at `data/eval/queries_v1.json`.

**Query set provenance.** `queries_v1.json` was authored on 2026-04-27 from a 30-item random sample (seed=42, first 1511 CS-filtered items drawn from the `open-phi/textbooks` HuggingFace dataset). The sample is dominated by the "programming" subfield (≈98%), with a long tail of algorithms, systems, and networking textbooks. Core CS concepts — TCP, binary search, stacks/queues — appear in 9–28% of items, confirming HIGH bands; OS and DB synthesis topics (isolation levels, page replacement) appear in 1–2%, confirming MEDIUM bands; cross-cutting topics (cache coherence, buffer overflow) appear in under 1.5%, confirming LOW bands.

## Day 6 — Experiment Day (2026-04-27)
