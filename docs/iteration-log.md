# Iteration Log

This file records experiment results for the P6 Digital Clone project, one entry per experiment. Each entry uses the six-field format from PRD §7g: Change, Reason, Metric Before, Metric After, Delta, Keep? Entries are newest-first within each day's H2 section. The query set driving experiments 6a/6b/6c/6e is versioned at `data/eval/queries_v1.json`.

**Query set provenance.** `queries_v1.json` was authored on 2026-04-27 from a 30-email random sample (seed=42) drawn from the combined Torvalds + Kroah-Hartman mbox corpus (31,450 emails total). Topics and groundedness bands reflect what actually appears in the corpus — LKML patch review, stable-tree process, driver error handling, locking — not generic CS pedagogy.

## Day 6 — Experiment Day (2026-04-27)
