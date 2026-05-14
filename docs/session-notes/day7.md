# Day 7 Session Notes

- **Date:** 2026-05-12
- **Branch:** feat/day7-ui-cli-diagrams
- **Baseline tests:** 437 passed
- **Baseline coverage:** 91% (src/)
- **Baseline main SHA:** 90f3cb1

---

## Phase 1: Click CLI + tests

- **Built.** `src/cli.py`:1-178 — Click group with 5 commands (`learn`, `index`, `query`, `compare`, `evaluate`). `tests/test_cli.py`:1-195 — 11 CliRunner tests covering happy paths and error paths for all 5 commands.
- **Why.** All commands wrap existing facades (`DigitalCloneFlow.kickoff`, `compare_leaders`, `RAGAgent.build`, `parse_mbox`+`build_profile_batch`, `load_corpus`+`chunk_documents`) per Architecture Rules — no direct LiteLLM/FAISS/Cohere imports. `learn` uses rebuild-only path (`build_profile_batch`) per Resolved Decision 4; `--incremental` not exposed.
- **Surprising.** `build_profile_batch` takes `(leader_name, features_list)` — not raw EmailMessage objects. The `learn` command needed an intermediate `extract_features(e) for e in emails` step that wasn't spelled out in the reuse map. Also `chunk_documents` takes `(docs, config)` not `(docs, chunk_size, chunk_overlap)` — the signature differs from what the reuse map implied, confirmed by reading the source.
- **Deferred (revised).** Originally noted `cli evaluate` chart generation as "scoped to Day 8 per plan." That deferral was wrong: PRD §7b explicitly requires `evaluate` to produce "scores + charts," PRD §7d enumerates 7 portfolio charts, and Day 8 has no chart scope. Triggered a Phase 3 plan expansion (1.5h → 4h): 5 missing chart functions added to `src/visualization.py`, `cli evaluate` wired to call them, `results/charts/` reorganized to PRD §7d order, Day 6 experiment exhibits moved to a separate `docs/experiments/charts/` directory. Uncovered line 178 (`if __name__ == "__main__": cli()`) is not exercisable via CliRunner; acceptable.
- **Follow-up edit (folded into Phase 1).** Expanded all 6 docstrings in `src/cli.py` (group + 5 commands) so `--help` output documents the typical workflow (`learn` → `index` → `query`/`compare`/`evaluate`), per-command "When to use" guidance, prerequisites, and outputs. Used Click's `\b` no-rewrap markers to preserve numbered list / bullet formatting in `--help`. No behavior change; all 11 tests still pass; amended into the Phase 1 commit rather than landing as a separate commit.
- **ADR candidate.** No new decision surfaced. The facade-wrapping pattern is the existing Architecture Rule 1 (CrewAI Flow as orchestrator) applied to adapters — captured in ADR-008 in Phase 2.

---

## Reference: what `cli evaluate` actually evaluates

Captured during Phase 1 review so Phase 3 chart implementations have a precise contract to plot against. Source: `src/evaluation/evaluator.py:75`.

For every (query × leader) pair in the JSON query set, `cli evaluate` produces an `EvaluationResult` containing **four scores**.

### The 3 component scores

| Score | What it measures | How | Target |
|---|---|---|---|
| `style_score` | Does the response *sound like* this leader? | Cosine similarity between the response's 15-dim feature vector (msg length, greetings, punctuation, caps, hedge frequency, vocab richness, reasoning markers, sentiment, formality, tech terms, code snippets, quote-reply, patch language, tech depth, phrase diversity) and the leader's `StyleProfile` vector. `src/style/style_scorer.py:29` | > 0.90 |
| `groundedness_score` | Is the response actually supported by retrieved evidence? | Splits response into sentences, embeds each, computes max cosine-sim against each retrieved chunk's embedding, then averages those per-sentence maxima. `src/evaluation/groundedness_scorer.py:1-11` | > 0.60 |
| `confidence_score` | Is the system confident in this response? | Heuristic blend of three equally-weighted sub-signals: (1) mean reranker score across top-5 retrieved chunks, (2) fraction of query keywords appearing in response, (3) hedge penalty (1 − min(1, hedge_count / 5)). No LLM call. `src/evaluation/confidence_scorer.py:1-14` | > 0.80 |

### The combined verdict

```
final_score = 0.4 × style + 0.4 × groundedness + 0.2 × confidence
decision    = "deliver" if final_score >= 0.75 else "fallback"
```

Weights and threshold are locked Architecture Rules (ADR-005). One LLM call per evaluation produces a one-sentence `explanation` of the decision. If `decision == "fallback"`, the user gets a calendar booking link instead of the response.

### What the JSON report captures per (query × leader)

The 4 scores + a `fallback` boolean. No raw response text, no per-sentence groundedness breakdown — just the aggregated numbers, which is what feeds the 5 PRD §7d charts Phase 3 wires in.

### What it does **not** evaluate

- **Factual accuracy** vs. ground truth — there's no expected-answer field; groundedness measures evidence-overlap, not correctness.
- **Latency** — the JSON does not currently capture per-query wall time. Phase 3 implication: `plot_latency_distribution` either requires adding timing wraps to the evaluate loop, or must be stubbed with a note. **Open question for Phase 3 — surface for replanning if it adds scope.**
- **Human style judgment** — `style_score` is vector cosine, not a human rating.
- **Query-set difficulty calibration** — `expected_groundedness_band` in `queries_v1.json` is metadata only; nothing currently checks predicted vs. expected band.
