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

---

## Phase 2: Streamlit App + ADR-008

- **Built.** `streamlit_app.py`:1-203 — 10-section Streamlit app (page_config, sidebar_visualizations, query_input, render_score_breakdown, render_fallback_card, render_response_card, render_single, render_compare, dispatch, footer). `docs/adr/ADR-008-hexagonal-adapters.md` — 5-section ADR documenting the hexagonal ports-and-adapters pattern visible across both CLI and Streamlit adapters.
- **Why.** PRD §7c requires a Streamlit UI with query input, leader dropdown (Torvalds / Kroah-Hartman / Compare Both), score breakdown, and fallback display. Architecture Rule requires no direct LiteLLM/FAISS/Cohere imports from adapter code — both adapters speak only to `DigitalCloneFlow.kickoff` and `compare_leaders`.
- **Surprising.** Helper functions (render_response_card, render_fallback_card, render_score_breakdown) must be defined before the dispatch block that calls them — Streamlit re-runs the entire script top-to-bottom on each interaction, so Python's definition-before-use rule applies normally. The plan's section ordering (dispatch listed before render helpers) would have caused NameErrors at runtime. Fixed by defining all render_ helpers before the dispatch block, preserving the section comment names.
- **Deferred.** `st.cache_resource` caching of `DigitalCloneFlow` deferred — every Streamlit rerun reinitializes FAISS + profiles. Latency penalty only, not correctness. Captured in ADR-008 Consequences. Ruby owns the Post-Portfolio Followups Notion page entry.
- **ADR candidate.** ADR-008 written: hexagonal architecture (ports-and-adapters) for CLI + Streamlit over `DigitalCloneFlow`. Decision: both adapters import only from `src/flow.py` façade + narrow style/RAG façades for `learn`/`index` commands. Verified by grep. Three alternatives considered: direct internal imports (rejected — breaks test isolation), shared adapter base class (rejected — premature abstraction given different output primitives), session_state caching (deferred per above).

### Phase 2 stop gate output

```
$ grep -nE "DigitalCloneFlow|compare_leaders" streamlit_app.py
3:All heavy I/O routes through DigitalCloneFlow (single-leader) or
4:compare_leaders (dual-leader). No direct LiteLLM / FAISS / Cohere imports.
14:from src.flow import DigitalCloneFlow
15:from src.flow import compare_leaders as _compare_leaders
182:            result = _compare_leaders(query_text.strip())
187:            flow = DigitalCloneFlow()

$ grep -E "^## (Context|Decision|Alternatives Considered|Quantified Validation|Consequences)" docs/adr/ADR-008-hexagonal-adapters.md
## Context
## Decision
## Alternatives Considered
## Quantified Validation
## Consequences

$ uv run streamlit run streamlit_app.py --server.headless true
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.0.0.132:8501
  External URL: http://73.143.108.98:8501

  [No exceptions — clean startup, page renders OK]
```

---

### What it does **not** evaluate

- **Factual accuracy** vs. ground truth — there's no expected-answer field; groundedness measures evidence-overlap, not correctness.
- **Latency** — captured in Phase 3 via `time.perf_counter()` wrapping `flow.kickoff`; `latency_ms` added to both fallback and scored records.
- **Human style judgment** — `style_score` is vector cosine, not a human rating.
- **Query-set difficulty calibration** — `expected_groundedness_band` in `queries_v1.json` is metadata only; nothing currently checks predicted vs. expected band.

---

## Phase 3: A1/A4/A5 diagrams + 5 chart functions + gallery split

- **Built.**
  - `docs/architecture/system-architecture.md` (A1, `graph TB`) — nodes for `DigitalCloneFlow`, `RAGAgent`, `StyleCrew`, `EvaluatorAgent`, `FallbackAgent` + all 5 externals.
  - `docs/architecture/data-models.md` (A4, `classDiagram`) — all 11 `src/schemas.py` Pydantic models with composition arrows.
  - `docs/architecture/data-flow.md` (A5, `graph LR`) — Offline (mbox→profile, corpus→FAISS) and Online (query→RAGAgent→StyleCrew→EvaluatorAgent→router→deliver|fallback) subgraphs.
  - 5 new chart functions in `src/visualization.py`: `plot_style_distribution`, `plot_groundedness_distribution`, `plot_score_breakdown`, `plot_fallback_rate`, `plot_latency_distribution` — all taking `list[dict]`, using matplotlib Agg, `dpi=150, bbox_inches="tight"`.
  - `src/cli.py::evaluate` wired: `time.perf_counter()` wraps each `flow.kickoff`; `latency_ms` added to both fallback and scored `records.append` dicts; 5 chart calls + `charts_dir.mkdir` appended after JSON write.
  - Gallery split: `results/charts/` renamed `style_radar.png` → `01-style-radar.png`; `docs/images/6d-style-evolution.png` → `results/charts/07-style-evolution.png`; 6 Day 6 exhibits → `docs/experiments/charts/`; `docs/images/` removed.
  - `results/charts/` gitignore line removed so charts can be committed.
  - Reference updates: ADR-006, `docs/iteration-log.md`, `docs/learning-journal.md`, `docs/plans/day6-plan.md`, `CLAUDE.md`, all 6 experiment scripts (one-line comment redirect per script; savefig paths untouched).
  - `tests/test_visualization.py` — 16 new tests covering all 6 chart functions (edge cases: empty records, all-fallback, no latency field).
  - `tests/test_cli.py` — evaluate tests updated to mock 5 chart functions and assert `latency_ms` in records.

- **Why.** PRD §7b requires `cli evaluate` to produce "scores + charts"; no chart scope existed in Day 8. PRD §7d specifies exactly 7 charts in `results/charts/` (radar, 5 evaluation charts, style evolution). The plan grew from 1.5h to 4h when Phase 1 surfaced that chart generation was silently deferred and that `docs/images/` conflated PRD deliverables with Day 6 methodology exhibits.

- **Surprising.** `results/charts/` was in `.gitignore` as "Generated artifacts — reproducible from source." The 2 manually moved PNGs (`01-style-radar`, `07-style-evolution`) could not be `git mv`d from their untracked locations. Removed the gitignore line to allow the full chart set to be committed — portfolio reviewers should see the charts without running `cli evaluate`. The latency question from Phase 2 ("no per-query wall time") resolved trivially: `time.perf_counter()` around `flow.kickoff` at `src/cli.py:264`, ~2 lines, no schema change.

- **Deferred.** Nothing new. `cli evaluate` docstring still references Phase 3 chart generation as complete; no follow-on items.

- **ADR candidate.** No new decision. The gallery split and gitignore removal are straightforward consequences of PRD §7d ownership. Latency capture is implementation detail, not an architectural decision.

---

## Phase 4: A2/A3 sequence diagrams

- **Built.**
  - `docs/architecture/single-query-sequence.md` (A2, `sequenceDiagram`) — actors: User, DigitalCloneFlow, RAGAgent, StyleCrew, EvaluatorAgent, FallbackAgent. `@router` branch on `final_score >= 0.75`; threshold sourced to ADR-005 via `Note over DCF` rather than inlined.
  - `docs/architecture/dual-leader-sequence.md` (A3, `sequenceDiagram`) — single `retrieve(query)` call to RAGAgent; shared chunks passed to both Torvalds and Kroah-Hartman style+evaluate branches via Mermaid `par`/`and` block; merge into `LeaderComparison`.

- **Why.** PRD §7f requires all 5 architecture diagrams (A1–A5). A2 and A3 document the runtime call sequence that is not visible in the static component (A1) or data-flow (A5) diagrams — specifically the `@router` fallback branch and the retrieve-once optimization in compare mode.

- **Surprising.** Nothing — pure documentation phase. The "retrieve once, style twice" structure was already locked in `src/flow.py`; A3 is a faithful transcription, not a design decision.

- **Deferred.** Nothing. All 5 PRD §7f diagrams now exist (`system-architecture.md`, `data-models.md`, `data-flow.md`, `single-query-sequence.md`, `dual-leader-sequence.md`).

- **ADR candidate.** No new decision surfaced.
