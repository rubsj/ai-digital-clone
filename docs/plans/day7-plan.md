# Day 7 Plan — Streamlit + CLI + Architecture Documentation

> Plan file mirrors `docs/plans/day7-plan.md` in the ai-digital-clone repo. That file is authoritative.

## Context

Days 1–6 are merged to main at commit `90f3cb1`. The system has 437 passing tests, `src/` coverage ≥90%, 5 Day 6 experiments, and ADRs 001–007. `src/cli.py` and `streamlit_app.py` exist as placeholders. `docs/architecture/` is empty. `docs/session-notes/` was scaffolded in PR #7 and is empty except for `.gitkeep`.

Day 7 ships the user-facing surface: a Click CLI, a Streamlit app, the five Mermaid architecture diagrams (A1–A5), and the consolidated chart gallery. Both CLI and Streamlit wrap `DigitalCloneFlow` only — no direct LiteLLM / FAISS / Cohere calls from adapter code. Architecture Rules (CLAUDE.md) are locked.

Branch: `feat/day7-ui-cli-diagrams`. Session notes go to `docs/session-notes/day7.md`. Per-phase cadence: stop-gate output + session-notes block + commit SHA before proceeding. All four phases ship today.

## Verification contract

Same as Day 6. Each phase stop gate requires actual terminal output (not descriptions). Test suite stays green (`uv run pytest -q`). `src/` coverage stays ≥90%. No new dependencies. No direct commits to `main`. The 5 invariant stop gates + 4 P6 specializations in CLAUDE.md (Verification Component 2) apply.

## Pre-flight (before Phase 1, ~5 min)

```
git checkout main && git pull
git checkout -b feat/day7-ui-cli-diagrams
uv run pytest -q 2>&1 | tail -3                                      # expect: 437 passed
uv run pytest --cov=src --cov-report=term-missing -q 2>&1 | tail -5  # baseline ≥90%
ls docs/images/                                                       # expect: 8 PNGs (7 to move + 6a-run2)
ls results/                                                           # expect: style_radar.png
ls docs/architecture/                                                 # expect: empty
grep -rln "docs/images" docs/                                         # expect: only docs/adr/ADR-006-...md
```

Create `docs/session-notes/day7.md` with a header: date, branch name, baseline test count, baseline coverage, baseline main SHA `90f3cb1`. Each phase appends one block (Built / Why / Surprising / Deferred / ADR candidate) per Verification Component 6.

---

## Phase 1 — Click CLI + tests (1.5h)

**Orientation.** Replace placeholder `src/cli.py` with a Click group of 5 commands per PRD §7b. Each command wraps an existing façade. Tests use `click.testing.CliRunner` with the heavy dependencies (LLM, FAISS, profile build) mocked at the façade boundary.

**Outputs.**
- `src/cli.py` — module-runnable via `python -m src.cli`.
- `tests/test_cli.py` — new file.

**Command signatures (no bodies):**

```python
@click.group()
def cli() -> None: ...

@cli.command()
@click.option("--leader", type=click.Choice(["torvalds", "kroah-hartman"]), required=True)
@click.option("--mbox", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def learn(leader: str, mbox: Path | None, output: Path | None) -> None: ...

@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
def index(config: Path | None) -> None: ...

@cli.command()
@click.argument("query_text")
@click.option("--leader", type=click.Choice(["torvalds", "kroah-hartman"]), required=True)
def query(query_text: str, leader: str) -> None: ...

@cli.command()
@click.argument("query_text")
def compare(query_text: str) -> None: ...

@cli.command()
@click.option("--queries", type=click.Path(exists=True, path_type=Path),
              default=Path("data/eval/queries_v1.json"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/"))
def evaluate(queries: Path, output_dir: Path) -> None: ...
```

**Acceptance criteria.**
1. `python -m src.cli --help` lists all 5 commands.
2. `tests/test_cli.py` covers each command with `CliRunner.invoke`, asserting `exit_code == 0` on happy paths and at least one error path (missing required option, missing queries file). Heavy calls (`learn`, `index`, `evaluate`, `query`, `compare`) mock the façade — no real LLM/FAISS/Cohere I/O in test suite.
3. `grep -E "litellm|faiss|cohere|openai" src/cli.py` returns no hits. CLI imports only from `src.flow`, `src.agents.*`, `src.style.*`, `src.rag.{corpus_loader,chunker}`, `src.schemas`, `src.config`, `src.visualization`.
4. Full suite passes (≥437 + new CLI tests); `src/` coverage ≥90%.

**Stop gate (paste into day7.md Phase 1 block).**

```
python -m src.cli --help
uv run pytest tests/test_cli.py -q
uv run pytest -q 2>&1 | tail -3
uv run pytest --cov=src.cli --cov-report=term-missing tests/test_cli.py -q | tail -10
git add src/cli.py tests/test_cli.py docs/session-notes/day7.md
git commit -m "feat(day7): phase 1 — Click CLI + CliRunner tests"
git rev-parse HEAD
```

---

## Phase 2 — Streamlit app (2.5h)

**Orientation.** Build `streamlit_app.py` at repo root per PRD §7c and §9. Wraps `DigitalCloneFlow.kickoff` for single-leader and `compare_leaders` for dual mode. All UI elements are typed views over `src/schemas.py` (StyledResponse, FallbackResponse, LeaderComparison, EvaluationResult). No caching this phase: each rerun reloads FAISS + profiles; deferral captured in ADR-008 Consequences. After the app is functional, write ADR-008 documenting the hexagonal-adapters pattern now visible in both `src/cli.py` and `streamlit_app.py`.

**Outputs.**
- `streamlit_app.py` — replaces placeholder.
- `docs/adr/ADR-008-hexagonal-adapters.md` — written after the Streamlit app is functional, before the Phase 2 stop gate. Standard 5-section ADR format (Context / Decision / Alternatives Considered / Quantified Validation / Consequences), per Engineering Protocols and the format already used in ADR-001–007. Consequences section references the Streamlit caching deferral (to be tracked on the Post-Portfolio Followups Notion page; Ruby owns page creation, plan only notes the deferral).
- Manual smoke output (15 lines of `streamlit run` output) pasted into day7.md Phase 2 block. No test file for Streamlit.

**Section structure (named, no bodies):**

```
# section: page_config              (st.set_page_config)
# section: sidebar_visualizations    (static image grid from results/charts/)
# section: query_input               (st.text_input + st.selectbox for leader/Compare Both)
# section: dispatch                  (route to render_single or render_compare)
# section: render_single             (calls DigitalCloneFlow().kickoff)
# section: render_compare            (calls compare_leaders)
# section: render_response_card      (helper: StyledResponse fields)
# section: render_fallback_card      (helper: FallbackResponse fields)
# section: render_score_breakdown    (style / groundedness / confidence / final + 0.75 line)
# section: footer
```

**Acceptance criteria.**
1. `streamlit run streamlit_app.py` launches without exceptions. Manual smoke output (the URL line + the first page render OK) pasted into day7.md.
2. Dropdown options exactly: `Torvalds`, `Kroah-Hartman`, `Compare Both`.
3. Single-leader render shows: response text, citations list, four scores (style, groundedness, confidence, final), confidence explanation string, threshold indicator. Fallback path renders booking slot + unstyled response when `flow.state.final_output` is a `FallbackResponse`.
4. Compare mode renders two cards via `st.columns(2)` from a single `LeaderComparison`.
5. `grep -E "litellm|faiss|cohere|openai\.|\.embeddings\.create" streamlit_app.py` returns no hits.
6. `docs/adr/ADR-008-hexagonal-adapters.md` exists with all 5 sections (Context, Decision, Alternatives Considered, Quantified Validation, Consequences); Consequences names the Streamlit caching deferral.

**Stop gate.**

```
streamlit run streamlit_app.py    # let it serve, hit it once, Ctrl+C; paste first ~15 lines
grep -nE "DigitalCloneFlow|compare_leaders" streamlit_app.py
grep -E "^## (Context|Decision|Alternatives Considered|Quantified Validation|Consequences)" docs/adr/ADR-008-hexagonal-adapters.md
git add streamlit_app.py docs/adr/ADR-008-hexagonal-adapters.md docs/session-notes/day7.md
git commit -m "feat(day7): phase 2 — Streamlit app + ADR-008 hexagonal adapters"
git rev-parse HEAD
```

---

## Phase 3 — A1/A4/A5 diagrams + 5 new chart functions + gallery split (4h)

**Orientation.** Author three Mermaid markdown files (A1, A4, A5 per PRD §7f); implement the 5 missing PRD §7d chart functions in `src/visualization.py`; wire `cli evaluate` to generate them at runtime; physically split portfolio charts (`results/charts/`, 7 PNGs in PRD §7d order) from Day 6 experiment exhibits (`docs/experiments/charts/`, 6 PNGs); remove the now-empty `docs/images/` directory.

Phase 3 grew from 1.5h to 4h after Phase 1 surfaced two gaps: (a) `cli evaluate` was promising "scores + charts" per PRD §7b but deferring charts to Day 8, which has no chart scope; (b) the 7 PNGs being moved to `results/charts/` conflated PRD §7d portfolio deliverables (2 of 7) with Day 6 experiment artifacts (5 of 7). Both fixed in this phase.

**Outputs.**

*Mermaid diagrams (3 new files):*
- `docs/architecture/system-architecture.md` (A1, ```` ```mermaid\ngraph TB ````).
- `docs/architecture/data-models.md` (A4, ```` ```mermaid\nclassDiagram ```` over the 11 schemas in `src/schemas.py`).
- `docs/architecture/data-flow.md` (A5, ```` ```mermaid\ngraph LR ```` with Offline / Online subgraphs).

*5 new chart function signatures in `src/visualization.py`* (signatures only — bodies authored in this phase, but plan does not pre-solve them). Each follows the existing `plot_style_radar` contract: matplotlib Agg backend, `savefig(..., dpi=150, bbox_inches="tight")`, no display.

```python
def plot_style_distribution(eval_results: list[EvaluationResult], output_path: Path) -> None: ...
def plot_groundedness_distribution(eval_results: list[EvaluationResult], output_path: Path) -> None: ...
def plot_score_breakdown(eval_results: list[EvaluationResult], output_path: Path) -> None: ...
def plot_fallback_rate(eval_results: list[EvaluationResult], output_path: Path) -> None: ...
def plot_latency_distribution(eval_results: list[EvaluationResult], output_path: Path) -> None: ...
```

*`src/cli.py::evaluate` wiring* (~10 new lines): after the existing `EvaluationResult` aggregation, call the 5 new functions and write directly to `results/charts/02-...` through `results/charts/06-...` (overwriting on each run).

*`results/charts/` — exactly 7 PNGs in PRD §7d order:*
- `01-style-radar.png` ← `git mv results/charts/style_radar.png results/charts/01-style-radar.png` (rename in place)
- `02-style-distribution.png` (runtime, generated by `cli evaluate`)
- `03-groundedness-distribution.png` (runtime, generated by `cli evaluate`)
- `04-score-breakdown.png` (runtime, generated by `cli evaluate`)
- `05-fallback-rate.png` (runtime, generated by `cli evaluate`)
- `06-latency-distribution.png` (runtime, generated by `cli evaluate`)
- `07-style-evolution.png` ← `git mv docs/images/6d-style-evolution.png results/charts/07-style-evolution.png`

*`docs/experiments/charts/` — new directory, exactly 6 PNGs (Day 6 methodology exhibits):*
- `6a-embeddings.png` ← `git mv docs/images/6a-embeddings.png`
- `6a-embeddings-run2.png` ← `git mv docs/images/6a-embeddings-run2.png`
- `6b-chunking.png` ← `git mv docs/images/6b-chunking.png`
- `6c-weight-sensitivity.png` ← `git mv docs/images/6c-weight-sensitivity.png`
- `6e-local-vs-api.png` ← `git mv docs/images/6e-local-vs-api.png`
- `6e-run2-groundedness-agreement.png` ← `git mv docs/images/6e-run2-groundedness-agreement.png`

*`docs/images/` removed entirely* (`rmdir docs/images/` after the 7 moves — directory must be empty).

*Reference updates* (executor uses Edit tool, not sed). All `docs/images/...` references redirected:
- `docs/iteration-log.md` — references update to either `results/charts/...` (for PRD §7d charts) or `docs/experiments/charts/...` (for Day 6 exhibits) per category.
- `docs/learning-journal.md` — same redirect logic.
- `docs/plans/day6-plan.md` — same redirect logic.
- `docs/plans/day7-plan.md` — self-reference in this mapping table is already correct; verify Reuse Map evaluate row reflects new paths.
- `docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md` — `docs/images/6d-style-evolution.png` → `results/charts/07-style-evolution.png` (PRD §7d chart); any other `docs/images/...` references → `docs/experiments/charts/...`.
- `scripts/experiment_6{a,b,c,e}_*.py` — one-line comment update noting output now lives at `docs/experiments/charts/`. Do not change `savefig` paths — scripts are one-time artifacts and re-running them is not part of Day 7 scope.

**Diagram block boundaries (no contents):**
- A1: nodes for `DigitalCloneFlow`, `RAGAgent`, `StyleCrew`, `EvaluatorAgent`, `FallbackAgent` + externals `FAISS`, `Cohere Rerank`, `OpenAI Embeddings`, `LiteLLM`, `LKML mbox`, `StyleProfile JSON`.
- A4: 11 classes from `src/schemas.py` with composition arrows.
- A5: two subgraphs — Offline (mbox → parser → features → profile; corpus → chunker → embedder → FAISS) and Online (query → RAGAgent → StyleCrew → EvaluatorAgent → router → deliver | FallbackAgent).

**Acceptance criteria.**
1. Three `.md` files exist; each contains exactly one fenced ```mermaid block.
2. A4 references each of the 11 Pydantic model class names verbatim — cross-check with `grep "^class " src/schemas.py`.
3. `ls results/charts/ | wc -l` returns **7**. Filenames match `^0[1-7]-.*\.png$` in PRD §7d order (radar, style-distribution, groundedness-distribution, score-breakdown, fallback-rate, latency-distribution, style-evolution).
4. `file results/charts/*.png` reports `PNG image data` for all 7.
5. `pyproject.toml` unchanged.
6. `ls docs/experiments/charts/ | wc -l` returns **6**. All Day 6 experiment exhibit PNGs present at the new location.
7. `docs/images/` directory does not exist — verified via `ls -la docs/` (no `images/` entry).
8. `grep -rln "docs/images" .` returns no hits, OR only hits inside `scripts/experiment_*.py` as one-line redirect comments. No live `docs/images/...` references remain in markdown.
9. 5 new chart functions exist in `src/visualization.py` with the exact signatures listed above; each uses `matplotlib.use("Agg")` and `plt.savefig(..., dpi=150, bbox_inches="tight")`.
10. `cli evaluate --queries data/eval/queries_v1.json` run produces 5 fresh PNGs at `results/charts/02-style-distribution.png` through `results/charts/06-latency-distribution.png` (file mtimes within the last minute). Captured in stop gate output.
11. ADR-006 redirect applied: `grep -n "results/charts/07-style-evolution.png" docs/adr/ADR-006-...md` returns one hit; old `docs/images/6d-style-evolution.png` reference is gone.

**Stop gate.**

```
# 1) PRD §7d portfolio chart moves into results/charts/
git mv results/charts/style_radar.png results/charts/01-style-radar.png
git mv docs/images/6d-style-evolution.png results/charts/07-style-evolution.png

# 2) Day 6 experiment exhibits into docs/experiments/charts/
mkdir -p docs/experiments/charts
git mv docs/images/6a-embeddings.png            docs/experiments/charts/6a-embeddings.png
git mv docs/images/6a-embeddings-run2.png       docs/experiments/charts/6a-embeddings-run2.png
git mv docs/images/6b-chunking.png              docs/experiments/charts/6b-chunking.png
git mv docs/images/6c-weight-sensitivity.png    docs/experiments/charts/6c-weight-sensitivity.png
git mv docs/images/6e-local-vs-api.png          docs/experiments/charts/6e-local-vs-api.png
git mv docs/images/6e-run2-groundedness-agreement.png docs/experiments/charts/6e-run2-groundedness-agreement.png

# 3) Remove now-empty docs/images/
rmdir docs/images

# 4) Reference updates (executor uses Edit tool, not sed) — see Outputs section for the 6 affected files

# 5) Generate the 5 runtime charts via cli evaluate
uv run python -m src.cli evaluate --queries data/eval/queries_v1.json --output-dir results/
ls -la results/charts/02-style-distribution.png results/charts/03-groundedness-distribution.png \
       results/charts/04-score-breakdown.png    results/charts/05-fallback-rate.png \
       results/charts/06-latency-distribution.png

# 6) Verification
ls results/charts/ | wc -l                                          # expect: 7
ls docs/experiments/charts/ | wc -l                                 # expect: 6
ls -la docs/ | grep -E "^d.*images"                                 # expect: no match
file results/charts/*.png
grep -rln "docs/images" .                                           # expect: no hits, or only comment lines in scripts/experiment_*.py
grep -n "docs/images/6d-style-evolution.png" docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md  # expect: no hits
grep -n "results/charts/07-style-evolution.png" docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md  # expect: 1 hit
grep -c '```mermaid' docs/architecture/system-architecture.md docs/architecture/data-models.md docs/architecture/data-flow.md
uv run pytest -q 2>&1 | tail -3                                     # expect: still green
uv run pytest --cov=src --cov-report=term-missing -q 2>&1 | tail -5 # expect: ≥90%
git status
git add docs/architecture/ results/charts/ docs/experiments/charts/ \
        src/visualization.py src/cli.py tests/test_cli.py \
        docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md \
        docs/iteration-log.md docs/learning-journal.md docs/plans/day6-plan.md \
        scripts/ docs/session-notes/day7.md
git rm -r docs/images 2>/dev/null || true
git commit -m "feat(day7): phase 3 — A1/A4/A5 diagrams + 5 chart functions + gallery split"
git rev-parse HEAD
```

**Note on test impact.** Adding 5 new functions to `src/visualization.py` and ~10 lines to `cli evaluate` may shift coverage. Existing `tests/test_cli.py::TestEvaluateCommand` mocks `DigitalCloneFlow` and so will not produce real `EvaluationResult` data — the new chart calls inside `cli evaluate` must either be mocked at the `src.cli.plot_*` boundary in tests, or guarded so they no-op when called with mocked outputs. Executor decides at implementation time; criterion 10 verifies end-to-end runtime generation independently.

---

## Phase 4 — A2, A3 sequence diagrams (1.5h)

**Orientation.** Document the single-query and dual-leader runtime paths per PRD §7f. Pure narrative — touches no Python and no other doc.

**Outputs.**
- `docs/architecture/single-query-sequence.md` (A2, ```` ```mermaid\nsequenceDiagram ````; actors: User, DigitalCloneFlow, RAGAgent, StyleCrew, EvaluatorAgent, FallbackAgent; `@router` branch at 0.75).
- `docs/architecture/dual-leader-sequence.md` (A3, ```` ```mermaid\nsequenceDiagram ````; one RAG call, two parallel style→evaluate sequences, merge into `LeaderComparison`).

**Acceptance criteria.**
1. Both files contain exactly one `sequenceDiagram` block.
2. A2 shows the `@router` branch on the delivery threshold; the threshold's source value is linked to ADR-005 rather than inlined.
3. A3 visually shows the "retrieve once, style twice" optimization (single RAG arrow, two style branches).
4. Neither file imports/edits Python or other markdown.

**Stop gate.**

```
ls docs/architecture/
grep -c 'sequenceDiagram' docs/architecture/single-query-sequence.md docs/architecture/dual-leader-sequence.md
git add docs/architecture/single-query-sequence.md docs/architecture/dual-leader-sequence.md docs/session-notes/day7.md
git commit -m "docs(day7): phase 4 — A2/A3 sequence diagrams"
git rev-parse HEAD
```

---

## Resolved decisions

1. **Chart consolidation — SPLIT BY CATEGORY.** PRD §7d portfolio charts live in `results/charts/` (7 PNGs in PRD §7d order). Day 6 methodology exhibits live in `docs/experiments/charts/` (6 PNGs). `docs/images/` is removed entirely. All `docs/images/...` references in markdown are redirected per category. `git mv` used throughout to preserve history.
2. **`results/charts/` composition.** Exactly 7 PNGs: 2 moved from existing locations (`01-style-radar.png`, `07-style-evolution.png`), 5 generated at runtime by `cli evaluate` (`02-` through `06-`). The 5 new chart functions in `src/visualization.py` are implemented in Phase 3 with signatures matching the existing `plot_style_radar` contract (matplotlib Agg, `dpi=150`, `bbox_inches="tight"`).
3. **Streamlit testing.** No test file. Manual smoke (15 lines of `streamlit run` output) pasted into the Phase 2 session-notes block.
4. **`cli learn` — rebuild only.** Calls `build_profile_batch` + `save_profile`. No `--incremental` flag; `update_profile_incremental` is not exposed by the CLI.
5. **`cli evaluate` query source.** Defaults to `data/eval/queries_v1.json` (confirmed present). `--queries PATH` override remains.
6. **Streamlit caching — NO caching in Phase 2.** Each rerun reloads FAISS + profiles. Deferral captured in ADR-008 Consequences; Ruby owns the Post-Portfolio Followups Notion page entry.
7. **ADR-008 — WRITE.** Trigger: Phase 2 completion, after the hexagonal pattern is visible in both adapters. Topic: hexagonal architecture (ports-and-adapters) for CLI + Streamlit over `DigitalCloneFlow`. Standard 5-section format. Output path: `docs/adr/ADR-008-hexagonal-adapters.md` (per Phase 2 outputs).

---

## Reuse map (call existing symbols — no new code paths)

| CLI command | Backing function(s) |
|---|---|
| `learn` | `src/style/email_parser.py::parse_mbox` → `src/style/profile_builder.py::build_profile_batch` → `src/style/profile_builder.py::save_profile` |
| `index` | `src/rag/corpus_loader.py::load_corpus` → `src/rag/chunker.py::chunk_documents` → `src/agents/rag_agent.py::RAGAgent.build` (calls `src/rag/indexer.py::build_index` + `save_index`) |
| `query` | `src/flow.py::DigitalCloneFlow.kickoff(inputs={"query": q, "leader": L})` → read `flow.state.final_output` |
| `compare` | `src/flow.py::compare_leaders(query: str) -> LeaderComparison` |
| `evaluate` | iterate queries from JSON → `DigitalCloneFlow.kickoff` per query → aggregate `EvaluationResult` fields → write `results/evaluation_<timestamp>.json` → calls 5 chart functions in `src/visualization.py` (newly implemented in Phase 3: `plot_style_distribution`, `plot_groundedness_distribution`, `plot_score_breakdown`, `plot_fallback_rate`, `plot_latency_distribution`), writing PNGs to `results/charts/02-...06-...` |

Streamlit reuse: `src/flow.py::DigitalCloneFlow`, `src/flow.py::compare_leaders`, `src/schemas.py::{StyledResponse, FallbackResponse, LeaderComparison, EvaluationResult}`. Sidebar images: static load from `results/charts/`.

A4 cross-references `src/schemas.py` verbatim — no new model names introduced.

---

## Risks

- **Streamlit re-entrancy.** `DigitalCloneFlow` is stateful; without caching (Resolved Decision 6), each rerun instantiates fresh state — slow but correct. If re-entrancy still bites in practice (e.g. shared FAISS file lock), Phase 2 eats into Phase 4 budget — Phase 4 is the buffer.
- **Phase 3 expansion (1.5h → 4h).** Adding 5 chart functions + runtime wiring + directory split pushed Phase 3 from a doc-only phase to a coding+docs phase. The 8-hour day budget is tight. Phase 4 (A2/A3) remains required and is **not cuttable** per Ruby's decision; if Phase 3 overruns significantly, surface for replanning rather than absorbing into Phase 4.
- **Latency data gap for `plot_latency_distribution`.** `cli evaluate` does not currently capture per-query wall time — `EvaluationResult` has no latency field and the evaluate loop does not time `DigitalCloneFlow.kickoff`. Two options at Phase 3 implementation time: (a) wrap each `kickoff` call with `time.perf_counter()` and add a `latency_ms` field to each JSON record (adds ~5 lines to `cli evaluate`, no schema change since records are dict-typed in the report), or (b) stub `plot_latency_distribution` with a "no data" placeholder PNG and a TODO. Recommend (a) — it's cheap and produces a real chart. Decision deferred to executor at Phase 3 start; if (a) reveals unexpected scope, escalate per the Phase 3 overrun rule above.

---

## Critical files

- `src/cli.py` (Phase 1 done; Phase 3 adds ~10 lines to `evaluate` for chart wiring)
- `streamlit_app.py` (replaces placeholder, Phase 2)
- `src/flow.py` (read-only; do not modify)
- `src/schemas.py` (read-only; A4 source of truth)
- `src/visualization.py` (Phase 3: adds 5 new chart functions alongside existing `plot_style_radar`)
- `docs/architecture/{system-architecture,data-models,data-flow,single-query-sequence,dual-leader-sequence}.md` (new)
- `results/charts/0[1-7]-*.png` (Phase 3: 2 moved + 5 runtime-generated by `cli evaluate`)
- `docs/experiments/charts/` (new directory; Phase 3: 6 Day 6 exhibit PNGs `git mv`d from `docs/images/`)
- `tests/test_cli.py` (Phase 1 done; Phase 3 may extend if evaluate chart calls need mocks)
- `docs/adr/ADR-008-hexagonal-adapters.md` (new, Phase 2)
- `docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md` (Phase 3 line edit: `6d-style-evolution.png` reference → `results/charts/07-style-evolution.png`)
- `docs/iteration-log.md`, `docs/learning-journal.md`, `docs/plans/day6-plan.md` (Phase 3: redirect `docs/images/...` references per category)
- `scripts/experiment_6{a,b,c,e}_*.py` (Phase 3: one-line comment redirect; savefig paths unchanged)
- `docs/session-notes/day7.md` (new; appended per phase)
- `docs/PRD.md` §7b, §7c, §7d, §7f, §9 (read-only; reference, do not duplicate)

## End-to-end verification (post all phases)

```
uv run pytest -q 2>&1 | tail -3
uv run pytest --cov=src --cov-report=term-missing -q 2>&1 | tail -5
python -m src.cli --help
python -m src.cli query "What is TCP/IP?" --leader torvalds          # full pipeline, real I/O
python -m src.cli compare "What is TCP/IP?"
streamlit run streamlit_app.py                                       # manual page check
ls results/charts/ && ls docs/architecture/
git log --oneline main..HEAD                                          # one commit per phase
```
