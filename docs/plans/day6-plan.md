# Day 6 Plan — Experiment Day

> Execution will be performed on a `feat/day6-experiments` branch — never direct to main, PR-only.

---

## Context

Days 1–5 are merged to main. The repo contains a working dual-leader pipeline:

- **RAG** — `RAGAgent.retrieve(query)` (FAISS top-20 → Cohere rerank → top-5). Provider hardcoded to OpenAI; MiniLM exists in `src/rag/embedder.py` but is not currently the default index path.
- **Style + Evaluator + Fallback** — full stack from Day 4, weighted formula `0.4 × style + 0.4 × groundedness + 0.2 × confidence`, threshold 0.75.
- **Flow** — `DigitalCloneFlow` with `@router` branching, `compare_leaders(query)` wrapper.
- **Tests** — 433 passing on main, ≥90% coverage on `src/`.

Day 6 is **measurement, not construction**. No new agents, no flow changes, no schema changes. Five experiments produce iteration-log entries and chart PNGs on top of the Day 5 system. ADR-006 is conditional on what 6c or 6e finds.

**Hard exit criteria** (PRD §8 Day 6 row, verbatim):
> All experiments run. Iteration log has ≥ 3 entries. Style evolution chart shows measurable shift. ADR-006 written if local vs API experiment produces interesting data.

This plan targets **5 entries** (one per experiment), exceeding the ≥3 floor.

**Carry-forward conventions** (do not deviate without an ADR):
- `instructor.from_litellm(litellm.completion)` for any structured output.
- Pydantic schemas live in `src/schemas.py` and **do not change today**.
- Façade pattern from `src/agents/*_steps.py` — no new agents.
- ADR-006 conditional. ADR format: 5 sections per CLAUDE.md.
- Branch naming: `feat/day6-experiments`. One branch for the day. PR to main.

---

## Verification contract (applies to every phase)

Each phase has a **stop gate**. Before moving on, paste into the chat:

1. The exact pytest invocation used and its full terminal output (pass/fail counts visible).
2. For experiment scripts: the actual stdout of the script, not a paraphrased summary.
3. For files created: file path + line count (`wc -l <path>`).
4. For each iteration-log entry: the rendered markdown for that entry, copied verbatim from the file.
5. For each chart: the saved PNG path and a one-line description of what it shows.

Self-reported "I did it" claims are not acceptable. If a gate fails, stop and surface the failure — do not silently proceed.

### Per-phase cadence (Phases 1–8): journal + commit before the stop gate

Every phase from 1 onward ends with **two sub-steps before** the stop-gate paste-back:

**a. Append a learning-journal entry** to `docs/learning-journal.md` under the Day 6 H2 (created in Phase 0). Each entry is an H3 of the form `### Phase N — <title>` and contains exactly three subsections:

- **What I built** — 1–2 sentences. Concrete: what artifact landed.
- **What surprised me** — 1–3 bullets covering non-obvious findings, debugging dead ends, or framework gotchas. If genuinely nothing surprised, write the literal sentence: `nothing surprised — straightforward execution`. Do not pad.
- **What I'd do differently** — one sentence, or `n/a`.

This is **reflection**, not a duplicate of the iteration-log. The iteration log captures experiment data per PRD §7g (Change / Reason / Metric Before / Metric After / Delta / Keep?). The journal captures Ruby's voice — the reasoning, the misses, the surprise. Do not copy numbers from the iteration log into the journal.

**b. Commit the phase's deliverables.** Conventional-commit message: `feat(day6): phase N — <one-line summary>` (use `docs(day6):` for Phases that ship only documentation, `chore(day6):` for housekeeping). Push the commit to the remote branch.

**Stop-gate output additions for every phase 1–8:**
- Commit SHA (`git log --oneline -1`).
- Full rendered learning-journal H3 entry, copied verbatim from the file.

### Cohere quota pre-flight (applies to Phases 2, 3, 4, 6 — every query-driven script)

Silent reranker degradation mid-experiment poisons the comparison: a run where 3 of 10 queries fall back to raw FAISS scores while the other 7 use Cohere is not a clean A/B. To prevent this, every query-driven script (`experiment_6a_*.py`, `experiment_6b_*.py`, `experiment_6c_*.py`, `experiment_6e_*.py`) does the following in its preamble before entering the 10-query loop:

1. **Test rerank call.** Issue one synthetic rerank request (e.g., `rerank(query="health check", documents=["foo", "bar"], top_n=1)`).
2. **Check for degraded mode.** If the response signals rate-limit-near-exhaustion (HTTP 429, or a near-limit header), or `src/rag/reranker.py`'s degraded-mode flag/log fires, **abort with a precise error message** identifying the cause. Do **not** proceed to the 10-query loop. The script exits non-zero so the executor cannot accidentally treat a degraded run as a successful experiment.
3. **Log the success.** On a clean check, print `cohere quota pre-check OK` to stdout — this string appears in the phase's stop-gate paste, providing audit evidence that the run was not silently degraded.

The pre-check itself consumes one rerank call per script (4 total across the day), well within the ~80-call envelope and the 1K/month free tier.

---

## Pre-flight notes the executor must read before Phase 1

**Iteration log file does not exist yet.** The repo has no `docs/iteration-log.md`. The PRD §7g spec is the contract:

> Each entry must include: Change, Reason, Metric Before, Metric After, Delta, Keep?

Phase 1 establishes the file using that exact set of fields. Do not invent a different schema. Do not split entries across multiple files. One markdown file, one entry per experiment, newest entry at the top under a `## Day 6 — Experiment Day (2026-04-27)` header. Use ATX-style H3 (`### 6a — Embedding comparison`) for each entry. Within an entry, render the six fields as a small markdown table or a labeled list — the executor picks one and uses it consistently across all 5 entries.

**Charts directory.** Save PNGs to `docs/images/` (the user-specified location for this day). The directory does not exist yet — Phase 1 creates it. The existing `results/charts/` directory is left untouched for Day 7's PRD-mandated 7-chart set.

**The 10-query set is shared across 6a/6b/6c/6e.** It is sourced once in Phase 1, persisted to a versioned artifact, and reused unchanged in 6a, 6b, 6c, and 6e. Phase 1 establishes the canonical source. The variation across phases is in **what is being measured**, not in what queries drive the measurement.

**Source for the 10 queries.** Pre-search performed before this plan was written:

- P6 repo (`/Users/rubyjha/repo/AI/ai-digital-clone`) — searched `tests/`, `data/`, `scripts/`, `docs/` for `*quer*` / `*eval*` files. **No reusable query set found.** The hits (`tests/test_evaluator.py`, `src/agents/evaluator_steps.py`, `src/evaluation/evaluator.py`) are evaluation code, not query corpora.
- P5 repo (`/Users/rubyjha/repo/AI/ai-rag-evaluation-framework`) — `data/output/qa_pairs.json` exists with 56 QA pairs, but they are over the **Armadale Capital Plc financial-report corpus**, not the `open-phi/textbooks` CS corpus P6 uses. **Domain mismatch — not reusable.** Forcing them through P6's RAG would produce uniformly low groundedness and tell us nothing about the configurations being compared.

**Conclusion: query-set creation is required and is Phase 1's first sub-step.** Persist the 10 queries to `data/eval/queries_v2.json` (versioned filename) and freeze the file for the rest of the day. Do not modify it between phases — reproducibility depends on a stable query set.

**Selection criteria for the 10 queries** (the executor authors them against these constraints, not freehand):
1. **Domain coverage.** Spread across the major CS/OS topics present in the indexed corpus — networking, operating systems, algorithms, data structures, databases, security. At least 6 distinct topics across the 10 queries; no more than 2 queries on the same narrow subtopic.
2. **Difficulty distribution.** Roughly 4 "easy" (single-fact, expected groundedness > 0.7 against the corpus), 4 "medium" (synthesis across 2–3 chunks, expected groundedness 0.4–0.7), 2 "edge" (cross-topic or partially-out-of-corpus, expected groundedness < 0.4). The mix matters: 6c needs queries that straddle the 0.75 final-score threshold so weight changes can move them; an all-easy set will pin everything above the threshold and produce a flat sensitivity chart.
3. **Expected groundedness range.** Documented per-query in the JSON itself as a `"expected_groundedness_band": "high" | "medium" | "low"` field. Not a ground truth — a hypothesis the experiments can compare against. This makes Phase 2's groundedness numbers interpretable rather than just numbers in a vacuum.
4. **Style-agnostic phrasing.** Queries are not phrased to favor one leader's voice over the other (no "ranty" or "diplomatic" framing). Style is downstream of retrieval; the query set's job is exercising RAG, not style.
5. **No proper-noun trivia.** Avoid queries answerable only by remembering a specific named entity (e.g., "What year was POSIX standardized?"). Those reward memorization, not retrieval+grounding. Prefer concept-explanation queries.

JSON schema for `data/eval/queries_v2.json` (executor uses exactly this):
```json
[
  {"id": "q01", "query": "...", "topic": "networking", "expected_groundedness_band": "high"},
  ...
]
```
This shape gives Phases 2/3/4/6 a canonical ID to key off when reporting per-query results in their charts and iteration-log entries.

**Reproducibility floor (applies to all experiments).**
- Any RNG-using helper accepts an explicit `seed` argument; `random.Random(seed)` per Day 4 pattern, never module-level `random.seed()`.
- All knobs (embedding provider, chunking strategy, weights, model name) live in source — config dicts at the top of the experiment script, not hidden behind env-var fallbacks.
- Outputs are versioned: markdown entry + chart PNG committed in the same commit as the experiment script that produced them.

**Tests on new helpers.** Experiments themselves do not need unit tests. But any new module under `src/` (a query loader, chart utility, Ollama client wrapper) does, and `src/` coverage must remain ≥ 90%. Experiment scripts live in `scripts/` and are exempt — they are reproducible artifacts, not library code.

**Stop-gate sequencing.** Each experiment is its own phase with its own stop gate. Do not run all five and write all five entries at the end. Run one, write its entry, save its chart, paste the stop-gate output, then move on. Batching defeats the verification contract.

---

## Phase 0 — Branch + scaffolding + draft PR (≤15 min)

**Orientation.** Cut `feat/day6-experiments` from current main. Verify the 433-test baseline still passes (Day 5 end-of-day count from CLAUDE.md "Current State"). Create `docs/images/` and `data/eval/` directories with `.gitkeep` markers so they survive the empty-directory git invariant. Add the `## Day 6 — Experiment Day (2026-04-27)` H2 header at the bottom of `docs/learning-journal.md` (the journal is oldest-first; Day 5 currently sits at the bottom). Commit, push, open a draft PR against main.

**Acceptance criteria.**
- Branch `feat/day6-experiments` exists and is checked out.
- `pytest tests/ -q` reports **exactly 433 passing**. Any deviation is a regression and must be diagnosed before Phase 1.
- `docs/images/.gitkeep` and `data/eval/.gitkeep` committed.
- `docs/learning-journal.md` has a new `## Day 6 — Experiment Day (2026-04-27)` H2 appended at the bottom (no H3 entries underneath yet — those land in Phases 1–8).
- First commit on the branch lands with message like `chore(day6): scaffold branch — gitkeep, journal H2`.
- Branch pushed; draft PR opened against `main`.

**Stop gate output to paste.**
- `git branch --show-current`
- `git status` (should be clean after the commit).
- Full pytest summary line including the count (`433 passed in Xs`).
- Commit SHA (`git log --oneline -1`).
- Draft PR URL.

---

## Phase 1 — Iteration log scaffold + 10-query set

**Orientation.** Two artifacts, both versioned, both prerequisites for every later phase.

1. **Query set (`data/eval/queries_v2.json`).** Author 10 queries spanning the textbook corpus topics. Spread them across difficulty: a few that should retrieve obviously well (general OS / networking concepts present in the corpus), a few mid-difficulty (specific algorithms, named protocols), a couple that probe edges (cross-topic reasoning, slightly out-of-corpus). The point is that 6a/6b need queries where embedding choice can plausibly differ; 6c needs queries where final-score will straddle the 0.75 threshold so weight changes move the deliver/fallback boundary; 6e needs queries that exercise the evaluator's full output range. One query set must serve all four use cases.
2. **Iteration log (`docs/iteration-log.md`).** Create with a top-of-file H1, a brief one-paragraph "what this file is" preamble, and the `## Day 6 — Experiment Day (2026-04-27)` H2 header. Empty under the H2 — Phases 2–6 each append one H3 entry.

**A small loader module is allowed if and only if it is reused by ≥ 2 experiment phases.** If only one phase needs to read the queries, inline the JSON load. Otherwise create `src/eval/query_loader.py` with `load_queries(path) -> list[str]` and a unit test (`tests/test_query_loader.py`). Same rule applies to a chart utility (`src/eval/charts.py`) — only create it when the second chart-producing phase forces the abstraction. Avoid speculative abstractions.

**Acceptance criteria.**
- `data/eval/queries_v2.json` exists, contains exactly 10 strings, valid JSON.
- `docs/iteration-log.md` exists with the structure described above and no per-experiment entries yet.
- If a loader module was created: `pytest tests/test_query_loader.py -v` passes; `src/` coverage still ≥ 90%.

**Stop gate output to paste.**
- `cat data/eval/queries_v2.json` (full content — 10 queries are short).
- `cat docs/iteration-log.md` (full content — should fit on a screen).
- Coverage report if a new `src/` module was added.
- (Per-phase cadence) Commit SHA and the rendered Phase 1 learning-journal H3 entry.

**Human-review gate (in addition to standard paste-back, not a replacement).**

> **STOP. Do not proceed to Phase 2.** Wait for Ruby to review the query set against the 5 selection criteria (domain coverage, difficulty distribution, expected groundedness range, style-agnostic phrasing, no proper-noun trivia) and explicitly approve in chat with `queries approved, proceed to Phase 2` or equivalent. If Ruby requests changes, revise `queries_v2.json`, re-run any helper-test commands, and re-paste the full file before re-requesting approval.

---

## Phase 2 — Experiment 6a: Embedding comparison (OpenAI vs MiniLM)

**Goal.** Run the 10 queries through two RAG configurations that differ **only** in the embedding model. Same chunking (500/50), same Cohere reranker, same retriever, same evaluator, same scoring weights. Measure groundedness, end-to-end final score, and retrieval latency.

**Inputs.** `data/eval/queries_v2.json`, the existing OpenAI FAISS index, and a freshly built MiniLM index (build it once at the top of the script if not already on disk; cache to `data/rag/faiss_index_minilm/`). The MiniLM index build is not a deliverable on its own — it is a prerequisite for this experiment. Reuse `embed_chunks(provider="minilm")` per the Day 3 implementation note in the learning journal.

**Outputs.**
- `scripts/experiment_6a_embeddings.py` (deterministic, all knobs in-source).
- `docs/images/6a-embeddings.png` — a chart with two side-by-side panels: (left) per-query final-score for each config, (right) groundedness mean ± stdev for each config. Style consistent with existing P6 charts (matplotlib, dpi=150, Agg backend).
- One iteration-log entry under Day 6, H3 `### 6a — Embedding comparison: OpenAI vs MiniLM`, with the six PRD §7g fields populated:
  - **Change:** swap OpenAI text-embedding-3-small for all-MiniLM-L6-v2.
  - **Reason:** verify P2's 26% Recall@5 lift on this corpus; ground ADR-002's claim in P6 data.
  - **Metric Before:** OpenAI mean groundedness, mean final, mean retrieval latency (3 numbers).
  - **Metric After:** MiniLM equivalents.
  - **Delta:** absolute and percent change for each metric.
  - **Keep?** decision sentence — likely "keep OpenAI" per P2 evidence, but the data decides.

**Stop gate.**
- `python scripts/experiment_6a_embeddings.py` runs to completion and prints the metrics table. Stdout must include the `cohere quota pre-check OK` line (per the shared pre-flight rule).
- `docs/images/6a-embeddings.png` exists and is non-empty.
- Iteration-log entry rendered (paste the entry's markdown into the gate).
- `pytest tests/ -q` still green at the post-Phase-1 baseline; no new tests expected in this phase.
- (Per-phase cadence) Commit SHA and the rendered Phase 2 learning-journal H3 entry.

---

## Phase 3 — Experiment 6b: Chunking comparison (fixed vs semantic)

**Goal.** Same 10 queries through two configurations that differ **only** in chunking. Same OpenAI embeddings, same Cohere reranker, same downstream pipeline. Measure groundedness and chunk-relevance.

**Chunk-relevance metric.** Define it explicitly in the script's docstring before measuring it. Suggested definition: mean of the top-5 reranker relevance scores per query, averaged across all 10 queries. Whatever the executor chooses, the definition is committed to source so Phase 6 (handover note) can reference it without ambiguity.

**Inputs.** `data/eval/queries_v2.json`, two FAISS indices: the existing baseline (500/50 fixed) and a fresh semantic-split index built via `chunk_semantic()` (Day 3, `src/rag/chunker.py`). Build the semantic index once and cache it to `data/rag/faiss_index_semantic/`.

**Outputs.**
- `scripts/experiment_6b_chunking.py`.
- `docs/images/6b-chunking.png` — paired bar or grouped bar of groundedness + chunk-relevance for each config across 10 queries.
- Iteration-log entry `### 6b — Chunking comparison: fixed 500/50 vs semantic markdown`, six fields per Phase 1 spec.

**Stop gate.**
- Script stdout pasted. Must include the `cohere quota pre-check OK` line (per the shared pre-flight rule).
- PNG path verified.
- Iteration-log entry rendered.
- `pytest tests/ -q` still green at the post-Phase-1 baseline; no new tests expected in this phase.
- (Per-phase cadence) Commit SHA and the rendered Phase 3 learning-journal H3 entry.

---

## Phase 4 — Experiment 6c: Scoring weight sensitivity

**Goal.** Hold everything else fixed and sweep the three weight configurations from PRD §6c across the 10 queries. Find the optimal config (or document that the system is insensitive in this region).

| Config | Style | Groundedness | Confidence |
|--------|-------|--------------|------------|
| Default | 0.4 | 0.4 | 0.2 |
| Style-heavy | 0.5 | 0.3 | 0.2 |
| Ground-heavy | 0.3 | 0.5 | 0.2 |

**Constraint.** PRD §3b D10 + CLAUDE.md Stop Gate #4 forbid changing the formula weights outside Day 6 experiments — this phase is the explicit exemption. Do not modify `src/evaluation/evaluator.py` itself; instead, parameterize the weights in the experiment script and recompute `final` from the raw component scores. The script must NOT mutate the production `evaluate()` function.

**Optimal-config criterion.** State up front, before running: "optimal" = the config whose mean final score on the 10 queries is highest **AND** whose deliver/fallback boundary behavior is closest to the PRD's 30–40% fallback rate target (PRD §2d). If those two criteria disagree, report both numbers and let the data — not the script — pick. The criterion lives in the script docstring so the choice is reproducible.

**Outputs.**
- `scripts/experiment_6c_weight_sensitivity.py`.
- `docs/images/6c-weight-sensitivity.png` — sensitivity plot. Suggested form: x-axis = the 10 queries, y-axis = final score, three lines (one per config), with a horizontal 0.75 threshold line marked. Per-config fallback rates printed in the legend or a small subplot.
- Iteration-log entry `### 6c — Scoring weight sensitivity (3 configs × 10 queries)`, six fields. **Metric Before** = default config numbers; **Metric After** = the chosen optimal config; **Delta** = changes; **Keep?** = decision on whether to update the production weights.

**ADR-006 trigger check (note, not action).** If the chosen optimal config differs materially from default (delta in mean final > 0.05, OR fallback rate moves outside the 30–40% band in a way default does not), this is a candidate trigger for ADR-006. **Do not write the ADR yet** — Phase 6e may also trigger it, and one ADR can cover both. The Phase 7 decision point consolidates.

**Stop gate.**
- Script stdout pasted (must include the `cohere quota pre-check OK` line, per-config mean final, and per-config fallback rate).
- PNG path verified.
- Iteration-log entry rendered.
- `pytest tests/ -q` still green at the post-Phase-1 baseline; no new tests expected in this phase.
- (Per-phase cadence) Commit SHA and the rendered Phase 4 learning-journal H3 entry.

---

## Phase 5 — Experiment 6d: Pre/post-2018 Torvalds style evolution

**Goal.** Partition Torvalds emails by `2018-09-01`. Recompute `StyleFeatures` on each partition independently. Report deltas on sentiment, capitalization, exclamations (use the relevant `punctuation_patterns` field), and formality. Visualize as a time-series with a vertical marker at 2018-09.

**Inputs.** The existing parsed Torvalds mbox (already validated on Day 1, ≥ 200 emails). Use `email.timestamp` to partition. If the corpus is too thin in one partition (< 30 emails one side), fall back to bucketing by year and document the alternative in the iteration-log entry's **Reason** field.

**Significance criterion (decide before measuring, not after).** A per-feature delta is reported as a **measurable shift** only if `|pre_mean - post_mean| > 2 × std(feature)` where `std` is computed on the **larger** of the two partitions (the more-data side gives the lower-variance estimate of the noise floor). Anything below that threshold is reported as `within noise` in the chart legend and the iteration-log **Reason** field. Do not promote a sub-noise delta to a "shift" in prose.

The PRD §8 exit criterion ("style evolution chart shows measurable shift") is satisfied if **at least one** of the four tracked features (sentiment, capitalization, exclamations via `punctuation_patterns`, formality) clears this threshold. If **none** clear it, the iteration-log **Keep?** field reads `n/a — no measurable shift detected at the 2σ threshold`, and the chart caption states the same. Do not force a conclusion that the data does not support — a null result is a valid finding and goes into the handover honestly.

**Constraint.** This experiment touches **only** style feature aggregation. Do not modify `feature_extractor.py` or `profile_builder.py`. Use them as-is, with the partition filter applied in the experiment script before passing emails into `extract_features` / `build_profile_batch`.

**Outputs.**
- `scripts/experiment_6d_style_evolution.py`.
- `docs/images/6d-style-evolution.png` — time-series. X-axis = email timestamp (or year-bucketed), Y-axis = the four chosen features (one line per feature, OR four small multiples — executor picks the cleaner one). Vertical dashed line at 2018-09-01 with a small label.
- Iteration-log entry `### 6d — Pre/post-2018 Torvalds style evolution`. Six fields. **Metric Before** = pre-2018 feature values; **Metric After** = post-2018; **Delta** = absolute changes, with each row marked `(measurable shift)` or `(within noise)` per the significance criterion above; **Keep?** = `n/a — diagnostic, not a config change` if at least one feature cleared the threshold, or `n/a — no measurable shift detected at the 2σ threshold` if none did.

**Stop gate.**
- Script stdout pasted (must include partition counts, per-feature pre/post means, and the std/threshold computation per feature).
- PNG path verified.
- Iteration-log entry rendered.
- `pytest tests/ -q` still green at the post-Phase-1 baseline; no new tests expected in this phase.
- (Per-phase cadence) Commit SHA and the rendered Phase 5 learning-journal H3 entry.

---

## Phase 6 — Experiment 6e: GPT-4o-mini vs local Ollama for evaluation scoring

**Goal.** Replace **only the evaluator's explanation-LLM call** (the single Instructor call inside `src/evaluation/evaluator.py`) with a local Ollama model (`qwen3:8b`). Same 10 queries, same prompts. Measure (a) Pearson correlation between the two configs' final scores, (b) latency per evaluation, (c) any qualitative drift in the explanation strings.

**Framing.** This experiment must produce a **decision**, not just numbers. The script's final printout is one of three outcomes:
1. **Quality parity** (Pearson ≥ 0.90 on final scores AND explanation strings are recognizably equivalent on a 5-sample manual spot check) → recommend Ollama for dev, GPT-4o-mini for prod.
2. **Quality drift** (Pearson < 0.90 OR spot-check shows hallucinated reasoning from Ollama) → recommend GPT-4o-mini for both, document the gap.
3. **Latency-quality tradeoff** (parity on quality, but Ollama is meaningfully slower or faster) → state the tradeoff and recommend per-environment.

**Ollama prerequisite (hard, not optional).** Ollama is installed on Ruby's machine. The default path for this phase is **the full experiment, not a skip.**

**Shell pre-flight — runs BEFORE the script is launched.** This is where novel misconfiguration (wrong base URL, wrong provider prefix, model not pulled) is discovered. The executor runs these in the shell and pastes the output into the stop gate:

1. **Daemon reachable:** `curl -s http://localhost:11434/api/tags` — must return JSON listing installed models. If empty/refused, start the daemon (`ollama serve` in a separate shell) and retry.
2. **Target model present:** verify `qwen3:8b` is in `ollama list`. If not present, run `ollama pull qwen3:8b` (~5GB, one-time) before continuing. **Do NOT substitute another model** — Ruby has committed to `qwen3:8b` for this experiment (chosen as the same parameter-count weight class as GPT-4o-mini for a fair API-vs-local comparison), and substitution introduces an artifact-traceability cost that is not worth the saved download time. Hardware (M5 Max, 128GB unified) handles the 8b without issue.
3. **End-to-end completion smoke test:** a one-liner that exercises the exact wiring the script will use:
   ```bash
   python -c "import litellm; r = litellm.completion(model='ollama/qwen3:8b', messages=[{'role':'user','content':'Reply with the single word: ready'}]); print(r.choices[0].message.content)"
   ```
   Must print a non-empty string. If this fails — wrong provider prefix, base URL, JSON parse — fix it here, **not** inside the experiment script.

**Script preamble — hard assertions only, not novel discovery.** The script's own startup checks the daemon (curl) and the model (`ollama list`) and aborts with a precise message ("ollama daemon not reachable at localhost:11434" / "model X not in `ollama list`") if either fails. By the time the script runs, all three pre-flight checks have already passed in the shell — the script-level assertions are the seatbelt, not the airbag.

**Skip path — fallback only, not the default.** If the daemon crashes mid-run or the model genuinely cannot be pulled (network, disk space), write a partial iteration-log entry: **Metric After** = "n/a — Ollama failed mid-run, see <error>"; **Keep?** = "deferred." This path exists so the day's other artifacts still ship even on infrastructure failure. It is **not** the planned outcome, and Phase 7 (ADR-006 trigger) treats a successful 6e as the expected case.

**Constraint.** Do not modify `src/evaluation/evaluator.py`. Parameterize the LLM client at the script level: the script calls the same scoring helpers but with a swapped Instructor client wired to Ollama (via `litellm`'s Ollama provider). If a thin wrapper is genuinely needed (e.g., an `_ollama_instructor_client()` helper used only by this script), put it in the script file, not in `src/`. A new `src/` module is only justified if it is reused — and nothing else uses Ollama in P6.

**Outputs.**
- `scripts/experiment_6e_local_vs_api.py`.
- `docs/images/6e-local-vs-api.png` — scatter plot of GPT-4o-mini final score vs Ollama final score across the 10 queries, with the y=x reference line and Pearson correlation in the title. If Ollama was skipped, write a placeholder PNG with the skip reason as text — keeps the artifact set complete.
- Iteration-log entry `### 6e — GPT-4o-mini vs local Ollama (qwen3:8b) for evaluation`, six fields, ending with the chosen recommendation per the framing above.

**Stop gate.**
- `ollama list` output pasted, showing `qwen3:8b` present. This output appears **before** the smoke-test commands and **before** the script run.
- Pre-flight outputs pasted: `curl http://localhost:11434/api/tags` JSON, the python `litellm.completion(...)` smoke-test stdout.
- Script stdout pasted. Must include the `cohere quota pre-check OK` line (per the shared pre-flight rule) AND the explicit decision sentence ("recommend X in dev, Y in prod" or "deferred — Ollama failed mid-run").
- PNG path verified (or placeholder PNG with skip reason).
- Iteration-log entry rendered.
- `pytest tests/ -q` still green at the post-Phase-1 baseline; no new tests expected in this phase.
- (Per-phase cadence) Commit SHA and the rendered Phase 6 learning-journal H3 entry.

---

## Phase 7 — ADR-006 decision point (CONDITIONAL)

**Decision rule.** Write `docs/adr/ADR-006-local-vs-api-llm-evaluation.md` if **any** of the following holds, taken from the data already produced in Phases 4 and 6:

- **6e produced an actionable decision** (parity → dev/prod split, or drift → stay on API). The recommendation itself is a non-obvious finding worth an ADR.
- **6c found an optimal config that differs materially from default** (delta in mean final > 0.05, OR fallback rate shift moves it from outside the 30–40% band into the band, or vice versa). Changing the production weights is a real architectural change and warrants an ADR.
- **Both 6c and 6e produced findings → STOP.** Do not write the ADR(s) yet. Paste a 3–5 line summary of each finding into the chat stop-gate, plus a recommendation: `one ADR covering both` OR `two ADRs (ADR-006 for finding X, ADR-007 for finding Y)`. The framing depends on whether the findings are related (both about evaluator behavior under perturbation) or independent (6c is trust-calibration, 6e is model-parity). Wait for Ruby's decision before writing. Once decided, write per the ADR-005 structure.

If none of those conditions hold (e.g., 6e was skipped due to Ollama unavailability AND 6c showed default is already optimal), **do not write an ADR**. Note in the handover that ADR-006 was considered and skipped, with the data that justified the skip. The PRD §7e explicitly marks ADR-006 as "Written only if Day 6 experiment produces interesting data" — skipping is a valid outcome, not a failure.

**If writing the ADR.**
- Match ADR-005's H2 section structure exactly: Context, Decision, Alternatives Considered, Quantified Validation, Consequences.
- Quantified Validation pulls numbers verbatim from the iteration-log entries — no fresh measurement.
- Java/TS parallel as one parenthetical at end of Consequences only.
- Open ADR-005 side by side while writing.

**Acceptance criteria.**
- A written decision (paste into the gate): either "ADR-006 written because <triggered condition>" with the file path, or "ADR-006 skipped because <data points>" with the iteration-log lines that justify the skip.

**Stop gate output to paste.**
- The decision sentence above.
- If written: `wc -l docs/adr/ADR-006-local-vs-api-llm-evaluation.md` and a quick H2 section list to confirm format match.
- (Per-phase cadence) Commit SHA and the rendered Phase 7 learning-journal H3 entry. The journal entry covers the decision-making (why an ADR was warranted or skipped), not a duplicate of the ADR's own content.

---

## Phase 8 — CLAUDE.md update + handover note + PR description + ready-for-review

**Orientation.** No new code. Three deliverables: CLAUDE.md "Current State" updated, Notion-ready handover note rendered to chat, GitHub PR description rendered to chat. After Ruby reviews the paste-back, mark the draft PR ready for review.

**CLAUDE.md updates.**
- "Last Updated" → 2026-04-27.
- "Current Day" → "Day 6 complete".
- "Tests" → new total. State the delta vs the 433 baseline explicitly: `433 → <new_total> (+N_new)`.
- Day 6 checklist items in the day-by-day section flipped from `[ ]` to `[x]`.
- "What's Done" extended with the five experiments + iteration log + (optional) ADR-006.
- "Key Decisions Made (Day 6)" section appended in the same shape as the Day 4 / Day 5 entries — bullet list of decisions whose rationale is non-obvious from the code or commits.

**Day 6 handover note (Notion-ready).** Same shape as the Day 5 handover. The Day 5 handover is **not** in the local repo — it lives in Notion.

**Notion access — explicit, not implied.** Fetch the Day 5 handover note from Notion using ID `34edb630640a81e0adb9c7855055b0ff` (URL: https://www.notion.so/34edb630640a81e0adb9c7855055b0ff). Use its structure as the template for the Day 6 handover. If the Notion MCP is not connected in the current session, ask Ruby to either (a) connect Notion MCP, or (b) paste the Day 5 handover content into chat. **Do NOT silently fall back to a different structure.**

The handover note lives **outside the repo** (it is pasted into Notion). The phase deliverable is the rendered markdown text in the chat, ready for Ruby to copy into Notion.

**PR description (third deliverable).** Render to chat, ready for Ruby to drop into the GitHub PR body. Required shape:

- **Summary** — one paragraph: what Day 6 produced (5 experiments, iteration-log entries, charts, conditional ADR), and the headline finding (the most interesting single number across the day).
- **Phase table** — one row per Phase 0–8, with columns: phase number, one-line summary, commit SHA. The SHAs come from each phase's per-phase cadence stop-gate output.
- **Artifacts shipped** — bulleted list: 5 experiment scripts (paths), 5 PNGs (paths), 5 iteration-log entries (link to `docs/iteration-log.md`), conditional ADR-006 (path or "skipped — see Phase 7 decision").
- **Test count delta** — `433 → <new_total> (+N)`. Both numbers visible. If `N == 0` (no new tests, only scripts and docs), say so explicitly.
- **Review focus** — point the reviewer at the 2–3 highest-judgment files. These are the experiment scripts where decisions were made (most likely `experiment_6c_weight_sensitivity.py` for the optimal-config criterion, `experiment_6e_local_vs_api.py` for the parity/drift recommendation, and the iteration-log entries themselves). Explicitly **do not** route the reviewer to chart-rendering boilerplate or matplotlib styling — that is mechanical.

**Mark PR ready for review.** After Ruby has seen the paste-back and signed off in chat, run `gh pr ready <PR_number>` (or via the GitHub UI) to lift the draft status. This step happens **after** Ruby's review, not during the executor's stop-gate paste.

**Acceptance criteria.**
- CLAUDE.md "Current State" reflects Day 6 complete with the new test count.
- Handover note rendered and pasted into the gate.
- PR description rendered and pasted into the gate.
- `pytest tests/ -v` final count matches CLAUDE.md.
- Draft PR marked ready for review (after Ruby's sign-off).

**Stop gate output to paste.**
- Diff of the CLAUDE.md "Current State" block (before → after).
- Full rendered handover-note markdown.
- Full rendered PR description markdown.
- Final pytest summary line.
- (Per-phase cadence) Commit SHA and the rendered Phase 8 learning-journal H3 entry.
- After Ruby's sign-off: `PR marked ready for review` confirmation line, with the `gh pr ready` command output or the PR URL showing the draft badge removed.

---

## Files touched

| Path | Action | Phase |
|------|--------|-------|
| `data/eval/queries_v2.json` | Create | 1 |
| `docs/iteration-log.md` | Create + append entries | 1, 2, 3, 4, 5, 6 |
| `docs/images/6a-embeddings.png` | Create | 2 |
| `docs/images/6b-chunking.png` | Create | 3 |
| `docs/images/6c-weight-sensitivity.png` | Create | 4 |
| `docs/images/6d-style-evolution.png` | Create | 5 |
| `docs/images/6e-local-vs-api.png` | Create | 6 |
| `scripts/experiment_6a_embeddings.py` | Create | 2 |
| `scripts/experiment_6b_chunking.py` | Create | 3 |
| `scripts/experiment_6c_weight_sensitivity.py` | Create | 4 |
| `scripts/experiment_6d_style_evolution.py` | Create | 5 |
| `scripts/experiment_6e_local_vs_api.py` | Create | 6 |
| `data/rag/faiss_index_minilm/` | Build (cached after first run) | 2 |
| `data/rag/faiss_index_semantic/` | Build (cached after first run) | 3 |
| `src/eval/query_loader.py` | Create **only if reused ≥ 2 phases** | 1 |
| `tests/test_query_loader.py` | Create iff loader created | 1 |
| `docs/adr/ADR-006-local-vs-api-llm-evaluation.md` | **Conditional** — only if Phase 7 trigger fires | 7 |
| `CLAUDE.md` | Update "Current State" | 8 |
| `src/schemas.py` | **No change** | — |
| `src/flow.py` | **No change** | — |
| `src/evaluation/evaluator.py` | **No change** — weights parameterized at script level only | — |

---

## Reuse map (do not duplicate)

- `src/rag/embedder.py::embed_chunks(provider="minilm")` — for the 6a MiniLM index build.
- `src/rag/chunker.py::chunk_semantic` — for the 6b semantic index build.
- `src/rag/indexer.py::build_index, save_index, load_index` — index lifecycle.
- `src/agents/rag_agent.py::RAGAgent` — full retrieval pipeline. Instantiate twice in 6a/6b with different index paths.
- `src/agents/evaluator_steps.py::EvaluatorAgent.evaluate` — call as-is in 6a/6b/6e. **Do not modify** for 6c — parameterize weights in the script.
- `src/style/feature_extractor.py::extract_features` + `src/style/profile_builder.py::build_profile_batch` — for 6d, with a partition filter applied before the call.
- `src/style/email_parser.py::parse_mbox` — already produces `EmailMessage` with `.timestamp` for 6d's partition.
- Chart styling pattern: see `src/visualization.py::plot_style_radar` for the matplotlib conventions (Agg backend, dpi=150, color choices) the new charts must match.

---

## End-to-end verification (after all phases land)

A reviewer should be able to run, in order:

1. `pytest tests/ -v` → all green; count ≥ Day 5 baseline of 433; src/ coverage still ≥ 90%.
2. Open `docs/iteration-log.md` → 5 H3 entries under the Day 6 H2, each with the six PRD §7g fields populated.
3. Open `docs/images/` → 5 PNGs (6a, 6b, 6c, 6d, 6e), all rendering.
4. Open `docs/adr/ADR-006-...md` if it exists → 5 H2 sections matching ADR-005's structure; Quantified Validation numbers trace to specific iteration-log entries.
5. CLAUDE.md "Current State" reflects Day 6 complete with the new test count.
6. The handover note is pasted into Notion (outside-repo deliverable, not verifiable from the repo alone — the PR description should link to the Notion page).

---

## Risks


- **MiniLM index build cost (Phase 2).** Building a fresh MiniLM index across the full ~900-chunk corpus is a one-time ~2–3 minute cost (CPU-only sentence-transformers). Acceptable, but the executor should run it before the first measurement so it is not on the experiment's hot path.
- **Cohere reranker quota.** Each query-driven phase pre-checks quota at script startup and aborts cleanly if degraded mode would activate. ~80 calls across the day; free-tier 1K/month is comfortable, but the pre-check prevents a partially-degraded run from producing meaningless comparison data.
- **6c boundary stability.** With only 10 queries, the per-config fallback rate has high variance (one query flipping = 10 percentage points). The "30–40% target" should be read as "in the right neighborhood," not a hard threshold. The optimal-config criterion in Phase 4 explicitly handles this by reporting both numbers when they disagree.
- **6d partition imbalance.** If Torvalds' parsed mbox is heavily weighted toward one period (e.g., mostly post-2018), the time-series chart will look noisy on the smaller side. The fallback to year-bucketing in Phase 5 mitigates, but the "measurable shift" exit criterion (PRD §8) could still fail. If pre-2018 has < 30 emails, document that as a partition-balance limitation in the iteration-log **Reason** field rather than forcing a conclusion.
- **6e Ollama mid-run failure.** Ollama is installed and expected to run successfully (Phase 6 hard prereq). The remaining risk is mid-run failure — daemon crash, OOM on the 8b model under load (unlikely on 128GB unified, but possible if other processes are competing), network hiccup if the model needs a re-pull. The skip path catches this and writes a partial entry, but the day's headline ADR-006 trigger then likely shifts to 6c. Mitigation: smoke-test the Ollama call before the 10-query loop (Phase 6 step 3) — most failures surface there, before 9 queries' worth of work is lost.

