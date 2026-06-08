# Day 15 — Session Notes

> Append per phase. This is the opening entry (Phase P1 — Archive results/). No source code changed this phase.

## Phase P1 — Archive results/

### Built

- Classified all 10 non-chart files in `results/` top level into three buckets: current-truth, audit-trail, worthless. The 7 chart PNGs in `results/charts/` were left in place; they are P3a scope.
- Confirmed the 7 pre-classified files from the day15 plan citation map against disk — no mismatches.
- Proposed buckets for the 3 plan-flagged candidates: `evaluation_day12.json` (audit-trail, live doc citations), `evaluation_day12_reeval.json` (audit-trail, session-note citations), `evaluation_20260523_121048.json` (worthless, pre-scoring-schema snapshot).
- Created `results/archive/` and archived 6 files via `git mv` (renames, not delete+add):
  - Audit-trail: `bakeoff_w1b0_day14.json`, `bakeoff_hhem_probe_day14.json`, `evaluation_day12_reeval2.json`, `evaluation_day12.json`, `evaluation_day12_reeval.json`
  - Worthless: `evaluation_20260523_121048.json`
- Kept 4 current-truth files at `results/` top level: `bakeoff_hhem_isolated_day14.json`, `w1b2_threshold_day14.json`, `w3a_metric_effect_day14.json`, `w3b_retrieval_effect_day14.json`.
- Updated 14 by-path doc citations across 7 files to point to the new `archive/` paths: `docs/day11-evaluation.md` (2 hits), `docs/evaluation-methodology.md` (1), `docs/session-notes/day12.md` (6), `docs/session-notes/day13.md` (1), `docs/session-notes/day14.md` (2), `docs/plans/day13-investigation.md` (1), `docs/experiments/day13/scorer_probe.md` (1).
- Left `docs/session-notes/day12.md:237` unchanged — the one remaining stale path is inside a verbatim ```` ``` ```` block reproducing the literal stdout of `scripts/analyze_reeval.py`.
- Wrote `results/MANIFEST.md`: per-file question, cited-by ADR/phase, bucket, beside-not-over sentence for `w3a_*`/`w3b_*`, coherence note, and zero-citation grep for the worthless file.

### Why

The ADRs cite evidence in prose ("the bake-off," "Probe A," "the W3a isolated metric effect," "q07 T W3a 0.285") rather than by file path — `grep -r "results/...json" docs/adr/` returns nothing. Moving a file therefore breaks no markdown link. But mis-bucketing one orphans the evidence behind a shipped ADR claim: the number stays in the ADR text with no recoverable source on disk. The manifest is the deliverable because the directory layout carries no semantic meaning; the manifest is the sole anchor between files and the claims that depend on them.

`w3a_metric_effect_day14.json` and `w3b_retrieval_effect_day14.json` stay current-truth rather than being archived because each ran the scorer(s) on frozen `(response, chunks)` to isolate one variable — the isolated-variable question that a live multi-pass run cannot answer. The forthcoming P2 run sits beside them, not over them.

The archive-not-delete default applies because the worthless snapshot (`evaluation_20260523_121048.json`) cleared the zero-citation grep, making deletion safe, but safe is not the same as worthwhile. Explicit approval is required to delete; absent that, archive.

`docs/session-notes/day12.md:237` was preserved without path rewriting because it is a verbatim ```` ``` ```` block reproducing the literal stdout of `scripts/analyze_reeval.py` on a specific day. Editing it to reflect the current file path would falsify a historical artifact.

### Surprising

- The day15-plan premise "the ADRs cite numbers not paths" was accurate for `docs/adr/` but false for `docs/` at large. `evaluation_day12.json` had 5 by-path citations outside the ADR directory: 2 in `docs/day11-evaluation.md`, 1 in `docs/evaluation-methodology.md`, and 2 in `docs/session-notes/day12.md`. The wider grep was necessary.
- The q07 Torvalds line in the drafted w3b manifest entry was inverted at the stop gate. The draft read "q07 T 0.335 > 0.40 threshold confirmed deliver under fixed retrieval." The fact is the opposite: HHEM 0.335 is below the 0.40 gate, so q07 T falls back after the dedup fix despite being oracle-grounded (53.6%). This is the accepted paraphrase-misroute limitation, not a deliver confirmation. Corrected before any file was written.
- `evaluation_day12_reeval.json` had a mixed citation profile in `docs/session-notes/day12.md`: two by-path hits (lines 135 and 397, updated) plus one hit inside a verbatim code block (line 237, preserved). The three lines required three different dispositions in the same file.

### Deferred

- `src/eval/harness.py:183` has a default output parameter of `"results/evaluation_day12.json"`. This is a write default, not a read; moving the file does not break the harness. Routed to P1b's §12.5 stale-comment/constant audit.
- `evaluation_20260523_121048.json` archived, not deleted. Zero-citation grep confirms deletion is safe; explicit approval needed to act on it.
- P1b (codebase audit + `docs/codebase-audit.md` template): independent of P1, may proceed in parallel from t0 per the plan dependency graph.
- P2 (canonical multi-pass run): gated on P1 merge. Cost-gate applies; call-count estimate must be computed fresh before any spend.

### ADR candidates

- None. P1 touched no decision. `results/` layout is operational housekeeping, not an architectural choice. No locked decision was reopened.

---

## Phase P1b — Codebase audit + reusable template

> Pass 1: run and propose. No code deleted or edited this pass. All destructive resolutions stop-gated for Ruby approval.

### §12.5 Audit — six categories

---

#### Category 1: Dead code from old architecture

**Command 1a:** `uv tool run vulture src/ --min-confidence 80`

```
src/evaluation/hhem/modeling_hhem_v2.py:4: unused import 'AutoModelForTokenClassification' (90% confidence)
src/schemas.py:125: unused variable 'cls' (100% confidence)
src/schemas.py:153: unused variable 'cls' (100% confidence)
src/style/feature_extractor.py:17: unused import 'Any' (90% confidence)
```

Per-finding decisions:

- `modeling_hhem_v2.py:4: AutoModelForTokenClassification` — unused import in vendored HHEM model file. **Document-and-defer** — HHEM vendor code is locked (ADR-020/021); do not touch. Routed to Ruby.
- `schemas.py:125, 153: cls` — false positive. Both are `@classmethod @field_validator` Pydantic v2 methods; `cls` is the required first parameter of any `@classmethod` and need not appear in the body. No action.
- `feature_extractor.py:17: Any` — confirmed genuine dead import. `from typing import Any` appears only on line 17; `Any` is not referenced anywhere else in the file. Zero-caller check: the specific import `Any` has zero uses within the file (grep for `\bAny\b` returns only the import line). **In-scope fix** — remove the unused import. Non-destructive; file has live importers. Stop-gated for pass 2.

**Command 1b:** `uv run pytest --cov=src --cov-report=term-missing -q`

```
Name                                           Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------
src/__init__.py                                    0      0   100%
src/agents/__init__.py                             0      0   100%
src/agents/clone_agent.py                         71      0   100%
src/agents/evaluator_agent.py                     64      0   100%
src/agents/fallback_agent.py                      77      0   100%
src/cli.py                                       120      8    93%   212-218, 297
src/components/__init__.py                         4      0   100%
src/components/gatekeeper.py                      35      0   100%
src/components/scoring_engine.py                  19      0   100%
src/components/style_profile_builder.py           27      0   100%
src/config.py                                     62      0   100%
src/eval/__init__.py                              0      0   100%
src/eval/harness.py                             107     87    19%   57-61, 70-102, 135-173, 195-291
src/eval/query_loader.py                           8      0   100%
src/evaluation/__init__.py                         3      0   100%
src/evaluation/confidence_scorer.py               26      0   100%
src/evaluation/groundedness_scorer.py             38      3    92%   79-81
src/evaluation/hhem/__init__.py                    0      0   100%
src/evaluation/hhem/configuration_hhem_v2.py       9      3    67%   12-14
src/evaluation/hhem/modeling_hhem_v2.py           40     20    50%   12, 43, 63-69, 73-84
src/components/retriever.py                       50      9    82%   71-72, 76-82
src/fallback/__init__.py                           4      0   100%
src/fallback/calendar_mock.py                     25      0   100%
src/fallback/context_summarizer.py                22      0   100%
src/fallback/unstyled_responder.py                17      0   100%
src/flow.py                                       96      3    97%   123-129, 239
src/rag/__init__.py                                8      0   100%
src/rag/chunker.py                                56     10    82%   73, 76-78, 82-93, 102, 112
src/rag/citation_extractor.py                     21      0   100%
src/rag/corpus_loader.py                          43      0   100%
src/rag/embedder.py                               92      9    90%   35-37, 73-74, 178-179, 194-195
src/rag/indexer.py                                33      0   100%
src/rag/reranker.py                               38      3    92%   58-60
src/rag/retriever.py                              21      1    95%   48
src/schemas.py                                   125      0   100%
src/style/__init__.py                              0      0   100%
src/style/email_parser.py                        144     16    89%   127-129, 159, 169, 285-295
src/style/feature_extractor.py                   165      0   100%
src/style/profile_builder.py                      77      0   100%
src/style/style_scorer.py                         12      0   100%
src/visualization.py                             124      0   100%
----------------------------------------------------------------------------
TOTAL                                           1883    172    91%
532 passed, 0 failed
```

Coverage findings:

- `src/eval/harness.py` 19% — expected; the harness is the runtime evaluation entrypoint, exercised by system runs not unit tests. Low coverage is not a dead-code signal here.
- `src/evaluation/hhem/configuration_hhem_v2.py` 67%, `src/evaluation/hhem/modeling_hhem_v2.py` 50% — HHEM vendor files. Low coverage is expected for vendored inference code exercised only during a live HHEM model call. **Document-and-defer** — HHEM locked.
- All other significant coverage gaps (`retriever.py:71-82`, `chunker.py:73-112`, `embedder.py:35-37,73-74,178-195`) are error-handling paths or alternate index-type branches, not dead architecture remnants.

---

#### Category 2: Dead documentation references

**Command 2a:** `grep -rn "rag_agent\|style_crew\|evaluator_steps\|fallback_steps" docs/ --include="*.md"`

```
docs/PRD.md:1136-1139  (§12.2 v1-to-v2 mapping table — deliberate historical record)
docs/PRD.md:1200, 1206  (§12.5 audit checklist grep patterns — meta-reference)
docs/learning-journal.md:291, 330, 338, 344, 356, 706  (historical engineering journal)
docs/adr/ADR-001-crewai-flow-pattern.md:16, 39  (v1→v2 context explanation)
docs/adr/ADR-014-agent-component-inventory.md:46  (reclassification rationale)
docs/plans/day1-plan.md through day14-plan.md  (historical plans)
docs/session-notes/day10.md through day14.md  (audit trail of retirements)
```

Decision: **no findings.** All matches are in historical documents deliberately recording the v1→v2 transition (PRD §12.2 mapping table, session-note audit trails, plan files, ADR explanations). These are the expected audit trail, not broken references to live files. The files named (`rag_agent.py`, `style_crew.py`, etc.) were retired and are correctly absent from `src/`.

**Command 2b:** `grep -rn "ADR-009.*threshold\|ADR-009.*0\.75" docs/`

```
docs/day8-findings.md:194: "ADR-009 (0.75 threshold). Still un-written."
docs/PRD.md:1201: (the audit checklist grep pattern itself — meta-reference)
docs/plans/day14-plan.md:276: (historical plan option-memo reference)
```

Decision: **no findings.** `docs/day8-findings.md` is the historical Day 8 engineering record carried forward unchanged per §7.5.1; the "Still un-written" entry was accurate at Day 8. The v2 system does not use ADR-009's threshold. The PRD and plan references are archival.

**Command 2c:** `ls docs/architecture/` → directory does not exist (P3b creates it). **No findings.**

---

#### Category 3: v1 vocabulary leaks

**Command 3a:** `grep -rn "ChatStyleAgent\|RAGAgent\|PlannerAgent\|evaluator_steps\|fallback_steps" src/ --include="*.py"`

No output. **No findings.**

**Command 3b:** `grep -rn "final_score" src/ --include="*.py"`

```
src/schemas.py:199:    come from one LLM call (ADR-011 hybrid). There is no combined final_score —
src/schemas.py:201:    (ADR-018). extra="forbid" so any caller still passing final_score or
```

Decision: **no finding.** Both lines are in the `EvaluationResult` docstring explaining *why* `final_score` was removed (anti-v1 documentation). The `extra="forbid"` config enforces that callers cannot pass `final_score` — these comments are load-bearing explanation.

**Command 3c:** `grep -rn "0\.75" src/ --include="*.py"`

```
src/agents/evaluator_agent.py:36: # prose and the LLM drifted to flagging at ~0.70-0.75 instead of 0.60.
src/agents/evaluator_agent.py:76: # delegated to LLM judgment which drifted to flagging at ~0.70-0.75 in practice.
```

Decision: **no finding.** These are ADR-017 RC-1 fix context comments explaining the historical drift that made arithmetic thresholds necessary. Load-bearing WHY comments.

**Command 3d:** `grep -rn "weighted.*formula\|0\.4.*style\|0\.4.*ground" src/ --include="*.py"`

```
src/visualization.py:116: ax.axvline(0.40, ..., label="GROUNDEDNESS_MIN 0.40 (HHEM)")
src/schemas.py:200:    routing is decided by the Gatekeeper, not a weighted formula
```

Decision: **no findings.** `visualization.py:116` matched `0\.4.*ground` against the correct v2 HHEM threshold label (false positive on the grep pattern). `schemas.py:200` is an anti-v1 statement.

---

#### Category 4: Orphaned data files

**Command 4a:** `ls -la data/cache/`

```
embeddings_minilm.json     11M  27 Apr 22:29
embeddings_minilm.npz     8.5M  28 Apr 00:18
embeddings_openai.json    921M  27 Apr 22:29
embeddings_openai.npz      47M   1 Jun 23:50
embeddings_openai_semantic.npz 30M  6 May 16:51
```

**Command 4b:** `ls -la results/` — already handled by P1; current-truth files remain at top level, archive/ holds audit-trail and worthless files. No action needed.

**Command 4c:** `find data/ -type d`

```
data/
data/fallback_logs    (empty)
data/cache
data/evaluations      (empty)
data/emails
data/models
data/rag
data/rag/faiss_index_minilm
data/rag/faiss_index_semantic
data/rag/faiss_index
data/eval
```

Per-finding decisions:

- **Finding 4a**: `data/cache/embeddings_openai.json` (921MB) — .json-format embedding cache. `grep -rn "embeddings_openai\.json" src/ tests/ scripts/` returns zero hits. The live pipeline reads only `.npz` format (`src/rag/embedder.py` defaults to `data/cache/embeddings_openai.npz`). **In-scope delete candidate** — zero-reader confirmed. Awaiting pass-2 approval.
- **Finding 4b**: `data/cache/embeddings_minilm.json` (11MB) — same pattern. Zero-reader confirmed. **In-scope delete candidate** — awaiting approval.
- **Finding 4c**: `data/cache/embeddings_openai_semantic.npz` (30MB) — referenced in `scripts/experiment_6b_chunking.py:88` only. Not live pipeline. **Document-and-defer** — deletes experiment reproducibility; route to Ruby.
- **Finding 4d**: `data/evaluations/` (empty directory) — no src/ or tests/ reference. **In-scope delete candidate** — empty dir, zero content to lose.
- **Finding 4e**: `data/fallback_logs/` (empty directory) — no src/ or tests/ reference. **In-scope delete candidate** — empty dir.
- **Finding 4f**: `data/rag/faiss_index_minilm/` — not referenced by live `src/` code (default is `data/rag/faiss_index`). **Document-and-defer** — experiment-era index; deletes experiment reproducibility.
- **Finding 4g**: `data/rag/faiss_index_semantic/` — not referenced by live `src/` code. `scripts/experiment_6b_chunking.py` references the semantic path. **Document-and-defer** — same reason.

---

#### Category 5: Stale comments and docstrings

**Command 5a:** `grep -rn "# .*final_score\|# .*threshold\|# .*0\.75\|# .*5.*agents" src/`

```
src/agents/evaluator_agent.py:34: # ADR-017 RC-1 fix: flag thresholds as named constants, not LLM judgment.
src/agents/evaluator_agent.py:35: # Previously the thresholds lived only as f-string literals in natural-language
src/agents/evaluator_agent.py:36: # prose and the LLM drifted to flagging at ~0.70-0.75 instead of 0.60.
src/agents/evaluator_agent.py:47: # never validated as a per-response flag threshold; 0.70 is the cosine proximity
src/agents/evaluator_agent.py:75: # ADR-017 RC-1 fix: threshold comparison is arithmetic, was previously
src/agents/evaluator_agent.py:76: # delegated to LLM judgment which drifted to flagging at ~0.70-0.75 in practice.
src/components/gatekeeper.py:17: # Import thresholds from the evaluator so the router and evaluator stay in sync.
src/components/gatekeeper.py:18: # Any future recalibration of a threshold automatically propagates here.
```

Decision: **no findings.** All are load-bearing WHY comments. `evaluator_agent.py:34-36, 75-76` are ADR-017 RC-1 context (the drift history is why constants were needed). `evaluator_agent.py:47` explains a specific threshold constant's semantics. `gatekeeper.py:17-18` documents a cross-module invariant.

**Command 5b:** `grep -rn '""".*final_score\|""".*threshold' src/`

No output. **No findings.**

**Command 5c:** `grep -rn "TODO\|FIXME\|XXX" src/`

```
src/evaluation/hhem/modeling_hhem_v2.py:45:    # TODO: Figure out how to publish only the adapter yet still able to do end-to-end pulling and inference.
src/evaluation/hhem/configuration_hhem_v2.py:17: # FIXME: The default values passed to the constructor are not used.
```

Decision: **document-and-defer** for both. Both are in HHEM vendor files. HHEM is a locked decision (ADR-020/021). The TODO is about model distribution logistics; the FIXME is about unused constructor defaults in the vendored config class. Routed to Ruby; do not touch vendor code here.

---

#### Category 6: Unused dependencies

**Command:** inspect `pyproject.toml`; for each dependency, `grep -rn "^import {name}\|^from {name}" src/ tests/`

```
crewai          : 4 imports in src/
crewai-tools    : 0 imports
instructor      : 4 imports
litellm         : 5 imports
faiss           : 6 imports
cohere          : 1 import
numpy           : 20 imports
pyyaml (yaml)   : 1 import
python-dotenv   : 0 imports in src/tests/ (used only in scripts/)
click           : 2 imports
rich            : 3 imports
matplotlib      : lazy-imported inside function bodies in visualization.py (confirmed via broader grep)
seaborn         : 0 imports anywhere in src/, tests/, or streamlit_app.py
streamlit       : imported in streamlit_app.py (confirmed)
sentence-transformers (sentence_transformers) : 1 import
datasets        : 1 import (src/rag/corpus_loader.py)
langchain-text-splitters (langchain_text_splitters) : 1 import
sentencepiece   : 0 direct imports
```

Per-finding decisions:

- **Finding 6a**: `crewai-tools>=0.40.0` — zero direct imports in src/ or tests/. May be a hard transitive dep of crewai. **Document-and-defer** — needs `uv remove crewai-tools && uv run python -c "from crewai import Agent"` verification before removal. Route to Ruby.
- **Finding 6b**: `seaborn>=0.13.0` — zero imports anywhere in src/, tests/, or streamlit_app.py. The only chart module (`src/visualization.py`) imports matplotlib directly inside function bodies; seaborn is never called. **In-scope delete candidate** from pyproject.toml — zero-use confirmed. Awaiting pass-2 approval.
- **Finding 6c**: `sentencepiece>=0.2.0` — zero direct imports. Indirect dependency of sentence-transformers or HHEM (T5-family tokenizer). **Document-and-defer** — removing risks breaking HHEM inference path; HHEM is locked. Route to Ruby.
- **Finding 6d**: `python-dotenv>=1.0.0` — zero imports in src/ or tests/; used in `scripts/diagnostic_6a_*.py`, `scripts/experiment_6*.py`, `scripts/w3b_*.py`. **Document-and-defer** — live use in experiment scripts; removing would break them.

---

#### Named deferred finding #5: prompt-vs-constant drift audit

Per Day-15 plan decision 4 and the scope fence: the full audit comparing LLM prompt text against code constants (GROUNDEDNESS_MIN, threshold values embedded in f-string prompts) to verify no drift has accumulated is **deferred finding #5**. It is named here as required and routed to Ruby. No commands run for this finding in P1b; the audit commands themselves are part of the deferred scope.

---

#### `src/eval/harness.py:183` — stale default output path

First surfaced in P1 Deferred, routed to P1b. The `run_measurement()` signature has `output: str | Path = "results/evaluation_day12.json"` as its default. P2 will always pass an explicit path (`results/evaluation_day15.json`), so the stale default does not affect the run. But the default is misleading for any future caller not passing an explicit path. **In-scope fix candidate** — change default to a neutral value (e.g. `"results/evaluation_latest.json"`) or a date-stamped template. Non-destructive. Awaiting pass-2 approval.

---

### In-scope fix list (proposed; stop-gated for pass-2 approval)

| # | File | Change | Zero-importer / zero-use proof |
|---|------|--------|-------------------------------|
| F1 | `src/style/feature_extractor.py:17` | Remove `from typing import Any` | `grep -n "\bAny\b" src/style/feature_extractor.py` returns only the import line |
| F2 | `data/cache/embeddings_openai.json` (921MB) | Delete | `grep -rn "embeddings_openai\.json" src/ tests/ scripts/` → no output |
| F3 | `data/cache/embeddings_minilm.json` (11MB) | Delete | `grep -rn "embeddings_minilm\.json" src/ tests/ scripts/` → no output |
| F4 | `data/evaluations/` (empty dir) | Delete | no src/ or tests/ reference; directory is empty |
| F5 | `data/fallback_logs/` (empty dir) | Delete | no src/ or tests/ reference; directory is empty |
| F6 | `seaborn>=0.13.0` in `pyproject.toml` | Remove dependency | `grep -rn "seaborn" src/ tests/ streamlit_app.py` → no output |
| F7 | `src/eval/harness.py:183` | Update stale default from `"results/evaluation_day12.json"` to neutral name | P2 will always pass explicit path; default only affects direct callers |

### Document-and-defer list (routed to Ruby; no action in P1b or P2)

| Finding | File(s) | Reason for deferral |
|---------|---------|---------------------|
| D1 | `src/evaluation/hhem/modeling_hhem_v2.py:4` | Unused import in HHEM vendor code — HHEM locked (ADR-020/021) |
| D2 | `src/evaluation/hhem/modeling_hhem_v2.py:45` | TODO in vendor code — HHEM locked |
| D3 | `src/evaluation/hhem/configuration_hhem_v2.py:17` | FIXME in vendor code — HHEM locked |
| D4 | `src/evaluation/hhem/configuration_hhem_v2.py` 67% / `modeling_hhem_v2.py` 50% coverage | Vendor code, locked decision |
| D5 | `data/cache/embeddings_openai_semantic.npz` | Experiment artifact; `scripts/experiment_6b_chunking.py:88` reads it |
| D6 | `data/rag/faiss_index_minilm/` | Experiment artifact; experiment scripts depend on it |
| D7 | `data/rag/faiss_index_semantic/` | Experiment artifact |
| D8 | `crewai-tools>=0.40.0` in pyproject.toml | Zero direct imports but may be hard transitive dep of crewai; needs live verification |
| D9 | `sentencepiece>=0.2.0` in pyproject.toml | Indirect dep of sentence-transformers / HHEM; removing risks breaking HHEM |
| D10 | `python-dotenv>=1.0.0` in pyproject.toml | Used in scripts/; removing breaks experiment reproducibility |
| D11 | Prompt-vs-constant drift audit (finding #5) | Deferred finding #5 per Day-15 plan scope fence; full audit deferred |

### Built

- Ran all six §12.5 categories with the exact PRD-specified commands. Captured raw output per category above.
- Wrote `docs/codebase-audit.md` as the reusable vocabulary-parameterized template (PRD §7.5.1).
- Proposed 7 in-scope fixes and 11 document-and-defer items.
- Suite confirmed green before and after: 532 passed, 0 failed.

### Why

Day-14 W4a ran the v1-residue retirements (`evaluator.py`, `rag_agent.py`, the prompt-string fixes) but did not run the complete §12.5 audit pass. P1b is the missing complete pass. The template is the reusable artifact: it is vocabulary-parameterized so P7-P9 and P1-P5 re-verification can copy it and swap the project-specific grep patterns without reimplementing the structure.

### Surprising

- Category 3 (v1 vocabulary leaks) produced **zero genuine findings** — the v2 codebase is clean. Every match was either an anti-v1 docstring (explicitly recording what was removed), an ADR-017 historical-context comment, or a false positive on the grep pattern.
- Category 2 (dead doc references) also produced zero findings among *current* doc files. All matches were in historical plan files, session notes, and the PRD §12.2 mapping table — which are exactly the expected audit-trail documents, not broken references.
- The largest in-scope candidate by byte count is `data/cache/embeddings_openai.json` at 921MB — a v1-era JSON embedding cache that has been superseded by the `.npz` format but never cleaned up. Its deletion is zero-risk but large-impact on disk.
- vulture reported `cls` as unused at 100% confidence in two Pydantic `@classmethod @field_validator` methods. This is a structural false positive in vulture: `@classmethod` requires `cls` as the first parameter by Python protocol, regardless of whether the body uses it.

### Deferred

- D1–D11 above (routed to Ruby; listed in document-and-defer table).
- Prompt-vs-constant drift audit (finding #5) — named and routed; no work done.
- `src/eval/harness.py:183` stale default — in-scope fix F7, stop-gated.

### ADR candidates

- None. P1b found no design decision requiring an ADR. All findings are either v1 residue (already covered by existing ADRs) or operational cleanup with no architectural implications.

---

### P1b Pass 2 — Approved resolutions

> Appended after Ruby's approval of the F1/F4/F5/F6 subset and the revised F7 form. Pass-2 scope: execute approved changes, reclassify F2/F3, record F7 form. No other changes.

**F6 prerequisite:** `grep -rn "seaborn" scripts/` → **no output.** Scripts are also clean; removal proceeded.

**F7 form chosen — option (a): keyword-only required argument.**

Caller check: `grep -rn "run_measurement" src/ tests/ scripts/ --include="*.py"` returned only the definition and its own module docstring — **zero external callers.** No caller depends on the default, so the default can be dropped cleanly. Applied change: added `*` to make both params keyword-only and removed the `output` default entirely:

```python
# Before
def run_measurement(
    path: str | Path = "data/eval/queries.json",
    output: str | Path = "results/evaluation_day12.json",
) -> dict:

# After
def run_measurement(
    *,
    path: str | Path = "data/eval/queries.json",
    output: str | Path,
) -> dict:
```

A call without `output` now raises `TypeError: run_measurement() missing 1 required keyword-only argument: 'output'` — fails loudly, as required.

**F1 — `src/style/feature_extractor.py:17`:** Removed `from typing import Any`. Zero-use confirmed (grep for `\bAny\b` in the file returned only the import line).

**F4 — `data/evaluations/`:** Zero-reader grep (`grep -rn "data/evaluations" src/ tests/ scripts/`) → no output. Directory confirmed empty. Deleted via `rmdir`.

**F5 — `data/fallback_logs/`:** Zero-reader grep (`grep -rn "data/fallback_logs" src/ tests/ scripts/`) → no output. Directory confirmed empty. Deleted via `rmdir`.

**F6 — `seaborn>=0.13.0` in `pyproject.toml`:** Removed. `uv sync` output:

```
Resolved 218 packages in 699ms
Uninstalled 2 packages in 3ms
 - seaborn==0.13.2
 ~ torvalds-digital-clone==0.1.0
```

Build succeeded, seaborn uninstalled.

**F2/F3 reclassified to defer:**

- `data/cache/embeddings_openai.json` (921MB) → **Document-and-defer.** Zero readers confirmed, but zero readers does not prove the file can be regenerated. This is a paid OpenAI artifact; deletion is irreversible without a confirmed regeneration path and cost. Added to the D-list as D12.
- `data/cache/embeddings_minilm.json` (11MB) → **Document-and-defer.** Same reasoning: regeneration path and cost must be confirmed before deletion. Added as D13.

**Full suite after all changes:**

```
532 passed, 35 warnings in 20.09s
```

Suite green. P1b complete.

---

## Phase P3b — Architecture diagrams A1–A6

### Built

- Confirmed `docs/architecture/` did not exist (P3b creates it). No stale old-named diagram files existed anywhere in the repo (`find` for the §12.1 Day-9 retired filenames returned no output).
- Read-only stale-inventory check: `grep -rn "GatekeeperAgent\|PlannerAgent" docs/architecture/` → directory absent, no hits. Post-write exit grep across the completed `docs/architecture/` → **empty** (PASS). The old class names appear nowhere in the new diagram files.
- Created `docs/architecture/` with six Mermaid diagrams matching the §7.5.3 file list:

| File | Diagram type | Summary |
|------|-------------|---------|
| `A1-system-architecture.md` | `graph TB` | High-level: Adapters → DigitalCloneFlow → 3 Agents + 4 Components → External Services. README hero. |
| `A2-single-query-sequence.md` | `sequenceDiagram` | Single `kickoff()`: retrieve → clone → evaluate → route → finalize OR handle\_fallback |
| `A3-dual-leader-sequence.md` | `sequenceDiagram` | `compare_leaders()`: shared retrieval once, then per-leader clone → evaluate → route (ADR-005) |
| `A4-data-models.md` | `classDiagram` | All 13 Pydantic schemas with composition arrows; CloneState as the Flow's typed state |
| `A5-data-flow.md` | `graph LR` | Offline lane (style learning + RAG indexing) vs online lane (per-query pipeline) |
| `A6-agent-vs-component.md` | `graph TB` | ADR-009/ADR-014 criterion applied: 3 LLM-driven Agents left, 4 deterministic Components right |

- Every diagram encodes the ADR-014 v2 inventory: **3 Agents** (CloneAgent, EvaluatorAgent, FallbackAgent), **4 Components** (Retriever, StyleProfileBuilder, ScoringEngine, Gatekeeper), **1 Flow** (DigitalCloneFlow). Gatekeeper is labelled deterministic Component with "No LLM — reclassified per ADR-018" in A6 and "Deliver-or-fallback arithmetic router" in A1.
- Flagged the PRD §7.5.3 prose contradiction (pre-ADR-018 inventory) into the deferred PRD reconciliation note at the bottom of A1, A2, A3, and A6. PRD not edited.
- A1 uses `graph TB` with clean subgraph structure — renderable as PNG for P4 README hero.
- Retire-old guard: no surviving old-named or stale-inventory diagram file found. Confirmed as clean.

### Why

ADR-014 (corrected 2026-06-01 per ADR-018) is the authoritative inventory. The PRD §7.5.3 prose predates ADR-018 and still names the pre-reclassification split. Following the locked ADR rather than the stale PRD prose is decision #3 from the Day-15 plan; editing the PRD in P3b is explicitly out of scope (deferred to the full PRD reconciliation pass). The deferred-reconciliation note in each affected diagram is the flag mechanism the plan requires.

The stale-file guard was non-trivial: `docs/evaluation-methodology.md` and `docs/day11-evaluation.md` contain the old agent names, but those are historical engineering documents (pre-Day-14 diagnosis records), not architecture diagram files. The guard targeted `docs/architecture/` only, and that directory was fully absent before P3b.

### Surprising

- No stale files to retire — the §12.1 Day-9 cleanup was complete. Both the old-named file check and the pre-write directory check returned empty. The retire-old guard found nothing to act on.
- The deferred-reconciliation notes initially caused the exit grep to fail because they quoted the old class names verbatim. The fix was to rephrase using "the pre-ADR-018 inventory" and "Gatekeeper classified as an Agent" rather than the literal class name strings — same information, grep-clean.
- A5 (data-flow) surfaces an important runtime detail: StyleProfileBuilder does not appear in the online query path at all. It runs offline and caches the StyleProfile to disk; the Flow loads from cache. This is easily missed when reading A1 alone.

### Deferred

- PRD §7.5.3 prose reconciliation (A1/A2/A3/A6 contradiction notes) — deferred to the full PRD reconciliation pass.
- A1 PNG generation for the README hero — deferred to P4 (P4 is the README phase; it uses A1 as input).

### ADR candidates

- None. P3b touched no decision. The diagrams record ADR-014/ADR-018/ADR-005 decisions already locked; no new decision was made.
