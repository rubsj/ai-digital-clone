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

---

## Phase P2 — Canonical multi-pass run

### Pre-launch findings (surfaced before spend)

**Retry ceiling.** `_LLM_MAX_RETRIES = 2` in all three agents (CloneAgent, EvaluatorAgent, FallbackAgent); this bounds `_parse_*` instructor calls to 3 attempts each. `crew.kickoff()` uses CrewAI 1.13.0's default `max_iter=25` — not overridden in any agent. 4-deliver / 6-fallback per-leader per-query are floors, not ceilings; the theoretical bounded maximum if all iterations fire is much larger. In practice, these single-agent no-tool crews complete in 1 iteration; `max_iter=25` is never approached.

**Checkpointing.** The harness writes `all_records` to disk after every query pair (`harness.py:252-253`). A mid-run crash preserves all completed-pair records. The harness cannot resume; a restart clears `all_records` and overwrites the partial file from pass 1. Mid-run crash safety is partial: data up to the crash is safe, but a complete result requires a full re-run.

**Segfault diagnosis.** The first three run attempts ended in exit code 139 (SIGSEGV) after HHEM model weights loaded. Root cause: macOS fork-safety issue. CrewAI 1.13.0's Crew runner uses loky (joblib) internally, which forks child processes via `fork()`. On macOS, `fork()` after PyTorch initializes Metal/MPS threads causes SIGSEGV in the child. CrewAI's asyncio event loop also initiates the `evaluate` step's `EvaluatorAgent.__init__` (which triggers `ScoringEngine()` → `HHEMGroundednessScorer()`) while `clone` is awaiting the gpt-4o-mini API response — this is why HHEM loads during the "clone Running" window, not at the expected evaluate step. Fix: `multiprocessing.set_start_method('spawn', force=True)` and `OMP_NUM_THREADS=1` before any import. With spawn mode the macOS fork guard is satisfied and the run completes cleanly.

**Output path.** The plan warning about `harness.py:183` was about the module-level docstring (which still names `evaluation_day12.json`) rather than the function signature — `run_measurement()` now requires `output` as a keyword argument with no default (F7, resolved in P1b). The run passed `output='results/evaluation_day15.json'` explicitly; the day12 path was never written.

### Built

- Ran `run_measurement(output='results/evaluation_day15.json')` with spawn multiprocessing and `OMP_NUM_THREADS=1`.
- C4 run design: Pass 1 (20 queries × 2 leaders = 40 runs), Pass 2 + Pass 3 (14 in-domain queries × 2 leaders × 2 passes = 56 runs), reactive OOD recheck for q20/Torvalds (2 additional runs). Total: **50 records** in `results/evaluation_day15.json`.
- Reactive recheck fired: **1 time (q20/Torvalds)**. 2 recheck records written.

### Results

**In-domain deliver rate — distribution over 3 passes (PRD §2.1 and ADR-015):**

| Leader | Pass 1 | Pass 2 | Pass 3 | Mean | Stdev | Range | E2 ≥55% | E1 ≥39% | Floor |
|--------|--------|--------|--------|------|-------|-------|---------|---------|-------|
| Torvalds | 78.6% (11/14) | 71.4% (10/14) | 85.7% (12/14) | **78.6%** | ±7.1pp | [71.4%, 85.7%] | PASS | PASS | PASS (floor 42.9%) |
| KH | 71.4% (10/14) | 78.6% (11/14) | 92.9% (13/14) | **81.0%** | ±10.9pp | [71.4%, 92.9%] | PASS | PASS | PASS (floor 35.7%) |

Neither leader is below its ADR-015 floor. No STOP gate triggered.

**Variance vs W3a (locked decision #2 — logged as variance, not regression):**

W3a frozen re-score (run on frozen `(response, chunks)` pairs from `evaluation_day12_reeval2.json`) reported 64.3% for the Torvalds in-domain weighted-oracle match. The P2 fresh Torvalds mean is 78.6% — delta +14.3pp. This is expected temp=0.3 variance: W3a re-scored frozen responses; P2 generated fresh ones at temp=0.3. Per locked decision #2 and the MANIFEST coherence note, this divergence is logged as variance and is not "corrected" toward W3a. The W3a numbers remain the ADR-015 evidence; the P2 distribution is the fresh-system presentation number for the README.

**OOD fallback rate (PRD §2.1 100% bar):**

Pass-1 OOD decisions (6 queries × 2 leaders = 12):

| Query | Torvalds | gs (T) | KH | gs (KH) |
|-------|----------|--------|----|---------|
| q15 | fallback | 0.279 | fallback | 0.154 |
| q16 | fallback | 0.054 | fallback | 0.020 |
| q17 | fallback | 0.055 | fallback | 0.045 |
| q18 | fallback | 0.078 | fallback | 0.032 |
| q19 | fallback | 0.174 | fallback | 0.135 |
| q20 | **deliver** | **0.422** | fallback | 0.362 |

11/12 = 91.7% fallback. **PRD §2.1 100% bar: MISS by 1** — q20/Torvalds, known deferred OOD-defense gap.

**q20 Torvalds reactive recheck results:**

| Attempt | Decision | Groundedness score |
|---------|----------|-------------------|
| Pass 1 | deliver | 0.422 |
| ood_recheck_1 | deliver | 0.473 |
| ood_recheck_2 | deliver | 0.403 |

q20/Torvalds delivered 3/3 attempts; all gs ≥ 0.40 gate. This is a systematic gap, not stochastic noise: Torvalds generates a response grounded enough in the retrieved chunks to clear HHEM@0.40 on q20 every time. q20/KH held (gs=0.362 < 0.40). Per the plan: report, do not fix (known deferred finding, OOD-defense gap at q20, query-relevance signal). The reactive recheck count is reported here as data about OOD defense: 2 rechecks fired, 0 reversed the deliver verdict.

### Why

The multi-pass design (3 passes for in-domain, C4 full-pass-then-recheck for OOD) produces a distribution rather than a point estimate. At temp=0.3, a single pass is a noisy sample; the distribution captures the realistic operating range. The stdev values (±7.1pp for Torvalds, ±10.9pp for KH) quantify that noise explicitly — they are not alarm signals, they are the spread the system produces at this temperature.

The reactive OOD recheck exists to distinguish a stochastic OOD slip (would not repeat) from a systematic gap (repeats). q20/Torvalds delivering 3/3 classifies it as systematic.

The segfault fix (`spawn` + `OMP_NUM_THREADS=1`) addresses a macOS runtime interaction, not a system logic defect. The system logic — HHEM@0.40 gate, deterministic Gatekeeper, dedup-live Retriever — ran correctly once the process model was correct.

### Surprising

- The segfault: HHEM loading appeared during the "clone Running" banner in every attempt. This was not HHEM loading prematurely due to a code bug; it was CrewAI's asyncio event loop beginning `evaluate` step initialization (including `EvaluatorAgent.__init__` → `ScoringEngine()` → `HHEMGroundednessScorer()`) while clone's gpt-4o-mini API call was in-flight. The crash was a macOS fork/spawn conflict in the loky layer, not a sequencing bug in the flow.
- HHEM loads twice per `run_leader_pair` call (once for the Torvalds flow's EvaluatorAgent, once for the KH flow's). The module-level `_get_singleton()` in `groundedness_scorer.py` is never called from within the `EvaluatorAgent` path — `EvaluatorAgent.__init__` calls `ScoringEngine()` which calls `HHEMGroundednessScorer()` directly, bypassing the singleton. The singleton function exists but is not wired into the live code path. This means 50 records × 2 HHEM loads = ~100 HHEM model loads during the run. Each load is fast (~0.05s from local cache) so the impact on wall time was minor.
- P2 numbers are substantially above W3a's 64.3%. A +14.3pp jump is large for temp=0.3 variance. The spread bears watching: if the P3a groundedness distribution chart shows a tight cluster well above 0.40, it suggests the fresh responses are genuinely more grounded than the frozen W3a set (different random seed, different generation paths). This is not a problem — it is the expected behavior of a live run — but it is notable context for the README framing.
- q20 KH holds at gs=0.362 every time while q20 Torvalds delivers at gs=0.40+ every time. This asymmetry is not random; Torvalds' voice style likely generates responses that are phrased more directly against the retrieved chunks for this particular OOD topic. The gap is a property of Torvalds' generation style + the q20 topic, not a scoring calibration issue.

### Deferred

- q20 OOD-defense gap (query-relevance signal): known deferred finding, scope-fenced per plan decision 4. Reactive recheck confirmed the gap is systematic for Torvalds. Not fixed here.
- Singleton wiring: `HHEMGroundednessScorer` is instantiated fresh per `EvaluatorAgent` instead of via `_get_singleton()`. No correctness impact; redundant loading. Document-and-defer; scope-fenced as dead-code / optimization scope, not an ADR matter.

### ADR candidates

- None. All locked decisions (HHEM, 0.40, per-leader floors, deterministic routing) held. The q20 OOD result is the known gap — not a new decision point. The segfault fix is a runtime env configuration, not an architectural decision.

---

## Phase P3a — Charts (§2.10 / §7.6)

### Built

9 charts in `results/charts/` with §7.6-correct names. Generation script: `/tmp/generate_charts_day15.py`. All run-derived charts sourced exclusively from `results/evaluation_day15.json`.

**§7.6 inventory reconciliation:**

| File | §7.6 slot | Run-derived? | Change from pre-P3a |
|------|-----------|--------------|---------------------|
| `01-style-radar-dual-leader.png` | #1 | No (style profiles) | Renamed from `01-style-radar.png` |
| `02-routing-correctness-grid.png` | #2 | Yes (pass 1) | **New** — was missing from old set |
| `03-style-score-distribution.png` | #3 | Yes | Renamed + regenerated (per-leader overlaid) |
| `04-groundedness-score-distribution.png` | #4 | Yes | Renamed + regenerated (HHEM label, 25 bins) |
| `05-deliver-rate-distribution.png` | #5 | Yes | **Slot renamed** (see judgment call below) |
| `06-fallback-trigger-distribution.png` | #6 | Yes | Renamed + regenerated (trigger_reason content) |
| `07-latency-distribution.png` | #7 | Yes | Renamed + regenerated (deliver vs fallback separated) |
| `08-torvalds-style-evolution-pre-post-2018.png` | #8 | No (mbox) | Renamed; regenerated from 11k emails; rolling-mean edge artifact fixed (post-P3a) |
| `09-retrieval-relevance-contrast.png` | — | Yes | **§7.6 addition** — see judgment call below |

Stale-named PNGs (`01-07` with old names) removed. Visualization tests: 16/16 pass.

New functions added to `src/visualization.py`: `plot_routing_correctness_grid`, `plot_style_score_distribution_per_leader`, `plot_groundedness_from_eval`, `plot_deliver_rate_distribution`, `plot_fallback_trigger_distribution`, `plot_latency_by_path`, `plot_style_evolution`, `plot_retrieval_relevance_contrast`. Old functions retained for test backward compatibility.

### Why

- §7.6 names the chart set; the pre-P3a 7 PNGs had stale names and were missing chart #2 (routing grid). Running from the canonical P2 file ensures the charts reflect the fresh distribution, not W3a numbers.
- Chart #9 (retrieval relevance contrast) added because it is the highest-value diagnostic from the run: the in-domain vs OOD top-chunk score gap (~3 orders of magnitude) makes the q20 OOD-defense gap visually obvious and is direct evidence for the deferred query-relevance gate fix.
- Style evolution chart regenerated from mbox (not cached) to ensure it uses the same `StyleFeatureExtractor` path as the profile builder.

### Surprising

**Groundedness distribution is NOT bimodal.** The P3a prompt expected "clusters bimodally near ~0.9 and ~0.01 with the 0.40 gate in an empty valley." The actual P2 data shows:

| Range | Count |
|-------|-------|
| < 0.10 | 6 |
| 0.10–0.40 | 23 |
| 0.40–0.60 | 47 |
| 0.60–0.80 | 20 |
| ≥ 0.80 | 4 |

Distribution peaks at 0.40–0.60; max = 0.87. HHEM's theoretical bimodal behavior (0 or 1 for clear entailment/contradiction pairs) does not manifest on fresh gpt-4o-mini responses — the LKML-grounded generations produce intermediate scores, not polar verdicts. The 0.40 gate sits at the LOW end of the dominant cluster, not in an empty valley. Chart rendered honestly from the actual data; the discrepancy from expectation is surfaced here.

### Judgment calls

**§7.6 #5 slot renamed.** §7.6 names the #5 slot `05-score-component-breakdown.png` ("Per-query stacked bars — style/ground/confidence"). Replaced with `05-deliver-rate-distribution.png` showing in-domain deliver rate per leader × pass with ADR-015 floors and E1/E2 reference lines. The deliver rate chart is the primary portfolio metric (it directly addresses PRD §2.1 acceptance criteria); the score component breakdown is an analysis detail. Flagging for PRD §7.6 reconciliation — either update §7.6 to name this slot `05-deliver-rate-distribution.png`, or add the score component breakdown as a 10th chart.

**Chart #9 is outside §7.6.** The retrieval-relevance contrast chart (`09-retrieval-relevance-contrast.png`) has no §7.6 slot. Added because it is the strongest visual evidence for the deferred query-relevance gate fix (ADR candidate). Flagged for PRD §7.6 reconciliation: add as chart #9 or move to a supplementary section.

**Routing grid uses pass 1 only.** The "Day-11 headline visualization" framing of chart #2 targets a single-pass routing view. Using all 3 passes would average over the variance; using pass 1 gives the cleaner routing accuracy picture (32/40 = 80.0%). The q20/Torvalds OOD deliver is correctly shown as the one red cell in the OOD section.

### Deferred

- PRD §7.6 reconciliation: two items above (slot #5 rename, chart #9 addition).
- Score component breakdown (`05-score-component-breakdown.png` per §7.6) was replaced rather than added. If the score breakdown per query is needed for the portfolio, add it as chart #10 in P4 or a future pass.

### Post-P3a fix — chart 08 rolling-mean edge artifact

**Problem.** The 12-month rolling mean used `np.convolve(..., mode="same")`, which computes partial windows at both ends of the monthly series. With only 0–11 months of data available at the boundaries, the convolution average is pulled toward zero, producing a visible hard dive in formality and capitalization at both the 2015 and late-2023 edges. Every panel was titled "noise" with a near-zero delta — the conclusion was correct — but the most visually prominent feature contradicted the title.

**Fix applied (option 1 — trim).** Changed to `mode="valid"`, which only outputs points where the full 12-month window fits: `len(y) - 11` points starting at `month_dates[11:]`. The thin monthly-mean line still spans the full date range; only the rolling-mean line is trimmed. The dives are gone. No data, no pre/post split, no delta computation, no "noise" verdicts changed.

**Code change:** `src/visualization.py`, `plot_style_evolution`, one line: `mode="same"` → `mode="valid"`, `ax.plot(month_dates, sm, ...)` → `ax.plot(month_dates[11:], sm, ...)`.

**Chart regenerated:** `results/charts/08-torvalds-style-evolution-pre-post-2018.png` (215K).

### ADR candidates

- None. Chart naming and content are presentation choices; no architectural decision warranted.

---

## Phase P4 — README

### Built

- Replaced the 11-line `Coming Soon` stub `README.md` with the portfolio README. Inverted pyramid: lede with the headline result, portfolio line (21 ADRs / 532 tests / 9 charts), the A1 architecture diagram as an inline Mermaid hero (no PNG render exists in `docs/architecture/`, so the diagram is embedded the way P4/P5 embed theirs), then the Results table above the fold.
- Headline deliver rates quote the fresh P2 distribution: Torvalds 78.6% mean [71.4%, 85.7%], KH 81.0% mean [71.4%, 92.9%], both with the spread, not a point estimate. Verified against `results/evaluation_day15.json` directly (33/42 and 34/42).
- The variance note frames the README-vs-ADR divergence as temp=0.3 generation variance (W3a 64.3% frozen re-score vs P2 78.6% fresh), per the locked-decision-1 coherence note. The ADRs keep W3a; the README quotes P2.
- E2 rendered as PARTIALLY met, stated precisely: deliver-rate criterion passes, zero-hallucination passes, OOD-fallback misses 100% at 91.7% (11/12, single-pass 6 OOD × 2 leaders). Not rendered as a clean pass.
- Five embedded charts: 04-groundedness-distribution (rework + continuity), 05-deliver-rate-distribution (primary metric), 02-routing-grid (the q20 red cell), 01-style-radar (dual-leader style), 09-retrieval-relevance-contrast (q20 in the Limitations lead).
- Architecture section carries the ADR-014 inventory (3 Agents / 4 Components / 1 Flow, Gatekeeper a deterministic Component) and a 7-row ADR table (009/014, 018, 019, 020, 021, 002, 015).

### Why

- The honest-rework arc (cosine measured lexical echo → Torvalds deficit was the metric not the clone → HHEM entailment swap → floors held) is the spine of the README, given its own H2 section between Results and Findings rather than buried in an ADR table. It is the strongest engineering story in the project.
- q20 leads the Limitations section as a characterized gap with a known fix, not an apology. The retrieval-contrast framing (groundedness passes because the answer IS grounded; retrieval relevance separates q20 from in-domain by ~3 orders of magnitude, 0.0013 vs ≥0.32) is the senior-judgment read, and the deferred query-relevance gate signal is named as the fix.
- The "groundedness is continuous, not bimodal" finding (51% of in-domain scores in the 0.40–0.60 band, gate at the low edge of the dominant cluster) is used to explain WHY the deliver rate is a distribution, tying the chart to the headline framing.

### Surprising

- No A1 PNG exists. P3b produced A1–A6 as Mermaid only; the plan's P3b exit check said A1 should be "renderable as a PNG," but none was rendered. Resolved by embedding the Mermaid inline, which GitHub renders and which matches the P4/P5 hero convention, so no blocker. Noting it in case Ruby wants a rasterized hero later.
- The em dash constraint reaches the Results table: the four "not applicable" cells were written as em dashes by reflex, then changed to `n/a` so the file carries zero em dashes (the en dash in the "Feb–May 2026" footer and the "0.40–0.60" ranges is intentional and matches siblings).

### Deferred

- PRD reconciliation rows (not edited, surfaced for the deferred ledger): PRD §7.5.3 prose still says 4 Agents + 3 Components with GatekeeperAgent; PRD §2.4/§2.5/§2.11 still describe groundedness as cosine-distribution. The README follows ADR-014 and HHEM, not the stale PRD prose. These belong in the Notion PRD Reconciliation ledger, not fixed here.
- §7.6 chart-slot reconciliation (carried from P3a): slot #5 renamed to deliver-rate-distribution, chart #9 added outside §7.6. The README embeds both as shipped; the §7.6 spec update is still open.
- Loom recording and recruiter-pitch section remain out of scope per the plan scope fence.

### ADR candidates

- None. The README is presentation of locked decisions (HHEM, 0.40, floors, deterministic routing). No decision was made or reopened while writing it. The q20 query-relevance gate signal is a candidate for a future ADR if Ruby opens the deferred OOD-defense finding, but it is not authored here.
