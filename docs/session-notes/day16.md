# Day 16 — Session Notes

## Phase P1 — Kill index re-embed + read-only diagnosis

### Built

Nothing. Read-only pass + kill.

### Kill

Killed PID 67636 (`python -m src.cli index`) after ~39 minutes of live OpenAI spend re-embedding 530K chunks. Confirmed dead via `pgrep`. Disk left as-is; no partial index or cache deleted.

### Diagnosis: `style_profile=None` crash

**Root cause: Cause A — loader missing at the query CLI call site.**

The flow has no internal profile-load step by design — `flow.py:6-7` documents this explicitly:
> *"Profile is caller-injected via `kickoff(inputs={"style_profile": ...})`; the Flow has no profile-load step."*

`CloneState.style_profile` defaults to `None` (`schemas.py:316`). Every caller that omits it from `kickoff(inputs={...})` crashes at `clone_agent.py:49` (`profile.features` on None).

Sites audited:
- `flow.py:218,228` (`compare_leaders`) — correct, loads and injects both profiles
- `harness.py:147,156` (`run_leader_pair`) — correct, loads and injects both profiles
- `cli.py:159` (`query` command) — **missing** — omitted `style_profile` entirely
- `streamlit_app.py:181` (single-leader path) — **missing** — omitted `style_profile` entirely

Not Cause B (no field rename mismatch) or Cause C (retrieval completed before the crash).

### Diagnosis: embedding cache re-embed

The embedder (`embedder.py:88`) reads/writes `data/cache/embeddings_openai.npz` (npz format). Cache logic is correct — cached chunks bypass the API. However, the codebase previously used a JSON cache (`data/cache/embeddings_openai.json`, 921 MB, April 27); the format changed to npz at some point. `_load_cache()` uses `np.load()` and never reads the `.json` file. The 921 MB JSON cache is a stranded dead artifact — all 530K chunks were submitted to OpenAI from scratch. This is why the killed run took 39+ minutes. Separate task; not fixed this session.

---

## Phase P2 — Fix profile injection at all kickoff sites

### Built

**`src/cli.py`** — two changes:
1. Added `load_profile` to the `profile_builder` import line.
2. In the `query` command: added `config = load_config()` + `profile = load_profile(Path(...))` before `flow.kickoff()`, then passed `style_profile=profile` in the inputs dict. Matches the pattern in `compare_leaders` (flow.py:216-222).
3. In the `if __name__ == "__main__":` block: added `os.environ["OMP_NUM_THREADS"] = "1"` and `multiprocessing.set_start_method("spawn", force=True)` before `cli()`. This fix was discovered in Day 15 (see day15.md §Segfault diagnosis) but was never committed; without it `query`, `compare`, and `evaluate` all SIGSEGV (exit 139) when HHEM loads inside CrewAI's asyncio executor.

**`streamlit_app.py`** — two changes:
1. Added `from src.config import load_config` and `from src.style.profile_builder import load_profile` imports.
2. In the single-leader dispatch block: derived `config_key` from `leader_choice`, loaded the profile, passed `style_profile=_profile` into `flow.kickoff()`.
3. Added `os.environ.setdefault("OMP_NUM_THREADS", "1")` and `multiprocessing.set_start_method("spawn", force=True)` at module top (before `import streamlit`) so the spawn mode is active before any flow execution.

**`tests/test_cli.py`** — updated `TestQueryCommand.test_styled_response`:
- Added `ANY` to mock imports (unused but available).
- Added `patch("src.cli.load_config")` and `patch("src.cli.load_profile")` to the context manager so the test remains a unit test (no real disk read).
- Updated `kickoff.assert_called_once_with` to include `"style_profile": mock_profile`.

Sites left unchanged: `flow.py:218,228` and `harness.py:147,156` — both already inject profiles correctly.

### Why

The flow's caller-inject design is intentional (ADR-005 shared retrieval requires chunks to be pre-populated, so a general pre-load step inside the flow would be architecturally wrong). The fix is at the two call sites that were omitted, not in the flow itself. The spawn/OMP fix is a macOS runtime necessity documented in day15 notes but was applied ad hoc then; committing it ensures all future entry points benefit automatically.

### Surprising

The Streamlit single-leader path (`streamlit_app.py:181`) was a second instance of the same bug, exactly as predicted in the diagnostic. `compare_leaders` (the dual-leader Streamlit path) was already correct because it delegates to `flow.py:compare_leaders()` which loads profiles internally.

---

## Verification

### Test suite

```
532 passed, 35 warnings in 20.53s
```

532/532 green (was 531/532 before test assertion update).

### Smoke tests (re-run of Day 16 blocked steps)

All entry points use the existing April-28 FAISS index (`data/rag/faiss_index/index.faiss`). No re-embed triggered.

| Step | Command | Result |
|------|---------|--------|
| 1. `learn` | `cli learn --leader torvalds` | PASS — 11,052 emails, profile saved |
| 2. `index` | `cli index` | RUNNING / no error — killed after 39 min (separate cache task) |
| 3. `query` | `cli query "…mutex vs spinlock…" --leader torvalds` | PASS — fallback response, no crash, profile injected |
| 4. `compare` | `cli compare "…mutex vs spinlock…"` | PASS — both leaders returned (fallback routing), no crash |
| 5. `evaluate` | `cli evaluate --queries /tmp/smoke_queries.json --output-dir /tmp/smoke_eval_out` | PASS — 2 queries, 4 flow completions, results written to `/tmp/smoke_eval_out/evaluation_20260609_222146.json`, 2×2 grid: both leaders 2/2 deliver |
| 6. Streamlit | `streamlit run streamlit_app.py --server.headless true` | PASS — bound at localhost:8501, HTTP 200, no import errors, no `ModuleNotFoundError: seaborn` |

**Blast radius checks:**
- F6 (seaborn removal): no `ModuleNotFoundError: seaborn` in any step.
- F7 (harness output kwarg): `evaluate` CLI passes an explicit timestamped path; no TypeError.
- HHEM / spawn: HHEM loads and scores without SIGSEGV after spawn fix.
- Dedup at Retriever: retrieval returned chunks in all 4+4 flow runs; reranking ran (Cohere).

### Decisions not revisited

None. The spawn fix is a runtime configuration, not an architectural change. The JSON→npz cache migration (stranded cache) is a separate task.

---

## Phase P3 — Corpus pin, cache conversion, zero-spend index

### Built

**`src/rag/corpus_loader.py`** — three changes:
1. Added `_EVALUATED_TOPICS: frozenset[str]` constant naming the 5 evaluated textbooks.
2. Added `topic_filter` parameter (default `_EVALUATED_TOPICS`) replacing positional `select(range(N))`. The default now selects by identity, not row position.
3. Added per-topic duplicate guard: the open-phi/textbooks CS slice has two rows for "Principles and Practice of Assistive Technology." (rows 1 and 14 of the CS slice). Row 1 (305,787 chars, 864 chunks) is the evaluated document — 864/864 chunk keys match the FAISS index. Row 14 (404,702 chars, 1,115 chunks) has 0/864 key overlap; it is a different version. Guard logs a warning and skips subsequent occurrences; first match wins.

**`tests/test_corpus_loader.py`** — three changes:
1. Added `_EVALUATED_TOPICS` to imports.
2. Added `topic_filter=None` to all existing tests using generic mock topics (so they test loading logic independently of the default filter).
3. Replaced `test_load_corpus_max_docs_caps_output` call with explicit `topic_filter=None, max_docs=3`.
4. Added `test_load_corpus_default_topic_filter_accepts_evaluated` — verifies default filter accepts evaluated topics and rejects others.
5. Added `test_load_corpus_duplicate_topic_skips_second` — verifies first-match semantics with the duplicate guard (no exception, 1 doc returned).

**`data/cache/embeddings_openai.npz`** — converted from JSON (921 MB, 26,913 entries) to npz (130 MB). Explicit schema mapping: old `{md5: [float...]}` → new `keys` str array + `vectors` float32 matrix (the schema `_load_cache` reads). Read-back verified: `allow_pickle=False` passes, dtype float32, first-entry round-trip allclose atol=1e-6.

### Key-match verification (stop-gate result: PASS)

- FAISS index chunks: 6,713 total, 5,856 unique MD5 keys (857 duplicate texts across the 5 docs; the cache deduplicates correctly).
- JSON cache: 26,913 entries (first 20 textbooks' worth, from the Day-6 max_docs=20 run that OOM'd loading the 921 MB JSON).
- Index keys missing from cache: **0**. All 5,856 unique chunk keys are present.

### Zero-spend build (step 4 evidence)

```
Loading corpus…
Skipping duplicate document for topic 'Principles and Practice of Assistive Technology.' (occurrence 2); keeping first match only.
  Loaded 5 documents.
  Created 6713 chunks.
  FAISS index saved.
```

No "Embedding (OpenAI)..." progress bar appeared — the `if uncached_texts:` branch in `embed_openai()` was never entered. Zero OpenAI API calls.

### Test suite

```
534 passed, 35 warnings in 19.31s
```

534/534 (was 532 at start of day, +2 new tests from P3).

### Judgment calls surfaced

**Duplicate handling:** Changed the duplicate guard from `raise ValueError` to `logger.warning + continue` (skip). Rationale: the first occurrence is always the evaluated document (100% key match); the second is a different/longer version with 0% match. Raising blocked the `index` command with no actionable path; warning + skip reproduces the evaluated corpus correctly. The content-based key-match step in the verification catches any future regression if dataset row ordering changes.

**`max_docs` retained:** The `max_docs` parameter is kept for tests and experimentation (pass `topic_filter=None, max_docs=N` for the old behavior). It is no longer the corpus-selection mechanism for production use.
