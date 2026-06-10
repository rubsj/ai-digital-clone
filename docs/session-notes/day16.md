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
