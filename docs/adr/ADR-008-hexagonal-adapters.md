# ADR-008: Hexagonal Adapters for CLI and Streamlit over DigitalCloneFlow

**Project:** P6: Torvalds Digital Clone
**Category:** Architecture
**Status:** Accepted
**Date:** 2026-05-13

---

## Context

By Day 6 the core pipeline ran through a single orchestrated entry point: `DigitalCloneFlow.kickoff(inputs)` and `compare_leaders(query)`. Both live in `src/flow.py` and wrap all FAISS retrieval, Cohere reranking, LiteLLM generation, style scoring, and evaluation logic. Nothing outside `src/` needs to know about any of those dependencies.

Day 7 added two user-facing surfaces: a Click CLI (`src/cli.py`) and a Streamlit app (`streamlit_app.py`). Both needed access to the full pipeline. I had to decide whether those surfaces should import from the pipeline internals directly, which is fast to write but couples the UI to the ML stack, or only through the existing flow façade.

The structure is ports-and-adapters. The domain logic in `src/flow.py`, `src/agents/`, `src/evaluation/`, and `src/style/` sits in the center. `DigitalCloneFlow` and `compare_leaders` are the ports. `src/cli.py` and `streamlit_app.py` are adapters that only talk to those ports.

---

## Decision

Both `src/cli.py` and `streamlit_app.py` import exclusively from `src/flow.py`, `src/schemas.py`, and `src/config.py`. For the CLI's `learn` and `index` commands, they also import the narrow style and RAG façades: `src/style/email_parser.py`, the StyleProfileBuilder Component (`src/components/style_profile_builder.py`), `src/rag/corpus_loader.py`, `src/rag/chunker.py`, and the Retriever Component (`src/components/retriever.py`). Neither adapter imports `litellm`, `faiss`, `cohere`, or `openai` directly.

The rule is documented in `CLAUDE.md` as the Architecture Rule (no direct LiteLLM, FAISS, or Cohere imports from adapter code) and enforced in CI by grep checks in the phase stop gates:

```
grep -E "litellm|faiss|cohere|openai\." src/cli.py         # 0 hits
grep -E "litellm|faiss|cohere|openai\." streamlit_app.py   # 0 hits
```

---

## Alternatives Considered

**Import pipeline internals directly from adapters.** Each adapter would import `Retriever`, `CloneAgent`, `EvaluatorAgent`, and `FallbackAgent` itself. I didn't do this because it breaks the encapsulation that makes the pipeline testable. All 11 CliRunner tests in `tests/test_cli.py` mock at `src.cli.DigitalCloneFlow` and `src.cli.compare_leaders`. Direct internal imports would force me to patch a dozen pipeline objects per test.

**Shared adapter base class.** Extract common rendering logic (score formatting, fallback detection) into `src/adapters/base.py` shared by CLI and Streamlit. I skipped this because the CLI and Streamlit have different output primitives (`click.echo` vs `st.metric`), and a shared base would either be too thin to pull its weight or would leak one channel's output format into the other. With only two adapters and three duplicated lines, the abstraction would cost more than it saves.

**Streamlit-native session state for pipeline objects.** Cache `DigitalCloneFlow` and `Retriever` in `st.session_state` to skip the FAISS reload on every Streamlit rerun. I deferred this. The rerun penalty is latency only, not correctness, and session state adds lifecycle complexity I don't need for a portfolio demo with one user.

---

## Consequences

If I later swap the retrieval backend or the LLM client, the change stays inside `src/`. Neither adapter file moves. The CLI test suite keeps mocking at the same boundary point.

`streamlit_app.py` reinitializes `DigitalCloneFlow` on every Streamlit rerun, which means FAISS and the style profiles reload from disk each time the user submits a query. The reload takes a few seconds on a single-user demo, so I've tracked the `@st.cache_resource` fix as a post-portfolio followup rather than burning Day 7 time on it.

The `learn` and `index` CLI commands import the corpus loader and profile builder directly rather than going through `DigitalCloneFlow`, because they run independently of any live query. The Architecture Rule prohibits direct ML library imports, not all imports from `src/`, so this carve-out fits. Those commands pull only domain façades (`parse_mbox`, `build_profile_batch`, `load_corpus`, `chunk_documents`, `Retriever.build()`), never the underlying libraries.
