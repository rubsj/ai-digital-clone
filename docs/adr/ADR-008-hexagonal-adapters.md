# ADR-008: Hexagonal Adapters for CLI and Streamlit over DigitalCloneFlow

**Project:** P6: Torvalds Digital Clone
**Category:** Architecture
**Status:** Accepted
**Date:** 2026-05-13

---

## Context

Days 1–6 built the core pipeline as a single orchestrated entry point: `DigitalCloneFlow.kickoff(inputs)` and `compare_leaders(query)`. Both functions live in `src/flow.py` and encapsulate all FAISS retrieval, Cohere reranking, LiteLLM generation, style scoring, and evaluation logic. Nothing outside `src/` needs to know about any of those dependencies.

Day 7 added two user-facing surfaces: a Click CLI (`src/cli.py`) and a Streamlit app (`streamlit_app.py`). Both needed access to the full pipeline. The architectural question was whether those surfaces should import from the pipeline internals directly (fast to write, high coupling) or only through the existing flow façade (more constrained, lower coupling).

The pattern that emerged across both adapters is textbook hexagonal architecture (ports-and-adapters): the domain logic (`src/flow.py`, `src/agents/`, `src/evaluation/`, `src/style/`) sits at the center, `DigitalCloneFlow` and `compare_leaders` are the ports, and `src/cli.py` / `streamlit_app.py` are adapters that speak only to those ports.

---

## Decision

Both `src/cli.py` and `streamlit_app.py` import exclusively from `src/flow.py`, `src/schemas.py`, `src/config.py`, and for the CLI's `learn`/`index` commands, the narrow style and RAG façades (`src/style/email_parser.py`, `src/style/profile_builder.py`, `src/rag/corpus_loader.py`, `src/rag/chunker.py`, `src/agents/rag_agent.py`). Neither adapter imports `litellm`, `faiss`, `cohere`, or `openai` directly.

The rule is enforced by the Architecture Rule in `CLAUDE.md` ("no direct LiteLLM/FAISS/Cohere imports from adapter code") and verified in CI by grep checks in the stop gates for each phase.

---

## Alternatives Considered

**Import pipeline internals directly from adapters.** Each adapter imports `RAGAgent`, `StyleCrew`, `EvaluatorAgent`, and `FallbackAgent` individually. Saves the thin façade layer. Rejected: breaks the encapsulation that makes the pipeline testable and replaceable. The CLI tests (`tests/test_cli.py`) mock at the façade boundary (`src.cli.DigitalCloneFlow`, `src.cli.compare_leaders`) — direct internal imports would require patching a dozen internals per test.

**Shared adapter base class.** Extract common rendering/output logic (score formatting, fallback detection) into an `src/adapters/base.py` shared by CLI and Streamlit. Rejected for Day 7: the CLI and Streamlit have different output primitives (`click.echo` vs `st.metric`). A shared base would either be too abstract to be useful or would couple the output format to one delivery channel. Three similar lines (in CLI + in Streamlit) is better than a premature abstraction.

**Streamlit-native session state for pipeline objects.** Cache `DigitalCloneFlow` and `RAGAgent` in `st.session_state` to avoid re-initializing FAISS on every rerun. Technically straightforward (see Consequences). Deferred from Day 7: the rerun penalty is latency only, not correctness, and adds state lifecycle complexity that isn't needed for a portfolio demo.

---

## Quantified Validation

The hexagonal constraint is verified by three grep checks, one per adapter and one covering the full project:

```
grep -E "litellm|faiss|cohere|openai\." src/cli.py         # 0 hits
grep -E "litellm|faiss|cohere|openai\." streamlit_app.py   # 0 hits
```

CLI test isolation: all 11 CliRunner tests in `tests/test_cli.py` mock at `src.cli.DigitalCloneFlow` and `src.cli.compare_leaders`. No internal pipeline symbols are patched. This would be impossible without the façade boundary.

---

## Consequences

The adapter boundary means any future swap of the retrieval or generation backend (e.g. replacing FAISS with ChromaDB, or LiteLLM with a direct OpenAI client) only touches `src/` internals — neither `src/cli.py` nor `streamlit_app.py` changes. The CLI test suite continues to mock at the same boundary point regardless of what's underneath.

**Streamlit caching deferral.** `streamlit_app.py` currently reinitializes `DigitalCloneFlow` (which loads FAISS and profiles) on every Streamlit rerun. The fix is `@st.cache_resource` on a factory function that returns the flow instance, keeping FAISS in memory across reruns. This is tracked as a post-portfolio followup (Ruby owns the Notion page entry). It is not implemented in Day 7 because the demo use case (low query volume, single user) tolerates the reload latency.

**`learn` and `index` commands bypass the flow façade.** The CLI `learn` and `index` commands need to invoke the corpus-loading and profile-building pipelines independently of a live query, so they import from the style and RAG façade modules directly rather than through `DigitalCloneFlow`. This is intentional and consistent with the Architecture Rule: the rule prohibits direct LiteLLM/FAISS/Cohere imports, not all imports from `src/`. The `learn`/`index` commands import only domain façades (`parse_mbox`, `build_profile_batch`, `load_corpus`, `chunk_documents`, `RAGAgent.build`), not the underlying ML libraries.
