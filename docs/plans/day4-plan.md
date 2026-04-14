# Day 4 — EvaluatorAgent + FallbackAgent

## Context

Days 1–3 are merged. Style profiling (Day 2) and RAG retrieval (Day 3) produce styled responses with citations but no quality gate. Day 4 adds the evaluation layer that decides whether to deliver a response or fall back to a calendar-booking message, per PRD Section 5b. The weighted score `0.4×style + 0.4×groundedness + 0.2×confidence` with a 0.75 threshold is the core deliverable; it unblocks Day 5 flow orchestration.

Two things to flag from exploration that diverge from the brief:
- The Day 2 style scorer function is `score_style(profile, response_features)` in [src/style/style_scorer.py](../../src/style/style_scorer.py), not `compute_style_score`. Plan uses the real name.
- The project standard is LiteLLM (not raw OpenAI). Instructor will wrap LiteLLM via `instructor.from_litellm(completion)`, consistent with the codebase's LLM conventions. The P5 `instructor.from_openai(OpenAI())` pattern will be adapted.

## Deliverables

### 1. Evaluation modules — `src/evaluation/`

**[src/evaluation/__init__.py](../../src/evaluation/__init__.py)** — explicit re-exports following the [src/rag/__init__.py](../../src/rag/__init__.py) pattern:
```python
from .groundedness_scorer import score_groundedness
from .confidence_scorer import score_confidence
from .evaluator import evaluate

__all__ = ["score_groundedness", "score_confidence", "evaluate"]
```


**[src/evaluation/groundedness_scorer.py](../../src/evaluation/groundedness_scorer.py)**
- `score_groundedness(response: str, chunks: list[RetrievalResult], top_k: int = 5) -> float`
- Split `response` into sentences via regex (`re.split(r'(?<=[.!?])\s+', ...)`) — no nltk dependency to keep install light.
- **Performance: batch embed all sentences in ONE call via `embed_openai(sentences)` (the underlying batched function in [src/rag/embedder.py](../../src/rag/embedder.py), batches of 100 with shared MD5 cache).** Calling `embed_query()` per sentence would issue N API calls; batching keeps it to 1 per response. Chunks are already embedded on `chunk.embedding` from the RAG pipeline — reuse, don't re-embed.
- For each sentence, take max cosine sim across the 5 chunk embeddings. Average the maxes.
- Edge cases: empty response → 0.0; zero chunks → 0.0; single-word response → single sentence path; chunks missing `.embedding` → fall back to `embed_openai(chunk_texts)` batch call (still 1 API call total).

**[src/evaluation/confidence_scorer.py](../../src/evaluation/confidence_scorer.py)**
- `score_confidence(query: str, response: str, retrieval_results: list[RetrievalResult]) -> float`
- Three signals, equally weighted (1/3 each), clipped to [0,1]:
  - `retrieval_relevance`: mean of `.score` from top-5 `RetrievalResult` items (reranker scores).
  - `completeness`: fraction of query keywords (lowercase, stopword-stripped, len>2) appearing in response.
  - `uncertainty_penalty`: `1 - min(1, hedge_count / 5)` where hedge_count matches phrases like "I think", "maybe", "possibly", "not sure", "might", "could be", "I believe" (case-insensitive regex).
- No LLM call here. The explanation LLM call lives in `evaluator.py` so it runs once per evaluation, not once per scorer.
- **Note: the equal 1/3 sub-weights are a starting assumption.** Flag for **Day 6 weight sensitivity experiment** — sweep weights against a labeled validation set to find the optimal mix; current values are placeholder, not principled.

**[src/evaluation/evaluator.py](../../src/evaluation/evaluator.py)**
- `evaluate(query, response, response_features, profile, retrieval_results) -> EvaluationResult`
- Calls `score_style` + `score_groundedness` + `score_confidence`.
- Computes `final = 0.4*style + 0.4*groundedness + 0.2*confidence`.
- Decision: `"deliver"` if `final >= 0.75` else `"fallback"`.
- One Instructor call producing `ExplanationModel(explanation: str)` — prompt includes all three sub-scores and decision. Use `instructor.from_litellm(litellm.completion)` with `model="gpt-4o-mini"`, `max_retries=3`.
- Returns `EvaluationResult` — schema's `@model_validator` will re-verify the weighted formula (tolerance already enforced in [src/schemas.py](../../src/schemas.py)).

### 2. Fallback modules — `src/fallback/`

**[src/fallback/__init__.py](../../src/fallback/__init__.py)** — explicit re-exports following the [src/rag/__init__.py](../../src/rag/__init__.py) pattern:
```python
from .calendar_mock import generate_available_slots
from .context_summarizer import summarize_context
from .unstyled_responder import generate_unstyled_response

__all__ = ["generate_available_slots", "summarize_context", "generate_unstyled_response"]
```


**[src/fallback/calendar_mock.py](../../src/fallback/calendar_mock.py)**
- `generate_available_slots(n: int = 3, seed: int | None = None) -> list[str]`
- Pure Python `datetime`. Skip weekends. Business hours 9am–5pm. Random times seeded for deterministic tests.
- Format: `"Tuesday, April 16, 2026 at 10:30 AM PT"`.

**[src/fallback/context_summarizer.py](../../src/fallback/context_summarizer.py)**
- `summarize_context(query: str, chunks: list[RetrievalResult]) -> str`
- Extract unique `chunk.source_topic` values from retrieval results. Template: `"Your question about {query_topic} touches on {topics_joined}. I'd like to discuss this in more depth."` No LLM — deterministic string composition.

**[src/fallback/unstyled_responder.py](../../src/fallback/unstyled_responder.py)**
- `generate_unstyled_response(query: str, chunks: list[RetrievalResult]) -> str`
- Instructor + LiteLLM call with `UnstyledAnswer(answer: str)` Pydantic model. Plain-factual system prompt explicitly instructing "no rhetorical style, no personality cues".

### 3. Agent facades

**[src/agents/evaluator_agent.py](../../src/agents/evaluator_agent.py)**
- `class EvaluatorAgent` with `evaluate(response, chunks, profile, query, response_features) -> EvaluationResult` — thin wrapper around `evaluator.evaluate()`.

**[src/agents/fallback_steps.py](../../src/agents/fallback_steps.py)**
- `build_fallback_response(query, chunks, trigger_reason) -> FallbackResponse`
- Composes: `generate_available_slots()` + `summarize_context()` + `generate_unstyled_response()` + hardcoded `calendar_link="https://cal.com/<placeholder>"` (PRD treats this as mock).

### 4. Tests — `tests/`

One test file per new module (6 files):
- `test_groundedness_scorer.py` — mock `embed_query` with `@patch("src.evaluation.groundedness_scorer.embed_query")`; assert sentence split, max-then-avg math, empty response → 0.0.
- `test_confidence_scorer.py` — hedge detection (pure hedging → low score), keyword coverage (perfect match → 1.0), reranker avg math.
- `test_evaluator.py` — mock all three scorers + Instructor client; assert weighted formula, boundary at exactly 0.75 (→ "deliver"), 0.7499 (→ "fallback").
- `test_calendar_mock.py` — weekend-skipping, n=3, deterministic with seed, format string regex.
- `test_context_summarizer.py` — topic extraction, dedup, empty chunks edge case.
- `test_unstyled_responder.py` — Instructor call mocked, verify prompt contains "no style".

Follow existing `_make_features(**kwargs)` / `tmp_path` / `@patch` patterns from [tests/test_embedder.py](../../tests/test_embedder.py) and [tests/test_style_scorer.py](../../tests/test_style_scorer.py). Target ≥90% coverage on new files.

### 5. Documentation

**[docs/adr/ADR-004-groundedness-scoring-approach.md](../adr/ADR-004-groundedness-scoring-approach.md)** — follow the **full gold-standard ADR template** from CLAUDE.md writing rules (not just ADR-001's basic structure). Required sections:

1. **Status / Date / Category** — Accepted, 2026-04-14, Evaluation.
2. **Context** — Why we need a groundedness gate; what "ungrounded response" looks like in this domain (leadership coaching emails); cost/latency budget per evaluation (<0.5s per PRD).
3. **Decision** — Sentence-level max-cosine-similarity heuristic, batch-embedded via `embed_openai()`. Include the formula and a small worked example.
4. **Alternatives Considered** — table with columns: **Alternative | How it works | Why Not**.
   - Pure LLM-as-judge (gpt-4o-mini): ~$0.002/call × 1000 evals = $2/run, 600–800ms latency, non-deterministic — Why Not: blows latency budget, harder to unit-test, cost compounds in Day 6 weight sweeps.
   - Token overlap (BLEU/ROUGE-L): <1ms, free — Why Not: semantically blind, paraphrases score 0 even when faithful.
   - NLI entailment model (e.g., DeBERTa-MNLI): more rigorous — Why Not: extra ~400MB model dependency, 100ms+ per pair, overkill for a weighted-formula component.
   - Per-sentence LLM judge: most accurate — Why Not: N×LLM calls per response, fails the <0.5s budget at N=5 sentences.
5. **Consequences** — three subsections:
   - **Easier**: deterministic unit tests (mock embeddings → exact scores), reuses Day 3 embedder + cache for free, sub-100ms evaluation, weight-sweep-friendly for Day 6.
   - **Harder**: heuristic can be fooled by topical-but-incorrect sentences (high cosine, wrong fact); requires periodic recalibration as corpus shifts; threshold (0.60) is empirical, not theoretical.
   - **Portability**: cosine sim works with any embedding provider — swap MiniLM for OpenAI in [src/rag/embedder.py](../../src/rag/embedder.py) without touching scorer code.
6. **Interview Signal** — "I picked a heuristic that hits the latency budget and unit-tests cleanly, then I planned a calibration step (5-sample LLM judge spot-check) to validate the threshold empirically rather than guessing." Demonstrates: cost-aware design, calibration discipline, knowing when LLM-as-judge is overkill.
7. **Java/TS Parallel** — analogous to choosing `String.equals()` + Levenshtein over a full NLP pipeline for fuzzy matching: you pick the cheapest measure that captures the signal you actually need, then validate on real data. **The key insight: the right tool is the one that meets your constraints (latency, cost, testability), not the most sophisticated one.**
8. **Cross-References**:
   - [ADR-002](../adr/ADR-002-embedding-strategy.md) — the embedding choice this scorer reuses; groundedness is "free" because Day 3 already pays the embedding cost.
   - **P2 (ai-rag-evaluation-framework) empirical findings** — semantic-similarity-based groundedness scored within 0.08 of LLM-judge on 50 labeled samples, validating the heuristic-first approach.
   - **P5 (ai-shoptalk-knowledge-agent)** — same RAG-eval pattern in production, threshold tuned to 0.55 for shorter answers; informs our 0.60 starting point.

**Calibration plan** (executed Day 4, documented in journal): run 5 sample evaluations through both the heuristic and a `gpt-4o-mini` LLM judge; report Spearman correlation and threshold delta. If the heuristic correlates poorly (<0.7), adjust the threshold or revisit the design.

**docs/journal/day4.md** — learning entries.

**[CLAUDE.md](../../CLAUDE.md) update** — check Day 4 boxes; mark Day 5 next.

## Critical files to modify
- Create: `src/evaluation/__init__.py`, `src/evaluation/groundedness_scorer.py`, `src/evaluation/confidence_scorer.py`, `src/evaluation/evaluator.py`
- Create: `src/fallback/__init__.py`, `src/fallback/calendar_mock.py`, `src/fallback/context_summarizer.py`, `src/fallback/unstyled_responder.py`
- Create: `src/agents/evaluator_agent.py`, `src/agents/fallback_steps.py`
- Create: 6 test files in `tests/`
- Create: `docs/adr/ADR-004-groundedness-scoring-approach.md`, `docs/journal/day4.md`
- Update: `CLAUDE.md` Day 4 checklist

## Functions/utilities reused (not rebuilt)
- `score_style` from [src/style/style_scorer.py](../../src/style/style_scorer.py)
- `embed_query` from [src/rag/embedder.py](../../src/rag/embedder.py) (MD5-cached — free on repeat text)
- `EvaluationResult`, `FallbackResponse`, `RetrievalResult`, `StyleProfile`, `StyleFeatures` from [src/schemas.py](../../src/schemas.py)
- Test fixture patterns from [tests/test_embedder.py](../../tests/test_embedder.py)

## Branch / commit plan
One branch `feat/day4-evaluator-fallback`, PR to main, no Co-Authored-By, no Claude attribution.

## Verification
1. `pytest tests/ -v` — all 305 prior tests still pass, ~50 new tests pass.
2. `pytest --cov=src/evaluation --cov=src/fallback --cov-report=term-missing` — ≥90% on new modules.
3. Smoke test in a scratch script: build a mock `StyleProfile` + retrieval results → call `EvaluatorAgent.evaluate()` → confirm `EvaluationResult.final_score` matches manually-computed weighted formula and `explanation` is non-empty.
4. Fallback path: invoke `build_fallback_response(...)` → confirm 3 slots, non-empty summary, unstyled response string.
5. Boundary test: construct inputs producing final_score ≈ 0.7499 → decision == "fallback"; 0.7500 → "deliver".
