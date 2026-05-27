# P6 Product Requirements Document (PRD) v2

> **This is the product requirements and architecture contract for P6 v2.**
>
> **Reading order before any implementation work:**
> 1. [Customized Requirements v2](https://www.notion.so/36ddb630640a811aae74d4c8d5b10565) (Notion) — the contract above this PRD
> 2. This PRD — what to build and why
> 3. CLAUDE.md — implementation conventions, code patterns, and Engineering Protocols
>
> **For Claude Code (Opus and Sonnet):**
> - Opus handles implementation planning; Sonnet executes plans
> - Do NOT re-debate architecture decisions — they are final, documented in ADRs
> - Do NOT implement v1 patterns (Python functions wrapped in Flow decorators called "agents")
> - DO follow the Agent vs Component vocabulary per ADR-009: LLM-driven work uses CrewAI Agent abstraction in `src/agents/`; deterministic work uses Component classes with `run()` method in `src/components/`
> - DO read each ADR referenced in §4 before implementing the corresponding subsystem
>
> **If implementation reveals a constraint that requires changing an architectural decision:**
> - STOP. Surface the conflict to Ruby in chat.
> - Do NOT silently work around it.
> - Architectural drift was the root cause of v1's failure (see docs/day8-findings.md).

**Project:** P6 — Torvalds Digital Clone (Multi-Agent System)
**Version:** 2.0
**Date:** 2026-05-26 (Day 9 of Portfolio Sprint)
**Author:** Ruby Jha
**Status:** Active — supersedes v1

**Deferred items:** Items deferred from active work live on the [Post-Portfolio Followups](https://www.notion.so/36cdb630640a812a9d99d79951011897) Notion page, not in this PRD.

---

## Revision History

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2026-04-02 | Ruby | Initial PRD against v1 customized requirements. Implementation followed this PRD through Day 7. |
| 2.0 | 2026-05-26 | Ruby | Day 8 verification revealed implementation gaps. v2 re-derives the architecture from the v2 customized requirements (4 Agents + 3 Components + Flow). Old PRD deleted; this is the only PRD. |

---

## §1 Overview

P6 builds a multi-agent system that creates digital clones of Linus Torvalds and Greg Kroah-Hartman from real Linux Kernel Mailing List (LKML) emails. The system learns each leader's writing style from their email history, retrieves technical knowledge from a textbook corpus, generates responses in the leader's voice grounded in retrieved content, evaluates the quality of those responses, decides whether the response is good enough to deliver or should fall back to a context-aware graceful redirection, and supports side-by-side dual-leader comparison.

The v2 architecture consists of **4 LLM-driven Agents** (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent) coordinated through a **CrewAI Flow** orchestrator that calls **3 deterministic Components** (Retriever, StyleProfileBuilder, ScoringEngine) at fixed points in the pipeline. The distinction between Agents and Components is the central architectural decision (per ADR-009): Agents are LLM-driven reasoning work using the CrewAI Agent abstraction; Components are deterministic Python classes with `run()` methods. Python functions are NOT called agents anywhere in v2.

The v2 architecture replaces v1's weighted scoring formula (`0.4×style + 0.4×ground + 0.2×conf >= 0.75`) with LLM-driven routing via GatekeeperAgent (per ADR-010). Day 8 verification measured the v1 formula producing a tight ~0.20-wide score distribution across 20 queries, making any threshold either deliver-everything or fallback-everything. GatekeeperAgent reasons over individual scores, the response, the chunks, and explanation flags from EvaluatorAgent. Routing decisions become defensible reasoning rather than brittle threshold comparisons.

**Portfolio narrative carried forward from v1:** P2 benchmarked RAG configurations. P5 built a production RAG system. P6 takes RAG into a multi-agent architecture. The v2 progression: "I measured RAG, built RAG, embedded RAG into a multi-agent system, then rebuilt the multi-agent system honestly when verification revealed my v1 implementation didn't match my v1 commitments."

### Project Goal

Build a production-quality multi-agent system that demonstrates:

1. **Multi-agent orchestration** through CrewAI Flow with 4 real Agents and deterministic routing
2. **Style transfer** from real LKML email corpus to generated responses (dual-leader)
3. **RAG-grounded generation** with citations and groundedness scoring
4. **LLM-driven quality routing** that reasons about deliver vs fallback beyond a fixed threshold
5. **Graceful failure handling** with leader-appropriate fallback prose and templated failsafe
6. **End-to-end evaluation** with a documented methodology and reproducible re-runs

### What's Different in v2

- v2 honors the multi-agent commitment from the customized requirements by building real CrewAI Agents where LLM reasoning adds value, instead of wrapping Python functions in Flow decorators
- v2 removes the weighted scoring formula and 0.75 threshold; routing is GatekeeperAgent's LLM reasoning
- v2 success metrics measure routing correctness as the headline outcome, with quantitative scores as informational metrics
- v2 corrects Cohere reranker silent failure (env var name bug from Day 3)
- v2 uses the corpus-aligned eval query set (data/eval/queries.json) designed during Day 8

---

## §2 Success Criteria

Success criteria are organized by the architectural layer they measure, with the routing-correctness metric as the headline.

### §2.1 Routing Correctness (Headline Metric)

The 2x2 routing-correctness grid is the primary system-level deliverable. Measured on the v2 evaluation query set (14 in-domain + 6 out-of-domain queries per leader, 40 records total in dual-leader mode).

|                | Delivered (GatekeeperAgent → deliver) | Fallback (GatekeeperAgent → fallback) |
|----------------|---------------------------------------|---------------------------------------|
| In-domain (14) | Target: ≥11/14 per leader (≥78%)      | ≤3/14                                 |
| OOD (6)        | Target: 0/6 per leader (no halluc.)   | Target: 6/6 per leader (100%)         |

**Day-11 acceptance criteria** (per ADR-015):

- **E2 target**: in-domain deliver rate ≥55%, OOD fallback = 100%, zero hallucinations on category-5 queries
- **E1 floor**: in-domain deliver rate ≥39% (no regression from v2 baseline measured Day 8)
- If E2 met → ship P6
- If between E1 and E2 → judgment call, document deliver rate as the system's operating point
- If below E1 → architecture has regressed, do not ship

### §2.2 Style Learning (StyleProfileBuilder — Component)

| Metric | Target |
|--------|--------|
| Style features extracted per email | 15 (11 base + 4 LKML-specific) |
| Within-leader style profile self-similarity | >0.70 (per ADR-003 calibration) |
| Email count contributing to each profile | ≥200 per leader |
| Cross-leader profile distinctness | Reported (not pass/fail) |
| Incremental learning | Working (alpha-weighted updates) |
| Dual profile generation | Both Torvalds and Kroah-Hartman produced |

### §2.3 Knowledge Retrieval (Retriever — Component)

| Metric | Target |
|--------|--------|
| FAISS index size | >900 chunks (actual: ~6,700) |
| Top-1 Cohere relevance on in-domain queries | >0.30 minimum gate; mean 0.85+ expected |
| Top-1 Cohere relevance on OOD queries | <0.10 |
| Citation coverage on delivered responses | 100% |
| Query latency (cold) | <1s |
| Query latency (warm) | <200ms |
| Indexing time (full corpus) | <5 minutes |

### §2.4 Quality Evaluation (EvaluatorAgent — Hybrid Agent)

Quantitative scores are computed by ScoringEngine and reported by EvaluatorAgent. They are informational metrics for GatekeeperAgent's reasoning and for system-level analysis; not pass/fail gates.

| Metric | Informational Target |
|--------|---------------------|
| Style score (cosine) distribution mean | 0.80–0.90 |
| Groundedness score distribution mean | 0.55–0.70 |
| Confidence score distribution mean | 0.75–0.90 |
| Explanation present per evaluation | Required (per C7) |
| Latency (score + LLM explanation) | <2s per response |

### §2.5 Routing Decision (GatekeeperAgent — Agent)

| Metric | Target |
|--------|--------|
| Routing reasoning attached to each decision | Required, human-readable |
| Determinism at temperature=0 | Same inputs → same decision |
| Hallucination detection on category-5 OOD queries | 100% correctly routed to fallback |
| Latency (LLM reasoning over inputs) | <2s per decision |

### §2.6 Fallback Generation (FallbackAgent — Agent)

| Metric | Target |
|--------|--------|
| Contextual acknowledgment per fallback | Generated in leader voice |
| Suggested in-domain redirections | 2-3 when adjacent topics exist |
| Calendar mock + 3 time slots | 100% (always generated) |
| Templated failsafe activates if LLM fails | Working |
| Latency | <2s |

### §2.7 Orchestration (DigitalCloneFlow)

| Metric | Target |
|--------|--------|
| End-to-end deliver latency | <8s |
| End-to-end fallback latency | <8s |
| Dual-leader mode with shared retrieval | Working |
| Error recovery on LLM failures | Working (templated failsafe path) |
| All 4 agents and 3 components integrated | 100% via Flow |

### §2.8 System Integration

- 5+ different query types processed end-to-end successfully
- Three-layer evaluation (unit + integration + system) all passing
- Streamlit dual-leader side-by-side display working (per C5)
- Production-quality error handling and logging

### §2.9 Test Coverage

| Target | Metric |
|--------|--------|
| Unit tests | All Agents and Components covered; no specific count required |
| Integration tests | Per-Agent contract tests with recorded LLM responses |
| End-to-end tests | Full pipeline via `cli evaluate` against v2 query set |
| Coverage | ≥90% on `src/` |
| Performance benchmarks | Per-Agent latency budgets verified; end-to-end <8s documented |

### §2.10 Visualizations (8 required)

Per §7.6 deliverables list. Each must be acceptance-tested as part of Day 13.

| # | Chart | Purpose |
|---|-------|---------|
| 1 | Dual-leader style feature radar chart | Hero visualization — overlaid profiles for both leaders |
| 2 | Routing correctness 2x2 grid | Day-11 headline visualization |
| 3 | Style score distribution histogram | Per-leader style consistency across queries |
| 4 | Groundedness score distribution histogram | RAG retrieval quality |
| 5 | Score component breakdown | Per-query stacked bars (style + ground + confidence) |
| 6 | Fallback trigger distribution | Which trigger_reason values appear most |
| 7 | End-to-end latency distribution | Performance profile, separated by deliver vs fallback path |
| 8 | Pre/post-2018 Torvalds style evolution (per C6) | Time-series with September 2018 marker |

### §2.11 What Is NOT a Pass/Fail Metric in v2

The following v1 metrics have been removed or recalibrated:

- Final score formula `0.4×style + 0.4×ground + 0.2×conf`: removed (no formula in v2)
- Final score >0.75 threshold: removed (no threshold; GatekeeperAgent reasons)
- Style score >0.90 as pass/fail: informational only
- Groundedness >0.60 as pass/fail: informational only
- Fallback rate 30-40% as pass/fail: replaced with separate in-domain and OOD targets
- End-to-end latency <1s: replaced with <8s (realistic for 4-agent pipeline)
- 23+ tests specific count: replaced with ≥90% coverage on src/

---

## §3 Tech Stack & Architecture

### §3.1 Core Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.12+ | Spec requires 3.10+; we use 3.12 for native generics |
| Multi-agent framework | CrewAI >=0.108.0 | Required by customized requirements; Flow + Agent abstractions |
| Vector search | FAISS IndexFlatIP | Exact search for <10K vectors; L2-normalized vectors |
| Primary embeddings | OpenAI text-embedding-3-small via LiteLLM | P2 grid search: 26% better than MiniLM on Recall@5 |
| Baseline embeddings | all-MiniLM-L6-v2 via SentenceTransformers | Bootcamp compliance comparison; offline option |
| Reranking | Cohere rerank-english-v3.0 | 2-stage retrieval (FAISS top-20 → Cohere top-5); P2-proven |
| LLM | OpenAI GPT-4o-mini via LiteLLM | Used by all 4 Agents; cost trivial for portfolio scope |
| Structured output | Instructor + Pydantic v2 | Portfolio standard; never raw json.loads; auto-retry |
| LLM routing | LiteLLM | Provider-agnostic; portfolio standard from P5 |
| Text splitting | LangChain text splitters | Spec-required for chunking; RecursiveCharacterTextSplitter at 500/50 |
| Data validation | Pydantic v2 | All data models; spec-required |
| Datasets | HuggingFace Datasets | Loading open-phi/textbooks corpus |
| Knowledge corpus | open-phi/textbooks (CS filter) | 5 CS textbooks, 6,713 chunks at 500/50 chunking |
| Style corpus | LKML mbox via lore.kernel.org | Filtered by `From:` header per leader |
| Web UI | Streamlit | Customization C5; side-by-side dual-leader display |
| CLI | Click | Portfolio standard from P2/P5; 5 commands |
| Testing | pytest | ≥90% coverage on `src/` |

### §3.2 Architecture at a Glance

The v2 architecture is a hexagonal arrangement: CLI and Streamlit are thin adapters; DigitalCloneFlow orchestrates 4 LLM-driven Agents calling 3 deterministic Components; external services (OpenAI, Cohere, FAISS) sit at the boundary.

Adapter rule (per ADR-008): `cli.py` and `streamlit_app.py` import only from `src/flow.py`, `src/schemas.py`, `src/config.py`, and narrow façades. Direct imports of LiteLLM, FAISS, Cohere, or OpenAI from adapter code are prohibited.

Orchestrator: `DigitalCloneFlow` in `src/flow.py` extends CrewAI `Flow[CloneState]` with `@start retrieve → @listen clone → @listen evaluate → @router route → @listen deliver | @listen fallback`.

Agents (LLM-driven, in `src/agents/`): CloneAgent (response generation), EvaluatorAgent (hybrid, calls ScoringEngine + LLM explanation), GatekeeperAgent (routing decision), FallbackAgent (contextual fallback + templated failsafe).

Components (deterministic, in `src/components/`): Retriever (FAISS + Cohere), StyleProfileBuilder (15 features), ScoringEngine (cosine math).

### §3.3 Why Agents vs Components

The Agent/Component distinction (ADR-009) reflects a real engineering principle: use LLMs where reasoning adds value, use deterministic code where computation is the work.

**An Agent (LLM-driven) is appropriate when:**
- The task requires reasoning about trade-offs (GatekeeperAgent weighing style vs groundedness)
- The task requires natural language generation (CloneAgent producing leader-voice prose)
- The task benefits from interpretation, not just computation (EvaluatorAgent explaining why scores look how they do)
- The output is qualitative (FallbackAgent crafting context-aware redirections)

**A Component (deterministic) is appropriate when:**
- The task is measurement (StyleProfileBuilder extracting 15 features per email)
- The task is search (Retriever finding nearest chunks)
- The task is mathematical (ScoringEngine computing cosine similarities)
- LLMs would degrade quality (numerical computation is one such case)

Calling deterministic Python "an agent" was the root cause of v1's gap between commitment and implementation. v2 enforces the distinction in vocabulary, file structure, and code review.

### §3.4 Why a Multi-Agent System Is the Right Pattern

A single LLM call cannot simultaneously: learn style patterns from corpus, retrieve grounded knowledge, generate styled response, evaluate response quality, reason about delivery vs fallback, and craft graceful fallback. The work is genuinely heterogeneous.

The v2 architecture splits this into 4 specialized Agents and 3 Components, each with one responsibility:
- Specialization: each Agent is tuned (via role/goal/backstory and prompts) for one job
- Testability: each Agent can be unit-tested with mocked dependencies
- Debugging: when output is wrong, the failure surface is bounded to one Agent
- Observability: per-Agent timing, prompt logs, and outputs can be inspected independently

This is the senior-engineer answer to "why not just one big prompt": specialization, testability, debuggability, observability. ADR-009 documents this reasoning.

### §3.5 Strategic Architecture Decisions Table

Compact summary of all locked decisions. Each entry references the ADR for full reasoning. Do not re-debate during implementation.

| # | Decision | Choice | Rationale | ADR |
|---|----------|--------|-----------|-----|
| D1 | Calendar integration | Mock (simulated slots, hardcoded link, 3 time strings) | Calendar booking not the learning objective; mock demonstrates the pattern | §4.10 |
| D2 | Embedding tiered approach | OpenAI text-embedding-3-small primary + MiniLM baseline | P2 evidence: 26% improvement on Recall@5; both run, comparison in iteration log | ADR-002 |
| D3 | CrewAI orchestration pattern | Flow with real Agents at each step (not Sequential, not Hierarchical) | Sequential can't branch; Hierarchical adds LLM latency for fixed-order steps; Flow provides @router branching with typed Pydantic state | ADR-001 |
| D4 | LLM for generation + evaluation | GPT-4o-mini for all 4 Agents | Single model simplifies; cost trivial at portfolio scale; Ollama deferred post-portfolio | ADR-007 |
| D5 | Data acquisition | lore.kernel.org mbox primary, FLOSSmole fallback | Full email headers; Python `mailbox` module handles parsing | §4.8 |
| D6 | Streamlit scope | Query + leader selector + scores + fallback; no admin/auth | Bounded feature set; consistent with P1-P5 Streamlit scope | (carried from v1) |
| D7 | Reranking | Cohere 2-stage (FAISS top-20 → Cohere top-5) | P2-proven ~20% lift; cost-effective at portfolio volume | ADR-002 |
| D8 | Structured output | Instructor + Pydantic v2 | Portfolio standard since P1; auto-retry on validation failure | (carried from v1) |
| D9 | LLM routing | LiteLLM | Provider-agnostic; portfolio standard from P5 | (carried from v1) |
| D10 | Chunking | 500 chars / 50 overlap (RecursiveCharacterTextSplitter) | Bootcamp spec baseline; P2 also tested semantic markdown split | (carried from v1) |
| **D11** | **Agent vs Component distinction** | **4 LLM-driven Agents + 3 deterministic Components + Flow** | **v2 vocabulary lock; Python functions not called agents; central decision** | **ADR-009** |
| **D12** | **Routing decision** | **LLM-driven GatekeeperAgent reasons over scores, response, chunks, flags** | **v1 formula produced tight ~0.20 score distribution; no threshold works; LLM reasoning more robust** | **ADR-010** |
| **D13** | **Style profile during rework** | **Frozen; re-measured Day 11; per-leader weighting only if KH deliver rate >20% below Torvalds** | **New architecture may absorb asymmetry; measure first, invest second** | **ADR-013** |
| **D14** | **Evaluation acceptance** | **E2 target ≥55% in-domain deliver; E1 floor ≥39%; OOD fallback = 100%** | **Routing correctness as headline; quantitative scores informational** | **ADR-015** |
| **D15** | **Three-layer evaluation methodology** | **Unit (continuous) + Integration (per agent) + System (`cli evaluate`)** | **Catches different failure modes at different scopes; Day-8 finding was a Layer-3 gap** | **ADR-016** |
| D16 | Style feature vector representation | Concatenated 15-dim numerical features (not LLM embeddings) | Interpretable per-feature; enables radar chart; cosine similarity measures style not topic; Schneider et al. 2016 validation | §4.6, ADR-003 |
| D17 | Dual-leader retrieval optimization | Retrieve once, style twice (shared chunks) | Factual content is leader-agnostic; halves the expensive operation; per-leader style application is cheap | §4.7, ADR-005 |

D11-D15 are new in v2; D1-D10, D16, D17 carry forward from v1 with possible light edits per the customized requirements changelog.

---

## §4 Strategic Decisions

The architectural decisions that shape v2. Each subsection references the relevant ADR. The compact summary table is in §3.5.

### §4.1 Agent vs Component Distinction (ADR-009)

**Decision:** Adopt explicit vocabulary distinguishing LLM-driven Agents (using CrewAI Agent abstraction) from deterministic Components (Python classes with `run()` method). Apply the distinction in file structure, code review, and documentation.

**Why:** v1's failure mode was conflating "Flow step" with "agent." Python functions wrapped in `@listen` decorators were called agents in the documentation, but they performed no LLM reasoning. The result was a system that claimed 5 agents but had 1. v2's vocabulary prevents this drift by name: a class with `role`, `goal`, `backstory` is an Agent; a class with `run()` and no LLM call is a Component.

**Implementation:** Agents live in `src/agents/`. Components live in `src/components/`. Code review enforces the distinction. CI grep checks for `def` functions named with "Agent" suffix that aren't CrewAI Agents (and fail the build if found).

### §4.2 LLM-Driven Routing via GatekeeperAgent (ADR-010)

**Decision:** Replace the v1 weighted formula (`0.4×style + 0.4×ground + 0.2×conf >= 0.75`) with an LLM-driven GatekeeperAgent that reasons about routing using individual scores, the response, retrieved chunks, and EvaluatorAgent's flags.

**Why:** Day 8 measured the v1 formula producing a tight ~0.20-wide score distribution across 20 queries (mean 0.66, range 0.56-0.76). At threshold 0.75, almost nothing cleared. At threshold 0.50, everything cleared. No threshold value produced a meaningful split, because the formula doesn't discriminate well across this corpus and query set. GatekeeperAgent can reason about trade-offs the formula can't: high style + low groundedness signals dangerous confidence; the response contradicts the most relevant chunk; topic mismatch even with reasonable scores.

**Cost:** ~1-2s of LLM latency per routing decision. End-to-end latency budget accommodates this (<8s total).

**Risk:** LLM reasoning has variance. Mitigated by setting temperature=0, providing structured output via Instructor, and writing the prompt to demand explicit reasoning that references specific scores and flags.

### §4.3 EvaluatorAgent Hybrid Design (ADR-011)

**Decision:** EvaluatorAgent is a hybrid: it delegates deterministic scoring to ScoringEngine Component and adds LLM reasoning to generate a human-readable explanation and flag specific issues. The Agent is real (LLM-driven for the explanation), but the scoring math stays out of the LLM path.

**Why:** Numerical scoring is bad LLM territory (LLMs hallucinate numbers). But explanation generation, flag identification, and interpretation are good LLM territory. Splitting these inside one Agent gives the right tool for each part of the job. EvaluatorAgent's output is `EvaluationResult` with three scores (from ScoringEngine), an LLM-generated explanation, and a list of LLM-identified flags.

**Note:** The scoring formula `final = 0.4×style + 0.4×ground + 0.2×conf` is gone in v2. `EvaluationResult` has no `final_score` field. GatekeeperAgent reasons over individual scores.

### §4.4 LLM-Driven FallbackAgent with Templated Failsafe (ADR-012)

**Decision:** FallbackAgent is LLM-driven (real CrewAI Agent) generating context-aware acknowledgments in leader voice, suggested in-domain redirections, and calendar booking mock. A templated failsafe path activates if the LLM call fails (5-line try/except).

**Why:** With 30-70% of queries expected to route to fallback (depending on corpus alignment), the fallback path is not an edge case — it's the dominant path. A system where the dominant path is mechanical templated text is barely multi-agent. LLM-driven fallback produces leader-appropriate tone, context-aware redirections, and a materially better user experience.

**Cost:** ~1-2s latency on fallback path. Templated failsafe avoids cascading failure if LLM is down.

### §4.5 CrewAI Flow with Real Agents at Each Step (ADR-001 rewritten)

**Problem:** CrewAI offers three orchestration approaches: Sequential Crews, Hierarchical Crews, and Flows. Which fits the digital clone workflow, and how do the v2 4 Agents wire into it?

**Analysis of the three orchestration patterns:**

| Option | How it works | P6 v2 fit? | Why not? |
|--------|--------------|------------|----------|
| Sequential Crew | Tasks execute in order, output feeds the next | Partial | No built-in conditional branching. Can't skip FallbackAgent when GatekeeperAgent decides deliver. Would need hacky no-op tasks. |
| Hierarchical Crew | Manager Agent delegates tasks to workers | No | Manager Agent would make every step decision via LLM (1-2s each). The pipeline order is fixed; runtime LLM reasoning over order is wasteful. Also documented as fragile in production (TDS Nov 2025). |
| Flow with real Agents | Event-driven orchestration via `@start`, `@listen`, `@router` decorators. State managed via Pydantic. Each step calls a real Agent or Component. | Yes | Native conditional branching via `@router`. Typed state via `Flow[CloneState]`. Real Agents at each step where LLM reasoning adds value. Components at each step where it doesn't. |

**Decision:** Flow as the deterministic orchestrator backbone. The Flow IS the orchestration. Each Flow step calls either a real CrewAI Agent (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent) or a real Component (Retriever). There is no separate PlannerAgent in v2.

**Difference from v1:** v1's ADR-001 reached the same conclusion (Flow over Sequential/Hierarchical) but the Flow steps were Python functions. v2's Flow steps call real Agents. The Flow shell is the same; what runs at each step is fundamentally different. This is the central architectural correction in v2.

**Why this is a strong portfolio signal:** "I evaluated CrewAI's three orchestration patterns. Sequential couldn't branch. Hierarchical adds LLM latency for fixed-order steps. Flow with real Agents at each step is CrewAI's recommended production pattern, and it preserves the separation between orchestration (Flow's job) and agency (each Agent's job). This is the conceptual distinction in multi-agent systems most candidates miss."

### §4.6 Style Feature Vector — Concatenated Numerical Features, Not LLM Embeddings (ADR-003)

**Problem:** How to represent a person's "style" as a comparable vector? Two approaches: (a) extract numerical features and concatenate into a vector, or (b) pass writing samples through an embedding model and average the embeddings.

**Decision:** Concatenated 15-dim numerical features (approach a). 11 base features + 4 LKML-specific features. See §5.3 for the full feature list.

**Rationale:**
- **Interpretability per feature:** "Torvalds uses 2x more dashes than Kroah-Hartman" is a concrete claim. LLM embeddings are opaque.
- **Radar chart visualization:** numerical features map directly to radar chart axes. Embedding vectors don't have human-readable axes.
- **Style vs topic discrimination:** cosine similarity on feature vectors measures style similarity in terms you can explain. Embedding similarity measures *semantic* similarity, which is the wrong thing — two people can discuss the same topic in completely different styles, and you want the feature vector to discriminate the style. ADR-003 documents this specifically with measurements showing LLM embeddings collapse "writes like Torvalds" and "writes about kernels" into the same direction.
- **Academic precedent:** Schneider et al. 2016 (OpenSym) extracted style features on this exact data and validated the approach. Building on a published methodology beats inventing one.

**Implementation:** Each feature is a float normalized to [0, 1]. The style vector is `np.ndarray` of length 15. Style score = cosine similarity between the leader's profile vector and the response's feature vector (computed by ScoringEngine Component).

**Trade-off accepted:** Hand-crafted features only capture patterns we explicitly code. Patterns we didn't anticipate are missed. The trade-off is interpretability + discriminability vs latent-feature coverage. For an interview-grade portfolio project where I need to *explain* what style is, interpretability wins.

### §4.7 Dual-Leader Mode — Shared RAG Retrieval, Per-Leader Style Application (ADR-005)

**Problem:** Running two leaders on the same query doubles compute. The cheap parts can run twice; the expensive parts should not. Where to share?

**Decision:** Retrieval runs ONCE. Style application (CloneAgent), evaluation (EvaluatorAgent), routing (GatekeeperAgent), and fallback (FallbackAgent if triggered) run per leader.

**Why:** The factual content retrieved from the knowledge base is the same regardless of who's "speaking." Only the style wrapper changes. This halves the most expensive operation in the pipeline (query embedding + FAISS top-20 + Cohere rerank, ~600ms per call), enabling dual-leader mode to stay under the latency budget.

**Implementation:**

```python
# WHY: retrieve once, run rest of pipeline per leader
# compare_leaders() in src/flow.py orchestrates the two Flow runs
def compare_leaders(query: str) -> LeaderComparison:
    torvalds_profile = load_profile("torvalds")
    kh_profile = load_profile("kroah_hartman")
    
    # First run: full pipeline for Torvalds (including retrieve)
    torvalds_flow = DigitalCloneFlow()
    torvalds_state = torvalds_flow.kickoff(inputs={
        "query": query, "leader": "torvalds", "style_profile": torvalds_profile,
    })
    shared_chunks = torvalds_state.chunks
    
    # Second run: skip retrieve via injected chunks
    kh_flow = DigitalCloneFlow()
    kh_state = kh_flow.kickoff(inputs={
        "query": query, "leader": "kroah_hartman", "style_profile": kh_profile,
        "chunks": shared_chunks,  # Retriever's run() sees state.chunks and returns early
    })
    
    return LeaderComparison(query=query, torvalds=..., kroah_hartman=...)
```

**Asymmetric outputs are expected:** One leader may deliver while the other falls back on the same query (Day 8 measured exactly this in 3 cases). The comparison surface honestly shows both routing decisions. The Streamlit UI displays them side by side regardless of route.

**Why dual-leader matters as a portfolio signal:** If the style learning pipeline genuinely captures individual style, the two leaders' delivered responses should read recognizably differently. Same chunks. Same query. Different voice. This is the built-in validation mechanism for the style work. If both leaders produce identical responses, the style agent has failed silently, and the comparison surface makes that failure visible.

### §4.8 Email Parsing — Python mailbox Module with Cleaning Pipeline (carried from v1)

**Problem:** LKML mbox archives contain decades of emails with inconsistent formatting, embedded patches, forwarded chains, encoding issues, and contamination from quoted reply text. Raw email body cannot feed directly into feature extraction.

**Decision:** Python's built-in `mailbox.mbox()` for primary parsing. Fallback to regex extraction for malformed entries. An 8-step cleaning pipeline runs before feature extraction.

**Cleaning pipeline (executed in order):**
1. Parse mbox → extract From, To, Subject, Body, Date, Message-ID
2. Filter by `From:` header containing `torvalds@` (or `gregkh@`) — only sent messages from the target leader
3. Strip quoted text (lines starting with `>`) — we want the leader's original content only
4. Remove email signatures (text after `-- \n` or common sign-off patterns)
5. Remove embedded patches/diffs (lines starting with `+`, `-`, `@@` in contiguous blocks)
6. Remove auto-generated content (mailing list footers, unsubscribe links)
7. Filter out very short emails (<20 words after cleaning) — these are "Applied, thanks" acknowledgments with no style signal
8. Validate: each cleaned email must have sender, date, and ≥20 words of body text

**Why strip quoted text:** On LKML, replies heavily quote the previous message. If quoted text is kept in the email body, style features will be contaminated by OTHER people's writing. We need only the leader's original words.

**Why filter short emails:** "Applied, thanks." is one of Torvalds' most frequent patterns but carries no discriminative style signal. Including these would skew the average-message-length feature artificially low and the patch-language feature artificially high without representing how he writes when he actually communicates.

**v2 status:** This pipeline already exists in v1 (`src/style/email_parser.py`). The StyleProfileBuilder Component wraps it. Day 9-14 rework does NOT rebuild this; the v1 cleaning logic is preserved per ADR-013 (style profile frozen during rework).

### §4.9 Style Profile Frozen During Rework (ADR-013)

**Decision:** The StyleProfileBuilder Component is frozen during the rework (May 26-31). The Day 8 asymmetry finding (Torvalds style mean 0.91, Kroah-Hartman 0.84) is re-measured under the new architecture on Day 11. Per-leader feature weighting is deferred to Day 12+ contingency work, triggered only if KH deliver rate is >20% below Torvalds.

**Why:** The new architecture (GatekeeperAgent reasoning over individual scores instead of formula) may absorb the asymmetry. Measuring first, then deciding whether to invest in per-leader extraction, is more efficient than building per-leader features speculatively.

### §4.10 Mock Calendar Integration

**Decision:** Calendar booking is mocked (hardcoded link + 3 generated time slot strings). No real Cal.com API integration.

**Why:** Real calendar integration is not the learning objective. The mock satisfies the customized requirements (C4 — enhanced fallback) and demonstrates the booking-link generation pattern without the operational complexity of OAuth, scheduling logic, or third-party API state. Same reasoning as v1.

---

## §5 Architecture Details

### §5.1 Agent Specifications

Each Agent is a class in `src/agents/`. Each class wraps a CrewAI `Agent` instance with `role`, `goal`, `backstory`, and uses a CrewAI `Task` to execute its work via a single-agent `Crew`. Pattern is consistent across all 4 Agents.

#### §5.1.1 CloneAgent

**Location:** `src/agents/clone_agent.py`

**Role:** "Voice Emulator — produces technically accurate responses in a specific Linux kernel maintainer's writing style."

**Goal:** "Generate response prose that answers the query, uses the retrieved chunks as factual grounding, and matches the leader's measured writing style."

**Backstory:** Encodes that the leader is a real person with documented communication patterns; lists key style features from the leader's StyleProfile; reminds the LLM to maintain technical accuracy and ground claims in chunks.

**Inputs:**
- `query: str` — the user's question
- `leader: Literal["torvalds", "kroah_hartman"]`
- `style_profile: StyleProfile` — measured features + 3-5 sample emails for in-context style examples
- `chunks: list[RetrievalResult]` — top-5 retrieved chunks

**Outputs:**
- `response_text: str` — generated styled response
- `citations: list[Citation]` — which chunks were used (LLM identifies via structured output)

**Tools:** None (CloneAgent reasons directly; no external tools needed for this task)

**LLM:** GPT-4o-mini via LiteLLM, temperature=0.3 (some variability for natural-sounding output)

#### §5.1.2 EvaluatorAgent

**Location:** `src/agents/evaluator_agent.py`

**Role:** "Quality Assessor — measures response quality across style, grounding, and confidence dimensions, and explains the measurement."

**Goal:** "Score the response against the leader's style profile and retrieved chunks, identify specific quality issues, and explain the measurement in natural language."

**Backstory:** Encodes that scoring is partly deterministic (cosine math) and partly interpretive (LLM explanation). The Agent's job is to do both correctly.

**Inputs:** `query`, `response_text` (from CloneAgent), `chunks`, `style_profile`, `leader`

**What it does:**
1. Calls `ScoringEngine.score()` Component for deterministic style_score, groundedness_score, confidence_score
2. LLM reads response + chunks + scores and generates `explanation` + `flags`

**Outputs:**
- `EvaluationResult`:
  - `style_score: float` (from ScoringEngine)
  - `groundedness_score: float` (from ScoringEngine)
  - `confidence_score: float` (from ScoringEngine)
  - `explanation: str` (LLM-generated, references specific scores)
  - `flags: list[str]` (LLM-identified issues, may be empty)
  - NO `final_score` field

**LLM:** GPT-4o-mini, temperature=0

#### §5.1.3 GatekeeperAgent

**Location:** `src/agents/gatekeeper_agent.py`

**Role:** "Quality Gatekeeper — decides whether a generated response is good enough to deliver or should fall back to graceful redirection."

**Goal:** "Reason about the response, retrieved chunks, and evaluation. Decide deliver or fallback. Explain the reasoning."

**Backstory:** Encodes the trade-offs to watch for. High style + low groundedness = confident hallucination risk → fallback. Medium-everything = potentially deliverable, weigh individual signals. Low groundedness + flag indicating chunk mismatch → fallback. Permissive by default; conservative when flags indicate specific issues.

**Inputs:** `query`, `response_text`, `chunks`, `evaluation` (scores + explanation + flags from EvaluatorAgent), `leader`

**Outputs:**
- `RoutingDecision`:
  - `decision: Literal["deliver", "fallback"]`
  - `reasoning: str` (LLM-generated, references specific scores and flags)
  - `trigger_reason: Optional[str]` (set if fallback; passed to FallbackAgent)

**LLM:** GPT-4o-mini, temperature=0

**Scope in v2:** Single-query routing only. Dual-leader cross-comparison is deferred (see Post-Portfolio Followups).

#### §5.1.4 FallbackAgent

**Location:** `src/agents/fallback_agent.py`

**Role:** "Graceful Redirect — when the system can't answer well, acknowledge the limitation in the leader's voice and offer alternatives."

**Goal:** "Generate a leader-appropriate acknowledgment explaining why the system can't answer, suggest 2-3 in-domain alternatives the system can answer, and offer a calendar booking."

**Backstory:** Encodes the leader's voice for fallback (terse Torvalds vs measured Kroah-Hartman). Reminds the LLM that honesty about limitations is the point.

**Inputs:** `query`, `leader`, `trigger_reason` (from GatekeeperAgent), `style_profile`, `chunks`

**Outputs:**
- `FallbackResponse`:
  - `acknowledgment: str` (leader-voice explanation of fallback)
  - `suggested_redirections: list[str]` (2-3 alternative questions)
  - `calendar_link: str` (hardcoded mock URL)
  - `available_slots: list[str]` (3 mock time slot strings)
  - `unstyled_response: str` (generic backup if generation fails or for failsafe)

**Templated failsafe:** If the LLM call fails (timeout, rate limit, structured output validation failure), a 5-line try/except in `fallback_agent.py` returns a templated `FallbackResponse` with leader-name-substituted acknowledgment and the same calendar mock data. The system always returns a usable response.

**LLM:** GPT-4o-mini, temperature=0.3

### §5.2 Component Specifications

Each Component is a class in `src/components/` with a `run()` method. No LLM calls. Deterministic given same inputs.

#### §5.2.1 Retriever

**Location:** `src/components/retriever.py`

**Purpose:** Retrieve top-5 most relevant knowledge chunks for a given query.

**What it does:**
1. Embed the query via OpenAI text-embedding-3-small (LiteLLM)
2. Search FAISS index for top-20 candidates (cosine similarity via dot product on normalized vectors)
3. Rerank top-20 via Cohere rerank-english-v3.0 to top-5
4. If Cohere call fails, log warning and return FAISS top-5 (graceful degradation per ADR-002)

**Outputs:** `list[RetrievalResult]` (5 items: chunk + score + rank)

**Notes:** Uses the same `data/rag/faiss_index/` from v1; Cohere reranking actually executes in v2 (env var name corrected per Day 8 finding); has `build()` method for one-time index construction.

#### §5.2.2 StyleProfileBuilder

**Location:** `src/components/style_profile_builder.py`

**Purpose:** Build a StyleProfile from a leader's email corpus.

**What it does:**
1. Parse mbox using Python `mailbox` module
2. Filter messages by `From:` header matching pattern
3. Apply 8-step email cleaning pipeline (see §4.8)
4. For each cleaned email: extract 15 style features via `extract_features()`
5. Aggregate features incrementally with weighted averaging: `updated = (1-alpha)*current + alpha*new`
6. Build `StyleFeatures` object; convert to 15-dim vector via `to_vector()`
7. Construct `StyleProfile`

**Outputs:** `StyleProfile` (Pydantic model, serializable to `data/profiles/{leader}.json`)

**Notes:** Self-similarity validation spot-check >0.70 on 20-email sample after profile built; email parsing pipeline preserved from v1 per ADR-013.

#### §5.2.3 ScoringEngine

**Location:** `src/components/scoring_engine.py`

**Purpose:** Compute the three quality scores deterministically.

**What it does:**
1. **Style score:** extract features from response_text → cosine similarity vs style_profile.style_vector
2. **Groundedness score:** split response into sentences (regex), batch-embed sentences via OpenAI, for each sentence take max cosine vs chunk embeddings, average across sentences (per ADR-004)
3. **Confidence score:** heuristic combining retrieval relevance (top-1 Cohere score), response completeness (length vs expected), uncertainty markers (hedging phrase count)

**Outputs:** `tuple[float, float, float]` (style_score, groundedness_score, confidence_score)

**Notes:** Used by EvaluatorAgent; pure computation, testable with synthetic inputs; does NOT compute `final_score` — the formula is gone in v2.

### §5.3 Style Features (15 total)

Per the customized requirements (C3):

**11 Base Features:** average message length (words per email), greeting patterns (frequency dict), punctuation patterns (ellipsis/exclamation/dashes frequency), capitalization ratio, question frequency, vocabulary richness (unique/total), common phrases (top 10), reasoning patterns (logical connector usage), sentiment distribution (pos/neutral/neg ratio), formality level, technical terminology frequency.

**4 LKML-Specific Features:** code snippet frequency, quote-reply ratio, patch language patterns (Applied/NAK frequency), technical depth indicator (technical/general vocab ratio).

`StyleFeatures.to_vector()` converts the 15 features into a 15-dim numpy array for cosine similarity. Categorical features reduced to scalars (frequency of dominant pattern); list features reduced to counts or set-overlap-with-baseline scalars.

### §5.4 Pydantic Data Models

All models in `src/schemas.py`. Pydantic v2. Key models:

```python
class EmailMessage(BaseModel):
    sender: str; recipients: list[str]; subject: str; body: str
    timestamp: datetime; message_id: str; is_patch: bool; quote_ratio: float

class StyleFeatures(BaseModel):
    # 11 base + 4 LKML-specific features
    def to_vector(self) -> np.ndarray: ...

class StyleProfile(BaseModel):
    leader_name: str; features: StyleFeatures; style_vector: list[float]  # 15-dim
    email_count: int; last_updated: datetime; alpha: float

class KnowledgeChunk(BaseModel):
    content: str; source_topic: str; source_field: str; chunk_index: int
    embedding: list[float] | None

class RetrievalResult(BaseModel):
    chunk: KnowledgeChunk; score: float; rank: int

class Citation(BaseModel):
    chunk_id: str; source_topic: str; text_snippet: str; relevance_score: float

class EvaluationResult(BaseModel):
    style_score: float; groundedness_score: float; confidence_score: float
    explanation: str; flags: list[str]
    # NO final_score in v2

class RoutingDecision(BaseModel):
    decision: Literal["deliver", "fallback"]; reasoning: str; trigger_reason: str | None

class FallbackResponse(BaseModel):
    acknowledgment: str; suggested_redirections: list[str]
    calendar_link: str; available_slots: list[str]; unstyled_response: str

class StyledResponse(BaseModel):
    query: str; leader: str; response: str; evaluation: EvaluationResult
    citations: list[Citation]; routing_decision: RoutingDecision

class LeaderComparison(BaseModel):
    query: str; torvalds: StyledResponse | FallbackResponse
    kroah_hartman: StyledResponse | FallbackResponse

class CloneState(BaseModel):
    """CrewAI Flow typed state, populated incrementally."""
    query: str; leader: Literal["torvalds", "kroah_hartman"]
    style_profile: StyleProfile | None = None
    chunks: list[RetrievalResult] = []
    response_text: str | None = None; citations: list[Citation] = []
    evaluation: EvaluationResult | None = None
    routing_decision: RoutingDecision | None = None
    styled_response: StyledResponse | None = None
    fallback_response: FallbackResponse | None = None
```

### §5.5 Flow Orchestration

`DigitalCloneFlow` in `src/flow.py` extends `Flow[CloneState]`. Pattern:

```python
class DigitalCloneFlow(Flow[CloneState]):
    @start()
    def retrieve(self):
        if self.state.chunks: return  # dual-leader: chunks already injected
        self.state.chunks = Retriever().run(self.state.query)
    
    @listen(retrieve)
    def clone(self):
        result = CloneAgent(leader=self.state.leader).run(
            query=self.state.query, leader=self.state.leader,
            style_profile=self.state.style_profile, chunks=self.state.chunks)
        self.state.response_text = result.response_text
        self.state.citations = result.citations
    
    @listen(clone)
    def evaluate(self):
        self.state.evaluation = EvaluatorAgent().run(
            query=self.state.query, response_text=self.state.response_text,
            chunks=self.state.chunks, style_profile=self.state.style_profile,
            leader=self.state.leader)
    
    @router(evaluate)
    def route(self) -> str:
        self.state.routing_decision = GatekeeperAgent().run(
            query=self.state.query, response_text=self.state.response_text,
            chunks=self.state.chunks, evaluation=self.state.evaluation,
            leader=self.state.leader)
        return self.state.routing_decision.decision  # "deliver" or "fallback"
    
    @listen("deliver")
    def deliver(self):
        self.state.styled_response = StyledResponse(...)
    
    @listen("fallback")
    def fallback(self):
        self.state.fallback_response = FallbackAgent().run(...)
```

The Flow IS the orchestration. There is no PlannerAgent class.

---

## §6 Pipeline Behavior

### §6.1 Single-Query Deliver Path

`cli query "How does kernel scheduling work?" --leader torvalds`:

1. CLI calls `DigitalCloneFlow.kickoff(inputs={...})`
2. `@start retrieve`: Retriever embeds query, FAISS top-20, Cohere rerank top-5 → state.chunks
3. `@listen clone`: CloneAgent generates Torvalds-style response → state.response_text, state.citations
4. `@listen evaluate`: EvaluatorAgent calls ScoringEngine + LLM explanation → state.evaluation
5. `@router route`: GatekeeperAgent reasons about delivery → state.routing_decision (decision="deliver")
6. `@listen deliver`: Package state into StyledResponse → state.styled_response
7. CLI prints StyledResponse with score breakdown, citations, evaluation explanation

End-to-end latency: <8s target. Per-step budgets: retrieve <1s, clone <3s, evaluate <2s, route <2s.

### §6.2 Single-Query Fallback Path

Same as deliver through step 5. At step 5, GatekeeperAgent decides "fallback". Then:

6. `@listen fallback`: FallbackAgent generates leader-voice acknowledgment, redirections, calendar mock → state.fallback_response
7. CLI prints FallbackResponse

If FallbackAgent's LLM call fails, the templated failsafe returns a pre-canned FallbackResponse. End-to-end latency: <8s.

### §6.3 Dual-Leader Comparison

`cli compare "How does kernel scheduling work?"` — see §4.7 for the shared-retrieval pattern. CLI prints side-by-side comparison; Streamlit shows the same in two columns. Asymmetric routing (one delivers, one falls back) is expected and displayed faithfully.

### §6.4 Offline Pipelines

**`cli learn`:** Build StyleProfile per leader — parse mbox → filter by `From:` → apply cleaning pipeline (§4.8) → extract features per email → save StyleProfile JSON.

**`cli index`:** Build FAISS index — load open-phi/textbooks (CS filter) → chunk via RecursiveCharacterTextSplitter (500/50) → embed via OpenAI (cached) → build FAISS IndexFlatIP → save.

These commands run once during setup.

### §6.5 Error Handling

| Failure mode | Behavior |
|--------------|----------|
| Retriever Cohere fails | Log warning, return FAISS top-5 (graceful degradation per ADR-002) |
| Retriever returns 0 chunks | EvaluatorAgent scores low groundedness, GatekeeperAgent routes to fallback |
| CloneAgent LLM fails | Flow raises; CLI catches and prints user-friendly error |
| EvaluatorAgent LLM fails (explanation) | Scores still computed; explanation set to "Evaluation failed to generate." Decision unaffected. |
| GatekeeperAgent LLM fails | Flow defaults to fallback (conservative); logs the failure |
| FallbackAgent LLM fails | Templated failsafe activates; returns pre-canned FallbackResponse |
| Style profile not found | CLI prints error: "Run `cli learn --leader <name>` first." |
| FAISS index not found | CLI prints error: "Run `cli index` first." |

---

## §7 Deliverables

### §7.1 Code Deliverables

```
src/
├── cli.py, config.py, flow.py, schemas.py, visualization.py
├── agents/
│   ├── clone_agent.py, evaluator_agent.py
│   ├── gatekeeper_agent.py, fallback_agent.py
└── components/
    ├── retriever.py, style_profile_builder.py, scoring_engine.py

streamlit_app.py
```

Lower-level modules (`src/rag/`, `src/style/`) are refactored into Component wrappers during the rework. Architectural rule: nothing outside `src/agents/` and `src/components/` is named like an Agent.

### §7.2 Test Deliverables

```
tests/
├── unit/        # one file per Agent and Component
├── integration/ # per-Agent contract tests with recorded LLM responses
└── e2e/
    └── test_cli_evaluate.py
```

Coverage target ≥90% on `src/`. LLM responses recorded for replay in CI.

**Test focus per Agent and Component:**

| Agent / Component | Key Files | Test Focus |
|-------------------|-----------|------------|
| CloneAgent | `src/agents/clone_agent.py` | Output is styled response + citations; LLM prompt includes leader's style features and chunks; structured output via Instructor parses correctly |
| EvaluatorAgent | `src/agents/evaluator_agent.py` | Hybrid contract: calls ScoringEngine; LLM generates explanation referencing scores; `flags` list correctly populated; NO `final_score` field |
| GatekeeperAgent | `src/agents/gatekeeper_agent.py` | RoutingDecision has decision + reasoning; reasoning references specific scores and flags; deterministic at temperature=0; routes category-5 OOD queries to fallback (hallucination detection) |
| FallbackAgent | `src/agents/fallback_agent.py` | Leader-voice acknowledgment generated; suggested redirections present when adjacent chunks exist; calendar mock always present; templated failsafe activates on LLM failure |
| Retriever | `src/components/retriever.py` | FAISS top-20 → Cohere top-5; Cohere fallback to FAISS top-5 if rerank fails; Cohere reranking actually executes (no env var bug); citation coverage |
| StyleProfileBuilder | `src/components/style_profile_builder.py` | mbox parses; `From:` filtering correct; 8-step cleaning pipeline executes; 15 features extracted per email; self-similarity >0.70 on 20-email sample |
| ScoringEngine | `src/components/scoring_engine.py` | Style cosine math correct; groundedness sentence-level math correct; confidence heuristic returns float in [0,1]; edge cases (empty response, no chunks) handled |
| DigitalCloneFlow | `src/flow.py` | All 5 steps execute in order; @router branches correctly on deliver vs fallback; CloneState typed throughout; compare_leaders() shares chunks correctly |

### §7.3 CLI Commands (5 total)

The CLI uses Click. Five commands, unchanged from v1. Internal implementations refactored for v2 Agent/Component architecture.

| Command | Purpose | Outputs |
|---------|---------|---------|
| `python -m src.cli learn --leader torvalds --mbox data/raw/lkml.mbox` | Build StyleProfile via StyleProfileBuilder Component | `data/profiles/torvalds.json` |
| `python -m src.cli index` | Build FAISS index via Retriever Component's `build()` method | `data/rag/faiss_index/` |
| `python -m src.cli query "What is TCP/IP?" --leader torvalds` | Run a single query through DigitalCloneFlow; print StyledResponse or FallbackResponse with score breakdown | Terminal output: response, scores, citations, evaluation explanation, routing reasoning |
| `python -m src.cli compare "What is TCP/IP?"` | Run dual-leader comparison via `compare_leaders()`; print side-by-side LeaderComparison | Terminal output: both leaders with their evaluations and routing decisions |
| `python -m src.cli evaluate --query-set data/eval/queries.json --output results/evaluation_dayN.json` | Run the v2 evaluation query set (20 queries × 2 leaders = 40 records) through the full pipeline; write results JSON | `results/evaluation_dayN.json` with all scores, routing decisions, latencies |

All commands enforce the ADR-008 hexagonal adapter boundary: `cli.py` imports only from `src/flow.py`, `src/schemas.py`, `src/config.py`. No direct LiteLLM/FAISS/Cohere imports.

### §7.4 Streamlit App (streamlit_app.py)

Single-file Streamlit app at the repo root. Imports only from `src.flow`, `src.schemas`, `src.config` (ADR-008 boundary).

**In-scope features:**

- **Text input** — single text field for the user's question
- **Leader selector** — dropdown with three options: "Torvalds", "Kroah-Hartman", "Compare Both"
- **Response display** — single response (single leader mode) or side-by-side columns (compare mode)
- **Score breakdown per response** — style_score, groundedness_score, confidence_score as bars or numbers (no final_score, per v2 ADR-010)
- **Routing reasoning** — display GatekeeperAgent's `reasoning` text for each response so the user sees WHY the system delivered vs fell back
- **Evaluation explanation** — display EvaluatorAgent's `explanation` text and any `flags`
- **Citations** — list of source topics + text snippets for delivered responses
- **Fallback display** — when GatekeeperAgent routes to fallback, show FallbackAgent's acknowledgment, suggested redirections, calendar link, and 3 time slots
- **Pre-generated visualizations** — show the 8 portfolio charts (per §2.10) as static PNGs in a separate tab or expandable section

**Out of scope:**

- No admin panel
- No authentication
- No real-time style profile training (profiles are loaded from disk)
- No email upload UI
- No chat history persistence across page reloads
- No user accounts

**Streamlit caching:** `@st.cache_resource` deferred to post-portfolio per the Post-Portfolio Followups page. v2 demo runs sequential queries; reload cost (~3-5s) is accepted.

### §7.5 Documentation Deliverables

#### §7.5.1 Top-Level Documentation

| File | Content | Authored |
|------|---------|----------|
| `docs/PRD.md` | This document — v2 implementation contract | Day 9 (this doc) |
| `docs/CLAUDE.md` | Implementation conventions and Engineering Protocols for Claude Code | Day 9 |
| `docs/codebase-audit.md` | Reusable audit-checklist template introduced Day 14; per §12.5; six categories of grep-based verification | Day 14 |
| `docs/day8-findings.md` | Historical engineering record of Day 8 verification gaps that triggered the v2 rework. Carries forward unchanged. | Pre-existing |
| `docs/day11-evaluation.md` | Post-rework evaluation report. Contains the 2x2 routing-correctness grid (Day-11 headline), per-leader deliver/fallback breakdowns, three-run variance analysis at temperature=0, comparison vs Day-8 v2 baseline, regression-anchor (q12, q13) check, and PRD §2 scorecard. | Day 12 |
| `docs/evaluation-methodology.md` | Three-layer evaluation approach per ADR-016: unit (continuous CI), integration (per-Agent contract tests with recorded LLM responses), system (`cli evaluate` against v2 query set). Documents regression detection methodology and the role of each layer. | Day 12 |
| `docs/eval-query-set-spec.md` | Methodology behind v2 query set design (14 in-domain + 6 OOD = 20 queries × 2 leaders). Five corpus-alignment categories, regression anchors (q12 binary search, q13 stacks/queues), category-5 hallucination probes. Carries forward from Day 8. | Pre-existing |
| `README.md` | Written fresh at Day 14 wrap. Gold standard per P1/P2 inverted pyramid pattern: visual proof above the fold, narrative-first results, inline screenshots, scannable engineering signals. | Day 14 |

#### §7.5.2 ADR Inventory (16 ADRs in `docs/adr/`)

8 light-edited from v1, 8 new in v2. Each ADR follows the 5-section format (Context, Decision, Alternatives Considered, Quantified Validation, Consequences) per the project ADR standard.

| ADR | Title | v2 Status |
|-----|-------|-----------|
| ADR-001 | CrewAI Flow with Real Agents at Each Step | Rewritten — v1 Flow shell preserved; what runs at each step changed from Python functions to real Agents |
| ADR-002 | RAG Configuration: Embeddings, Reranking, Chunking | Light edit — Cohere env var correction note (CO_API_KEY → COHERE_API_KEY) |
| ADR-003 | Hand-Crafted Feature Vectors over LLM Embeddings | Light edit — naming aligned with v2 vocabulary |
| ADR-004 | Groundedness Scoring via Cosine Similarity | Light edit — scope clarified (used by ScoringEngine Component, not LLM) |
| ADR-005 | Shared RAG Retrieval for Dual-Leader Mode | Light edit — Agent names updated (CloneAgent, not ChatStyleAgent) |
| ADR-006 | Day 6 Methodology and Corpus-Shape Limits | Light edit — Day 8 confirmation of corpus-shape findings; §10 notes which experiments are obsolete |
| ADR-007 | LLM Roles in the Pipeline | Rewritten — supersedes v1's LLM-scoring-viability ADR. Documents where LLMs are used (4 Agents) and where they are deliberately not (3 Components) |
| ADR-008 | Hexagonal Adapters for CLI and Streamlit | Kept as-is — boundary rule unchanged; CI grep checks added |
| ADR-009 | Agent vs Component Distinction | New — central v2 vocabulary decision; criteria for what is an Agent vs a Component |
| ADR-010 | LLM-Driven Routing via GatekeeperAgent | New — documents removal of weighted formula and 0.75 threshold; tight ~0.20 score distribution evidence from Day 8 |
| ADR-011 | EvaluatorAgent Hybrid Design | New — scoring delegated to ScoringEngine Component; LLM generates explanation and flags |
| ADR-012 | LLM-Driven FallbackAgent with Templated Failsafe | New — fallback is dominant path (30-70% of queries) so it earns a real Agent; 5-line try/except failsafe |
| ADR-013 | Style Profile Frozen, Re-Measured Day 11 | New — freeze during rework; per-leader weighting only if KH deliver >20% below Torvalds |
| ADR-014 | Agent and Component Inventory | New — explicit list of 4 Agents + 3 Components + Flow; documents why no PlannerAgent in v2 |
| ADR-015 | Post-Rework Evaluation Acceptance Criteria | New — E2 target ≥55% in-domain deliver, E1 floor ≥39%, OOD fallback = 100% |
| ADR-016 | Evaluation Methodology — Three-Layer Approach | New — locks unit/integration/system structure and role of each layer |

#### §7.5.3 Architecture Diagrams (6 in `docs/architecture/`)

All diagrams rendered as Mermaid in markdown for native GitHub rendering. PNG exports for README embedding generated at Day 13.

| # | File | Type | Purpose | Generated |
|---|------|------|---------|-----------|
| A1 | `A1-system-architecture.md` | Mermaid `graph TB` | High-level system: Adapters → Flow → 4 Agents + 3 Components → External Services. README hero diagram. | Day 13 |
| A2 | `A2-single-query-sequence.md` | Mermaid `sequenceDiagram` | Single-query path: User → Flow → Retriever → CloneAgent → EvaluatorAgent → GatekeeperAgent → deliver branch OR FallbackAgent. Shows @router conditional branching. | Day 13 |
| A3 | `A3-dual-leader-sequence.md` | Mermaid `sequenceDiagram` | Dual-leader compare_leaders() pattern: shared retrieval, then per-leader CloneAgent → EvaluatorAgent → GatekeeperAgent. Highlights retrieve-once optimization (ADR-005). | Day 13 |
| A4 | `A4-data-models.md` | Mermaid `classDiagram` | Pydantic models: EmailMessage, StyleFeatures, StyleProfile, KnowledgeChunk, RetrievalResult, Citation, EvaluationResult, RoutingDecision, FallbackResponse, StyledResponse, LeaderComparison, CloneState. Shows composition. | Day 13 |
| A5 | `A5-data-flow.md` | Mermaid `graph LR` | Two swim lanes: Offline (style learning + RAG indexing) and Online (per-query Flow execution). Shows what runs once vs per-query. | Day 13 |
| A6 | `A6-agent-vs-component.md` | Mermaid `graph TB` | NEW in v2: Visual of the 4 Agents + 3 Components distinction (ADR-009). Left column LLM-driven Agents; right column deterministic Components; clear separation. | Day 13 |

#### §7.5.4 Session Notes (`docs/session-notes/`)

Per-day implementation notes per Verification Protocol Component 6. One file per day: `day9.md` through `day14.md`. `day14.md` includes the codebase audit results per §12.5.

### §7.6 Visualization Deliverables

8 portfolio charts in `results/charts/` (per §2.10):

1. `01-style-radar-dual-leader.png` — Hero: 15 features for both leaders overlaid
2. `02-routing-correctness-grid.png` — Day-11 headline 2x2 grid
3. `03-style-score-distribution.png` — Histogram per leader
4. `04-groundedness-score-distribution.png` — Histogram per leader
5. `05-score-component-breakdown.png` — Per-query stacked bars (style/ground/confidence)
6. `06-fallback-trigger-distribution.png` — Bar chart of trigger_reason values
7. `07-latency-distribution.png` — Histogram with deliver vs fallback separated
8. `08-torvalds-style-evolution-pre-post-2018.png` — Time-series with September 2018 marker (C6)

All charts: Matplotlib/Seaborn/Plotly, colorblind-friendly palettes (`Set2`, `tab10`), >=10pt fonts, descriptive titles.

### §7.7 Demo Deliverables

- Streamlit app (per §7.4)
- CLI (per §7.3)
- Screen recording (~3-5 min, MP4) showing a query trace through the 4 agents

### §7.8 Iteration Logs (Already Completed)

v1 produced an iteration log (`results/iteration_log.json`). v2 inherits it as historical record. v2 does NOT require new iteration log entries during the rework. The Day-11 evaluation report (`docs/day11-evaluation.md`) is the v2-equivalent: captures the new architecture's measured performance with comparison to v1 baseline. If post-rework tuning produces measurable changes, those go in `docs/day11-evaluation.md` as labeled re-runs, not in a separate iteration log.

---

## §8 Session Plan

Rework runs Day 9 through Day 14 (May 26-31). Each day ends with the Verification Protocol check against the relevant PRD section, per the Engineering Protocols documented in CLAUDE.md.

### Day 9 (May 26) — Foundation

**Scope:**
- Branch creation: `refactor/p6-multi-agent-rework` off `main`
- Cleanup deletions per Day-9-morning task list (see §12.1)
- Rename `data/eval/queries_v2.json` → `data/eval/queries.json`
- Write this PRD v2 (locked Day 9 morning)
- Write CLAUDE.md v2
- Write new ADRs (009, 010, 014, 015, 016)
- Light edits to ADRs 001-008

**End-of-day check:** All docs in place; ADR-009 vocabulary defined; branch exists, cleanup committed.

### Day 10 (May 27) — Components and First Two Agents

**Scope:**
- Build/refactor 3 Components (Retriever, StyleProfileBuilder preserving §4.8 pipeline, ScoringEngine)
- Build 2 Agents (CloneAgent, EvaluatorAgent hybrid)
- Unit tests for all Components and these 2 Agents

**End-of-day check:** Components instantiate and `run()` successfully; CloneAgent generates plausible Torvalds-style response; EvaluatorAgent produces EvaluationResult; unit tests pass.

### Day 11 (May 28) — Remaining Agents and Integration

**Scope:**
- Build GatekeeperAgent and FallbackAgent (with templated failsafe)
- Update CloneState schema with `routing_decision` field
- Refactor `src/flow.py` DigitalCloneFlow to call Agents at each step
- Update `compare_leaders()` for shared retrieval (preserve v1 ADR-005 pattern per §4.7)
- Integration tests per Agent
- End-of-day smoke test: run one query end-to-end

**End-of-day check:** One query traces successfully through all 5 Flow steps; each Agent's output is the expected Pydantic type; architecture honesty check passes (4 Agents with role/goal/backstory, 3 Components with `run()`).

### Day 12 (May 29) — End-to-End and Evaluation

**Scope:**
- Run end-to-end smoke test on diverse queries
- Run `cli evaluate` against v2 query set (`data/eval/queries.json`, 20 queries)
- Generate `docs/day11-evaluation.md` with 2x2 routing-correctness grid, per-leader breakdowns, regression anchor check, PRD scorecard
- Decision gate: E2 target hit? E1 floor hit? Decide ship/no-ship per ADR-015

**End-of-day check:** Day-11 evaluation report complete with real numbers; decision gate evaluated and recorded.

### Day 13 (May 30) — Polish and Documentation

**Scope:**
- Redraw 6 architecture diagrams (A1-A6) reflecting v2 architecture
- Generate 8 visualization charts in `results/charts/` (per §2.10 and §7.6)
- Streamlit polish: verify side-by-side display, score breakdowns, fallback rendering
- Run full test suite, fix any failures
- Update CLAUDE.md if implementation revealed any conventions to document

**End-of-day check:** All architecture diagrams reflect v2; all 8 charts generated; Streamlit demo runs cleanly; all tests pass, coverage ≥90% on src/.

### Day 14 (May 31) — Wrap

**Scope:**
- Write README from scratch (gold standard per P1/P2 inverted pyramid pattern)
- **Codebase audit (per §12.5):** run the v2 architecture verification checklist; document results in `docs/session-notes/day14.md`; resolve each finding (delete, fix, or defer to Post-Portfolio Followups). Create `docs/codebase-audit.md` as the reusable template for P7/P8/P9 and P1-P5 re-verification.
- Record screen recording demo (3-5 min)
- Final integration check: clone repo from scratch, follow README, verify everything runs
- Update Notion Project Tracker entry for P6 (Complete=true, all properties)
- Hand-off to P7 (June 1 start)

**End-of-day check:** README is gold standard; codebase audit complete with documented findings; screen recording saved; all success criteria from §2 verified with evidence; Notion Project Tracker updated; P6 v2 closed; P7 ready to start.

---

## §9 Risk and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GatekeeperAgent's LLM reasoning is inconsistent (different decisions on same input) | Medium | High (routing instability) | Temperature=0; structured output via Instructor; prompt demands explicit reasoning referencing specific scores |
| Day-11 eval shows in-domain deliver rate below E1 floor (39%) | Low | High (architecture regressed) | Stop-condition; investigate before declaring complete; possibly revert to v1 routing as diagnostic |
| Per-leader style asymmetry persists despite new architecture | Medium | Medium (KH deliver rate < Torvalds) | Day-12+ contingency per ADR-013: per-leader feature weighting (1 day) if KH deliver >20% below Torvalds |
| Agent prompt drift between Days 10-13 produces inconsistent outputs | Medium | Medium (regression on regression anchors) | Regression anchors (q12, q13) in eval set; re-run end-to-end at end of Days 11, 12, 13; flag any drift |
| Refactoring rag/ and style/ modules into Component wrappers breaks existing tests | Medium | Medium (debug time during rework) | Keep underlying logic intact; Components are thin wrappers; existing unit tests should mostly pass |
| Day-11 evaluation reveals corpus is too narrow (in-domain deliver capped at ~50%) | Low | Medium (E2 target unattainable) | Operating-point characterization story; ship at E1 floor; document corpus as bounded scope |
| Cohere rate limit hits during eval runs | Low | Medium (silent degradation if not caught) | Throttle wrapper from Day 8 carried forward; watch for Cohere fallback warnings in logs |
| Implementation drifts from PRD/CLAUDE.md during 5-day rework | Medium | High (re-introduces v1's failure mode) | Daily Verification Protocol with architecture honesty check (per ADR-009); CI grep checks for Agent/Component vocabulary; Day 14 codebase audit (§12.5) |
| Email cleaning pipeline regression during StyleProfileBuilder refactor | Low | Medium (style profiles change unexpectedly) | Style profile frozen during rework per ADR-013; if rebuild needed, validate self-similarity >0.70 on sample |

---

## §10 Experiment Plan — Already Completed

v1 ran four experiments on Day 6 (April 2026). Their results are part of the project's engineering record. v2 does NOT re-run these; findings carry forward where still valid, and obsoleted where the new architecture supersedes them.

### §10.1 Embedding Comparison (Completed — Still Valid)

OpenAI text-embedding-3-small materially better than MiniLM on this corpus, consistent with P2. Result in iteration log and ADR-002. Retriever uses OpenAI as primary.

### §10.2 Chunking Comparison (Completed — Still Valid)

Light differentiation between fixed (500/50) and semantic markdown split; not enough to drive a change. Locked on 500/50. RecursiveCharacterTextSplitter remains the chunking strategy.

### §10.3 Scoring Weight Sensitivity (Obsolete)

Pinned at proxy-regime artifact (style scores ~0.50 across all queries because queries used as proxy for responses; weight changes didn't discriminate). **Obsolete:** the new architecture has no weighted formula. GatekeeperAgent reasons over individual scores.

### §10.4 Pre/Post-2018 Style Evolution (Completed — Carry Forward as Visualization)

No detectable shift in Torvalds' style at per-email resolution at September 2018. Documented as a measured null result. **Carry forward as deliverable:** chart #8 (§2.10, §7.6) is still produced. The null result itself is a portfolio signal — "I measured for a hypothesized behavioral change and reported the null result honestly."

### §10.5 LLM Scoring Viability (Obsolete per new ADR-007)

GPT-4o-mini correlated Pearson 0.82 with cosine baseline on groundedness scoring. **Obsolete:** v2 doesn't use LLMs for numerical scoring at all. ScoringEngine handles scoring deterministically. ADR-007 rewritten to document the new pattern.

### §10.6 New Experiments in v2

The only new "experiment" is the Day-11 evaluation itself (`docs/day11-evaluation.md`), which is the system test against the v2 query set. No exploratory experiments planned during the rework. Post-rework, if asymmetry persists (ADR-013), per-leader feature weighting becomes a Day 12+ scoped experiment.

---

## §11 Interview Talking Points

Reference for portfolio walkthroughs. Each talking point is a question + answer + evidence.

### §11.1 "Have you built a multi-agent system?"

**Answer:** P6 v2: 4 specialized LLM-driven Agents (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent) coordinated through CrewAI Flow with deterministic routing, supported by 3 Components (Retriever, StyleProfileBuilder, ScoringEngine) that handle the measurable parts of the pipeline. Event-driven workflow with `@router` conditional branching at the GatekeeperAgent. Each Agent independently testable with mocked dependencies. Dual-leader comparison mode proves the style learning captures genuine individual patterns.

**Evidence:** 6 Mermaid architecture diagrams (A1-A6); src/agents/ with 4 files; src/components/ with 3 files; src/flow.py with DigitalCloneFlow.

### §11.2 "What's the difference between an Agent and a Component in your system?"

**Answer:** Agents are LLM-driven reasoning work using the CrewAI Agent abstraction with `role`, `goal`, `backstory`. They live in `src/agents/`. Components are deterministic Python classes with a `run()` method that do measurement, search, or math without LLM calls. They live in `src/components/`. The distinction matters because conflating them was the failure mode in my v1 — Python functions wrapped in Flow decorators were called "agents" in documentation but performed no LLM reasoning. v2 enforces the vocabulary in code, file structure, and CI checks.

**Evidence:** ADR-009; ADR-014; docs/day8-findings.md.

### §11.3 "How did you make data-driven decisions across projects?"

**Answer:** P2 benchmarked 16 RAG configurations and found OpenAI embeddings beat MiniLM by 26% on Recall@5 and Cohere reranking added ~20% lift. P6 carries that evidence forward — the Retriever Component uses OpenAI primary, MiniLM baseline, Cohere reranking. Each architectural decision in P6 has an ADR with the reasoning and the data point that backs it. When v1's measurements revealed the architecture wasn't doing what I claimed it did (Day 8), I diagnosed the root causes, documented the findings honestly in day8-findings.md, and re-derived v2 from first principles.

**Evidence:** ADR-002; docs/day8-findings.md; customized-requirements-v2 changelog C9-C12.

### §11.4 "How do you handle AI system failures?"

**Answer:** Two layers. First, the GatekeeperAgent reasons about whether each response is good enough to deliver. If not, the system routes to FallbackAgent which produces a context-aware acknowledgment in the leader's voice plus suggested in-domain redirections. Second, the FallbackAgent itself has a templated failsafe — if its LLM call fails, a hardcoded template returns a usable response. The architecture has no path that leaves the user without a response.

**Evidence:** ADR-010; ADR-012; src/agents/fallback_agent.py.

### §11.5 "What's the most interesting thing you found?"

**Answer:** On Day 8, I ran end-to-end verification before declaring P6 complete and discovered three things: only 1 of 5 "agents" was a real CrewAI Agent; Cohere reranking had been silently failing since Day 3 due to an env var name bug; and the weighted scoring formula produced a tight ~0.20-wide score distribution where no threshold produced meaningful routing. The architecture documentation said one thing, the implementation did another. I documented the findings honestly, re-derived the architecture from first principles, and rebuilt as v2. The verification gap that surfaced this became Component 6 of my Verification Protocol — a permanent change to how I work with Claude Code.

**Evidence:** docs/day8-findings.md; the v1→v2 rework; Engineering Protocols Notion page.

### §11.6 "How does this connect to your portfolio?"

**Answer:** P2 measured RAG configurations. P5 built a production RAG system from first principles. P6 embedded RAG into a multi-agent system — the same FAISS + Cohere reranking pattern appears in all three, improving each time. P6 v2 is also the project where I formalized the Verification Protocol's architecture-honesty check, which now applies retroactively to P1-P5 during re-verification and forward to P7-P9.

**Evidence:** The progression itself; portfolio README; ADR cross-references.

### §11.7 "Why CrewAI over LangChain agents?"

**Answer:** The customized requirements locked CrewAI. But the interesting choice within CrewAI was the orchestration pattern. CrewAI offers Sequential, Hierarchical, and Flow. Sequential can't branch conditionally, so it can't handle the deliver/fallback split. Hierarchical adds LLM latency on every step decision when the order is fixed — wasteful for a deterministic pipeline. Flow with `@router` is the production pattern: deterministic step ordering, typed state via Pydantic, conditional branching via the router decorator. My v2 architecture uses Flow as the shell calling real CrewAI Agents at each step where LLM reasoning adds value.

**Evidence:** ADR-001 (rewritten); src/flow.py.

### §11.8 "Why clone Torvalds instead of a generic employee?"

**Answer:** Torvalds has the most distinctive communication style in open source software. There's a published paper (Schneider et al. 2016, OpenSym) that validated style feature extraction on exactly this data. And the 2018 behavioral change provided ground truth for testing whether my style pipeline could detect a real shift. I measured pre/post-September-2018 and reported a null result — the per-email resolution wasn't enough to detect the shift. That null result is itself a portfolio signal: I measured for a hypothesized change and reported honestly when I didn't find it.

**Evidence:** ADR-006; chart 8; customized-requirements doc C6.

---

## §12 Appendices

### §12.1 Cleanup List (Day 9 Morning)

Files to delete before rework begins:

```
docs/plans/day8-plan.md
results/evaluation_threshold_diagnostic.json
results/evaluation_cohere_fixed.json
results/evaluation_v1_cohere_fixed.json
results/evaluation_final.json
docs/P6-PRD.md                          (old PRD v1)
docs/CLAUDE.md                          (old CLAUDE.md v1)
docs/architecture/system-architecture.md (old A1)
docs/architecture/single-query-sequence.md (old A2)
docs/architecture/dual-leader-sequence.md  (old A3)
docs/architecture/data-models.md           (old A4)
docs/architecture/data-flow.md             (old A5)
```

Files to rename: `data/eval/queries_v2.json` → `data/eval/queries.json`

Files to keep: `docs/day8-findings.md`, `docs/eval-query-set-spec.md`, `data/rag/`, `data/profiles/`, `scripts/anchor_trajectory.py`, `scripts/inspect_q12_chunks.py`, `results/iteration_log.json` (v1 historical record), `src/`, `tests/` (refactored during rework).

### §12.2 Mapping of v1 Code to v2 Architecture

| v1 location | v2 location | Status |
|-------------|-------------|--------|
| `src/agents/rag_agent.py` (façade) | `src/components/retriever.py` | Renamed and reclassified as Component |
| `src/agents/style_crew.py` (real Agent) | `src/agents/clone_agent.py` | Renamed; stays Agent |
| `src/agents/evaluator_steps.py` (Python functions) | `src/agents/evaluator_agent.py` | Rewritten as real CrewAI Agent (hybrid) |
| `src/agents/fallback_steps.py` (Python functions) | `src/agents/fallback_agent.py` | Rewritten as real CrewAI Agent + templated failsafe |
| Flow router (Python `@router` returning string) | `src/agents/gatekeeper_agent.py` + Flow @router calling it | Routing decision becomes a real Agent |
| `src/rag/*.py` | Used by `src/components/retriever.py` | Stays as low-level modules; wrapped by Component |
| `src/style/*.py` | Used by `src/components/style_profile_builder.py` | Stays as low-level modules; wrapped. Email cleaning pipeline (§4.8) preserved. |
| `src/evaluation/*.py` | Used by `src/components/scoring_engine.py` | Scoring math wrapped by Component |
| `src/fallback/*.py` | Used by `src/agents/fallback_agent.py` | Helper modules; templated failsafe uses them |

### §12.3 ADR Inventory (16 ADRs)

| ADR | Title | Status in v2 |
|-----|-------|--------------|
| ADR-001 | CrewAI Flow with Real Agents at Each Step | Rewritten in place |
| ADR-002 | RAG Configuration: Embeddings, Reranking, Chunking | Light edit (Cohere correction note) |
| ADR-003 | Hand-Crafted Feature Vectors over LLM Embeddings | Light edit (naming) |
| ADR-004 | Groundedness Scoring via Cosine Similarity | Light edit (scope clarified) |
| ADR-005 | Shared RAG Retrieval for Dual-Leader Mode | Light edit (agent names) |
| ADR-006 | Day 6 Methodology and Corpus-Shape Limits | Light edit (Day 8 confirmation) |
| ADR-007 | LLM Roles in the Pipeline | Rewritten in place |
| ADR-008 | Hexagonal Adapters for CLI and Streamlit | Kept as-is |
| ADR-009 | Agent vs Component Distinction | New |
| ADR-010 | LLM-Driven Routing via GatekeeperAgent | New |
| ADR-011 | EvaluatorAgent Hybrid Design | New |
| ADR-012 | LLM-Driven FallbackAgent with Templated Failsafe | New |
| ADR-013 | Style Profile Asymmetry — Frozen, Re-Measured Day 11 | New |
| ADR-014 | Agent and Component Inventory | New |
| ADR-015 | Post-Rework Evaluation Acceptance Criteria | New |
| ADR-016 | Evaluation Methodology — Three-Layer Approach | New |

### §12.4 Customized Requirements Traceability

Every v2 PRD section traces to one or more sections of the customized requirements doc. Full traceability matrix lives as a separate Notion page. Summary:

| PRD Section | Customized Req §  | Notes |
|-------------|-------------------|-------|
| §2 Success Criteria | Customized Req §Success Metrics (v2) | v2 redefined per C11 |
| §3 Tech Stack | Customized Req §Technical Requirements | Same stack |
| §4 Strategic Decisions | Customized Req §Architectural Decisions + ADRs | Decisions delegated to ADRs |
| §5 Architecture Details | Customized Req §System Architecture Overview (v2) | 4 Agents + 3 Components |
| §6 Pipeline Behavior | Customized Req §System Architecture Pipeline Flow | Same pipeline shape |
| §7 Deliverables | Customized Req §Deliverables (v2) | CLI commands + Streamlit features detailed |
| §8 Session Plan | Customized Req §Getting Started Hints (v2) | Day 9-14 |
| §10 Experiment Plan | Customized Req §Iteration Logs (carried as completed work) | Day 6 experiments not re-run |
| §11 Interview Talking Points | (Portfolio narrative, no direct req mapping) | EM-interview-ready answers |

### §12.5 Codebase Audit Checklist

The Day 14 codebase audit verifies that no v1 residue remains in the shipped artifact. The framing is verification, not cleanup: the audit either passes (nothing found) or surfaces specific items, each resolved by one of three actions — delete, fix, or defer to Post-Portfolio Followups.

**Template location:** `docs/codebase-audit.md` (created Day 14, reusable across P7/P8/P9 and P1-P5 re-verification). Each run captures its findings in `docs/session-notes/dayN.md`.

**Time budget:** 1-1.5 hours done properly. The grep checks are fast; reading findings and deciding actions takes time. Refactoring is out of scope — if dead code is complex, delete it; don't refactor the live code.

**Six audit categories:**

**1. Dead code from old architecture.**
- Run `uv run vulture src/ --min-confidence 80` to find unused functions, classes, imports
- Run `uv run pytest --cov=src --cov-report=term-missing` and inspect uncovered branches; uncovered = potentially unused, investigate
- Decision rule: if a Python file in `src/` is no longer imported anywhere, delete it. If a function/class is defined but not called, delete it.

**2. Dead documentation references.**
- `grep -rn "rag_agent\|style_crew\|evaluator_steps\|fallback_steps" docs/ --include="*.md"` — references to v1 file paths
- `grep -rn "ADR-009.*threshold\|ADR-009.*0\.75" docs/` — references to the obsoleted ADR-009 (threshold)
- Inspect `docs/architecture/*.md` for old agent names
- Decision rule: if a doc reference points to a v1 file that no longer exists or a v1 concept (0.75 threshold, weighted formula, 5 agents), fix the doc.

**3. v1 vocabulary leaks.**
- `grep -rn "ChatStyleAgent\|RAGAgent\|PlannerAgent\|evaluator_steps\|fallback_steps" src/ --include="*.py"` — old class/file names in code
- `grep -rn "final_score" src/ --include="*.py"` — should not appear in v2 outside deprecation comments
- `grep -rn "0\.75" src/ --include="*.py"` — old threshold (false positives possible; review each)
- `grep -rn "weighted.*formula\|0\.4.*style\|0\.4.*ground" src/ --include="*.py"` — v1 routing formula
- Decision rule: all findings are bugs. Fix them.

**4. Orphaned data files.**
- `ls -la data/cache/` — embedding caches the new pipeline doesn't use
- `ls -la results/` — v1-era evaluation JSONs that survived Day 9 cleanup
- `ls -la data/` recursively — any directory the new pipeline doesn't read or write
- Decision rule: if the new pipeline never reads or writes a data file, delete it. Keep `results/iteration_log.json` (v1 historical record per §12.1).

**5. Stale comments and docstrings.**
- `grep -rn "# .*final_score\|# .*threshold\|# .*0\.75\|# .*5.*agents" src/` — comments referencing v1 concepts
- `grep -rn '""".*final_score\|""".*threshold' src/` — docstrings referencing v1 concepts
- `grep -rn "TODO\|FIXME\|XXX" src/` — outstanding TODOs (each one decided: resolved, kept with date, or moved to Post-Portfolio Followups)
- Decision rule: fix the comment to match current code, or delete if no longer relevant.

**6. Unused dependencies.**
- Inspect `pyproject.toml` — for each dependency, `grep -rn "^import {name}\|^from {name}" src/ tests/` to confirm it's imported
- Decision rule: if a package is in `pyproject.toml` but not imported anywhere, remove from `pyproject.toml` and re-run `uv sync` to verify the project still builds.

**Audit output format (in `docs/session-notes/day14.md`):**

For each category, document: (a) the grep/tool command run, (b) the raw output captured, (c) for each finding, the decision made (delete/fix/defer) and reasoning. Empty output is data — record "no findings" with the command that produced it. Clean output is the proof, not the absence of bad output.

**Reusability across projects:**

`docs/codebase-audit.md` is the durable template. Each project's Day-N wrap copies the template, customizes the grep patterns for that project's vocabulary, runs the audit, and saves results in that project's `docs/session-notes/`. For P7/P8/P9, the v2-vocabulary greps will be replaced with project-specific patterns (e.g., P7's vocabulary may be different). For P1-P5 re-verification (June 10-14), the same audit runs against each project's main branch with its own vocabulary.

---

*End of PRD v2. Lives at `docs/PRD.md` in the repo. Supersedes v1.*

*Deferred items tracked on [Post-Portfolio Followups](https://www.notion.so/36cdb630640a812a9d99d79951011897), not in this PRD.*
