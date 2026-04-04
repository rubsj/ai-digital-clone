# PRD: P6 — Torvalds Digital Clone (Multi-Agent System)

> **This is the product requirements and architecture contract.**
> Claude Code: read this + CLAUDE.md before starting. Use this for WHAT to build and WHY.
> Claude Code Opus handles implementation planning. Do NOT re-debate architecture decisions — they are final.

**Project:** P6 — Torvalds Digital Clone: Multi-Agent Style-Matching System
**Timeline:** 8 sessions (~32h total), revised plan with learn day + experiment day prioritizing depth
**Owner:** Developer (Java/TS background, completed P1–P5)
**Source of Truth:** [Notion Customized Requirements](https://www.notion.so/336db630640a81f2882bcbdf53723796)
**Original Bootcamp Spec:** [Notion Original Requirements](https://www.notion.so/335db630640a816680d4f12d00e14afd)
**PRD Version:** v1

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1 | Apr 2026 | Initial PRD. Domain customized from Enron to LKML/Torvalds. All bootcamp technical requirements preserved. Added dual-leader comparison, LKML-specific style features, Streamlit UI, pre/post-2018 style evolution, confidence explanations, P2/P5-informed RAG config. |

---

## 1. Objective

Build a **multi-agent system** that creates digital clones of two real open-source technical leaders (Linus Torvalds and Greg Kroah-Hartman), capable of mimicking their communication styles while providing accurate, knowledge-grounded responses. The system learns style from real LKML email archives, retrieves factual knowledge from a CS textbook corpus, evaluates response quality across three dimensions, and gracefully handles uncertainty through intelligent fallback.

**The Portfolio Narrative:** P2 benchmarked RAG configurations (16 configs, 3 embedding models). P5 built a production RAG system from first principles. P6 takes RAG into a multi-agent architecture — the same retrieval pipeline now serves as one agent among five, orchestrated by CrewAI. The progression: "I measured RAG, built RAG, then embedded RAG into a multi-agent system with quality control and graceful degradation."

**The output is a COMPLETE MULTI-AGENT SYSTEM with EXPLAINABLE DECISIONS.** Every response must be scored, explained, and either delivered with confidence or gracefully deferred. The dual-leader comparison proves the style learning pipeline captures genuine individual patterns, not generic text generation.

**Why Torvalds/LKML over Enron:** Enron is the "TODO app" of NLP datasets. Torvalds has the most recognizable communication style in tech, with an academically validated style feature dataset (Schneider et al. 2016, OpenSym). The 2018 behavioral change provides natural ground truth for validating style detection sensitivity. This is an EM-relevant narrative about measurable culture change, not just a technical exercise.

---

## 2. Success Criteria

> These are the hard targets from the customized requirements document. The project is not complete until all are met.

### 2a. Style Learning Targets (ChatStyleAgent)

| Metric | Target |
|--------|--------|
| Style Score (cosine similarity) | > 0.90 |
| Emails processed per leader | >= 200 |
| Style features extracted | 15+ (11 base + 4 LKML-specific) |
| Incremental learning | Working (alpha-weighted updates) |
| Dual profiles distinct | Torvalds and Kroah-Hartman produce visually different radar charts |

### 2b. Knowledge Retrieval Targets (RAGAgent)

| Metric | Target |
|--------|--------|
| Knowledge base size | > 900 chunks |
| Groundedness score | > 0.60 |
| Citation coverage | 100% of responses |
| Query speed | < 1s |
| Indexing time | < 5 minutes |

### 2c. Quality Evaluation Targets (EvaluatorAgent)

| Metric | Target |
|--------|--------|
| Final Score (weighted) | > 0.75 on in-domain queries |
| Style Score | > 0.90 |
| Groundedness Score | > 0.60 |
| Confidence Score | > 0.80 |
| Evaluation speed | < 0.5s |
| Confidence explanation | Human-readable string per evaluation |

**Formula:** `FinalScore = 0.4 × StyleScore + 0.4 × GroundednessScore + 0.2 × ConfidenceScore`

### 2d. Fallback Targets (FallbackAgent)

| Metric | Target |
|--------|--------|
| Fallback rate | 30-40% across diverse queries |
| Trigger accuracy | > 90% |
| Calendar booking generation | 100% |
| Unstyled fallback option | Working |

### 2e. Orchestration Targets (PlannerAgent)

| Metric | Target |
|--------|--------|
| End-to-end latency | < 1s average |
| System reliability | > 95% |
| Agent coordination | 100% (all 4 agent steps + Flow orchestration) |
| Error recovery | Graceful fallback on component failure |
| Dual-leader mode | Working |

### 2f. Test Coverage

| Target | Metric |
|--------|--------|
| Unit tests | 23+ tests covering all agents |
| Integration tests | Multi-agent workflow end-to-end |
| End-to-end tests | Real queries through full pipeline |
| Performance benchmarks | < 1s average latency documented |

### 2g. Visualizations (7 required)

1. Dual-leader style feature radar chart (hero visualization)
2. Style score distribution histogram
3. Groundedness score distribution histogram
4. Final score breakdown chart with 0.75 threshold line
5. Fallback rate by query type bar chart
6. End-to-end latency distribution histogram
7. Pre/post-2018 Torvalds style evolution time-series chart

---

## 3. Architecture Decisions (FINAL)

> These are locked from the requirements page. Claude Code must implement as specified — do not re-evaluate or propose alternatives.

### 3a. Core Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.12+ | Spec requires 3.10+. We use 3.12 for native generics. |
| Multi-agent | CrewAI | Spec-required. Role-based agents with defined tasks. |
| Vector search | FAISS IndexFlatIP | Spec-required. Exact search for < 1K vectors. Consistent with P2/P5. |
| Embeddings (primary) | OpenAI text-embedding-3-small via LiteLLM | P2 grid search: 26% better than MiniLM on Recall@5. Default config. |
| Embeddings (baseline) | all-MiniLM-L6-v2 via SentenceTransformers | Bootcamp compliance. Documented comparison in iteration log. |
| Reranking | Cohere Rerank API (top-20 → top-5) | P2: ~20% lift at $0.05/168 reranks. No longer optional. |
| LLM | OpenAI GPT-4o-mini via LiteLLM | Generation + evaluation scoring. Single model to start. |
| Structured output | Instructor + Pydantic v2 | Portfolio standard. Never raw `json.loads`. Auto-retry on validation failure. |
| LLM routing | LiteLLM | Portfolio standard from P5. Provider-agnostic wrapper. |
| Text splitting | LangChain text splitters | Spec-required for chunking utilities. |
| Data validation | Pydantic v2 | Spec-required. All data models. |
| Web UI | Streamlit | Customization C5. Side-by-side dual-leader display. |
| CLI | Click | Portfolio standard from P2/P5. |
| Testing | pytest | 23+ tests required. |
| Datasets | HuggingFace Datasets | Loading open-phi/textbooks corpus. |

### 3b. Strategic Architecture Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Calendar integration | Mock (simulated slots) | Calendar booking not the learning objective. Mocked slots demonstrate the pattern. |
| D2 | Embedding tiered approach | OpenAI primary + MiniLM baseline | P2 evidence: 26% improvement. Both run, comparison in iteration log. |
| D3 | CrewAI orchestration pattern | **Flow + single-agent Crew for style generation** | See Decision 1 below. Flow provides conditional branching, typed state, dual-leader loop. Avoids broken hierarchical process. CrewAI's recommended production pattern. |
| D4 | LLM for generation + eval | GPT-4o-mini for both | Simplifies initial implementation. Local vs API is Day 6 experiment candidate. |
| D5 | Data acquisition | lore.kernel.org mbox (primary), FLOSSmole (fallback) | Full email headers, proper format. Python `mailbox` module handles parsing. |
| D6 | Streamlit scope | Query + leader selector + scores + fallback. No admin/auth. | Bounded feature set. Consistent with P1-P5 Streamlit scope. |
| D7 | Reranking | Cohere 2-stage (top-20 → top-5) | P2 proven. Cost-effective. |
| D8 | Structured output | Instructor + Pydantic | Portfolio standard since P1. |
| D9 | LLM routing | LiteLLM | Portfolio standard from P5. |
| D10 | Chunking | 500 chars / 50 overlap (baseline) + semantic markdown split (experiment) | Bootcamp spec baseline + P2 best config experiment. |

---

## 4. Finalized Design Decisions

> These are engineering problems that required careful design. Each decision below is FINAL.

### Decision 1: CrewAI Orchestration — Flow + Agent Steps (Not Sequential Crew, Not Hierarchical)

**Problem:** CrewAI offers three orchestration approaches: sequential Crews, hierarchical Crews, and Flows. Which one fits the digital clone workflow?

**Analysis of all three options:**

| Option | How it works | P6 fit? | Why not? |
|--------|-------------|---------|----------|
| **Sequential Crew** | Tasks execute in order, output of one feeds the next | Partial | No built-in conditional branching. Can't skip FallbackAgent when score >= 0.75. Would need hacky no-op tasks. Dual-leader requires running entire Crew twice including redundant RAG retrieval. |
| **Hierarchical Crew** | Manager agent delegates tasks to workers | No | Documented as broken in production (Towards Data Science, Nov 2025). Manager loops instead of delegating. Incorrect agent invocation, inflated latency/token usage. CrewAI community confirms. |
| **Flow + agent steps** | Event-driven orchestration with `@start`, `@listen`, `@router` decorators. State managed via Pydantic model. | Yes | Native conditional branching via `@router`. Typed state via `FlowState`. Dual-leader mode via looping with shared state. CrewAI's own recommended production pattern (blog Dec 2025, DocuSign case study). |

**Decision:** Flow as the deterministic backbone, with individual agent functions as Flow steps. ChatStyleAgent uses a single-agent CrewAI Crew for LLM-powered style generation (where CrewAI's Agent abstraction adds value: role, goal, backstory, retry). Other agents are direct function calls within Flow steps (no Crew overhead needed for deterministic computation).

**Architecture:**
```python
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel

class CloneState(BaseModel):
    """Typed state passed between all Flow steps."""
    query: str = ""
    leader: str = ""
    retrieved_chunks: list = []
    styled_response: str = ""
    evaluation: dict = {}
    final_output: dict = {}

class DigitalCloneFlow(Flow[CloneState]):

    @start()
    def retrieve_knowledge(self):
        """RAGAgent step — deterministic pipeline, no Crew needed."""
        # WHY: embed query → FAISS search top-20 → Cohere rerank → top-5
        self.state.retrieved_chunks = rag_retrieve(self.state.query)

    @listen(retrieve_knowledge)
    def apply_style(self):
        """ChatStyleAgent step — single-agent Crew for LLM generation."""
        # WHY: This IS where CrewAI's Agent abstraction adds value.
        # The style agent needs role/goal/backstory for the LLM prompt.
        style_crew = create_style_crew(self.state.leader)
        result = style_crew.kickoff(inputs={
            "chunks": self.state.retrieved_chunks,
            "query": self.state.query,
            "profile": load_profile(self.state.leader)
        })
        self.state.styled_response = result.raw

    @listen(apply_style)
    @router()
    def evaluate_response(self) -> str:
        """EvaluatorAgent step — computation + explanation string."""
        # WHY: Mostly math (cosine similarity, weighted formula).
        # One LLM call for explanation string via Instructor.
        eval_result = evaluate(
            self.state.styled_response,
            self.state.retrieved_chunks,
            self.state.leader
        )
        self.state.evaluation = eval_result
        if eval_result["final_score"] >= 0.75:
            return "deliver"   # Routes to @listen("deliver")
        return "fallback"      # Routes to @listen("fallback")

    @listen("deliver")
    def deliver_response(self):
        """Return the styled, scored response."""
        self.state.final_output = build_styled_response(self.state)

    @listen("fallback")
    def handle_fallback(self):
        """FallbackAgent step — booking + unstyled option."""
        self.state.final_output = build_fallback_response(self.state)
```

**Dual-leader mode:** The Flow runs twice — once with `state.leader = "torvalds"`, once with `state.leader = "kroah_hartman"`. The `retrieve_knowledge` step is called only on the first run; on the second run, the shared `retrieved_chunks` are reused. This is managed by a wrapper function, not by the Flow itself.

**The PlannerAgent IS the Flow class.** It's not a separate 5th agent. It's `DigitalCloneFlow` — the class that defines the step order, state management, conditional routing, and dual-leader orchestration. This matches CrewAI's own architecture: "Flows are the project plan that coordinates multiple teams."

**Why this is the strongest interview signal:**
- "I evaluated CrewAI's three process patterns. Sequential couldn't branch. Hierarchical is broken in production. I chose Flows — CrewAI's recommended production pattern — with a single-agent Crew for the style generation step where LLM agency adds value, and direct function calls for deterministic steps like retrieval and evaluation."
- This demonstrates understanding of the DIFFERENCE between orchestration (Flow) and agency (Crew/Agent), which is the key conceptual distinction in multi-agent systems.

### Decision 2: Style Feature Vector — Concatenated Numerical Features, Not LLM Embeddings

**Problem:** How to represent a person's "style" as a comparable vector? Two approaches: (a) extract numerical features and concatenate into a vector, or (b) pass writing samples through an embedding model and average the embeddings.

**Decision:** Concatenated numerical features (approach a).

**Rationale:**
- The 15 features are individually interpretable — you can say "Torvalds uses 2x more dashes than Kroah-Hartman." LLM embeddings are opaque.
- Feature vectors enable the radar chart visualization directly. Embedding vectors don't map to human-readable axes.
- Cosine similarity on feature vectors measures style similarity in terms you can explain. Embedding similarity measures semantic similarity, which is not what we want — two people can discuss the same topic in completely different styles.
- This is what the Schneider et al. 2016 paper did, providing academic validation.

**Implementation:** Each feature is a float in [0, 1] (normalized). The style vector is `np.array` of length 15. Style score = cosine similarity between the leader's profile vector and the response's feature vector.

### Decision 3: Dual-Leader Mode — Shared RAG, Separate Style Application

**Problem:** Running two leaders on the same query doubles compute. Where to share?

**Decision:** RAG retrieval runs ONCE. Style application + evaluation runs per leader.

**Why:** The factual content retrieved from the knowledge base is the same regardless of who's "speaking." Only the style wrapper changes. This halves the most expensive operation (embedding + FAISS search + Cohere rerank).

**Implementation:**
```python
# WHY: retrieve once, style twice — RAG is the expensive part
chunks = rag_agent.retrieve(query)  # One call
torvalds_response = style_agent.apply(chunks, query, profile="torvalds")
kroah_hartman_response = style_agent.apply(chunks, query, profile="kroah_hartman")
torvalds_eval = evaluator_agent.score(torvalds_response, chunks, "torvalds")
kroah_hartman_eval = evaluator_agent.score(kroah_hartman_response, chunks, "kroah_hartman")
```

### Decision 4: Groundedness Scoring — Chunk Coverage Heuristic, Not LLM Judge

**Problem:** How to score whether a response is grounded in the retrieved chunks? Two approaches: (a) LLM-as-judge comparing response to chunks, or (b) heuristic measuring token overlap between response claims and chunk content.

**Decision:** Hybrid — semantic similarity between response sentences and retrieved chunks (using the embedding model), with an LLM judge as calibration.

**Rationale:**
- Pure token overlap misses paraphrasing (response says "three-step process" while chunk says "3-phase approach")
- Pure LLM judge is expensive for every evaluation call and adds latency
- Sentence-level cosine similarity between response segments and top chunks is fast (embeddings already computed) and captures paraphrasing
- LLM judge runs on a sample of 5 responses to calibrate the heuristic threshold

### Decision 5: Email Parsing — Python mailbox Module with Fallback

**Problem:** LKML mbox archives contain decades of emails with inconsistent formatting, embedded patches, forwarded chains, and encoding issues.

**Decision:** Python's built-in `mailbox.mbox()` for primary parsing. Fallback to regex extraction for malformed entries.

**Cleaning pipeline:**
1. Parse mbox → extract From, To, Subject, Body, Date, Message-ID
2. Filter by `From:` containing `torvalds@` or `gregkh@`
3. Strip quoted text (lines starting with `>`) — we want original content only
4. Remove email signatures (text after `-- \n` or common sign-off patterns)
5. Remove embedded patches/diffs (lines starting with `+`, `-`, `@@` in contiguous blocks)
6. Remove auto-generated content (mailing list footers, unsubscribe links)
7. Filter out very short emails (< 20 words after cleaning) — these are typically "Applied, thanks" acknowledgments that don't carry enough style signal
8. Validate: each cleaned email must have sender, date, and >= 20 words of body text

**Why strip quoted text:** On LKML, replies heavily quote the previous message. If we include quoted text, the style features would be contaminated by OTHER people's writing. We need only the leader's original words.

---

## 5. Component Architecture

### 5a. Data Models (Pydantic v2)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `EmailMessage` | Parsed LKML email | `sender: str`, `recipients: list[str]`, `subject: str`, `body: str`, `timestamp: datetime`, `message_id: str`, `is_patch: bool` |
| `StyleFeatures` | Extracted features for one email or aggregate | `avg_message_length: float`, `greeting_patterns: dict[str, float]`, `punctuation_patterns: dict[str, float]`, `capitalization_ratio: float`, `question_frequency: float`, `vocabulary_richness: float`, `common_phrases: list[str]`, `reasoning_patterns: dict[str, float]`, `sentiment_distribution: dict[str, float]`, `formality_level: float`, `technical_terminology: float`, `code_snippet_freq: float`, `quote_reply_ratio: float`, `patch_language: dict[str, float]`, `technical_depth: float` |
| `StyleProfile` | Aggregate learned profile | `leader_name: str`, `features: StyleFeatures`, `style_vector: np.ndarray`, `email_count: int`, `last_updated: datetime`, `alpha: float` |
| `KnowledgeChunk` | Textbook content unit | `content: str`, `source_topic: str`, `source_field: str`, `chunk_index: int`, `embedding: Optional[np.ndarray]` |
| `RetrievalResult` | Retrieved chunk + score | `chunk: KnowledgeChunk`, `score: float`, `rank: int` |
| `EvaluationResult` | Quality scores | `style_score: float`, `groundedness_score: float`, `confidence_score: float`, `final_score: float`, `explanation: str`, `decision: Literal["deliver", "fallback"]` |
| `FallbackResponse` | Fallback output | `trigger_reason: str`, `context_summary: str`, `calendar_link: str`, `available_slots: list[str]`, `unstyled_response: Optional[str]` |
| `StyledResponse` | Final agent output | `query: str`, `leader: str`, `response: str`, `evaluation: EvaluationResult`, `citations: list[Citation]`, `fallback: Optional[FallbackResponse]` |
| `LeaderComparison` | Side-by-side output | `query: str`, `torvalds: StyledResponse`, `kroah_hartman: StyledResponse` |
| `Citation` | Source reference | `chunk_id: str`, `source_topic: str`, `text_snippet: str`, `relevance_score: float` |

### 5b. Agent Architecture (CrewAI Flow + Crew Hybrid)

The system uses CrewAI's dual architecture: a **Flow** for deterministic orchestration and a **single-agent Crew** for LLM-powered style generation. Each "agent" maps to a Flow step with the appropriate level of autonomy.

| Agent (Conceptual) | CrewAI Implementation | Why This Level |
|---|---|---|
| **RAGAgent** | Flow step — direct function call | Deterministic pipeline (embed → FAISS → Cohere rerank). No LLM reasoning. No benefit from Crew overhead. |
| **ChatStyleAgent** | Flow step invoking a **single-agent Crew** | This IS where CrewAI's Agent abstraction adds value. The style agent needs role/goal/backstory to prompt the LLM for style-matched generation. Crew gives retry + structured output. |
| **EvaluatorAgent** | Flow step — computation + single Instructor LLM call | Mostly math (cosine similarity, weighted formula). One LLM call for explanation string. No autonomy needed. |
| **FallbackAgent** | Flow step — deterministic logic | Context summarization + mock calendar slots. Straightforward function. |
| **PlannerAgent** | **The Flow class itself** (`DigitalCloneFlow`) | The Flow IS the orchestrator. It defines step order, state management, conditional routing (`@router`), and dual-leader mode. Not a separate agent. |

**State management:** `CloneState` (Pydantic BaseModel) is passed between all Flow steps automatically. Contains query, leader, retrieved_chunks, styled_response, evaluation, and final_output.

**ChatStyleAgent Crew definition:**
```python
# WHY: CrewAI Agent gives us role/goal/backstory for prompt engineering,
# plus auto-retry and structured output parsing.
style_agent = Agent(
    role="Communication style specialist for {leader_name}",
    goal="Generate a response that matches {leader_name}'s writing style "
         "while accurately conveying the retrieved knowledge",
    backstory="You are an expert at mimicking the communication patterns of "
              "technical leaders. You have studied {leader_name}'s emails "
              "and understand their sentence structure, vocabulary, tone, "
              "and distinctive phrases.",
    llm="gpt-4o-mini",  # Via LiteLLM
    verbose=True
)

style_task = Task(
    description="Given these knowledge chunks: {chunks}\n\n"
                "Answer this query in {leader_name}'s style: {query}\n\n"
                "Style profile: {profile_summary}",
    expected_output="A response written in {leader_name}'s communication style, "
                    "grounded in the provided knowledge chunks, with [N] citations.",
    agent=style_agent
)

style_crew = Crew(agents=[style_agent], tasks=[style_task])
```

**Error recovery:** If any Flow step throws an exception, the Flow catches it and routes to the fallback step. This is implemented via try/except in the Flow's step methods, NOT via CrewAI's built-in error handling (which is designed for Crew-level failures, not Flow-level).

### 5c. Pipeline Architecture

**Phase 1: Offline — Style Learning (run once per leader)**
```
LKML mbox → Parse → Filter by sender → Clean → Extract 15 features → Aggregate → Style Profile JSON
```

**Phase 2: Offline — Knowledge Indexing (run once)**
```
open-phi/textbooks (CS filter) → Chunk (500/50 or semantic) → Embed (OpenAI) → FAISS Index → Save
```

**Phase 3: Online — Query Processing (DigitalCloneFlow)**
```
User Query → Flow.start()
  → retrieve_knowledge: embed query → FAISS search top-20 → Cohere rerank → top-5 chunks
  → apply_style: chunks + query + profile → style Crew generates styled response
  → evaluate_response (@router): score style + groundedness + confidence → explanation
      → "deliver": return StyledResponse
      → "fallback": handle_fallback → return FallbackResponse (booking + unstyled option)
```

**Phase 4: Online — Dual-Leader Comparison**
```
User Query → run DigitalCloneFlow with leader="torvalds"
  → retrieve_knowledge runs (chunks stored in state)
  → apply_style → evaluate → deliver or fallback
→ run DigitalCloneFlow with leader="kroah_hartman"
  → retrieve_knowledge SKIPPED (reuse chunks from state)
  → apply_style → evaluate → deliver or fallback
→ Return LeaderComparison (side-by-side)
```

---

## 6. Experiment Plan

### 6a. Embedding Comparison (Iteration Log Entry)

Run the same 10 test queries through two RAG configurations:

| Config | Embeddings | Reranking | Chunking |
|--------|-----------|-----------|----------|
| A (default) | OpenAI text-embedding-3-small | Cohere top-20→5 | 500/50 |
| B (baseline) | all-MiniLM-L6-v2 | Cohere top-20→5 | 500/50 |

Compare: Groundedness score, retrieval latency, final score. Document delta in iteration log. Expected: Config A wins by ~20-26% based on P2 data.

### 6b. Chunking Comparison (Iteration Log Entry)

| Config | Embeddings | Chunking |
|--------|-----------|----------|
| C (bootcamp spec) | OpenAI | 500 chars / 50 overlap |
| D (P2 winner) | OpenAI | Semantic split on markdown headers |

Compare: Groundedness score, chunk relevance. Document in iteration log.

### 6c. Scoring Weight Sensitivity (Day 6 Experiment)

Run 10+ test queries with 3 weight configurations:

| Config | Style | Groundedness | Confidence | Expected Effect |
|--------|-------|-------------|------------|----------------|
| Default | 0.4 | 0.4 | 0.2 | Balanced |
| Style-heavy | 0.5 | 0.3 | 0.2 | Lower fallback rate, risk of hallucination |
| Ground-heavy | 0.3 | 0.5 | 0.2 | Higher fallback rate, more accurate |

Plot sensitivity chart. Document findings in iteration log.

### 6d. Pre/Post-2018 Style Evolution

Partition Torvalds emails by September 2018. Compute style features for each period separately. Measure:
- Sentiment distribution shift
- Capitalization ratio change
- Exclamation frequency change
- Formality level change

Visualize as time-series with vertical marker at 2018-09.

---

## 7. Deliverables

### 7a. Core Agents

| Agent | Key Files | Tests |
|-------|-----------|-------|
| ChatStyleAgent | `src/agents/style_agent.py`, `src/style/feature_extractor.py`, `src/style/profile_builder.py` | Style feature accuracy, incremental learning, dual-profile distinctness |
| RAGAgent | `src/agents/rag_agent.py`, `src/rag/chunker.py`, `src/rag/embedder.py`, `src/rag/retriever.py`, `src/rag/reranker.py` | Chunk count, retrieval speed, citation coverage |
| EvaluatorAgent | `src/agents/evaluator_agent.py`, `src/evaluation/style_scorer.py`, `src/evaluation/groundedness_scorer.py`, `src/evaluation/confidence_scorer.py` | Score ranges, formula correctness, explanation generation |
| FallbackAgent | `src/agents/fallback_agent.py`, `src/fallback/calendar_mock.py`, `src/fallback/context_summarizer.py` | Trigger conditions, booking format, unstyled response quality |
| PlannerAgent (Flow) | `src/flow.py`, `src/agents/style_crew.py` | End-to-end flow, dual-leader mode, error recovery, @router branching |

### 7b. CLI Commands

| Command | Purpose |
|---------|---------|
| `python -m src.cli learn --leader torvalds` | Build style profile from LKML emails |
| `python -m src.cli index` | Build FAISS index from textbook corpus |
| `python -m src.cli query "What is TCP/IP?" --leader torvalds` | Single query through full pipeline |
| `python -m src.cli compare "What is TCP/IP?"` | Dual-leader comparison |
| `python -m src.cli evaluate` | Run 10+ test queries, generate scores + charts |

### 7c. Streamlit App

- Text input for query
- Dropdown: "Torvalds" / "Kroah-Hartman" / "Compare Both"
- Response display (single or side-by-side)
- Score breakdown per response (style, groundedness, confidence, final)
- Confidence explanation string
- Fallback display with both options (booking + unstyled)
- Pre-generated visualizations as static images

### 7d. Visualizations (7 required + style evolution)

| # | Chart | Purpose |
|---|-------|---------|
| 1 | Dual-leader style feature radar chart | Hero visualization — overlaid profiles for both leaders |
| 2 | Style score distribution histogram | Per-leader style consistency across queries |
| 3 | Groundedness score distribution histogram | RAG retrieval quality |
| 4 | Final score breakdown (grouped bar) | Three components per query, 0.75 threshold line |
| 5 | Fallback rate by query type | Which queries the system can't handle |
| 6 | End-to-end latency distribution | Performance profile, 1s target line |
| 7 | Pre/post-2018 Torvalds style evolution | Time-series with September 2018 marker |

### 7e. Documentation

| Deliverable | Content | Day |
|-------------|---------|-----|
| README.md | Gold standard — results above fold, architecture diagram, demo link | Day 8 |
| ADR-001: CrewAI Flow vs Sequential vs Hierarchical | Why Flow pattern. Analysis of all three options. Hierarchical documented as broken. Production pattern evidence. | Day 1 |
| ADR-002: RAG Config — Embeddings, Reranking, and Chunking (P2 Evidence) | Why OpenAI over MiniLM (26% P2 delta). Why Cohere 2-stage reranking (20% lift). Why test semantic chunking. Three decisions, one ADR, all backed by P2 grid search data. | Day 3 |
| ADR-003: Feature Vectors vs LLM Embeddings for Style | Interpretability + radar chart compatibility. Schneider et al. validation. Why cosine similarity on features beats embedding similarity for style matching. | Day 2 |
| ADR-004: Groundedness Scoring — Semantic Similarity Heuristic vs LLM Judge | Why hybrid approach (fast heuristic + calibration). Alternatives: pure LLM judge (expensive), pure token overlap (misses paraphrasing), no scoring. | Day 4 |
| ADR-005: Shared RAG Retrieval for Dual-Leader Mode | Why retrieve once, style twice. Alternative: independent pipelines per leader. Latency optimization that enables the 1s target in comparison mode. | Day 5 |
| ADR-006: Local vs API LLM for Evaluation (experiment-driven) | Written only if Day 6 experiment produces interesting data. Does GPT-4o-mini quality justify cost vs local Ollama? If quality parity, recommend local for development. | Day 6 (conditional) |
| Iteration Log | Every config change with before/after metrics. >= 3 entries. | Days 4-6 |
| Learning Journal | Concept entries for multi-agent patterns, CrewAI, style transfer | Day 8 |

### 7f. Architecture Documents (Mermaid)

All diagrams produced as Mermaid in markdown files under `docs/architecture/`. Also rendered as PNG for README embedding. These are PLANNED deliverables, not afterthoughts — the multi-agent architecture is complex enough that visual documentation is essential for both interviews and maintainability.

| # | Diagram | Type | Purpose | Generated |
|---|---------|------|---------|-----------|
| A1 | System Architecture | Mermaid `graph TB` | High-level: Flow orchestrator, 4 agent steps, FAISS, Cohere, OpenAI, LKML mbox, style profiles. The README hero diagram. | Day 7 (with README) |
| A2 | Single Query Sequence | Mermaid `sequenceDiagram` | User → Flow → RAGAgent → StyleCrew → EvaluatorAgent → `@router` branch → deliver or FallbackAgent. Shows conditional branching visually. | Day 5 (after Flow integration) |
| A3 | Dual-Leader Comparison Sequence | Mermaid `sequenceDiagram` | Same as A2 but shows shared RAG retrieval, then two parallel style→evaluate→decide passes. Highlights the "retrieve once, style twice" optimization. | Day 5 |
| A4 | Data Model Relationships | Mermaid `classDiagram` | Pydantic models: EmailMessage, StyleFeatures, StyleProfile, KnowledgeChunk, RetrievalResult, EvaluationResult, StyledResponse, LeaderComparison, FallbackResponse. Shows composition and inheritance. | Day 7 |
| A5 | Offline vs Online Data Flow | Mermaid `graph LR` | Two swim lanes: Offline (style learning from mbox + knowledge indexing from textbooks) and Online (query processing through Flow). Shows what runs once vs per-query. | Day 7 |

**Why Mermaid:** Renders natively in GitHub markdown. No external tool needed. Same approach as P5. Interview signal: "I documented the multi-agent architecture with 5 Mermaid diagrams covering system overview, query lifecycle, data model relationships, and offline/online data flow."

**Directory:**
```
docs/architecture/
├── system-architecture.md          ← A1: high-level system diagram
├── single-query-sequence.md        ← A2: single leader query flow
├── dual-leader-sequence.md         ← A3: comparison mode flow
├── data-models.md                  ← A4: Pydantic model relationships
└── data-flow.md                    ← A5: offline vs online pipelines
```

### 7g. Iteration Logs

**Format:** Each entry must include: Change, Reason, Metric Before, Metric After, Delta, Keep?

**What must be logged:**
- Embedding model comparison (OpenAI vs MiniLM)
- Chunking strategy comparison (fixed vs semantic)
- Scoring weight experiments (3 configurations)
- Any threshold adjustments
- Style feature additions or removals

---

## 8. Session Plan (8 Days)

| Session | Focus | ADRs | Exit Criteria |
|---------|-------|------|---------------|
| Day 1 (~4h) | **Learn Day + Foundation.** Study CrewAI Flows (event-driven orchestration), understand `@start`/`@listen`/`@router` decorators and `FlowState`. Project setup, Pydantic schemas, email parsing pipeline. Download LKML mbox for both leaders. | ADR-001 | CrewAI Flows understood (can explain Flow vs Crew vs hierarchical tradeoffs). Schemas defined. Email parser extracts clean messages from mbox. 200+ emails per leader validated. |
| Day 2 (~4h) | **ChatStyleAgent.** Feature extractor (15 features), style profile builder, incremental learning with alpha. Build profiles for both leaders. | ADR-003 | Both style profiles generated. Radar chart shows distinct signatures. Style score > 0.90 on training data. |
| Day 3 (~6-8h) | **RAGAgent.** Textbook corpus loading (CS filter), chunking pipeline, OpenAI embeddings via LiteLLM, FAISS index, Cohere reranking, citation extraction. Also build MiniLM baseline index. | ADR-002 | 900+ chunks indexed. Retrieval returns relevant chunks in < 1s. Citations working. |
| Day 4 (~4h) | **EvaluatorAgent + FallbackAgent.** Style scorer (cosine similarity), groundedness scorer (semantic similarity heuristic), confidence scorer (with explanation string), weighted formula, threshold logic. FallbackAgent with mock calendar + unstyled response option. | ADR-004 | Evaluation pipeline complete. Fallback triggers correctly. Explanation strings generated. ADR-004 documents groundedness scoring approach. |
| Day 5 (~4h) | **Flow Orchestration + Integration.** Wire all agent steps into DigitalCloneFlow. Implement @router for deliver/fallback branching. Implement dual-leader comparison mode with shared RAG state. Error handling via try/except in Flow steps. | ADR-005 | End-to-end Flow works. Dual-leader comparison produces two scored responses. @router correctly branches on threshold. Error recovery tested. ADR-005 documents shared RAG optimization. |
| Day 6 (~4h) | **Experiment Day.** Embedding comparison (OpenAI vs MiniLM). Chunking comparison. Scoring weight sensitivity (3 configs × 10 queries). Pre/post-2018 style evolution analysis. Iteration log entries. Optional: local vs API LLM experiment for evaluation scoring. | ADR-006 (conditional) | All experiments run. Iteration log has >= 3 entries. Style evolution chart shows measurable shift. ADR-006 written if local vs API experiment produces interesting data. |
| Day 7 (~4h) | **Streamlit + CLI + Architecture Docs.** Build Streamlit app (query input, leader selector, side-by-side display, score breakdown, fallback display). Click CLI commands. All 7 visualizations generated as PNGs. Architecture diagrams A1-A5 as Mermaid markdown. | — | Streamlit demo working. CLI commands functional. All charts saved. All 5 architecture diagrams committed. |
| Day 8 (~4h) | **Documentation Sprint.** README (gold standard). Humanize all 5-6 ADRs (first-person voice, real debugging stories). Learning Journal. Concept Library entries: "Multi-Agent Topologies", "CrewAI Flows", "Style Transfer via Feature Vectors". Loom recording. Final success criteria checklist. | — | All deliverables complete. P6 DONE. |

---

## 9. Directory Structure

```
06-torvalds-digital-clone/
├── CLAUDE.md                           ← Claude Code persistent memory
├── README.md                           ← Gold standard README
├── pyproject.toml
├── .env                                ← API keys (OPENAI, COHERE) — git-ignored
├── .env.example                        ← Template with required key names, no values
├── docs/
│   ├── PRD.md                          ← This file
│   ├── learning/
│   │   └── concepts-primer.html        ← Multi-agent concepts reference
│   ├── architecture/
│   │   ├── system-architecture.md      ← A1: high-level Mermaid diagram
│   │   ├── single-query-sequence.md    ← A2: single leader query flow
│   │   ├── dual-leader-sequence.md     ← A3: comparison mode flow
│   │   ├── data-models.md              ← A4: Pydantic model relationships
│   │   └── data-flow.md               ← A5: offline vs online pipelines
│   └── adr/
│       ├── ADR-001-crewai-flow-pattern.md
│       ├── ADR-002-rag-config-embeddings-reranking-chunking.md
│       ├── ADR-003-feature-vectors-vs-llm-embeddings.md
│       ├── ADR-004-groundedness-scoring-approach.md
│       ├── ADR-005-shared-rag-dual-leader-mode.md
│       └── ADR-006-local-vs-api-llm-evaluation.md  ← conditional (Day 6 experiment)
├── src/
│   ├── __init__.py
│   ├── schemas.py                      ← All Pydantic models (including CloneState)
│   ├── config.py                       ← YAML config loading + validation
│   ├── flow.py                         ← DigitalCloneFlow — the orchestrator (PlannerAgent)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── style_crew.py              ← ChatStyleAgent as single-agent CrewAI Crew
│   │   ├── rag_steps.py               ← RAGAgent logic (Flow step functions)
│   │   ├── evaluator_steps.py         ← EvaluatorAgent logic (Flow step functions)
│   │   └── fallback_steps.py          ← FallbackAgent logic (Flow step functions)
│   ├── style/
│   │   ├── __init__.py
│   │   ├── email_parser.py             ← LKML mbox parsing + cleaning
│   │   ├── feature_extractor.py        ← 15 style feature extraction
│   │   ├── profile_builder.py          ← Incremental style profile
│   │   └── style_scorer.py             ← Cosine similarity scoring
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── corpus_loader.py            ← HuggingFace open-phi/textbooks
│   │   ├── chunker.py                  ← Fixed + semantic chunking
│   │   ├── embedder.py                 ← OpenAI + MiniLM via LiteLLM
│   │   ├── indexer.py                  ← FAISS index build + save/load
│   │   ├── retriever.py                ← Vector search + reranking
│   │   └── citation_extractor.py       ← Parse [N] references
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── groundedness_scorer.py      ← Semantic similarity heuristic
│   │   ├── confidence_scorer.py        ← Multi-factor + explanation
│   │   └── evaluator.py               ← Weighted formula + decision
│   ├── fallback/
│   │   ├── __init__.py
│   │   ├── calendar_mock.py            ← Simulated booking slots
│   │   ├── context_summarizer.py       ← Query + conversation summary
│   │   └── unstyled_responder.py       ← Grounded response sans style
│   ├── visualization.py                ← All 7 charts
│   └── cli.py                          ← Click CLI commands
├── streamlit_app.py                    ← Streamlit web UI
├── data/
│   ├── emails/                         ← Downloaded LKML mbox files
│   ├── models/                         ← Style profile JSONs
│   ├── rag/
│   │   ├── faiss_index/                ← Saved FAISS indices
│   │   └── chunks_metadata.json        ← Chunk metadata
│   ├── evaluations/                    ← Evaluation result JSONs
│   ├── fallback_logs/                  ← Fallback trigger logs
│   └── cache/                          ← LLM response cache
├── results/
│   ├── charts/                         ← Generated PNG visualizations
│   └── iteration_log.json              ← Config change log
├── configs/
│   └── default.yaml                    ← Default pipeline configuration
└── tests/
    ├── test_email_parser.py
    ├── test_feature_extractor.py
    ├── test_profile_builder.py
    ├── test_chunker.py
    ├── test_retriever.py
    ├── test_evaluator.py
    ├── test_fallback.py
    ├── test_flow.py                    ← Flow orchestration tests
    └── test_integration.py
```

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LKML mbox parsing fails (encoding, format issues) | No style corpus | Fallback to FLOSSmole pre-extracted dataset. Budget 1h on Day 1 for parsing troubleshooting. |
| Torvalds emails too noisy (patches, forwards) | Style features contaminated | Aggressive cleaning pipeline (Decision 5). Strip quoted text, patches, signatures. Validate min 20 words per email. |
| CrewAI Flows learning curve | Blocked on Day 1 | Learn Day dedicated to understanding CrewAI Flows. Study `@start`/`@listen`/`@router` decorators, FlowState, and the Lead Score Flow example (uses @router for conditional branching — closest to P6's pattern). ADR-001 forces engagement. |
| Style score < 0.90 | Success criteria unmet | Check feature normalization. Compare raw features between leaders — if too similar, add more discriminative features. The 4 LKML-specific features were added for this reason. |
| Groundedness score < 0.60 | RAG retrieval insufficient | Try semantic chunking (P2 winner). Increase top-K from 5 to 10 before reranking. Filter corpus more aggressively to CS/OS topics. |
| Fallback rate outside 30-40% | Calibration off | Adjust threshold (start at 0.75, iterate). Weight sensitivity experiment on Day 6 designed to find optimal config. |
| OpenAI rate limits | Stalls pipeline | Cache all LLM responses (MD5-keyed JSON). MiniLM baseline doesn't need API. Cohere free tier handles reranking volume. |
| Dual-leader mode too slow | > 2s latency | Shared RAG retrieval (Decision 3). Style application + evaluation is lightweight compared to retrieval. |
| 8-day timeline too tight | Incomplete features | Days 1-5 = core pipeline (MVP). Days 6-7 = experiments + UI. Day 8 = docs. If behind: drop Streamlit (Loom demo instead), reduce experiment scope. |

---

## 11. Interview Talking Points

1. **"Have you built a multi-agent system?"** → P6: 5 specialized agents orchestrated by CrewAI Flows. Event-driven workflow with `@router` conditional branching at evaluation. Each agent independently testable. Dual-leader comparison mode proves the style learning captures genuine individual patterns. Documented with 5 Mermaid architecture diagrams: system overview, query sequences, data models, data flow.

2. **"How did you make data-driven decisions across projects?"** → P2 benchmarked 16 RAG configs and found OpenAI embeddings beat local models by 26%. P6 carries that evidence forward — the RAG pipeline uses P2's winning config as default, with MiniLM as documented baseline. The iteration log shows the comparison on P6's actual data.

3. **"How do you handle AI system failures?"** → Confidence-based fallback with explainable decisions. Every response gets a three-dimensional score. Below 0.75, the system offers two alternatives: schedule a call with the real person, or get a factually grounded response without style matching. The confidence explanation tells you WHY it failed.

4. **"What's the most interesting thing you found?"** → The pre/post-2018 style evolution chart detected a statistically significant shift in Torvalds' communication patterns after his public commitment to change. [Fill with actual data after Day 6.]

5. **"How does this connect to your portfolio?"** → P2 measured RAG. P5 built production RAG. P6 embeds RAG into a multi-agent system. Each project builds on the last. The same FAISS + Cohere reranking pattern appears in all three, improving each time.

6. **"Why CrewAI over LangChain agents?"** → Bootcamp spec required CrewAI. But I went deeper: I evaluated all three CrewAI orchestration patterns. Sequential Crews can't branch conditionally. Hierarchical is documented as broken in production. I chose Flows — CrewAI's event-driven orchestration layer — as the deterministic backbone, with a single-agent Crew only for the style generation step where LLM agency adds real value. This is CrewAI's own recommended production pattern, validated by their DocuSign case study.

7. **"Why clone Torvalds instead of a generic employee?"** → Torvalds has the most distinctive communication style in open source. There's a published paper (Schneider et al. 2016) that validated style feature extraction on exactly this data. And the 2018 behavioral change provides ground truth — if my style pipeline can detect a real change, it's not just measuring noise.
