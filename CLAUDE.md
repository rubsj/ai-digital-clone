# CLAUDE.md — P6: Torvalds Digital Clone (Multi-Agent System)

> **Read this file + docs/PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P6 — Torvalds Digital Clone: Multi-Agent Style-Matching System
- **Location:** `rubsj/06-torvalds-digital-clone` (standalone repo)
- **Timeline:** 8 sessions (~32h total), with learn day + experiment day prioritizing depth
- **PRD:** `docs/PRD.md` — the product requirements contract (v1)
- **Requirements:** [Notion Customized Requirements](https://www.notion.so/336db630640a81f2882bcbdf53723796)
- **Original Bootcamp Spec:** [Notion Original Requirements](https://www.notion.so/335db630640a816680d4f12d00e14afd)

---

## Model Routing Protocol (CRITICAL)

**Opus plans. Sonnet executes.**

### Opus (Planning & Architecture)
- Start of each day: read PRD tasks, create detailed implementation plan with file-by-file approach
- Design Pydantic schemas, CrewAI Flow structure, function signatures
- Debug non-trivial issues (conceptual, not typos)
- Analyze experiment results and decide what findings matter
- Any ambiguity in the PRD

### Sonnet (Implementation)
- All code writing — implement what Opus planned
- File creation, dependency setup, test writing
- Running commands (uv sync, pytest, experiment runs)
- Routine fixes (imports, parameters, formatting)
- Chart generation, documentation

### Session Workflow
```
1. Opus: "Read CLAUDE.md and docs/PRD.md. Today is Day [N]. Plan implementation."
2. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
3. Ruby reviews plan for gaps against PRD
4. Sonnet: "Execute the plan. Start with [first file]."
5. Sonnet implements, tests, commits
6. If blocked → Opus for debugging
7. Session end → Sonnet for git commit, journal entry, CLAUDE.md update
```

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1–P5.
- **Learning Priority:** Learning depth over speed. Rabbit holes encouraged if they produce genuine insight.
- **Hardware:** Mac Pro M5 Max, 128GB unified RAM, 40 GPU cores.
- **IDE:** VS Code + Claude Code extension

### Patterns Proven in P1–P5 (Reuse These)
| Pattern | Source | P6 Application |
|---------|--------|----------------|
| Pydantic models + validators | P1–P5 | EmailMessage, StyleProfile, EvaluationResult, CloneState, all schemas |
| Instructor + auto-retry | P1–P5 | Style-matched response generation, confidence explanation strings |
| JSON file cache (MD5 key) | P1/P2/P4/P5 | Cache LLM responses + embeddings. Never re-call OpenAI for same input. |
| FAISS IndexFlatIP + L2 normalization | P2/P5 | Knowledge base vector index. ALWAYS normalize before add() and search(). |
| OpenAI text-embedding-3-small via LiteLLM | P2/P5 | Primary embeddings (26% better than MiniLM — P2 evidence) |
| Cohere Rerank API (top-20 → top-5) | P2/P5 | 2-stage retrieval. 20% lift proven in P2. |
| LiteLLM for LLM routing | P5 | Provider-agnostic wrapper. `from litellm import completion`. |
| matplotlib/seaborn charts | P1–P5 | 7 visualizations + style evolution chart |
| Click CLI | P2/P5 | 5 commands: learn, index, query, compare, evaluate |
| Rich progress bars | P2/P5 | Email parsing, chunk indexing progress display |
| ADR template (5 sections — see Writing Rules below) | P1–P5 | 5-6 ADRs distributed across Days 1-6 |
| `yaml.safe_load()` exclusively | P5 | Config loading. NEVER `yaml.load()`. |

### New for P6 (Learn These)
- **CrewAI Flows** — `from crewai.flow.flow import Flow, listen, start, router`. Event-driven orchestration with `@start()`, `@listen(method)`, `@router()` decorators. FlowState = Pydantic BaseModel passed between steps. Think of it as a Spring event bus with typed state — each step listens for the previous step's completion, and `@router` enables conditional branching (like Java's `switch` on method return value).
- **CrewAI Agent + Crew** — `from crewai import Agent, Task, Crew`. Used ONLY for ChatStyleAgent (style generation needs LLM agency with role/goal/backstory). Other agents are direct function calls in Flow steps. Don't over-use Crew — it adds overhead for deterministic steps.
- **Python `mailbox` module** — `import mailbox; mbox = mailbox.mbox("path/to/file.mbox")`. Parses mbox email archives into message objects. `msg["From"]`, `msg["Subject"]`, `msg.get_payload()`. Like Java's `MimeMessage` parsing but simpler. Watch for encoding issues — LKML archives span decades.
- **Style feature extraction** — Numerical features from text: punctuation frequency, vocabulary richness (unique/total), sentence length distribution, capitalization ratio. All must normalize to [0,1] for cosine similarity. This is NLP feature engineering, not embedding-based.
- **Cosine similarity on feature vectors** — `from numpy import dot; from numpy.linalg import norm; sim = dot(a, b) / (norm(a) * norm(b))`. For 15-feature style vectors. NOT the same as embedding similarity — here each dimension has a human-readable meaning (unlike 384d/1536d embedding vectors).
- **Incremental learning** — `updated = (1 - alpha) * current + alpha * new`. Weighted average for streaming updates. alpha=0.3 default. Like an exponential moving average (EMA) in trading systems. The key insight: you never recompute from scratch.

---

## Writing Rules

> Inherited from portfolio-wide standards. Applies to all docs — ADRs, journal entries, READMEs, code comments.

- Write as a practitioner documenting real decisions, not a consultant producing a deliverable
- First person is allowed and preferred where natural ("I picked X because", "this burned us")
- Never narrate the document's own importance — if it mattered, just state what happened
- No section whose only purpose is to make the author look good
- Analogies go inline as parentheticals — never in their own dedicated section
- Bold emotional category labels ("Easier:", "Harder:") are banned — write plain prose or plain bullets
- Numbers and benchmarks stay where they're contextually relevant — never aggregate into a "Validation" section
- Section headers are plain nouns — not action phrases, not corporate labels
- If a sentence could have been written without knowing anything specific about this project, delete it
- Code comments explain WHY, never what — if the code is readable, no comment needed
- No hedging openers in comments: ban "Note that", "This ensures", "It's worth mentioning"
- Docstrings: one sentence what + one sentence non-obvious how/why — no parameter narration
- Inline comments for short context, block comments only for genuinely non-obvious decisions
- Comment like you're explaining to a teammate at 11pm — direct, no filler

### ADR Format (STRICT — follow ADR-001/002/003 exactly)

Every ADR has exactly **5 sections** in this order: Context, Decision, Alternatives Considered, Quantified Validation, Consequences. No more, no fewer.

**Banned sections** — never add these regardless of plan instructions:
- "Interview Signal" — embed any interview-relevant insight as prose inside Consequences
- "Java/TS Parallel" or any named analogy section — one parenthetical sentence at the END of Consequences only, not a dedicated section
- "Cross-References" — inline mentions in the relevant section only

**Alternatives Considered format:**
- Each alternative is a `**bold name** — prose paragraph` entry
- Never use a markdown table with "Why Not" columns
- Explain why you didn't pick it in the paragraph, not in a separate column

**Quantified Validation:**
- A table or numbered list of actual measurements — agreement rates, latency numbers, cost calculations, Recall@5 scores
- Numbers that were inputs to the decision, not post-hoc justification

**Consequences:**
- Single flowing section, no sub-headers
- Cover the actual operational tradeoffs: what gets easier, what gets harder, what you'd have to redo to port it
- End with the Java/TS/domain parallel as one parenthetical sentence inline — not a header, not a bullet

---

## Architecture Rules (FINAL — Do Not Re-Debate)

These come from PRD Sections 3, 4, and 5. All design decisions are finalized.

1. **CrewAI Flow** — the orchestrator. `DigitalCloneFlow(Flow[CloneState])` with `@start`, `@listen`, `@router`. NOT a sequential Crew (can't branch). NOT hierarchical (documented broken).
2. **Single-agent Crew** — ONLY for ChatStyleAgent (style generation needs role/goal/backstory). All other agents are direct function calls within Flow steps.
3. **CloneState** — Pydantic BaseModel passed between all Flow steps. Contains: query, leader, retrieved_chunks, styled_response, evaluation, final_output.
4. **FAISS IndexFlatIP** — L2-normalized vectors. Exact brute-force search for <1K chunks.
5. **OpenAI text-embedding-3-small** (primary) + **MiniLM** (baseline comparison). Both via LiteLLM.
6. **Cohere Rerank** — 2-stage: top-20 → rerank → top-5. Not optional.
7. **GPT-4o-mini** via LiteLLM — for style-matched generation AND evaluation explanation strings.
8. **Instructor + Pydantic** — ALL structured LLM output. Never raw `json.loads`.
9. **Feature vectors for style** — 15 numerical features (11 base + 4 LKML-specific), NOT LLM embeddings. Enables radar chart + interpretable cosine similarity.
10. **Shared RAG for dual-leader** — retrieve once, style twice. RAG is the expensive step.
11. **Groundedness scoring** — semantic similarity heuristic (fast), calibrated by 5-sample LLM judge. Not full LLM judge per query.
12. **Calendar booking** — mocked (simulated slots). No real Cal.com API.
13. **Fallback** — two options: calendar booking + unstyled-but-grounded response.
14. **Email cleaning** — strip quoted text (lines with `>`), patches, signatures, auto-generated content. Min 20 words after cleaning.
15. **Chunking** — 500 chars / 50 overlap (baseline) + semantic markdown split (experiment). Both in iteration log.

---

## Stop Gates (CRITICAL)

Claude Code MUST stop and get Ruby's approval before:

1. **Any destructive operation** — deleting files, overwriting existing data, dropping indices
2. **Changing architecture decisions** — anything in the "Architecture Rules" section above
3. **Adding new dependencies** beyond what's in pyproject.toml
4. **Modifying the scoring formula** (0.4/0.4/0.2 weights) outside of Day 6 experiments
5. **Any operation that calls OpenAI API more than 100 times** in a single run (cost guard)
6. **Committing directly to main** — always work on feature branches

---

## Verification Protocol

Claude Code reporting steps as "done" is not sufficient. For each deliverable:

1. **Echo-back**: Paste the actual terminal output or file content showing it works
2. **File:line references**: Point to specific code locations, not just file names
3. **Test evidence**: Show pytest output with pass counts
4. **Plan-diff**: If implementation deviated from plan, explain what changed and why

---

## Current State

> **Update this section at the end of EVERY session.**

### Last Updated: 2026-04-26

**Current Day:** Day 5 complete
**Branch:** main (feat/day5-flow-orchestration pending PR)
**Tests:** 433 passing
**Coverage:** 90% src/flow.py, 100% src/agents/style_crew.py (target ≥90% met)

### What's Done
- [x] Customized requirements page created in Notion
- [x] PRD v1 created (docs/PRD.md)
- [x] CLAUDE.md created (this file)
- [x] All 10 architectural decisions locked (D1-D10)
- [x] CrewAI Flow pattern selected (over Sequential and Hierarchical)
- [x] 5-6 ADRs planned and distributed across days
- [x] scratch/flow_poc.py — CrewAI Flows learning artifact (@start, @listen, @router validated)
- [x] pyproject.toml + .python-version (3.13) + .env.example + .gitignore
- [x] Full directory structure per PRD Section 9
- [x] src/schemas.py — all 11 Pydantic models including CloneState
- [x] configs/default.yaml + src/config.py with Pydantic validation
- [x] src/style/email_parser.py — full cleaning pipeline (quotes, patches, signatures, footers)
- [x] scripts/validate_emails.py — validates mbox files once downloaded
- [x] docs/adr/ADR-001-crewai-flow-pattern.md — P1-P5 template, first-person voice
- [x] tests/test_schemas.py (34 tests), tests/test_email_parser.py (33 tests), tests/test_config.py (11 tests)
- [x] src/style/feature_extractor.py — 15 features (11 base + 4 LKML-specific), all [0,1]
- [x] src/style/profile_builder.py — batch aggregation + incremental EMA update (alpha=0.3)
- [x] src/style/scorer.py — cosine similarity on feature vectors
- [x] scripts/build_profiles.py — end-to-end pipeline + variance table + radar chart
- [x] src/visualization.py — matplotlib polar radar chart (15 axes, dual-leader)
- [x] docs/adr/ADR-003-feature-vectors-vs-llm-embeddings.md
- [x] src/rag/corpus_loader.py — HuggingFace open-phi/textbooks, 1,511 CS docs, Rich progress
- [x] src/rag/chunker.py — chunk_baseline (RecursiveCharacter) + chunk_semantic (MarkdownHeader)
- [x] src/rag/embedder.py — OpenAI text-embedding-3-small (LiteLLM) + MiniLM, MD5 JSON cache
- [x] src/rag/indexer.py — FAISS IndexFlatIP build/save/load, _validate_norms
- [x] src/rag/retriever.py — embed query → FAISS top-20 → RetrievalResult list
- [x] src/rag/reranker.py — Cohere ClientV2 rerank top-5 with try/except fallback
- [x] src/rag/citation_extractor.py — [N] parsing, 1-based, dedup, score clamp
- [x] src/rag/__init__.py — re-exports all 13 public functions
- [x] src/agents/rag_agent.py — RAGAgent facade (build + retrieve)
- [x] scripts/test_rag_pipeline.py — 7-step e2e validation with Rich tables
- [x] docs/adr/ADR-002-rag-config-embeddings-reranking-chunking.md
- [x] 7 new test files (305 total passing)
- [x] src/evaluation/groundedness_scorer.py — sentence-level max cosine sim, batch embed, chunk.embedding reuse
- [x] src/evaluation/confidence_scorer.py — 3-signal heuristic (retrieval relevance + completeness + uncertainty penalty)
- [x] src/evaluation/evaluator.py — weighted formula 0.4/0.4/0.2, single Instructor call, EvaluationResult
- [x] src/evaluation/__init__.py — re-exports
- [x] src/fallback/calendar_mock.py — pure Python datetime, seeded RNG, business-day skipping
- [x] src/fallback/context_summarizer.py — deterministic topic string, dedup, query truncation
- [x] src/fallback/unstyled_responder.py — Instructor + LiteLLM, plain-factual system prompt
- [x] src/fallback/__init__.py — re-exports
- [x] src/agents/evaluator_steps.py — EvaluatorAgent thin facade
- [x] src/agents/fallback_steps.py — build_fallback_response() composing all fallback modules
- [x] 6 new test files (382 total, 99% coverage on new modules)
- [x] docs/adr/ADR-004-groundedness-scoring-approach.md
- [x] docs/learning-journal.md Day 4 entry

### What's Next
- Day 5: Flow Orchestration + Integration
  - `src/flow.py`: DigitalCloneFlow with @start, @listen, @router
  - `src/agents/style_crew.py`: Single-agent CrewAI Crew for style generation
  - Wire: retrieve_knowledge → apply_style → evaluate_response → deliver/fallback
  - @router: return "deliver" or "fallback" based on EvaluationResult.decision
  - Dual-leader comparison: run Flow twice, share retrieved_chunks via CloneState
  - End-to-end test: query → scored response (single leader)
  - ADR-005: Shared RAG for Dual-Leader Mode

### Blockers
- None

### Key Decisions Made (Day 4)
- Batch embedding over per-sentence calls: `embed_openai(sentences)` once for all response sentences, then one more call for any chunks missing `.embedding`. Avoids N API calls for N sentences.
- `EvaluationResult @model_validator` enforces weighted formula — round `final` to 6 decimal places before passing to avoid IEEE 754 drift failures.
- Equal 1/3 weights for confidence sub-signals are a placeholder; Day 6 weight sensitivity sweep will calibrate.
- `random.Random(seed)` (isolated instance) rather than `random.seed()` (module-level global) for seeded calendar slots — required for test isolation.
- `evaluator_steps.py` (not `evaluator_agent.py`) — matched the on-disk stub name for consistency with the `_steps` suffix pattern in the agents directory.

### Key Decisions Made (Day 3)
- FAISS -1 padding: `index.search()` returns -1 when k > ntotal. Filter in retriever or metadata[-1] silently returns wrong result.
- `faiss.normalize_L2()` mutates in-place — not functional style. Called before `index.add()` AND before `index.search()`.
- `cohere.ClientV2` (not deprecated `cohere.Client`) — response shape differs between versions.
- Pydantic v2: `model_copy(update={"embedding": vec})` for immutable chunk update (not in-place mutation).
- Dataset has direct `topic` column — plan assumed it needed parsing from `outline`.

---

## Day-by-Day Checklist

### Day 1 — Learn Day + Foundation
- [x] Study CrewAI Flows: `@start`, `@listen`, `@router` decorators, FlowState
- [x] Study the Lead Score Flow example (closest to P6's @router pattern)
- [x] Project setup: pyproject.toml, .env.example, directory structure
- [x] Pydantic schemas: EmailMessage, StyleFeatures, StyleProfile, KnowledgeChunk, RetrievalResult, EvaluationResult, FallbackResponse, StyledResponse, LeaderComparison, Citation, CloneState
- [x] Email parser: Python `mailbox.mbox()` → parse From/To/Subject/Body/Date/Message-ID
- [x] Email cleaner: strip quoted text (`>`), patches, signatures, footers, min 20 words
- [x] Download LKML mbox for Torvalds and Kroah-Hartman
- [x] Validate: ≥200 clean emails per leader
- [x] Tests for schemas + email parser (95% total coverage — exceeds 90% target)
- [x] **ADR-001: CrewAI Flow vs Sequential vs Hierarchical** written and committed
- [x] **Checkpoint:** Email parser works. 200+ emails per leader extracted and cleaned. PASSED.

### Day 2 — ChatStyleAgent
- [ ] Feature extractor: 15 features (11 base + 4 LKML-specific)
  - Base: avg_message_length, greeting_patterns, punctuation_patterns, capitalization_ratio, question_frequency, vocabulary_richness, common_phrases, reasoning_patterns, sentiment_distribution, formality_level, technical_terminology
  - LKML: code_snippet_freq, quote_reply_ratio, patch_language, technical_depth
- [ ] All features normalized to [0, 1]
- [ ] Style profile builder: aggregate features across all emails → StyleProfile
- [ ] Incremental learning: alpha-weighted update (`updated = (1-α)*current + α*new`)
- [ ] Build profiles for BOTH Torvalds and Kroah-Hartman
- [ ] Verify: radar chart shows visually distinct profiles
- [ ] Style scorer: cosine similarity between profile vector and response feature vector
- [ ] Tests for feature extractor + profile builder
- [ ] **ADR-003: Feature vectors vs LLM embeddings for style** written and committed
- [ ] **Checkpoint:** Two distinct style profiles. Style score > 0.90 on training emails.

### Day 3 — RAGAgent
- [x] Corpus loader: HuggingFace `open-phi/textbooks`, filter field="computer science"
- [x] Chunker: 500 chars / 50 overlap (baseline) + semantic markdown header split (experiment)
- [x] Embedder: OpenAI text-embedding-3-small via LiteLLM (primary) + MiniLM (baseline)
- [x] FAISS indexer: build + save/load. L2-normalize before add().
- [x] Retriever: embed query → FAISS search top-20 → Cohere rerank → top-5
- [x] Citation extractor: parse [N] references from generated text
- [x] Validate: ≥900 chunks indexed, retrieval < 1s, citations working
- [x] Tests for chunker, embedder, indexer, retriever (305 total, 94% RAG coverage)
- [x] **ADR-002: RAG Config — Embeddings, Reranking, Chunking (P2 Evidence)** written and committed
- [x] **Checkpoint:** RAG pipeline end-to-end. Query → relevant cited chunks. PASSED.

### Day 4 — EvaluatorAgent + FallbackAgent
- [x] Style scorer: cosine similarity between leader profile and response features
- [x] Groundedness scorer: semantic similarity between response sentences and retrieved chunks
- [x] Confidence scorer: retrieval relevance + response completeness + uncertainty penalty + explanation string
- [x] Evaluator: weighted formula (0.4 style + 0.4 groundedness + 0.2 confidence)
- [x] Decision logic: ≥0.75 deliver, <0.75 fallback
- [x] FallbackAgent: trigger detection, context summarizer, calendar mock, unstyled responder
- [x] Tests for all scoring components + fallback triggers
- [x] **ADR-004: Groundedness Scoring — Semantic Similarity vs LLM Judge** written and committed
- [x] **Checkpoint:** Evaluation pipeline scores responses. Fallback triggers correctly.

### Day 5 — Flow Orchestration + Integration
- [ ] `src/flow.py`: DigitalCloneFlow with @start, @listen, @router
- [ ] `src/agents/style_crew.py`: Single-agent CrewAI Crew for style generation
- [ ] Wire: retrieve_knowledge → apply_style → evaluate_response → deliver/fallback
- [ ] @router: return "deliver" or "fallback" based on final_score threshold
- [ ] Dual-leader comparison: run Flow twice, share retrieved_chunks via state
- [ ] Error recovery: try/except in Flow steps → fallback on any failure
- [ ] End-to-end test: query → scored response (single leader)
- [ ] End-to-end test: query → LeaderComparison (dual mode)
- [ ] Architecture diagrams A2 (single query sequence) + A3 (dual-leader sequence)
- [ ] **ADR-005: Shared RAG for Dual-Leader Mode** written and committed
- [ ] **Checkpoint:** Full pipeline works. Dual-leader comparison produces two scored responses.

### Day 6 — Experiment Day
- [ ] Embedding comparison: OpenAI vs MiniLM on same 10 queries → iteration log entry
- [ ] Chunking comparison: 500/50 vs semantic markdown split → iteration log entry
- [ ] Scoring weight sensitivity: 3 configs (0.4/0.4/0.2, 0.5/0.3/0.2, 0.3/0.5/0.2) × 10 queries
- [ ] Pre/post-2018 style evolution: partition Torvalds emails, compute features, plot time-series
- [ ] Iteration log: ≥3 entries with before/after metrics
- [ ] Optional: local vs API LLM experiment → ADR-006 if interesting
- [ ] **Checkpoint:** All experiments complete. Iteration log populated. Style evolution chart generated.

### Day 7 — Streamlit + CLI + Architecture Docs
- [ ] Streamlit app: query input, leader selector dropdown, response display, score breakdown, confidence explanation, fallback display, side-by-side comparison mode
- [ ] Click CLI: learn, index, query, compare, evaluate commands
- [ ] All 7 visualization PNGs generated and saved to results/charts/
- [ ] Architecture diagrams A1 (system), A4 (data models), A5 (data flow) as Mermaid markdown
- [ ] Tests for CLI commands
- [ ] **Checkpoint:** Streamlit demo working. CLI functional. All charts + architecture docs committed.

### Day 8 — Documentation Sprint
- [ ] README.md (gold standard: results above fold, architecture diagram, findings, ADR table, tech stack, quick start, known gaps, demo link/Loom)
- [ ] Humanize all 5-6 ADRs (first-person voice, real debugging stories, varied sentence structure)
- [ ] Learning Journal entry (Notion): multi-agent patterns, CrewAI Flows, style transfer
- [ ] Concept Library entries (Notion): "Multi-Agent Topologies", "CrewAI Flows vs Crews", "Style Transfer via Feature Vectors"
- [ ] Loom recording (2-min walkthrough)
- [ ] Final success criteria checklist pass (all checkboxes in Notion requirements)
- [ ] Portfolio footer: "Part of a [9-project AI engineering sprint](https://github.com/rubsj/ai-portfolio). Built Feb-May 2026."
- [ ] Final git push
- [ ] **P6 COMPLETE**

---

## Troubleshooting Guide

### "LKML mbox parsing fails with encoding errors"
LKML archives span decades — expect mixed encodings (ASCII, UTF-8, Latin-1). Use `msg.get_payload(decode=True)` with fallback: try UTF-8 first, then Latin-1, then `errors='replace'`. If mbox is too large, filter by date range (2015-2023 is a good balance of modern style + pre/post-2018 coverage).

### "Not enough clean emails after filtering"
The 20-word minimum and quote stripping aggressively reduce count. Lower to 10 words, or include short but meaningful messages like "Applied, thanks" as a separate category (don't use for style features, but count them for patch_language feature). If still under 200, use FLOSSmole pre-extracted dataset as fallback.

### "Style profiles look identical for both leaders"
Features aren't discriminative enough. Check: are features properly normalized to [0,1]? Is quote stripping working (contamination from quoted text makes everyone look similar)? Add more LKML-specific features. The 4 domain features (code_snippet_freq, quote_reply_ratio, patch_language, technical_depth) were specifically chosen for discrimination.

### "FAISS IndexFlatIP returns low similarity scores"
IndexFlatIP = inner product. Equals cosine similarity ONLY if vectors are L2-normalized. Call `faiss.normalize_L2(embeddings)` before `add()` AND before `search()`. Verify: `np.linalg.norm(emb)` should be ≈ 1.0 for every vector.

### "Cohere reranking returns error"
Check: API key in .env, free tier hasn't hit limit (1K calls/month). Fallback: skip reranking and use raw FAISS top-5 scores. Document in iteration log as a configuration change.

### "CrewAI Flow @router not branching"
The `@router()` decorated method must return a string. Methods listening must use `@listen("string_value")` matching the return. Common mistake: returning `True/False` instead of `"deliver"/"fallback"`. Check: the router method has the `@router()` decorator AND returns a string.

### "CrewAI style Crew generates generic response ignoring style"
The Agent's backstory and task description must include specific style features. Don't just say "write like Torvalds" — include: "Use short, direct sentences. Heavy use of dashes (--). Start sentences with 'The point is' or 'The thing is'. Avoid formal greetings. Sign off with just 'Linus'." Extract these from the style profile and inject into the prompt.

### "Dual-leader mode takes >2s"
The RAG retrieval should only run once (shared). If it's running twice, check that `retrieved_chunks` is populated in CloneState after the first Flow run and reused in the second. The style + evaluation steps are lightweight (<500ms each).

### "Groundedness score always near 0 or always near 1"
The semantic similarity threshold needs calibration. Run 5 test queries, manually label groundedness (0-1), compare to heuristic. Adjust the similarity threshold. The LLM judge calibration step on Day 4 is specifically for this.

### "Fallback rate outside 30-40%"
If too high (>40%): lower threshold from 0.75 toward 0.70. If too low (<30%): raise threshold or tighten confidence scoring (increase uncertainty penalty). Day 6 weight sensitivity experiment is designed to find the optimal config.

### "Pre/post-2018 chart shows no change"
Check: are you correctly partitioning by September 2018? The shift is most visible in sentiment_distribution (more positive post-2018), capitalization_ratio (less ALL CAPS), and exclamation frequency (fewer !!). If using only post-2018 emails, you won't see the contrast — need both periods.

### "LiteLLM model string not working"
LiteLLM model strings: `"gpt-4o-mini"` for OpenAI (not `"openai/gpt-4o-mini"`). For embeddings: `litellm.embedding(model="text-embedding-3-small", input=[text])`. Check `OPENAI_API_KEY` is set in `.env`.

### "Instructor validation error on LLM response"
Instructor retries up to `max_retries` (default 3). If the LLM consistently produces invalid output, the Pydantic model may be too strict. Check: are Optional fields marked Optional? Are enum values matching exactly? Common fix: loosen validators, add `description` to fields for LLM guidance.
