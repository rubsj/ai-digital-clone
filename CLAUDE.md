# CLAUDE.md — P6: Torvalds Digital Clone (Multi-Agent System)

> **Read this file + docs/PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P6 — Torvalds Digital Clone: Multi-Agent Style-Matching System
- **Location:** `rubsj/06-torvalds-digital-clone` (standalone repo)
- **Timeline:** 8 sessions (~32h total), with learn day + experiment day prioritizing depth
- **PRD:** `docs/PRD.md` — the product requirements contract (v1)
- **Engineering Protocols (canonical standard):** [Notion → Engineering Protocols](https://www.notion.so/35ddb630640a818aa961d003d43c0200). This CLAUDE.md copies the protocol invariants verbatim into the Prompt Discipline, Verification, and Teach-Back sections below, then adds P6-specific specializations. The Notion page is the canonical source; this file is the project-frozen snapshot.
- **Requirements:** [Notion Customized Requirements](https://www.notion.so/336db630640a81f2882bcbdf53723796)
- **Original Bootcamp Spec:** [Notion Original Requirements](https://www.notion.so/335db630640a816680d4f12d00e14afd)

---

## Prompt Discipline Protocol

> Canonical standard: [Engineering Protocols → Prompt Discipline Protocol](https://www.notion.so/35ddb630640a81458abcf79d51973120). The invariants below are copied verbatim from that page. P6 specializations are noted under each component.

### Component 1: Model Routing (invariant)

**Opus plans. Sonnet executes.** Non-negotiable.

The invariant: Opus handles design, debugging, and analysis. Sonnet handles implementation. If I find myself opening Sonnet for planning or Opus for routine implementation, I have crossed a wire.

**P6 specialization — detailed responsibility split:**

*Opus (Planning & Architecture):*
- Start of each day: read PRD/CLAUDE.md, produce file-by-file implementation plan
- Design Pydantic schemas, CrewAI Flow structure, function signatures
- ChatStyleAgent role/goal/backstory generation — the LLM-agency framing affects every downstream feature injection, so this goes to Opus even when it looks mechanical
- Style profile + Flow architecture decisions regardless of how mechanical they feel
- Debug non-trivial issues (conceptual, not typos)
- Analyze experiment results and decide what findings matter
- Any ambiguity in the PRD

*Sonnet (Implementation):*
- All code writing — implement what Opus planned
- File creation, dependency setup, test writing
- Running commands (uv sync, pytest, experiment runs)
- Routine fixes (imports, parameters, formatting)
- Chart generation, documentation
- Session notes (`docs/session-notes/dayN.md`) — raw phase notes per Verification Component 6
- Git commits, CLAUDE.md state updates

Note: Sonnet no longer writes journal entries or handover notes. Those moved to me under Teach-Back Author Pass (Day 7+).

### Component 2: Planning Prompts (invariant)

Constraint-heavy. Contain: orientation, source-of-truth references, deliverables, guardrails. Do NOT contain function signatures, pseudocode, pre-solved tradeoffs, or "the code should..." instructions. Cap ~30 lines.

**P6 specialization:** Planning prompts must reference Architecture Rules section in this file as the locked-decision boundary. Opus is not allowed to re-debate any of the 15 Architecture Rules; if a plan touches one, Opus must flag it back to me rather than work around it.

### Component 3: Execution Prompts (invariant)

Lean. Contain: session context, plan-file authority instruction, Verification Protocol reference, phased stop gates. Do NOT re-type the plan, duplicate implementation guidance, or introduce new design decisions. Typical length 40-60 lines.

**P6 specialization:** Execution prompts must include the trigger_reason: str = "" propagation convention on CloneState. Cross-step error context is project-specific and Sonnet has dropped it twice without the explicit reminder.

### Component 4: Plan-File Authority Rule (invariant)

Every execution prompt opens with: *"Re-read docs/plans/dayN-plan.md from disk before proceeding. The file on disk is authoritative over any version in your context."*

**P6 specialization:** Plan files for P6 live at `docs/plans/dayN-plan.md`. Day 4 onward use the `dayN-plan.md` naming convention strictly — earlier days had inconsistent names that broke this rule once.

### Session Workflow

```
1. Claude.ai (Opus): Socratic Gate — answer 3-5 questions on day's concepts before any plan
2. Claude.ai (Opus): "Read CLAUDE.md and docs/PRD.md. Today is Day [N]. Plan implementation."
3. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
4. Ruby reviews plan for gaps against PRD
5. Claude.ai: Ruby drafts execution prompt for Sonnet (Prompt Discipline)
6. Claude Code (Sonnet): "Re-read docs/plans/dayN-plan.md from disk. Execute. Start with [first file]."
7. Sonnet implements Phase 1, runs tests, appends Phase 1 block to docs/session-notes/dayN.md, stops at gate
8. Claude.ai: Ruby pastes Phase 1 terminal output; Claude reviews against Verification Components 3 and 4
9. If verification passes, Ruby writes Phase Defense (2-3 prompts from menu, at least one Category A + at least one B/C/D)
10. Claude grades defense; on clear, Ruby reads Phase Notes block, drafts journal entry (100-200 words), Claude redlines, push to Notion
11. Claude Code: Sonnet continues to Phase 2 (repeat steps 7-10 for each phase)
12. If blocked → Opus for debugging
13. Session end in Claude Code: Sonnet runs Plan-Diff, commits
14. Session end in Claude.ai: Ruby reads full session notes + Plan-Diff, drafts handover (10 min max), Claude redlines, push to Notion
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

## Verification Protocol

> Canonical standard: [Engineering Protocols → Verification Protocol](https://www.notion.so/35ddb630640a81929b92e02c79c9b9c3). The 6 components below are copied verbatim. P6 specializations follow each component.

### Component 1: Echo-Back (invariant — first step, before any code)

Before writing any code, Claude Code echoes every deliverable as a numbered implementation plan. For EACH item: (1) file(s), (2) function/class, (3) verification command or test. If anything is unclear or conflicts with existing code, ASK before proceeding. Wait for explicit "approved, proceed" before implementation begins.

**P6 specialization:** Echo must explicitly name which agent (RAGAgent, EvaluatorAgent, FallbackAgent, ChatStyleAgent) the phase modifies and which Pydantic schemas in `src/schemas.py` are touched. Schema changes that affect CloneState require special attention since they propagate across every Flow step.

### Component 2: Phased Execution with Stop Gates (invariant)

Split plan into 2-4 phases. Max 5-6 items per phase. After each phase, report with `file:line` references and STOP. Do not proceed until "continue" is approved.

**Approval stop gates (always require explicit OK, invariant):**
1. Any destructive operation — deleting files, overwriting data, dropping indices
2. Changing architecture decisions — anything in the "Architecture Rules" section above
3. Adding new dependencies beyond what's in pyproject.toml
4. Any operation that calls OpenAI API more than 100 times in a single run (cost guard)
5. Committing directly to main — always work on feature branches

**P6 specializations (additional project-specific stop gates):**
- Modifying the scoring formula (0.4 style + 0.4 groundedness + 0.2 confidence) outside of Day 6 experiments
- Bypassing the 0.75 delivery threshold in the @router branching logic
- Importing from FallbackAgent inside EvaluatorAgent (circular dependency risk; build_fallback_response composes fallbacks at the agent facade layer only)
- Calling LiteLLM directly from agent code — always go through the wrapper

### Component 3: Verification Contract (invariant — raw terminal output, never descriptions)

Before reporting any session complete, Claude Code MUST run and paste ACTUAL terminal output:

- **Smoke tests**: specific commands proving each deliverable works
- **Grep verification**: `grep -n "function_name" src/file.py | head -5` proving code exists
- **Test suite**: `python -m pytest tests/ -x --tb=short 2>&1 | tail -10` (all pass, 0 failures)
- **Coverage**: `python -m pytest --cov=src --cov-report=term-missing 2>&1 | tail -20` (≥90%)

**P6 specialization:** Coverage threshold is ≥90% on any file under `src/`. The threshold is a hard gate. Day 4 set the precedent at 99% on new modules; Day 6 maintained ≥90% across src/ overall while experiment scripts (in `scripts/`) are exempt.

### Component 4: Plan-Diff at Session End (invariant)

For EVERY numbered plan item, report one of:
- **DONE**: `file:line` + 1-sentence description
- **SKIPPED**: reason + whether acceptable
- **PARTIAL**: what's missing + impact

No DONE without specific file:line citation. "Implemented as planned" or "completed successfully" without proof is insufficient.

**P6 specialization (no addition).** Component 4 is fully invariant for P6.

### Component 5: Anti-Patterns Checklist (invariant)

| Anti-Pattern | Prevention |
|---|---|
| Report "done" without running code | Verification Contract (component 3) |
| Skip plan items silently | Plan-Diff (component 4) forces accounting |
| Happy-path-only tests | Plan must specify edge cases; Plan-Diff catches |
| Guess environment variables | `echo $VAR \| head -c4` to verify before using |
| Proceed past STOP gate | Explicit gate in prompt, wait for "continue" |
| Duplicate PRD instead of reading it | "Read PRD directly, don't re-derive" |

**P6 specializations (additional project-specific anti-patterns):**

| Anti-Pattern | Prevention |
|---|---|
| Style score collapses to ~0.50 across all queries | Verify generated responses (not queries) are passed to scorer; Day 6 6c surfaced this |
| FAISS returns -1 padding for k > ntotal | Filter -1 in retriever before metadata lookup |
| `faiss.normalize_L2()` skipped before search() | Must call before BOTH `add()` and `search()` — mutates in-place |
| Cohere reranking treated as universally +20% Recall | Corpus-shape sensitive; verify per project (P5: +20%, P6 CS textbooks: +2.5%) |
| `@router()` returning True/False instead of string | Must return string matching `@listen("string_value")` |
| Calendar mock uses `random.seed()` globally | Use isolated `random.Random(seed)` instance for test isolation |

### Component 6: Session Notes (invariant — raw material for Author Pass)

After each phase clears the stop gate, Claude Code appends 3-5 bullets to `docs/session-notes/dayN.md`. One block per phase. Each block covers five fields:

- **Built.** What got built. `file:line`.
- **Why.** Why this approach over the alternative we discussed in planning.
- **Surprising.** Anything unexpected during implementation. Edge case, library quirk, dead end.
- **Deferred.** Anything that came up as future work, not done today.
- **ADR candidate.** Any decision worth a separate ADR if not already one.

These notes are raw material I read at session close before drafting the journal entry. They are NOT the final journal entry. They are NOT a handover note. They are NOT a place for synthesis or cross-phase patterns.

The file is committed to the repo at `docs/session-notes/dayN.md`. Same path across all projects.

**Format example:**

```
## Phase 2: EvaluatorAgent groundedness scorer

- Built: src/evaluation/groundedness_scorer.py:1-87, sentence-level max cosine similarity
- Why: chose semantic similarity over LLM judge to keep latency under 100ms; LLM judge stays as 5-sample calibration only
- Surprising: chunk.embedding was None on cache miss; had to add lazy embed fallback inside the scorer
- Deferred: per-domain calibration threshold; today's threshold is global
- ADR candidate: yes, ADR-004 covers this. Done.
```

**P6 specializations:**
- Component 6 is new as of Day 7. Days 1-6 have no session notes file — that history lives in the existing learning-journal.md and ADRs. Do not retroactively create session notes for Days 1-6.
- The `docs/session-notes/` directory must be created on Day 7 alongside the first phase block.
- For P6 specifically, the "Why" field should reference Architecture Rule numbers when the decision touches one (e.g., "Why: per Architecture Rule 9, feature vectors over LLM embeddings").

---

## Teach-Back Protocol

> Canonical standard: [Engineering Protocols → Teach-Back Protocol](https://www.notion.so/35ddb630640a81e0a086ce5812853b9b). New as of Day 7. The 3 components below are invariant; P6 specialization noted at the end.

### Component 1: Socratic Gate (invariant — before any plan or prompt)

Before Claude generates the day's plan or Claude Code prompt, Claude asks 3-5 questions about the day's concepts in increasing specificity. Ruby answers in own words, no lookup, 2-3 sentences per question. Claude grades honestly and closes gaps via first-principles reasoning before any plan generation begins.

### Component 2: Phase Defense (invariant — Claude.ai, after each Claude Code phase)

This is a debrief, not a closed-book exam. By the time the phase finishes, I have evidence: the terminal output, the Phase Notes block Claude Code wrote in `docs/session-notes/dayN.md`, and my memory of the planning conversation. The defense uses all of it. The retrieval value comes from articulating my model in my own words, not from withholding information.

**Per-phase workflow (invariant ordering):**

1. Claude Code finishes phase, reports file:line, appends Phase Notes block to `docs/session-notes/dayN.md`, stops at gate
2. I paste Claude Code terminal output into Claude.ai; Claude reviews it against Verification Components 3 and 4 for this phase's items
3. If verification passes, I write the Phase Defense
4. Claude grades the defense
5. Defense clears
6. I read the full Phase Notes block again as source material
7. I draft the journal entry (Author Pass)

Verification first establishes that the code works. Then the defense tests whether I understand what works. Then the journal entry synthesizes.

**How to write the defense.**

Pick 2-3 prompts from the menu below. At least one from Category A. At least one from B, C, or D. Each answer 1-3 sentences. Total defense 4-8 sentences.

*Category A — What got built (always at least one):*
- What does this phase add to the system that wasn't there before? Name the new capability in one sentence.
- Which files changed and what each one is responsible for now.
- What's the entry point for this new capability? How would a caller invoke it?

*Category B — Design and tradeoffs (at least one if a real decision was made):*
- Why this approach over the alternative we discussed in planning. (Skip if no alternative was discussed.)
- What tradeoff did this phase accept? (Latency vs accuracy, simplicity vs flexibility, etc.)
- Which architecture rule did this phase implement or constrain against?
- Why this library, this pattern, this data structure over the obvious alternative.

*Category C — Risk and edge cases (pick one when applicable):*
- What's the most likely way this code breaks in production?
- Which test case would I write next if I had more time?
- What's the dumbest input that would crash this?
- What downstream code now depends on this in a way that wasn't true before?

*Category D — Connection (pick one when applicable):*
- How does this phase connect to a previous phase today or a previous day?
- What does this phase set up for the next phase?
- Has a similar problem come up in a prior project? How was it solved differently there?

**What Claude pushes back on:**
- Category coverage (all-A defenses skip synthesis; B/C/D required)
- Hollow answers ("built the scorer, tests pass")
- Paraphrases of terminal output or Phase Notes (the category prompts require synthesis beyond what's on screen)
- Claims that contradict the terminal output or Phase Notes (signals I misunderstood the phase)

**P6 specialization:** When Category B is selected and the decision touches an Architecture Rule, the defense must name the rule number (e.g., "Architecture Rule 9: feature vectors over LLM embeddings"). This ties phase-level decisions back to the locked design boundary.

### Component 3: End-of-Day Author Pass (invariant — Claude.ai, per phase + session close)

I write two artifacts: the journal entry for each phase, and the session handover note. Both used to be Claude Code's job in P1-P5. Both are mine now.

The raw material comes from `docs/session-notes/dayN.md` (written by Claude Code under Verification Component 6).

**Before drafting either artifact, I read `docs/session-notes/dayN.md` end to end.** The session notes are my source. They are not my draft.

**Journal entry — per phase, right after Phase Defense clears.**

- Read the Phase Notes block for this phase from `docs/session-notes/dayN.md`
- Draft the journal entry in Claude.ai. 100-200 words per phase, minimum
- Claude redlines for vagueness, fabricated certainty, missing tradeoffs, weak verbs, and passages that paraphrase the session notes instead of synthesizing
- Revise once, push to Notion

**Synthesis bar (invariant):** the journal entry must add something the session notes do not have. One of three things:
- Cross-phase pattern (how this phase connects to earlier phases today or earlier days)
- Analogy (Java/TS parallel, prior project parallel, or real-world analog)
- Connection to prior projects (P2 evidence used here, or P5 pattern applied)

If the entry is recognizably a paraphrase of the session notes, Claude pushes back and I redraft.

**Handover note — at session close.**

- Read the full `docs/session-notes/dayN.md` + the Verification Plan-Diff
- Draft the handover in Claude.ai. 10 minutes maximum
- Claude redlines for vagueness, fabricated certainty, missing tradeoffs, weak verbs
- Revise once, push to Notion

Claude Code does not write journal entries. Claude Code does not write the handover note.

**Per-phase journaling is the default for P6 and all future projects.** Batching is rejected because by session close, Phase 1 details have faded and the entry collapses to plan-diff paraphrase.

**P6 specialization:** Socratic Gate question banks for P6 concentrate on: RAG metrics (groundedness, faithfulness, recall@k), multi-agent topologies (Flow vs Crew vs Hierarchical), CrewAI Flow decorators (@start/@listen/@router state propagation), feature-vector style transfer (vs LLM embeddings), and weighted scoring formula sensitivity. Day 7 onward, every session begins with a Socratic Gate drawn from these domains.

---

## Current State

> **Update this section at the end of EVERY session.**

### Last Updated: 2026-04-27

**Current Day:** Day 6 complete
**Branch:** feat/day6-experiments (PR #6 open — pending review)
**Tests:** 437 passing (433 → 437, +4 new)
**Coverage:** ≥90% src/ maintained; no new src/ modules added in Day 6 experiment scripts

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
- [x] src/agents/style_crew.py — single-agent CrewAI Crew (ChatStyleAgent), injects concrete StyleProfile numerics into role/goal/backstory
- [x] src/flow.py — DigitalCloneFlow with 5 steps: retrieve → style_response → evaluate_response (@router) → finalize / handle_fallback
- [x] src/schemas.py — trigger_reason: str = "" added to CloneState for cross-step error propagation
- [x] scripts/timing_dual_leader.py — timing harness (mocked RAG 100ms, LLM 50ms, 5-run avg): shared 413.6ms vs independent 460.9ms
- [x] tests/test_style_crew.py (21 tests), tests/test_flow.py (37 tests)
- [x] docs/adr/ADR-005-shared-rag-dual-leader-mode.md
- [x] docs/learning-journal.md Day 5 entries
- [x] data/eval/queries_v1.json — 10 CS queries (seed=42, open-phi/textbooks), versioned for all Day 6 experiments
- [x] docs/iteration-log.md — 7 H3 entries (6a, 6a Run 2, 6b, 6c, 6d, 6e Run 1, 6e Run 2) per PRD §7g six-field format
- [x] scripts/experiment_6a_embeddings.py — OpenAI vs MiniLM embedding comparison (Cohere bimodal finding)
- [x] scripts/experiment_6b_chunking.py — fixed 500/50 vs semantic markdown chunking (+0 measurable Δ)
- [x] scripts/experiment_6c_weight_sensitivity.py — 3 weight configs × 10 queries (proxy pins style; weights retained by inertia)
- [x] scripts/experiment_6d_style_evolution.py — pre/post-2018 Torvalds style evolution (null result at per-email resolution)
- [x] scripts/experiment_6e_local_vs_api.py — Run 1 explanation-generation latency comparison (degenerate Pearson=1.0; preserved as audit trail)
- [x] scripts/experiment_6e_run2_groundedness_agreement.py — Run 2 independent groundedness scoring (Pearson(GPT,baseline)=0.82)
- [x] docs/experiments/charts/6a-embeddings.png, 6b-chunking.png, 6c-weight-sensitivity.png, 6e-local-vs-api.png, 6e-run2-groundedness-agreement.png; results/charts/07-style-evolution.png
- [x] src/eval/query_loader.py + tests/test_query_loader.py (loader reused across ≥2 experiment scripts)
- [x] docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md — three methodology-limit findings clustered
- [x] docs/adr/ADR-007-llm-evaluation-scoring-viability.md — GPT-4o-mini validated at Pearson=0.82; Ollama for explanation only
- [x] docs/learning-journal.md Day 6 entries (Phases 1–7)

### What's Next
- Day 7: Streamlit + CLI + Architecture Docs
  - Streamlit app: query input, leader selector, response display, score breakdown, side-by-side comparison
  - Click CLI: learn, index, query, compare, evaluate commands
  - All 7 visualization PNGs in results/charts/
  - Architecture diagrams A1, A4, A5 as Mermaid markdown
  - Tests for CLI commands

### Deferred Items (carried forward from Day 6)
- **Re-measure weight sensitivity against generated responses** — Day 6's sweep used queries as input proxy, which pinned style at ≈0.50 (production range 0.80–0.95). No valid measurement of weight sensitivity was produced. Re-run 6c against actual StyleCrew-generated responses when available. Source: ADR-006 §Consequences.
- **Re-measure Torvalds style evolution at population level** — Per-email significance test was the wrong instrument (within-email variance swamps between-period shifts). Re-run 6d with monthly rolling means and a mixed-effects model partitioning within-email vs between-period variance. Source: ADR-006 §Consequences.
- **ADR-002 amendment: Cohere Recall@5 lift is corpus-shape sensitive** — The ~20% Recall@5 claim came from P5 financial reports; on CS textbooks it was +2.5%. ADR-002 needs an amendment noting the corpus-shape dependency before it is cited for a new corpus. Source: ADR-006 §Consequences.

### Blockers
- None

### Key Decisions Made (Day 6)
- **Cohere bimodal verdict on CS textbook corpus (6a).** ADR-002's ~20% Recall@5 lift claim is corpus-shape sensitive. On open-phi/textbooks, Cohere collapses both embedding models to near-identical scores on 6/10 queries (Cohere max < 0.05). Embedding choice produces +2.5% post-rerank Δ groundedness — far below the P5 prior +26%. ADR-006 documents as a known limit; ADR-002 amendment is flagged as future work.
- **Production weights 0.4/0.4/0.2 retained by inertia, not evidence (6c).** Query-as-proxy pins style at ≈0.50 (std=0.0101), making weight sensitivity on the style dimension unmeasurable. All three weight configs produce Δ < 0.004. Re-measurement against actual generated responses (style ≥ 0.80) is the proper validation path; out of scope for Day 6.
- **PRD §8 style evolution exit criterion not met at per-email resolution (6d).** Within-email variance (std ≈ 0.10–0.21) swamps between-period shifts (|Δ| = 0.0002–0.017). Formality moved +0.017 in the expected direction (8% of 2σ band). Null result documented honestly — it reflects measurement resolution, not absence of behavioral change.
- **Two-ADR split: ADR-006 for methodology-limit cluster, ADR-007 for LLM scoring viability (6e).** Phase 6 Run 2 produced a positive, actionable finding — not a measurement-design limit. Merging it into the methodology cluster would have obscured a calibrated, usable result. Structurally different findings warrant separate ADRs regardless of how they land on a pre-data agreement-band framework.
- **GPT-4o-mini validated for scoring; Ollama qwen3:8b approved for explanation generation only (6e Run 2).** Pearson(GPT, baseline)=0.82 at latency parity (1504ms vs 1465ms). Pearson(Ollama, baseline)=0.68; the 0.14 gap produces directional disagreements sufficient to flip routing decisions (q03: Ollama=0.9 vs baseline=0.60 → +0.16 final score impact). Do not use Ollama qwen3:8b for evaluation scoring in production.
- **Run 1 latency advantage (2.1x) was task-specific (6e audit trail).** Run 1 measured explanation-generation latency on a degenerate task (arithmetic over pre-computed scores). Run 2 found parity on the harder scoring task. Both runs preserved in iteration log; any future citation of Run 1 speed must specify the sub-task.

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
- [x] Feature extractor: 15 features (11 base + 4 LKML-specific)
  - Base: avg_message_length, greeting_patterns, punctuation_patterns, capitalization_ratio, question_frequency, vocabulary_richness, common_phrases, reasoning_patterns, sentiment_distribution, formality_level, technical_terminology
  - LKML: code_snippet_freq, quote_reply_ratio, patch_language, technical_depth
- [x] All features normalized to [0, 1]
- [x] Style profile builder: aggregate features across all emails → StyleProfile
- [x] Incremental learning: alpha-weighted update (`updated = (1-α)*current + α*new`)
- [x] Build profiles for BOTH Torvalds and Kroah-Hartman
- [x] Verify: radar chart shows visually distinct profiles
- [x] Style scorer: cosine similarity between profile vector and response feature vector
- [x] Tests for feature extractor + profile builder
- [x] **ADR-003: Feature vectors vs LLM embeddings for style** written and committed
- [x] **Checkpoint:** Two distinct style profiles. Style score > 0.90 on training emails.

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
- [x] `src/flow.py`: DigitalCloneFlow with @start, @listen, @router
- [x] `src/agents/style_crew.py`: Single-agent CrewAI Crew for style generation
- [x] Wire: retrieve_knowledge → apply_style → evaluate_response → deliver/fallback
- [x] @router: return "deliver" or "fallback" based on final_score threshold
- [x] Dual-leader comparison: run Flow twice, share retrieved_chunks via state
- [x] Error recovery: try/except in Flow steps → fallback on any failure
- [x] End-to-end test: query → scored response (single leader)
- [x] End-to-end test: query → LeaderComparison (dual mode)
- [x] Architecture diagrams A2 (single query sequence) + A3 (dual-leader sequence)
- [x] **ADR-005: Shared RAG for Dual-Leader Mode** written and committed
- [x] **Checkpoint:** Full pipeline works. Dual-leader comparison produces two scored responses.

### Day 6 — Experiment Day
- [x] Embedding comparison: OpenAI vs MiniLM on same 10 queries → iteration log entry
- [x] Chunking comparison: 500/50 vs semantic markdown split → iteration log entry
- [x] Scoring weight sensitivity: 3 configs (0.4/0.4/0.2, 0.5/0.3/0.2, 0.3/0.5/0.2) × 10 queries
- [x] Pre/post-2018 style evolution: partition Torvalds emails, compute features, plot time-series
- [x] Iteration log: 7 entries (≥3 required; 5 experiments + 2 corrected runs)
- [x] Local vs API LLM experiment → ADR-006 + ADR-007 (two ADRs, both written)
- [x] **Checkpoint:** All experiments complete. Iteration log populated. Style evolution chart generated.

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
