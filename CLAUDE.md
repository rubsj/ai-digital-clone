# CLAUDE.md — P6 v2: Torvalds Digital Clone (Multi-Agent System)

> **Read this file + docs/PRD.md at the start of EVERY session.**
> This is the persistent memory across sessions for working on P6 v2.
> Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P6 — Torvalds Digital Clone: Multi-Agent Style-Matching System
- **Version:** 2.0 (supersedes v1 after Day 8 verification)
- **Location:** `rubsj/06-torvalds-digital-clone` (standalone repo, branch `refactor/p6-multi-agent-rework`)
- **Timeline:** Day 9 through Day 14 (May 26-31, 2026) — 6 days of rework
- **PRD:** `docs/PRD.md` — the product requirements contract (v2)
- **Customized Requirements (the contract above the PRD):** [Notion → P6 Customized v2](https://www.notion.so/36ddb630640a811aae74d4c8d5b10565)
- **Engineering Protocols (canonical standard):** [Notion → Engineering Protocols](https://www.notion.so/35ddb630640a818aa961d003d43c0200). This CLAUDE.md copies the protocol invariants verbatim into the Prompt Discipline, Verification, and Teach-Back sections below, then adds P6 v2-specific specializations. The Notion page is the canonical source; this file is the project-frozen snapshot.
- **Historical engineering record:** `docs/day8-findings.md` — captures the v1 architecture gap that necessitated v2

---

## Why v2 Exists (Critical Context)

Day 8 verification revealed three implementation gaps in v1:

1. **Only 1 of 5 "agents" was a real CrewAI Agent.** ChatStyleAgent was real. RAGAgent, EvaluatorAgent, FallbackAgent, and PlannerAgent were Python functions wrapped in Flow decorators called "agents" in documentation. The 5-agent commitment was not honored in implementation.

2. **Cohere reranker had been silently failing since Day 3** due to environment variable name mismatch (`CO_API_KEY` instead of `COHERE_API_KEY`). All prior evaluation data was vector-only-FAISS, not Cohere-reranked.

3. **The weighted scoring formula was fragile.** `final_score = 0.4×style + 0.4×ground + 0.2×conf` produced a tight ~0.20-wide score distribution across 20 queries, making any threshold either deliver-everything or fallback-everything.

v2 re-derives the architecture from first principles:
- 4 LLM-driven Agents + 3 deterministic Components + Flow orchestrator
- No weighted formula; LLM-driven GatekeeperAgent reasons about routing
- All agent code is real CrewAI Agent with role/goal/backstory
- Vocabulary lock: Python functions are NOT called agents anywhere

Do not implement v1 patterns. When in doubt, re-read PRD §1 (Overview) and PRD §4 (Strategic Decisions).

---

## Vocabulary Lock (per ADR-009)

This is the central architectural decision of v2. The vocabulary must be enforced in code, documentation, and review.

### Agent (LLM-driven)

A class that uses the CrewAI Agent abstraction with `role`, `goal`, `backstory`. Wraps a CrewAI Agent + Task + Crew. Performs LLM reasoning. Lives in `src/agents/`. v2 has 4: CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent.

### Component (deterministic Python)

A class with a `run()` method. Performs measurement, computation, or search. No LLM calls. Lives in `src/components/`. v2 has 3: Retriever, StyleProfileBuilder, ScoringEngine.

### Orchestrator

The CrewAI Flow (`DigitalCloneFlow` in `src/flow.py`) that coordinates Agents and Components through `@start`, `@listen`, and `@router` decorators. The Flow IS the orchestration. There is no PlannerAgent in v2.

### Prohibited Patterns

- Python functions named with "Agent" suffix (e.g., `process_evaluator_agent`)
- Files in `src/agents/` that don't use the CrewAI Agent abstraction
- Files in `src/components/` that call LLM APIs
- Documentation referring to deterministic code as "agent"

CI grep checks enforce these rules. The build fails if a violation is detected.

---

## Architecture Rules (Locked Decisions)

These rules are derived from ADRs and the PRD. They are not subject to re-debate during implementation. If implementation reveals a constraint requiring a rule change, STOP and surface to Ruby.

### Rule 1: 4 Agents + 3 Components + 1 Flow (ADR-009, ADR-014)

Code organization:
```
src/
├── agents/        # 4 LLM-driven CrewAI Agents
├── components/    # 3 deterministic Python classes with run()
├── flow.py        # DigitalCloneFlow orchestrator
├── schemas.py     # All Pydantic v2 models
├── cli.py         # Click CLI (per ADR-008)
└── config.py      # Configuration
```

### Rule 2: Hexagonal Adapter Boundary (ADR-008)

Adapters (`src/cli.py`, `streamlit_app.py`) only import from:
- `src/flow.py` (DigitalCloneFlow, compare_leaders)
- `src/schemas.py` (Pydantic models)
- `src/config.py` (configuration)
- Narrow façades exposed by Agents/Components for setup operations

Direct imports of LiteLLM, FAISS, Cohere, OpenAI from adapter code are PROHIBITED. CI grep enforces this.

### Rule 3: No Weighted Formula (ADR-010)

There is no `final_score` field on `EvaluationResult`. There is no `0.75` threshold anywhere in the codebase. GatekeeperAgent reasons over individual scores; do not compute a weighted combination.

### Rule 4: Structured LLM Output via Instructor (carried from v1)

All LLM responses are validated via Instructor + Pydantic v2. Never use raw `json.loads` on LLM output. Every Agent's output is a typed Pydantic model.

### Rule 5: Style Profile Frozen During Rework (ADR-013)

Do not modify `StyleProfileBuilder` or the 15 features during the Day 9-14 rework. Day-11 re-evaluation determines whether per-leader feature weighting is needed (Day 12+ contingency).

### Rule 6: Cohere Env Var Name (ADR-002 corrected)

Cohere requires `COHERE_API_KEY` (NOT `CO_API_KEY`). This was the silent failure from Day 3 to Day 8. If Cohere call fails, Retriever falls back gracefully to FAISS top-5 with a warning log.

### Rule 7: Plan-File Authority (Prompt Discipline Component 4)

Every Sonnet execution prompt opens with: *"Re-read docs/plans/dayN-plan.md from disk before proceeding. The file on disk is authoritative over any version in your context."*

---

## Writing Rules

> Applies to all written output — ADRs, journal entries, session notes, READMEs, docstrings, code comments, commit messages.

**Voice and stance:**

- Write as a practitioner documenting real decisions, not a consultant producing a deliverable.
- First person is allowed and preferred where natural ("I picked X because", "this burned us on Day 8").
- Never narrate the document's own importance — if it matters, just state what happened.
- No section whose only purpose is to make the author look good.

**Banned constructions:**

- No AI-cadence: no em-dashes, no tricolons ("X, Y, and Z"), no "not just X but Y" parallel structures, no signposting phrases ("The key finding:", "Importantly,", "proven with data").
- No bold emotional category labels ("Easier:", "Harder:", "The Win:") — write plain prose or plain bullets.
- Bullets are at least 1-2 sentences each. Single-fragment bullets are usually compressible into prose.
- Section headers are plain nouns or noun phrases — not action phrases ("Achieving X"), not corporate labels ("Strategic Initiatives").

**ADR-specific:**

- 5 H2 sections only: Context, Decision, Alternatives Considered, Quantified Validation, Consequences.
- No Cross-References, Interview Signal, or Java/TS Parallel as separate H2 sections.
- Java/TS parallel goes inline as a parenthetical at the end of Consequences if relevant — never its own section.
- No Easier/Harder/Portability sub-headers under Consequences.
- Numbers and benchmarks stay where they're contextually relevant — never aggregate into a "Validation" section heading distinct from "Quantified Validation."

**Journal entry / session note specific:**

- Structure-then-draft. Outline the phase, the build, the surprise, the deferred item — then write the prose. No blank-page drafting.
- Per-phase entries are the default. One block per phase. Phase boundary = stop gate boundary.
- Session notes (`docs/session-notes/dayN.md`) are the bridge between phase journal entries and the End-of-Day Author Pass synthesis. They capture what happened during execution, not what should happen.
- Architecture honesty check is non-optional for any Agent or Component work (see Verification Protocol Component 6).

**README-specific:**

- Inverted pyramid. Visual proof above the fold (architecture diagram, hero screenshot, key result). Narrative-first results below. Engineering signals scannable in the second screen.
- No emoji in headers. No Table of Contents. No placeholder links. No horizontal rules between sections except above the footer.
- Images: `<p align="center"><img width="700–800"/></p>` pattern with raw GitHub URLs.
- P1/P2 READMEs are the concrete reference. Match their structure.

**Code comments and docstrings:**

- Comments explain WHY, never what. If the code is readable, no comment needed.
- Banned openers in comments: "Note that", "This ensures", "It's worth mentioning", "Importantly".
- Docstrings: one sentence what, one sentence non-obvious how or why. No parameter narration in prose form (parameter types are in the signature).
- Inline comments for short context. Block comments only for genuinely non-obvious decisions.
- Comment like you're explaining to a teammate at 11pm — direct, no filler.

**Test for whether a sentence belongs:**

If a sentence could have been written without knowing anything specific about this project, delete it.

---

## Prompt Discipline Protocol

> Canonical standard: [Engineering Protocols → Prompt Discipline Protocol](https://www.notion.so/35ddb630640a81458abcf79d51973120). The invariants below are copied verbatim from that page. P6 v2 specializations are noted under each component.

### Component 1: Model Routing (invariant)

**Opus plans. Sonnet executes.** Non-negotiable.

The invariant: Opus handles design, debugging, and analysis. Sonnet handles implementation. If I find myself opening Sonnet for planning or Opus for routine implementation, I have crossed a wire.

**Opus does design, debugging, and analysis:**
- File-by-file planning from PRD and CLAUDE.md
- Schema design
- Non-trivial debugging
- Experiment analysis
- Anything that requires reading the PRD as authoritative

**Sonnet does implementation:**
- Code writing
- File creation, dependency setup
- Test writing
- Running commands
- Routine fixes
- Chart generation, documentation
- Session notes (`docs/session-notes/dayN.md`) — raw phase notes per Verification Component 6

**P6 v2 specialization — detailed responsibility split:**

*Opus (Planning & Architecture for v2):*
- Start of each day: read PRD/CLAUDE.md, produce file-by-file implementation plan
- Design CrewAI Agent role/goal/backstory for each of the 4 Agents
- Design Pydantic schema changes (especially the new RoutingDecision model)
- Plan the Flow refactor: which functions become Agent calls, which Components get extracted
- Debug non-trivial issues (conceptual, not typos)
- Analyze Day-11 evaluation results and decide what findings matter
- Any ambiguity in the PRD

*Sonnet (Implementation for v2):*
- All code writing — implement what Opus planned
- File creation, dependency setup, test writing
- Running commands (uv sync, pytest, evaluation runs)
- Routine fixes (imports, parameters, formatting)
- Chart generation, documentation
- Session notes (`docs/session-notes/dayN.md`)
- Git commits, CLAUDE.md state updates

Sonnet does not write journal entries or handover notes. Those are mine under Teach-Back Author Pass.

### Component 2: Planning Prompts (invariant)

Constraint-heavy. Contain: orientation, source-of-truth references, deliverables, guardrails. Do NOT contain function signatures, pseudocode, pre-solved tradeoffs, or "the code should..." instructions. Cap ~30 lines.

**P6 v2 specialization:** Planning prompts must reference the Architecture Rules section in this CLAUDE.md as the locked-decision boundary. Opus is not allowed to re-debate any of the 7 Architecture Rules; if a plan touches one, Opus must flag it back to me rather than work around it. Specifically: Opus does not propose weighted-formula routing, does not propose merging Agents and Components, does not propose removing the templated failsafe from FallbackAgent.

### Component 3: Execution Prompts (invariant)

Lean. Contain: session context, plan-file authority instruction, Verification Protocol reference, phased stop gates. Do NOT re-type the plan, duplicate implementation guidance, or introduce new design decisions. Typical length 40-60 lines.

**P6 v2 specialization:** Execution prompts must reference the Agent/Component vocabulary explicitly. Sample line: "All work in this session must follow the Agent vs Component distinction (ADR-009). LLM-driven work → src/agents/ with CrewAI Agent abstraction. Deterministic work → src/components/ with run() method. Do not call Components 'Agents' in code, comments, docstrings, or session notes."

This is the explicit guardrail against the v1 failure mode. Do not omit it.

### Component 4: Plan-File Authority Rule (invariant)

Every execution prompt opens with: *"Re-read docs/plans/dayN-plan.md from disk before proceeding. The file on disk is authoritative over any version in your context."*

**P6 v2 specialization:** Plan files for P6 v2 live at `docs/plans/dayN-plan.md` where N is 9 through 14. The Day 9 plan documents the cleanup and refactor scope; Days 10-14 plans cover implementation phases. Naming convention is strict: `day9-plan.md`, `day10-plan.md`, etc.

### Component 5: PRD Coverage Check (invariant)

After Opus produces the plan and before Sonnet executes it, check that every PRD deliverable for the day is covered by at least one phase.

For each PRD section in scope for the day, confirm:
- Which phase covers it
- That the phase's outputs and acceptance criteria actually match what the PRD asks for
- That nothing was silently deferred to a later day without explicit re-scoping

The planning prompt to Opus should include: *"After producing the plan, list every PRD section-N deliverable for this day and confirm which phase covers it. Flag any that have no phase covering them, or that you've deferred. Do not silently defer scope."*

**P6 v2 specialization:** PRD §7 (Deliverables) and PRD §8 (Session Plan) together specify what each Day 9-14 produces. The PRD Coverage Check verifies the day plan covers all PRD §8 scope items for that day. If a deliverable is in PRD §7 but not assigned to a day in PRD §8, that is a PRD gap and should be flagged to me — not silently re-scoped.

### Session Workflow

```
1. Claude.ai (Opus): Socratic Gate — answer 3-5 questions on day's concepts before any plan
2. Claude.ai (Opus): "Read CLAUDE.md and docs/PRD.md. Today is Day [N]. Plan implementation."
3. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
4. Ruby reviews plan for gaps against PRD (PRD Coverage Check)
5. Claude.ai: Ruby drafts execution prompt for Sonnet (Prompt Discipline)
6. Claude Code (Sonnet): "Re-read docs/plans/dayN-plan.md from disk. Execute. Start with [first file]."
7. Sonnet implements Phase 1, runs tests, appends Phase 1 block to docs/session-notes/dayN.md, stops at gate
8. Claude.ai: Ruby pastes Phase 1 terminal output; Claude reviews against Verification Components 3 and 4
9. Claude.ai: Ruby writes Phase 1 Defense (Teach-Back Component 2)
10. Claude.ai: Ruby writes journal entry for Phase 1 (Teach-Back Component 3 Author Pass)
11. Repeat 7-10 for Phase 2, Phase 3, etc.
12. Session close: Sonnet writes Plan-Diff (Verification Component 4)
13. Claude.ai: Ruby writes handover note (Teach-Back Component 3 Author Pass)
```

---

## Verification Protocol

> Canonical standard: [Engineering Protocols → Verification Protocol](https://www.notion.so/35ddb630640a81929b92e02c79c9b9c3). The invariants below are copied verbatim from that page. P6 v2 specializations are noted under each component.

### Component 1: Echo-Back (invariant)

Before any code gets written, Claude Code echoes back the day's deliverables as a numbered plan.

For each item, it states three things:
- The file(s) to create or modify
- The function or class to add or change
- The command or test that will prove it works

If anything is unclear, Claude Code asks before proceeding. No code starts until I reply "approved, proceed."

**P6 v2 specialization:** Echo-back must classify each deliverable as Agent work, Component work, Flow work, schema work, or test work. The classification surfaces vocabulary mismatches before code is written. If an item is labeled "agent" but the implementation would be deterministic, the echo catches it.

### Component 2: Phased Execution with Stop Gates (invariant)

Split the work into 2-4 phases. Maximum 5-6 plan items per phase.

After each phase, Claude Code reports what it did with file:line references. Then it stops. It does not continue until I say "continue."

These stop gates always require explicit approval:
- Destructive operations (delete, overwrite, drop index)
- Architecture decision changes
- New dependencies beyond pyproject.toml
- More than 100 API calls in a single run (cost guard)
- Direct commits to main

**P6 v2 specialization:** Additional stop gates for v2:
- Any change that would create a Python function named with "Agent" suffix
- Any change to the EvaluationResult schema that would add a `final_score` field
- Any change to the routing logic that would introduce a threshold comparison
- Any change to StyleProfileBuilder during the rework (frozen per ADR-013)

These additions exist to prevent regression to v1 patterns.

### Component 3: Verification Contract (invariant)

Before reporting a session complete, Claude Code runs four checks and pastes actual terminal output. Summaries are not enough.

- **Smoke tests.** Specific commands proving each deliverable works.
- **Grep check.** `grep -n "function_name" src/file.py | head -5` shows the code exists where claimed.
- **Test suite.** `python -m pytest tests/ -x --tb=short 2>&1 | tail -10`. All pass. Zero failures.
- **Coverage.** `python -m pytest --cov=src --cov-report=term-missing 2>&1 | tail -20`. At least 90% on new modules.

**P6 v2 specialization — Architecture Honesty Check:**

In addition to the four invariant checks, every session must include an architecture honesty check:

```bash
# Verify all 4 Agents are real CrewAI Agents
grep -l "from crewai import Agent" src/agents/*.py
# Expected: 4 files (clone_agent.py, evaluator_agent.py, gatekeeper_agent.py, fallback_agent.py)

# Verify each Agent has role/goal/backstory
for f in src/agents/*.py; do
  echo "=== $f ==="
  grep -E "(role|goal|backstory)" "$f" | head -5
done

# Verify Components have run() method and no LLM imports
for f in src/components/*.py; do
  echo "=== $f ==="
  grep -E "def run\(" "$f"
  grep -E "(litellm|openai|cohere|instructor)" "$f" && echo "WARNING: Component imports LLM"
done

# Verify no Python functions named with Agent suffix outside src/agents/
grep -rn "def.*Agent[^a-z]" src/ --include="*.py" | grep -v "src/agents/"
# Expected: empty output

# Verify no final_score field in EvaluationResult
grep -n "final_score" src/schemas.py
# Expected: empty output (or only in deprecation comment)
```

These checks must pass before any session is reported complete. If any check fails, the implementation has drifted back to v1 patterns and must be corrected.

### Component 4: Plan-Diff at Session End (invariant)

For every numbered item in the original plan, Claude Code reports one of three states:

- **DONE.** file:line plus a one-sentence description.
- **SKIPPED.** Reason. Whether the skip is acceptable.
- **PARTIAL.** What's missing. Impact.

"Implemented as planned" is not DONE. Without a file:line citation, the claim doesn't count.

**P6 v2 specialization:** Plan-Diff entries for Agent or Component creation must include the Agent/Component classification confirmation. Sample entry: "DONE: src/agents/gatekeeper_agent.py:1-187, GatekeeperAgent class wraps CrewAI Agent with role/goal/backstory, run() method orchestrates Crew kickoff, returns RoutingDecision Pydantic model. Vocabulary verified: this is a real Agent (LLM-driven)."

### Component 5: Anti-Patterns Checklist (invariant)

| Anti-Pattern | Prevention |
|--------------|-----------|
| Report "done" without running the code | Verification Contract (component 3) |
| Skip plan items silently | Plan-Diff (component 4) |
| Happy-path-only tests | Plan specifies edge cases. Plan-Diff catches gaps. |
| Guess environment variables | `echo $VAR | head -c4` before using |
| Proceed past STOP gate | Wait for explicit "continue" |
| Duplicate PRD content instead of reading it | "Read PRD directly. Don't re-derive." |

**P6 v2 specialization — Additional anti-patterns:**

| v2 Anti-Pattern | Prevention |
|-----------------|-----------|
| Call deterministic code "an agent" | Vocabulary lock (this file); CI grep check |
| Add `final_score` field to EvaluationResult | Architecture Rule 3; PR review |
| Add 0.75 threshold to routing | Architecture Rule 3; PR review |
| Skip Cohere when Cohere call fails (silent) | Graceful degradation must log warning; Day-11 eval verifies Cohere actually ran |
| Modify StyleProfileBuilder during rework | Architecture Rule 5; stop gate |
| Use raw json.loads on LLM output | Architecture Rule 4; Instructor everywhere |
| Bypass adapter boundary (direct LLM imports in cli.py or streamlit_app.py) | Architecture Rule 2; CI grep enforces |

### Component 6: Session Notes (invariant)

After each phase clears the stop gate, Claude Code appends 3-5 bullets to `docs/session-notes/dayN.md`. One block per phase.

Each block covers five things:
- **Built.** What got built. file:line.
- **Why.** Why this approach over the alternative we discussed in planning.
- **Surprising.** Anything unexpected during implementation. Edge case, library quirk, dead end.
- **Deferred.** Anything that came up as future work, not done today.
- **ADR candidate.** Any decision worth a separate ADR if not already one.

These are not the final journal entry. They are raw material I read at session close before drafting the journal entry myself.

The file is committed to the repo. It lives at `docs/session-notes/dayN.md`. Same path across all projects.

**P6 v2 specialization — Architecture honesty check in session notes:**

Every session note block for an Agent or Component creation MUST include an explicit architecture confirmation. Format:

```
## Phase 2: GatekeeperAgent implementation

- Built: src/agents/gatekeeper_agent.py:1-187
- Why: LLM-driven routing per ADR-010, replaces v1 weighted formula
- Architecture check: Real CrewAI Agent ✓ (role/goal/backstory at lines 23-45); 
  uses Instructor for RoutingDecision output ✓; no threshold comparison ✓; 
  no `final_score` reference ✓
- Surprising: needed to add prompt instruction "reference specific scores in your reasoning" 
  because initial outputs were generic
- Deferred: cross-leader routing comparison (single-query only in v2 per PRD §5.1.3)
- ADR candidate: no, ADR-010 covers this
```

The architecture check field is non-optional. If a session note for Agent or Component work doesn't include it, the phase is incomplete.

---

## Teach-Back Protocol

> Canonical standard: [Engineering Protocols → Teach-Back Protocol](https://www.notion.so/35ddb630640a81e0a086ce5812853b9b). The invariants below are copied verbatim from that page. P6 v2 specializations are noted under each component.

### Component 1: Socratic Gate (invariant)

At the start of a session, before Claude.ai generates the day's plan, Claude asks me 3-5 questions about the day's concepts.

I answer in my own words. 2-3 sentences each. No lookups.

Claude grades my answers. If something is shaky, Claude pushes back. We close the gap with first-principles reasoning before any plan gets drafted.

No plan generation until the gate clears.

**P6 v2 specialization — question bank:**

Day 10 (Components + first 2 Agents) questions:
- What makes CloneAgent a real Agent vs the v1 ChatStyleAgent that was also real? What's the same, what's different?
- ScoringEngine is a Component, but EvaluatorAgent calls it and is itself an Agent. How is that not a vocabulary violation?
- Why does Retriever stay a Component if the customized requirements call its predecessor RAGAgent an Agent?

Day 11 (GatekeeperAgent + FallbackAgent + integration) questions:
- GatekeeperAgent reasons about routing. What three pieces of evidence does it use that the v1 formula could not?
- What's the failure mode if FallbackAgent's templated failsafe doesn't exist?
- Why is the routing decision LLM-driven but the order of Flow steps deterministic?

Day 12 (E2E eval) questions:
- What's the difference between the E1 floor and E2 target? Why have both?
- Why is OOD fallback at 100% the hard gate (zero hallucinations) but in-domain deliver rate is a target band?
- If Day-11 measures KH deliver rate 20% below Torvalds, what does ADR-013 say to do?

These questions probe whether I can articulate v2's architectural distinctions, not just recite them.

### Component 2: Phase Defense (invariant)

This is a debrief, not a closed-book exam. By the time the phase finishes, I have evidence available: the terminal output, the Phase Notes block Claude Code wrote, and my memory of the planning conversation. The defense uses all of it.

**Per-phase workflow:**
1. Claude Code finishes the phase. Reports file:line. Appends Phase Notes block to `docs/session-notes/dayN.md`. Stops at gate.
2. I paste the Claude Code terminal output into Claude.ai. Claude reviews it against Verification Protocol Components 3 and 4 for this phase's items.
3. If verification passes, I write the Phase Defense.
4. Claude grades the defense.
5. Defense clears.
6. I read the full Phase Notes block again as source material.
7. I draft the journal entry (Author Pass, Component 3 below).

**How I write the defense.**

Pick 2-3 prompts from the menu below. At least one from Category A. At least one from B, C, or D.

Each answer is 1-3 sentences. Total defense length: 4-8 sentences.

**Category A — What got built (always at least one).**
- What does this phase add to the system that wasn't there before? Name the new capability in one sentence.
- Which files changed and what each one is responsible for now.
- What's the entry point for this new capability? How would a caller invoke it?

**Category B — Design and tradeoffs (at least one if a real decision was made).**
- Why this approach over the alternative we discussed in planning.
- What tradeoff did this phase accept? (Latency vs accuracy, simplicity vs flexibility, etc.)
- Which architecture rule did this phase implement or constrain against?
- Why this library, this pattern, this data structure over the obvious alternative.

**Category C — Risk and edge cases (pick one when applicable).**
- What's the most likely way this code breaks in production?
- Which test case would I write next if I had more time?
- What's the dumbest input that would crash this?
- What downstream code now depends on this in a way that wasn't true before?

**Category D — Connection (pick one when applicable).**
- How does this phase connect to a previous phase today or a previous day?
- What does this phase set up for the next phase?
- Has a similar problem come up in a prior project? How was it solved differently there?

**P6 v2 specialization — v1 drift check in Phase Defense:**

For phases that touch Agent or Component code, the Phase Defense must include one prompt from this v2-specific category:

**Category V — v1 drift check (mandatory for Agent/Component phases):**
- Did this phase introduce any pattern that resembles v1? (e.g., Python function called "agent," weighted formula, threshold comparison, deterministic code in src/agents/)
- If yes, was that intentional and documented? Or did it slip through?
- What would have caught it if I hadn't asked?

This category exists because v1's failure mode was silent drift — the architecture documentation said one thing while the code did another. The Author Pass debrief is the moment to catch drift before it gets to the next session.

### Component 3: End-of-Day Author Pass (invariant)

I write two artifacts: the journal entry for each phase, and the session handover note.

The raw material comes from `docs/session-notes/dayN.md`. Claude Code writes that file under Verification Protocol Component 6.

**Before drafting either artifact, I read `docs/session-notes/dayN.md` end to end.** The session notes are my source. They are not my draft.

**The learning journal entry — per phase, right after Phase Defense clears.**

What it does: captures what I learned in this phase, in my own words, in the format I'll later use for interview prep. STAR for behavioral material. First-principles framing for technical material.

How it gets written:
- I read the Phase Notes block for this phase from `docs/session-notes/dayN.md`.
- I draft the journal entry in Claude.ai. 100-200 words per phase, minimum.
- Claude redlines for vagueness, fabricated certainty, missing tradeoffs, weak verbs, and crucially: passages that paraphrase the session notes instead of synthesizing.
- I revise once.
- I push the revised entry to Notion.

The synthesis bar: my journal entry must add something the session notes do not have. Usually one of three things:
- The cross-phase pattern (how this phase connects to earlier phases in the day or earlier days in the project)
- The analogy (Java/TS parallel, prior project parallel, or a real-world analog)
- The connection to prior projects (P2 evidence used here, or P5 pattern applied)

If my journal entry is recognizably a paraphrase of the session notes, the entry doesn't count. Claude pushes back and I redraft.

**The handover note — at session close.**

What it does: keeps continuity across sessions. The next session opens by reading the previous session's handover.

How it gets written:
- I read the full `docs/session-notes/dayN.md` file.
- I read the Plan-Diff from the Verification Protocol output.
- I draft the handover in Claude.ai. 10 minutes maximum.
- Claude redlines for vagueness, fabricated certainty, missing tradeoffs, weak verbs.
- I revise once.
- I push the revised version to Notion.

Claude Code does not write the journal entries. Claude Code does not write the handover note. Both are mine.

**P6 v2 specialization — v1 drift verification in Author Pass:**

Before closing each session, the handover note must explicitly answer: "Did today's work drift back toward v1 patterns at any point? If yes, what was caught and how?"

This question exists to make drift detection a session-close ritual, not just a phase-by-phase reactive check. If three consecutive sessions answer "no drift detected," the question stays in the template but its answer can be short. If any session answers "yes," the handover documents what happened and what would have caught it earlier.

The point is to keep the v1 failure mode visible during the rework. The cost of forgetting is high; the cost of one extra line in the handover note is trivial.

---

## Developer Context

**Background:** Java/TypeScript developer with 20+ years of software engineering experience, learning Python through the portfolio sprint. Completed P1–P5 before P6. Hardware: Mac Pro M5 Max, 128GB unified RAM, 40 GPU cores. IDE: VS Code + Claude Code extension.

**Learning priority:** Depth over speed. Understanding "why" before "how." Rabbit holes are encouraged if they produce genuine insight — surface them rather than skip them.

**Python-vs-Java surfacing rule:** When implementing a pattern that differs meaningfully from Java/TS, the agent commenting the code or writing the journal entry calls out the difference inline as a brief parenthetical. Not a teaching essay — a one-line orientation. Example: `# WHY: Pydantic field_validator runs on assignment, not just at construction (like Bean Validation's @Valid on a setter, not a constructor-only check)`.

### Patterns Carried Forward from P1–P5 (Reuse, Don't Re-Justify)

| Pattern | Source | P6 v2 Application |
|---------|--------|-------------------|
| Pydantic v2 models + validators | P1–P5 | All schemas in `src/schemas.py`: EmailMessage, StyleProfile, EvaluationResult, RoutingDecision, CloneState, etc. |
| Instructor + auto-retry for structured LLM output | P1–P5 | All 4 Agents use Instructor; never raw `json.loads`. Auto-retry on validation failure. |
| JSON file cache (MD5 key) | P1/P2/P4/P5 | Cache OpenAI embeddings to `data/cache/embeddings_openai.json`. Never re-call OpenAI for same input. |
| FAISS IndexFlatIP + L2 normalization | P2/P5 | Retriever's FAISS layer. ALWAYS normalize before `add()` and `search()`. |
| OpenAI text-embedding-3-small via LiteLLM | P2/P5 | Primary embedding in Retriever (26% better than MiniLM, per P2). |
| Cohere Rerank API (top-20 → top-5) | P2/P5 | Retriever's 2-stage retrieval. 20% lift proven in P2. v1 had silent failure due to env var name (corrected in v2 — see Architecture Rule 6). |
| LiteLLM for LLM routing | P5 | All Agent LLM calls go through LiteLLM. Provider-agnostic. |
| matplotlib/seaborn/plotly charts | P1–P5 | 8 portfolio charts in `results/charts/` per PRD §7.6. |
| Click CLI | P2/P5 | 5 commands per PRD §7.3: learn, index, query, compare, evaluate. |
| Rich progress bars | P2/P5 | Email parsing and chunk indexing progress display. |
| ADR template (5 H2 sections) | P1–P5, refined in v2 | See Writing Rules → ADR-specific. Context, Decision, Alternatives, Quantified Validation, Consequences. |
| `yaml.safe_load()` exclusively | P5 | Config loading in `src/config.py`. NEVER `yaml.load()`. |
| Hexagonal adapter boundary | P5 | CLI and Streamlit import only from `flow.py`, `schemas.py`, `config.py`. See Architecture Rule 2. |

Reuse these silently. Don't re-justify why we use Pydantic when adding a new model — the decision was made in P1.

### New for P6 v2 (Understand These Before Day 10)

**CrewAI Flow** — `from crewai.flow.flow import Flow, listen, start, router`. Event-driven orchestration with `@start()`, `@listen(method)`, `@router()` decorators. `Flow[CloneState]` carries a typed Pydantic state across steps. Think of it as a Spring event bus with typed state — each step listens for the previous step's completion, and `@router` enables conditional branching (like Java's `switch` on the routed string). The Flow IS the orchestration; v2 has no separate PlannerAgent.

**CrewAI Agent + Task + Crew** — `from crewai import Agent, Task, Crew`. v2 uses this abstraction for all 4 LLM-driven Agents (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent). Pattern per Agent: a class wraps an `Agent` instance (with role, goal, backstory), constructs a single `Task`, runs it via a single-agent `Crew`, and parses the output into a Pydantic model via Instructor. Components do NOT use this abstraction — they're plain Python classes with `run()` methods.

**Python `mailbox` module** — `import mailbox; mbox = mailbox.mbox("path/to/file.mbox")`. Parses mbox email archives into message objects. `msg["From"]`, `msg["Subject"]`, `msg.get_payload()`. Like Java's `MimeMessage` parsing but lower-level. Watch for encoding issues — LKML archives span decades and contain mixed encodings. The v1 StyleProfileBuilder's 8-step email cleaning pipeline handles this; do not rewrite it.

**Hand-crafted style feature extraction** — 15 numerical features from text (11 base + 4 LKML-specific): punctuation frequency, vocabulary richness, capitalization ratio, etc. All normalize to `[0, 1]` for cosine similarity. This is NLP feature engineering, NOT embedding-based. The features have human-readable meaning per dimension; LLM embeddings would not (see ADR-003).

**Cosine similarity on feature vectors** — `from numpy import dot; from numpy.linalg import norm; sim = dot(a, b) / (norm(a) * norm(b))`. For 15-dim style vectors and embedding vectors alike. NOT the same conceptual operation in both cases: on style vectors, each dimension is interpretable; on embedding vectors, dimensions are opaque latent features.

**Incremental learning with alpha** — `updated = (1 - alpha) * current + alpha * new`. Weighted average for streaming updates. `alpha=0.3` default. Like an exponential moving average (EMA) in trading systems — you never recompute from scratch.

**Three-layer evaluation methodology** — Unit (CI), Integration (per Agent/Component with recorded LLM responses), System (`cli evaluate` against the 20-query v2 set). The Day 8 finding (only Layer 3 catches end-to-end behavior gaps) shaped this. Do not collapse layers. See ADR-016.

---

## Code Conventions

### Python Version and Tooling

- Python 3.12+
- `uv` as package manager (`uv sync`, `uv run`, `uv add`)
- `ruff` for linting and formatting (line length 100, double quotes)
- `mypy` for type checking on `src/` and `tests/`

### Pydantic v2 Patterns

Ruby's background is Java/TypeScript. Pydantic v2 is the closest Python equivalent to TS interfaces/zod.

**Use Pydantic v2 syntax everywhere:**
```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class RoutingDecision(BaseModel):
    decision: Literal["deliver", "fallback"]
    reasoning: str = Field(min_length=20, description="LLM-generated reasoning")
    trigger_reason: str | None = None
    
    @field_validator("reasoning")
    @classmethod
    def reasoning_references_scores(cls, v: str) -> str:
        # Could enforce that reasoning mentions specific scores
        return v
```

**Do not use Pydantic v1 syntax** (`@validator`, `Config`, `parse_obj`). Use `model_validate`, `model_dump`, etc.

### Instructor + LLM Output Pattern

All LLM outputs validated via Instructor:

```python
import instructor
from litellm import completion

client = instructor.from_litellm(completion)

result: RoutingDecision = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=RoutingDecision,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_retries=2,
)
```

Never use raw `json.loads` on LLM output. Instructor's retry logic handles parsing failures.

### Type Hints

All public functions and methods have type hints. Pydantic models are the preferred input/output types.

```python
# Good
def score(
    response_text: str,
    chunks: list[RetrievalResult],
    style_profile: StyleProfile,
) -> tuple[float, float, float]:
    ...

# Avoid
def score(response_text, chunks, style_profile):
    ...
```

### Imports Inside Agents/Components

**Agent files import:**
- CrewAI: `from crewai import Agent, Task, Crew`
- Instructor + LiteLLM
- Pydantic schemas from `src/schemas.py`
- Components if delegation needed (e.g., EvaluatorAgent imports ScoringEngine)

**Component files import:**
- Standard library, numpy, FAISS, etc.
- Schemas from `src/schemas.py`
- NEVER: LiteLLM, OpenAI, Cohere, Instructor (these are LLM tools; Components don't use LLMs)

If a Component file imports LLM libraries, the architecture honesty check will fail.

### Commenting Style

Brief, declarative, in own words. No AI-cadence. No tricolons. No em-dashes.

```python
# Good
# Cosine similarity via dot product on L2-normalized vectors
similarity = np.dot(query_vec, chunk_vec)

# Avoid (AI-cadence)
# This function computes cosine similarity — not just any similarity,
# but a normalized one — to capture the directional alignment...
```

### Test Naming and Structure

```
tests/
├── unit/                           # Pure unit tests, mocked dependencies
├── integration/                    # Per-Agent with mocked LLM responses (recorded)
└── e2e/                            # Full pipeline via cli evaluate
```

Test names follow `test_<unit>_<scenario>` pattern. Use `pytest.mark.parametrize` for table-driven tests.

LLM responses in integration tests are recorded once and replayed in CI (via VCR or similar). Live LLM calls in CI cost money and introduce flakiness.

---

## File Structure Rules

### What Lives Where

| Directory | Purpose | What goes here |
|-----------|---------|----------------|
| `src/agents/` | LLM-driven Agents | 4 Agent classes, each wrapping CrewAI Agent/Task/Crew |
| `src/components/` | Deterministic Components | 3 Component classes with `run()` method |
| `src/` (root) | Orchestration + adapters | flow.py, schemas.py, cli.py, config.py, visualization.py |
| `src/rag/`, `src/style/`, `src/evaluation/`, `src/fallback/` | Low-level utilities | Helper modules used BY Components and Agents; not directly imported by adapters |
| `data/` | Persistent data | FAISS index, StyleProfile JSONs, evaluation query set, mbox files |
| `tests/` | All tests | unit/, integration/, e2e/ subdirs |
| `docs/` | Documentation | PRD.md, CLAUDE.md, day8-findings.md, session-notes/, plans/, adr/, architecture/ |
| `results/` | Evaluation outputs | charts/, evaluation_*.json |
| `scripts/` | One-off utilities | anchor_trajectory.py, inspect_q12_chunks.py |
| `streamlit_app.py` | Streamlit entry point | Top-level; imports only from src.flow, src.schemas, src.config |

### What Does NOT Go in Each Directory

- `src/agents/` does NOT contain Python functions named with Agent suffix
- `src/components/` does NOT contain LLM library imports
- `src/cli.py` does NOT directly import LiteLLM, FAISS, OpenAI, Cohere
- `streamlit_app.py` does NOT directly import LiteLLM, FAISS, OpenAI, Cohere
- `docs/` does NOT contain code (only markdown documentation)

---

## CLI Commands

The CLI has 5 commands (unchanged from v1; internals refactored):

```bash
# Build StyleProfile for a leader from mbox archive
cli learn --leader torvalds --mbox data/raw/lkml.mbox

# Build FAISS index from textbook corpus
cli index

# Run a single query through the pipeline
cli query "How does kernel scheduling work?" --leader torvalds

# Run a query against both leaders side-by-side
cli compare "How does kernel scheduling work?"

# Run the full evaluation query set and write results
cli evaluate --query-set data/eval/queries.json --output results/evaluation_dayN.json
```

CLI commands stay identical across v1 → v2. Internal implementation changes; user-facing surface does not.

---

## Day-by-Day Plan Reference

Day plans live at `docs/plans/dayN-plan.md` where N = 9, 10, 11, 12, 13, 14.

Each day's plan is created by Opus at the start of that session, reviewed by Ruby against the PRD §8 (Session Plan) scope for that day, then handed to Sonnet for execution.

The plan file on disk is authoritative (per Prompt Discipline Component 4). Sonnet re-reads from disk at each phase boundary.

---

## What to Ask Ruby vs Decide Yourself

### Decide Yourself (Sonnet)

- Implementation details within a phase that the plan covers
- Variable names, helper function organization, internal docstring wording
- Test case naming and structure
- Routine fixes during phase execution (typos, missing imports, formatting)
- File paths within the established directory structure
- Choice of error messages and log output formatting

### Ask Ruby (stop and surface)

- Any architecture rule conflict (see Architecture Rules section)
- Any new dependency not already in pyproject.toml
- Any pattern that resembles v1 (Python functions as agents, weighted formula, threshold comparison)
- Any change to the agent count or component count
- Any change to the EvaluationResult schema beyond what the plan specifies
- Any decision that would change a documented ADR
- Any test that would require live LLM calls in CI
- Any deletion of files not in the cleanup list (PRD §11.1)
- Any commit to main (always require explicit "commit to main" instruction)
- Anything that "seems like the right thing but isn't in the plan"

The cost of asking is one round trip. The cost of silent drift is days of rework. Default to asking.

---

## Current State

> Update this section before ending each session. Append; don't overwrite.

### Session 1 — Day 9 (May 26, 2026) — Foundation

**Status:** In progress as of PRD/CLAUDE.md generation

**Completed:**
- v1 customized requirements deleted from Notion
- v2 customized requirements published to Notion: https://www.notion.so/36ddb630640a811aae74d4c8d5b10565
- PRD v2 drafted (this session)
- CLAUDE.md v2 drafted (this session)

**Pending (Day 9 remaining):**
- Create `refactor/p6-multi-agent-rework` branch
- Execute cleanup deletions per PRD §11.1
- Rename `data/eval/queries_v2.json` → `data/eval/queries.json`
- Commit PRD.md and CLAUDE.md to the rework branch
- Write new ADRs: 009, 010, 014, 015, 016
- Light edits to ADRs 001-008
- Create Traceability Matrix Notion page (separate task)

**Next session opens with:**
- Day 10 plan (Components + first 2 Agents)
- Opus to read PRD §5.2 (Components) and §5.1.1-2 (CloneAgent + EvaluatorAgent) and produce file-by-file implementation plan

---

*End of CLAUDE.md v2. Lives at `docs/CLAUDE.md` in the repo. Frozen snapshot of the Engineering Protocols Notion page as of 2026-05-26, with P6 v2 specializations. The Notion page is the canonical source; this file is project-frozen.*
