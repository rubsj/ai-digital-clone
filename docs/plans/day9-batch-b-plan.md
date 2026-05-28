# Day 9 ADR Batch B Plan

Plan for the second and final batch of v2 ADR work, to be drafted by Opus in a follow-up session. This file is planning input only. No ADR files are written this session. Stop after this plan is reviewed.

Sources read for this plan: `docs/PRD.md` (full, with attention to §2, §3, §5.1, §7.5.2, §10, §12.3), `CLAUDE.md` (Writing Rules and ADR format contract), `docs/day8-findings.md`, `docs/notes/adr-batch-b-context.md` (substantive source for Groups 1 and 2), `docs/plans/day9-plan.md` (Batch A locked plan, used for format and depth), and the eight existing ADR files in `docs/adr/` (001 through 008 for current state; 009 through 012 for Batch A voice and the on-disk cross-reference targets).

Batch B covers 12 items:

- Group 1: four new ADRs (013, 014, 015, 016)
- Group 2: two full rewrites in place (001, 007)
- Group 3: six light edits (002, 003, 004, 005, 006, 008)

---

## Format and conventions (applies to every item below)

These hold for all 12 items so they are not repeated per ADR.

**Five H2 sections only, in order:** Context, Decision, Alternatives Considered, Quantified Validation, Consequences. No Cross-References section, no Interview Signal section, no Java/TS Parallel section. Java/TS parallel is an inline parenthetical at the end of Consequences when it lands naturally, omitted otherwise. The cross-references listed per item in this plan are planning aids for the author; they are folded into Context or Consequences prose in the ADR, not given their own heading.

**Banned constructions** (CLAUDE.md Writing Rules): no em-dashes, no tricolons ("X, Y, and Z"), no "not just X but Y", no signposting ("The key finding:", "Importantly,"), no bold emotional category labels ("Easier:", "The Win:"). First person is allowed and preferred. Bullets are at least one to two sentences each.

**Tricolons in this plan are shorthand, not prose to copy.** Several Decision and Quantified Validation outlines below pack three-element comma series (for example "mbox parse, 15-feature extraction, profile build") as compact planning notes. The CLAUDE.md tricolon ban applies to the ADR output, not to this plan file. When rendering these into ADR prose, unpack each three-element series into a two-element form, separate sentences, or a named-list construction (for example "X and Y. It also Z." or "X, Y. The third element is Z."). Do not transcribe the plan's comma series verbatim into the ADR.

**Codebase substance check (per ADR, at writing time).** Earlier humanization passes may have stripped substantive material from existing ADRs. For each ADR in Batch B, before finalizing the prose:

1. Read the on-disk ADR file (for rewrites and light edits) or the context brief and PRD references (for new ADRs) as the baseline.
2. Read the relevant source files in `src/` and tests in `tests/` for the area the ADR covers. For ADR-001, that is `src/flow.py` and the agent classes in `src/agents/`. For ADR-007, that is every file under `src/agents/` and `src/components/`. For ADR-002, that is `src/components/retriever.py` and the Cohere reranking code. The plan section per ADR notes the relevant paths where it can.
3. For each of the five ADR sections (Context, Decision, Alternatives Considered, Quantified Validation, Consequences), ask: is there empirical material in the codebase that would make this section's existing argument concrete and is not currently in the ADR? Examples worth adding: actual function signatures that prove the Agent versus Component distinction, real measured numbers from tests or experiment runs, specific exception types raised by the failsafe path, line counts that prove the templated failsafe is small, the actual list of imports in `src/components/` that proves the no-LiteLLM rule.
4. Add only material that strengthens an argument that is already in the ADR. Do not add new arguments. Do not expand scope. Do not introduce decisions that were not in the context brief or the v1 ADR. If material is genuinely new (a decision the ADR does not currently document), flag it for human review rather than silently incorporating it.
5. For each addition, note briefly in the planning notes (not in the ADR prose) why the addition strengthens that section. One line is enough.

The guardrail is: codebase substance strengthens existing arguments. It does not expand scope or invent decisions. If the codebase reveals that an ADR's existing claim is wrong or incomplete in a substantive way, surface that to the human; do not silently rewrite the claim.

**Sequencing note.** The v2 implementation lands Days 10–11 per PRD §8. At Day 9 writing time, the substance check applies fully to: (a) light edits (002, 003, 004, 005, 006, 008), since v1 code is on disk, and (b) v1 evidence cited in the rewrites (ADR-001 citing v1 `src/flow.py` step structure, ADR-007 citing v1 LLM-scoring path). For v2-specific empirical material (the actual import lists in `src/components/`, the real Agent class signatures, the failsafe exception types in the rewritten FallbackAgent), the substance check is partial at Day 9: cite PRD §5.1, §5.2, §5.4, §5.5 as specification evidence, cite Day 8 measured data for the baseline regression evidence, and accept that some v2 code-level material is forward-looking specification rather than reflecting code on disk. This is not a weakness of the ADRs; ADRs as up-front design records, validated by subsequent implementation, are the engineering practice these documents are meant to capture. The Day 14 codebase audit (PRD §12.5) is the verification pass that checks ADR claims against the implemented code.

**Header style decision.** The four new ADRs (013 through 016) use the Batch A minimal header that 009 through 012 already use on disk:

```
# ADR-0NN: Title

**Status:** Accepted
**Date:** 2026-05-26

## Context
```

The two rewrites (001, 007) keep their existing fuller header block (Project, Category, Status, Date, horizontal rule) because the surrounding light-edit ADRs all carry it and a rewrite-in-place should not gratuitously reformat the header. The rewrites update the Status line to read `Accepted (rewritten for v2, 2026-05-26)` and keep the original Date line so the v1 authorship date is preserved as record (the Date line is the v1 authorship date, not the rewrite date; ADR-001 stays `2026-04-03`, ADR-007 stays `2026-04-27`). This header inconsistency between the two batches is real; it is called out in the closing summary as a thing for Ruby to confirm rather than something I should silently normalize across all 16 files.

The Category property must be updated, or the rewrites carry a v1 category string under a v2 body. The v1 files use a single-word Category format (ADR-001 is `Orchestration`, ADR-007 is `Evaluation`), not a compound slash format, so the slash-preservation pattern does not apply. Set both rewrites to `Category: Architecture`: the orchestration-pattern decision in ADR-001 and the LLM-placement decision in ADR-007 are both architectural concerns rather than separate categories. ADR-007's old `Evaluation` category is stale under the v2 body (the rewrite is about LLM roles across the pipeline, not evaluation), which is the concrete reason this property cannot be left at its v1 value.

**Date stamp.** New ADRs and the rewrite Status notes use 2026-05-26 (Day 9), matching ADR-009 through 012 and PRD §8.

**Filenames.** New ADRs:
- `docs/adr/ADR-013-style-profile-frozen.md`
- `docs/adr/ADR-014-agent-component-inventory.md`
- `docs/adr/ADR-015-post-rework-eval-acceptance-criteria.md`
- `docs/adr/ADR-016-evaluation-methodology-three-layer.md`

ADR-001 keeps its current filename (`ADR-001-crewai-flow-pattern.md`), which still fits the rewritten title. ADR-007 is renamed from `ADR-007-llm-evaluation-scoring-viability.md` to `ADR-007-llm-roles-in-the-pipeline.md` as part of the rewrite, because the v1 filename no longer matches v2 content. The execution session does the `git mv` rename and does not leave a stub redirect at the old name.

---

# Group 1 — New ADRs

## ADR-013: Style Profile Frozen, Re-Measured Day 11

### 1. Context (outline)

Day 8 v2 evaluation surfaced a measurement asymmetry in style scoring: mean style score 0.91 for Torvalds against 0.84 for Kroah-Hartman across the same 14 in-domain queries (the precise per-record figures in day8-findings.md are 0.9025 Torvalds, 0.8355 KH across the 11 scored records). The asymmetry is a property of how the 15 hand-crafted features behave on each leader's corpus, because the feature set was derived from Schneider et al. on Torvalds-distinctive LKML signals and applied uniformly to both leaders. It is design provenance, not a bug. The v2 architecture changes how the style score is used: GatekeeperAgent reasons over it qualitatively instead of feeding it into a 0.75 threshold at 40% weight. The open question is whether qualitative routing absorbs the asymmetry or whether it persists, and what to commit to during the rework before that is known.

### 2. Decision

Freeze StyleProfileBuilder and the 15 features during the rework (Days 9 through 10). Re-measure the asymmetry on Day 11 under the new architecture. Per-leader feature weighting (the Option S2 path from day8-findings.md) becomes scheduled Day 12 to 13 work only if Kroah-Hartman's in-domain deliver rate is more than 20 percentage points below Torvalds's at Day 11. Below that gap the asymmetry is treated as measurement noise; above it, as signal that earns the engineering investment.

### 3. Alternatives Considered

- **Do S2 now (per-leader feature weighting during the rework).** Rejected. The rework is already a six-day investment, and S2 might be wasted effort if GatekeeperAgent's qualitative reasoning absorbs the asymmetry. Testing the new architecture first, then adding S2 if measurement demands it, is the cheaper sequence and the stronger interview narrative (an evidence-based call rather than preemptive complexity).
- **Drop the numeric style_score from EvaluationResult and let GatekeeperAgent read sample emails directly (Option S3).** Rejected. It removes the radar-chart visualization, which is a portfolio deliverable (PRD §2.10 chart 1), eliminates a quantitative metric from PRD §2.2, and swaps measurable signal for unconstrained judgment. Too radical for rework scope.
- **Ship as-is with the gap documented and no commitment to fix.** Rejected. A documented gap with no trigger is a weaker engineering signal than a documented gap with an explicit threshold and scheduled contingency. The 20-point trigger plus the Day 12 to 13 commitment turns hand-waving into a plan.

### 4. Quantified Validation (evidence to cite)

- Day 8 v2 figures: Torvalds style mean 0.9025, KH 0.8355, delta +0.067 favoring Torvalds, from the Finding 1 table in `docs/day8-findings.md`. Cite the 11-scored-record basis honestly rather than rounding to "0.91 vs 0.84" without the sample note.
- The KH compensation detail (KH groundedness mean 0.6673 against Torvalds 0.5913) so final means came out near-identical under the old formula. This is why the asymmetry did not fully propagate to v1 deliver decisions and why it may behave differently under qualitative routing.
- The 20-point threshold is a judgment call, not a derivation. State that plainly: small gaps over a 14-query in-domain set are within measurement noise; gaps over 20 points across 14 queries are unlikely to be random.
- PRD §2.1 Day-11 acceptance criteria and PRD §4.9 reference this freeze-and-remeasure plan.

### 5. Consequences

The rework stays focused on architectural change rather than feature engineering. Style scoring remains one input GatekeeperAgent reasons over, not a primary routing driver. Day 11 re-measurement is committed work with a defined trigger, not aspiration. If the trigger fires, Days 12 to 13 carry per-leader weighting; if it does not, the resolved asymmetry is documented as a Day 11 finding and nothing further is built. No Java/TS parallel; the context brief notes it is not natural here, so omit it.

### 6. Cross-references

Depends on ADR-010 (LLM-driven routing is the mechanism expected to absorb the asymmetry) and ADR-003 (the hand-crafted feature design that produces the asymmetry). Referenced by PRD §4.9 and §2.1. ADR-003's light edit (Group 3) adds a one-line pointer back to this ADR, so the link is bidirectional.

---

## ADR-014: Agent and Component Inventory

### 1. Context (outline)

ADR-009 defined the reusable vocabulary criterion (Agent is LLM-driven via the CrewAI Agent abstraction; Component is deterministic Python with a `run()` method). That criterion is portable across projects and deliberately abstract. P6 v2 still needs a concrete inventory: which specific units exist, why each one is classified the way it is, and what the split decision was. Without this ADR the architecture lives only in `src/agents/` and `src/components/` directory listings, which is implementation rather than a decision record. A reviewer cannot reconstruct why Retriever is a Component or why GatekeeperAgent is an Agent from a file tree alone.

### 2. Decision

P6 v2 has exactly 4 Agents, 3 Components, and 1 Flow orchestrator. Agents (LLM-driven): CloneAgent (response generation in leader voice), EvaluatorAgent (hybrid, wraps ScoringEngine for math and adds an LLM call for explanation and flags), GatekeeperAgent (LLM routing decision over scores, chunks, and response), FallbackAgent (LLM leader-appropriate fallback with templated failsafe). Components (deterministic): Retriever (FAISS plus Cohere rerank), StyleProfileBuilder (mbox parse, 15-feature extraction, profile build), ScoringEngine (cosine style math, sentence-level groundedness, completeness-based confidence). Orchestrator: DigitalCloneFlow, a CrewAI Flow subclass with `@start`, `@listen`, and `@router` decorators. The Flow is not an Agent and there is no PlannerAgent.

### 3. Alternatives Considered

- **Make Retriever and ScoringEngine Agents.** Rejected. Both are deterministic pipelines. Retriever's chain (embed, FAISS, rerank) has no LLM reasoning step; ScoringEngine is cosine math. Wrapping either in the CrewAI Agent abstraction adds prompt scaffolding with no autonomy benefit and would fail the ADR-009 criterion.
- **Make DigitalCloneFlow an Agent (a PlannerAgent on top of the Flow).** Rejected. The Flow already owns step order, state, and routing, and there is no per-step LLM reasoning in the Flow itself. Adding a PlannerAgent over it is the "agent that manages agents" anti-pattern, and it was part of the v1 confusion that v2 exists to remove.
- **Collapse CloneAgent and FallbackAgent into one Agent with a mode parameter.** Rejected. They have different goals (generate versus gracefully decline), different prompts, and different evaluation expectations. Merging them obscures responsibility and complicates testing. Two Agents, two prompts, two test suites is cleaner.

(The two-ADR split itself, ADR-009 for the criterion and ADR-014 for the concrete inventory, is the adopted structure and is explained in Context rather than listed as a rejected alternative.)

### 4. Quantified Validation (evidence to cite)

- PRD §5.1 (agent specifications) and §5.2 (component specifications) are the implementation specification this ADR justifies; cite them as the authoritative inventory source.
- PRD §3.2 and §3.3 (architecture at a glance, why Agents vs Components) supply the count and the classification principle.
- PRD §12.2 (v1-to-v2 code mapping) is the evidence that the inventory is a deliberate reclassification: rag_agent.py became a Component (Retriever), the two Python-function "agents" (evaluator_steps, fallback_steps) became real Agents, and the Flow `@router` string-return became GatekeeperAgent.
- This ADR rests largely on architectural reasoning; the load-bearing evidence is the PRD §5.1/§5.2 specification and the §12.2 mapping, not a measurement.

### 5. Consequences

The `src/` directory follows the inventory exactly: `src/agents/` has four files, `src/components/` has three, and nothing outside those directories is named like an Agent. Any new Agent or Component in future work has to pass the ADR-009 criterion check before it is added, so the inventory is a gate rather than a snapshot. Java/TS parallel lands naturally and the context brief suggests it: a Spring application distinguishes stateless business logic from worker components, and the cleaner mapping here is "is this a strategy that reasons or a worker that computes." Include it as a one-line parenthetical at the end of Consequences only if it reads cleanly; per the going-forward Writing Rule, drop it if it needs explanation to land.

### 6. Cross-references

Builds directly on ADR-009 (the criterion). This is the central dependency and must be on disk; confirmed present (`docs/adr/ADR-009-agent-vs-component-distinction.md`). References ADR-011 for the EvaluatorAgent hybrid detail and ADR-001 (rewritten) for the Flow-is-orchestrator point. CLAUDE.md Architecture Rule 1 cites both ADR-009 and ADR-014.

---

## ADR-015: Post-Rework Evaluation Acceptance Criteria

### 1. Context (outline)

After the rework completes on Day 10, Day 11 runs the v2 evaluation query set (14 in-domain plus 6 out-of-domain per leader) through the new architecture. The question is what counts as success. Without criteria locked before the data lands, the post-rework evaluation degrades into "look at the numbers and decide if they feel good," which is post-hoc rationalization. The original PRD aspirational target (around 70% in-domain deliver) was set against the v1 architecture and never measured under v2, so using it as the bar risks declaring failure on numbers a well-functioning new architecture may legitimately never hit.

### 2. Decision

Three-tier acceptance criteria, locked in advance of Day 11. E2 (target): in-domain deliver rate at or above 55% per leader, OOD fallback at 100%, zero hallucinations on category-5 queries; if E2 is met, ship P6. E1 (floor): in-domain deliver rate at or above 39% per leader, matching the v2 baseline measured Day 8; below E1 the architecture has regressed and P6 does not ship. Between E1 and E2 is a judgment call: document the deliver rate as the system's operating point and ship if the other criteria pass. OOD fallback at 100% is non-negotiable for shipping regardless of where the in-domain rate lands.

### 3. Alternatives Considered

- **Use the PRD aspirational target (around 70% in-domain deliver, call it E3) as the ship gate.** Rejected as the bar. That number was set for v1, and v2's GatekeeperAgent makes qualitative routing decisions that may legitimately deliver fewer responses than a 0.75-threshold formula. Treating E3 as the gate risks "architecture works correctly, misses an aspirational number, declared a failure." E3 stays in the PRD as stretch, not as gate.
- **A single threshold (for example deliver rate at or above 50%, pass or fail).** Rejected. A binary outcome hides the difference between "barely over the line, architecture is fragile" and "well over the line, architecture is robust." Three tiers force explicit reasoning about the middle band instead of pretending it does not exist.
- **No pre-locked criteria, decide at Day 11 when the numbers are in.** Rejected. Deciding after the data lands is rationalization. Lock the criteria first, measure against them second.

### 4. Quantified Validation (evidence to cite)

- E1 floor anchored to the Day 8 measured v2 baseline of 39% in-domain deliver rate (full 40-record matrix: 11 of 28 in-domain records delivered). Source: the routing-correctness 2x2 and the §2d scorecard row in `docs/day8-findings.md`.
- E2 target of 55% chosen as substantive improvement over the baseline, roughly four additional in-domain queries delivering per leader out of 14, without claiming the rework should reach aspirational numbers. State that 55% is a chosen step, not a derived value.
- OOD fallback at 100% is a measured property of the Day 8 run (0 of 12 OOD records delivered, zero hallucinations), not a stretch target. Cite the bottom-left cell of the day8 2x2.

### 5. Consequences

Day 11 produces an unambiguous pass, fail, or judgment outcome rather than a vibe. This ADR becomes the citation point for the Day 11 evaluation report (`docs/day11-evaluation.md`). The three-tier structure with a defined judgment band is the interview signal: criteria locked before measurement, including an explicit plan for between-criteria results. Java/TS parallel (setting SLO error budgets before a deploy rather than after) is available but thin; the context brief says skip it unless the ADR needs the analogy beat, so default to omitting it.

### 6. Cross-references

Depends on ADR-010 (qualitative routing is why the v1 target does not transfer) and ADR-016 (the Layer-3 system evaluation is the measurement that these criteria judge). Depended on by PRD §2.1 (Day-11 acceptance criteria reproduce E1/E2 verbatim) and by the future `docs/day11-evaluation.md`. ADR-013's trigger (KH deliver rate more than 20 points below Torvalds) is measured in the same Day 11 run, so 013 and 015 share the evaluation event.

---

## ADR-016: Evaluation Methodology, Three-Layer Approach

### 1. Context (outline)

P6 v2 has three distinct evaluation surfaces: per-component math correctness (ScoringEngine), per-agent behavior under controlled inputs (CloneAgent, GatekeeperAgent, and the others), and end-to-end system behavior on the v2 query set. Without explicit layering these bleed into each other: unit tests start asserting system behavior, integration tests need a live LLM call to pass, system tests fail for reasons a unit test should have caught. The Day 8 finding was specifically a Layer-3 gap (end-to-end behavior diverged from the per-step claims), which is the motivating evidence that the layers catch different failure modes and must not be collapsed.

### 2. Decision

Three layers, each with a defined guarantee. Layer 1 (unit, continuous): every Component and Agent has unit tests in `tests/unit/`, deterministic, LLM calls mocked, coverage target at or above 90% on `src/`, run in CI on every commit. Layer 2 (integration, per Agent or Component): each unit tested in isolation with real LLM calls (Agents) or real dependencies such as FAISS (Components), with LLM responses recorded for replay in CI to keep determinism, located in `tests/integration/`, asserting contract behavior rather than exact output content. Layer 3 (system, end-to-end): the v2 query set runs through the full Flow via `cli evaluate`, producing JSON with all scores, routing decisions, and latencies, captured in the Day 11 report with the 2x2 grid, the Day-8 baseline comparison, and the PRD §2 scorecard. A methodology document (`docs/evaluation-methodology.md`) explains the three layers and how regression detection works across them.

### 3. Alternatives Considered

- **Two layers (unit plus system, skip integration).** Rejected. CloneAgent and GatekeeperAgent need contract tests with real LLM calls that are neither unit tests nor full-system runs. Without an integration layer those tests either bloat the unit layer and slow CI, or get deferred to system runs and lose isolation.
- **Four layers (add a smoke layer between integration and system).** Rejected as overkill. The v2 query set is small (20 queries per leader) so system runs are not slow, and a smoke layer adds maintenance without coverage the other three layers do not already provide.
- **LLM-as-judge as a separate evaluation layer.** Rejected as a layer, included as a methodology element inside Layer 3. EvaluatorAgent's explanation already supplies qualitative LLM judgment per the rewritten ADR-007. Layering it separately multiplies evaluation surfaces without adding rigor.

### 4. Quantified Validation (evidence to cite)

- The Day 8 finding is the load-bearing evidence: end-to-end verification (Layer 3) surfaced the agent-count gap, the silent Cohere failure, and the score-distribution problem that no unit or integration test had caught. Cite `docs/day8-findings.md` and the Day 8 followup note that promoted side-effect verification into the Verification Protocol.
- The three-layer structure was implicit in P5's testing approach and is made explicit here; cite that lineage rather than inventing novelty.
- LLM-response recording for replay is a known pattern; point at the v2 implementation target `tests/integration/conftest.py` (to be created in the rework) so the ADR has a concrete artifact reference.
- This ADR is primarily methodology; coverage target (90% on `src/`) and query-set size (20 per leader) are the quantitative anchors.

### 5. Consequences

Test files organize by layer (`tests/unit/`, `tests/integration/`, `tests/e2e/`). CI runs Layer 1 on every commit; Layers 2 and 3 run on PR and on release. The methodology document becomes a portable artifact reusable in P7 and beyond. The Day 11 evaluation report is constrained to Layer 3 results, so Layer 1 and Layer 2 outcomes do not appear in it. Java/TS parallel lands cleanly and the context brief endorses it: this roughly matches the Mike Cohn test pyramid (unit over integration over end-to-end), with the Layer-3 system evaluation doubling as the regression suite. Include it as a one-line parenthetical at the end of Consequences.

### 6. Cross-references

Depended on by ADR-015 (the Day 11 acceptance criteria judge the Layer-3 output this ADR defines). References ADR-007 rewritten (LLM-as-judge sits inside Layer 3 via EvaluatorAgent's explanation) and ADR-011 (EvaluatorAgent contract is a Layer-2 target). PRD §2.8, §2.9, and §7.5.2 reference the three-layer methodology and the methodology document.

---

# Group 2 — Rewrites

Both rewrites replace the v1 ADR body entirely while keeping the filename. The v1 content is not preserved as an appendix; the PRD §10 obsolescence notes already record what changed and why, so the ADR does not need to carry its own superseded text.

## ADR-001 (rewrite): CrewAI Flow with Real Agents at Each Step

Replaces the v1 ADR-001 ("CrewAI Flow over Sequential and Hierarchical Patterns"). Title updates to "CrewAI Flow with Real Agents at Each Step." The v1 decision (Flow over Sequential and Hierarchical) survives; the reasoning and the implementation pattern change substantially.

### 1. Context (outline)

CrewAI offers three orchestration approaches: Sequential Crews, Hierarchical Crews, and Flows. v1's ADR-001 chose Flow but wired every Flow step as a Python function calling deterministic pipeline code, and labeled four of those functions "agents" in documentation. The v2 question is not "which orchestration pattern" (the answer is still Flow) but "what runs at each Flow step." The v2 PRD's acceptance criteria (§2) and its multi-agent architectural intent (§3) define what the system needs to be. Measuring the v1 implementation against those requirements showed it did not meet them: the multi-agent claim was directory-deep rather than execution-deep, since only one of five "agents" used the CrewAI Agent abstraction while the other four were Python functions with the vocabulary applied as labels rather than as structure. The architecture had to change because the requirements were not being met, not because an audit produced a finding. Day 8 is the date the gap was measured, not the reason for the rework. The v1 ADR's own Decision section is the evidence, where it states only ChatStyleAgent used the Agent abstraction and "the other four agents are just functions."

### 2. Decision

CrewAI Flow is the deterministic orchestrator backbone. Each Flow step calls either a real CrewAI Agent (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent) or a real Component (Retriever). The Flow is the orchestration; there is no separate PlannerAgent. State is managed through `Flow[CloneState]` (a Pydantic BaseModel populated incrementally). Conditional branching happens via the `@router` decorator on the GatekeeperAgent step, returning the string `"deliver"` or `"fallback"`. This is the central architectural correction in v2: the Flow shell is unchanged from v1, but what runs inside each step changes from Python functions to real Agents and Components.

### 3. Alternatives Considered

- **Sequential Crew.** Rejected. No native conditional branching, so skipping FallbackAgent when GatekeeperAgent decides deliver would require no-op tasks or post-hoc filtering. The pipeline needs branching and Sequential cannot express it.
- **Hierarchical Crew.** Rejected. A Manager Agent making LLM-based delegation decisions on every fixed-order step adds 1 to 2 seconds of LLM latency per step with no autonomy benefit, because the pipeline order is deterministic. It is also documented as fragile in production (Towards Data Science, November 2025, which recorded hierarchical Crews looping instead of delegating).
- **Flow with Python-function steps (the v1 pattern).** Rejected for v2. It loses the Agent abstraction benefits (role/goal/backstory for prompt engineering, retry, Instructor structured output) and is the exact pattern that made v1's multi-agent claim directory-deep rather than execution-deep.

### 4. Quantified Validation (evidence to cite)

- The requirements the v1 implementation did not meet: PRD §2 (acceptance criteria, including the routing-correctness headline and the multi-agent orchestration goal) and PRD §3 (the 4-Agent-plus-3-Component architectural intent). These are the bar the v1 system was measured against. The Day 8 measurement is the empirical anchor that exposed the gap, framed as the measurement that exposed it, not as the event that caused the rework.
- v1 measured behavior: of five units called "agent," one (ChatStyleAgent) was a real CrewAI Agent and four were Python functions wrapped in `@listen` decorators. Cite PRD §12.2 and the v1 ADR-001 Decision text being replaced.
- The CrewAI engineering guidance (Dec 2025 blog) recommending a deterministic Flow backbone with individual steps leveraging different levels of agents, and the DocuSign migration from Sequential to Flows for conditional logic with typed state. These carry forward from the v1 ADR's own validation, reframed for the v2 "real Agents at each step" pattern.
- v2 structure: 4 real CrewAI Agents at 4 Flow steps where LLM reasoning is the work, plus 1 Component step (Retriever). Cite PRD §5.5 (the Flow code skeleton) as the concrete shape.
- Note for the author: the v1 ADR cited a `scratch/flow_poc.py` decorator POC and a `score >= 0.75` router example. Do not carry the 0.75 router example into the rewrite; routing is now GatekeeperAgent's string return, not a threshold comparison (ADR-010). The POC reference can stay only if reframed as validating decorator mechanics, not threshold routing.

### 5. Consequences

This is the decision every other v2 ADR builds on. The Flow shell stays; what runs inside moves from "Python functions with multi-agent vocabulary" to "real Agents where LLM reasoning lives, real Components where determinism lives." The visible artifact is the `src/agents/` versus `src/components/` split (ADR-014). The CrewAI dependency surface stays small and isolated to `src/flow.py` and the Agent classes, so a Flows API change is contained. Java/TS parallel lands and the context brief endorses it: this is closer to Spring Integration flow definitions calling typed beans than to a saga orchestrator, where the Flow is structure and the Agents and Components are behavior. Include as a one-line parenthetical at the end of Consequences.

### 6. Cross-references

Foundational for ADR-009 and ADR-014 (the Agent/Component split is the visible result of this decision), ADR-010 (the `@router` step is GatekeeperAgent), and ADR-005 (the dual-leader shared-retrieval pattern runs two Flow instances). The v1 ADR-001's dual-leader paragraph and `retrieved_chunks` early-exit detail should be kept in spirit but cross-referenced to ADR-005 rather than re-explained in full.

## ADR-007 (rewrite): LLM Roles in the Pipeline

Replaces the v1 ADR-007 ("LLM-Based Evaluation Scoring Viability"). Title updates to "LLM Roles in the Pipeline." PRD §10.5 marks the v1 Pearson 0.82 LLM-scoring experiment obsolete and points here. The v1 question (can GPT-4o-mini's LLM-as-judge scores be trusted in production) is gone because v2 stops using LLMs for numerical scoring entirely.

**Filename change.** This rewrite renames the file from `ADR-007-llm-evaluation-scoring-viability.md` to `ADR-007-llm-roles-in-the-pipeline.md` (via `git mv`, no stub at the old name). The new title and filename align, so any future link points at a name that matches the content.

### 1. Context (outline)

v1 ADR-007 asked whether GPT-4o-mini's LLM-as-judge groundedness scores correlated well enough with the deterministic cosine baseline to be used in production. The answer was yes (Pearson 0.82), and v1 used the LLM for both scoring and explanation. v2 removes LLMs from numerical scoring: ScoringEngine handles all three scores deterministically, and LLM use shifts to response generation, explanation, routing, and contextual fallback. The v1 question is obsolete. The v2 question is broader: across the whole pipeline, where do LLMs earn their cost and where are they the wrong tool.

### 2. Decision

LLMs are used at four pipeline locations, each justified by a need for reasoning over inputs: CloneAgent (response generation in leader voice), EvaluatorAgent (explanation and flags over deterministic scores), GatekeeperAgent (routing judgment over scores, chunks, and response), FallbackAgent (leader-appropriate fallback generation). LLMs are deliberately not used for numerical scoring (ScoringEngine is deterministic math), retrieval (Retriever is FAISS plus Cohere with no LLM in the chain), orchestration (the Flow decorators are deterministic), or routing the Flow itself (only the GatekeeperAgent decision inside the Flow is LLM-driven). The principle: LLM where reasoning is the work, deterministic where reasoning is the obstacle.

### 3. Alternatives Considered

- **LLM everywhere, including numerical scoring (the v1 pattern).** Rejected for v2. Numerical scoring is math; an LLM adds latency, non-determinism, and cost with no quality benefit over the cosine baseline. The v1 Pearson 0.82 result showed LLM scoring agreed with cosine, which is an argument for keeping cosine, not for keeping the LLM in that path.
- **LLM nowhere (fully deterministic).** Rejected. Leader-voice response generation cannot be done well deterministically, and routing judgment that absorbs the style-score asymmetry (ADR-013) needs qualitative reasoning. Removing all LLMs collapses the system to a templated chatbot.
- **LLM only for response generation (CloneAgent), deterministic everywhere else.** Rejected. GatekeeperAgent's reasoning is the mechanism that absorbs measurement asymmetry, FallbackAgent's leader-voice generation is the user-facing differentiator on the dominant path, and EvaluatorAgent's explanation makes the system inspectable. All three earn their cost.

### 4. Quantified Validation (evidence to cite)

- v1 ADR-007 plus PRD §10.5 obsolete the Pearson 0.82 LLM-scoring experiment. Cite the correlation honestly as the reason the LLM-scoring path was retired (LLM agreed with cosine, so cosine wins on determinism and cost), reframing rather than discarding the v1 number.
- ADR-009, ADR-010, ADR-011, and ADR-012 each justify one specific LLM placement; this ADR consolidates the cross-cutting principle and points at them rather than re-deriving each.
- ScoringEngine's determinism is grounded in ADR-004 (groundedness cosine) and ADR-003 (style features). Cite both as the basis for the "no LLM in scoring" half of the decision.
- This ADR is principle-level; the one hard number is the retired Pearson 0.82, used to show the LLM-scoring path was measured and then deliberately removed rather than never tried.

### 5. Consequences

Every Agent in the v2 inventory has an explicit LLM-use justification traceable to this ADR, and new Agents in future work must pass the "reasoning is the work" test. Components stay LLM-free, enforced architecturally by the absence of LiteLLM imports in `src/components/` (the architecture honesty check greps for this). Java/TS parallel is available (choosing between a deterministic strategy pattern and a rules engine with side effects, where the wrong placement yields either rigid systems that cannot adapt or unpredictable ones that cannot be tested) and the context brief endorses it; include as a one-line parenthetical at the end of Consequences if it reads cleanly.

### 6. Cross-references

Consolidates ADR-009, 010, 011, 012 (each a specific LLM placement) and depends on ADR-003 and ADR-004 (the deterministic scoring it relies on). Referenced by ADR-016 (LLM-as-judge sits inside Layer 3 as EvaluatorAgent's explanation, not a separate layer) and by PRD §10.5. The v1 ADR-007's Ollama-versus-GPT material is dropped from the rewrite; Ollama is deferred post-portfolio per PRD §3.5 D4, so a one-line note that local-model substitution is deferred is enough, with no need to carry the full Run 1 and Run 2 latency tables.

---

# Group 3 — Light edits

Each light edit preserves the existing five-section structure and the full v1 header. The edits are surgical. Read the on-disk file again at edit time; the current-state summaries below are the orientation, not a substitute for the file.

## ADR-002: RAG Configuration (Embeddings, Reranking, Chunking)

### Current state
Locks OpenAI text-embedding-3-small primary with MiniLM baseline, RecursiveCharacterTextSplitter 500/50, FAISS IndexFlatIP with L2 normalization, and Cohere rerank-english-v3.0 top-20 to top-5. Quantified Validation cites P2 grid numbers including a Cohere precision lift (0.52 to 0.74).

### Edit scope per PRD §7.5.2
Light edit: Cohere env var correction note (CO_API_KEY to COHERE_API_KEY) and a correction that the cited Cohere lift was inherited from P5/P2 and did not transfer to P6's corpus as a smooth percentage.

### Specific edits required
- Add a correction note (a short labeled paragraph inside Consequences, or a dated amendment block at the end of Consequences) stating that from Day 3 through Day 8 the Cohere reranker silently fell back to vector-only because the code read `CO_API_KEY` while `.env` set `COHERE_API_KEY`, and that the one-line fix (commit 206c232) means Cohere actually executes in v2. Cite `docs/day8-findings.md` Finding 2.
- Add that v2 measured in-domain top-1 Cohere relevance mean around 0.89 on the v2 in-domain query set (day8-findings.md "Step B verification" / Finding 2 context), correcting the implication that the P2 precision-lift table reflects P6 behavior.
- Cross-reference ADR-006, which already amended the magnitude claim (Cohere is directionally right but the 20% figure does not generalize to programming-textbook corpora).
- Keep graceful degradation described: on Cohere failure, Retriever returns FAISS top-5 with a warning log.

### What does NOT change
The embedding choice (OpenAI primary, MiniLM baseline), chunking (500/50, RecursiveCharacterTextSplitter), FAISS IndexFlatIP with L2 normalization, the two-stage top-20 to top-5 retrieval shape, and the P2 grid evidence for the embedding decision all stand. Do not rewrite the Decision section; the configuration is unchanged.

## ADR-003: Hand-Crafted Feature Vectors over LLM Embeddings

### Current state
Decides hand-crafted 15-dim feature vectors over LLM embeddings, with interpretability, radar-chart compatibility, style-versus-topic discrimination, and Schneider et al. precedent as the rationale. References `feature_extractor.py`, `StyleFeatures.to_vector()`, and full-corpus self-similarity numbers.

### Edit scope per PRD §7.5.2
Light edit: naming aligned with v2 vocabulary.

### Specific edits required
- Update any "ChatStyleAgent" reference to "CloneAgent" and align profile-building references to the StyleProfileBuilder Component (the on-disk text references `feature_extractor.py` and `scripts/build_profiles.py`; reframe profile construction as StyleProfileBuilder's job per PRD §5.2.2 without deleting the underlying-module references, which still exist as low-level helpers per PRD §12.2).
- Add a one-line pointer to ADR-013 noting the style profile is frozen during the rework and re-measured Day 11, so the asymmetry finding has a home.
- Confirm the cosine-similarity-on-feature-vectors framing matches ScoringEngine's role (style score is computed by ScoringEngine per PRD §5.2.3).

### What does NOT change
The core decision (hand-crafted 15-dim vectors over LLM embeddings) and its full rationale (interpretability per dimension, radar-chart axes, style-versus-topic discrimination, Schneider et al. validation, reproducibility versus model drift, the coverage tradeoff) all survive. The self-similarity and per-feature delta tables stay. Do not touch the CodeBERT Java/TS parenthetical already at the end of Consequences.

## ADR-004: Groundedness Scoring via Cosine Similarity

### Current state
Decides sentence-level max cosine similarity against top-5 chunks, batched embeddings, calibrated against a 5-sample LLM judge. Context states groundedness is 40% of the final formula `0.4*style + 0.4*groundedness + 0.2*confidence`. Code reference is `src/evaluation/groundedness_scorer.py`.

### Edit scope per PRD §7.5.2
Light edit: scope clarified (the scorer is now a method on ScoringEngine Component; LLM judgment for groundedness happens at EvaluatorAgent's explanation step, not inside the scorer).

### Specific edits required
- Clarify that groundedness scoring is a ScoringEngine Component method, deterministic and LLM-free, and that any LLM interpretation of groundedness happens in EvaluatorAgent's explanation generation (ADR-011), not in the scorer.
- Soften or correct the Context sentence that frames groundedness as "40% of the final quality score." The weighted formula and final_score are gone in v2 (ADR-010, Architecture Rule 3). Reframe as: groundedness is one of three individual scores GatekeeperAgent reasons over. Do not leave a live reference implying the 0.4 weight still drives routing.
- Update the code-location reference toward ScoringEngine (`src/components/scoring_engine.py`) while noting the math is wrapped from the existing `src/evaluation/` helper per PRD §12.2.
- Cross-reference ADR-007 rewritten (groundedness is deterministic; LLM does not score) and ADR-011.

### What does NOT change
The core decision (sentence-level max cosine, batched embeddings, reuse of chunk embeddings from the RAG pipeline, the regex sentence split) survives. The LLM-judge calibration table and the known failure mode (semantic similarity cannot detect contradiction) stay. The BLEU/ROUGE, BERTScore, and per-sentence-LLM-judge alternatives stay.

## ADR-005: Shared RAG Retrieval for Dual-Leader Mode

### Current state
Decides retrieve-once, style-twice via a `compare_leaders()` wrapper and a `retrieved_chunks` early-exit in the Flow's retrieve step. Contains two Mermaid sequence diagrams (A2 single-query, A3 dual-leader) that name RAGAgent, StyleCrew, and EvaluatorAgent only.

### Edit scope per PRD §7.5.2
Light edit: agent names updated (CloneAgent, not ChatStyleAgent/StyleCrew), and the per-leader branch extended to show the v2 agents.

### Specific edits required
- In both Mermaid diagrams and inline text, rename `StyleCrew`/`ChatStyleAgent` to CloneAgent and `RAGAgent` to the Retriever Component. EvaluatorAgent keeps its name.
- Extend the per-leader branch in the A3 diagram (and A2) to show the v2 pipeline: Retriever once, then per leader CloneAgent, EvaluatorAgent, GatekeeperAgent, and FallbackAgent on the fallback branch. The current diagrams stop at EvaluatorAgent returning a decision; v2 splits evaluation (EvaluatorAgent) from routing (GatekeeperAgent) and adds the explicit FallbackAgent step.
- Update the `EvaluationResult {decision}` diagram label, since in v2 the decision lives on RoutingDecision from GatekeeperAgent, not on EvaluationResult (which has no decision field). Show the `@router` branching on GatekeeperAgent's output.
- Align the early-exit code snippet field name with the v2 schema (`CloneState.chunks` per PRD §5.4, versus the v1 `retrieved_chunks`); flag this rename for the author to confirm against the actual v2 schema before editing, since it touches a code-accurate snippet.
- Cross-reference ADR-001 rewritten (the Flow runs two instances) and note the timing-harness numbers were measured under the v1 step shape.

### What does NOT change
The core decision (retrieve once, reuse chunks across leaders, request-scoped sharing through the wrapper rather than a cache) survives unchanged, along with the rejected alternatives (independent pipelines, cached RAG with TTL) and the timing measurements. The asymmetric-outcome handling (one leader delivers, the other falls back) stays and now aligns with PRD §4.7.

## ADR-006: Corpus-Shape Limits on Retrieval

**Title change is part of this light edit.** The v1 title "Day 6 Methodology and Corpus-Shape Limits" is misleading after the scope reduction, since the Day 6 experiments are deprecated and the surviving value is the corpus-shape findings that Day 8 re-confirmed. The H1 title in the ADR prose changes to "Corpus-Shape Limits on Retrieval." The filename does NOT change: the file stays `docs/adr/ADR-006-day6-methodology-and-corpus-shape-limits.md`. Do not `git mv` this file; only the title-in-prose and the PRD title rows change.

### Current state
Documents three Day 6 null results as measurement artifacts (Cohere bimodal behavior, weight-sensitivity sweep under the proxy regime, Torvalds 2018 style-evolution null at per-email resolution). Decision keeps production on weights 0.4/0.4/0.2, OpenAI embeddings, and Cohere. References a chart at `results/charts/07-style-evolution.png`.

### Edit scope per PRD §7.5.2
Light edit with a scope reduction. Day 6 experiments are deprecated per PRD §10, so ADR-006's scope narrows from "active methodology plus corpus-shape findings" to "historical record of Day 6 methodology, plus the durable corpus-shape findings that Day 8 confirmed." The methodology becomes record; the corpus-shape insight is the part that carries forward.

### Specific edits required
- Rewrite current-tense prose about the Day 6 experiments into past tense as historical record. Sentences that read as live methodology ("The production scorer combines...", "I wanted to see whether...") become past-tense accounts of what was run on Day 6. The goal is that a reader sees this ADR as a record of a completed measurement pass, not as a description of the current system.
- Add a brief dated amendment (at the end of Consequences) noting the Day 8 q12 binary-search verification re-confirmed the corpus-shape constraint: same query, same index, same Cohere, flipped from deliver to fallback because the corpus has one substantive binary-search chunk and two-clause questions need two well-aligned chunks. Cite day8-findings.md "Verification 2." This is the durable insight that carries forward, so it stays present-tense as a confirmed finding.
- Note that the weight-sweep null is superseded for production purposes by ADR-010 (no weighted formula in v2), so the "production stays on 0.4/0.4/0.2" decision is historical. Add a one-line pointer to PRD §10.3 (marked obsolete) and ADR-010 rather than rewriting the section.
- Cross-reference ADR-002 (the Cohere-magnitude amendment ADR-006 already made about the 20% figure not generalizing is the same correction ADR-002's light edit now formalizes).

### Chart reference (resolved by PRD §10.4)
PRD §10.4 settles this: the pre/post-2018 style-evolution chart carries forward as a deliverable ("chart #8 (§2.10, §7.6) is still produced"). It survives Day 6 experiment deprecation and is regenerated at Day 12 from raw data, not removed. So the edit is non-conditional: update the on-disk reference from `results/charts/07-style-evolution.png` (v1 numbering) to PRD §7.6's v2 name `results/charts/08-torvalds-style-evolution-pre-post-2018.png`, citing PRD §10.4 as the authority for the chart's survival. Disk catches up at Day 12 to 13 regeneration. The context brief's `05-style-evolution.png` is wrong regardless; slot 05 on disk is the fallback-rate chart.

### What does NOT change
- The corpus-shape findings survive as durable insight, not as a deprecated experiment. This is the part of ADR-006 that carries forward into v2.
- The structural reason for the null results survives: the nulls trace to corpus shape, input proxy, and feature resolution, not to a retrieval bug. That diagnosis is the load-bearing point and stays.
- The cross-reference to ADR-002 (Cohere magnitude does not generalize to programming-textbook corpora) stays and is reinforced.
- The three documented measurement-artifact diagnoses stay as the historical Day 6 record (now past-tense). The Spring mocked-dependency parenthetical stays.

## ADR-008: Hexagonal Adapters for CLI and Streamlit

### Current state
Decides that `src/cli.py` and `streamlit_app.py` import only from `src/flow.py`, `src/schemas.py`, `src/config.py`, plus narrow façades for learn/index, and never import litellm/faiss/cohere/openai directly. References RAGAgent, StyleCrew, EvaluatorAgent, FallbackAgent and the v1 façade paths (`src/style/profile_builder.py`, `src/rag/corpus_loader.py`, `src/agents/rag_agent.py`).

### Edit scope per PRD §7.5.2
Light edit, confirmed (not a rewrite). The hexagonal adapter pattern is unchanged. PRD §7.5.2 and §12.3 both say "Kept as-is," which applies to the decision, not to the literal prose. The edit surface is v1 class names appearing in current-tense text across three of the five sections, confirmed by grep on the on-disk file. Renaming them in place is a terminology pass, not a structural change, and it prevents the Day 14 codebase audit (PRD §12.5 category 3) from later flagging the same names as vocabulary leaks.

### Specific edits required
Five surgical renames, located by line number against the current on-disk file:
- Line 22 (Decision): `src/agents/rag_agent.py` becomes `src/components/retriever.py`.
- Line 22 (Decision): `src/style/profile_builder.py` maps to StyleProfileBuilder per PRD §5.2.2 (the underlying helper module path may persist per PRD §12.2; reframe the façade name, do not invent a new path without confirming it).
- Line 35 (Alternatives Considered): `RAGAgent` becomes Retriever; `StyleCrew` becomes CloneAgent.
- Line 39 (Alternatives Considered): `RAGAgent` in the `st.session_state` caching example becomes Retriever.
- Line 49 (Consequences): `RAGAgent.build` becomes `Retriever.build()`.

Confirmations (no edit, just verify during execution): `EvaluatorAgent`, `FallbackAgent`, `DigitalCloneFlow`, and `compare_leaders` references are already v2-correct and stay unchanged; the `@st.cache_resource` deferral note in Consequences stays; the grep check list (`litellm`, `faiss`, `cohere`, `openai`) matches CLAUDE.md Architecture Rule 2.

### What does NOT change
The decision (adapters import only through the flow façade and never the ML libraries), the ports-and-adapters framing, the test-boundary rationale (CliRunner tests mock at `src.cli.DigitalCloneFlow`), the rejected alternatives (shared adapter base class, Streamlit session-state caching), and the learn/index carve-out all survive. No structural change to the five sections.

---

# Summary

## ADR count after Batch B

The contiguous range is **001 through 016, which is 16 ADRs, not 14.** The planning prompt's expectation of 14 is a mismatch with the PRD. Both PRD §7.5.2 and §12.3 list 16 ADRs explicitly. Accounting: 001 and 007 rewritten (2), 002 to 006 and 008 light-edited (6), 009 to 012 written in Batch A (4), 013 to 016 written in Batch B (4). That is 16 files, contiguous, no gaps. After Batch B completes, all 16 ADR files exist on disk. The "14" figure in the prompt should be treated as an error to confirm with Ruby, not a target.

## Resolved decisions (carried into the per-item sections above)

These were open ambiguities in the first draft of this plan; Ruby has now resolved them, and the per-item sections reflect the resolutions.

1. **ADR-008 stays a light edit, not a rewrite.** The hexagonal pattern is unchanged. "Kept as-is" in PRD §7.5.2 and §12.3 applies to the decision, not the literal prose. The five v1-vocabulary renames (located by line number in the ADR-008 section above) are applied in place.

2. **ADR-007 filename renamed.** The rewrite renames `ADR-007-llm-evaluation-scoring-viability.md` to `ADR-007-llm-roles-in-the-pipeline.md` via `git mv`, with no stub at the old name.

3. **ADR-013 title is "Style Profile Frozen, Re-Measured Day 11"** (the PRD §7.5.2 form). PRD §12.3 line 1162 carries the longer "Style Profile Asymmetry — Frozen, Re-Measured Day 11" form, which also contains a banned em-dash; it is corrected to match §7.5.2 as part of the PRD-edit sub-item below.

4. **ADR-016 title is "Evaluation Methodology, Three-Layer Approach"** (comma, not em-dash). PRD §7.5.2 line 888 and §12.3 line 1165 both use the em-dash form and are corrected in the PRD-edit sub-item below.

5. **ADR-006 scope reduces to historical record plus durable corpus-shape findings.** Day 6 experiments are deprecated per PRD §10, so the light edit rewrites Day 6 experiment prose into past tense, keeps the Day 8 corpus-shape confirmation as the carried-forward insight, and points the superseded weight-sweep at ADR-010 and PRD §10.3. The chart reference is resolved by PRD §10.4 (chart #8 carries forward as a deliverable): update `07-style-evolution.png` to `08-torvalds-style-evolution-pre-post-2018.png`, no conditional.

### PRD-edit sub-item (one commit, five title fixes)

Batch B execution also makes one small PRD edit alongside the ADR work, fixing five stale or em-dashed titles so the PRD agrees with the ADR files:
- PRD §7.5.2 line 888: ADR-016 title em-dash becomes a comma.
- PRD §12.3 line 1162: ADR-013 title becomes "Style Profile Frozen, Re-Measured Day 11" (drop "Asymmetry" and the em-dash, matching §7.5.2).
- PRD §12.3 line 1165: ADR-016 title em-dash becomes a comma.
- PRD §7.5.2 line 878: ADR-006 title becomes "Corpus-Shape Limits on Retrieval" (was "Day 6 Methodology and Corpus-Shape Limits").
- PRD §12.3 line 1155: ADR-006 title becomes "Corpus-Shape Limits on Retrieval."

## Remaining flags (no blocker, surface for awareness)

1. **ADR-005 schema field rename in a code snippet.** The v1 early-exit snippet uses `CloneState.retrieved_chunks`; PRD §5.4 defines the v2 field as `chunks`. The v2 schema is not implemented yet (it lands Day 10 to 11). Recommendation, carried in the ADR-005 section: align the snippet to PRD §5.4 (`chunks`) and note it reflects the planned schema.

2. **Header style split between batches.** Batch A ADRs (009 to 012) use a minimal header; the v1 ADRs (001 to 008) use a fuller Project/Category/Status/Date header. This plan keeps both rewrites on the fuller header and the four new ADRs on the minimal header, leaving 16 files with two header styles. A single consistent header across all 16 would be a separate normalization pass.

3. **Chart inventory drift (Day 12 verification item, not a Batch B blocker).** Disk currently holds 7 charts under v1 numbering; PRD §7.6 specifies 8 charts under v2 numbering. The missing chart is `02-routing-correctness-grid.png`, which is a Day 11 deliverable rather than a Day 9 one. This is broader than Batch B and is flagged for Day 12 chart-regeneration verification, not for resolution during the ADR work.

4. **Day 6 experiment chart caption correction (separate day8 follow-up, not Batch B work).** The day8-findings follow-up flagged that the Day 6 reranking charts in `docs/experiments/charts/` ran without Cohere and may need a "measured pre-Cohere-fix" caption; the two affected files are `6a-embeddings.png` and `6a-embeddings-run2.png`. This is tracked separately and is not folded into ADR-002's light edit, which stays scoped to the Cohere env var correction note inside the ADR file. The third candidate (`6e-run2-groundedness-agreement.png`, the Pearson-0.82 experiment) needs no separate caption work; its obsolescence is already absorbed by the ADR-007 rewrite's Quantified Validation section.

## Cross-references requiring Batch A ADRs on disk

Confirmed by directory listing: `docs/adr/` contains ADR-009 through ADR-012 (all four Batch A files, dated 27 May). Every Batch B cross-reference into Batch A is therefore satisfiable now:

- ADR-014 depends on ADR-009 (the criterion). Present.
- ADR-013 references ADR-010. Present.
- ADR-007 rewrite consolidates ADR-009, 010, 011, 012. All present.
- ADR-001 rewrite is referenced by ADR-009 and ADR-014 and points forward to ADR-010 and ADR-011. All present.
- ADR-015 depends on ADR-010 and ADR-016; ADR-016 references ADR-007 (rewritten) and ADR-011. ADR-011 present; ADR-007 and ADR-016 are written in this batch.

No Batch B item depends on a file that does not yet exist.

**Execution order within Batch B.** Drafting order privileges ADRs whose substance is fully available at Day 9 writing time. The recommended order:
1. Group 3 light edits first (002, 003, 004, 005, 006, 008). The v1 code and the v1 ADR text are both on disk, so the substance check applies fully and the edits are surgical.
2. ADR-001 rewrite next. The Context and Quantified Validation lean on v1 code (which exists) and on PRD §2 and §3 requirements (which exist). The forward-looking v2 material is structural (the Flow shell stays; what runs inside changes) and is well-grounded in PRD §5.5.
3. ADR-007 rewrite. Substance is the v1 LLM-scoring path (on disk) plus PRD §10.5 obsolescence (on disk) plus the cross-cutting LLM-placement principle (specified in PRD §5.1 and §5.2).
4. The four new ADRs (013, 014, 015, 016). These lean most heavily on PRD specification and on Day 8 measured data, and least on v2 code. Within this group, ADR-007 must be done before ADR-016 because ADR-016 references ADR-007's rewritten content (cross-reference confirmed in the Batch A cross-references list above).

This order is a recommendation, not a strict requirement; deviations are fine as long as ADR-007 precedes ADR-016 and as long as forward-looking v2 material in any ADR is flagged appropriately per the substance-check sequencing note.

---

*Stop after review. ADR files in `docs/adr/` are written only after this plan is approved.*
