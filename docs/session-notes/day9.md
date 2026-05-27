# Day 9 Session Notes

## Phase 1: ADR-009 Agent vs Component Distinction

- Built: `docs/adr/009-agent-vs-component-distinction.md`. 5 H2 sections (Context, Decision, Alternatives Considered, Quantified Validation, Consequences). Decision is verbatim from `docs/plans/day9-plan.md`; Context is the plan outline rewritten as three prose paragraphs; Alternatives and Consequences are the plan's bullets with v1-references preserved.
- Why: this ADR formalizes the vocabulary lock that v1's failure mode required. Wrote Context as prose (per CLAUDE.md ADR-specific rule) rather than carrying the plan's bullet outline straight through. Java/TS parenthetical (Spring `@Service` vs `@Repository`) goes inline at the end of Consequences, not as its own section.
- Surprising: the input path in the execution prompt was `docs/plans/day9-adr-batch-a-plan.md`, but the file on disk is `docs/plans/day9-plan.md`. Same content. Flagged the naming mismatch and proceeded with the on-disk file as the contract.
- Deferred: ADR-010, ADR-011, ADR-012 — Phases 2-4 of this session, each behind its own stop gate.
- ADR candidate: no, this phase is the ADR.

## Phase 2: ADR-010 LLM-Driven Routing via GatekeeperAgent

- Built: `docs/adr/ADR-010-llm-driven-routing-gatekeeper.md`. 5 H2 sections. Decision verbatim from the plan; Context rewritten as three prose paragraphs from the outline; Alternatives keep the v1 weighted-formula and threshold references; Quantified Validation transcribes the seven-item evidence list; Drools rule-engine parenthetical inline at the end of Consequences.
- Why: documents the replacement of v1's `0.4×style + 0.4×ground + 0.2×conf` formula and 0.75 threshold with LLM-driven routing. Matched ADR-009's edited house style (no bold lead-ins in Alternatives, `Positive:`/`Negative:` retained in Consequences).
- Verified against day8-findings.md before transcribing: 95.0% / 90.0% / 72.5% fallback (Three-run comparison 19/20, 18/20, 29/40); style means T 0.9025 / KH 0.8355, Δ +0.067 (Finding 1); groundedness 0.6258 up from 0.5173 (Three-run); OOD 12/12 100% zero hallucinations (2×2 matrix); q12 v1 May-23 delivered at final=0.7525, both fall back post-Cohere-fix (Verification 2). All seven numbers match source.
- Surprising: nothing. No conflicts between plan and CLAUDE.md format rules.
- Deferred: ADR-011 (Phase 3), ADR-012 (Phase 4).
- ADR candidate: no, this phase is the ADR.

## Phase 3: ADR-011 EvaluatorAgent Hybrid Design

- Built: `docs/adr/ADR-011-evaluator-agent-hybrid-design.md`. 5 H2 sections. Decision verbatim from the plan; Context as three prose paragraphs; Alternatives keep the v1-design reference on the pure-Component option; Quantified Validation transcribes the three-item evidence list. No Java/TS parenthetical, and the plan's editorial scaffolding note explaining the omission was left out of the ADR (it is plan-file material, not ADR content). Consequences ends at the last bullet.
- Why: documents the hybrid Agent that delegates numerical scoring to ScoringEngine and uses the LLM only for explanation and flags. The hybrid pattern is the case ADR-009 flagged as needing its own treatment.
- Verified: Pearson 0.82 against PRD §10.5 ("GPT-4o-mini correlated Pearson 0.82 with cosine baseline on groundedness scoring") — matches. Finding 1 numbers (style 0.9025, groundedness 0.6258, 0.28 gap) consistent with day8-findings.md.
- Surprising: nothing. Confirmed the no-parenthetical and no-scaffolding-note instructions per the plan.
- Deferred: ADR-012 (Phase 4).
- ADR candidate: no, this phase is the ADR.
