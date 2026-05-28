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

## Phase 4: ADR-012 LLM-Driven FallbackAgent with Templated Failsafe

- Built: `docs/adr/ADR-012-fallback-agent-with-failsafe.md`. 5 H2 sections. Decision verbatim from the plan; Context as three prose paragraphs; Alternatives keep the v1-templated-fallback reference; Quantified Validation transcribes the four-item evidence list. resilience4j/Hystrix circuit-breaker parenthetical inline at the end of Consequences.
- Why: documents the LLM-driven fallback with a 5-line templated failsafe. The dominant-path framing (60-75% of in-domain experience) is what justifies an Agent here rather than a template.
- Verified against day8-findings.md: fallback 95% / 90% / 72.5% (Three-run comparison); in-domain 60.7% = 17/28 (scorecard §2d + 2×2 matrix); per-leader Torvalds 14/20 70.0%, KH 15/20 75.0% (Three-run); mean fallback latency 11,445 ms (Three-run). All four numbers match source.
- Surprising: nothing. No conflicts between plan and CLAUDE.md format rules.
- Deferred: none. ADR batch A (009-012) complete.
- ADR candidate: no, this phase is the ADR.

End of session: all four Day 9 ADRs (009-012) written. 009-011 committed (3244295, 325a489, c7c62f7); 012 awaiting review.

## Batch B Phase 1: six ADR light edits and PRD title fixes

- Built: light edits to the six surviving v1 ADRs (`ADR-002` through `ADR-006`, `ADR-008`) and five title fixes in `docs/PRD.md` §7.5.2 and §12.3, committed as d756fea. ADR-002 took a Cohere correction (the silent CO_API_KEY/COHERE_API_KEY failure and the 0.89 measured top-1 relevance after fix 206c232); ADR-003 took v2 vocabulary plus an ADR-013 freeze pointer; ADR-004 reframed groundedness off the dead weighted formula; ADR-005 rebuilt both sequence diagrams to the v2 pipeline and renamed `CloneState.retrieved_chunks` to `chunks`; ADR-006 was retitled "Corpus-Shape Limits on Retrieval" with the Day 6 experiment prose moved to past tense; ADR-008 took five vocabulary renames.
- Why: surgical alignment per the locked `docs/plans/day9-batch-b-plan.md`, not rewrites. The substance check held additions to material that strengthens an argument already in the ADR, such as the verified env var name and the real v2 schema field `chunks`, and anything that would have contradicted the on-disk file was surfaced rather than silently rewritten.
- Reviewed and corrected: a read-through after d756fea caught AI-cadence and v2 leaks the first pass left in, fixed across ADR-002/003/004. The ADR-002 amendment paragraph narrated the Cohere incident as operational history and was deleted down to the 0.89 architectural evidence; ADR-003's Context still called 0.70 a "quality gate before delivery" (a v1 routing claim) and was reframed as feature-design validation citing ADR-010; ADR-004 still had a future-tense Day 6 sweep sentence and "0.60 threshold" language, both reframed to past tense and to a calibration point. The bold-label-dash pattern (`**X** -`) in Alternatives was converted to plain sentence-starters per the CLAUDE.md ban. ADR-005 also dropped a stale "Phase 3 error-recovery" reference during review.
- Surprising: the first humanization pass reintroduced the exact constructions the Writing Rules ban, bold-dash labels and an incident-narrative amendment. The substance-check additions need the same banned-construction sweep the rest of the ADR gets.
- Deferred: Batch B Phase 2 (ADR-001 and ADR-007 rewrites) and Phase 3 (new ADR-013 through ADR-016), each behind its own stop gate.
- ADR candidate: no. These edits maintain existing ADRs and the PRD inventory; no new decision surfaced.

End of Batch B Phase 1: six light-edited ADRs and the PRD title fixes committed (d756fea); review corrections to ADR-002/003/004 and ADR-005's stale-reference cleanup in the follow-up commit. Phases 2 and 3 await review gates.
