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

## Batch B Phase 2: ADR-001 and ADR-007 rewrites

- Built: full in-place rewrites of the two v1 ADRs whose decisions no longer hold under v2. ADR-001 retitled "CrewAI Flow with Real Agents at Each Step" (filename unchanged, committed 5b89ba9); ADR-007 retitled "LLM Roles in the Pipeline" and renamed via `git mv` from `ADR-007-llm-evaluation-scoring-viability.md` to `ADR-007-llm-roles-in-the-pipeline.md`, no stub left (committed 889be13, pushed to draft PR #13). Both keep the fuller v1 header block with Category updated to Architecture, Status to "Accepted (rewritten for v2, 2026-05-26)", and the original v1 authorship Date preserved (001 stays 2026-04-03, 007 stays 2026-04-27).
- Why: v1 ADR-001 described Flow with Python-function steps and v1 ADR-007 was about LLM-scoring viability; both are obsolete under the v2 four-Agent/three-Component architecture, so PRD §7.5.2 marks them rewrites rather than light edits. ADR-001 keeps the Flow-over-Sequential/Hierarchical decision but moves the reasoning to real Agents at each step; ADR-007 reframes the retired Pearson 0.82 result as the reason the LLM-scoring path was dropped and consolidates the four LLM placements and four non-uses across the pipeline.
- Verified per the substance check before each commit, with surgical corrections applied: ADR-001 Context import quote corrected to `from crewai import Agent, Crew, LLM, Task` (LLM was missing) and the Consequences pin sentence repointed from "v1 pinned for that reason" to the actual API-churn reason carried forward from v1; ADR-007 Pearson confirmed at 0.82 (0.8172, p=0.0039) against PRD §10.5, the v1 reasoning-LLM paths confirmed at `src/evaluation/evaluator.py` and `src/fallback/unstyled_responder.py`, and the "fails the build" overstatement corrected to "flags a warning at the session stop gate" to match CLAUDE.md's grep check (warns, does not fail CI).
- Surprising: the first ADR-007 commit (1a8f42c, never pushed) captured the `git mv` rename but with the old v1 body, because `git add` aborted on the already-deleted old pathspec and staged nothing new. Caught by reading the committed content, fixed by staging the new path only and `git commit --amend` to 889be13.
- Deferred: Batch B Phase 3 (new ADR-013 through ADR-016), behind its own stop gate.
- ADR candidate: no. These are rewrites of existing decisions, not new ones.

End of Batch B Phase 2: ADR-001 rewrite committed (5b89ba9) and ADR-007 rewrite committed and pushed (889be13, draft PR #13). Phase 3 awaits its review gate.

## Batch B Phase 3: four new ADRs (013-016)

- Built: the four new Batch B ADRs, each on the Batch A minimal header (Status/Date, five H2 sections, no Cross-References or Java/TS headings). ADR-013 Style Profile Frozen, Re-Measured Day 11 (committed af6d55b, alongside the ADR-003 Schneider citation restore); ADR-014 Agent and Component Inventory (a0e0b2f); ADR-015 Post-Rework Evaluation Acceptance Criteria (06a10c8); ADR-016 Evaluation Methodology, Three-Layer Approach (3af0be0). Each written one sub-phase at a time behind its own stop gate, with a verification pass between writing and commit.
- Why: ADR-013 commits to freezing StyleProfileBuilder during the rework and re-measuring the Day 8 style asymmetry on Day 11, with per-leader weighting deferred behind a 20-point trigger. ADR-014 records the concrete 4-Agent/3-Component/1-Flow inventory that applies the ADR-009 criterion and rejects a PlannerAgent. ADR-015 locks the three-tier E1/E2 acceptance criteria before Day 11 measures anything. ADR-016 locks the three-layer evaluation methodology, motivated by the Day 8 Layer-3 gap.
- Framing discipline (Phase 3 throughout): kept measured, chosen, and future cleanly separated. Day 8 figures are measured (style means 0.9025/0.8355, in-domain deliver 39% = 11/28, OOD 0/12 zero hallucinations); E2 55% is a chosen target stated as such; the Day 11 evaluation and the v2 Agent/Component code are committed-future and never written as if they exist. The four Agents and three Components are framed as specified per PRD §5.1/§5.2, not as files on disk.
- Verified per the substance check before each commit; corrections applied where verification caught a problem. ADR-013 had attributed the Schneider et al. (2016) citation to ADR-003, which surfaced that ADR-003 no longer carried it (stripped in an earlier humanization pass); ADR-013 was repointed to day8-findings.md and PRD §3, and ADR-003's citation was restored in the same commit. ADR-015's "§2d scorecard row" citation was scoped explicitly to day8-findings (legacy label internal to the frozen scorecard, not a current PRD section). ADR-016 dropped the agent-count gap from Quantified Validation because day8-findings does not document it (its §2e row marks orchestration HIT); only the Cohere silent failure and the score-distribution problem, which the file does carry, were cited as Layer-3 evidence.
- Surprising: the ADR-003 stripped-citation finding is the first confirmed instance of the humanization-stripped-substance failure mode in a committed ADR. Flagged for the Day 14 audit to scan the other early-humanized ADRs for similarly stripped citations.
- Deferred: Day 14 codebase audit verifies these ADR claims against the implemented v2 src/ (PRD §12.5). The contiguous ADR set after Batch B is 001-016, 16 files; the planning prompt's "14" figure is a mismatch to confirm with Ruby.
- ADR candidate: no. These four are the decisions; no further decision surfaced during authoring.

End of Batch B Phase 3: all four new ADRs committed (af6d55b, a0e0b2f, 06a10c8, 3af0be0) on feat/day9-adr-batch-b. Batch B new-ADR authoring complete; all 16 ADRs (001-016) now exist on disk.

