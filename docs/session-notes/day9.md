# Day 9 Session Notes

## Phase 1: ADR-009 Agent vs Component Distinction

- Built: `docs/adr/009-agent-vs-component-distinction.md`. 5 H2 sections (Context, Decision, Alternatives Considered, Quantified Validation, Consequences). Decision is verbatim from `docs/plans/day9-plan.md`; Context is the plan outline rewritten as three prose paragraphs; Alternatives and Consequences are the plan's bullets with v1-references preserved.
- Why: this ADR formalizes the vocabulary lock that v1's failure mode required. Wrote Context as prose (per CLAUDE.md ADR-specific rule) rather than carrying the plan's bullet outline straight through. Java/TS parenthetical (Spring `@Service` vs `@Repository`) goes inline at the end of Consequences, not as its own section.
- Surprising: the input path in the execution prompt was `docs/plans/day9-adr-batch-a-plan.md`, but the file on disk is `docs/plans/day9-plan.md`. Same content. Flagged the naming mismatch and proceeded with the on-disk file as the contract.
- Deferred: ADR-010, ADR-011, ADR-012 — Phases 2-4 of this session, each behind its own stop gate.
- ADR candidate: no, this phase is the ADR.
