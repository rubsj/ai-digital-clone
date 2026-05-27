# ADR-011: EvaluatorAgent Hybrid Design

**Status:** Accepted
**Date:** 2026-05-26

## Context

EvaluatorAgent has two responsibilities in v2 that don't share a tool. It produces three numerical scores (style, groundedness, confidence), and it produces a human-readable explanation plus a list of flags for GatekeeperAgent to reason over.

The scoring side is deterministic. Cosine similarity on style vectors and sentence-level groundedness math have exact answers, and downstream routing depends on scores being trustworthy and reproducible. The LLM has no judgment to add on top of the math. The explanation and flags side is the opposite. Identifying that a response mentions slab allocator while the chunks discuss buddy allocator is interpretation over text, which deterministic Python cannot do without becoming an LLM.

The vocabulary lock from ADR-009 has to accommodate this: an Agent that internally delegates the deterministic part to a Component (ScoringEngine), rather than two siblings at the Flow level. The v1 `EvaluationResult` had a `final_score` field consumed by the routing formula. In v2 there is no `final_score`; GatekeeperAgent reasons over the three individual scores plus explanation and flags (per ADR-010).

## Decision

EvaluatorAgent is a CrewAI Agent that delegates numerical scoring to the ScoringEngine Component and uses the LLM only to generate the explanation and flags. The Agent's output is an `EvaluationResult` with three scores from ScoringEngine, an LLM-generated `explanation` string that references the scores, and an LLM-identified `flags: list[str]`. `EvaluationResult` has no `final_score` field, and the weighted formula is removed from the codebase.

## Alternatives Considered

- Pure Component (no LLM, just scoring). This is the v1 design. It loses the explanation and flag capability that GatekeeperAgent needs to do better-than-formula routing. Without flags, GatekeeperAgent is back to reasoning over three numbers, which barely improves on the formula and re-creates the Day 8 problem.
- Pure Agent (LLM does scoring too). The deterministic cosine math is already strongly correlated with LLM judgment on groundedness (Pearson 0.82, PRD §10.5), so an LLM scorer adds no measurable lift while widening the system's failure surface and making scores non-reproducible. The deterministic backbone is what the rest of the system reasons over.
- Two separate top-level units (ScoringAgent + ExplanationAgent). ScoringAgent would fail the ADR-009 criterion (no LLM reasoning) and collapse to a Component anyway. The split would be one Agent plus one Component named badly, which is the hybrid pattern with worse naming. The hybrid framing is cleaner because the Component sits inside the Agent's responsibility surface.

## Quantified Validation

- Pearson 0.82 correlation between deterministic cosine groundedness scoring and GPT-4o-mini judgment (PRD §10.5). An LLM scorer on top of the deterministic one offers no measurable lift and expands the system's failure surface.
- Day 8 surfaced cases where high style scores masked low groundedness (Finding 1: Torvalds style mean 0.9025 vs groundedness mean 0.6258, a 0.28 gap). Flags are what let GatekeeperAgent catch this; pure scoring without flags re-creates the Day 8 problem.
- The third reason is architectural rather than empirical: numerical computation belongs to deterministic code, and interpretation belongs to the LLM.

## Consequences

- Positive: scoring stays deterministic and testable with synthetic inputs, while explanation and flags get LLM interpretation where it actually helps.
- Positive: `EvaluationResult` carries more signal for GatekeeperAgent than three numbers, which is what makes ADR-010's routing improvement work.
- Positive: ADR-009's vocabulary holds inside a single responsibility surface, without forcing the Agent/Component split up to the Flow level.
- Negative: the Agent-calls-Component pattern has to be made explicit in code structure. ScoringEngine is injected into EvaluatorAgent, not co-located.
- Negative: integration testing requires recording the LLM call for the explanation and flag portion. Handled by Layer-2 contract tests per ADR-016.
