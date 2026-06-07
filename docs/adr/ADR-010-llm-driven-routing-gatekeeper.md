# ADR-010: LLM-Driven Routing via GatekeeperAgent

**Status:** Superseded
**Date:** 2026-06-01

## Context

v1 routed deliver vs fallback with a fixed weighted formula, `final = 0.4×style + 0.4×ground + 0.2×conf`, followed by a hard threshold at `final ≥ 0.75`. Day 8 verification on the v2 query set found the formula does not discriminate on this corpus.

The formula compresses three signals into one number and loses their interaction. A high-style low-groundedness response reads the same as one that is mediocre across the board, even though the first is dangerous and the second is unimpressive. The score distribution is corpus-dependent, so no threshold ports to a new corpus. And the 0.4/0.4/0.2 weights came from intuition rather than measurement.

The fallback path dominates under any threshold reachable in v2, landing in the 30-70% range depending on the choice. Routing quality is the everyday user experience, not an edge case.

A correct routing decision has to reason about flag interactions that no formula reduction can express. EvaluatorAgent's `flags` list surfaces things like "response mentions slab allocator but chunks discuss buddy allocator," which must force fallback regardless of the numeric scores.

## Decision

Replace the weighted formula and 0.75 threshold with a GatekeeperAgent that reasons over the query, the styled response, the retrieved chunks, EvaluatorAgent's three individual scores, and the explanation and flags. The Agent outputs a `RoutingDecision` containing `decision: Literal["deliver", "fallback"]`, a `reasoning` string that must reference specific scores and flags, and an optional `trigger_reason` passed to FallbackAgent. The Agent runs at `temperature=0` with Instructor-structured output. No `final_score` field exists anywhere in v2.

## Alternatives Considered

- Recalibrate the formula weights via sensitivity sweep. Day 6's scoring-weight sensitivity experiment was a null result (PRD §10.3 marks it obsolete). The score band is too narrow on this corpus for reweighting to move the deliver/fallback split meaningfully. The problem is signal compression, not weight choice.
- Replace the single threshold with multi-criteria thresholds (e.g., `style > 0.7 AND ground > 0.5`). Same problem, one level up. The thresholds are still arbitrary, still corpus-dependent, and still blind to flag interactions like a confident-hallucination pattern where high style hides low groundedness.
- Remove routing entirely and always deliver with score annotations. PRD §1's product principle says it is better to say "I don't know" than to fabricate while pretending to be the leader. Removing the deliver/fallback decision violates this and produces confident hallucinations on out-of-domain queries, the failure mode Category 5 OOD probes target.

## Quantified Validation

- At threshold 0.75 the v1 May-23 eval (Cohere broken) produced 19/20 (95.0%) fallback. Source: day8-findings.md "Three-run comparison" table.
- At threshold 0.75 the v1 + Cohere working control run produced 18/20 (90.0%) fallback. Source: same table.
- The v2 final run (40 records, Cohere working) produced 29/40 (72.5%) fallback at threshold 0.75. Source: same table.
- Scored-record style means in the v2 final run: Torvalds 0.9025, Kroah-Hartman 0.8355, an asymmetry of +0.067 favoring Torvalds. Source: "Finding 1" table.
- Scored-record groundedness mean (v2 final): 0.6258, up from the 0.5173 broken-Cohere baseline. Source: "Three-run comparison."
- Category 5 OOD probes: 12/12 (100%) fallback with zero hallucinations. Source: routing-correctness 2×2.
- q12 binary-search regression. Same query, same index, same threshold (0.75): v1 May-23 delivered at `final=0.7525`; v1 + Cohere working and v2 q12 both fell back. day8-findings.md "Verification 2" traces the flip to downstream LLM and groundedness-scoring stochasticity combined with the single-good-chunk corpus shape, not to any infrastructure change. Routing flips across runs with no change to the scoring inputs the formula can see. No fixed threshold is stable on this corpus.

## Consequences

- Positive: Routing decisions become defensible. The Agent's `reasoning` string is the artifact pointed at in review.
- Positive: The decision ports to a new corpus or query set without re-tuning a threshold. The LLM reasons over whatever the scores happen to be on that corpus.
- Positive: Confident-hallucination handling improves through flag awareness. High style with low groundedness can be routed to fallback because the flag is explicit, where the formula was blind to it.
- Positive: This aligns with PRD §2.1 routing-correctness as the headline metric and the E1/E2 acceptance criteria in ADR-015.
- Negative: ~1-2s additional LLM latency per routing decision, covered by the 8s end-to-end budget per PRD §2.7.
- Negative: LLM reasoning has variance. `temperature=0`, Instructor structured output, and a prompt that demands explicit reference to specific scores and flags keep it bounded.
- Negative: The Agent is harder to unit-test than a formula. Mitigated by Layer-2 integration tests with recorded LLM responses per ADR-016.

(Similar to replacing a Drools-style rule engine with a domain expert service that reasons over the same facts: the rule engine is faster and deterministic but only useful when the rules cleanly partition the input space, and when they cannot, you need judgement rather than more rules.)
