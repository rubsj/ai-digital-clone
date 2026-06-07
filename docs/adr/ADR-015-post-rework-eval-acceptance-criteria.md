# ADR-015: Post-Rework Evaluation Acceptance Criteria

**Status:** Accepted

**Date:** 2026-05-26

## Context

After the rework completes, Day 11 runs the v2 evaluation query set (14 in-domain and 6 out-of-domain queries per leader, 40 records in dual-leader mode) through the new architecture. The question this ADR answers is what counts as success. Without criteria locked before the data lands, the post-rework evaluation degrades into reading the numbers and deciding whether they feel acceptable, which is rationalization after the fact.

The aspirational target cannot be the bar. PRD §2.1's routing-correctness grid sets an aspirational in-domain deliver target of 11 of 14 per leader (about 78%), but that figure was framed against the v1 architecture and was never measured under v2. v2's GatekeeperAgent reasons over the individual scores qualitatively rather than applying the v1 0.75-threshold formula (ADR-010), and a qualitative gate may legitimately deliver fewer responses than a fixed threshold did. Holding v2 to the v1 aspiration risks calling a correctly functioning architecture a failure for missing a number set under different assumptions.

The criteria therefore separate a target from a floor, with a defined judgment band between them, and they are fixed before Day 11 measures anything.

## Decision

Three-tier acceptance criteria, locked in advance of the Day 11 evaluation.

- E2 (target): in-domain deliver rate at or above 55% per leader, with OOD fallback at 100% and zero hallucinations on category-5 queries. If E2 is met, P6 ships.
- E1 (floor): in-domain deliver rate at or above 39% per leader, the level measured as the v2 baseline on Day 8. Below E1 the architecture has regressed and P6 does not ship. Between E1 and E2 is a judgment call, where the deliver rate is documented as the system's operating point and P6 ships if the other criteria pass.

OOD fallback at 100% is non-negotiable for shipping regardless of where the in-domain rate lands, because a hallucination on an out-of-domain query is the worst signal the system can produce.

## Alternatives Considered

- Use the aspirational target as the ship gate (PRD §2.1's roughly 78% in-domain, treated as E3). Rejected as the bar. That target was set for v1, and v2's GatekeeperAgent makes qualitative routing decisions that may deliver fewer responses than a fixed-threshold formula (ADR-010). Treating E3 as the gate invites the outcome where the architecture works correctly, misses an aspirational number, and gets declared a failure anyway. E3 stays in PRD §2.1 as stretch, not as the ship gate.
- Use a single threshold, for example a deliver rate at or above 50% as one pass-or-fail line. Rejected. A binary line hides the difference between barely clearing it, which signals a fragile architecture, and clearing it comfortably, which signals a robust one. Three tiers force explicit reasoning about the middle band instead of pretending it does not exist.
- Lock no criteria and decide at Day 11 once the numbers are in. Rejected. Deciding after the data lands is post-hoc rationalization. The criteria are locked first and measured against second.

## Quantified Validation

- E1's floor of 39% is measured, not chosen. It is the in-domain deliver rate from the Day 8 v2 run, where 11 of the 28 in-domain records in the full routing matrix delivered (`docs/day8-findings.md`, the routing-correctness 2x2 and the §2d fallback-rate scorecard row in that file, where §2d is a legacy label internal to the scorecard rather than a current PRD section). Day 11 has to clear that level per leader to count as no regression.
- The OOD criterion is measured rather than aspirational. On Day 8 the system fell back on all 12 out-of-domain records (0 of 12 delivered) with zero hallucinations, so the 100% OOD fallback criterion locks in a property the architecture already showed rather than a stretch it has to reach.
- E2's 55% is a chosen target, not a derived or measured value. It sits above the 39% baseline by roughly four additional in-domain queries delivering per leader out of 14, picked as a substantive improvement without holding the rework to the v1 aspiration.
- The measurement these criteria judge is the Layer-3 system evaluation defined in ADR-016 (Evaluation Methodology, Three-Layer Approach), run on Day 11. ADR-013's per-leader style trigger, which fires if Kroah-Hartman's in-domain deliver rate lands more than 20 points below Torvalds's, is measured in the same Day 11 run, so the two ADRs share one evaluation event.

## Consequences

The Day 11 decision is a defined outcome rather than a verdict argued after the fact, because the criteria are fixed before the run. The system either clears E2 and ships, or it falls below E1 and does not, and the band in between has its handling fixed here in advance: the deliver rate is documented as the operating point and P6 ships if the other criteria pass. This ADR is the citation point for the Day 11 evaluation report (`docs/day11-evaluation.md`), which records the measured rates against E1 and E2. Locking the criteria before the measurement, including the rule for the middle band, is what keeps the Day 11 decision from being reasoned backward from whatever the numbers happen to be.

---

## Amendment — Day 12 (2026-06-01)

**Floor correction.** The E1 floor of 39% per leader was derived from the pooled Day-8 baseline of 11/28 = 39.3%. Applied per leader, 39% would set Kroah-Hartman's floor at 5.46/14, above Kroah-Hartman's own Day-8 per-leader baseline of 5/14 = 35.7%. This would cause a false regression if Kroah-Hartman matched its own Day-8 level exactly. The floor is therefore corrected to honest per-leader baselines:

- Torvalds floor: 42.9% (6/14), matching the Torvalds Day-8 per-leader baseline.
- Kroah-Hartman floor: 35.7% (5/14), matching the Kroah-Hartman Day-8 per-leader baseline.

This change does not alter the original Decision section's E2 target, OOD non-negotiable, or the band-handling rule. It corrects a mis-derivation in E1 that would have fired a false regression alarm against Kroah-Hartman.

**Sub-floor handling.** The Day-12 measurement produced 0/14 deliver rate for both leaders across all three passes. This falls below the corrected floors. Per the original Decision: "Below E1 the architecture has regressed and P6 does not ship." That rule stands. The investigation branch applies: read the GatekeeperAgent reasoning and scores to distinguish correct conservatism from pathological punting on groundable queries.

Day-12 investigation finding: two compound root causes were identified.

Root cause 1 — EvaluatorAgent flag threshold calibration. The EvaluatorAgent prompt specifies groundedness target 0.60 and instructs the LLM to flag any dimension below its target. The LLM is raising `low_groundedness` for scores up to 0.706 (q08 Kroah-Hartman). The effective flag threshold is approximately 0.70-0.75, not the stated 0.60.

Root cause 2 — GatekeeperAgent routes on flag presence. The Gatekeeper routes to fallback whenever `low_groundedness` appears in the EvaluatorAgent flags, regardless of the actual score value. In several cases the Gatekeeper also makes arithmetic errors in its reasoning (for example, describing gs=0.651 as "below the acceptable threshold of 0.60"). The Gatekeeper is following the flag, not re-evaluating the number.

These two root causes compound: EvaluatorAgent over-flags above the stated threshold, then Gatekeeper follows the flag. The sub-floor deliver rate is therefore not evidence that the corpus is wrong or that the CloneAgent responses are ungroundable. It is evidence that the scoring and routing layer is more conservative than its own stated targets.

The pre-floor branch outcome as defined in this ADR: stop ship, investigate, document the root cause. Investigation is complete (see `docs/day11-evaluation.md`). The fix is a re-calibration of the EvaluatorAgent flag threshold and a Gatekeeper prompt update to evaluate scores numerically, not just flag presence. A re-measurement after those changes determines whether the floor is cleared.

Notion sync of this amendment is scheduled for Phase 2 alongside the ADR-010 trigger_category amendment logged Day 11.

---

## Amendment (2026-06-05, Day 14 W3c): floors confirmed on HHEM; paraphrase misroute documented as accepted limitation

The per-leader floors (Torvalds 42.9%, KH 35.7%) are unchanged. This amendment validates them on the corrected metric and reframes their role. It revises no prior number.

**Floors validated on the corrected metric.** The Day-14 W3a isolated metric effect (frozen inputs, HHEM at GROUNDEDNESS_MIN 0.40, W2 dedup active) measured Torvalds at 9/14 (64.3%) and KH at 11/14 (78.6%). Both clear their ADR-015 floors by 21 and 43 points respectively, and both clear PRD 2.1 E2 (at least 55%) on the corrected metric. The floors hold; they are not moved.

**Floors reframed as regression guardrails, not bias compensation.** ADR-021 committed to compensating the intrinsic paraphrase bias "at the per-leader floor." W3c establishes that this is the wrong instrument and declines to do it. The bias is per-query: on hard queries a grounded Torvalds response scores below the shared gate and falls back. A per-leader floor is an aggregate rate guardrail; it cannot discriminate a paraphrase-penalized grounded response from a genuinely weak one, so it cannot fix a per-query misroute. Lowering Torvalds' threshold to rescue these cases would weaken the shared OOD/hallucination defense (the same GROUNDEDNESS_MIN gates OOD fallback at AUC 0.942) and is the leader-keyed calibration ADR-021's Door A already rejected. The floor's honest role on HHEM is to confirm delivery has not regressed below the ADR-015 baseline. That role is confirmed.

**Accepted limitation (ADR-021 Door C, made concrete).** On hard paraphrased queries a grounded Torvalds response can score below 0.40 and route to fallback. Surviving in-sample exemplar after the W2 retrieval fix: q07 T (oracle grounding 53.6%, W3a HHEM 0.285, W3b 3-pass mean 0.335; the dedup improved it reliably yet it remains below the gate). Broader class: q14 T (oracle 73.4%, W3a 0.369) and q03 T (oracle 64.2%, W3a 0.385; resolved by dedup in W3b). This is a characterized, bounded limitation of a deterministic gate on a paraphrase-sensitive metric. It is not a generation defect (ADR-019 verdict), not a retrieval defect (W3b improved q07 T and it still fell back), and not a floor-calibration defect. It is the operational cost ADR-021 accepted when it chose a local deterministic gate over a paraphrase-robust LLM judge: occasional fallback on hard grounded Torvalds queries, documented rather than hidden.

**Regime-shape question closed.** ADR-021 left open whether a flat per-leader offset was the right compensation shape. It is not. A flat offset is aggregate and would license delivery for any Torvalds response just under the floor, including legitimately weak ones, while still not targeting the specific hard-query misroutes. The compensation shape and the misroute shape do not match, which is why (a)+(c) declines to use the floor as a compensation instrument at all.

Relates: ADR-021 (bias characterized, Door C), ADR-019 (metric confound), ADR-020 (HHEM selected, threshold 0.40), W3a (isolated metric effect). Additive amendment; original Decision numbers (42.9%, 35.7%) and derivation rationale intact.
