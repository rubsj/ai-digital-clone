# ADR-013: Style Profile Frozen, Re-Measured Day 11

**Status:** Accepted

**Date:** 2026-05-26

## Context

Day 8 v2 evaluation surfaced a measurement asymmetry in style scoring. Across the 11 records that reached the evaluator (6 Torvalds, 5 Kroah-Hartman), Torvalds scored a style mean of 0.9025 against Kroah-Hartman's 0.8355, a gap of +0.067 favoring Torvalds. PRD §4.9 carries the rounded form (0.91 against 0.84); the per-record figures in `docs/day8-findings.md` Finding 1 are the precise basis.

The asymmetry is a property of how the 15 hand-crafted style features behave on each leader's corpus, not a defect in the scorer. The feature set was derived from Schneider et al. (2016) on Torvalds-specific LKML signals (`docs/day8-findings.md`, PRD §3) and is applied uniformly to both leaders, which is the hand-crafted feature-vector decision recorded in ADR-003. Several dimensions encode habits that are natural in Torvalds' writing, so a Kroah-Hartman response written in his own authentic voice lands further from his profile on those dimensions than a Torvalds response lands from Torvalds' profile. This is design provenance.

The v2 architecture changes how the style score is used. v1 fed it into a 0.75 deliver threshold at 40% weight; v2 drops that formula, and GatekeeperAgent reasons over the three individual scores qualitatively instead (ADR-010). Qualitative routing might absorb the asymmetry, or the asymmetry might persist into the deliver decision. That is not yet known. The open question is what to commit to during the rework before the new architecture has been measured.

## Decision

Freeze StyleProfileBuilder and the 15 features for the first two rework days (Days 9 and 10), with no feature-engineering changes during that window. Re-measure the asymmetry on Day 11 under the new architecture.

Per-leader feature weighting (the Option S2 path from `docs/day8-findings.md`) becomes scheduled Day 12 to 13 work, triggered only if Kroah-Hartman's in-domain deliver rate at Day 11 is more than 20 percentage points below Torvalds's. Below that gap the asymmetry is treated as measurement noise and nothing further is built. Above it, it is treated as signal that earns the engineering investment.

## Alternatives Considered

- Do S2 now, building per-leader feature weighting during the rework. Rejected. The rework is already a six-day investment, and S2 may be wasted effort if GatekeeperAgent's qualitative reasoning absorbs the asymmetry on its own. Testing the new architecture first and adding S2 only if the measurement demands it is the cheaper sequence, and it is the more honest engineering narrative (an evidence-based call rather than preemptive complexity).
- Drop the numeric style_score from EvaluationResult and let GatekeeperAgent read sample emails directly (Option S3). Rejected. It removes the radar-chart visualization, a portfolio deliverable under PRD §2.10, and it eliminates a quantitative style metric from PRD §2.2. It also swaps a measurable signal for unconstrained judgment. That is too radical for rework scope.
- Ship as-is with the gap documented and no commitment to fix. Rejected. A documented gap with no trigger is a weaker engineering signal than a documented gap with an explicit threshold and scheduled contingency. The 20-point trigger and the Day 12 to 13 commitment turn an open question into a plan.

## Quantified Validation

- The asymmetry is measured Day 8 data. Torvalds style mean 0.9025 against Kroah-Hartman 0.8355, a +0.067 gap, from the Finding 1 table in `docs/day8-findings.md`. The basis is 11 scored records (6 Torvalds, 5 Kroah-Hartman), not the full 14 in-domain queries per leader, because only those records reached the evaluator.
- The asymmetry did not fully propagate to v1 deliver decisions because Kroah-Hartman compensated on groundedness. His groundedness mean was 0.6673 against Torvalds' 0.5913, so the two final means came out nearly identical under the old weighted formula (0.7808 Torvalds, 0.7804 Kroah-Hartman). This is the reason the style gap stayed hidden in v1 routing, and the reason it may behave differently once GatekeeperAgent reasons over the individual scores instead of a blended number.
- The 20-percentage-point trigger is a judgment call, not a value derived from a calculation. Small gaps across a 14-query in-domain set sit within measurement noise; a gap wider than 20 points across that set is unlikely to be random.
- The Day 11 re-measurement is committed future work, not a result. It runs under the new architecture as part of the Day 11 evaluation, and PRD §2.1 (the Day-11 acceptance criteria) and PRD §4.9 reference this freeze-and-re-measure plan.

## Consequences

The rework stays focused on architectural change rather than feature engineering during the days it is most fragile. Style scoring remains one input GatekeeperAgent reasons over (ADR-010), not a primary routing driver, so the frozen profile is not load-bearing for the deliver decision in the way it was in v1. Day 11 re-measurement is committed work with a defined trigger rather than an aspiration. If the trigger fires, Days 12 to 13 carry per-leader feature weighting; if it does not, the resolved asymmetry is documented as a Day 11 finding and nothing further is built.
