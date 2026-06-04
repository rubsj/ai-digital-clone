# ADR-019: The groundedness scorer measures lexical echo, not containment

- Status: Accepted
- Date: 2026-06-03
- Project: P6 Digital Clone (v2)
- Supersedes/relates: informs ADR-020 (replacement metric), ADR-004 amendment (pending), ADR-015 (floor, conditional)

## Context

v2 routing is a deterministic gate. The Gatekeeper Component floors a per-response groundedness number against `GROUNDEDNESS_MIN` and routes deliver or fallback (ADR-018). That number is produced live in the ScoringEngine Component as per-sentence max cosine against the top-5 retrieved chunks, then a mean over sentences.

Days 12 and 13 opened as a ship/no-ship investigation into a Torvalds groundedness deficit: the Torvalds clone scored below the KH clone on groundedness across the in-domain set, and below the ADR-015 floor on several queries. The working assumption was a generation deficit in the clone, which framed an H2 versus H3 fork (two competing theories of what to excise from the Torvalds output). Before acting on either, Day 13 asked the prior question: is the deficit a property of the clone, or an artifact of how the score is computed.

## Decision

We conclude the groundedness scorer measures lexical echo, the degree to which a sentence reuses the source's vocabulary, rather than containment, whether the claim is supported by the source. Cosine similarity is symmetric and rewards shared wording; groundedness is a directional support question that should be indifferent to wording. The two coincide only when the clone restates the source in the source's own words.

Torvalds is not less grounded. He writes terse, synthesized prose that does not parrot the source, and cosine penalizes exactly that. The metric is the defect.

Three consequences follow from this decision. We do not change the clone; there is no containment deficit to excise, so the H2/H3 fork is closed as resting on a false premise. Routing (ADR-018) is honest and stays untouched; the router applied the floor correctly to a number that was itself misleading. And the metric is replaced rather than tuned, which is decided separately in ADR-020.

## Alternatives Considered

- **Excise a Torvalds containment deficit (the H2/H3 fork).** Rejected. The investigation showed no deficit exists; both forks assumed one.
- **Length-normalize cosine to remove the verbosity correlation.** Rejected. It treats a symptom. A normalized cosine still answers a similarity question, so it still rewards echo over support.
- **Accept the deficit and lower the Torvalds floor.** Rejected. This moves the floor to rescue a rate against an instrument we now know is broken. Floor decisions wait for a corrected measurement.

## Quantified Validation

The verdict was confirmed three independent ways on the in-domain set, plus a baseline disagreement check against a blind containment instrument (Opus per-span grounded/inferable/free markup, produced without sight of the scorer's numbers).

- **Probe A, verbosity.** Score correlated with response length at r = +0.39. Longer, more source-echoing text scored higher independent of whether it was better grounded.
- **Probe B, containment held equal.** On queries the blind instrument rated as equally grounded across the two clones, the scorer still split them 7 of 7 toward KH, with a gap near 0.065.
- **Probe C, per-sentence mechanism.** Torvalds own-words sentences scored 0.17 to 0.27 cosine; KH source-echoing sentences scored 0.71 to 0.73 on the same containment-held-equal queries.
- **Baseline disagreement.** The blind instrument and the scorer agreed on the Torvalds-versus-KH direction on only 5 of 14 queries.

**Reinforcement (2026-06-04, Day 14).** The verdict held under wider testing. The held-equal lean toward KH reproduced across three further model families in the W1b.0 bake-off (DeBERTa-v3 NLI, MiniCheck flan-t5-large, HHEM-2.1-Open), and across four aggregation variants in Probe A, while the W2 retrieval-dedup bug was shown not to touch any held-equal query. The oracle's own inference labels show no systematic lean toward Torvalds (he carries fewer inferable spans than KH, 12.2% versus 15.1%), so the gap is not a grounding deficit hiding as a metric artifact. It is surface-form sensitivity shared by similarity and entailment metrics on this corpus. This strengthens the original verdict rather than softening it: the clones are equally grounded, and the metric family carries the bias.

## Consequences

A replacement metric is required, decided in ADR-020. ADR-004, which justified the cosine heuristic, must be amended to mark it confounded and to record the routing threshold on the replacement's scale; that amendment waits until the new threshold is derived (end of W1b). The ADR-015 per-leader floors were calibrated against this broken metric and cannot be trusted until re-measured on the corrected one; the corrected re-gate (W3) decides whether they hold, and the floor does not move to rescue a rate.

A separate, leader-agnostic side finding surfaced during the investigation: 6 of 14 in-domain queries retrieved duplicate chunks (the same passage 2 to 3 times in the top-5), cutting effective context. It does not explain the per-leader gap and is handled as its own workstream.

(For a JVM/TS reader: this is the shape of discovering a test that asserted on the string form of a formatted value when it should have asserted on the parsed value. The assertion passed and failed for the wrong reason, and every decision it gated is suspect until the assertion is corrected, even though the surrounding harness was sound.)
