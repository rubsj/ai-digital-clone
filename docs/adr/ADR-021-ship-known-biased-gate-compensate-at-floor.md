# ADR-021: Ship the groundedness gate with a known paraphrase bias, compensate at the per-leader floor

- Status: Accepted
- Date: 2026-06-04
- Project: P6 Digital Clone (v2)
- Relates: depends on ADR-020 (HHEM selected); reinforces ADR-019 (confound); constrains ADR-015 (per-leader floor, W3c)

## Context

ADR-020 selects HHEM-2.1-Open, and the bake-off that selected it returned "none clears." The residual that no candidate cleared is a held-equal paraphrase lean toward KH: on the 7 queries the oracle rates as equally grounded across clones, the entailment models still score KH slightly above Torvalds. KH echoes the source vocabulary, Torvalds paraphrases supported claims in his own words, and the models retain a mild surface-form sensitivity at the per-(span, chunk) level that rewards the echo.

The investigation closed off the innocent explanations. The lean reproduced across three model families (cosine, DeBERTa NLI, MiniCheck, plus HHEM), so it is not a single model's quirk. It survived a clean-retrieval check (the W2 duplicate-chunk bug was shown not to touch any held-equal query), so it is not a retrieval artifact. It survived four aggregation variants in Probe A and a monotonic calibration was excluded by construction, so it is not reachable by reshaping the score. And the oracle's own inference labels show no systematic lean toward Torvalds (he carries fewer inferable spans than KH, 12.2% versus 15.1%), so the gap is not a real grounding deficit. The conclusion is narrow and well-supported: the clones are equally grounded, and surface-sensitive metrics on this corpus carry a small bias against paraphrase. No local strict-entailment model matches the oracle's paraphrase-robust construct, because that construct is an LLM-style judgment we have ruled off the live deterministic path.

This forces a three-way decision that three constraints cannot jointly satisfy: a live deterministic local gate, a groundedness number robust to paraphrase, and selection only from local strict-entailment models. The bake-off proved the first and third cannot deliver the second on this corpus. Something has to give.

## Decision

Ship the live gate with HHEM and its characterized residual bias, documented explicitly, and compensate for the bias at the per-leader floor (W3c), not at the metric. The bias is leader-blind in mechanism (it penalizes paraphrase distance from the source, which correlates with leader identity but is not keyed on it) and small in magnitude (held-equal mean absolute clone difference about 0.06). The live gate's actual job is to route deliver-versus-fallback and to send out-of-domain queries to fallback; HHEM does that job (OOD separability AUC 0.942). The held-equal parity that the metric fails is a clone-versus-clone comparison the router never performs, since the router scores one response against one threshold and does not compare the two clones.

## Alternatives Considered

This is the three-door fork, decided on the bake-off and probe evidence.

- **Door A, reshape the metric to remove the bias.** Rejected. Probe A tested four aggregations against a pre-registered bar and none cleared; the bias is intrinsic at the per-pair level and aggregation-invariant, and a leader-blind monotonic calibration cannot move the direction metric. Per-leader calibration would close it but is banned as the floor-rescuing anti-pattern relocated into the metric.
- **Door B, demote groundedness to an offline metric and route the live gate on something else.** Rejected. It reopens the live-versus-offline boundary settled on Day 12, costs real rework, and weakens the live OOD-hallucination defense that the groundedness gate currently provides.
- **Door C, accept and compensate at the floor.** Chosen. Keeps the gate local, deterministic, and shipping, at the cost of a documented bias and a per-leader floor adjustment.
- **Keep searching for a clean local model.** Rejected. The bake-off across three families shows no clean local metric exists for this corpus.

## Quantified Validation

The bias is bounded and characterized. Held-equal mean absolute clone difference is 0.06 with a KH lean on 6 of 7 queries (HHEM); the single Torvalds-leaning held-equal query (q04) confirms the lean is a phrasing effect, not a consistent grounding signal. The oracle inference-label cross-check shows no Torvalds inference lean (Torvalds grounded 87.8% / inferable 12.2%; KH grounded 84.9% / inferable 15.1%). The gate does its routing job: OOD separability AUC 0.942, well clear of the 0.85 bar, so deliver-versus-fallback and the OOD catch are intact. The floor compensation itself is pending W3c and is not validated here; it must be pre-registered against the measured bias magnitude, not back-fit to make Torvalds pass.

## Consequences

The live gate ships and works at its real job, with a known, measured, and documented bias on a comparison the router does not make. The load-bearing risk moves downstream: the W3c per-leader floor must be pre-registered and oracle-grounded, or "compensate at the floor" silently becomes the floor-rescuing move this project has refused throughout. The same discipline that governed the bake-off criterion governs the floor: the adjustment is justified by the measured bias magnitude, decided before the rate it produces is seen. The ADR-015 floor amendment at W3c will record this compensation, additive, with the original numbers intact. For the portfolio and interview, the bias is documented openly as a senior-judgment decision: the bias was measured, shown to be intrinsic rather than retrieval or grounding, and compensated transparently rather than hidden or overfit.

(For a JVM/TS reader: this is shipping a validator with a documented false-negative rate on one input class, with the tolerance handled by a calibrated threshold per class rather than by hacking the validator's internals. You accept a measured, bounded error and correct for it at the policy layer, with the correction itself reviewed so it does not quietly become "tune until the test passes.")
