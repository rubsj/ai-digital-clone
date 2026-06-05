# ADR-020: Replace cosine groundedness with a local entailment scorer (HHEM-2.1-Open)

- Status: Accepted. Model resolved: HHEM-2.1-Open. Routing threshold GROUNDEDNESS_MIN = 0.40.
- Date: 2026-06-03 (decided), completed 2026-06-04 (model resolved by bake-off; threshold derived W1b.2)
- Project: P6 Digital Clone (v2)
- Relates: follows ADR-019 (confound); residual bias handled in ADR-021; keeps ADR-009/014/018 (boundary, deterministic router); amends ADR-004 (pending, end of W1b)

## Context

ADR-019 established that the cosine groundedness scorer measures lexical echo, not containment. It has to be replaced. Three constraints fixed the shape of the replacement, from the architecture as it ships.

The router is live and deterministic (ADR-018): the Gatekeeper Component floors a per-response groundedness number against `GROUNDEDNESS_MIN` at inference time, and that number is the live out-of-domain hallucination defense (PRD section 2.1 makes OOD-fallback = 100% non-negotiable). So the replacement must run live, in-process, within the evaluate per-step budget, and produce a number a deterministic floor can trust. ADR-004's original rejection of an LLM judge on latency grounds therefore hardens here: an LLM-produced number adds per-query latency and breaks the router's determinism, since a deterministic comparison over a non-deterministic input is not deterministic, and temperature zero is not a guarantee. The corpus (LKML kernel-development prose) is also out-of-domain for the general-domain data these models train on, so the model could not be chosen from a public leaderboard.

## Decision

Replace cosine with **HHEM-2.1-Open** (Vectara), a local FLAN-T5-based factual-consistency model (110M parameters), run in-process in the ScoringEngine Component. Groundedness is scored per-sentence with the chunk as premise and the generated sentence as hypothesis, max over the top-5 chunks per sentence, mean over sentences (the V0 aggregation; Probe A confirmed alternatives do not improve it). HHEM is invoked through its own `model.predict()` path with its internal tokenizer and prompt template, which is the correct call path (the generic pipeline interface produced flat scores).

The number stays deterministic and Component-owned. There is no Agent/Component boundary move, so ADR-009, ADR-014, and ADR-018 hold. The metric is local and in-process: weights pinned at build time, loaded at startup like the FAISS index, no external call on the routing path.

HHEM was selected by the pre-registered bake-off against the Day-13 containment oracle (no API spend, no generation). The bake-off returned **none clears**: no candidate passed all four pre-registered gates. HHEM was chosen as the best-aligned and least-confounded model in the field, not as a gate-passer. Its residual held-equal paraphrase bias is accepted and handled at the per-leader floor per ADR-021, rather than treated as disqualifying. The routing threshold was derived at W1b.2 (`GROUNDEDNESS_MIN = 0.40`, see Quantified Validation) and is recorded with the supersede note in the ADR-004 amendment.

## Alternatives Considered

- **The W1 metric family.** LLM-faithfulness judge: rejected for the live gate (non-deterministic, latency), parked as a possible offline oracle. Promoted Opus instrument: its role is the calibration oracle the bake-off scores against, not the live metric. Hybrid: only coherent as local-live plus LLM-offline, and the local-live half is what this ADR fixes.
- **The bake-off field, within the entailment family.** DeBERTa-v3 NLI (control): weakest, fails the length gate (r 0.307) and only 8/14 direction. MiniCheck flan-t5-large: clean on length (r 0.052) but worst on direction (6/14) and a fully systematic KH lean (7/7). HHEM: best oracle agreement (sentence-AUC 0.858), least length-coupling (r 0.174), passes OOD separability (AUC 0.942). HHEM wins on every axis that the others split, which is why it is the model despite none-clears.
- **Keep searching for a clean local model.** Rejected. The bake-off ran three model families and the held-equal bias appeared in all three, so a cleaner local model is not on the shelf for this corpus. The bias is a property of surface-sensitive metrics on terse synthesized prose, not a model-quality gap.

## Quantified Validation

Bake-off against the Day-13 oracle, four pre-registered gates, no spend. None cleared.

| Gate | Bar | DeBERTa-v3 NLI | MiniCheck flan-t5-large | HHEM-2.1-Open |
|---|---|---|---|---|
| G1 held-equal `|T-KH|` | <= 0.05, no systematic lean | 0.084 (fail) | 0.057, systematic KH 7/7 (fail) | 0.060, KH 6/7 (fail) |
| G2 direction vs oracle | >= 12/14 | 8/14 (fail) | 6/14 (fail) | 6/14 (fail) |
| G3 length `|r|` | <= 0.20, p > 0.05 | 0.307 (fail) | 0.052 (pass) | 0.174 (pass) |
| G4 OOD separability AUC | >= 0.85 | 0.875 (pass) | 0.940 (pass) | 0.942 (pass) |
| Tiebreak sentence-AUC | higher is better | 0.800 | 0.826 | 0.858 |

Probe A (metric reshape, no spend) tested four aggregations over HHEM's per-pair scores against a pre-registered bar (improve G1 and G2 while holding G4 >= 0.90). None cleared. Mean-over-chunks gave the best G1 (0.035) but dropped G2 to 5/14; top-2 and softmax variants behaved the same way. A leader-blind monotonic calibration was excluded by construction, since it cannot change per-query direction and so cannot move G2. The held-equal lean is intrinsic at the per-(span, chunk) level and aggregation-invariant. This evidence is the basis for ADR-021.

Pending: the corrected re-gate (W3a metric effect, W3b retrieval effect) and the W3c per-leader floor.

Threshold derivation (W1b.2, no spend, oracle labels). `GROUNDEDNESS_MIN = 0.40` on HHEM's scale, derived by a pre-registered safety-asymmetric rule: maximize grounded deliver rate subject to catching at least 90 percent of should-fall-back content (ungrounded plus OOD), on a query-level train split, validated held-out. The cosine-era 0.60 does not transfer; at 0.60 HHEM would route about 42 percent of oracle-grounded content to fallback, because HHEM's entailment scores run lower than cosine's lexical-echo-inflated scores. The safety-constrained train point was 0.4368 and 0.40 was chosen over it for robustness: 0.40 sits mid-band in the [0.38, 0.44] stability interval and does not balance the zero-bias property on a single response's score. The oracle deliver-versus-fallback cutoff (grounded < 0.50) is an analyst choice, not pre-registered, and is recorded as such.

## Consequences

HHEM sits on the live path, local and deterministic, in ScoringEngine, with no boundary move, so the rework's settled architecture holds. Two real costs follow.

First, a production dependency conflict, now resolved by vendoring. HHEM loaded only under transformers ~4.x via `model.predict()` and failed on the project's transformers 5.x. The mechanism: transformers 5.x sets `all_tied_weights_keys` during weight loading and reads it back, but HHEM's remote `HHEMv2ForSequenceClassification` never has it set, so loading raises `AttributeError` (not a sentencepiece issue; sentencepiece installs cleanly). The dependency investigation established that nothing in the live stack hard-requires transformers 5.x (only `sentence-transformers` declares a constraint, `>=4.41,<6`, which 4.x satisfies); the 5.x in the lock was merely the newest resolvable version. Three paths were costed (pin 4.x project-wide, vendor the modeling code, subprocess isolation) and there is no transformers-5-compatible upstream HHEM revision. **Decision: vendor.** HHEM's `modeling_hhem_v2.py` and `configuration_hhem_v2.py` are copied into `src/evaluation/hhem/`, pinned to hub commit `8e4a2e6e96c708cc76c2344f7e4757df2515292c`, with one semantically-correct change (`all_tied_weights_keys: dict = {}` at class level, correct because HHEM ties no weights) and `trust_remote_code=True` removed. This keeps the live env on 5.x, localizes the change to two files, removes remote-code execution from the live routing path, and preserves the local-in-process-deterministic property ADR-020 requires (subprocess isolation was rejected precisely because a process boundary would break "in-process"). The vendored load path must reproduce the full working call path, not only the tied-weights fix: `DebertaV2Tokenizer` substitution, `token_type_ids` stripped before the forward pass, and `model.predict()` with its prompt template (the generic pipeline interface produced flat ~0.29 scores). `sentencepiece` is a production runtime dependency for HHEM's tokenizer and belongs in main dependencies. Provenance (source commit and the one-line change) is recorded in the vendored directory so the modification is auditable.

Second, a residual leader-blind paraphrase bias on the held-equal queries, accepted here and compensated at the W3c per-leader floor under ADR-021, not at the metric. ADR-004 must be amended at the end of W1b to mark cosine superseded and record the threshold on HHEM's scale. PRD sections 2.4, 2.5, 2.11 and the cosine-specific distribution are stale against this gate and are flagged for the Wrap-time PRD reconciliation pass.

(For a JVM/TS reader: this is swapping a `String.contains` check for a parser-backed validator inside a request-validation filter. Same place in the pipeline, still an in-process call rather than a per-request network service, and it stays inside the deterministic filter so the routing logic above it does not change. The dependency conflict is the equivalent of a validator library that needs a tweak to run on your runtime: rather than pin the whole service back to an old runtime to suit it, you vendor the library's relevant source, apply the one correct fix, and own it, which also drops its remote-code loading.)
