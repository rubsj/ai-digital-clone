# Day 14 — Session Notes

> Append per phase. This is the opening entry (Phase W1, the decision gate). No source code changed in this phase.

## Phase W1 — Metric decision gate (decision only, no code)

### Built
- The W1 decision: replace cosine groundedness with a local entailment scorer, in-process in the ScoringEngine Component, model and threshold selected empirically. Recorded in ADR-019 (confound, complete) and ADR-020 (replacement, drafted; model and threshold pending the bake-off).
- The bake-off spec and the pre-registered win criterion below, which is the next executable step (W1b.0).
- No `src/` changes this phase.

### Why
- The day14 plan contradicted itself on whether the groundedness gate is live or offline. ADR-018 settles it: the gate is a live deterministic routing control and the live OOD-hallucination defense (PRD section 2.1, OOD-fallback = 100%). So the replacement must run live, in-process, within the evaluate per-step budget, and measure containment.
- Prior fork resolved: the router stays deterministic (locked Day 12). A deterministic floor over a non-deterministic LLM number is not deterministic in effect, and temperature zero is not a determinism guarantee, so LLM-produced numbers are out of the live-gate slot. The entailment family in ScoringEngine is the clean fit, with no Agent/Component boundary move (ADR-009/014/018 hold).
- Category error named: cosine answers a similarity question (symmetric, vocabulary-sensitive); groundedness is an entailment question (directional, wording-indifferent). Patching cosine stays in the wrong family.

### Bake-off spec (W1b.0) — next executable step
- Goal: pick the live-gate model empirically against our own oracle. No API spend, no generation. Scoring stored data only.
- Candidates: HHEM-2.1-Open (front-runner); MiniCheck as flan-t5-large (its flagship sub-1B checkpoint, see the W1b.0 load-verification amendment below); a general DeBERTa-v3 NLI cross-encoder (control).
- Oracle and data: the Day-13 blind containment markup (Opus per-span grounded/inferable/free) on the in-domain set; the 7 oracle-equal queries; the 14 in-domain queries for both leaders; the OOD records for the separability gate.
- Metric shape under test: per-sentence entailment with chunk as premise and sentence as hypothesis, max over the top-5 chunks per sentence, mean over sentences. Same pipeline shape as the current cosine scorer, per-pair operation swapped.
- Pre-registered win criterion, LOCKED before the run:
  - Gate 1, held equal: mean `|score_T - score_KH| <= 0.05` on the 7 equal queries, no systematic lean. (Cosine: split 7/7 toward KH, gap ~0.065.)
  - Gate 2, direction: per-query Torvalds-vs-KH ordering agrees with the oracle on at least 12 of 14. (Cosine: 5/14.)
  - Gate 3, length: `|r(score, length)| <= 0.20` and not significant. (Cosine: +0.39.)
  - Gate 4, separability: grounded-vs-OOD AUC `>= 0.85`, or every OOD record below the grounded minimum. (Maps to PRD section 2.1.)
  - Tiebreak: among gate-passers, highest sentence-level agreement (balanced accuracy or AUC) with the oracle labels.
  - "None clears" is a permitted outcome: STOP and reconsider, do not ship the least-bad model.
- Sequence after the bake-off: W1b.0 bake-off, then STOP for Ruby review, then W1b.1 implement the winning scorer in `src/evaluation/` and wire `scoring_engine.py`, then W1b.2 derive `GROUNDEDNESS_MIN` on the winner's scale against the oracle. ADR-020 is completed and ADR-004 amended only after the threshold exists.

### W1b.0 load-verification amendment (2026-06-03)

The load check before the run surfaced a factual error in this spec and two load issues. Recorded here rather than silently patched. The four gates and the win criterion are unchanged, so pre-registration holds.

- **MiniCheck premise was wrong.** The original line said "pin the library's default supported checkpoint." The minicheck library is GitHub-only (not on PyPI), and its `__init__` default is `Bespoke-MiniCheck-7B`, the exact model marked out of scope. "Pin the default" and "7B is out of scope" cannot both hold. Resolution: drop the anti-hand-pick rule (it existed to stop pre-judging the winner, but the gates decide the winner, not which variant enters), and run MiniCheck as **flan-t5-large** (770M, its documented best sub-1B checkpoint), loaded directly from HuggingFace rather than via the GitHub package and its vllm dependency. Hard rule: MiniCheck's Flan-T5 expects a specific document-plus-claim input format; it must be fed in that format or it fails for the wrong reason. If correct formatting cannot be assured, drop MiniCheck rather than run it mis-fed.
- **HHEM-2.1-Open** loads from a public, non-gated repo but needs `sentencepiece`, which is absent from the venv. Resolution: add `sentencepiece` to the project's main dependencies (not dev-only, since HHEM would need it in production if it wins) and install. Not a workaround, a correct dependency.
- **DeBERTa-v3 NLI control** (`cross-encoder/nli-deberta-v3-base`, 184M) loaded cleanly, labels confirmed, eval mode.
- **Fallback:** if MiniCheck formatting stays uncertain, run HHEM + control only. Two models fed correctly beats three with one mis-fed; HHEM-vs-control still answers the core question.
- **Carried second-order flags:** if HHEM wins, `sentencepiece` is a production runtime dependency. If flan-t5-large wins, its 770M size needs a live-latency check against the evaluate per-step budget before W1b.1 locks it.

### Surprising
- The plan I believed was settled carried a live-vs-offline contradiction. The decision gate was genuine, not a formality.
- The three Day-13 gates test only relative clone discrimination. The live gate thresholds one absolute score, so a model can pass all three and still be unusable if grounded and OOD content share a score band. Added Gate 4 to cover the Day-8 tight-distribution failure mode the others miss.

### Deferred
- Model and threshold: pending the bake-off and W1b.2.
- ADR-004 amendment: end of W1b, needs the new threshold.
- ADR-015 floor: W3c, conditional on the corrected re-gate; floor does not move to rescue a rate.
- PRD reconciliation: Wrap-time pass (after W3 numbers and W4 field names). Checklist below.
- Offline LLM-faithfulness (RAGAS) as a possible reported or calibration oracle: parked, not the live metric.

### PRD reconciliation checklist (for the Wrap; do NOT edit the PRD now)
- Sections 2.4 / 2.5 / 2.11: groundedness reclassified from "informational, not a pass/fail gate" to a deterministic routing gate (ADR-018).
- Section 2.4 distribution "mean 0.55-0.70": cosine-specific; replace with the new metric's scale from W3a.
- Sections 1 / 5.2: "4 LLM-driven Agents (including GatekeeperAgent) + 3 Components" becomes 3 Agents + 4 Components, Gatekeeper as deterministic Component (ADR-014).
- ADR-010 LLM-routing references: mark superseded by ADR-018.
- Sections 2.7 / 5.2: ScoringEngine groundedness is now a local entailment model; note the latency budget and the warm-process deployment implication.

### ADR candidates
- ADR-019 confound: complete.
- ADR-020 replacement: drafted, model and threshold pending bake-off.
- ADR-004 amendment: flagged, end of W1b.
- ADR-015 amendment: flagged, conditional, W3c.

---

## Phase W1b.0 — Three-candidate bake-off (no code)

### Built
- Scored DeBERTa-v3-NLI and MiniCheck flan-t5-large in the main bakeoff script; HHEM scored separately in an isolated venv (see below). Results: `results/bakeoff_w1b0_day14.json` (DeBERTa, MiniCheck) and `results/bakeoff_hhem_isolated_day14.json` (HHEM). Winner: none.

| Model | G1 mean\|diff\| | G2 agree | G3 | G4 AUC | TB AUC |
|---|---|---|---|---|---|
| DeBERTa-v3-NLI | 0.084 **FAIL** | 8/14 **FAIL** | **FAIL** r=0.307 | PASS 0.875 | 0.800 |
| MiniCheck flan-t5-large | 0.057 **FAIL** (7/7 KH systematic) | 6/14 **FAIL** | PASS r=0.052 | PASS 0.940 | 0.826 |
| HHEM-2.1-Open | 0.060 **FAIL** | 6/14 **FAIL** | PASS r=0.174 | PASS 0.942 | **0.858** |

- All three fail G1 and G2. "None clears" outcome stands per the locked spec.
- Framing decision after result: **Door C** — ship HHEM (best on G3, G4, and tiebreak AUC) with documented paraphrase bias, compensated at the W3c floor.

**Task B — HHEM isolated-env load (part of this phase).**
The bakeoff script failed HHEM with `'HHEMv2ForSequenceClassification' object has no attribute 'all_tied_weights_keys'` — a transformers 5.x incompatibility. Fixed by creating `/tmp/hhem_isolated_venv` with `transformers==4.44.2` and `sentencepiece`. Confirmed: `HHEMv2ForSequenceClassification`, flan-t5-base foundation, 110M params, `model.predict()` interface with internal T5 tokenizer + prompt template. `sentencepiece` added as a production dependency (not dev-only).

**Task A — Oracle label inspection (part of this phase).**
Per-query oracle groundedness-fraction (GF) from `markup_output.json` + `blind_key.json`, to confirm HHEM's G2 failures are genuine disagreements rather than oracle noise:

| qid | T GF | KH GF | Oracle T>KH | HHEM agrees |
|---|---|---|---|---|
| q01 | 0.812 | 0.806 | YES (marginal) | NO — KH lean |
| q02 | 1.000 | 1.000 | TIE | NO — KH lean |
| q04 | 0.816 | 0.857 | NO | NO — T lean (wrong dir) |
| q05 | 0.352 | 0.333 | YES (marginal) | NO — KH lean |
| q06 | 0.963 | 0.944 | YES | NO — KH lean |
| q10 | 0.888 | 0.694 | YES (large) | NO — KH lean |
| q13 | 0.863 | 0.861 | YES (tiny) | NO — KH lean |
| q14 | 0.734 | 0.621 | YES | NO — KH lean |

q06 is the clearest failure: oracle T=0.963 vs KH=0.944, HHEM scores KH=0.743 vs T=0.610. q02: both leaders have GF=1.0 (every span grounded across all 3 passes); HHEM still scores KH 0.017 higher. Both confirm the lean is a scoring artifact, not an oracle ambiguity.

### Why
- All three fail G1 and G2. No candidate reduces the held-equal mean gap below 0.05 AND achieves at least 12/14 direction agreements simultaneously.
- DeBERTa additionally fails G3 (length correlation r=0.307, p<0.01) — the same confound family as the cosine metric it was meant to replace. Its entailment probability space clusters near zero on many pairs (q05: T=0.001, KH=0.001), producing near-degenerate scores on low-similarity queries.
- MiniCheck's G1 failure is structurally sharper: 7/7 KH lean (perfectly systematic), vs HHEM's 6/7. The systematic-lean flag alone disqualifies it per the gate definition, independent of the magnitude.
- HHEM's isolation requirement makes it a heavier production dependency than the spec anticipated: `transformers==4.44.2` pin and `sentencepiece` are both required at runtime.
- The oracle inspection confirmed the framing decision: HHEM's failures are real (the model disagrees with oracle-grounded reality in a KH-biased direction), not label noise. Door C is not "ship a broken metric" — it is "ship the best available metric with the bias quantified, and compensate at the floor."

### Surprising
- DeBERTa's G3 failure: an NLI cross-encoder scoring (premise, hypothesis) pairs should not be length-sensitive once entailment is properly framed; this suggests partial string-overlap signal surviving into the NLI head.
- MiniCheck's 7/7 KH systematic lean despite correct document+claim formatting. The bias is intrinsic to the model, not a preprocessing artifact.
- The HHEM transformers-version conflict is not documented on the model hub page; it is only discoverable at load time. Any production deployment of HHEM will need a version pin strategy or subprocess boundary.
- q04 is the only held-equal query where HHEM leans T — and the oracle has KH slightly more grounded on q04 (KH=0.857 vs T=0.816). HHEM's T lean on q04 is doubly wrong: wrong direction and inconsistent with the KH-echo hypothesis.

### Deferred
- W1b.1 (implement HHEM scorer in `src/evaluation/`, wire `scoring_engine.py`): blocked on Ruby's explicit Door C authorization.
- W1b.2 (derive GROUNDEDNESS_MIN on HHEM's scale): blocked on W1b.1.
- Production isolation strategy for transformers 4.44.2 pin: subprocess boundary vs project-level pin. W1b.1 architectural note.
- ADR-020 completion (model confirmed = HHEM V0 max; threshold pending W1b.2).
- ADR-004 amendment: blocked on ADR-020.

---

## Phase W2 diagnostic — dedup / held-equal scope gate (stopped)

### Built
- Confirmed via content inspection of `docs/experiments/day13/markup_input.json` that the dedup-affected queries and held-equal queries are completely disjoint:
  - **Dedup-affected** (1–3 duplicate chunks in top-5): q03, q07, q09, q10, q11, q14.
  - **Held-equal** (oracle both leaders equally grounded): q01, q02, q04, q05, q06, q12, q13.
  - Zero overlap. Scope gate triggered; stopped per pre-registered instruction.

### Why
- If the dedup bug affected any held-equal query, clean retrieval might give the model a different effective context and shrink the T-vs-KH gap. With zero overlap, it cannot. The held-equal queries all have clean (zero-duplicate) top-5 retrievals; their HHEM lean is fully explained by per-(span,chunk) score behavior, not by retrieval contamination.

### Surprising
- The split is clean and complete — not one held-equal query has even a single duplicate chunk. This makes the diagnostic unusually unambiguous; no partial-overlap edge cases to reason about.

### Deferred
- W2 production fix (dedup in `src/components/retriever.py` for the six named queries): independent of the metric decision, confirmed to have zero impact on held-equal queries. Can proceed independently at any time.

---

## Phase W1b.0 Probe A — Metric reshape (pre-registered, time-boxed, no variant clears)

### Built
- Single HHEM scoring pass (3,455 in-domain + 520 OOD pairs, zero paid API calls); four aggregation variants applied post-hoc to the same raw scores. Results written to `results/bakeoff_hhem_probe_day14.json`.
- Calibration excluded pre-run: a leader-blind monotonic transform preserves per-query T-vs-KH direction; G2 is a direction metric. Calibration cannot clear a bar requiring both G1 and G2.

| Variant | G1 | G1 Δ | G2 | G4 | TB AUC | Clears |
|---|---|---|---|---|---|---|
| V0 max (baseline) | 0.0597 | — | 6/14 | 0.9415 | 0.8576 | NO |
| V1 mean-over-chunks | 0.0351 | −0.025 | 5/14 | 0.9653 | 0.8399 | NO |
| V2 top-2 mean | 0.0500 | −0.010 | 5/14 | 0.9464 | 0.8581 | NO |
| V3 softmax-weighted (α=3) | 0.0487 | −0.011 | 5/14 | 0.9435 | 0.8590 | NO |

- Verdict: no variant clears. **Ship Door C — HHEM V0 max with documented paraphrase bias.**

### Why
- G1 improves with all non-max variants (V1 best: 0.0351, a 41% reduction). G2 regresses to 5/14 for all three. The simultaneous condition cannot be met.
- G2 regression mechanism: for V1 (mean-over-chunks), q08 flips from AGREE to DISAGREE (baseline: model T=0.727 > KH=0.721, oracle T>KH; mean: model T=0.349 < KH=0.356); q12 also flips (baseline: both KH>T in model and oracle; mean: model T=0.216 > KH=0.203 — now disagrees with oracle). q13 gains (DISAGREE to AGREE). Net: −2 + 1 = −1. V2 and V3 produce the same −1 pattern via identical q08 and q12 flips.
- The KH vocabulary-echo advantage is distributed across all five retrieved chunks, not only the single best-match chunk. Max preserves the highest per-span entailment probability; mean dilutes it. But the KH advantage survives the dilution on close-call queries (q08, q12), and the averaging flips those directions the wrong way.

### Surprising
- G4 (OOD separability AUC) improves with softer aggregation — V1 reaches 0.9653 vs baseline 0.9415. Averaging reduces within-class score variance, and the grounded-vs-OOD gap widens. This is a useful diagnostic property but irrelevant to the bar since G2 fails.
- q01 and q06 remain KH-leaning across all four variants including V1 mean (q06: diff=0.133 at V0, 0.086 at V1). The two dominant KH-lean queries are not correctable by aggregation — their KH advantage spans all five chunk scores uniformly.
- The pre-run calibration exclusion was correct before seeing the data: once the G1-G2 tension became visible in the variant results, it was clear calibration was never going to flip query-level rankings.

### Deferred
- All blocked workstreams remain: W1b.1, W1b.2, W3a–W3c, W4b, W4c, W6.
- Bias documentation for ADR-020: the paraphrase lean on q01, q06 (persistently KH-dominant) and q04 (T-leaning against oracle) must be recorded in the shipped artifact.

### ADR candidates
- ADR-020 completion: model = HHEM-2.1-Open, aggregation = V0 max-over-chunks, paraphrase bias quantified (G1=0.060, systematic KH lean 6/7 held-equal queries), GROUNDEDNESS_MIN pending W1b.2.

---

## Phase W1b.1 precondition — HHEM live-stack dependency resolution (investigation only, no code)

### Built
- Dependency graph audit against `pyproject.toml` and `uv.lock`. Zero API calls; all inspection local.
- Established the exact transformers 5.x failure mechanism and costed three resolution paths. Results below; no path selected (decision deferred to Ruby).

**Dependency constraint finding.** No package in the live stack hard-requires transformers ≥ 5.x. The lockfile resolves to `transformers==5.5.0` only because that is the resolver's latest compatible selection — not because any consumer sets a floor. The sole direct consumer is `sentence-transformers 5.3.0`, which declares `transformers>=4.41.0,<6.0.0`. `crewai 1.13.0`, `litellm 1.83.0`, `datasets 4.8.4`, and `instructor 1.15.1` declare no transformers dependency (they pull `tokenizers` or `huggingface-hub` directly). Pinning to `4.44.2` satisfies sentence-transformers' declared range.

**Failure mechanism (transformers 5.x).** In transformers 5.5.0, `all_tied_weights_keys` is not a class-level property of `PreTrainedModel` — it is an instance dict set by `mark_tied_weights_as_initialized()` (modeling_utils.py:1298). The loading code accesses it at lines 4526, 4615, 4634, and 4740 during the weight-loading phase. For `HHEMv2ForSequenceClassification`, this dict is never set before those accesses, raising `AttributeError`. HHEM has no tied weights, so the correct value is `{}`. The vendoring fix is one class-level line: `all_tied_weights_keys: dict = {}`.

**Upstream revision check.** `refs/main` points to `8e4a2e6e96c708cc76c2344f7e4757df2515292c` — the only snapshot Vectara has published. The remote `modeling_hhem_v2.py` has not been updated for transformers 5.x compatibility. No newer revision exists; this path does not exist.

**Three-path table.**

| Path | Mechanism | Blast radius | Latency | Determinism | ADR-020 |
|---|---|---|---|---|---|
| 1 — Pin 4.x project-wide | `transformers==4.44.2` in pyproject, lock regen | Lock regen; sentence-transformers 5.3.0 needs smoke test under 4.44.2 (may use 5.x APIs; fallback: downgrade ST to 3.x) | None — in-process | YES | Compatible |
| 2 — Vendor modeling code | Copy `modeling_hhem_v2.py` + `configuration_hhem_v2.py` into `src/evaluation/hhem/`; add `all_tied_weights_keys: dict = {}`; remove `trust_remote_code=True` | 2 files (~70 lines), 1-line fix, 1 loader change; live env stays at 5.5.0 | None — in-process | YES; stronger — project owns the code, no trust_remote_code surprise | Compatible; ownership improved |
| 3 — Subprocess isolation | Per-query subprocess or persistent sidecar using the Python 3.9.6 hhem_isolated_venv | Two managed envs, sidecar lifecycle, error handling for crash/hang | CLI is ephemeral: fresh subprocess = ~3–6s model-load cost per query; persistent sidecar avoids startup but requires daemon management (~5–50ms IPC/query) | Technically yes (same weights), but NOT in-process — crosses a process boundary | Violates ADR-020 "local and in-process, no external call on the routing path" as written |

### Why
- Path 2 has the smallest blast radius: the live env stays at 5.5.0, no lock regen, no embedding-path regression risk, and project ownership of the modeling code is strictly better than `trust_remote_code=True` against a single-revision model hub that has not been updated since 2023.
- Path 1 is viable but carries a sentence-transformers compatibility gamble: ST 5.3.0 was released in March 2026 (after transformers 5.x shipped) and may use 5.x-specific internals despite its declared `>=4.41.0` lower bound. One `from sentence_transformers import SentenceTransformer` smoke test in a downgraded env would gate this decision.
- Path 3 violates ADR-020's "in-process" requirement and is architecturally disproportionate for a CLI tool: there is no long-lived process to amortize model loading, so the per-query cost is dominated by subprocess startup and weight loading (~3–6s), not inference. It is surfaced for completeness only.

### Surprising
- No direct dependency pins transformers at all. The 5.5.0 resolution is entirely the solver's choice; a pin to 4.44.2 would resolve with zero constraint conflicts.
- The incompatibility fix in the vendored code is one line. The HHEM modeling file is ~70 lines; it is smaller and simpler than most `src/` modules in this project. Vendoring it is not a significant maintenance surface.
- The `predict()` call path (which is the only correct call path — the pipeline interface gives flat ~0.29 scores) does not touch the `all_tied_weights_keys` machinery at all. The loading-time failure blocks import but does not implicate any inference logic.

### Deferred
- Path selection: Ruby's decision. All three are documented; none has been implemented.
- W1b.1 implementation (scorer code in `src/evaluation/`, wire `scoring_engine.py`): blocked on path selection.
- W1b.2 (GROUNDEDNESS_MIN derivation): blocked on W1b.1.

---

## Phase W1b.1 — Vendor HHEM and wire into ScoringEngine

### Built
- `src/evaluation/hhem/` — vendored `modeling_hhem_v2.py` and `configuration_hhem_v2.py` pinned to hub commit `8e4a2e6e96c708cc76c2344f7e4757df2515292c`. One fix applied: `all_tied_weights_keys: dict = {}` on `HHEMv2ForSequenceClassification`. `trust_remote_code` removed. `PROVENANCE.md` records source, commit, change, and why.
- `src/evaluation/groundedness_scorer.py` — rewritten around `HHEMGroundednessScorer`. Cosine path removed. V0 aggregation preserved: per-sentence `model.predict()`, max over top-5 chunks per sentence, mean over sentences.
- `src/components/scoring_engine.py` — `ScoringEngine.__init__` added; loads `HHEMGroundednessScorer` once at construction (FAISS-index lifecycle). `score()` calls `self._gscorer.score()`.
- `tests/test_groundedness_scorer.py` and `tests/test_components_scoring_engine.py` — rewritten to mock `HHEMGroundednessScorer`; cosine-shape tests replaced with HHEM scorer shape tests.

### Why
- Path 2 (vendor) selected, as documented in the precondition investigation.
- `sentencepiece` was already in main deps from the bakeoff setup. No additional dep change needed.

### Surprising
**The `all_tied_weights_keys: dict = {}` fix is necessary but not sufficient.** Under transformers 5.x the loader replaces tensor objects rather than copying into them, which silently breaks T5's `encoder.embed_tokens → shared` weight tying after loading. Without re-tying, `model.predict()` returns flat **~0.50** on all pairs (not the 0.29 pipeline-path failure — a different, subtler fault). The fix is one line in `HHEMGroundednessScorer.__init__` after `from_pretrained`: `self._model.t5.tie_weights()`. This restores the correct scores immediately.

The session-notes precondition entry said "The `predict()` call path does not touch the `all_tied_weights_keys` machinery at all." That remains true — the `all_tied_weights_keys` fix is load-only. The tying issue is a separate mechanism: tensor-replace vs tensor-copy on load. Both fixes are needed together.

### Smoke-check scores (Step 2 gate, passed before wiring)

| Pair | Score |
|---|---|
| Grounded: "The buddy allocator manages physical memory pages in Linux." vs kernel chunk | **0.9699** |
| Grounded: "Linux uses a buddy allocator for physical memory management." vs kernel chunk | **0.9349** |
| Ungrounded: "Python is a high-level scripting language for web development." vs kernel chunk | **0.0142** |
| Ungrounded: "The Moon is made of green cheese and orbits Neptune." vs kernel chunk | **0.0015** |
| Gap | **0.9627** (threshold 0.30) — PASS |

### Architecture-honesty check
- No `final_score`, no weighted formula, no LLM routing number in changed files.
- `GROUNDEDNESS_MIN = 0.60` in `evaluator_agent.py` — **unchanged**.
- Scorer stays deterministic and Component-owned; no Agent/Component boundary move.
- Zero paid API calls; all inference local and in-process.

### Test suite
- 26 scorer + engine tests green. 494 passed, 37 skipped, 8 pre-existing failures (all pre-date this branch).

### Deferred
- W1b.2 — threshold derivation on HHEM's scale against the oracle; `GROUNDEDNESS_MIN` not yet set.
- ADR-020 completion and ADR-004 amendment — blocked on W1b.2.
- W3 re-gate (metric effect, retrieval effect) — blocked on threshold.

### ADR candidates
- ADR-020: model confirmed = HHEM-2.1-Open, aggregation = V0 max-over-chunks. One additional fix needed beyond documented: `t5.tie_weights()` post-load under transformers 5.x. Consequences paragraph should note this. GROUNDEDNESS_MIN still pending W1b.2.

---

## Phase W1b.1 post-hoc — `tie_weights` fix relocated to vendored model

### Built
- The `t5.tie_weights()` call moved from `HHEMGroundednessScorer.__init__` into a `tie_weights(**kwargs)` override on `HHEMv2ForSequenceClassification` in the vendored `modeling_hhem_v2.py`. The override calls `super().tie_weights(**kwargs)` then `self.t5.tie_weights()`.
- `PROVENANCE.md` updated with Change 2, including the smoke-check results and the why.

### Why
The `__init__` call was a workaround; the right owner is the model class. Any caller constructing `HHEMv2ForSequenceClassification` directly (not via `HHEMGroundednessScorer`) would have gotten a silently broken model without the override. Making it an override ensures correctness for any future caller and is the principled location for a post-load operation that the model spec requires.

### Surprising
Nothing new: the relocation confirmed what the W1b.1 note said. Smoke-check scores unchanged (grounded ≈ 0.97, ungrounded ≈ 0.007, gap ≈ 0.963).

### Deferred
Same as W1b.1.

---

## Phase TEST-FIX — Routing/evaluation test stale fixes (ADR-017 / ADR-018)

### Built
- 7 previously-failing tests fixed across two files; no production code touched.
- `tests/integration/test_fallback_agent.py` — 6 tests updated to the ADR-018 `_build_task_description` and `_build_crew` signatures (8 and 9 parameters respectively: `query`, `trigger_reason`/`leader`, `chunks`, `trigger_category`, `groundedness_score`, `style_score`, `confidence_score`, `style_profile`). ADR-018 behavioral assertions added beyond arity: `"Failure category: {trigger_category}"` line appears when set; `"groundedness=0.450"` score line appears when scores are provided; `"Style examples from your own emails"` appears when `style_profile` has `sample_emails` populated.
- `tests/test_evaluator_agent.py` — `test_run_propagates_flags` updated. Removed `flags=["low_style", "low_groundedness"]` from `_ReviewDraft(...)` (Pydantic silently ignores extra fields; `_ReviewDraft` has no `flags` field). Assertion changed to `["low_groundedness", "low_style", "low_confidence"]` — the ADR-017 RC-1 spec-correct order (groundedness first as the safety gate) with all three flags present (confidence=0.5 is below CONFIDENCE_MIN=0.80).

### Classification of the 7 failures (all TEST-STALE)
All 7 were stale assertions against production code that had correctly moved ahead per the ADRs. None were INCOMPLETE, ARCH-VIOLATION, or LOGIC-CHANGED.

| Failures | Root cause |
|---|---|
| 1–6 (FallbackAgent arity) | `_build_task_description` and `_build_crew` signatures enriched by ADR-018 (trigger_category + 3 scores + style_profile); tests still called the old shorter signature |
| 7 (test_run_propagates_flags) | Old test: wrong flag order (style first), missing third flag (confidence), and passed `flags=` kwarg to `_ReviewDraft` which has no such field — all three errors in one assertion |

### Exit check (confirmed terminal output)
- 7 previously-failing tests: all pass.
- Full suite: 501 passed, 1 failed, 37 skipped. The 1 remaining failure is `tests/test_query_loader.py::test_load_queries_canonical_file` — pre-existing missing data file (W4, out of scope).
- `git diff HEAD -- src/ | wc -c` = 0. `GROUNDEDNESS_MIN` unchanged.

### Why
The tests were the obligation — they guard the groundedness sentinel signal and the routing path. Without green coverage asserting ADR-017 and ADR-018 behavior, the threshold derivation in W1b.2 would be running on an untested routing path.

### Surprising
- `_ReviewDraft` silently drops `flags=[...]` kwargs because Pydantic ignores extra fields by default. The test appeared to be asserting flag behavior but was asserting nothing about flags — the `flags=` argument was never stored anywhere.
- The three bugs in `test_run_propagates_flags` (order, missing flag, invalid kwarg) were independent; any one of them would have produced a different wrong result.

### Deferred
- W1b.2 threshold derivation: unblocked by this phase.

---

## Phase W1b.2 — GROUNDEDNESS_MIN derivation on HHEM's scale (DERIVE AND SURFACE ONLY)

### Built
- `scripts/w1b2_threshold_derivation.py` — threshold analysis script. Zero paid API calls; all local.
- `results/w1b2_threshold_day14.json` — full sweep tables, per-leader bias, OOD individual scores.
- OOD HHEM scores computed fresh (never stored individually in prior artifacts); in-domain scores reused from `bakeoff_hhem_isolated_day14.json`.

### Method (pre-registered)
- **Positive class (fallback):** oracle_gf < 0.50 in-domain + all OOD records.
- **Negative class (deliver-worthy):** oracle_gf ≥ 0.50 in-domain.
- **Score:** HHEM V0 aggregation (same path as the live gate).
- **Split:** held-equal 7 queries (train); non-held-equal 7 (held-out). OOD included in both.
- **Selection:** highest T with fallback-recall ≥ 0.90 and GDR maximized (train-derived).

### Dataset
- 23 oracle-grounded (deliver), 5 oracle-ungrounded in-domain (fallback), 12 OOD (fallback). Total 40 data points.
- In-domain fallback scores: q05 T (0.251), q05 KH (0.276), q07 KH (0.252), q09 T (0.378), q09 KH (0.343).
- OOD scores: range 0.021–0.571, mean 0.139. One outlier: q20 KH (0.571) — the sole OOD response that scores above 0.40 and would escape the proposed threshold.

### Threshold sweep (compact)

**Train (12 deliver, 14 fallback):**

| T | Fallback-Recall | GDR | J |
|---|---|---|---|
| 0.28 | 0.929 | 1.000 | 0.929 |
| 0.35 | 0.929 | 1.000 | 0.929 |
| 0.40 | 0.929 | 1.000 | 0.929 |
| **0.4368** | **0.929** | **1.000** | **0.929** | ← safety-constrained + Youden both |
| 0.45 | 0.929 | 0.833 | 0.762 |
| 0.60 | 1.000 | 0.583 | 0.583 | ← current GROUNDEDNESS_MIN |

The J=0.929 plateau spans T∈[0.276, 0.4368]. The safety-constrained and Youden points are identical in performance; the selection rule (highest T maximizing GDR subject to safety) pins the operating point at T=0.4368.

**Held-out (11 deliver, 15 fallback):**

| T | Fallback-Recall | GDR | J |
|---|---|---|---|
| 0.27 | 0.800 | 1.000 | 0.800 | ← Youden (below safety constraint) |
| 0.35 | 0.867 | 0.909 | 0.776 |
| **0.3848** | **0.933** | **0.818** | **0.751** | ← safety-constrained |
| 0.40 | 0.933 | 0.727 | 0.661 |
| 0.60 | 1.000 | 0.364 | 0.364 |

### Train-held-out divergence
Train safety threshold 0.4368, held-out 0.3848 — gap of 0.052. Expected at N=14. Stability interval: [0.38, 0.44] satisfies the safety constraint on both splits. The held-out GDR cost (0.818 at T=0.385, 0.727 at T=0.40–0.44) comes from three Torvalds responses on harder queries scoring below the proposed threshold — the paraphrase bias operating on non-held-equal queries.

Held-out responses that would be mis-routed at T=0.40 (oracle-grounded, HHEM too low):
- q14 T: oracle_gf=0.734, HHEM=0.369
- q07 T: oracle_gf=0.536, HHEM=0.285
- q03 T: oracle_gf=0.642, HHEM=0.385

All three are Torvalds responses; all three reflect the paraphrase bias. W3c addresses them.

### Per-leader deliver-rate gap on held-equal oracle-grounded queries

| T | Torvalds | KH | Gap |
|---|---|---|---|
| 0.40 | 6/6 = 1.000 | 6/6 = 1.000 | **0.000** |
| 0.4368 | 6/6 = 1.000 | 6/6 = 1.000 | **0.000** |
| 0.44 | 4/6 = 0.667 | 6/6 = 1.000 | **−0.333** |

The per-leader deliver-rate gap is zero at T≤0.4368. The 0.06 mean absolute score gap (ADR-021 G1 raw number) does not translate to a deliver-rate gap at the proposed threshold. The gap first activates above T=0.4368 (Torvalds' minimum held-equal oracle-grounded score). The proposed threshold sits at the exact boundary.

### Current GROUNDEDNESS_MIN=0.60 assessment
T=0.60 was calibrated for the cosine scorer, which inflated scores via lexical echo. At T=0.60 on train: GDR=0.583 — only 7 of 12 oracle-grounded train responses correctly delivered (~42% routed incorrectly to fallback). The value cannot carry over to HHEM.

### Proposed operating point
T=0.43 — a round number just below the train-derived T=0.4368, preserving train GDR=1.000 (q12 T score 0.4368 > 0.43, correctly delivered). Satisfies safety on both splits. Per-leader bias zero at T=0.43 on held-equal oracle-grounded queries. Held-out GDR=0.727 (q07 T, q14 T, q03 T scored below threshold due to paraphrase bias — W3c addresses). The range [0.38, 0.43] is defensible; uncertainty is ±0.05 given N=14.

**GROUNDEDNESS_MIN not written. Awaiting Ruby confirmation of operating point.**

### Why
The train J=0.929 plateau being flat across a wide range ([0.276, 0.4368]) means the data cannot distinguish between 0.28 and 0.44 on performance; the selection rule breaks the tie. The held-out validates the train-derived point as reasonable (safety constraint holds, GDR cost is the known bias, not a metric failure).

### Surprising
- The safety-constrained and Youden thresholds converge to the same J=0.929 on train. The plateau is unusually flat because GDR=1.000 and fallback-recall=0.929 both hold across a 0.16-wide threshold range.
- The per-leader deliver-rate gap is exactly zero at the proposed threshold. The 0.06 raw score gap from the bake-off translates to zero operational impact at T≤0.4368 — the bias only activates above the Torvalds minimum score.
- One OOD response (q20 KH: HHEM=0.571) scores well above the OOD cluster (mean 0.139) and above the proposed threshold — it would be delivered rather than routed to fallback. This is the sole safety miss at T=0.43.

### Deferred
- `GROUNDEDNESS_MIN` code write: blocked on Ruby confirmation of T=0.43.
- ADR-020 completion and ADR-004 amendment: blocked on code write.
- W3 re-gate (metric effect, retrieval effect, per-leader floor): blocked on threshold.
- W3c per-leader floor: blocked on W3a/W3b; will address the q07 T, q14 T, q03 T mis-routing.
- q20 KH OOD outlier: investigate whether the response scored high due to topical overlap between the OOD query and the retrieved chunks (retrieval artifact) vs genuine model failure.
