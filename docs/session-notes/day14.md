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
- Scored DeBERTa-v3-NLI and MiniCheck flan-t5-large in the main bakeoff script; HHEM scored separately in an isolated venv (see below). Results: `results/archive/bakeoff_w1b0_day14.json` (DeBERTa, MiniCheck) and `results/bakeoff_hhem_isolated_day14.json` (HHEM). Winner: none.

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
- Single HHEM scoring pass (3,455 in-domain + 520 OOD pairs, zero paid API calls); four aggregation variants applied post-hoc to the same raw scores. Results written to `results/archive/bakeoff_hhem_probe_day14.json`.
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

---

## Phase W1b.2 CLOSEOUT — GROUNDEDNESS_MIN = 0.40 confirmed and landed

### Built
- `src/agents/evaluator_agent.py` — `GROUNDEDNESS_MIN` changed from 0.60 to **0.40**. WHY comment added: explains that 0.40 is the W1b.2-derived HHEM operating point, that the cosine-era 0.60 does not transfer (HHEM scores run lower, carrying 0.60 would route ~42% of oracle-grounded content to fallback), and that per-leader floor is deferred to W3c.
- `docs/adr/ADR-004-groundedness-scoring-approach.md` — replaced wholesale with merged version. Status set to **Superseded**. Amendment block (cosine superseded by ADR-020, threshold change, residual-bias note, cosine-in-retrieval carve-out) folded directly into the body. The staging file `ADR-004-amendment-block.md` deleted by Ruby before this session; not committed.
- `docs/adr/ADR-020-replace-cosine-with-local-entailment-scorer.md` — pre-updated by Ruby: status line now records `GROUNDEDNESS_MIN = 0.40 (W1b.2, confirmed 2026-06-04)`; Date field extended to record both decision date and threshold-confirmed date; Decision paragraph updated to say threshold confirmed (not pending); Quantified Validation now carries the full W1b.2 threshold-derivation paragraph (safety-asymmetric rule, train T=0.4368, operating point T=0.40, stability interval [0.38, 0.44], per-leader gap = 0.000 at T=0.40, sole OOD miss q20 KH).
- `docs/adr/ADR-021-ship-known-biased-gate-compensate-at-floor.md` — pre-updated by Ruby: Decision now frames bias as regime-dependent ("zero per-leader deliver-rate gap at T=0.40 on held-equal queries, real misroutes on harder queries"); "small in magnitude" removed. Quantified Validation adds the threshold-relative perspective: the operational bias is zero at the confirmed operating point; the honest statement is regime-dependent, not "small."
- `docs/adr/ADR-019-groundedness-measures-lexical-echo.md` — confirmed unchanged on disk (reinforcement note already present from the bake-off phase). Not included in the commit.
- `results/w1b2_threshold_day14.json` and `scripts/w1b2_threshold_derivation.py` — added to repo.
- Commit: `08b43da` on `feat/day14-hhem-scorer`.

### Operating point
Ruby confirmed **T=0.40**, not T=0.43 as proposed. Both sit in the stability interval [0.38, 0.44]. 0.40 is slightly more conservative (further from the train ceiling of 0.4368), and is a cleaner round number on HHEM's scale.

### Architecture-honesty check (confirmed clean)
- Single definition: `GROUNDEDNESS_MIN = 0.40` in `evaluator_agent.py`. Imported (not redefined) by `gatekeeper.py`.
- No per-leader floor, no `final_score`, no weighted formula, no LLM routing number introduced.
- Two stale `"target > 0.60"` strings remain in LLM prompt text (`evaluator_agent.py:111`, `evaluation/evaluator.py:46`) — display-only, no routing effect. Noted for a future production-code pass.

### Surprising
- ADR-020 and ADR-021 were already pre-updated on disk by Ruby before the session; the code write was the only production change. ADR-004-amendment-block.md (the staging file from the prior phase) had been deleted.
- ADR-004 was also absent from disk — Ruby deleted both ADR-004 files (the original and the amendment-block staging file) and wanted a single merged file written fresh.

### Deferred
- W3 re-gate (W3a metric effect, W3b retrieval effect, W3c per-leader floor): next phase, unblocked.
- W3c per-leader floor: addresses q07 T (HHEM=0.285), q14 T (HHEM=0.369), q03 T (HHEM=0.385) — three oracle-grounded Torvalds responses mis-routed at T=0.40 due to paraphrase bias.
- q20 KH OOD outlier (HHEM=0.571, sole safety miss): investigate before W3b.
- ADR-004 amendment sync to Notion (the Notion version was pushed to Notion earlier in the session from the pre-merged disk state; Notion ADR-004 page needs updating to the merged/Superseded version).
- Stale prompt strings (`"target > 0.60"` in LLM prompts): independent cleanup, not a routing issue.

---

## Phase TEST-FIX-2 — Gatekeeper routing test stale fixes

### Built
- `tests/integration/test_gatekeeper.py` — 4 failing tests fixed + 2 vacuously-passing stale tests updated. No production code touched.

**4 failing tests (hardcoded cosine-era scores 0.55/0.50 now above the 0.40 floor):**

| Test | Old value | New value |
|---|---|---|
| `test_compute_flags_low_groundedness` | `0.55` | `GROUNDEDNESS_MIN - 0.10` |
| `test_compute_flags_multiple` | gs=`0.50` | `GROUNDEDNESS_MIN - 0.10` |
| `test_run_falls_back_when_groundedness_below_floor` | `0.55`, `"0.55" in reason` | `gs = GROUNDEDNESS_MIN - 0.10`, `f"{gs:.2f}" in reason` |
| `test_run_fallback_low_groundedness_with_low_confidence` | `0.55` | `GROUNDEDNESS_MIN - 0.10` |

**2 stale passing tests also updated:**

| Test | Change |
|---|---|
| `test_compute_flags_at_floor_is_clear` | score `0.60` → `GROUNDEDNESS_MIN`; comment updated |
| `test_run_delivers_when_groundedness_at_floor` | score `0.60` → `GROUNDEDNESS_MIN`; comment added |

`test_run_fallback_low_groundedness_quality_flags_no_blocking` (score 0.55) was passing vacuously (as a deliver path) — also updated to `GROUNDEDNESS_MIN - 0.10` so it tests the fallback path it was designed for.

**Import added:** `from src.agents.evaluator_agent import GROUNDEDNESS_MIN` — test values derived from the constant, not bare literals, so they cannot silently flip on future threshold changes.

### Classification (all TEST-STALE)
Identical in kind to the 7 TEST-STALE fixes in the prior phase: stale assertions against production code that had correctly moved ahead. Not INCOMPLETE, ARCH-VIOLATION, or LOGIC-CHANGED.

### Coverage check
- Below-floor → fallback path: all four fixed tests.
- Above-floor → deliver path: `test_run_delivers_when_groundedness_at_floor` (exactly `GROUNDEDNESS_MIN`) and `test_run_delivers_when_groundedness_above_floor` (0.80). Both sides of the 0.40 floor are exercised.

### Exit check
- Suite: **501 passed, 1 failed, 37 skipped**. The 1 failure is `test_load_queries_canonical_file` — pre-existing missing data file (W4, out of scope). Zero new failures.
- `GROUNDEDNESS_MIN = 0.40` at its single definition; no stale routing literal anywhere in `src/`.

### Surprising
- `test_run_fallback_low_groundedness_quality_flags_no_blocking` was passing even with score 0.55 because with the new threshold the test was exercising the deliver path (where `low_groundedness` is also not in `quality_flags`). The test was vacuously correct for the wrong reason.

### Deferred
- W3 re-gate: unblocked.

---

## Phase W3a — Metric effect, isolated (no-spend re-gate)

### Built
- The W3a metric-effect measurement: re-thresholded the frozen stored outputs under two scorers on identical inputs (cosine at GROUNDEDNESS_MIN 0.60, HHEM at 0.40), holding the duplicate-laden retrieval fixed. No generation, no re-retrieval, no paid call (both scorers' per-record scores already existed on the frozen inputs). Results in results/w3a_metric_effect_day14.json.
- Re-scorability precondition passed for all groups: 84/84 in-domain and 12/12 OOD records carried candidate text plus exact top-5 chunks. Nothing deferred to W3b. Granularity is per-(query, leader) mean, matching W1b.2 and ADR-021 so the number feeds W3c on the same basis.

### Why
- The metric swap and the pending retrieval fix are separate causes; entangling them would feed the W3c floor a number containing a retrieval artifact it must not compensate. W3a isolates the metric by re-scoring frozen inputs, so the cosine-vs-HHEM routing-verdict diff is the metric effect alone.

### Findings (the isolated metric effect, input to W3c)
- 10 of 28 verdicts flip on the swap. 9 flip fallback to deliver (cosine was over-conservative against synthesized prose, every flip lands on an oracle-deliver-worthy record); 1 flips deliver to fallback, q09 KH, a genuinely ungrounded response (oracle 0.198) that cosine's lexical echo had passed. The metric tightened where it should and loosened where the oracle agrees.
- Deliver rate: Torvalds 28.6% to 64.3%, KH 57.1% to 78.6%. Under HHEM both leaders clear both the ADR-015 per-leader floors (Torvalds 42.9%, KH 35.7%) and the PRD 2.1 bars (E2 55%, E1 39%), on the still-buggy retrieval, before W3b.
- KH-over-Torvalds direction narrows from a perfectly systematic 14/14 under cosine to 10/14 under HHEM. The residual is the ADR-021 paraphrase lean, now at score level only; it is the bias W3c compensates.
- Regression anchors q12, q13: both fell back under cosine (3 of 4 cells), both deliver under HHEM, matching the oracle (both deliver-worthy). Anchors move the right way.

### Surprising
- Torvalds clears the 42.9% ADR-015 floor at 64.3% on the corrected metric before any retrieval fix. The floor this whole investigation was triggered by is cleared once the broken metric is replaced. The original "Torvalds groundedness deficit" was a measurement artifact, not a generation deficit. This confirms the Day-13 verdict at the routing level.
- The OOD bar broke at one cell (q20 KH), see the flagged finding below.

### Deferred / flagged finding — OOD defense gap (NOT a retrieval-workstream item)
- q20 KH (an off-topic microcontroller-selection query) delivered under HHEM at 0.571, breaching the PRD 2.1 OOD-fallback=100% bar. The category-5 read on the delivered text: NOT a fabrication. It invents no false facts and faithfully paraphrases the retrieved chunks, which happened to contain topically-adjacent in-corpus sensor/microcontroller prose. Benign in content.
- The mechanism is the important part. Groundedness measures response-versus-chunks support; it never sees the query. On an OOD query where retrieval surfaces topically-adjacent chunks, a faithful paraphrase of those wrong chunks scores HIGH, so the groundedness gate certifies it. Two failures compound: retrieval surfaces plausible chunks, and groundedness then certifies faithfulness-to-the-wrong-chunks. Groundedness alone cannot catch this class of OOD by construction.
- This is an architectural limit of using groundedness as the sole OOD defense, and it predates HHEM. Cosine scored 12/12 OOD-fallback here by accident (its lexical-echo bias happened to push q20 KH below 0.60); HHEM scores the faithful paraphrase honestly at 0.571 and thereby exposes the gap cosine was masking. HHEM is not worse here; it is more honest.
- NOT dedup-fixable: q20 is not in the dedup set (q03/q07/q09/q10/q11/q14), and even perfect dedup leaves the compound failure intact, since one plausible on-topic-looking chunk is enough. So this is explicitly not a W3b retrieval item.
- Candidate remedy (surface, not decided): a query-relevance signal at the gate, distinct from groundedness, so the router can catch "answer is faithful to chunks that should not have been retrieved for this query." Disposition deferred to Ruby. Not a today ship-blocker (benign content), but a real OOD-containment gap to track.

### ADR candidate
- OOD defense gap: groundedness cannot catch topically-adjacent OOD; consider a query-relevance gate signal separate from groundedness. Flagged, not written; disposition is Ruby's.

### Scope
- No generation, no re-retrieval, no W2 fix, no floor change, GROUNDEDNESS_MIN untouched at 0.40, no ADR edits.

---

## Phase W2 — Retrieval dedup fix (fix-point A, dedup before rerank)

### Built
- A dedup step in Retriever.run between retrieve() and rerank_with_status: removes duplicate-content candidates from the FAISS pool before Cohere rerank, keeping the highest-scored copy (FAISS returns descending order, so keep first; deterministic tie-break). ~9 lines, rerank_with_status untouched, top_n_initial/top_n_final unchanged. Two zero-network unit tests: dedup removes duplicates order-preserving keeping the higher-scored copy, and is a no-op on already-distinct input.

### Why
- 6 of 14 in-domain queries returned the same passage 2-3 times in the top-5, cutting effective context to 2-4 distinct passages and depressing groundedness on those queries. Fixed at the retrieval layer, not the generation layer; a generation-side workaround would paper over a retrieval defect.
- Fix-point A (before rerank) over B (after rerank) because HHEM V0 is max-over-chunks: B would shrink the max pool to 2-4 for affected queries, scoring them on fewer chunks than every other query, trading the context defect for an uneven-scoring defect. A backfills distinct passages from the 20-slot pool, keeping effective-k=5 across all queries, and gives Cohere a clean pool to pick the best 5 from.

### Verified (zero paid spend)
- All 6 affected queries: deduped pool fully distinct and 12+ entries, so Cohere's top-5 is 5 distinct passages by construction (no Cohere call needed to verify). 8 unaffected queries: no-op, effective-k stays 5. Suite green except the known unrelated test_load_queries_canonical_file.

### Surprising
- The duplication is pervasive, not confined to the 6 visibly-broken queries: 6 of the 8 "unaffected" queries also carry duplicate-content entries in their FAISS top-20; they just never had a duplicate win a final top-5 slot. So the retrieval-time dedup now protects every query, and the root cause is clearly index-wide.

### Deferred finding — corpus-level duplication (root cause, masked not fixed)
- The dedup bug's root cause is 857 duplicate-content entries in the persisted FAISS index (6,713 metadata entries for 5,856 unique strings), baked in at build time, not produced at query time. Fix-point A dedups at retrieval time, which fully corrects the gate behavior W3b measures, but it is a mask: the index still carries 857 redundant entries, the dedup invariant lives only in Retriever.run() so any caller reading metadata.json directly still sees duplicates, and every query searches a 6,713-entry index that should be 5,856. The duplication is pervasive (reaches 6 of the 8 "unaffected" queries' candidate pools too). The proper fix is a deduped index rebuild and re-embed, deferred out of the Day-14 sprint as heavier and not required for the gate. The build producing redundant entries (likely overlapping chunk windows or re-indexed documents) is plausibly a cross-project P5/RAG-pipeline data-quality issue worth checking there too. Tracked, not scoped today.

### ADR candidate
- ADR-002 amendment note (top-20 to up to 20 distinct): authored and landed this phase.
- Corpus-level dedup rebuild: deferred; possible cross-project P5 data-quality issue.

### Scope
- Retriever.run plus one test file only. rerank_with_status untouched. No metric/floor/ADR-body change beyond the ADR-002 amendment note. Zero paid calls.

---

## Phase W3b — Retrieval effect, isolated (cost spend, HHEM@0.40 fixed)

### Built
- W3b runner `scripts/w3b_retrieval_effect.py`: 6 dedup-affected queries (q03, q07, q09, q10, q11, q14) × 2 leaders × 3 passes through the full pipeline with the W2 dedup fix live. HHEM@0.40 held fixed throughout (metric unchanged from W3a). Actual spend: 18 pipeline runs, ~252 completions (within the pre-registered ~500 envelope). Results in `results/w3b_retrieval_effect_day14.json` (incremental, complete).
- Fixed a process-order crash (exit 139, SIGSEGV) caused by FAISS's BLAS initializing before PyTorch/flan-t5-base in the same process: pre-load HHEM before importing the harness (which loads FAISS), and monkey-patch ScoringEngine to reuse the shared instance across all 18 runs. No src/ changes; the fix is in the runner script only.

### Why
- W3a measured the metric effect on frozen retrieval (old chunks, old responses). W3b measures the marginal retrieval effect: what does deduped retrieval contribute to grounding, holding the metric fixed? The two contributions need to be separately attributable so W3c's floor is set against the W3a metric effect only, not the W3b-entangled number.

### Dedup fix confirmed active
- All 6 queries: 5/5 distinct chunks in pass 1. W2 is engaged throughout.

### Per-query results (3-pass spread vs W3a frozen baseline, HHEM@0.40)

| qid | leader | W3a HHEM | P1 | P2 | P3 | spread | Δ mean | > noise? | W3b majority |
|-----|--------|----------|----|----|-----|--------|--------|----------|--------------|
| q03 | torvalds | 0.3848 | 0.4384 | 0.4577 | 0.4423 | 0.019 | **+0.061** | **YES** | **deliver** |
| q03 | kroah_hartman | 0.4813 | 0.5934 | 0.4375 | 0.3840 | 0.209 | −0.010 | no | deliver |
| q07 | torvalds | 0.2852 | 0.3336 | 0.3379 | 0.3343 | 0.004 | **+0.050** | **YES** | fallback |
| q07 | kroah_hartman | 0.2524 | 0.4731 | 0.4243 | 0.5087 | 0.084 | **+0.216** | **YES** | **deliver** |
| q09 | torvalds | 0.3778 | 0.5375 | 0.5635 | 0.5700 | 0.033 | **+0.179** | **YES** | **deliver** |
| q09 | kroah_hartman | 0.3426 | 0.5811 | 0.4875 | 0.5189 | 0.094 | **+0.187** | **YES** | **deliver** |
| q10 | torvalds | 0.5528 | 0.5597 | 0.4357 | 0.5346 | 0.124 | −0.043 | no | deliver |
| q10 | kroah_hartman | 0.6004 | 0.6694 | 0.5596 | 0.6559 | 0.110 | +0.028 | no | deliver |
| q11 | torvalds | 0.5994 | 0.5868 | 0.5017 | 0.5957 | 0.094 | −0.038 | no | deliver |
| q11 | kroah_hartman | 0.7245 | 0.5997 | 0.4806 | 0.6553 | 0.175 | −0.146 | no | deliver |
| q14 | torvalds | 0.3687 | 0.5060 | 0.3677 | 0.4704 | 0.138 | +0.079 | no | deliver |
| q14 | kroah_hartman | 0.4414 | 0.3375 | 0.4772 | 0.3663 | 0.140 | −0.048 | no | fallback |

### Marginal retrieval delta

Retrieval fix demonstrably moved grounding above generation noise (delta > spread): q03 T, q07 KH, q09 T, q09 KH — four reliable verdict flips fallback→deliver. q07 T: delta exceeds noise (+0.050 > spread 0.004) but stays below 0.40; the fix brought better chunks but the query is inherently low-HHEM for Torvalds' generation style.

Within generation noise (pre-registered: "did not move grounding measurably above noise"): q03 KH, q10 T/KH, q11 T/KH — no retrieval effect attributable. Per the mechanism caveat: HHEM V0 is max-over-chunks; if the best-supporting chunk was already in the pre-dedup top-2, restoring effective-k to 5 does not change the max. q14 T and q14 KH flip majority verdict but the delta is within spread — uncertain, not attributed to the retrieval fix.

### Mechanism caveat (pre-registered)
HHEM V0 aggregation (max-over-chunks, mean-over-sentences) means restoring effective-k from 2-4 to 5 only raises the score when a better-supporting chunk enters the pool. For q10 and q11 (already high-HHEM, best chunk already in top-2), dedup made no measurable difference. For q09 and q07 KH, a materially better-fitting chunk entered the deduped pool and changed the score dramatically.

### Combined corrected picture (W3a metric effect + W3b retrieval delta)

Reliable-signal combined (only >-noise flips applied on top of W3a):
- Torvalds: W3a 9/14 + q03 T (+1) + q09 T (+1) = **11/14 = 78.6%**
- KH: W3a 11/14 + q07 KH (+1) + q09 KH (+1) = **13/14 = 92.9%**

Best-estimate combined (majority verdicts for dedup queries):
- Torvalds: 12/14 = 85.7% (q14 T flips in, q07 T stays out)
- KH: 12/14 = 85.7% (q14 KH flips out)

Both leaders clear PRD §2.1 E2 ≥ 55% and E1 ≥ 39% on either counting method.

### Surprising
- q09 is the largest signal: both leaders jumped from ~0.38 (fallback) to ~0.54 (deliver) with tight spreads. The dedup fix clearly surfaced a better-fitting chunk for this query. q07 KH also: +0.22, all 3 passes above threshold.
- q11 KH: largest negative spread (0.175), suggesting high generation variance for this query regardless of retrieval. The dedup fix neither helped nor hurt; q11 KH stays deliver across all 3 passes.
- The process-order crash (HHEM + FAISS same process) means the live pipeline has never scored responses with HHEM and fresh retrieval in the same process before this run. All prior HHEM scores (W3a bakeoff, threshold derivation) were on frozen data. This run is the first live HHEM-scored pipeline run.

### Deferred
- q14 verdict uncertainty: delta within noise for both leaders; q14 T and q14 KH flips are not reliably attributed to the retrieval fix. Not actionable until W3c floor sets the operating point.
- q07 T persistent fallback: dedup improved grounding above noise but not above 0.40. Whether a per-leader floor or a harder floor adjustment is needed is W3c's decision, not W3b's.

### ADR candidate
- None from W3b alone. The mechanism caveat (max-over-chunks, effective-k only helps when better chunk enters pool) is already captured in the W3a framing and the ADR-020 V0 aggregation note.

### Scope
- scripts/w3b_retrieval_effect.py (new runner) and results/w3b_retrieval_effect_day14.json (new results file). No src/ changes. GROUNDEDNESS_MIN unchanged. No ADR edits. No floor touched.

---

## Phase W3b — Retrieval effect, isolated (cost spend, 3-pass)

### Built
- The W3b retrieval-effect measurement: regenerated the 6 dedup-affected queries (q03, q07, q09, q10, q11, q14) x 2 leaders x 3 passes through the full pipeline with the W2-fixed (deduped) retriever, metric held fixed at HHEM 0.40, scored against the W3a baseline. Results in results/w3b_retrieval_effect_day14.json. Dedup confirmed live: all 6 queries returned 5/5 distinct chunks.
- Spend came in under envelope: ~252 completions against the approved ~500 ceiling (the realized completions-per-pair ran lighter than the estimate). 18 pipeline runs, 18 Cohere calls.

### Why
- W3b isolates the retrieval contribution: only retrieval changed vs W3a (now deduped), metric held at HHEM 0.40. The delta vs W3a is the retrieval effect on these 6 queries. 3 passes because CloneAgent runs at temp 0.3 and generation noise must be averaged out for the retrieval signal to show through.

### Findings (read against generation noise, pre-registered)
- 5 of 12 cells show a retrieval effect above the 3-pass spread: q03 T (+0.061, flips to deliver), q07 T (+0.050, stays fallback — mean 0.335 still below 0.40), q07 KH (+0.216, flips to deliver), q09 T (+0.179, flips to deliver), q09 KH (+0.187, flips to deliver).
- 5 cells: delta within generation noise, no reliable retrieval effect (q03 KH, q10 T, q10 KH, q11 T, q11 KH). Pre-registered interpretation applies: the dedup did not move grounding measurably above noise for these.
- 2 cells uncertain (q14 T, q14 KH): majority verdict flips but the spread (0.14) exceeds the delta, and one pass straddles the threshold each way. Generation variance, not retrieval, drives these flips. Not attributable to the fix.

### Surprising / mechanism confirmed
- The effect tracks the max-over-chunks mechanism exactly. HHEM V0 is max-over-chunks; restoring effective-k from 2-4 to 5 only raises the score if a better-supporting chunk enters the pool. The queries that improved (q07 KH, q09 both) are where dedup brought in a materially better-fitting chunk; the queries that did not (q10, q11) are where the best chunk was already in the pre-dedup top-2. The data confirmed the pre-registered mechanism caveat rather than contradicting it. This is a stronger result than a flat "dedup helped" because it explains when and why.
- q07 T improved reliably yet stayed in fallback (mean 0.335 < 0.40). It is one of the three Torvalds hard-query misroutes ADR-021 named as the regime-dependent bias: dedup helped it but did not rescue it, consistent with the bias being real on hard queries independent of retrieval.

### How to carry these numbers (caveat)
- The W3a and W3b numbers are NOT on the same footing. W3a is a re-threshold of frozen Day-12 responses (point value per cell, no generation noise). W3b is fresh 3-pass generation (a distribution per cell). So any "combined corrected deliver rate" is a hybrid of point estimates (the 8 non-dedup queries) and noisy 3-pass majorities (the 6 dedup queries). Read it conservatively. The reliable-signal version (counting only above-noise flips) is Torvalds 11/14, KH 13/14; the majority-verdict "best estimate" (T 12/14, KH 12/14) is a noisier upper read, not a stable system property. Both clear PRD 2.1 (E2 55%, E1 39%) either way.

### Does NOT feed W3c
- W3b is portfolio evidence that the dedup fix works; it is NOT an input to the W3c floor. The floor compensates the intrinsic metric paraphrase bias, measured on the held-equal set, which is disjoint from these 6 dedup queries. The W3c floor input is the W3a isolated metric effect only. This disjointness is by design and is the wall that keeps the retrieval fix from contaminating the floor decision.

### Scope
- Exactly the 6 dedup queries, 3 passes, metric held fixed at HHEM 0.40. No floor change, no metric change, no other queries, no OOD, no ADR edits. Cost spend under envelope.

---

## Phase W3c — Per-leader floor disposition (a)+(c): floors confirmed not moved, misroute documented

### Decided (pre-registered before reading rates)
- The per-leader floors are aggregate regression guardrails, confirmed on the corrected metric and NOT moved. No leader-specific threshold, no GROUNDEDNESS_MIN change. The paraphrase bias is handled as a documented accepted limitation (ADR-021 Door C), not by any floor or threshold adjustment. The rule was locked before any rate was read.

### Why
- The ADR-021 bias is per-query misrouting (a grounded Torvalds answer on a hard query scoring below the shared gate). A per-leader aggregate rate floor cannot fix a per-query misroute; it cannot tell a paraphrase-penalized grounded answer from a weak one. Lowering Torvalds' threshold would weaken the shared OOD defense (same GROUNDEDNESS_MIN, OOD AUC 0.942) and is the leader-keyed calibration ADR-021 Door A rejected. So the floor's honest role is regression guardrail, not compensation.

### Rates vs floors (from W3a isolated metric effect; W3b NOT used as input)
- Torvalds 9/14 = 64.3% vs 42.9% floor (+21pp) and PRD E1 39%: clears both.
- KH 11/14 = 78.6% vs 35.7% floor (+43pp) and PRD E2 55%: clears both.
- Both clear both bars on the corrected metric, frozen inputs. Floors hold, not moved. This is the complete W3c rate read.

### Accepted limitation (documented, not engineered around)
- On hard paraphrased queries a grounded Torvalds response can score below 0.40 and fall back. Surviving exemplar after the W2 dedup fix: q07 T (oracle 53.6%, W3a 0.285, W3b mean 0.335 — improved reliably by retrieval, still below the gate). Broader class: q14 T (oracle 73.4%, W3a 0.369), q03 T (oracle 64.2%, W3a 0.385; resolved by dedup). Not a generation defect (ADR-019), not a retrieval defect (W3b improved q07 T and it still fell back), not a floor-calibration defect. It is the operational cost of choosing a local deterministic gate over a paraphrase-robust LLM judge. Chose to document it rather than soften a safety gate to hide it.

### Regime-shape question closed
- A flat per-leader offset is the wrong shape: it is aggregate, would pass weak answers just under the floor, and still would not target the per-query misroutes. (a)+(c) declines to use the floor as compensation at all. Open question from the plan is resolved.

### Surfaced for ADR
- ADR-015 amendment (floors confirmed on HHEM, reframed as guardrails, misroute documented): authored, landed this phase. Numbers unchanged.

### Scope
- No threshold change, no leader-specific threshold, no routing code change, no floor number change. W3b not used as a rate input. Zero paid calls.

---

## Phase W4b-3 — Repoint cli index from RAGAgent to Retriever.build()

### Built
- `src/cli.py` — `from src.agents.rag_agent import RAGAgent` replaced with `from src.components.retriever import Retriever`. Index command body changed from `agent = RAGAgent(config=config); agent.build(chunks)` to `Retriever(config=config).build(chunks)`.
- `tests/test_cli.py` — `TestIndexCommand.test_success` updated. `patch("src.cli.RAGAgent", ...)` replaced with `patch("src.cli.Retriever", ...)`. Added sentinel-chunks pattern: `sentinel_chunks = [MagicMock()]` returned by `chunk_documents` mock; asserted `mock_retriever.build.assert_called_once_with(sentinel_chunks)` so the test verifies data actually flows through, not just that a name is patchable.

### Why
- ADR-014 claimed "v1 rag_agent.py façade becomes the Retriever Component" but the cli index command was never migrated. `RAGAgent.build()` and `Retriever.build()` are structurally identical (same primitives, same `_DEFAULT_INDEX_DIR = Path("data/rag/faiss_index")`), so the repoint is safe. Without this, rag_agent.py would remain a live importer and could not be deleted in W4c.
- The stop-gate investigation (W4c precondition) confirmed the miss: ADR-014's claim was half accurate — the retrieve-side was absorbed into Retriever, but the cli build-side was never migrated.

### Surprising
- Nothing new surfaced. The precondition investigation had already established that ADR-014 was half accurate, so the fix was mechanical once the decision to proceed was confirmed.

### Deferred
- W4c (delete rag_agent.py): now unblocked — zero importers confirmed post-edit.

---

## Phase W4c — Delete dead v1 rag_agent.py

### Built
- `src/agents/rag_agent.py` — deleted. Pre-delete grep (`grep -rn "rag_agent|RAGAgent" src/ tests/`) returned only the file itself; zero external importers. `src/agents/__init__.py` had no re-export. Post-delete grep returned empty.

### Why
- With W4b-3 complete, rag_agent.py was a zero-importer file with no re-export. The v2 Retriever Component owns both the build and retrieve paths. Keeping a dead v1 facade in the agents package creates misleading inventory and contradicts ADR-014's stated clean-sweep outcome.

### Surprising
- `src/agents/__init__.py` was empty — rag_agent.py had never been re-exported, so the deletion needed no __init__ cleanup. The test mock in test_cli.py was the only caller ever, and W4b-3 had already repointed it.

### Deferred
- None. W4d (stale prompt strings in evaluator_agent.py) next.

---

## Phase W4d — Fix stale groundedness-target string in evaluator prompt

### Built
- `src/agents/evaluator_agent.py` line 111 — `_build_task_description` prompt string changed:

  **Before:** `f"  Groundedness: {scores.groundedness_score:.3f} (target > 0.60)\n"`

  **After:** `f"  Groundedness: {scores.groundedness_score:.3f} (HHEM entailment; well-grounded in the retrieved context)\n"`

### Why
- `0.60` was the cosine-era groundedness threshold. It is meaningless on HHEM's entailment scale and contradicts the live `GROUNDEDNESS_MIN = 0.40` constant in the same file. This is an LLM-consumed prompt string, not a code-read gate; hardcoding `0.40` would be equally stale on the next threshold revision. Qualitative phrasing names the metric and the quality intent without a number that can go out of date.
- Routing is controlled by the deterministic `GROUNDEDNESS_MIN = 0.40` in `_compute_flags()`. The prompt string change has no routing effect.

### Surprising
- The investigation surfaced a second stale string: line 110 `(target > 0.90)` for style, but `STYLE_MIN = 0.70` (ADR-017 Amendment 1 corrected it from the synthetic-data calibration target). Surfaced as an observation; out of scope for this phase, addressed in W4d-continued.
- Lines 36/40/41 carry `0.60` in ADR-context comments explaining the historical cosine threshold — not prompt instructions, not actionable stale strings.

### Deferred
- Line 110 style string and line 112 confidence string: out of scope, addressed in W4d-continued.

---

## Phase W4d-continued — Make all three score-target annotations qualitative

### Built
- `src/agents/evaluator_agent.py` lines 110 and 112 — `_build_task_description` prompt:

  **Before:**
  ```
  f"  Style:        {scores.style_score:.3f} (target > 0.90)\n"
  f"  Groundedness: {scores.groundedness_score:.3f} (HHEM entailment; well-grounded in the retrieved context)\n"
  f"  Confidence:   {scores.confidence_score:.3f} (target > 0.80)\n\n"
  ```
  **After:**
  ```
  f"  Style:        {scores.style_score:.3f} (stylistic match to the leader's voice)\n"
  f"  Groundedness: {scores.groundedness_score:.3f} (HHEM entailment; well-grounded in the retrieved context)\n"
  f"  Confidence:   {scores.confidence_score:.3f} (model's expressed certainty in the response)\n\n"
  ```

### Why
- Line 110 `(target > 0.90)` was stale: `STYLE_MIN = 0.70` per ADR-017 Amendment 1 (corrected from the synthetic-data calibration target 0.90). Any style score in [0.70, 0.89] is correctly delivered but the prompt characterized it as below-target, potentially biasing the LLM reviewer's explanation.
- Line 112 `(target > 0.80)` was numerically accurate (`CONFIDENCE_MIN = 0.80`) but inconsistent with the other two now-qualitative lines; a future threshold change would re-introduce the staleness problem.
- Result: all three lines qualitative, no hardcoded thresholds in LLM-consumed prompt text. The constants (`STYLE_MIN`, `GROUNDEDNESS_MIN`, `CONFIDENCE_MIN`) remain the single source of truth for routing and flag logic.

### Surprising
- Nothing new. The fix was straightforward once the live constants were confirmed.

### Deferred
- W5: architecture diagram (Mermaid, `docs/architecture/`).
- W6: §2.10 results-chart regeneration (gated on W3+W4b).

---

## Day 14 — CLOSED (2026-06-04 to 2026-06-06)

Day 14 is complete. Three calendar days, one defect chased to ground and the system rebuilt around what it exposed.

**What it was:** a reported Torvalds groundedness deficit (28.6% vs a 42.9% floor) turned out to be a measurement artifact. The cosine metric measured lexical echo, not containment, and punished the clone that paraphrased. Fixing the metric (HHEM entailment at GROUNDEDNESS_MIN 0.40), not the clone, lifted Torvalds to 64.3%.

**Delivered:** metric replacement (ADR-019/020, HHEM vendored), threshold derivation (0.40, safety-asymmetric), bias disposition (ADR-021 Door C, regime-dependent), retrieval dedup (W2, ADR-002 amended), re-gate (W3a/W3b/W3c), v1 residue cleanup (W4a-d: evaluator.py and rag_agent.py retired, cli/visualization/streamlit refactored, prompt strings fixed), ADR reconciliation (8 ADRs on disk + Notion), 5 phase journals + 1 capstone. Suite green: 532 passed, 0 failed.

**Carried to Day 15 (deferred findings, see handover):** corpus-level index duplication (rebuild needed), OOD-defense gap (q20, query-relevance signal candidate), per-query paraphrase misroute (q07, accepted limitation), Wrap-time PRD reconciliation, prompt-vs-constant drift audit.

**Day 15 is presentation, not fix:** archive results/ first, then a full multi-pass run on the fixed system (14x2x3, reported as a distribution, real spend), then charts (W6) and the architecture diagram (W5) from that canonical file, then README. A single run will not reproduce W3a's numbers (temp=0.3 is stochastic); that is variance, not regression.

---
