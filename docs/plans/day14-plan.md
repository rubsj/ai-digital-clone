# Day 14 Plan: Metric Fix, Retrieval Dedup, Re-Gate, Phase 2 Remainder

**Planning model:** Opus (this document; plan only)
**Executing model:** Sonnet (per-phase, plan-from-disk)
**ADR authorship:** flagged here, written separately — never by the executing model mid-phase
**Data:** `results/evaluation_day12_reeval2.json` (84 in-domain records, ADR-018 deterministic router live)
**Query set:** `data/eval/queries.json`
**Deviation policy:** if code or data contradicts a stated premise, STOP and report. Do not improvise past a gate. Surface, do not silently choose.

---

## Verdict this plan builds on

Day 13 closed with a verdict, confirmed three ways (Probe A verbosity, Probe B containment-held-equal, Probe C per-sentence mechanism): **the groundedness scorer — per-sentence max cosine against the top-5 chunks, then mean — measures lexical echo, not containment.** Torvalds is not less grounded; he is less source-aligned. The scorer rewards sentences that restate chunk vocabulary and penalizes synthesis in the author's own voice, even where Opus markup labels both as grounded. The metric is the defect. Routing is honest (ADR-018 deterministic router applies the floor correctly). **No clone fix is warranted.**

Two consequences propagate into this plan:
- The H2/H3 fork is dead. There is no containment deficit to excise.
- The 42.9% Torvalds floor (ADR-015) was calibrated against a broken metric. It cannot be trusted until re-measured, and it must not be moved to rescue a delivery rate.

---

## Dependency map and parallelism

```
W1 Metric + threshold ── DECISION GATE (STOP) ──▶ W1b implement (scorer + GROUNDEDNESS_MIN) ──┐
                                                                                              │
W2 Retrieval dedup (parallel from t0) ──────────────────────────────────┐                    │
                                                                         │                    ▼
                              W3a metric effect — re-score FROZEN outputs, NO generation ◀─────┘
                                  (in-domain 84 + OOD 6×2; q12/q13 anchors; 2 scorers cosine@0.60 vs HHEM@0.40 → verdict diff)
                                                         │
                                                         ▼
                              W3b retrieval effect — REGENERATE the 6 dedup queries (cost spend)
                                  (needs W2; isolates the retrieval fix as a marginal delta vs W3a)
                                                         │
                                                         ▼
                              W3c floor recalibration (gated; must NOT move to rescue a rate)
W4 Phase 2 remainder:
     W4a dead-file retirement + ADR-014 Notion sync  ── parallel from t0 (independent)
     W4b cli/visualization refactor + cli evaluate→harness ── AFTER W1 decision (groundedness display)
     W4c rag_agent.py retirement ── AFTER W4b (cli.py is its last live importer)
W5 Architecture diagram (docs/architecture/, §7.5.3) ── parallel from t0 (light, independent)
W6 §2.10 results-chart regeneration ── AFTER W3 (new numbers) AND W4b (visualization refactor)
```

- **Opening step is W1's DECISION GATE.** It is a decision, not an implementation, and it now also fixes the **routing threshold** (does 0.60 transfer to the new metric, or is it re-derived). Everything that re-measures (W3) waits on it.
- **The re-gate is split to isolate one variable at a time.** W3a re-scores the **frozen** `(response, chunks)` under two scorers on identical inputs — the old cosine@0.60 and the new HHEM@0.40 — and reads the metric effect off the routing-verdict diff; no generation, no re-retrieval. W3b regenerates only the 6 dedup-affected queries with W2 live — isolating the retrieval effect as a marginal delta. These are separate phases because W3a is a no-spend measurement and W3b is a cost spend.
- **W2 runs in parallel** with the W1 decision wait, but **must land before W3b** (not W3a — W3a uses the stored old chunks by design).
- **OOD is inside the re-gate, not excluded.** A metric *replacement* (unlike Day-12's threshold change) can move OOD scores; PRD §2.1 makes OOD-fallback=100% and zero category-5 hallucinations non-negotiable. W3a re-scores the 6 OOD records too.
- **W4a, W5 are independent** and may run from t0. **W4b is gated on the W1 decision** (not its implementation) so the groundedness display is touched once. **W4c is gated on W4b. W6 is gated on W3 + W4b.**
- These are six workstreams. Do not collapse them into one block. Each clears its own gate.

---

## W1 — Metric fix: decision first

**Objective.** Choose the replacement for the lexically-confounded groundedness metric **and decide the routing threshold on the new metric's scale.** This workstream's opening and only mandatory step is a **DECISION**, not code. Lay out the options with tradeoffs, then STOP for Ruby to choose. Implementation (W1b) is a separate phase that does not begin until the decision is recorded and the ADRs are flagged.

**Gate before it.** None — this is the opening step of Day 14. Re-read this section, ADR-004, and `src/evaluation/groundedness_scorer.py` + `src/components/scoring_engine.py` from disk before assembling the options.

**What Sonnet does (decision phase).** Produce a one-screen options memo. For each option, state **cost** (API/latency/dependency), **determinism**, **defensibility** (does it measure containment rather than echo, and can it be defended in a portfolio writeup), and **location** (see below). Do not recommend; present. Candidate options to cost out — Sonnet may add others it finds, but must not silently drop any:

1. **Entailment / NLI scorer.** Local cross-encoder NLI (claim entailed by chunk?). Cost: model dependency (~the BERTScore concern ADR-004 weighed), no per-call LLM spend. Determinism: deterministic given a pinned model. Defensibility: measures entailment, directly addresses the confound; needs its own calibration sample on LKML vocabulary.
2. **LLM-faithfulness judge, RAGAS-style.** Per-claim faithfulness against retrieved context. Ties to P2 (RAG evaluation framework) — reuse, not re-justify. Cost: per-call LLM. Determinism: needs temp=0 + MD5 cache to be reproducible. Defensibility: industry-standard. **Note the ADR-004 nuance:** ADR-004 rejected an LLM judge for *inference-time* latency (>3s SLA, PRD §2.7). The groundedness gate is an *offline eval* path, where that latency objection is weaker — surface this explicitly so the decision is made on the right axis.
3. **Promote the Day 13 containment instrument to scorer.** The blind Opus per-span grounded/inferable/free markup, formalized. Cost: Opus per-eval, highest. Determinism: low (LLM, span-level). Defensibility: already validated against the confound on this exact data; but expensive and Opus-dependent.
4. **Hybrid.** e.g. cosine retained as a cheap inference-time signal + entailment/LLM faithfulness as the offline gate metric; or cosine pre-filter feeding an LLM judge. State which path each metric governs.

**Metric LOCATION — a first-class decision criterion, not a W1b implementation detail.** Per option, state **where the number is produced and what architecture that implies**, because the choice can reopen the Agent/Component boundary the rework just made honest:
- Options 1 (NLI) and the deterministic side of 4 keep scoring **deterministic in the ScoringEngine Component** — no boundary change; the current `score()` path holds.
- Options 2 (LLM-faithfulness) and 3 (promoted Opus instrument) make the number **LLM-produced**, which ADR-009 bars from a Component. The score would relocate into the **EvaluatorAgent** (the Hybrid Agent, ADR-011) and ScoringEngine would no longer own groundedness. That is a real architecture cost — it re-opens the deterministic-Component boundary ADR-018 just settled — and it must be weighed at the decision, not discovered in W1b.
State this consequence per option so Ruby chooses the metric and the boundary move together. The replacement-metric ADR records where the metric lives.

**Threshold transfer — part of the same decision (do not defer).** The routing floor `GROUNDEDNESS_MIN = 0.60` (`src/agents/evaluator_agent.py`, imported by the deterministic Gatekeeper) was calibrated to cosine (ADR-004's 5-sample LLM-judge agreement level). It does not automatically carry to a new metric's scale. The memo must state, per chosen metric: **does 0.60 transfer as-is, map to an equivalent point, or must it be re-derived** (e.g. a fresh small judge-agreement sample, the way ADR-004 derived 0.60). The threshold is a deterministic gate (ADR-018), so a number on the wrong scale silently breaks routing. This is not the per-leader floor (that is W3c) — it is the per-response groundedness gate.

**Stop gate W1 (DECISION GATE).** Deliver the options memo covering metric **and** threshold. **STOP.** Ruby chooses the metric, the threshold basis, and confirms which ADRs to write. Do not implement, do not write the ADRs, do not pick a default. This is the hardest gate in the plan; the rest of the metric track is downstream of Ruby's choice.

**W1b — implement (separate phase, post-decision, post-ADR).** Re-read the chosen-metric ADR from disk first. Implement the chosen scorer in `src/evaluation/` and wire through `src/components/scoring_engine.py` (the EvaluatorAgent's numerical path; per ADR-007 no LLM number inside a Component — if the chosen metric is an LLM judge, confirm where it is allowed to live before writing). **Set `GROUNDEDNESS_MIN` to the W1-chosen value in `src/agents/evaluator_agent.py`** so the deterministic Gatekeeper routes on the new scale; this must be in place before any W3 run. Architecture-honesty guard applies: no `final_score`, no weighted formula, no LLM-judged routing number reintroduced.

**Exit check (W1b).** New scorer unit-tested against its calibration sample; `GROUNDEDNESS_MIN` set to the chosen value with the basis cited in the ADR-004 amendment; the confound case from Day 13 (q02 fully-grounded both clones, scorer gapped +0.065) no longer separates the two clones under the new metric — or, if it still does, the residual is explained. Architecture honesty grep clean.

---

## W2 — Retrieval dedup bug

**Objective.** Fix the duplicate-chunk defect: 6 of 14 in-domain queries (q03, q07, q09, q10, q11, q14) retrieved the same passage 2–3× in the top-5, cutting effective context to 3–4 distinct passages. Leader-agnostic (chunks are shared per record, byte-identity confirmed Day 13). It does not explain the per-leader scorer gap, but it depresses absolute grounding on ~43% of in-domain queries. Bounded fix in the retrieval path.

**Gate before it.** None — independent of the metric decision, runs in parallel from t0. **But it must land before W3.** First step is read-only confirmation (below); the code change waits on that confirmation being surfaced.

**What Sonnet does.**
1. **Confirm the dedup point in code before specifying the fix** (read-only). The current path is `Retriever.run()` (`src/components/retriever.py:84–117`): embed → FAISS `top_n_initial` → `rerank_with_status(...)` → returns `reranked` (top-5). Determine where duplicates enter:
   - duplicate `KnowledgeChunk`s present at index `build()` time (corpus-level), or
   - FAISS returning the same passage at multiple ranks, or
   - duplicates surviving the Cohere rerank into the top-5.
   The fix point differs by cause: dedup before rerank lets rerank backfill the top-5 with distinct passages (preserves 5 effective chunks); dedup after rerank shrinks the top-5. Surface the confirmed cause and both fix-point options with their effect on effective-k. **Do not pick silently** — if the choice changes retrieval semantics enough to touch ADR-002, flag it.
2. Implement the bounded fix at the confirmed point with a deterministic tie-break (e.g. keep highest-ranked instance). No change to `rerank_with_status` (Retriever depends on it; ADR-018 Phase-2 note).

**Exit check.** The six named queries return 5 distinct passages in the top-5 (or the corrected effective-k is reported per query); a regression check that the other 8 in-domain queries are unchanged; suite green. Result feeds W3.

---

## W3 — Re-gate: measure first, one variable at a time

The metric, its threshold, and the bias framing are now **settled inputs, not open questions.** ADR-020 fixes the scorer (HHEM-2.1-Open, V0 max-over-chunks aggregation, local and in-process in the ScoringEngine Component) and the routing threshold (`GROUNDEDNESS_MIN = 0.40`, derived at W1b.2, already landed in `src/agents/evaluator_agent.py`). ADR-021 fixes the framing of the residual paraphrase bias: leader-blind in mechanism, **regime-dependent** in size (zero per-leader deliver-rate gap in HHEM's polarized zone, real misroutes on harder queries), shipped under Door C and compensated at the per-leader floor (W3c), not at the metric. W3 does not re-decide any of this; it measures the consequence and sets the floor.

Groundedness is a pure function of stored `(response, chunks)`. That fact sets the design: anything whose chunks did **not** change can be re-scored from disk with no regeneration; only the dedup-affected queries get new chunks and therefore new responses. The re-gate is split so the metric effect (W3a) and the retrieval effect (W3b) are separately attributable, and so the no-spend measurement clears before the cost spend.

**Measurement precedes recalibration throughout. The floor is not touched until W3c, and never to rescue a rate.**

### W3a — Metric effect, isolated (no generation, no re-retrieval)

**Objective.** Quantify what the **metric swap alone** does to routing, holding retrieval fixed. Re-score the **frozen stored outputs** from the prior runs — each response plus its exact top-5 retrieved chunks, **including the current duplicate-chunk retrieval** — under **two scorers on identical inputs**: the old cosine at its `GROUNDEDNESS_MIN = 0.60`, and the new HHEM at `GROUNDEDNESS_MIN = 0.40`. Apply the deterministic router under each, and read the metric effect off the **difference in routing verdicts** between the two. No generation, no re-retrieval.

**Precondition (read-only, before any scoring) — confirm the frozen inputs are re-scorable.** W3a must first confirm that the stored artifacts carry, per record, the **response text and its exact top-5 chunks** in a form complete enough to re-score without re-running any part of the pipeline. If they do not — **STOP and surface.** Two notes on where this is most likely to bite:
- The in-domain records are `results/evaluation_day12_reeval2.json` (84 records, both leaders).
- OOD is the exposed case: groundedness scores the clone **candidate** response (the text the router scored), not the delivered fallback text, and every OOD record fell back. If only the fallback text / routing reasoning was persisted for an OOD record and not the candidate response + its chunks, that record **cannot** be re-scored from disk — do not substitute fallback text for the candidate. Surface it and route it to W3b's regeneration rather than faking a frozen re-score.

**Gate before it.** W1b landed (HHEM scorer wired, `GROUNDEDNESS_MIN = 0.40` live — both already done). Re-read this section, `src/eval/harness.py`, the cosine and HHEM scorer paths, and the deterministic router from disk.

**No paid API spend — and possibly no fresh scoring.** Both scorers are local (cosine over a local-encoder embedding path; HHEM in-process via `model.predict()`), so nothing calls a paid endpoint. Moreover the per-record scores may already exist: cosine scores are stored in `evaluation_day12_reeval2.json`, and HHEM in-domain scores were computed at the W1b.0 bake-off (`results/bakeoff_hhem_isolated_day14.json`). The precondition step confirms whether W3a reduces to **re-applying the two thresholds (0.60, 0.40) to already-stored scores on identical frozen inputs** — the cleanest form — or whether any record must be (re-)scored locally. Either way: no paid call, paste the compute note, the 100-call cost guard is not tripped.

**What Sonnet does.**
1. Re-score (or, where the scores already exist on identical frozen inputs, re-threshold) the 84 in-domain frozen records, and the OOD records **only for those confirmed re-scorable by the precondition** (the rest move to W3b), under **both** scorers — cosine@0.60 and HHEM@0.40 — applying the deterministic router in code to each. Write to a new results file; do not overwrite the Day-12 artifacts.
2. **In-domain:** report, per leader, the 2×2 deliver/fallback grid under each scorer, and the **routing-verdict diff** (which records flip deliver↔fallback when cosine@0.60 is swapped for HHEM@0.40) — that diff is the isolated metric effect. Report the Torvalds–KH picture under HHEM versus cosine and whether the old cosine direction (KH > T) persists / narrows / reverses. Report the in-domain deliver rate against **both** the PRD §2.1 headline bars (E2 ≥55%, E1 ≥39%) **and** the ADR-015 per-leader floors (Torvalds 42.9%, KH 35.7%) — §2.1 is the ship criterion, the per-leader floors its amended refinement; report both, do not replace one with the other.
3. **Regression anchors:** report q12 and q13 explicitly (Day-8 in-domain anchors; both fell back under cosine per Day-12) — state their score and deliver/fallback under each scorer so the anchor check is auditable, not inferred.
4. **OOD (PRD §2.1, non-negotiable):** report the OOD fallback rate under HHEM@0.40. It must be 100% (6/6 both leaders). **If any OOD record flips to deliver, run the category-5 hallucination check on the actual delivered text** (prose judgment, paste the text) — a delivered OOD answer with fabricated content is a ship-blocker, not a tuning note. (W1b.2 flagged q20 KH at HHEM 0.571 as the sole OOD score above 0.40 — confirm its verdict here.)

**Stop gate W3a.** Deliver the two-scorer in-domain grids + the routing-verdict diff (the isolated metric effect) + q12/q13 anchors + OOD fallback count and any hallucination read. STOP. This isolates the metric on frozen inputs; the retrieval effect is W3b, and the floor that compensates the metric's residual bias is W3c (which uses **this** isolated number, not the W3b-entangled one).

### W3b — Retrieval effect, isolated (cost spend)

**Objective.** Quantify the marginal effect of the dedup fix, **holding the metric fixed at HHEM `GROUNDEDNESS_MIN = 0.40`**, by re-retrieving and regenerating only the queries whose chunks actually change.

**Gate before it.** W2 merged AND W3a delivered. **Paste a call-count estimate before the run** (full pipeline regeneration): the 6 dedup-affected in-domain queries (q03, q07, q09, q10, q11, q14) × 2 leaders × 3 passes = 36 pairs — at ~14 completions/pair this is ~500 completions, well over the 100-call cost guard, so this run requires explicit cost-spend approval. Plus any OOD record the W3a precondition routed here, and any OOD query found dedup-affected in W2.

**Scope note — disjoint from the bias measurement.** The 6 dedup-affected queries (q03, q07, q09, q10, q11, q14) are **disjoint** from the 7 held-equal queries the ADR-021 paraphrase bias was measured on (q01, q02, q04, q05, q06, q12, q13) — confirmed zero-overlap in the W2 diagnostic. So the dedup fix cannot move the held-equal bias number, and W3b's retrieval effect does not contaminate the W3a metric effect that W3c's floor is set from.

**What Sonnet does.** Apply the W2 dedup fix and re-retrieve for those 6 queries, regenerate them through the full pipeline, and re-score with HHEM@0.40 (metric held fixed), into the W3a results file (or a sibling). Report the per-query delta vs W3a for the same queries, so the retrieval contribution is separated from the metric contribution. Note that CloneAgent temp=0.3 adds regeneration variance (Day-12 saw Torvalds 3–4/14); report the 3-pass spread, do not over-read a single pass.

**Stop gate W3b.** Deliver the marginal-delta table (W3b vs W3a on the 6 queries) and the combined corrected grid. STOP. Floor still untouched.

### W3c — Per-leader floor: compensate the ADR-021 bias (separately gated, must not rescue a rate)

**Objective.** Set the per-leader floor that **compensates the ADR-021 intrinsic paraphrase bias** — the leader-blind, regime-dependent KH lean HHEM carries on held-equal queries — and decide whether ADR-015's existing per-leader floors (Torvalds 42.9%, KH 35.7%) hold, move, or are re-derived on HHEM's scale. The floor is set from the **isolated W3a metric effect only** — the cosine@0.60 vs HHEM@0.40 verdict diff on the frozen inputs — **not** the W3b retrieval-entangled picture. The metric effect is the bias the floor must offset; the retrieval fix is a separate correction that must not be folded into the floor.

**OPENING GATE (ADR-021 pre-registration obligation).** Before any deliver rate under the candidate floor is computed or seen: the floor adjustment must be **justified by the measured bias magnitude and pre-registered** against it. A floor set to rescue a deliver rate — chosen after seeing the rate it produces — is **forbidden** by ADR-021 and is the floor-rescuing anti-pattern this project has refused throughout. State the bias magnitude (from W3a), state the floor rule it implies, lock it, *then* read the resulting rate. If the locked floor still puts Torvalds below a defensible bar — or below PRD §2.1 E1/E2 — that is the honest result, reported as such.

**Open design question (surface, do NOT resolve here).** ADR-021 establishes the bias is **regime-dependent**: zero per-leader deliver-rate gap in HHEM's polarized zone (the held-equal queries at T=0.40), but real misroutes on harder queries where Torvalds' grounded scores run low (W1b.2 named q07 T 0.285, q14 T 0.369, q03 T 0.385). So whether a **single flat per-leader floor offset** is even the right *shape* of compensation — versus something regime-aware — is itself an open question to settle at W3c. Lay it out; do not pick.

**Gate before it.** W3a and W3b delivered (the corrected measurement exists; the W3a isolated metric effect is the input to the floor).

**What Sonnet does.** Present the floor options and their basis (e.g. re-derive from HHEM's distribution the way the style floor was derived; or hold the ADR-015 numbers and document), framed against **both** PRD §2.1's E2/E1 headline bars and the ADR-015 per-leader floors they refine, with the regime-dependence shape question surfaced. Recalibration follows the measurement; it does not move to make Torvalds pass.

**Stop gate W3c.** Options surfaced, bias magnitude and any candidate floor rule pre-registered before the rate is read. STOP for Ruby. This is ADR territory (ADR-015 amendment). Do not write it.

**Exit check (W3 track).** Metric effect (cosine@0.60 vs HHEM@0.40 verdict diff on frozen inputs) and metric+dedup measurements both recorded, with q12/q13 anchors; OOD fallback rate reported (and hallucination-checked if anything flipped); the per-leader floor pre-registered against the W3a bias magnitude before its rate is seen; floor decision recorded by Ruby (not the executing model); any floor change lands as a flagged ADR-015 amendment, additive, original numbers untouched.

---

## W4 — Phase 2 remainder (carried from Day 12)

Day 12 left Phase 2 blocked behind STOP GATE 1.6b + 1.7 *and* "Torvalds floor not cleared." The cli/visualization refactor was additionally parked on collision risk with a Torvalds clone fix. **Under the Day 13 verdict there is no clone fix**, so that collision risk is gone. The remaining question is whether the metric fix collides with the refactor. **Code-confirmed assessment below.**

### W4a — dead v1 file retirement + ADR-014 Notion sync (independent, parallel from t0)

**Objective.** Retire the dead v1 files Day 12 deferred, and complete the ADR Notion sync.

**Gate before it.** None for the audit/grep step. **STOP GATE before any delete** (destructive, multi-file) — paste a per-file zero-live-importer grep first. Re-read the Day 12 Phase 2 section from disk.

**What Sonnet does (confirmed against code today).**
- `src/agents/rag_agent.py` — live importers: `src/cli.py:16,137` and `tests/test_cli.py:162`. **Not yet retirable** — depends on W4b (cli refactor) removing the import. Sequenced as W4c.
- `src/evaluation/evaluator.py` — live importers: `src/evaluation/__init__.py:4` (re-export) and `tests/test_evaluator.py`. `scoring_engine.py` imports `confidence_scorer` and `groundedness_scorer` directly, **not** `evaluator`. Retirable after removing the `__init__` re-export and its test. **No collision with W1:** deleting `evaluator.py` does not touch the live scorer the metric fix edits.
- `reranker.rerank()` — after `rag_agent.py` goes, `rerank()` is called only by experiment scripts (`scripts/test_rag_pipeline.py`, `diagnostic_6a_*`, `experiment_6b_*`). **Do not delete `reranker.py`** — it exports `rerank_with_status`, which `Retriever` uses. Audit the scripts; if any still call `rerank()`, surface it — do not force-delete.
- ADR-014 Notion sync: confirm the Day-12 ADR-014 inventory correction (4→3 agents, 3→4 components) is synced to the Notion **📐 ADR Log** (filter by Project = "P6: Digital Clone"; ADR numbers collide across projects). Sync the other pending ADRs Day 12 deferred (ADR-010 superseded-by-018, ADR-015 amendment, ADR-017, ADR-018). **Notion is an external write — confirm scope before writing.**

**Exit check.** Per-file zero-importer grep pasted before each delete; suite green after each; Notion ADR Log shows the corrected inventory and the pending ADRs. This step also feeds PRD §12.5 (the codebase audit's v1-residue verification — see PRD Coverage Check).

### W4b — cli/visualization refactor (gated on W1 decision)

**Objective.** Rewire `src/cli.py` and `src/visualization.py` off the dead v1 schema onto v2 field names.

**Unblock assessment, code-confirmed.** The v1 tokens are present and isolated: `src/cli.py` lines 149/161/168/172/173/181/183/219/220/226/227/281/292/294 (`final_output`, `final_score`, `trigger_reason`, `context_summary`, `0.75`) and `src/visualization.py` lines 5/97/130/138/154 (`final_score` series, `0.75` threshold). **None of these tokens appears in `groundedness_scorer.py` or `scoring_engine.py`** — grep-confirmed. So the v1-field-name refactor is **independent of the metric fix and unblocked from the collision side.** The collision risk Day 12 named (a Torvalds clone fix) no longer exists.

**One soft coupling, surfaced — not a block.** `visualization.py` has `plot_groundedness_distribution` and `cli.py` prints `groundedness_score`. If the W1 decision renames/replaces the groundedness field or adds a faithfulness field, the groundedness *display* would want updating. To touch that display once rather than twice, **sequence W4b after the W1 decision** (the decision, not the full W1b implementation — the field shape is known once the metric is chosen). This is a sequencing call, not a dependency that blocks the v1-token cleanup.

**Streamlit demo is in scope — confirmed by grep.** The shipped `streamlit_app.py` (repo root) renders `ev.groundedness_score` with a cosine-flavored help label ("Semantic overlap with retrieved chunks", `:83`) **and** carries live v1 residue: `ev.final_score` with a "vs threshold" delta (`:87-88`) and the literal `0.4×style + 0.4×groundedness + 0.2×confidence` formula string (`:95`), plus a `THRESHOLD` constant. It is the same v1-residue class as cli/visualization on the user-facing surface. Align its groundedness label to the W1-chosen metric and strip the dead `final_score`/formula/threshold display in the same pass — do not ship a demo showing a removed metric and a removed formula.

**Gate before it.** W1 decision recorded. Re-read the Day 12 Phase 2 section and `src/cli.py`, `src/visualization.py`, `src/flow.py`, `src/schemas.py` from disk.

**What Sonnet does.**
1. Rewire `query`/`compare` to `flow.state.styled_response`/`fallback_response` and the v2 `FallbackResponse` fields (`acknowledgment`, `suggested_redirections`, `calendar_link`, `available_slots`); drop every `final_score` print and the `0.75` docstring/threshold references; align the groundedness display with the W1-chosen metric.
2. **Wire `cli evaluate` to the Phase-1 harness (Day-12 Phase 2 item 2, dropped in the prior draft — restored here).** `evaluate` (`src/cli.py:238`) must call `src/eval/harness.py` (per-stage latency, 2×2 grid, PRD scorecard) rather than rebuild it, and the `--queries` default must change from `data/eval/queries_v1.json` (`src/cli.py:235`, docstring `:254-255`) to `data/eval/queries.json`. This also removes the pre-existing `test_load_queries_canonical_file` provenance for the v1 path.
3. **Align `streamlit_app.py` (repo root).** Strip the dead `final_score` metric, the "vs threshold" delta, the `THRESHOLD` constant, and the `0.4×style + 0.4×groundedness + 0.2×confidence` formula string; relabel the Groundedness metric's help text for the W1-chosen metric (drop the cosine "semantic overlap" wording if the metric changed). Same v2-field discipline as `query`/`compare`.
4. Re-enable the two skipped test files (`tests/test_cli.py`, `tests/test_visualization.py`) against the v2 shape.

**Exit check.** `grep -n "final_score\|final_output\|0\.75\|context_summary\|trigger_reason" src/cli.py src/visualization.py` returns only intentional matches; `grep -n "final_score\|0\.4×style\|THRESHOLD" streamlit_app.py` empty; `grep -n "queries_v1" src/cli.py` empty; `cli query`, `cli compare`, and `cli evaluate` run end-to-end (the last writes a results JSON matching the harness schema and prints the grid); `streamlit_app.py` imports/renders without referencing a removed field; both re-enabled test files collect and pass.

### W4c — rag_agent.py retirement (gated on W4b)

**Objective.** Delete `src/agents/rag_agent.py` once `cli.py` no longer imports it.

**Gate before it.** W4b merged (removes `cli.py:16,137`). **STOP GATE (destructive)** — paste zero-live-importer grep.

**What Sonnet does.** Confirm `index` and all of `src/` no longer import `RAGAgent`; update/remove `tests/test_cli.py:162` mock; delete the file.

**Exit check.** `grep -rn "rag_agent\|RAGAgent" src/ tests/` empty; suite green.

### W4d — Stale groundedness target strings in prompt text (cleanup)

**Objective.** Replace the stale `target > 0.60` strings in two LLM prompt templates — `src/agents/evaluator_agent.py:111` and `src/evaluation/evaluator.py:46` — with the HHEM-scale target (the 0.40 gate) or a qualitative intent. `0.60` is the cosine-era threshold; it is meaningless on HHEM's scale and contradicts the live `GROUNDEDNESS_MIN = 0.40` gate. These are prompt text, display-only with no routing effect, which is exactly why **no test catches them** — flagged at the W1b.2 closeout and carried here.

**Gate before it.** None for `evaluator_agent.py` (live agent prompt, must be fixed regardless). **Sequence after W4a for `evaluation/evaluator.py`:** W4a retires `evaluation/evaluator.py`, so if that file is deleted first the `:46` string disappears with it — do not fix a string in a file about to be deleted. Confirm W4a's disposition of `evaluator.py` before touching its `:46` line.

**What Sonnet does.** In `evaluator_agent.py:111`, replace `target > 0.60` with the 0.40-gate phrasing (or a qualitative "well-grounded in the retrieved context" intent that does not hard-code a scale). For `evaluation/evaluator.py:46`, fix it the same way **only if W4a did not retire the file**; otherwise note it resolved by deletion.

**Exit check.** `grep -n "0\.60\|target > 0" src/agents/evaluator_agent.py` returns no stale groundedness-target string; `evaluation/evaluator.py:46` either fixed or gone with the file; suite green (no test asserts on this prompt text, so this is a grep-and-read check, not a test pass).

---

## W5 — Architecture diagram (light, parallel)

**Objective.** A current architecture diagram: 3 Agents (CloneAgent, EvaluatorAgent, FallbackAgent) + 4 Components (Retriever, ScoringEngine, StyleProfileBuilder, **Gatekeeper-as-component**) + 1 Flow orchestrator, matching ADR-014 as corrected 2026-06-01. No such diagram exists under `docs/` today (confirmed).

**Gate before it.** None — independent, slot wherever it fits. Re-read ADR-014 from disk for the canonical inventory.

**What Sonnet does.** Produce a light diagram in `docs/architecture/` (PRD §7.5.3 is the home for architecture diagrams; Mermaid in a markdown doc is the cheap default). Show the Flow orchestration path and mark the Gatekeeper as a deterministic Component, not an Agent. **This is the §7.5.3 architecture diagram and is distinct from the §2.10 results charts in W6** — do not conflate them.

**Exit check.** Diagram committed under `docs/architecture/`; agent/component counts match ADR-014 (3/4/1); no `GatekeeperAgent` label.

---

## W6 — §2.10 results-chart regeneration (after the re-gate)

**Objective.** The §2.10 portfolio charts that encode evaluation numbers — groundedness distribution (`04-groundedness-score-distribution.png`), the per-response score breakdown, the fallback-trigger chart, and the routing 2×2 — go stale the moment the re-gate produces new numbers. Regenerate them from the corrected results. Distinct from the W5 architecture diagram (§7.5.3).

**Gate before it.** W3 delivered (new results file exists) AND W4b merged (`visualization.py` rewired off `final_score`/`0.75`, so the charts no longer plot the removed series). Re-read §2.10, the chart list, and `src/visualization.py` from disk.

**What Sonnet does.** Regenerate the affected `results/charts/` PNGs from the W3 results file using the refactored `visualization.py`. Confirm no chart still renders a `final_score` series or a `0.75` threshold line, and that the groundedness chart is labeled for the W1-chosen metric (not "cosine groundedness" if the metric changed).

**Exit check.** Charts regenerated from the corrected results; `grep`/visual confirm no `final_score` series and no stale `0.75`/`0.60` cosine threshold line; groundedness chart labeled for the new metric.

---

## ADRs this plan implies (flagged, not written)

The executing model writes **none** of these. Each is authored separately and Notion-synced.

1. **Confound-finding ADR** — records the Day 13 verdict: the groundedness scorer measures lexical echo, not containment; confirmed three ways. New ADR (Evaluation category).
2. **Replacement-metric ADR** — records the W1 decision (entailment / LLM-faithfulness / promoted-instrument / hybrid), **including where the metric's number lives** (ScoringEngine Component vs relocated into the EvaluatorAgent per ADR-009/ADR-011) and any Agent/Component boundary move that implies. New ADR; content depends on Ruby's choice.
3. **ADR-004 amendment** — marks the cosine heuristic as confounded / superseded by the replacement metric, **and records the routing threshold on the new metric's scale** (whether 0.60 transferred, mapped, or was re-derived, with its basis). The threshold belongs in this amendment, not only the confound. Additive amendment, original Decision untouched (or a supersede note, per Ruby).
4. **ADR-015 amendment (W3c)** — only if the floor moves after the corrected re-measurement. Must follow the measurement, not rescue a rate.
5. **ADR-002 amendment (W2)** — only if the dedup fix changes retrieval semantics (effective-k, tie-break) enough to amend the RAG-config decision. Surface at W2 step 1; may be a plain note rather than an amendment — Ruby decides.
6. Not new ADRs: the ADR-014 inventory correction (already an amendment, needs Notion sync) and the ADR-010→018 supersede mark.

---

## Plan discipline

- **Model split.** Opus plans (this file); Sonnet executes per phase. ADRs are authored outside the execution loop — the executing model never invents or writes an ADR mid-phase (it flags, then stops).
- **Plan-from-disk per phase.** Each workstream opens by re-reading its own section here and its named source files from disk, not from context. The file on disk is authoritative.
- **Stop gates before destructive, multi-file, or cost-spend changes.** W1 decision gate; W3a no-spend measurement gate; **W3b cost-spend gate (regeneration, ~500 completions — explicit approval)**; W3c floor gate; W4a delete gate; W4c delete gate. Each requires explicit approval; deletes require a pasted zero-importer grep first; every run that calls an LLM/embedding pastes a **call-count estimate before spending**, and anything over the 100-call cost guard stops and surfaces.
- **Surface, do not silently choose.** W1 presents metric *and threshold* options without a default. W2 surfaces the dedup-point cause and both fix points. W3 reports measurement before any floor move and isolates the metric effect (W3a) from the retrieval effect (W3b). No recommendation is substituted for Ruby's decision at a gate.
- **No unsurfaced ADRs.** All six ADR implications are listed above and flagged; none is written by the executing model.
- **PRD Coverage Check.**
  - PRD §8 assigns Day 14 the **"Wrap"**: the §12.5 codebase audit, a fresh `README.md` (P1/P2 inverted-pyramid, results above the fold), and the `docs/codebase-audit.md` reusable template. **This plan does not deliver the Wrap, and that is a flagged re-sequence, not a silent drop.** The floor investigation (Days 12–13) reframed into a metric defect; the corrected metric and re-gate (W1–W3) must land before the README can state honest results above the fold, and the audit's v1-residue check (§12.5) is partly executed by W4a's retirements. Recommendation to surface to Ruby: the Wrap moves to a Day 15, fed by W1–W4 + W6; W4a's grep-confirmed retirements double as the audit's v1-residue pass. Flagging the displacement, not deciding it.
  - **§2.10 charts** are covered by W6 (regeneration after the re-gate); **§7.5.3 architecture diagram** by W5. Day-12 Phase 2 item 2 (`cli evaluate` → harness, default off `queries_v1.json`) is **restored into W4b**, not deferred.
  - **Streamlit demo (`streamlit_app.py`) is in scope, not scoped out.** Grep confirms it reads `groundedness_score` (cosine-labeled) and still renders the removed v1 `final_score`/weighted-formula/threshold; W4b aligns its groundedness label to the W1 metric and strips the v1 residue. The CLAUDE.md "CLI commands stay identical v1→v2" rule applies to *command surface*, not to displayed metric labels — relabeling a stale metric is not a command-surface change.
  - **§2.1 headline bars reported, not replaced.** W3a (and the W3c basis) report the in-domain deliver rate against PRD §2.1 E2 (≥55%) and E1 (≥39%) **alongside** the ADR-015 per-leader floors, since §2.1 is the headline ship criterion and the per-leader floors are its amendment. OOD-fallback=100% and zero category-5 hallucinations (§2.1, non-negotiable) are reported in W3a.
  - **Flag for correction — do NOT edit the PRD in this plan.** PRD §2.4 (line 118), §2.5, and §2.11 (lines 195–196) describe groundedness as an *informational* metric, "not a pass/fail gate." ADR-018 made it a **deterministic routing gate** — the PRD prose is now stale and contradicts the shipped architecture. Separately, §2.4's "groundedness distribution mean 0.55–0.70" (line 123) is **cosine-specific** and will not describe the W1 replacement metric's scale. Both are flagged here for Ruby to correct in a PRD pass; this plan does not touch PRD text.
- **Verb-and-count audit.** Below. Every imperative has a file target and a verification step.
- **Phase Defence wired into the exit gate.** Each workstream closes with the four-category Phase Defence (Category A always; one of B/C/D; **Category V v1-drift mandatory** for W1b, W2 (touches the Retriever Component), W4b, W4c — all touch Agent/Component/scorer code). Session notes appended to `docs/session-notes/day14.md` per phase (Built / Why / Surprising / Deferred / ADR candidate). The handover answers the standing v1-drift question.

## Verb-and-count audit

| Item | Verb | File target | Verification |
|------|------|-------------|--------------|
| W1 | decide | options memo (no code) | per option: cost/determinism/defensibility/**location** (ScoringEngine Component vs EvaluatorAgent per ADR-009/011, with the boundary-move consequence); ADR-004 offline-vs-inference nuance surfaced; **threshold transfer (0.60 transfers / maps / re-derived) stated per metric**; STOP at decision gate, no default picked |
| W1b | implement | `src/evaluation/` (new scorer), `src/components/scoring_engine.py`, `src/agents/evaluator_agent.py` (`GROUNDEDNESS_MIN`) | new scorer unit-tested vs calibration sample; threshold set with ADR-cited basis; Day-13 q02 confound case no longer separates clones (or residual explained); architecture-honesty grep clean |
| W2.1 | confirm (read-only) | `src/components/retriever.py:84–117`, index build path | duplicate-entry cause identified; both fix points + effective-k effect surfaced; ADR-002 touch flagged if semantics change |
| W2.2 | implement | confirmed dedup point in `src/components/retriever.py` | 6 named queries return 5 distinct top-5 passages (or corrected effective-k reported); other 8 unchanged; suite green |
| W3a | re-score, 2 scorers (no paid spend) | cosine + HHEM scorer paths, new `results/*.json` (in-domain 84 from `reeval2`; OOD 6×2 from `evaluation_day12.json` **iff re-scorable**) | **frozen-input re-scorability precondition stated (else STOP), OOD candidate-text+chunks branch declared — insufficient → OOD moves to W3b**; **no paid spend (both scorers local, HHEM in-process), compute note pasted**; in-domain 2×2 under each scorer + **routing-verdict diff (metric effect)** reported **against PRD §2.1 E2(55%)/E1(39%) and ADR-015 per-leader floors**; **q12/q13 anchors reported**; **OOD fallback = 100% or category-5 hallucination text pasted**; floor untouched |
| W3b | regenerate (cost spend) | full pipeline on q03/q07/q09/q10/q11/q14 (+ any dedup-affected OOD), into W3a results file | call-count estimate pre-run (~500 completions → cost-spend approval); per-query marginal delta vs W3a; 3-pass spread reported; floor untouched |
| W3c | surface | ADR-015 (flagged) | floor options + basis presented; recalibration follows measurement, does not rescue a rate; STOP for Ruby |
| W4a | retire + sync | `src/evaluation/evaluator.py` (+`__init__` re-export, `tests/test_evaluator.py`); audit `reranker.rerank()`; Notion ADR Log | per-file zero-importer grep pre-delete; suite green; ADR-014 inventory + pending ADRs present in Notion |
| W4b | refactor + wire | `src/cli.py` (incl. `evaluate`→harness, default off `queries_v1.json`), `src/visualization.py`, **`streamlit_app.py`**, `tests/test_cli.py`, `tests/test_visualization.py` | v1-token grep clean; `grep queries_v1 src/cli.py` empty; `grep "final_score\|0.4×style\|THRESHOLD" streamlit_app.py` empty; `query`/`compare`/`evaluate` run end-to-end (evaluate writes harness-schema JSON + grid); streamlit renders no removed field; both test files collect and pass |
| W4c | delete | `src/agents/rag_agent.py` (+`tests/test_cli.py` mock) | `grep -rn "rag_agent\|RAGAgent" src/ tests/` empty; suite green |
| W5 | diagram | `docs/architecture/` (Mermaid, §7.5.3) | counts match ADR-014 (3 agents / 4 components / 1 flow); Gatekeeper marked Component; no `GatekeeperAgent` label |
| W6 | regenerate charts | `results/charts/` via refactored `src/visualization.py` | regenerated from W3 results; no `final_score` series / stale `0.75`/`0.60` line; groundedness chart labeled for the new metric |
