# Day 15 Plan: The Wrap — Archive, Canonical Run, Charts + Diagrams, README

**Planning model:** Opus (this document; plan only)
**Executing model:** Sonnet (per-phase, plan-from-disk)
**ADR authorship:** none expected this day (metric/threshold/floor/routing are LOCKED); anything that would touch an ADR is flagged and STOPped, never written by the executing model
**System under test:** the fixed Day-14 system — HHEM-2.1-Open groundedness in the ScoringEngine Component, `GROUNDEDNESS_MIN = 0.40`, dedup-before-rerank live in the Retriever, deterministic router (ADR-018). Suite green at Day-14 close (532 passed, 0 failed).
**Query set:** `data/eval/queries.json` (the canonical set; the `queries_v1.json` default was removed in W4b)
**Precondition for the day:** branch base is `main`, working tree clean.
**Deviation policy:** if code or data contradicts a stated premise, STOP and report. Do not improvise past a gate. Surface, do not silently choose.

---

## What this day is, and is not

Day 15 is the Wrap that Day 14 displaced: present a fixed system, do not fix it. The defect investigation is closed (ADR-019 confound, ADR-020 HHEM replacement, ADR-021 Door C bias disposition, ADR-015 floors confirmed, ADR-002 dedup amendment). **The metric, the `0.40` threshold, the per-leader floors, and the deterministic routing are LOCKED. This plan does not reopen any of them.** If a phase appears to require reopening one, that is a STOP-and-surface to Ruby, not a license to re-decide.

Four decisions are already locked from a Socratic gate and are encoded below, not re-derived. If the executing model disagrees with any, it STOPs and surfaces to Ruby rather than overriding.

1. **Archive axis — three buckets.** Every file in `results/` sorts into *current-truth-for-a-question* / *audit-trail-evidence* / *worthless*. The fresh canonical run (P2) does **not** supersede the frozen-input isolated-effect measurements `w3a_metric_effect_day14.json` and `w3b_retrieval_effect_day14.json`: those ran the scorer(s) on **frozen** `(response, chunks)` to isolate one variable, so they answer a *different question* than a live run and sit **beside** it, not under it. Both keep current-truth status. The manifest names each kept file's question and states the beside-not-over relationship. **Coherence note to carry forward to P4:** the README quotes the **fresh P2 distribution**; the ADRs keep W3a's numbers; the divergence between them is expected temp=0.3 variance, not a contradiction.

2. **The run is a distribution, not a point estimate.** CloneAgent at temp=0.3 makes a single pass a noisy sample. P2 runs multiple passes and reports spread, not one number. The fresh numbers **will not** match W3a's frozen re-score, by design. The run is cost-gated: a call-count estimate is pasted before any spend.

3. **Diagrams follow ADR-014, not the PRD.** Redraw all six diagrams A1–A6 (the PRD §7.5.3 *file list* is the inventory of what to draw). The PRD §7.5.3 *prose* is stale: it describes the pre-Day-14 "4 Agents + 3 Components (incl. GatekeeperAgent)" split. ADR-014's current inventory wins — **3 Agents + 4 Components + 1 Flow, with Gatekeeper as a deterministic Component**. The PRD contradiction is flagged into the deferred PRD reconciliation; the executing model does not follow the PRD prose and does not edit the PRD.

4. **Scope fence.** Loom recording and recruiter-pitch are OUT (post-June-13). The Day-14 deferred open findings stay OUT unless Ruby explicitly opens them: corpus-level index duplication (rebuild/re-embed), the OOD-defense gap at q20 (query-relevance signal), the q07 T paraphrase misroute (accepted limitation), the full PRD reconciliation pass, and the prompt-vs-constant drift audit. These are *reported where they surface*, never *fixed* in this plan.

---

## Dependency chain

This is a sequential presentation pipeline, not the six-parallel-workstream shape of Day 14. One thing must be true before the next can start.

```
P1  Archive results/ ──STOP (manifest + per-file disposition; moves need a one-line basis, deletes need a grep)─┐
                                                                                                                │
P1b Codebase audit + docs/codebase-audit.md template ── INDEPENDENT (does not gate P2, P2 does not gate it) ─── │ ─▶ feeds P4 close
    (parallel from t0; destructive deletes stop-gated, grep-before-delete)                                      │
                                                                                                                ▼
P2  Canonical multi-pass run ──COST GATE (call-count estimate before spend; >100 calls → explicit approval)─────┐
    (in-domain 14×2 + OOD×2, multi-pass; writes results/evaluation_day15.json as a per-(query,leader) dist.)    │
                                                                                                                ▼
                              P3a Charts (§2.10/§7.6) — run-derived charts regenerate from the P2 file ──┐
                              P3b Diagrams (A1–A6) — ADR-014 inventory; INDEPENDENT of the run,          │
                                  may begin as early as P1 (light, like Day-14 W5)                       │
                                                                                                         ▼
                                                              P4 README — results above the fold from the
                                                                  FRESH P2 distribution; A1 hero from P3b;
                                                                  charts from P3a ──STOP for Ruby review
```

- **P1 and P1b are fully specified below** — their inputs (every file now in `results/`; the codebase) exist today.
- **P1b is independent of the run** — it audits the shipped codebase, so it neither gates nor is gated by P2 and may run from t0 alongside P1. Its findings inform the P4 close but do not block the run.
- **P2, P3a, P3b, P4 are gated stubs.** Their inputs (the run's numbers, the distribution's shape, the chart images) do not exist until the prior phase lands, so each stub specifies only its **input, gate, exit check, and ADR/PRD touch-points** — not file-by-file work that cannot be planned before the numbers exist.
- **P3b (diagrams) is independent of the run.** It reflects architecture, not measured numbers, so it may run in parallel from P1. It is grouped with P3a only because both feed P4. This is an honest parallelism note, not a re-sequencing.

---

## P1 — Archive `results/` (fully specified)

**Objective.** Sort every file in `results/` into the three locked buckets, write a manifest that makes the sort auditable and preserves the file→ADR-claim mapping, then physically separate audit-trail and worthless files from current-truth files without breaking any ADR's evidence trail.

**Gate before it.** None — opening step, inputs exist now. First action is read-only: enumerate `results/` (top level and `results/charts/`) and re-read this section. The move/delete step waits on the manifest being surfaced and approved.

**Load-bearing finding that shapes this phase (confirmed by Opus before planning).** The ADRs cite the *numbers*, not the file paths — grep for `results/...json` across `docs/adr/` returns nothing; ADR-020/021/015 cite evidence in prose ("the bake-off," "Probe A," "the W3a isolated metric effect," "q07 T W3a 0.285"). So **moving a results file breaks no markdown link, but losing or mis-bucketing one orphans the evidence behind a shipped ADR claim.** The manifest is the only thing that preserves traceability. This is why the manifest, not the directory layout, is the deliverable.

**The ADR→results-file citation map (the load-bearing set — none of these may be deleted; archive-with-pointer at most).**

| File | Answers the question | Cited by | Bucket |
|------|----------------------|----------|--------|
| `bakeoff_w1b0_day14.json` | Which entailment model best matches the oracle (DeBERTa, MiniCheck)? | ADR-020 G1–G4 table; ADR-021 alternatives | audit-trail evidence |
| `bakeoff_hhem_isolated_day14.json` | HHEM's bake-off scores + the in-domain scores W1b.2/W3a reuse | ADR-020 HHEM column; W1b.2 input | current-truth (HHEM bake-off) |
| `bakeoff_hhem_probe_day14.json` | Do aggregation variants fix the bias (Probe A)? | ADR-020 Probe A paragraph; ADR-021 Door A | audit-trail evidence |
| `w1b2_threshold_day14.json` | Where does `GROUNDEDNESS_MIN` sit on HHEM's scale? | ADR-020 threshold derivation; ADR-021 regime-dependence | current-truth (threshold) |
| `w3a_metric_effect_day14.json` | Metric effect alone, on frozen inputs (the W3c floor input) | ADR-015 amendment (64.3% / 78.6%); ADR-021 | **current-truth — beside, not under, the P2 run** |
| `w3b_retrieval_effect_day14.json` | Retrieval (dedup) effect alone, metric held fixed | ADR-002 amendment; ADR-015 amendment (q07 T 0.335) | **current-truth — beside, not under, the P2 run** |
| `evaluation_day12_reeval2.json` | The 84 frozen in-domain records W3a re-scored | the frozen source the isolated re-scores derive from | audit-trail evidence (do not delete; W3a is meaningless without its inputs) |

**What Sonnet does.**
1. **Enumerate and propose a bucket per file** (read-only). For every file in `results/` (including the older `evaluation_day12.json`, `evaluation_day12_reeval.json`, and `evaluation_20260523_121048.json`, which are candidates for audit-trail or worthless), state the bucket and a one-line basis. The seven files in the citation map above are pre-classified; the executing model confirms each against disk and surfaces any mismatch rather than re-bucketing silently. `results/charts/` PNGs are handled in P3a, not moved here — note them and leave them.
2. **Write `results/MANIFEST.md`.** For each *current-truth* file: the question it answers, and — for `w3a_*` and `w3b_*` — the explicit **beside-not-over** sentence (these isolate one variable on frozen inputs; the P2 live run answers a different question and does not supersede them). Carry the **coherence note** verbatim into the manifest: README will quote the fresh P2 distribution, the ADRs keep W3a's numbers, the gap is temp=0.3 variance. For each audit-trail file: the ADR/phase whose evidence it is. For each worthless file: why it carries no current question and no live citation.
3. **Separate the buckets.** Create `results/archive/` and **move** (not delete) audit-trail and worthless files into it; keep current-truth files at the `results/` top level next to the forthcoming `evaluation_day15.json`. Deletion of any file requires a STOP and a pasted zero-citation check (grep the file's basename and its distinctive numbers across `docs/`); default to archive-not-delete. Nothing in the load-bearing set is deleted.

**Stop gate P1.** Deliver the per-file bucket table with bases + the drafted `MANIFEST.md`. **STOP.** Ruby approves the buckets and the moves before any file is moved or deleted. This gate exists because the moves touch shipped-ADR evidence; the manifest is the artifact under review, not just the directory.

**Exit check.** `MANIFEST.md` exists and accounts for every file (top-level + `archive/`); every current-truth file names its question; `w3a_*`/`w3b_*` carry the beside-not-over sentence and the coherence note is recorded; no load-bearing file deleted; `git status` shows moves (renames), not content rewrites, for archived files; suite still green (archiving results JSON touches no importer — confirm none of the moved files is read by `src/` or `tests/` with a grep before the move).

---

## P1b — Codebase audit + reusable template (fully specified, parallel to P1)

**Objective.** Run the PRD §12.5 v2-architecture verification audit on the shipped codebase, capture the findings in the Day-15 session notes, and write `docs/codebase-audit.md` as the durable reusable template the §12.5 deliverable calls for. This is the §12.5 audit the Day-14 Wrap displaced (Ruby's resolution of decision 5, taken 2026-06-08: add as a phase, not defer).

**Gate before it.** None — its input is the codebase, which exists now. **Independent of the run: P2 does not gate P1b, and P1b does not gate P2.** May proceed from t0, in parallel with P1 and P2. Re-read PRD §12.5 (the six audit categories and the audit-output format) and §7.5.1 (the template's reusability intent) from disk before starting.

**What Sonnet does.**
1. **Run the §12.5 checklist as written — reference it, do not restate it.** Execute the six audit categories (dead code, dead documentation references, v1 vocabulary leaks, orphaned data files, stale comments/docstrings, unused dependencies) with the exact grep/tool commands PRD §12.5 specifies. Capture raw output per category in `docs/session-notes/day15.md` in the §12.5 audit-output format (command, raw output, per-finding decision). Empty output is recorded as "no findings" with its command — clean output is the proof.
2. **Resolve only the clearly-in-scope findings.** Genuine dead code (zero-importer files, defined-but-uncalled functions) and unambiguous v1-vocabulary/stale-doc leaks are fixed or deleted here, exactly as §12.5's decision rules direct. **Refactoring is out of scope** (§12.5): if dead code is entangled, delete it, do not refactor live code.
3. **Document-and-defer everything that touches a deferred finding or a locked decision — never fix it here.** The five Day-14 deferred findings stay deferred (decision 4): corpus index duplication, the OOD-defense gap (q20), the q07 paraphrase misroute, the full PRD reconciliation, and **the prompt-vs-constant drift audit (deferred finding #5) — the audit names it as a finding and routes it to Ruby, it does not resolve it.** Anything implicating a locked decision (HHEM, 0.40, the floors, deterministic routing) is surfaced to Ruby, not changed.
4. **Write `docs/codebase-audit.md` to double as the reusable template** (PRD §7.5.1): the six-category checklist with the grep patterns parameterized for per-project vocabulary, so P7–P9 and the P1–P5 re-verification copy it and swap the patterns. The Day-15 run's findings live in the session notes; the template is vocabulary-agnostic.

**Stop gate (destructive deletes only).** Any delete is stop-gated and grep-before-delete, like Day-14 W4a: paste a per-file zero-importer grep before removing it; suite green after each. Non-destructive fixes (doc/comment edits) and the template write proceed without a gate.

**Exit check.** `docs/codebase-audit.md` exists and is vocabulary-parameterized (reusable, not P6-hardcoded); `docs/session-notes/day15.md` carries all six categories in the §12.5 output format with raw command output and per-finding decisions ("no findings" recorded where clean); in-scope dead code/vocabulary leaks resolved with deletes grep-gated; every deferred-finding and locked-decision touch documented-and-deferred, none fixed; suite green.

**ADR / PRD touch-points.** PRD §12.5 (the audit), §7.5.1 (the template). No ADR. The prompt-vs-constant drift audit surfaces here as a named finding routed to Ruby (deferred finding #5), not resolved.

---

## P2 — Canonical multi-pass run (gated stub)

**Objective.** Produce the single canonical evaluation artifact the charts and README are built from: a multi-pass run of the fixed system, reported as a per-leader **distribution**.

**Input.** The fixed system (HHEM@0.40, dedup live, deterministic router) and `data/eval/queries.json`. Scope is the full set: 14 in-domain queries × 2 leaders **plus** the OOD records × 2 leaders, over the multi-pass count (the Day-14 close named "14×2×3"). **Read the OOD count from `queries.json` — the `expected_behavior == "fallback"` records — do not hardcode it.** Writes `results/evaluation_day15.json`.

**Gate before it.** P1 merged (clean `results/` with the manifest, so the new file lands beside current-truth and not among archived artifacts). P1b does **not** gate P2. **Output path:** the run must pass `results/evaluation_day15.json` as an **explicit** argument — the harness default is still `results/evaluation_day12.json` (`src/eval/harness.py:183`) until P1b repoints it, and since P1b does not gate P2, a default run would write the day12 path back to the cleaned top level and collide by name with the archived file. **COST GATE:** before any spend, **compute the call-count estimate fresh** from the full in-domain-plus-OOD scope — pipeline-runs × passes × completions-per-run, the last measured against the live system, not anchored to a prior day's figure. This run is well over the 100-call cost guard, so it requires explicit cost-spend approval. No run starts until the freshly-computed estimate is surfaced and approved.

**Exit check.** The output path `results/evaluation_day15.json` was passed **explicitly**, not inherited from the harness default (which would have written `results/evaluation_day12.json`). `results/evaluation_day15.json` carries per-(query, leader) **multi-pass** results (not a single pass). The in-domain deliver rate is reported **as a distribution** (mean + spread per leader) against PRD §2.1 E2 (≥55%) and E1 (≥39%) **and** the ADR-015 per-leader floors (Torvalds 42.9%, KH 35.7%) — report all bars, replace none. OOD fallback rate reported against the §2.1 100% bar. The fresh numbers are expected to differ from W3a's frozen re-score; that divergence is logged **as variance, not regression** (locked decision #2) and is **not** "corrected" to match W3a.

**ADR / PRD touch-points.** No ADR edits — metric, threshold, and floor are LOCKED. If the fresh distribution lands a leader **below its ADR-015 floor**, that is a STOP-and-surface (a genuine regression signal), not a tuning task. If q20 KH delivers again under OOD, that is the **known OOD-defense gap** (Day-14 deferred finding, scope-fenced out) — report it, do not fix it. PRD §2.1 bars are reported; the stale §2.4 cosine-distribution prose is flagged for the deferred PRD reconciliation, not edited.

---

## P3a — Charts (§2.10 / §7.6) (gated stub)

**Objective.** Regenerate the portfolio chart set so every chart that encodes evaluation numbers reflects the canonical P2 run, and the set matches the §7.6 inventory.

**Input.** `results/evaluation_day15.json` (P2) for the run-derived charts; the frozen StyleProfiles for the style-derived charts; the W4b-refactored `src/visualization.py` (already off `final_score`/`0.75`).

**Gate before it.** P2 delivered for the run-derived charts (#2 routing grid, #4 groundedness distribution, #5 score breakdown, #6 fallback trigger, #7 latency). The style-derived charts (#1 dual-leader radar, #3 style distribution, #8 pre/post-2018 evolution) do not depend on P2 and could regenerate earlier, but are kept in this phase for one coherent chart pass. Re-read §2.10, §7.6, and `src/visualization.py` from disk.

**Exit check.** Eight charts present per §7.6, regenerated from the P2 file where run-derived; the groundedness chart labeled for **HHEM entailment**, not cosine; no chart renders a `final_score` series or a stale `0.75`/`0.60` threshold line. **Naming/count reconciliation is in scope:** the current `results/charts/` holds seven PNGs with names that do not match §7.6 (e.g. `01-style-radar.png` vs the spec's `01-style-radar-dual-leader.png`) and is **missing the #2 routing-correctness 2×2 grid** — surface the discrepancy and align to the §7.6 names and count, or flag any §7.6 chart that cannot be produced from available data rather than silently shipping seven.

**ADR / PRD touch-points.** §2.10 (chart list), §7.6 (file names), `src/visualization.py`. No ADR. No metric relabel beyond cosine→HHEM on the groundedness chart.

---

## P3b — Diagrams A1–A6 (gated stub; independent of the run)

**Objective.** Produce the six architecture diagrams the PRD §7.5.3 file list calls for, drawn to **ADR-014's current inventory**, in `docs/architecture/` (the directory does not exist yet — confirmed).

**Input.** ADR-014 as corrected 2026-06-01, confirmed against disk: the v2 agent set is exactly **3 Agents (CloneAgent, EvaluatorAgent, FallbackAgent) + 4 Components (Retriever, StyleProfileBuilder, ScoringEngine, Gatekeeper) + 1 Flow orchestrator (DigitalCloneFlow)**. Both `GatekeeperAgent` and `PlannerAgent` are real v1 residue to guard against: GatekeeperAgent was an Agent until ADR-018 reclassified it a Component, and v1 referred to the Flow itself as the PlannerAgent (ADR-014, ADR-001). Re-read ADR-014 from disk for the canonical inventory and the file-by-file purpose of A1–A6.

**Gate before it.** None — independent of P2; may begin as early as P1. Re-read ADR-014. First action is a read-only confirm that no stale architecture file survives anywhere (the §12.1 Day-9 cleanup deleted the old v1 `docs/architecture/*.md` diagrams and `find` shows none today — re-verify): grep the repo for old-named diagram files and for `GatekeeperAgent`/`PlannerAgent` in any `.md`/`.png` companion.

**Exit check.** Six Mermaid diagrams (`A1`–`A6`) under `docs/architecture/` matching the §7.5.3 filenames; every diagram shows **3 Agents / 4 Components / 1 Flow** with Gatekeeper marked a deterministic **Component** (not `GatekeeperAgent`) and no `PlannerAgent`; A1 (system architecture, the README hero) is renderable as a PNG for P4. A grep for `GatekeeperAgent`/`PlannerAgent` across `docs/architecture/` is empty. **Retire-old guard:** if the read-only confirm found any surviving old-named or stale-inventory diagram file, retire it in this phase so a pre-Day-14 GatekeeperAgent diagram cannot leak into the README; if none survive, record that as confirmed.

**ADR / PRD touch-points.** **Decision #3 is binding here:** follow ADR-014, not the PRD §7.5.3 prose. The PRD §7.5.3 table prose ("4 Agents + 3 Components," GatekeeperAgent in the A1/A2/A3/A6 descriptions) is **stale** and contradicts the shipped architecture — **flag it into the deferred PRD reconciliation, do not follow it, do not edit the PRD in this phase.**

---

## P4 — README (gated stub)

**Objective.** Replace the `Coming Soon` stub (`README.md`, 11 lines) with the portfolio README: inverted pyramid, visual proof and the headline result above the fold.

**Input.** The **fresh P2 distribution** for the headline numbers; the A1 hero diagram (P3b) and the P3a charts for the above-the-fold visuals; the P1/P2 portfolio READMEs as the concrete structural reference (per CLAUDE.md README rules).

**Gate before it.** P2, P3a, and P3b delivered. Re-read the CLAUDE.md "README-specific" writing rules and a P1/P2 README from disk before drafting.

**Exit check.** README follows the inverted pyramid (hero diagram + key result above the fold, narrative-first results below, engineering signals on the second screen); the headline deliver rate quotes the **P2 fresh distribution with its spread** (locked decision #1 coherence note: the README does **not** quote W3a's frozen numbers, and a one-line note frames any ADR-vs-README divergence as temp=0.3 variance); the OOD-fallback=100% claim quotes the **live P2 OOD number** from the fresh distribution, not W3a's frozen 6×2 (decision 6, taken 2026-06-08); groundedness described as **HHEM entailment at the 0.40 gate**, never cosine; no v1 residue (`final_score`, `0.75`, weighted formula, "semantic overlap" cosine wording); no emoji in headers, no ToC, no placeholder links (CLAUDE.md rules). **Scope fence:** no Loom/recording embed, no recruiter-pitch section. **STOP for Ruby review** — the README is the outward-facing surface; the executing model drafts it and stops, it does not publish.

**ADR / PRD touch-points.** PRD §7.5.1 (top-level docs), CLAUDE.md README rules. No ADR.

---

## Scope decisions — taken by Ruby (2026-06-08)

Two scope decisions belonged to Ruby. Both are now taken; recorded here with their resolution rather than deleted, to keep the audit trail.

5. **The Wrap deliverables the Day-15 scope dropped — TAKEN: add as a phase.** PRD §8 and `day14-plan.md` assign the Wrap two deliverables the original four phases did not cover: the full **§12.5 codebase audit** (Day-14 W4a executed only its v1-residue retirements — `evaluator.py`, `rag_agent.py`, the prompt-string fixes — not the complete audit pass) and the reusable **`docs/codebase-audit.md` template** (PRD §7.5.1). **Resolution: added as P1b**, a fully-specified phase parallel to P1, independent of the run. The five deferred findings and any locked decision stay document-and-defer in P1b, never fixed there.

6. **OOD in the canonical run — TAKEN: include OOD.** P2 runs the full in-domain-plus-OOD scope, and the README (P4) quotes a **live** OOD-fallback=100% number from the fresh distribution rather than citing W3a's frozen 6×2. **Resolution folded into P2 Input/Gate and P4 exit.** The q20 OOD-defense gap stays a reported-not-fixed deferred finding (decision 4).

---

## ADRs this plan implies

**None expected.** Day 15 is presentation of a system whose decisions are already recorded (ADR-019/020/021/015/002 closed Day 14). The executing model writes no ADR. Two conditional flags only, both STOP-and-surface rather than write:
- If the P2 fresh distribution puts a leader **below its ADR-015 floor**, that is a regression signal to surface to Ruby — not an ADR-015 re-amendment by the executing model.
- The deferred findings (OOD gap q20, corpus index duplication, q07 misroute, full PRD reconciliation, prompt-vs-constant drift audit) remain tracked and **out of scope**; if one demands attention mid-phase, surface it for Ruby to open, do not author its ADR.

---

## Plan discipline

- **Model split.** Opus plans (this file); Sonnet executes per phase. No ADR is authored in the execution loop; the executing model flags and stops.
- **Plan-from-disk per phase.** Each phase opens by re-reading its own section here and its named source files from disk (ADR-014 for P3b, §2.10/§7.6 + `visualization.py` for P3a, the CLAUDE.md README rules + a P1/P2 README for P4). The file on disk is authoritative.
- **Stop / cost gates.** P1 archive gate (moves touch shipped-ADR evidence — manifest reviewed, deletes need a zero-citation grep); P1b delete gate (destructive deletes only, grep-before-delete like Day-14 W4a); **P2 cost-spend gate** (call-count estimate computed fresh from the run scope and pasted before approval); P4 README review gate (outward-facing surface). Anything over the 100-call cost guard stops and surfaces.
- **Locked, not reopened.** Metric (HHEM), threshold (0.40), per-leader floors (42.9% / 35.7%), and deterministic routing are LOCKED. A phase that seems to need one reopened is a STOP, not a re-decision.
- **The run is a distribution.** P2 reports spread; its divergence from W3a's frozen numbers is variance, recorded as such, never "corrected" toward W3a (decision #2).
- **Diagrams follow ADR-014, not the PRD** (decision #3); the §7.5.3 prose contradiction is flagged into the deferred PRD reconciliation, not followed and not edited.
- **Scope fence held** (decision #4): Loom and recruiter-pitch out; the five deferred findings reported-not-fixed unless Ruby opens them.
- **PRD Coverage Check.**
  - PRD §8 assigns the **Wrap** (the §12.5 codebase audit, a fresh README, the `docs/codebase-audit.md` reusable template). Day 14 displaced the Wrap to Day 15. This plan now covers all of it: the **§12.5 audit + the reusable template** (P1b), the **README** (P4), the **§2.10/§7.6 charts** (P3a), and the **§7.5.3 diagrams** (P3b). The Wrap's screen recording and Notion tracker update remain out of Day-15 scope per the scope fence (decision 4).
  - **§7.5.3 prose is stale** (4 Agents + 3 Components, GatekeeperAgent) and contradicts ADR-014; **§2.4/§2.5/§2.11** describe groundedness as informational and quote a cosine-specific distribution; both are flagged for the deferred PRD reconciliation — this plan does not edit the PRD.
- **Session notes.** Each phase appends a Built / Why / Surprising / Deferred / ADR-candidate block to `docs/session-notes/day15.md` (the file does not exist yet; the first phase creates it). The only code that changes this day is P1b's dead-code deletions (no Agent/Component *logic* change), so the architecture-honesty check is a confirm-clean: P1b's deletes are grep-gated, and P3b's diagrams and P4's README must not reintroduce the `GatekeeperAgent`/`PlannerAgent`/`final_score`/cosine vocabulary — that vocabulary check applies to the documentation surface as well as the code.

## Verb-and-count audit

| Phase | Verb | Target | Verification |
|-------|------|--------|--------------|
| P1 | archive | `results/MANIFEST.md`, `results/archive/` | every file bucketed with a one-line basis; current-truth files name their question; `w3a_*`/`w3b_*` carry the beside-not-over sentence + coherence note; no load-bearing file deleted; moves are git renames; STOP before any move/delete, deletes need a zero-citation grep |
| P1b | audit + template | `docs/codebase-audit.md`, `docs/session-notes/day15.md` | six §12.5 categories run with raw output + per-finding decision ("no findings" recorded where clean); in-scope dead code/vocab leaks resolved, deletes grep-gated; five deferred findings + locked decisions document-and-defer (incl. prompt-vs-constant drift, finding #5); template vocabulary-parameterized/reusable; suite green |
| P2 | run (cost spend) | `results/evaluation_day15.json` | call-count estimate computed fresh from full in-domain+OOD scope + pasted pre-run → approval; OOD count read from `queries.json`, not hardcoded; multi-pass per-(query,leader) distribution; in-domain deliver rate reported **as spread** vs PRD §2.1 E2/E1 **and** ADR-015 floors; live OOD fallback vs the 100% bar; divergence from W3a logged as variance; below-floor → STOP |
| P3a | regenerate charts | `results/charts/` via `src/visualization.py` | 8 charts per §7.6, run-derived ones from the P2 file; groundedness labeled HHEM not cosine; no `final_score`/`0.75`/`0.60` series; naming + missing #2 routing-grid reconciled to §7.6 or flagged |
| P3b | diagram | `docs/architecture/A1–A6` (Mermaid) | counts 3 Agents / 4 Components / 1 Flow per ADR-014; Gatekeeper marked Component; no `GatekeeperAgent`/`PlannerAgent`; A1 PNG for the README; PRD §7.5.3 prose contradiction flagged, not followed |
| P4 | write README | `README.md` | inverted pyramid, hero + key result above the fold; headline quotes the **fresh P2 distribution** (not W3a) with the variance note; HHEM@0.40 wording, no v1/cosine residue; no Loom/recruiter section; STOP for Ruby review |
