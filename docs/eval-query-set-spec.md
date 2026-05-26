# Eval Query Set v2 — Specification

**Status:** Revised (rev 1, 2026-05-25). Used to generate `data/eval/queries_v2.json` in Step B.
**Date:** 2026-05-25 (Day 8)
**Author:** Day 8 wrap session, P6 Digital Writing Clone.
**Replaces:** None. v1 (`data/eval/queries_v1.json`) retained as the historical artifact that surfaced the system-test gap. Two v1 queries reused as regression anchors (see A5).

---

## A1 — Design intent

This is a **system test**, not a benchmark. The eval set exercises both pipeline branches the orchestrator was designed to support: the **deliver path** (queries the corpus can substantively answer, expected to produce a styled response with `final_score >= 0.75`) and the **fallback path** (queries the corpus cannot answer, expected to route to `FallbackResponse` with a `trigger_reason` reflecting absence of grounded content).

Beyond pass/fail on the deliver path, the run is also checked for a specific **negative signal**: a category-5 out-of-domain query that comes back with a styled response and `final >= 0.75` is **not** a success — it is a hallucination failure. The LLM produced confident styled content from parametric knowledge while retrieval surfaced no relevant chunks, and the groundedness scorer failed to catch it. Such records will be flagged in the Step D analysis as a real finding, not absorbed into the deliver rate.

The original v1 set tested only the deliver path with topics outside corpus coverage, which made the entire run register as fallback and obscured whether the deliver path worked at all. v2 mixes both kinds of queries at a 70/30 split so a single run produces signal on both branches.

---

## A2 — Corpus topic analysis

The production index (`data/rag/faiss_index/`, 6,713 chunks at 1536-dim OpenAI embeddings) is built from `open-phi/textbooks` filtered to `field="computer_science"`. The filter returns five textbooks, not a broad CS corpus.

### Per-textbook chunk distribution

| Source textbook | Chunks | % corpus |
|---|---:|---:|
| Statistical Learning Theory and Applications | 2,073 | 30.9% |
| Introduction to Computers and Engineering Problem Solving | 1,476 | 22.0% |
| Numerical Methods Applied to Chemical Engineering | 1,437 | 21.4% |
| Principles and Practice of Assistive Technology | 864 | 12.9% |
| Data Mining | 863 | 12.9% |

Sampled 200 chunks at random — distribution matches the population within ±2.5pp.

### Topic coverage (keyword-probed, 6,713-chunk full-scan)

| Topic | Hits | Coverage assessment |
|---|---:|---|
| Regularization (ridge, lasso, kernel) | 318 | well-covered |
| Assistive technology design process | 396 | well-covered (single-book monopoly) |
| Cryptography keywords (encryption/rsa/aes) | 159 | misleading — most hits are "data encryption" references in Data Mining, not crypto theory. Treat as **sparse**, not in-domain. |
| Support Vector Machines | 158 | well-covered |
| Neural networks (basic) | 113 | well-covered |
| ODE / PDE / finite element / Runge-Kutta | 93 | well-covered |
| Recursion / sorting | 79 | covered |
| Decision trees / random forests | 69 | covered |
| Stacks / queues / linked lists | 63 | covered |
| Numerical integration (Simpson, trapezoidal) | 58 | covered |
| Object-oriented Java basics | 52 | covered |
| Sparse matrix / linear systems | 47 | covered |
| Stochastic programming / gradient descent | 45 | covered |
| Collaborative filtering | 38 | covered |
| k-Nearest Neighbors | 31 | covered |
| Forward/backward selection | 29 | covered |
| Confusion matrix / precision/recall | 26 | covered |
| Distributed systems keywords | 20 | sparse — "distributed memory" in numerical methods, not Raft/Paxos. Treat as **absent**. |
| Databases / SQL / transaction isolation | 8 | absent (incidental hits) |
| Compilers / parsers | 3 | absent |
| Git / version control | 2 | absent |
| Web / HTTP | 1 | absent |
| **Networking / TCP / routing** | **0** | **absent** |
| **Operating systems / virtual memory** | **0** | **absent** |
| **Security / buffer overflow / ASLR** | **0** | **absent** |
| Embedded / hardware (Arduino, Pi) | 0 | absent |
| Cooking / sports / non-technical | 0 | absent |

The keyword counts are scans against `c["content"].lower()` for the listed substrings — they overcount (a chunk mentioning "buffer" in the cache sense counts toward "buffer overflow") but don't undercount, so a zero is genuine. The result: classic systems-CS topics (networking, OS, security) are **structurally absent** from the corpus — exactly the topics that dominated v1.

---

## A3 — Query categories

Six categories, totaling 20 queries. 14 in-domain (70%), 6 out-of-domain (30%).

### In-domain (14 queries, target: deliver with `final >= 0.75`)

| # | Category | Count | Tests | Expected behavior |
|---|---|---:|---|---|
| 1 | Statistical learning & ML theory | 4 | Whether the deliver path produces grounded styled responses on the corpus's largest topic. Tests retrieval into the SVM / regularization / kernel-methods / NN material in *Statistical Learning Theory*. | deliver, `final >= 0.75` |
| 2 | Data mining methods | 4 | Tests retrieval into kNN, decision trees, confusion matrix, feature selection, collaborative filtering material in *Data Mining*. | deliver, `final >= 0.75` |
| 3 | Numerical methods | 3 | Tests retrieval into ODE/PDE solvers, sparse matrix, numerical integration, fixed-point iteration material in *Numerical Methods*. | deliver, `final >= 0.75` |
| 4 | Programming fundamentals & data structures | 3 | Tests retrieval into recursion, sorting, stacks/queues, binary search material in *Intro to Computers*. Holds the two v1 regression anchors. | deliver, `final >= 0.75` |

### Out-of-domain (6 queries, target: fallback with grounded trigger reason)

| # | Category | Count | Tests | Expected behavior |
|---|---|---:|---|---|
| 5 | Systems CS absent from corpus | 4 | Networking, OS internals, security, kernel internals. Confirmed 0 keyword hits in A2. Stylistically plausible for both leaders (kernel maintainers' domain) — but the corpus can't ground them. **Hallucination failure mode:** a styled response with `final >= 0.75` on these is a real finding, not a deliver. The grading rule looks at `trigger_reason` and groundedness specifically. | fallback (with `trigger_reason` indicating no grounded content); a deliver here is logged as a hallucination finding |
| 6 | Off-topic technical | 2 | Embedded hardware / microcontroller / RTOS. Stylistically plausible for kernel-adjacent folks but absent from the corpus. | fallback |

### Total: 14 + 6 = 20

The 70/30 split has a hard arithmetic constraint: hitting PRD §2d's 30-40% fallback band requires the in-domain deliver rate to land at ≥86% (because the 6 OOD queries already account for 30% fallback as designed; any in-domain miss adds to that count). See A8 contingency clause.

---

## A4 — Per-leader balance

10 queries per leader. Split per category as follows:

| Category | Torvalds | Kroah-Hartman |
|---|---:|---:|
| 1. Statistical learning & ML | 2 | 2 |
| 2. Data mining | 2 | 2 |
| 3. Numerical methods | 2 | 1 |
| 4. Programming fundamentals & data structures | 1 (q03_v1 anchor) | 2 (q04_v1 anchor + 1 new) |
| 5. Systems CS absent from corpus | 2 | 2 |
| 6. Off-topic technical | 1 | 1 |
| **Total** | **10** | **10** |

Stylistic-plausibility notes:

- Both leaders get two category-5 systems queries: this is the most pointed test — they have public, strong opinions on these topics, but the corpus doesn't, so fallback should fire on `groundedness` / `final < 0.75` rather than on "leader can't answer." Inspecting `trigger_reason` after the run distinguishes correct fallback from hallucinated deliver.
- The third numerical-methods query goes to Torvalds (2T vs 1KH); Torvalds plausibly has views on numerical precision and floating-point pitfalls.
- The third programming-fundamentals query goes to Kroah-Hartman; he writes extensively about Linux driver code style which lines up with "recursion / sorting / data structures" tone.
- No conversational/small-talk category. N=1 wasn't a test, and small-talk fallback exercises a different upstream code path than knowledge-question fallback. Dropped in revision 1.

---

## A5 — Per-query quality criteria

Each generated query must satisfy all of:

1. **Verifiable corpus alignment.** In-domain queries: top-1 chunk Cohere relevance > 0.3 in Step B2. Out-of-domain queries: top-1 chunk Cohere relevance < 0.1 in Step B3. This is the gate, not author judgment.
2. **Realistic phrasing.** A user might plausibly type the query into a chat box. No stub forms ("tell me about X," "explain Y"). Use natural interrogative phrasing.
3. **Length cap.** 1–2 sentences. No multi-clause compound questions.
4. **No telegraphing.** Out-of-domain queries must not signal "this is out-of-domain" (no "explain a topic the textbook doesn't cover"). They should read as ordinary questions a curious user would ask.
5. **Style-leader independence.** The query must be answerable in either leader's voice. No query that depends on a personal anecdote only one leader could give.
6. **Regression anchors.** Two queries are reused verbatim from v1 with `regression_anchor: true`:
   - **q03 v1** (`"How does binary search work and what is its time complexity?"`) — the May-23 single deliver (final=0.7525 with broken Cohere). Reused in v2 Cat 4 as a Torvalds query.
   - **q04 v1** (`"What is the difference between a stack and a queue, and when is each used?"`) — the Cohere-fixed single deliver (final=0.7738, groundedness=0.6029). Reused in v2 Cat 4 as a Kroah-Hartman query.
   - Rationale: if these score materially differently in v2, something other than query composition has shifted (style profile drift, LLM run-to-run variance beyond expected stochasticity, retrieval determinism issue).
   - No other v1 reuse.

---

## A6 — Out of scope

This spec covers a **20-query system test**. Explicitly out of scope:

- Larger-scale benchmark eval sets (P5-style 200+ QA pairs). Those belong to a future eval-framework project, not P6's wrap day.
- Modifying the corpus (e.g., adding a networking textbook to fix coverage gaps). The corpus is what it is; the eval set adapts to it.
- Modifying scoring functions (groundedness scorer, style scorer, formula weights, threshold). All scoring stays at production values for this run.
- Modifying the RAG pipeline beyond the already-applied Cohere env-var fix.
- Multi-turn or conversational eval. All queries are single-shot.
- Adversarial / red-team queries (jailbreak attempts, prompt injection). Out of scope for a portfolio-wrap day.
- Generating Loom-demo queries. Loom queries are curated separately for narrative, not for system testing.
- Conversational / non-technical small talk. Different upstream code path; not a deliver/fallback knowledge test.

---

## A7 — Headline README artifact: routing-correctness 2×2

The first table in `docs/day8-findings.md` (and the analogous table in the README "Results" section) is a 2×2 routing-correctness grid:

```
                       │  Delivered (final≥0.75)  │  Fallback (final<0.75)  │
───────────────────────┼──────────────────────────┼─────────────────────────┤
 In-domain  (14 queries) │  ✅ Correct routing      │  ⚠ Deliver-path miss   │
 Out-of-domain (6 queries)│  ❌ Hallucination       │  ✅ Correct routing     │
```

Cells are populated with counts after the Step C run. The diagonal (top-left + bottom-right) is the "system worked as designed" count. Off-diagonal cells are explicit failure modes:

- **Top-right (in-domain → fallback)**: deliver-path miss. The corpus has the content but the pipeline didn't bring final-score over 0.75. Diagnosis: groundedness calibration, retrieval quality, or style/confidence dragging the composite down.
- **Bottom-left (OOD → deliver)**: hallucination. The corpus has no grounded content but the pipeline produced confident styled output anyway. Diagnosis: groundedness scorer is too generous (parametric LLM completion happens to share embedding space with random chunks), or style/confidence is masking weak groundedness.

The grid is the headline because it makes routing correctness legible without burying it in raw score tables. Every recruiter who's never run a CrewAI Flow can read this table.

---

## A8 — Contingency: in-domain deliver rate < 85%

The 70/30 split places a hard floor under the in-domain deliver rate: to hit PRD §2d's 30-40% fallback band, in-domain delivery must reach **≥86%** (12 of 14 in-domain queries must score `final >= 0.75`).

If Step C's run produces an in-domain deliver rate below 85%:

- **Do not** retroactively rebalance the query set to inflate the deliver count. Replacing in-domain queries with easier ones after seeing the results gates the test to its outcome — test-integrity failure.
- **Do** document the gap in `docs/day8-findings.md` as a finding: "70/30 split predicts X% fallback; measured Y%; in-domain deliver rate Z%."
- **Do** propose recalibrating PRD §2d in the README narrative based on the measured evidence (e.g., "with this corpus, in-domain deliver tops out at Z% because of [groundedness floor / style-scorer weight / threshold calibration]; PRD §2d's 30-40% band assumed a higher in-domain ceiling").
- **Do not** silently fail the PRD criterion. Either flag it as missed and explain why, or re-derive the target from measured evidence and label it as a re-calibration.

This clause is here so the spec can't be retroactively edited to make the run pass.

---

## Step A exit criteria — APPROVED 2026-05-25 (per user revisions rev 1)

Spec is the binding artifact for Step B query generation. Step B will generate queries matching A3/A4/A5, verify against A5.1 gates, and stop for the user to review the Cohere score distribution before Step C.
