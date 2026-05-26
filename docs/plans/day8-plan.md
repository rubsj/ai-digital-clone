# Day 8 Plan — P6 Wrap (README + ADR Spot-Check + Hub + Tracker)

**Date:** 2026-05-25 (Day 3 of Portfolio Sprint, final day of P6)
**Budget:** 3 hours hard cap. Polish/wrap day, not implementation.
**Branch:** `feat/day8-readme-and-wrap` off main (current main: `162949a` on `ai-portfolio`, P6 main: `2c52b1f`)
**Going-in state:** Days 1–7 merged. 464 tests passing, src/ 94% coverage. All PRD §7 deliverables shipped. 8 ADRs, 5 architecture diagrams, 7-chart gallery in `results/charts/`, 6 methodology charts in `docs/experiments/charts/`. No outstanding code work.

---

## Goal of the day

Take P6 from "code complete on disk" to "shippable as a portfolio artifact." That means: a recruiter or hiring engineer landing on the GitHub repo can read the README in 3 minutes and walk away with (1) what was built, (2) the headline numbers, (3) the non-obvious design decisions, (4) the known gaps. Plus the hub README and the Notion Project Tracker reflect the same story.

This is not a writing day for new content — Day 7 produced the diagrams, charts, ADR-008 narrative, and session notes that the README will draw from. Day 8 is rearrangement and condensation against the P1/P2 reference pattern.

---

## Reference: the "gold standard" P1/P2 pattern

After reading [P1 README](../../../ai-synthetic-data-generator/README.md) and [P2 README](../../../ai-rag-evaluation-framework/README.md) the inverted-pyramid structure is:

1. **Title + one-paragraph hook.** What was built + the single most surprising finding, in ~3 sentences. P2's opening: "I tested 16 RAG configurations and found semantic chunking + OpenAI embeddings + Cohere reranking gets 0.747 Recall@5. This is how I got there."
2. **Badges + dashboard screenshot + Live Dashboard placeholder.** Python/License badges, a centered hero image, "Live Dashboard: Deploying in Week 8 of the portfolio sprint."
3. **Results.** Tabular headline metrics. P1 shows the failure-reduction stages. P2 shows pre/post-rerank deltas.
4. **Findings / What the numbers mean.** Short prose paragraphs unpacking the surprises. "Overlap is overrated." "Reranking is the cheapest win." Each finding gets a one-line lede and 1–3 sentences of evidence.
5. **Architecture.** Mermaid diagram inline + 1–2 paragraphs explaining the experimental structure.
6. **Design decisions (ADR table).** 3-column markdown table: ADR link · Decision · Trade-off. One row per ADR.
7. **Evaluation charts.** Centered images with a 1-sentence reading caption below each.
8. **Tech Stack table.** Component · Library · Why this one. P2 has 11 rows. Forces explicit justification per dependency.
9. **Quick Start.** Copy-pasteable shell block. Setup → run pipeline → run tests → launch demo.
10. **Known Gaps.** Honest list of what's not covered and why. P2 has 5 bullets. P1 has 2.
11. **Footer.** "Part of a [9-project AI engineering sprint]. Built Feb–May 2026." Plus byline with LinkedIn + GitHub.

The pattern's core discipline: **lead with the result, not the architecture**. P2 doesn't explain FAISS until §5. The headline numbers and the surprising finding go first because that's what a reader actually needs to decide whether to keep reading.

---

## Task 1 — P6 README rewrite (90 minutes, the bulk of the day)

**Current state:** [README.md](../../README.md) is a 12-line "Coming Soon" placeholder. Full rewrite, not edit.

**Approach:** I draft the structure section-by-section, Ruby reviews and rewrites in their own voice (per the structure-then-draft pattern named in Day 7 process learnings). No prose comes from me into the file directly — I propose, Ruby writes, I redline.

**Section-by-section plan:**

- **§1 Hook (1 paragraph, ~3 sentences).** Anchor: "Multi-agent CrewAI Flow that writes in two real people's styles (Torvalds and Kroah-Hartman), grounds answers in their LKML email history via RAG, and routes to a fallback when groundedness drops below 0.75." The single most surprising finding to lead with: **TBD with Ruby during review** — candidates: (a) groundedness scoring viability with cosine vs LLM judge (ADR-007's negative result), (b) the dual-leader retrieve-once-style-twice optimization, (c) the LoRA-via-prompt style cloning result vs heavyweight fine-tuning. Recommend (a) — negative-result honesty is a portfolio differentiator that P1/P2 don't lean on.

- **§2 Badges + hero image + Live Dashboard placeholder.** Use the existing Streamlit screenshot if one was captured during Day 7; if not, generate one in 5 min by running `uv run streamlit run streamlit_app.py`, taking a screenshot of Compare mode, saving to `docs/screenshots/dashboard.png`, and committing on the same Day 8 branch.

- **§3 Results table.** Three rows minimum: average style score (Torvalds), average style score (Kroah-Hartman), groundedness pass rate at 0.75 threshold. Pulled from `results/` JSONs (need to check what's there). Plus the headline test/coverage counts (464 tests, 94% coverage) as a single sentence below the table — these belong to engineering rigor, not to the model's quality story.

- **§4 Findings.** 3-5 short prose paragraphs. Candidate findings from Day 6 methodology and Day 7 process learnings:
  - "Cosine similarity for groundedness is not viable" — ADR-007 negative result.
  - "Shared retrieval halves cost in dual-leader mode" — ADR-005.
  - "Feature vectors beat LLM embeddings for style scoring" — ADR-003.
  - "The 0.75 threshold is load-bearing but undocumented" — naming the gap (referenced to Post-Portfolio Followups as ADR-009 candidate) is the kind of honesty hiring engineers read for signal.
  - "Corpus-shape limits constrain LKML's style signal" — ADR-006.

- **§5 Architecture.** Embed the A1 system diagram (`docs/architecture/system-architecture.md`) as the primary Mermaid block. Reference A2-A5 by link below it. 1-paragraph explanation of the 5-agent structure + the @router branch.

- **§6 Design decisions table.** Eight rows, one per ADR. Each Trade-off cell is ≤15 words. Sample rows:
  - ADR-001: CrewAI Flow over Sequential/Hierarchical · Typed state + explicit @router branching, deterministic routing
  - ADR-003: Feature vectors over LLM embeddings · Faster, deterministic, interpretable per-dimension contribution
  - ADR-007: Cosine groundedness not viable · Negative result; LLM judge is the only viable path
  - ADR-008: Hexagonal adapters · CLI and Streamlit wrap Flow façade, never import LLM/FAISS/Cohere
  - (Fill remaining 4 from ADR fronts during execution.)

- **§7 Evaluation charts.** 4-5 of the 7 PRD §7d charts inline. The full 7-chart gallery referenced by directory link. Each centered image needs a 1-sentence caption telling the reader what to notice. Best candidates for inline (per Day 7 handover):
  - `01-style-radar.png` — visual hook, recruiter-friendly
  - `03-groundedness-distribution.png` — shows the 0.75 split working
  - `04-score-breakdown.png` — explains the weighted formula
  - `05-fallback-rate.png` — shows the routing actually triggers

- **§8 Tech Stack table.** ~12 rows. Use P2's column structure (Component · Library · Why). Must include the CrewAI / LiteLLM / FAISS / Cohere / Click / Streamlit / pytest stack at minimum. Each "Why" cell is the justification the matching ADR landed on, condensed to one sentence.

- **§9 Quick Start.** Copy-pasteable. Verify each command actually runs before committing:
  ```bash
  uv sync
  cp .env.example .env  # Add OPENAI_API_KEY, COHERE_API_KEY
  uv run python -m src.cli learn data/torvalds.mbox --leader torvalds
  uv run python -m src.cli index data/lkml_corpus/
  uv run python -m src.cli query "Why does the kernel reject this patch?" --leader torvalds
  uv run python -m src.cli compare "Why does the kernel reject this patch?"
  uv run python -m src.cli evaluate
  uv run streamlit run streamlit_app.py
  ```

- **§10 Known Gaps.** Pull directly from Day 7 handover's "Known Gaps" section — already written, well-calibrated. Four bullets: 0.75 threshold has no ADR, Streamlit caching deferred, latency only wall-time, diagram-set 0.75 cleanup. Add a fifth: the LKML corpus is style-signal-poor compared to long-form prose (Day 6 finding, ADR-006).

- **§11 Footer.** Exact P1/P2 footer pattern.

**Out of scope for the README rewrite:**
- New screenshots beyond the dashboard hero (Loom is a separate property, not embedded).
- Streamlit caching fix (Post-Portfolio Followups).
- ADR-009 for the 0.75 threshold (Post-Portfolio Followups).
- Any code or test changes.

---

## Task 2 — ADR-001 to ADR-004 spot-check (10 minutes, lightweight)

**Scope:** ADR-001 through ADR-004 were written before the 5-section Engineering Protocols standard was formalized in Notion. Confirm they match the current standard. ADR-005 through ADR-008 were written after, and Day 7 humanization already covered them — no work needed there.

**Section headers already confirmed present** (just verified via grep): all four files have Context, Decision, Alternatives Considered, Quantified Validation, Consequences. So the structural check passes.

**Content depth spot-check (the real work):**
- Open each ADR. Skim the **Quantified Validation** section specifically (this is the section most likely to have been thin in early drafts).
- Verify each one has at least one of: concrete number, named source (paper / blog / docs reference), or POC-script reference.
- Verify Consequences names a real downstream effect, not just rephrases the Decision.

**Output:**
- If all four pass: 1-line confirmation in the Day 8 journal. No file edits today.
- If any ADR is thin: do not fix today. Log the specific delta (e.g., "ADR-002 Quantified Validation has no numbers, only qualitative claims") to the [Post-Portfolio Followups page](https://www.notion.so/36cdb630640a812a9d99d79951011897) as a new entry. Day 8 budget can't absorb an ADR humanization pass on top of the README.

**Why log-don't-fix:** the Engineering Protocols standard explicitly accepts that early ADRs may need a humanization sweep, and that sweep should be a discrete deliberate session, not a Day 8 polish-day rabbit hole.

---

## Task 3 — Hub README update (15 minutes)

**Current state of [ai-portfolio/README.md](../../../ai-portfolio/README.md):**
- P1-P5 are under `## Completed Projects` with the full P1/P2 entry shape: 1-paragraph description, Result line, tech badges, repo badge.
- P6 is currently under `## Upcoming Projects` with a planned-state description, no Result line.
- Portfolio Stats reads: "2,038 tests across P1-P5".

**Edits required:**
1. **Move the P6 block** from `## Upcoming Projects` to the bottom of `## Completed Projects` (after P5).
2. **Rewrite P6 description** in P1/P2 voice. The current text is forward-looking ("StyleAnalyzer extracts...EvaluatorAgent scores...PlannerAgent orchestrates") and reads as a plan. Replace with delivery-state prose anchored on the actual built system + the headline finding. Match the 3-4-sentence length of P5's entry.
3. **Add a Result line** for P6. Format: `**Result:** <style score result> · <groundedness result> · 464 tests · 94% coverage · 8 ADRs · 5 architecture diagrams`. Exact numeric values pulled from Day 7 handover Key Metrics block.
4. **Update Portfolio Stats:**
   - "2,038 tests across P1-P5" → "2,502 tests across P1-P6" (2,038 + 464)
   - "25 ADRs" → "33 ADRs" (25 + 8)
   - Leave the "97% on P5, 99% on P4" line untouched; optionally add "94% on P6".
5. **Update the Mermaid graph in `## How the Projects Connect`:** change `P6[P6: Writing Clone]:::upcoming` to `:::done`. The classDef line stays; the existing P3→P6 and P5→P6 edges stay.
6. **Update tech badges** for the P6 entry: current is `Python | CrewAI | OpenAI | Sentence-Transformers`. Actual stack is closer to `Python | CrewAI | LiteLLM | FAISS | Cohere | Click | Streamlit`. Match what the new P6 README §8 Tech Stack lands on.

**Commit + PR strategy:** the hub README change is a separate PR on the `ai-portfolio` repo, on a `feat/p6-hub-entry` branch. Per the feedback in memory (no direct commits to main), even one-line hub changes go through a PR.

---

## Task 4 — P6 Project Tracker (Notion) update (20 minutes)

**Current Notion state** (from fetching [P6 page](https://www.notion.so/2ffdb630640a81e69e42c595d95ee5af)):
```
Status: Backlog
Completed: false
Demo Link: empty
Loom Link: empty
Est. Hours: 22
GitHub Folder: 06-digital-writing-clone
Key Tech: [OpenAI API, ChromaDB, Sentence-Transformers]
Start Date: 2026-03-03
End Date: 2026-03-08
Priority: 6
Tier: Intermediate
Week: Week 5
```

**Target values for properties:**

| Property | New value | Source |
|----------|-----------|--------|
| Status | Complete | n/a |
| Completed | true | n/a |
| Demo Link | https://github.com/rubsj/ai-digital-clone (or live Streamlit URL if deployed by EOD; otherwise repo URL is the accepted placeholder per P1/P2 precedent) | Repo URL is canonical |
| Loom Link | empty for now — Loom recording is a Portfolio Sprint deliverable not yet scoped for today. Track as a separate followup if not done. Recommend leaving empty rather than placeholder. | n/a |
| Est. Hours | Keep at 22 (estimate) OR add an "Actual Hours" — Notion property doesn't currently exist for actual, so leave Est. Hours alone. Total actual hours: rough estimate 26-30h across 7 working sessions per handovers. Optional: add an Actual Hours property to the Tracker schema. Recommend NOT adding schema today — too much yak-shaving on a wrap day. | Day handovers |
| GitHub Folder | 06-digital-writing-clone — verify this matches the actual repo location (`ai-digital-clone`, no `06-` prefix in the repo names). May need correction to `ai-digital-clone`. | Repo on disk |
| Key Tech | Update to match actual stack: `[Python, CrewAI, LiteLLM, FAISS, Cohere, Click, Streamlit, Pydantic]`. Current `ChromaDB` and `Sentence-Transformers` are wrong — the project uses FAISS and OpenAI/LiteLLM embeddings, not Chroma or local sentence-transformers (post-ADR-002 decision). | ADR-002, ADR-008 |
| Start Date | Keep as 2026-03-03 (original planned start) OR change to actual first-commit date. The Day 1 handover is dated Apr 3, 2026 — that's the actual start. Recommend changing to actual `2026-04-03`. | Day 1 handover |
| End Date | 2026-05-25 (today, Day 8 wrap) — change from the original planned 2026-03-08. | Today |
| Priority / Tier / Week | Leave unchanged. | n/a |

**Open question for Ruby's call during plan review:** Loom Link policy. If the portfolio sprint expects Loom by June 13 (Miami), then leaving Loom Link empty on May 25 is fine and matches reality. If the expectation is that Loom is part of P6's "complete" definition, then P6 stays at Status=In Progress with Completed=false until the Loom exists. **Recommend Complete=true with Loom Link empty** — the code/docs deliverable is done, Loom is a separate sprint-wide track.

**Mechanism:** use `notion-update-page` against page ID `2ffdb630640a81e69e42c595d95ee5af`. Each property update is one call. Will batch but sequence is fine.

---

## Task 5 — Final integration check (10 minutes)

Before declaring P6 done:
1. From `/Users/rubyjha/repo/AI/ai-digital-clone`: `git status` clean, `git log --oneline -1` shows `2c52b1f` on main.
2. From the same dir: `uv run pytest -q` — confirm 464 passing, 0 failing.
3. `uv run python -m src.cli --help` runs and lists all 5 commands.
4. The new README renders cleanly when previewed (Mermaid diagram parses, all image paths resolve, ADR links resolve).
5. The hub `clone-all.sh` includes `ai-digital-clone` (verify, since the script was added recently).

If anything fails, the README/hub PRs do not merge today.

---

## Task 6 — Day 8 journal entry (15 minutes, end-of-day)

Per the Teach-Back Protocol, one entry in [docs/learning-journal.md](../../docs/learning-journal.md) at end-of-day:
- What was wrapped (README, hub, tracker).
- What the README-writing process taught about the project (often you don't know what the headline finding *is* until you have to compress to 3 sentences).
- ADR spot-check result.
- Loose ends going into P7 (May 26 start).

Structure-then-draft per the failure mode named in Day 7: I'll provide the paragraph skeleton during execution. Ruby fills the prose.

---

## Sequencing within the 3-hour budget

| Block | Task | Time |
|-------|------|------|
| 0:00–0:10 | Branch off main: `feat/day8-readme-and-wrap`. ADR-001 to ADR-004 spot-check (Task 2). Log deltas if any. | 10 min |
| 0:10–1:40 | P6 README rewrite (Task 1). Structure-then-draft loop per section. | 90 min |
| 1:40–2:00 | Run integration checks (Task 5). Commit P6 README + dashboard screenshot. Open PR on `ai-digital-clone`. | 20 min |
| 2:00–2:15 | Hub README rewrite (Task 3). Separate branch on `ai-portfolio`. PR opened. | 15 min |
| 2:15–2:35 | Project Tracker updates (Task 4). | 20 min |
| 2:35–2:50 | Journal entry (Task 6). | 15 min |
| 2:50–3:00 | Buffer / overflow. | 10 min |

**Stop-loss triggers** (if any of these hit, drop the lower-priority work):
- README rewrite passes 90 min: drop a section's polish, ship what's there, log polish as followup.
- Integration check fails: stop. Don't open PRs on a broken state. Investigate root cause; reschedule wrap.
- Project Tracker tooling balks (Notion property type mismatches): hand-update via UI rather than burning time on tool fixes.

---

## What gets shipped at EOD

Two PRs:
1. `ai-digital-clone` repo: `feat/day8-readme-and-wrap` → main. Files: `README.md` (full rewrite), `docs/screenshots/dashboard.png` (new), `docs/plans/day8-plan.md` (this file).
2. `ai-portfolio` repo: `feat/p6-hub-entry` → main. Files: `README.md` (P6 entry promoted to completed + stats updated + Mermaid edge style updated).

Plus:
- Notion P6 Project Tracker entry flipped to Complete with corrected properties.
- Day 8 entry in `learning-journal.md`.
- Followups page may gain 0-N entries from the ADR spot-check.

After EOD, P6 is closed. P7 setup begins May 26.

---

## Open questions for plan review

1. **Headline finding to lead the README hook with.** Recommend ADR-007's groundedness negative result. Alternates: ADR-005 dual-leader optimization, ADR-003 feature vectors over LLM embeddings.
2. **Loom Link policy.** Recommend Complete=true with Loom empty (Loom is a sprint-wide track, not a P6 deliverable definition).
3. **Whether to capture a fresh dashboard screenshot today or use an existing one.** Recommend capture-today (5 min) if no Day 7 screenshot exists; this is the README hero image.
4. **Whether to add an Actual Hours property to the Tracker schema.** Recommend no — schema change is yak-shaving on a wrap day; defer.
5. **Start Date correction** (2026-03-03 planned → 2026-04-03 actual). Recommend correcting since End Date is also being corrected; consistency matters.
