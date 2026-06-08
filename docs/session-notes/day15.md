# Day 15 — Session Notes

> Append per phase. This is the opening entry (Phase P1 — Archive results/). No source code changed this phase.

## Phase P1 — Archive results/

### Built

- Classified all 10 non-chart files in `results/` top level into three buckets: current-truth, audit-trail, worthless. The 7 chart PNGs in `results/charts/` were left in place; they are P3a scope.
- Confirmed the 7 pre-classified files from the day15 plan citation map against disk — no mismatches.
- Proposed buckets for the 3 plan-flagged candidates: `evaluation_day12.json` (audit-trail, live doc citations), `evaluation_day12_reeval.json` (audit-trail, session-note citations), `evaluation_20260523_121048.json` (worthless, pre-scoring-schema snapshot).
- Created `results/archive/` and archived 6 files via `git mv` (renames, not delete+add):
  - Audit-trail: `bakeoff_w1b0_day14.json`, `bakeoff_hhem_probe_day14.json`, `evaluation_day12_reeval2.json`, `evaluation_day12.json`, `evaluation_day12_reeval.json`
  - Worthless: `evaluation_20260523_121048.json`
- Kept 4 current-truth files at `results/` top level: `bakeoff_hhem_isolated_day14.json`, `w1b2_threshold_day14.json`, `w3a_metric_effect_day14.json`, `w3b_retrieval_effect_day14.json`.
- Updated 14 by-path doc citations across 7 files to point to the new `archive/` paths: `docs/day11-evaluation.md` (2 hits), `docs/evaluation-methodology.md` (1), `docs/session-notes/day12.md` (6), `docs/session-notes/day13.md` (1), `docs/session-notes/day14.md` (2), `docs/plans/day13-investigation.md` (1), `docs/experiments/day13/scorer_probe.md` (1).
- Left `docs/session-notes/day12.md:237` unchanged — the one remaining stale path is inside a verbatim ```` ``` ```` block reproducing the literal stdout of `scripts/analyze_reeval.py`.
- Wrote `results/MANIFEST.md`: per-file question, cited-by ADR/phase, bucket, beside-not-over sentence for `w3a_*`/`w3b_*`, coherence note, and zero-citation grep for the worthless file.

### Why

The ADRs cite evidence in prose ("the bake-off," "Probe A," "the W3a isolated metric effect," "q07 T W3a 0.285") rather than by file path — `grep -r "results/...json" docs/adr/` returns nothing. Moving a file therefore breaks no markdown link. But mis-bucketing one orphans the evidence behind a shipped ADR claim: the number stays in the ADR text with no recoverable source on disk. The manifest is the deliverable because the directory layout carries no semantic meaning; the manifest is the sole anchor between files and the claims that depend on them.

`w3a_metric_effect_day14.json` and `w3b_retrieval_effect_day14.json` stay current-truth rather than being archived because each ran the scorer(s) on frozen `(response, chunks)` to isolate one variable — the isolated-variable question that a live multi-pass run cannot answer. The forthcoming P2 run sits beside them, not over them.

The archive-not-delete default applies because the worthless snapshot (`evaluation_20260523_121048.json`) cleared the zero-citation grep, making deletion safe, but safe is not the same as worthwhile. Explicit approval is required to delete; absent that, archive.

`docs/session-notes/day12.md:237` was preserved without path rewriting because it is a verbatim ```` ``` ```` block reproducing the literal stdout of `scripts/analyze_reeval.py` on a specific day. Editing it to reflect the current file path would falsify a historical artifact.

### Surprising

- The day15-plan premise "the ADRs cite numbers not paths" was accurate for `docs/adr/` but false for `docs/` at large. `evaluation_day12.json` had 5 by-path citations outside the ADR directory: 2 in `docs/day11-evaluation.md`, 1 in `docs/evaluation-methodology.md`, and 2 in `docs/session-notes/day12.md`. The wider grep was necessary.
- The q07 Torvalds line in the drafted w3b manifest entry was inverted at the stop gate. The draft read "q07 T 0.335 > 0.40 threshold confirmed deliver under fixed retrieval." The fact is the opposite: HHEM 0.335 is below the 0.40 gate, so q07 T falls back after the dedup fix despite being oracle-grounded (53.6%). This is the accepted paraphrase-misroute limitation, not a deliver confirmation. Corrected before any file was written.
- `evaluation_day12_reeval.json` had a mixed citation profile in `docs/session-notes/day12.md`: two by-path hits (lines 135 and 397, updated) plus one hit inside a verbatim code block (line 237, preserved). The three lines required three different dispositions in the same file.

### Deferred

- `src/eval/harness.py:183` has a default output parameter of `"results/evaluation_day12.json"`. This is a write default, not a read; moving the file does not break the harness. Routed to P1b's §12.5 stale-comment/constant audit.
- `evaluation_20260523_121048.json` archived, not deleted. Zero-citation grep confirms deletion is safe; explicit approval needed to act on it.
- P1b (codebase audit + `docs/codebase-audit.md` template): independent of P1, may proceed in parallel from t0 per the plan dependency graph.
- P2 (canonical multi-pass run): gated on P1 merge. Cost-gate applies; call-count estimate must be computed fresh before any spend.

### ADR candidates

- None. P1 touched no decision. `results/` layout is operational housekeeping, not an architectural choice. No locked decision was reopened.
