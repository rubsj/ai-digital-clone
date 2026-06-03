# Day 13 Investigation Plan: Torvalds Floor Adjudication

**Executing model:** Sonnet  
**Judge model:** Opus (blind, inter-phase)  
**Data:** results/evaluation_day12_reeval2.json  
**Query set:** data/eval/queries.json  
**Deviation policy:** if data contradicts a stated premise, STOP and report. Do not improvise past a gate.

---

## Goal

Adjudicate why Torvalds sits below his 42.9% floor while KH ships, using comparative span analysis on existing data. The fork is:

- **H2:** off-source persona tails, localized and removable, a clone-prompt bug, floor stays
- **H3:** terse voice asserts past the chunks throughout, a legitimate deficit, forces a floor/persona decision

Blend is possible; the investigation quantifies the split.

---

## PHASE A — BUILD THE BLINDED MARKUP ARTIFACT

### Step 1: Assertions
Load reeval2. Assert:
- 42 records = 14 in-domain queries x 3 runs
- both clones per record
- chunk_contents byte-identical across the two clones within each record

If any assertion fails, STOP and report.

### Step 2: Assemble per-(query, run, clone) records
For each (query, run, clone) response (84 total), assemble:
- query text
- category
- the 5 shared chunks (stored once per query+run, referenced by index)
- clone_response_text segmented into numbered sentence-level spans

### Step 3: Blind it
Strip everything that leaks leader or prior verdict before Opus sees it:
- Fields to strip: leader_name/key, groundedness_score, style_score, confidence_score, trigger_category, trigger_reason, routing_reasoning, decision, output_type, flags, delivered_text
- Mark the CLONE response only, never the fallback text
- Relabel the two clones as A/B, RANDOMIZED per query (so A does not track one leader across queries)
- Write per-record A/B <-> leader mapping to: docs/experiments/day13/blind_key.json
- Write blinded artifact to: docs/experiments/day13/markup_input.json

### Step 4: Mechanical coverage lines
Compute per-query coverage summary for the later audit cards (not for Opus):
- query, category
- mechanical summary: chunk count, lengths, most frequent salient noun phrases
- No judgment about answer quality
- Unit/method stated

### STOP GATE 1
Report: artifact path, key path, strip confirmation, span counts.
Wait for Opus markup file at docs/experiments/day13/markup_output.json before Phase B.

---

## PHASE B — EXCISION ARITHMETIC + AUDIT CARDS
(run only after markup_output.json exists)

### Step 6: Ingest Opus span labels
Labels: grounded / inferable / free; plus checked-chunk per span

### Step 7: Length-weighted arithmetic
State unit. Per (query, clone):
- grounded_len, inferable_len, free_len, total_len, supported_fraction
- Aggregate 3 runs per query+clone (mean)
- FLAG any span whose label differs across runs as run-instability

### Step 8: Within-query discriminator
Same chunks, two clones. Compare:
- grounded core (grounded_len A vs B)
- free portion (free_len A vs B)
- free-span DISTRIBUTION: contiguous/trailing = localized; interleaved = distributed (use span positions)

### Step 9: Per-query classification rule
- H2 if grounded cores are comparable AND lower clone carries extra LOCALIZED free spans
- H3 if lower clone's grounded core is itself smaller AND/OR its free spans are distributed
- else blend, quantified

### Step 10: Pivotal spans + audit cards
- Pivotal span: any span whose label, if flipped, would change its query's H2/H3 classification
- Per pivotal span, emit a self-contained audit card:
  - span verbatim
  - Opus label
  - top-2 chunks by overlap with "show all 5" expansion available
  - "why pivotal" line with exact numeric swing
  - blinded clone label
- Print four-test rubric ONCE as header above all cards, not per card
- Attach mechanical coverage line per query
- Emit one-screen full-markup summary per query (all spans, pivotal or not)

### STOP GATE 2
Deliver cards + summary. Wait for human's pivot adjudication. Do not compute the verdict.

---

## PHASE C — VERDICT
(run only after human returns pivot overrides)

### Step 12: Apply overrides
Apply human overrides. Recompute. Un-blind via blind_key.json for reporting.

### Step 13: Verdict
H2 / H3 / blend, with quantified split:
- what fraction of aggregate Torvalds-vs-KH deficit is excision-recoverable
- how many of the 14 queries fall each way
- Report in BOTH units: groundedness magnitude and implied pass-rate effect at 0.60 cutoff

### Step 14: Recommended next actions (OPTIONS only, not executed)
- H2 -> clone-prompt fix then re-gate
- H3 -> floor decision or persona re-spec (flag: ADR required, do NOT write it)
- blend -> both with residual quantified

### Step 15: Verb-and-count audit
Reconcile: 14 queries, 84 responses marked, total spans, pivots audited, deltas computed.
Then Phase Defence, 4-category debrief per CLAUDE.md.

### EXIT GATE
Verdict + memo delivered. FIX NOT STARTED. No clone edits, no floor change, no ADR written, no Phase 2 work, no Notion writes. Hand the verdict memo back and stop.
