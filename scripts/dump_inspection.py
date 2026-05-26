"""Dump 5 stratified samples to scripts/diagnostic_inspection.md for manual annotation."""
from __future__ import annotations

import json
from pathlib import Path

records = [json.loads(line) for line in Path("scripts/diagnostic_floor.jsonl").read_text().splitlines() if line]
records = [r for r in records if r.get("scores")]
# Sort by groundedness
records.sort(key=lambda r: r["scores"]["groundedness"])

# Stratified pick: bottom 2, middle 2, top 1
picks = [records[0], records[1], records[len(records)//2 - 1], records[len(records)//2], records[-1]]

lines = ["# Diagnostic Inspection — 5 stratified samples\n"]
lines.append("Selected by cosine groundedness: 2 lowest, 2 middle, 1 top.\n")
for r in picks:
    sc = r["scores"]
    lines.append(f"## {r['id']} — {r['leader']} (cosine groundedness = {sc['groundedness']:.4f})\n")
    lines.append(f"**Topic:** {r.get('topic','?')}  ·  **Final:** {sc['final']:.4f}  ·  **Style:** {sc['style']:.4f}  ·  **Confidence:** {sc['confidence']:.4f}  ·  **Decision:** {sc['decision']}\n")
    lines.append(f"**Query:** {r['query']}\n")
    lines.append("### Styled response\n")
    lines.append("```\n" + r["styled_response"] + "\n```\n")
    lines.append("### Retrieved chunks (top-5)\n")
    for i, c in enumerate(r["retrieved_chunks"][:5]):
        snippet = c["content"][:500].replace("\n", " ")
        lines.append(f"**Chunk {i} [{c['source_topic']}] score={c['score']:.3f}:**\n\n> {snippet}…\n")
    lines.append("---\n")

Path("scripts/diagnostic_inspection.md").write_text("\n".join(lines))
print("Wrote scripts/diagnostic_inspection.md")
print(f"Picks: {[(r['id'], r['leader'], round(r['scores']['groundedness'],4)) for r in picks]}")
