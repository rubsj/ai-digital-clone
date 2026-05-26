"""Diagnostic: re-run eval queries, capture full state for groundedness-floor analysis.

Writes scripts/diagnostic_floor.jsonl with one record per (query, leader):
  id, leader, query, styled_response, trigger_reason,
  retrieved_chunks: [{chunk_index, content, score}],
  scores: {style, groundedness, confidence, final, decision}  # null on fallback

Not committed. Not part of src/. Run from repo root with: uv run python scripts/diagnostic_floor.py
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.flow import DigitalCloneFlow  # noqa: E402

QUERY_FILE = Path("data/eval/queries_v1.json")
OUT_FILE = Path("scripts/diagnostic_floor.jsonl")

LEADERS = [("torvalds", "Linus Torvalds"), ("kroah_hartman", "Greg Kroah-Hartman")]


def main() -> None:
    queries = json.loads(QUERY_FILE.read_text())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as out:
        for item in queries:
            for cfg_key, display in LEADERS:
                flow = DigitalCloneFlow()
                try:
                    flow.kickoff(inputs={"query": item["query"], "leader": display})
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "id": item["id"], "leader": cfg_key, "query": item["query"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    out.write(json.dumps(record) + "\n")
                    print(f"  [{item['id']}] {cfg_key}: ERROR {exc}")
                    continue
                s = flow.state
                chunks = [
                    {
                        "chunk_index": r.chunk.chunk_index,
                        "source_topic": r.chunk.source_topic,
                        "content": r.chunk.content,
                        "score": r.score,
                        "rank": r.rank,
                    }
                    for r in s.retrieved_chunks
                ]
                ev = s.evaluation
                scores = None
                if ev is not None:
                    scores = {
                        "style": ev.style_score,
                        "groundedness": ev.groundedness_score,
                        "confidence": ev.confidence_score,
                        "final": ev.final_score,
                        "decision": ev.decision,
                    }
                record = {
                    "id": item["id"],
                    "leader": cfg_key,
                    "query": item["query"],
                    "topic": item.get("topic"),
                    "styled_response": s.styled_response,
                    "trigger_reason": s.trigger_reason,
                    "retrieved_chunks": chunks,
                    "scores": scores,
                }
                out.write(json.dumps(record) + "\n")
                summary = "fallback" if scores is None else f"final={scores['final']:.4f}"
                print(f"  [{item['id']}] {cfg_key}: {summary} trigger={s.trigger_reason!r}")
    print(f"\nWritten: {OUT_FILE}")


if __name__ == "__main__":
    main()
