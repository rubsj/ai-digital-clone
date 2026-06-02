"""Equivalence guard for ADR-017 step 1.5.2.

Uses Phase 1 results (clone_response_text, chunk_contents, stored scores) to
reconstruct the EvaluatorAgent inputs, then runs kickoff + _parse_review to
compare .raw vs draft.explanation. No full flow run; no re-eval budget consumed.
Throwaway script; not part of the production codebase.
"""

from __future__ import annotations
import json, sys
sys.path.insert(0, ".")

from src.agents.evaluator_agent import EvaluatorAgent
from src.components.scoring_engine import Scores
from src.schemas import KnowledgeChunk, RetrievalResult


GUARD_QUERY_IDS = ("q01", "q08", "q13")
GUARD_LEADER = "torvalds"


def _chunks_from_stored(chunk_contents: list[dict]) -> list[RetrievalResult]:
    results = []
    for c in chunk_contents:
        kc = KnowledgeChunk(
            content=c["content"],
            source_topic=c["source_topic"],
            source_field="",
            chunk_index=c["rank"],
        )
        results.append(RetrievalResult(chunk=kc, score=c["score"], rank=c["rank"]))
    return results


def main() -> None:
    records_by_id: dict[str, dict] = {}
    with open("results/evaluation_day12.json") as fh:
        all_records = json.load(fh)
    for rec in all_records:
        if rec["pass"] == 1 and rec["query_id"] in GUARD_QUERY_IDS:
            records_by_id[rec["query_id"]] = rec

    agent = EvaluatorAgent()

    print("=" * 72)
    print("EQUIVALENCE GUARD — .raw vs _parse_review explanation (3 records)")
    print("=" * 72)

    for qid in GUARD_QUERY_IDS:
        rec = records_by_id[qid]
        leader_rec = rec[GUARD_LEADER]
        query = rec["query"]
        response = leader_rec["clone_response_text"]
        chunks = _chunks_from_stored(leader_rec["chunk_contents"])

        scores = Scores(
            style_score=leader_rec["style_score"],
            groundedness_score=leader_rec["groundedness_score"],
            confidence_score=leader_rec["confidence_score"],
        )

        print(f"\n--- {qid}: {query[:70]}...")
        print(f"    gs={scores.groundedness_score:.3f}  "
              f"style={scores.style_score:.3f}  conf={scores.confidence_score:.3f}")

        crew = agent._build_crew(query, response, scores, chunks)
        raw = crew.kickoff().raw
        draft = agent._parse_review(raw)

        print(f"\n  [.raw] ({len(raw)} chars):\n  {raw}\n")
        print(f"  [_parse_review explanation] ({len(draft.explanation)} chars):\n  {draft.explanation}\n")
        print(f"  [_parse_review flags] {draft.flags}")
        print("-" * 72)


if __name__ == "__main__":
    main()
