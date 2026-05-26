"""Step B verification: FAISS retrieve + Cohere rerank, no full pipeline.

For each query in data/eval/queries_v2.json:
  1. FAISS top-20 via src.rag.retriever
  2. Cohere rerank to top-5 via src.rag.reranker
  3. Capture top-1 Cohere score and top-5 mean
  4. Apply gates: in-domain >0.3, out-of-domain <0.1

Reports per-query table and pass/fail aggregate. Does NOT call the full Flow.
"""
from __future__ import annotations

import json
import statistics as s
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import faiss  # noqa: E402

from src.rag.retriever import retrieve  # noqa: E402
from src.rag.reranker import rerank  # noqa: E402
from src.rag.indexer import load_index  # noqa: E402

QUERIES = Path("data/eval/queries_v2.json")
INDEX_DIR = Path("data/rag/faiss_index")

IN_DOMAIN_THRESHOLD = 0.3
OOD_THRESHOLD = 0.1


def main() -> None:
    queries = json.loads(QUERIES.read_text())
    index, metadata = load_index(INDEX_DIR)
    print(f"Loaded index: ntotal={index.ntotal}, metadata entries={len(metadata)}")

    rows = []
    last_cohere_t = 0.0
    THROTTLE_S = 7.0  # Cohere trial-key limit: 10 calls/min → 6s. Use 7s for headroom.
    for q in queries:
        # FAISS top-20 → Cohere rerank to top-5
        retrieved = retrieve(q["query"], index, metadata, top_n=20)
        # Throttle so we don't trip the 10/min trial-key limit
        elapsed = time.monotonic() - last_cohere_t
        if elapsed < THROTTLE_S:
            time.sleep(THROTTLE_S - elapsed)
        last_cohere_t = time.monotonic()
        reranked = rerank(query=q["query"], results=retrieved, top_n=5)
        top1 = reranked[0] if reranked else None
        top1_score = top1.score if top1 else 0.0
        top5_scores = [r.score for r in reranked]
        top5_mean = s.mean(top5_scores) if top5_scores else 0.0
        snippet = (top1.chunk.content[:90].replace("\n", " ")) if top1 else ""
        rows.append({
            "id": q["id"],
            "leader": q["leader"],
            "category": q["category"],
            "expected": q["expected_behavior"],
            "anchor": q.get("regression_anchor", False),
            "top1_score": top1_score,
            "top5_mean": top5_mean,
            "top1_snippet": snippet,
            "top1_topic": top1.chunk.source_topic if top1 else "",
        })

    # In-domain table
    in_dom = [r for r in rows if r["expected"] == "deliver"]
    ood = [r for r in rows if r["expected"] == "fallback"]

    def fmt_row(r):
        flag = ""
        if r["expected"] == "deliver":
            flag = "PASS" if r["top1_score"] > IN_DOMAIN_THRESHOLD else "FAIL"
        else:
            flag = "PASS" if r["top1_score"] < OOD_THRESHOLD else "FAIL"
        anchor = "  *anchor*" if r["anchor"] else ""
        return (f"  {r['id']:4s} {r['leader']:14s} {r['category']:30s} "
                f"top1={r['top1_score']:7.4f}  top5_mean={r['top5_mean']:7.4f}  "
                f"[{flag}]{anchor}\n     → {r['top1_topic'][:50]} :: {r['top1_snippet']}")

    print("\n" + "=" * 110)
    print(f"IN-DOMAIN (gate: top1 > {IN_DOMAIN_THRESHOLD})")
    print("=" * 110)
    for r in in_dom:
        print(fmt_row(r))

    print("\n" + "=" * 110)
    print(f"OUT-OF-DOMAIN (gate: top1 < {OOD_THRESHOLD})")
    print("=" * 110)
    for r in ood:
        print(fmt_row(r))

    # Distribution stats on in-domain top-1
    scores_in = [r["top1_score"] for r in in_dom]
    print("\n" + "=" * 110)
    print("IN-DOMAIN top-1 Cohere score distribution")
    print("=" * 110)
    print(f"  n={len(scores_in)}  min={min(scores_in):.4f}  max={max(scores_in):.4f}  "
          f"mean={s.mean(scores_in):.4f}  median={s.median(scores_in):.4f}")
    print(f"  PASS (>{IN_DOMAIN_THRESHOLD}): {sum(1 for x in scores_in if x > IN_DOMAIN_THRESHOLD)}/{len(scores_in)}")

    scores_ood = [r["top1_score"] for r in ood]
    print("\nOOD top-1 Cohere score distribution")
    print(f"  n={len(scores_ood)}  min={min(scores_ood):.4f}  max={max(scores_ood):.4f}  "
          f"mean={s.mean(scores_ood):.4f}  median={s.median(scores_ood):.4f}")
    print(f"  PASS (<{OOD_THRESHOLD}): {sum(1 for x in scores_ood if x < OOD_THRESHOLD)}/{len(scores_ood)}")

    # Decision
    flagged = []
    for r in rows:
        if r["expected"] == "deliver" and r["top1_score"] <= IN_DOMAIN_THRESHOLD:
            flagged.append(r["id"])
        if r["expected"] == "fallback" and r["top1_score"] >= OOD_THRESHOLD:
            flagged.append(r["id"])
    print("\n" + "=" * 110)
    if not flagged:
        print("DECISION: B PASSED — all 20 queries clear their respective gates.")
    else:
        print(f"DECISION: B FAILED — {len(flagged)} queries flagged: {flagged}")

    # Persist for later steps
    Path("scripts/verify_queries_v2.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
