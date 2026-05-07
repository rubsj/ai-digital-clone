"""Diagnostic for Experiment 6a rejection — three problems.

Problem 2: Chunk pool overlap
  For q01 (bit-identical groundedness) and q06 (different groundedness),
  log top-20 FAISS candidate chunk indices for each embedding, compute
  overlap, then log post-Cohere top-5 and identify where scores collapse.

Problem 3: Confidence sub-score breakdown
  For q01, print the three confidence sub-scores (retrieval_relevance,
  completeness, uncertainty_penalty) for both embeddings.

Usage:
    cd /Users/rubyjha/repo/AI/ai-digital-clone
    uv run python scripts/diagnostic_6a_chunk_overlap.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("CO_API_KEY") and os.environ.get("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]

from pathlib import Path
import json

from src.eval.query_loader import load_queries
from src.evaluation.confidence_scorer import (
    _completeness,
    _retrieval_relevance,
    _uncertainty_penalty,
)
from src.rag.indexer import load_index
from src.rag.retriever import retrieve
from src.rag.reranker import rerank

QUERIES_PATH = Path("data/eval/queries_v1.json")
OPENAI_INDEX_DIR = Path("data/rag/faiss_index")
MINILM_INDEX_DIR = Path("data/rag/faiss_index_minilm")

RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5

DIAG_QUERY_IDS = {"q01", "q06"}


def _chunk_index_set(results, top_n=20) -> set[int]:
    return {r.chunk.chunk_index for r in results[:top_n]}


def main() -> None:
    print("Loading indices...")
    oa_index, oa_meta = load_index(OPENAI_INDEX_DIR)
    ml_index, ml_meta = load_index(MINILM_INDEX_DIR)
    print(f"  OpenAI index: {oa_index.ntotal} vectors")
    print(f"  MiniLM index: {ml_index.ntotal} vectors")

    queries = load_queries(QUERIES_PATH)
    diag_queries = [q for q in queries if q["id"] in DIAG_QUERY_IDS]

    sep = "=" * 72

    for qr in diag_queries:
        qid = qr["id"]
        query = qr["query"]
        print(f"\n{sep}")
        print(f"QUERY {qid}: {query[:80]}")
        print(sep)

        # --- Retrieve top-20 from each embedding ---
        oa_top20 = retrieve(query, oa_index, oa_meta, top_n=RETRIEVAL_TOP_N, provider="openai")
        ml_top20 = retrieve(query, ml_index, ml_meta, top_n=RETRIEVAL_TOP_N, provider="minilm")

        oa_ids = [r.chunk.chunk_index for r in oa_top20]
        ml_ids = [r.chunk.chunk_index for r in ml_top20]

        oa_set = set(oa_ids)
        ml_set = set(ml_ids)
        overlap = oa_set & ml_set
        union = oa_set | ml_set

        print(f"\n[PRE-RERANK top-20 chunk_index lists]")
        print(f"  OpenAI : {oa_ids}")
        print(f"  MiniLM : {ml_ids}")
        print(f"\n  Intersection : {len(overlap)}/{RETRIEVAL_TOP_N} chunks identical")
        print(f"  Overlap %    : {len(overlap)/RETRIEVAL_TOP_N*100:.1f}%  ({sorted(overlap)})")
        print(f"  OpenAI-only  : {sorted(oa_set - ml_set)}")
        print(f"  MiniLM-only  : {sorted(ml_set - oa_set)}")

        # --- Rerank each ---
        oa_top5 = rerank(query, oa_top20, top_n=RERANK_TOP_N)
        ml_top5 = rerank(query, ml_top20, top_n=RERANK_TOP_N)

        oa_top5_ids = [r.chunk.chunk_index for r in oa_top5]
        ml_top5_ids = [r.chunk.chunk_index for r in ml_top5]
        oa_scores = [r.score for r in oa_top5]
        ml_scores = [r.score for r in ml_top5]

        print(f"\n[POST-RERANK top-5 chunk_index and Cohere relevance_score]")
        print(f"  OpenAI : {list(zip(oa_top5_ids, [f'{s:.4f}' for s in oa_scores]))}")
        print(f"  MiniLM : {list(zip(ml_top5_ids, [f'{s:.4f}' for s in ml_scores]))}")
        print(f"  Top-5 identical? {oa_top5_ids == ml_top5_ids}")
        print(f"  Scores identical? {[round(s, 4) for s in oa_scores] == [round(s, 4) for s in ml_scores]}")

        # --- Confidence sub-scores breakdown ---
        # response proxy = query text (same as experiment_6a_embeddings.py)
        response = query

        print(f"\n[CONFIDENCE SUB-SCORES — response proxy = query text]")
        for name, top5 in [("OpenAI", oa_top5), ("MiniLM", ml_top5)]:
            rr = _retrieval_relevance(top5)
            comp = _completeness(query, response)
            penalty = _uncertainty_penalty(response)
            total = (rr + comp + penalty) / 3.0
            print(f"  {name}:")
            print(f"    retrieval_relevance  = {rr:.4f}  (mean Cohere score: top-5 mean)")
            print(f"    completeness         = {comp:.4f}  (query keywords in response)")
            print(f"    uncertainty_penalty  = {penalty:.4f}  (1 - hedge_count/5)")
            print(f"    confidence           = ({rr:.4f} + {comp:.4f} + {penalty:.4f}) / 3 = {total:.4f}")

        # Show Cohere score distributions for context
        print(f"\n[COHERE SCORE DISTRIBUTIONS — all top-5]")
        print(f"  OpenAI Cohere scores: {[f'{s:.4f}' for s in oa_scores]}")
        print(f"  MiniLM Cohere scores: {[f'{s:.4f}' for s in ml_scores]}")
        print(f"  OpenAI mean: {sum(oa_scores)/len(oa_scores):.4f}")
        print(f"  MiniLM mean: {sum(ml_scores)/len(ml_scores):.4f}")


if __name__ == "__main__":
    main()
