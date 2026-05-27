"""Cohere Rerank API: 2-stage retrieval top-N → top-K.

Graceful fallback: if the Cohere API call fails for any reason, log a warning
and return the original results sliced to top_n.
"""

from __future__ import annotations

import logging
import os
import time

import cohere

from src.schemas import RetrievalResult

logger = logging.getLogger(__name__)

# Module-level last-call timestamp for COHERE_THROTTLE_SECONDS support.
# Cohere's free/trial tier caps at 10 calls/minute; set the env var to e.g.
# "7" to space calls and avoid silent rate-limit fallbacks during eval runs.
_last_call_t: float = 0.0


def rerank(
    query: str,
    results: list[RetrievalResult],
    model: str = "rerank-english-v3.0",
    top_n: int = 5,
) -> list[RetrievalResult]:
    """Cohere Rerank: reduce results list to top_n by relevance.

    Maps Cohere's ranked indices back to the original RetrievalResult objects
    and re-assigns rank 0..top_n-1. Falls back to original[:top_n] on error.
    """
    if not results:
        return []

    effective_n = min(top_n, len(results))

    global _last_call_t
    throttle_s = float(os.environ.get("COHERE_THROTTLE_SECONDS", "0") or "0")
    if throttle_s > 0:
        wait = throttle_s - (time.monotonic() - _last_call_t)
        if wait > 0:
            time.sleep(wait)
    _last_call_t = time.monotonic()

    try:
        client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY", ""))
        documents = [r.chunk.content for r in results]
        response = client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=effective_n,
        )
        reranked: list[RetrievalResult] = []
        for new_rank, item in enumerate(response.results):
            original = results[item.index]
            reranked.append(
                RetrievalResult(
                    chunk=original.chunk,
                    score=float(item.relevance_score),
                    rank=new_rank,
                )
            )
        return reranked

    except Exception as exc:
        logger.warning("Cohere rerank failed (%s); falling back to top-%d.", exc, effective_n)
        return [
            RetrievalResult(chunk=r.chunk, score=r.score, rank=i)
            for i, r in enumerate(results[:effective_n])
        ]
