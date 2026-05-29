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


def _fallback(results: list[RetrievalResult], effective_n: int) -> list[RetrievalResult]:
    """FAISS top-n passthrough with re-assigned ranks (ADR-002 graceful fallback)."""
    return [
        RetrievalResult(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(results[:effective_n])
    ]


def rerank_with_status(
    query: str,
    results: list[RetrievalResult],
    model: str = "rerank-english-v3.0",
    top_n: int = 5,
) -> tuple[list[RetrievalResult], bool]:
    """Cohere Rerank with an explicit ran-flag.

    Returns (reranked_results, rerank_ran). rerank_ran is True only when the
    Cohere API call actually succeeded; False on missing key or API error
    (results then fall back to FAISS top-n per ADR-002).

    Fail-loud (constraint #3 / v2 raison d'être): a missing/empty COHERE_API_KEY
    is the Day-3 silent-failure bug. We log a loud, specific WARNING naming the
    env var so a misconfigured key can never again degrade precision silently.
    """
    if not results:
        return [], False

    effective_n = min(top_n, len(results))

    global _last_call_t
    throttle_s = float(os.environ.get("COHERE_THROTTLE_SECONDS", "0") or "0")
    if throttle_s > 0:
        wait = throttle_s - (time.monotonic() - _last_call_t)
        if wait > 0:
            time.sleep(wait)
    _last_call_t = time.monotonic()

    api_key = os.environ.get("COHERE_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "COHERE_API_KEY is missing or empty — Cohere reranking cannot run; "
            "falling back to FAISS top-%d. Set COHERE_API_KEY to enable reranking "
            "(ADR-002: 0.52 → 0.74 Precision@5).",
            effective_n,
        )

    try:
        client = cohere.ClientV2(api_key=api_key)
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
        return reranked, True

    except Exception as exc:
        logger.warning("Cohere rerank failed (%s); falling back to top-%d.", exc, effective_n)
        return _fallback(results, effective_n), False


def rerank(
    query: str,
    results: list[RetrievalResult],
    model: str = "rerank-english-v3.0",
    top_n: int = 5,
) -> list[RetrievalResult]:
    """Cohere Rerank: reduce results list to top_n by relevance.

    Maps Cohere's ranked indices back to the original RetrievalResult objects
    and re-assigns rank 0..top_n-1. Falls back to original[:top_n] on error.
    Thin wrapper over rerank_with_status() for callers that don't need the flag.
    """
    reranked, _ = rerank_with_status(query, results, model=model, top_n=top_n)
    return reranked
