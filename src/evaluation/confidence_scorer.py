"""Confidence scorer: multi-factor heuristic, no LLM calls.

Three sub-signals, each in [0, 1], weighted equally (1/3 each):

  1. retrieval_relevance  — mean reranker score across top-5 results.
  2. completeness         — fraction of query keywords found in response.
  3. uncertainty_penalty  — 1 − min(1, hedge_count / 5) where hedge_count
                            counts hedging phrases in the response.

NOTE: The equal 1/3 sub-weights are a starting assumption.
      Day 6 will sweep these weights against a labeled validation set.

Target: > 0.80 (per PRD quality table).
"""

from __future__ import annotations

import re

from src.schemas import RetrievalResult

# Stopwords to strip from query keywords before coverage check.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for",
        "on", "with", "that", "this", "are", "was", "were", "be", "as", "at",
        "by", "not", "but", "from", "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "may", "might", "shall", "should",
        "about", "into", "than", "then", "when", "what", "how", "why", "who",
        "which", "their", "there", "they", "you", "your", "we", "our", "us",
        "he", "she", "him", "her", "his", "hers", "its",
    }
)

# Hedging phrases that signal uncertainty.
_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bi think\b", re.IGNORECASE),
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bnot sure\b", re.IGNORECASE),
    re.compile(r"\bmight\b", re.IGNORECASE),
    re.compile(r"\bcould be\b", re.IGNORECASE),
    re.compile(r"\bi believe\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bi('m| am) not certain\b", re.IGNORECASE),
]


def _retrieval_relevance(results: list[RetrievalResult], top_k: int = 5) -> float:
    """Mean reranker score for the top-k retrieved chunks."""
    if not results:
        return 0.0
    scores = [rr.score for rr in results[:top_k]]
    return float(sum(scores) / len(scores))


def _completeness(query: str, response: str) -> float:
    """Fraction of meaningful query keywords present in the response."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if not keywords:
        return 1.0  # no meaningful keywords → assume complete
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw in response_lower)
    return hits / len(keywords)


def _uncertainty_penalty(response: str) -> float:
    """1 − min(1, hedge_count / 5). More hedges → lower penalty score."""
    hedge_count = sum(
        1 for pattern in _HEDGE_PATTERNS if pattern.search(response)
    )
    return 1.0 - min(1.0, hedge_count / 5.0)


def score_confidence(
    query: str,
    response: str,
    retrieval_results: list[RetrievalResult],
) -> float:
    """Composite confidence score: equally-weighted average of three signals.

    Args:
        query:             Original user query.
        response:          Generated response text.
        retrieval_results: Top-k results from the RAG pipeline.

    Returns:
        float in [0.0, 1.0].
    """
    relevance = _retrieval_relevance(retrieval_results)
    completeness = _completeness(query, response)
    penalty = _uncertainty_penalty(response)

    return float((relevance + completeness + penalty) / 3.0)
