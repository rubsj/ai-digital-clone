"""Tests for src/evaluation/confidence_scorer.py.

Pure math — no API calls, no mocking needed.
"""

from __future__ import annotations

import pytest

from src.evaluation.confidence_scorer import (
    _completeness,
    _retrieval_relevance,
    _uncertainty_penalty,
    score_confidence,
)
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_result(score: float = 0.8, rank: int = 0) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content="kernel memory management",
        source_topic="Linux Kernel",
        source_field="computer_science",
        chunk_index=rank,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


# ---------------------------------------------------------------------------
# _retrieval_relevance
# ---------------------------------------------------------------------------


def test_retrieval_relevance_empty():
    assert _retrieval_relevance([]) == 0.0


def test_retrieval_relevance_single():
    assert _retrieval_relevance([_make_result(score=0.6)]) == pytest.approx(0.6)


def test_retrieval_relevance_average():
    results = [_make_result(score=s) for s in [0.8, 0.6, 0.4]]
    assert _retrieval_relevance(results) == pytest.approx(0.6)


def test_retrieval_relevance_top_k():
    # Only first 2 used when top_k=2
    results = [_make_result(score=s) for s in [1.0, 1.0, 0.0]]
    assert _retrieval_relevance(results, top_k=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _completeness
# ---------------------------------------------------------------------------


def test_completeness_perfect_match():
    # All meaningful keywords appear in response
    assert _completeness("memory management kernel", "memory management kernel works") == pytest.approx(1.0)


def test_completeness_no_keywords():
    # Query has only stopwords
    assert _completeness("the is a an", "some response here") == pytest.approx(1.0)


def test_completeness_no_match():
    assert _completeness("memory kernel allocation", "nothing relevant here") == pytest.approx(0.0)


def test_completeness_partial():
    result = _completeness("memory kernel allocation scheduler", "memory and kernel are discussed")
    assert 0.0 < result < 1.0


def test_completeness_case_insensitive():
    assert _completeness("Memory Kernel", "memory kernel are important") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _uncertainty_penalty
# ---------------------------------------------------------------------------


def test_uncertainty_penalty_no_hedges():
    assert _uncertainty_penalty("The kernel uses slab allocators.") == pytest.approx(1.0)


def test_uncertainty_penalty_single_hedge():
    result = _uncertainty_penalty("I think the kernel uses slab allocators.")
    assert result == pytest.approx(0.8)  # 1 - 1/5


def test_uncertainty_penalty_five_hedges_max():
    text = "I think maybe possibly not sure I believe this might be correct."
    # ≥5 hedges → penalty = 0.0
    assert _uncertainty_penalty(text) == pytest.approx(0.0)


def test_uncertainty_penalty_saturates_at_zero():
    # More than 5 hedges still floors at 0.0
    text = "I think maybe possibly not sure I believe might could be perhaps I'm not certain."
    assert _uncertainty_penalty(text) == pytest.approx(0.0)


def test_uncertainty_penalty_case_insensitive():
    assert _uncertainty_penalty("MAYBE the kernel is right.") == pytest.approx(0.8)


def test_uncertainty_penalty_in_range():
    for text in ["", "I think maybe possibly.", "Definitely correct."]:
        r = _uncertainty_penalty(text)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# score_confidence
# ---------------------------------------------------------------------------


def test_score_confidence_returns_float():
    result = score_confidence("memory kernel", "memory kernel allocators", [_make_result()])
    assert isinstance(result, float)


def test_score_confidence_in_range():
    result = score_confidence("memory kernel", "memory kernel allocators", [_make_result()])
    assert 0.0 <= result <= 1.0


def test_score_confidence_high_on_good_inputs():
    # All three signals should be high: good reranker score, keyword coverage, no hedges
    results = [_make_result(score=0.9) for _ in range(5)]
    score = score_confidence(
        "memory kernel",
        "memory kernel slab allocators are used extensively",
        results,
    )
    assert score > 0.7


def test_score_confidence_low_on_hedging():
    results = [_make_result(score=0.5)]
    score = score_confidence(
        "memory kernel allocation",
        "I think maybe possibly not sure I believe this could be right.",
        results,
    )
    assert score < 0.5


def test_score_confidence_empty_chunks():
    # retrieval_relevance = 0 → lower overall score
    score = score_confidence("memory", "memory is important", [])
    assert 0.0 <= score <= 1.0
    assert score < 0.7  # relevance = 0 drags it down
