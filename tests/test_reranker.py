"""Tests for src/rag/reranker.py.

All Cohere API calls are mocked — never calls the real API.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.reranker import rerank
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_result(idx: int, score: float = 0.5) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=f"Content of chunk {idx}",
        source_topic=f"Topic {idx}",
        source_field="computer_science",
        chunk_index=idx,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=idx)


def _make_cohere_response(ranked_indices: list[int], scores: list[float]) -> MagicMock:
    """Build a mock Cohere rerank response."""
    response = MagicMock()
    items = []
    for idx, score in zip(ranked_indices, scores):
        item = MagicMock()
        item.index = idx
        item.relevance_score = score
        items.append(item)
    response.results = items
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_reduces_to_top_n(mock_client_cls):
    results = [_make_result(i) for i in range(20)]
    mock_client = MagicMock()
    mock_client.rerank.return_value = _make_cohere_response(
        list(range(5)), [0.9, 0.8, 0.7, 0.6, 0.5]
    )
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=5)

    assert len(reranked) == 5


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_reassigns_ranks(mock_client_cls):
    results = [_make_result(i) for i in range(10)]
    mock_client = MagicMock()
    mock_client.rerank.return_value = _make_cohere_response(
        [7, 3, 1, 9, 0], [0.95, 0.85, 0.75, 0.65, 0.55]
    )
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=5)

    assert [r.rank for r in reranked] == [0, 1, 2, 3, 4]


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_maps_back_correct_chunks(mock_client_cls):
    results = [_make_result(i) for i in range(10)]
    mock_client = MagicMock()
    # Cohere says index 7 is most relevant
    mock_client.rerank.return_value = _make_cohere_response([7, 3], [0.9, 0.8])
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=2)

    assert reranked[0].chunk.chunk_index == 7
    assert reranked[1].chunk.chunk_index == 3


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_uses_cohere_scores(mock_client_cls):
    results = [_make_result(i) for i in range(5)]
    mock_client = MagicMock()
    mock_client.rerank.return_value = _make_cohere_response([0, 1], [0.99, 0.77])
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=2)

    assert abs(reranked[0].score - 0.99) < 1e-6
    assert abs(reranked[1].score - 0.77) < 1e-6


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_fallback_on_api_error(mock_client_cls):
    """API raises → fallback returns original top-5 with re-assigned ranks."""
    results = [_make_result(i, score=float(10 - i)) for i in range(20)]
    mock_client = MagicMock()
    mock_client.rerank.side_effect = RuntimeError("Cohere unavailable")
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=5)

    assert len(reranked) == 5
    assert [r.rank for r in reranked] == [0, 1, 2, 3, 4]
    # Fallback preserves original order
    assert reranked[0].chunk.chunk_index == 0


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_empty_results_returns_empty(mock_client_cls):
    reranked = rerank("query", [], top_n=5)
    assert reranked == []
    mock_client_cls.assert_not_called()


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_top_n_larger_than_results(mock_client_cls):
    """top_n=10 with only 3 results → returns at most 3."""
    results = [_make_result(i) for i in range(3)]
    mock_client = MagicMock()
    mock_client.rerank.return_value = _make_cohere_response([0, 1, 2], [0.9, 0.8, 0.7])
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=10)

    assert len(reranked) == 3


@patch("src.rag.reranker.cohere.ClientV2")
def test_rerank_returns_retrieval_result_objects(mock_client_cls):
    results = [_make_result(i) for i in range(5)]
    mock_client = MagicMock()
    mock_client.rerank.return_value = _make_cohere_response(
        [0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]
    )
    mock_client_cls.return_value = mock_client

    reranked = rerank("query", results, top_n=5)

    assert all(isinstance(r, RetrievalResult) for r in reranked)
