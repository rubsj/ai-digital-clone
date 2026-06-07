"""Tests for src/components/retriever.py.

A small FAISS index with synthetic normalized embeddings is persisted to a temp
dir and loaded through the real disk path (Retriever has no test-only seams);
embed_query and cohere.ClientV2 are mocked — never touches the network.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from src.components.retriever import Retriever
from src.config import load_config
from src.rag.indexer import save_index
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normalized(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _small_index(n: int = 20):
    dim = load_config().embedding.dimension
    embeddings = _normalized(n, dim)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    metadata = [
        {
            "content": f"chunk content {i}",
            "source_topic": f"Topic {i}",
            "source_field": "computer_science",
            "chunk_index": i,
            "embedding": None,
        }
        for i in range(n)
    ]
    return index, metadata, embeddings


def _cohere_response(ranked_indices: list[int], scores: list[float]) -> MagicMock:
    response = MagicMock()
    items = []
    for idx, score in zip(ranked_indices, scores):
        item = MagicMock()
        item.index = idx
        item.relevance_score = score
        items.append(item)
    response.results = items
    return response


def _retriever_with_index(index_dir, n: int = 20) -> tuple[Retriever, np.ndarray]:
    """Persist a small index to index_dir, then load it via the real disk path."""
    index, metadata, embeddings = _small_index(n)
    save_index(index, metadata, index_dir=index_dir)
    r = Retriever(index_dir=index_dir)
    return r, embeddings


# ---------------------------------------------------------------------------
# FAISS → Cohere pipeline
# ---------------------------------------------------------------------------


@patch("src.rag.reranker.cohere.ClientV2")
def test_run_returns_top_5(mock_client_cls, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    r, embeddings = _retriever_with_index(tmp_path, 20)
    mock_client = MagicMock()
    mock_client.rerank.return_value = _cohere_response(
        list(range(5)), [0.9, 0.8, 0.7, 0.6, 0.5]
    )
    mock_client_cls.return_value = mock_client

    with patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()):
        results = r.run("any query")

    assert len(results) == 5
    assert all(isinstance(x, RetrievalResult) for x in results)


@patch("src.rag.reranker.cohere.ClientV2")
def test_run_cohere_actually_invoked(mock_client_cls, monkeypatch, tmp_path):
    """Constraint #3: assert Cohere ran — both the client call and the flag."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    r, embeddings = _retriever_with_index(tmp_path, 20)
    mock_client = MagicMock()
    mock_client.rerank.return_value = _cohere_response(
        list(range(5)), [0.9, 0.8, 0.7, 0.6, 0.5]
    )
    mock_client_cls.return_value = mock_client

    with patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()):
        r.run("any query")

    mock_client.rerank.assert_called_once()
    assert r.last_rerank_ran is True


@patch("src.rag.reranker.cohere.ClientV2")
def test_run_fallback_on_cohere_error_flags_not_ran(mock_client_cls, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    r, embeddings = _retriever_with_index(tmp_path, 20)
    mock_client = MagicMock()
    mock_client.rerank.side_effect = RuntimeError("Cohere unavailable")
    mock_client_cls.return_value = mock_client

    with patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()):
        results = r.run("any query")

    assert r.last_rerank_ran is False
    assert len(results) == 5  # graceful fallback to FAISS top-5 (ADR-002)


def test_run_missing_key_warns_and_falls_back(monkeypatch, caplog, tmp_path):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    r, embeddings = _retriever_with_index(tmp_path, 20)

    import logging

    with caplog.at_level(logging.WARNING), patch(
        "src.rag.retriever.embed_query", return_value=embeddings[0].copy()
    ):
        results = r.run("any query")

    assert r.last_rerank_ran is False
    assert len(results) == 5
    assert any("COHERE_API_KEY" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_run_without_index_raises(tmp_path):
    # Empty index_dir → nothing to load → run() must raise.
    r = Retriever(index_dir=tmp_path)
    with pytest.raises(RuntimeError):
        r.run("query")


@patch("src.rag.reranker.cohere.ClientV2")
def test_run_top_5_when_fewer_candidates(mock_client_cls, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    r, embeddings = _retriever_with_index(tmp_path, 3)
    mock_client = MagicMock()
    mock_client.rerank.return_value = _cohere_response([0, 1, 2], [0.9, 0.8, 0.7])
    mock_client_cls.return_value = mock_client

    with patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()):
        results = r.run("query")

    assert len(results) == 3


# ---------------------------------------------------------------------------
# Latency smoke (< 1s cold)
# ---------------------------------------------------------------------------


@patch("src.rag.reranker.cohere.ClientV2")
def test_run_latency_under_1s(mock_client_cls, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    r, embeddings = _retriever_with_index(tmp_path, 20)
    mock_client = MagicMock()
    mock_client.rerank.return_value = _cohere_response(
        list(range(5)), [0.9, 0.8, 0.7, 0.6, 0.5]
    )
    mock_client_cls.return_value = mock_client

    with patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()):
        start = time.perf_counter()
        r.run("any query")
        elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"retrieval took {elapsed:.3f}s (budget 1s; embed/Cohere mocked)"


# ---------------------------------------------------------------------------
# Fix-point A dedup: corpus duplicate-content entries (W2)
# ---------------------------------------------------------------------------


def _make_result(content: str, score: float, rank: int) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic="t",
        source_field="f",
        chunk_index=rank,
        embedding=None,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


def test_run_deduplicates_candidates_before_rerank(tmp_path):
    """Dedup at fix-point A: duplicate-content candidates stripped before rerank.

    Verifies:
    - duplicate-content entries are removed (keeps first = highest FAISS score)
    - relative order among surviving entries is preserved
    - rerank_with_status receives only the distinct-content pool
    """
    r, embeddings = _retriever_with_index(tmp_path, 20)

    # ranks 0 and 2 share content "alpha"; ranks 1 and 4 share content "beta"
    synthetic = [
        _make_result("alpha", 0.90, 0),
        _make_result("beta",  0.80, 1),
        _make_result("alpha", 0.70, 2),  # dup — must be dropped
        _make_result("gamma", 0.60, 3),
        _make_result("beta",  0.50, 4),  # dup — must be dropped
    ]

    captured: dict = {}

    def fake_rerank(query, candidates, model, top_n):
        captured["pool"] = list(candidates)
        deduped_top = candidates[: min(top_n, len(candidates))]
        return (
            [RetrievalResult(chunk=c.chunk, score=c.score, rank=i) for i, c in enumerate(deduped_top)],
            True,
        )

    with (
        patch("src.components.retriever.retrieve", return_value=synthetic),
        patch("src.components.retriever.rerank_with_status", side_effect=fake_rerank),
        patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()),
    ):
        r.run("test query")

    pool = captured["pool"]
    pool_contents = [c.chunk.content for c in pool]

    assert pool_contents == ["alpha", "beta", "gamma"], (
        f"expected ['alpha','beta','gamma'], got {pool_contents}"
    )
    # highest-ranked copy of each survives (rank 0 alpha, rank 1 beta)
    assert pool[0].score == pytest.approx(0.90)
    assert pool[1].score == pytest.approx(0.80)


def test_run_dedup_is_noop_for_distinct_candidates(tmp_path):
    """Dedup is a no-op when all candidates already have distinct content."""
    r, embeddings = _retriever_with_index(tmp_path, 20)

    distinct = [_make_result(f"chunk {i}", 1.0 - i * 0.1, i) for i in range(5)]

    captured: dict = {}

    def fake_rerank(query, candidates, model, top_n):
        captured["pool"] = list(candidates)
        return (
            [RetrievalResult(chunk=c.chunk, score=c.score, rank=i) for i, c in enumerate(candidates)],
            True,
        )

    with (
        patch("src.components.retriever.retrieve", return_value=distinct),
        patch("src.components.retriever.rerank_with_status", side_effect=fake_rerank),
        patch("src.rag.retriever.embed_query", return_value=embeddings[0].copy()),
    ):
        r.run("test query")

    pool = captured["pool"]
    assert len(pool) == 5
    assert [c.chunk.content for c in pool] == [f"chunk {i}" for i in range(5)]
