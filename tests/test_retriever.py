"""Tests for src/rag/retriever.py.

Uses a small real FAISS index with synthetic normalized embeddings.
embed_query is mocked to return a controlled vector.
"""

from __future__ import annotations

from unittest.mock import patch

import faiss
import numpy as np
import pytest

from src.rag.retriever import retrieve
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_normalized(n: int, dim: int = 1536, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_small_index(n: int = 10, dim: int = 1536):
    embeddings = _random_normalized(n, dim)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_correct_count():
    index, metadata, embeddings = _build_small_index(10)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("any query", index, metadata, top_n=5)

    assert len(results) == 5


def test_retrieve_returns_retrieval_result_objects():
    index, metadata, embeddings = _build_small_index(10)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("any query", index, metadata, top_n=3)

    assert all(isinstance(r, RetrievalResult) for r in results)


def test_retrieve_chunk_fields_populated():
    index, metadata, embeddings = _build_small_index(5)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("any query", index, metadata, top_n=1)

    chunk = results[0].chunk
    assert isinstance(chunk, KnowledgeChunk)
    assert chunk.source_field == "computer_science"
    assert chunk.chunk_index >= 0


def test_retrieve_scores_are_float():
    index, metadata, embeddings = _build_small_index(5)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("query", index, metadata, top_n=3)

    assert all(isinstance(r.score, float) for r in results)


def test_retrieve_nearest_neighbor_ranks_first():
    """Query vector identical to chunk 4 should be top result."""
    index, metadata, embeddings = _build_small_index(10)
    query_vec = embeddings[4].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("query", index, metadata, top_n=1)

    assert results[0].chunk.chunk_index == 4


def test_retrieve_top_n_larger_than_index_capped():
    """top_n=100 with only 5 chunks should not crash."""
    index, metadata, embeddings = _build_small_index(5)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("query", index, metadata, top_n=100)

    assert len(results) == 5


def test_retrieve_empty_index_returns_empty():
    dim = 1536
    index = faiss.IndexFlatIP(dim)
    metadata: list[dict] = []

    with patch("src.rag.retriever.embed_query", return_value=np.ones(dim, dtype=np.float32)):
        results = retrieve("query", index, metadata, top_n=5)

    assert results == []


def test_retrieve_rank_assigned_sequentially():
    index, metadata, embeddings = _build_small_index(5)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("query", index, metadata, top_n=5)

    ranks = [r.rank for r in results]
    assert ranks == list(range(len(results)))


def test_retrieve_top_n_one_returns_single_result():
    index, metadata, embeddings = _build_small_index(5)
    query_vec = embeddings[0].copy()

    with patch("src.rag.retriever.embed_query", return_value=query_vec):
        results = retrieve("query", index, metadata, top_n=1)

    assert len(results) == 1
