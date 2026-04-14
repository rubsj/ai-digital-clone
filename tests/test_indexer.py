"""Tests for src/rag/indexer.py.

No API calls — uses synthetic normalized embeddings.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from src.rag.indexer import (
    _validate_norms,
    build_index,
    load_index,
    save_index,
)
from src.schemas import KnowledgeChunk


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_chunk(idx: int, embedding: np.ndarray) -> KnowledgeChunk:
    return KnowledgeChunk(
        content=f"Chunk content {idx}",
        source_topic=f"Topic {idx}",
        source_field="computer_science",
        chunk_index=idx,
        embedding=embedding,
    )


def _random_normalized(n: int, dim: int = 1536, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# _validate_norms
# ---------------------------------------------------------------------------


def test_validate_norms_passes_for_normalized():
    vecs = _random_normalized(10)
    _validate_norms(vecs)  # should not raise


def test_validate_norms_raises_for_unnormalized():
    vecs = _random_normalized(5)
    vecs[2] = vecs[2] * 5.0  # unnormalize one row
    with pytest.raises(ValueError, match="not L2-normalized"):
        _validate_norms(vecs)


def test_validate_norms_raises_zero_vector():
    vecs = _random_normalized(3)
    vecs[1] = np.zeros(1536, dtype=np.float32)
    with pytest.raises(ValueError):
        _validate_norms(vecs)


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


def test_build_index_ntotal_matches_chunks():
    embeddings = _random_normalized(10)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(10)]
    index, metadata = build_index(chunks, dimension=1536)
    assert index.ntotal == 10


def test_build_index_returns_faiss_index():
    embeddings = _random_normalized(5)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(5)]
    index, _ = build_index(chunks, dimension=1536)
    assert isinstance(index, faiss.IndexFlatIP)


def test_build_index_metadata_length_matches():
    embeddings = _random_normalized(7)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(7)]
    _, metadata = build_index(chunks, dimension=1536)
    assert len(metadata) == 7


def test_build_index_metadata_no_embedding():
    embeddings = _random_normalized(3)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(3)]
    _, metadata = build_index(chunks, dimension=1536)
    for m in metadata:
        assert "embedding" not in m


def test_build_index_metadata_has_content():
    embeddings = _random_normalized(3)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(3)]
    _, metadata = build_index(chunks, dimension=1536)
    assert metadata[0]["content"] == "Chunk content 0"
    assert metadata[0]["chunk_index"] == 0


def test_build_index_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        build_index([], dimension=1536)


def test_build_index_wrong_dimension_raises():
    embeddings = _random_normalized(3, dim=384)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(3)]
    with pytest.raises(ValueError, match="shape"):
        build_index(chunks, dimension=1536)  # mismatch: 384 vs 1536


# ---------------------------------------------------------------------------
# save_index + load_index roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_ntotal(tmp_path):
    embeddings = _random_normalized(8)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(8)]
    index, metadata = build_index(chunks, dimension=1536)

    save_index(index, metadata, index_dir=tmp_path)
    loaded_index, loaded_metadata = load_index(index_dir=tmp_path)

    assert loaded_index.ntotal == 8


def test_save_load_roundtrip_metadata(tmp_path):
    embeddings = _random_normalized(4)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(4)]
    index, metadata = build_index(chunks, dimension=1536)

    save_index(index, metadata, index_dir=tmp_path)
    _, loaded_metadata = load_index(index_dir=tmp_path)

    assert len(loaded_metadata) == 4
    assert loaded_metadata[2]["chunk_index"] == 2


def test_save_creates_index_files(tmp_path):
    embeddings = _random_normalized(2)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(2)]
    index, metadata = build_index(chunks, dimension=1536)

    save_index(index, metadata, index_dir=tmp_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_save_creates_nested_dir(tmp_path):
    nested = tmp_path / "rag" / "faiss_index"
    embeddings = _random_normalized(2)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(2)]
    index, metadata = build_index(chunks, dimension=1536)

    save_index(index, metadata, index_dir=nested)

    assert nested.exists()
    assert (nested / "index.faiss").exists()


# ---------------------------------------------------------------------------
# Search correctness
# ---------------------------------------------------------------------------


def test_search_returns_nearest_neighbor(tmp_path):
    """Query vector identical to chunk 3 should rank chunk 3 first."""
    embeddings = _random_normalized(10)
    chunks = [_make_chunk(i, embeddings[i]) for i in range(10)]
    index, metadata = build_index(chunks, dimension=1536)

    query = embeddings[3:4].copy().astype(np.float32)
    faiss.normalize_L2(query)
    scores, indices = index.search(query, 1)

    assert indices[0][0] == 3
