"""Tests for src/rag/embedder.py.

All API calls (LiteLLM, SentenceTransformers) are mocked — never calls real APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.embedder import (
    _load_cache,
    _md5,
    _save_cache,
    embed_chunks,
    embed_minilm,
    embed_openai,
    embed_query,
)
from src.schemas import KnowledgeChunk


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "hello world", idx: int = 0) -> KnowledgeChunk:
    return KnowledgeChunk(
        content=content,
        source_topic="Test",
        source_field="computer_science",
        chunk_index=idx,
    )


def _normalized_vec(dim: int = 1536) -> list[float]:
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_litellm_response(texts: list[str], dim: int = 1536) -> MagicMock:
    """Build a mock LiteLLM embedding response matching current dict format."""
    response = MagicMock()
    response.data = [{"embedding": _normalized_vec(dim)} for _ in texts]
    return response


# ---------------------------------------------------------------------------
# _md5
# ---------------------------------------------------------------------------


def test_md5_returns_string():
    assert isinstance(_md5("hello"), str)


def test_md5_deterministic():
    assert _md5("hello") == _md5("hello")


def test_md5_different_inputs_differ():
    assert _md5("hello") != _md5("world")


# ---------------------------------------------------------------------------
# _load_cache / _save_cache
# ---------------------------------------------------------------------------


def test_load_cache_missing_file_returns_empty(tmp_path):
    cache = _load_cache(tmp_path / "nonexistent.json")
    assert cache == {}


def test_save_and_load_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    data = {"abc": [1.0, 2.0, 3.0]}
    _save_cache(data, path)
    loaded = _load_cache(path)
    assert loaded == data


def test_save_cache_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "cache.json"
    _save_cache({"k": [1.0]}, path)
    assert path.exists()


def test_load_cache_corrupt_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not valid json{{{")
    assert _load_cache(path) == {}


# ---------------------------------------------------------------------------
# embed_openai
# ---------------------------------------------------------------------------


@patch("src.rag.embedder.litellm")
def test_embed_openai_returns_correct_count(mock_litellm, tmp_path):
    texts = ["text one", "text two", "text three"]
    mock_litellm.embedding.return_value = _make_litellm_response(texts)

    results = embed_openai(texts, cache_path=tmp_path / "cache.json")

    assert len(results) == 3


@patch("src.rag.embedder.litellm")
def test_embed_openai_returns_numpy_arrays(mock_litellm, tmp_path):
    texts = ["hello"]
    mock_litellm.embedding.return_value = _make_litellm_response(texts)

    results = embed_openai(texts, cache_path=tmp_path / "cache.json")

    assert isinstance(results[0], np.ndarray)


@patch("src.rag.embedder.litellm")
def test_embed_openai_normalized(mock_litellm, tmp_path):
    texts = ["check normalization"]
    mock_litellm.embedding.return_value = _make_litellm_response(texts)

    results = embed_openai(texts, cache_path=tmp_path / "cache.json")

    norm = np.linalg.norm(results[0])
    assert abs(norm - 1.0) < 1e-5


@patch("src.rag.embedder.litellm")
def test_embed_openai_caches_result(mock_litellm, tmp_path):
    texts = ["cached text"]
    mock_litellm.embedding.return_value = _make_litellm_response(texts)

    cache_path = tmp_path / "cache.json"
    embed_openai(texts, cache_path=cache_path)
    embed_openai(texts, cache_path=cache_path)

    # API called only once despite two embed_openai calls
    assert mock_litellm.embedding.call_count == 1


@patch("src.rag.embedder.litellm")
def test_embed_openai_batch_splitting(mock_litellm, tmp_path):
    """250 texts / batch_size=100 → 3 API calls."""
    texts = [f"text_{i}" for i in range(250)]

    def side_effect(model, input):
        return _make_litellm_response(input)

    mock_litellm.embedding.side_effect = side_effect

    embed_openai(texts, cache_path=tmp_path / "cache.json", batch_size=100)

    assert mock_litellm.embedding.call_count == 3


@patch("src.rag.embedder.litellm")
def test_embed_openai_empty_input(mock_litellm, tmp_path):
    results = embed_openai([], cache_path=tmp_path / "cache.json")
    assert results == []
    mock_litellm.embedding.assert_not_called()


# ---------------------------------------------------------------------------
# embed_minilm
# ---------------------------------------------------------------------------


@patch("src.rag.embedder._get_minilm")
def test_embed_minilm_returns_correct_count(mock_get, tmp_path):
    texts = ["alpha", "beta", "gamma"]
    mock_model = MagicMock()
    vecs = np.random.randn(3, 384).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    mock_model.encode.return_value = vecs
    mock_get.return_value = mock_model

    results = embed_minilm(texts, cache_path=tmp_path / "cache.json")
    assert len(results) == 3


@patch("src.rag.embedder._get_minilm")
def test_embed_minilm_caches_result(mock_get, tmp_path):
    texts = ["cached minilm"]
    mock_model = MagicMock()
    vecs = np.ones((1, 384), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    mock_model.encode.return_value = vecs
    mock_get.return_value = mock_model

    cache_path = tmp_path / "cache.json"
    embed_minilm(texts, cache_path=cache_path)
    embed_minilm(texts, cache_path=cache_path)

    assert mock_model.encode.call_count == 1


@patch("src.rag.embedder._get_minilm")
def test_embed_minilm_empty_input(mock_get, tmp_path):
    results = embed_minilm([], cache_path=tmp_path / "cache.json")
    assert results == []
    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


@patch("src.rag.embedder.litellm")
def test_embed_chunks_sets_embedding(mock_litellm, tmp_path):
    chunk = _make_chunk("some content")
    mock_litellm.embedding.return_value = _make_litellm_response(["some content"])

    result = embed_chunks([chunk], provider="openai", cache_path=tmp_path / "c.json")

    assert len(result) == 1
    assert result[0].embedding is not None
    assert isinstance(result[0].embedding, np.ndarray)


@patch("src.rag.embedder.litellm")
def test_embed_chunks_returns_new_objects(mock_litellm, tmp_path):
    """embed_chunks must use model_copy — original chunk unchanged."""
    chunk = _make_chunk("content")
    mock_litellm.embedding.return_value = _make_litellm_response(["content"])

    result = embed_chunks([chunk], provider="openai", cache_path=tmp_path / "c.json")

    assert chunk.embedding is None  # original unchanged
    assert result[0].embedding is not None


@patch("src.rag.embedder.litellm")
def test_embed_chunks_empty_input(mock_litellm, tmp_path):
    result = embed_chunks([], provider="openai", cache_path=tmp_path / "c.json")
    assert result == []


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


@patch("src.rag.embedder.litellm")
def test_embed_query_returns_single_array(mock_litellm, tmp_path):
    mock_litellm.embedding.return_value = _make_litellm_response(["query"])

    result = embed_query("query", provider="openai", cache_path=tmp_path / "q.json")

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


@patch("src.rag.embedder.litellm")
def test_embed_query_normalized(mock_litellm, tmp_path):
    mock_litellm.embedding.return_value = _make_litellm_response(["query"])

    result = embed_query("query", provider="openai", cache_path=tmp_path / "q.json")

    assert abs(np.linalg.norm(result) - 1.0) < 1e-5
