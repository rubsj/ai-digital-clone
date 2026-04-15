"""Tests for src/evaluation/groundedness_scorer.py.

embed_openai is mocked throughout — no real API calls.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.evaluation.groundedness_scorer import _cosine, _split_sentences, score_groundedness
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _unit_vec(dim: int = 1536, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_chunk(content: str = "kernel memory management", seed: int = 0) -> KnowledgeChunk:
    return KnowledgeChunk(
        content=content,
        source_topic="Linux Kernel",
        source_field="computer_science",
        chunk_index=0,
        embedding=_unit_vec(seed=seed),
    )


def _make_result(content: str = "kernel memory management", seed: int = 0) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(content=content, seed=seed), score=0.9, rank=0)


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "First sentence. Second sentence. Third sentence."
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_exclamation():
    sentences = _split_sentences("Great idea! Let me check the code.")
    assert len(sentences) == 2


def test_split_sentences_filters_short():
    # "Yes." is < MIN_SENTENCE_CHARS, should be filtered
    sentences = _split_sentences("Yes. This is a much longer sentence with meaning.")
    assert len(sentences) == 1
    assert "longer" in sentences[0]


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_single_word():
    # Single word below threshold
    assert _split_sentences("Hello.") == []


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


def test_cosine_identical():
    v = _unit_vec()
    assert _cosine(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_zero_a():
    a = np.zeros(5, dtype=np.float32)
    b = _unit_vec(dim=5)
    assert _cosine(a, b) == 0.0


def test_cosine_zero_b():
    a = _unit_vec(dim=5)
    b = np.zeros(5, dtype=np.float32)
    assert _cosine(a, b) == 0.0


def test_cosine_result_in_range():
    for seed in range(10):
        a = _unit_vec(dim=16, seed=seed)
        b = _unit_vec(dim=16, seed=seed + 100)
        assert 0.0 <= _cosine(a, b) <= 1.0


# ---------------------------------------------------------------------------
# score_groundedness — edge cases
# ---------------------------------------------------------------------------


def test_score_groundedness_empty_response():
    result = score_groundedness("", [_make_result()])
    assert result == 0.0


def test_score_groundedness_empty_chunks():
    result = score_groundedness("This is a meaningful response sentence.", [])
    assert result == 0.0


def test_score_groundedness_both_empty():
    assert score_groundedness("", []) == 0.0


def test_score_groundedness_returns_float():
    sentence_vec = _unit_vec(seed=1)
    chunk_vec = _unit_vec(seed=1)  # identical → max sim = 1.0

    with patch("src.evaluation.groundedness_scorer.embed_openai", return_value=[sentence_vec]):
        result = score_groundedness(
            "This is a long enough sentence to pass the filter.",
            [_make_result()],
        )
    assert isinstance(result, float)


def test_score_groundedness_in_range():
    sentence_vec = _unit_vec(seed=5)
    chunk_vec = _unit_vec(seed=5)

    with patch("src.evaluation.groundedness_scorer.embed_openai", return_value=[sentence_vec]):
        result = score_groundedness(
            "The kernel manages memory using slab allocators in the core.",
            [_make_result()],
        )
    assert 0.0 <= result <= 1.0


def test_score_groundedness_identical_vecs_near_one():
    """When sentence embedding == chunk embedding, score should be ~1.0."""
    v = _unit_vec(seed=42)
    chunk = _make_chunk()
    chunk = chunk.model_copy(update={"embedding": v})
    rr = RetrievalResult(chunk=chunk, score=0.9, rank=0)

    with patch("src.evaluation.groundedness_scorer.embed_openai", return_value=[v]):
        result = score_groundedness(
            "The kernel manages memory using slab allocators correctly.",
            [rr],
        )
    assert result == pytest.approx(1.0, abs=1e-4)


def test_score_groundedness_missing_chunk_embedding_triggers_batch():
    """Chunks with embedding=None should be batch-embedded (one extra embed_openai call)."""
    chunk = KnowledgeChunk(
        content="kernel memory management details",
        source_topic="Linux",
        source_field="cs",
        chunk_index=0,
        embedding=None,
    )
    rr = RetrievalResult(chunk=chunk, score=0.9, rank=0)

    sentence_vec = _unit_vec(seed=1)
    chunk_vec = _unit_vec(seed=2)

    call_args = []

    def fake_embed(texts):
        call_args.append(texts)
        if len(texts) == 1 and texts[0].startswith("The"):
            return [sentence_vec]
        return [chunk_vec]

    with patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=fake_embed):
        result = score_groundedness(
            "The kernel manages memory using slab allocators in practice.",
            [rr],
        )
    # Two embed_openai calls: one for sentences, one for missing chunk
    assert len(call_args) == 2
    assert 0.0 <= result <= 1.0


def test_score_groundedness_top_k_limits_chunks():
    """Only top_k chunks are used even if more are passed."""
    v_match = _unit_vec(seed=99)
    # First chunk matches, rest do not matter when top_k=1
    chunks = [
        RetrievalResult(
            chunk=_make_chunk().model_copy(update={"embedding": v_match}),
            score=0.9,
            rank=i,
        )
        for i in range(5)
    ]

    with patch("src.evaluation.groundedness_scorer.embed_openai", return_value=[v_match]):
        result = score_groundedness(
            "The slab allocator is essential for kernel memory management.",
            chunks,
            top_k=1,
        )
    assert result == pytest.approx(1.0, abs=1e-4)
