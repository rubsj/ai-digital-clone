"""Tests for src/evaluation/groundedness_scorer.py (HHEM-2.1-Open, ADR-020).

The HHEM model is mocked throughout — no real model loads or API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.evaluation.groundedness_scorer import HHEMGroundednessScorer, _split_sentences, score_groundedness
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "kernel memory management") -> KnowledgeChunk:
    return KnowledgeChunk(
        content=content,
        source_topic="Linux Kernel",
        source_field="computer_science",
        chunk_index=0,
    )


def _make_result(content: str = "kernel memory management") -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(content=content), score=0.9, rank=0)


def _mock_scorer(scores: list[float] | None = None) -> HHEMGroundednessScorer:
    """Return an HHEMGroundednessScorer with a mocked model.

    predict() returns a tensor of the given scores (cycled if needed),
    or [0.8] by default.
    """
    scorer = HHEMGroundednessScorer.__new__(HHEMGroundednessScorer)
    mock_model = MagicMock()

    if scores is None:
        scores = [0.8]

    def _predict(pairs):
        n = len(pairs)
        repeated = (scores * ((n // len(scores)) + 1))[:n]
        return torch.tensor(repeated, dtype=torch.float32)

    mock_model.predict.side_effect = _predict
    scorer._model = mock_model
    return scorer


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    sentences = _split_sentences("First sentence. Second sentence. Third sentence.")
    assert len(sentences) == 3


def test_split_sentences_exclamation():
    sentences = _split_sentences("Great idea! Let me check the code.")
    assert len(sentences) == 2


def test_split_sentences_filters_short():
    sentences = _split_sentences("Yes. This is a much longer sentence with meaning.")
    assert len(sentences) == 1
    assert "longer" in sentences[0]


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_single_word():
    assert _split_sentences("Hello.") == []


# ---------------------------------------------------------------------------
# HHEMGroundednessScorer.score — edge cases (no model needed)
# ---------------------------------------------------------------------------


def test_score_empty_response():
    scorer = _mock_scorer()
    assert scorer.score("", [_make_result()]) == 0.0


def test_score_empty_chunks():
    scorer = _mock_scorer()
    assert scorer.score("This is a meaningful response sentence.", []) == 0.0


def test_score_both_empty():
    scorer = _mock_scorer()
    assert scorer.score("", []) == 0.0


def test_score_response_below_min_chars():
    scorer = _mock_scorer()
    # "Yes." is shorter than _MIN_SENTENCE_CHARS
    assert scorer.score("Yes.", [_make_result()]) == 0.0


# ---------------------------------------------------------------------------
# HHEMGroundednessScorer.score — aggregation shape
# ---------------------------------------------------------------------------


def test_score_returns_float():
    scorer = _mock_scorer(scores=[0.7])
    result = scorer.score(
        "The kernel manages memory using slab allocators in the core.",
        [_make_result()],
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_max_over_chunks_single_sentence():
    """Single sentence: per-sentence score = max over chunk scores."""
    scorer = _mock_scorer(scores=[0.9, 0.4, 0.6])
    chunks = [_make_result(f"chunk {i}") for i in range(3)]
    result = scorer.score(
        "The slab allocator manages kernel memory efficiently.",
        chunks,
    )
    assert result == pytest.approx(0.9, abs=1e-4)


def test_score_mean_over_sentences():
    """Two sentences: result = mean of per-sentence maxima."""
    scorer = _mock_scorer(scores=[0.8])
    chunks = [_make_result()]
    result = scorer.score(
        "The kernel manages memory. Interrupts are handled by the scheduler.",
        chunks,
    )
    # Each sentence gets predict([0.8]) → max=0.8. Mean([0.8, 0.8]) = 0.8
    assert result == pytest.approx(0.8, abs=1e-4)


def test_score_top_k_limits_chunks():
    """Only the first top_k chunks are used."""
    scorer = _mock_scorer(scores=[0.9])
    chunks = [_make_result(f"chunk {i}") for i in range(10)]

    # top_k=2 → predict called with 2 pairs per sentence
    scorer.score(
        "The slab allocator is essential for kernel memory management.",
        chunks,
        top_k=2,
    )
    # Each predict call should have received exactly 2 pairs
    for call_args in scorer._model.predict.call_args_list:
        pairs = call_args[0][0]
        assert len(pairs) == 2


def test_score_high_when_all_chunks_match():
    """If all chunk scores are near 1.0, groundedness_score is near 1.0."""
    scorer = _mock_scorer(scores=[0.99])
    result = scorer.score(
        "The kernel manages memory using slab allocators correctly.",
        [_make_result()],
    )
    assert result == pytest.approx(0.99, abs=1e-4)


def test_score_low_when_all_chunks_mismatch():
    """If all chunk scores are near 0.0, groundedness_score is near 0.0."""
    scorer = _mock_scorer(scores=[0.02])
    result = scorer.score(
        "The kernel manages memory using slab allocators correctly.",
        [_make_result()],
    )
    assert result == pytest.approx(0.02, abs=1e-4)


# ---------------------------------------------------------------------------
# score_groundedness convenience wrapper
# ---------------------------------------------------------------------------


def test_score_groundedness_passes_through_to_scorer():
    scorer = _mock_scorer(scores=[0.75])
    result = score_groundedness(
        "The kernel uses slab allocators for memory management.",
        [_make_result()],
        scorer=scorer,
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_groundedness_empty_response():
    scorer = _mock_scorer()
    assert score_groundedness("", [_make_result()], scorer=scorer) == 0.0


def test_score_groundedness_empty_chunks():
    scorer = _mock_scorer()
    assert score_groundedness("This is a sentence.", [], scorer=scorer) == 0.0
