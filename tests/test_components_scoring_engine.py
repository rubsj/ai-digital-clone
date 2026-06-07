"""Tests for src/components/scoring_engine.py.

HHEMGroundednessScorer is mocked — no real model loads or API calls.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.components.scoring_engine import ScoringEngine, Scores
from src.evaluation.groundedness_scorer import HHEMGroundednessScorer
from src.schemas import EmailMessage, KnowledgeChunk, RetrievalResult, StyleProfile
from src.style.feature_extractor import extract_features

_TECHY = (
    "The kernel scheduler uses a spinlock to protect the run queue. Because the "
    "interrupt handler can preempt the context switch, we must disable IRQs. "
    "The mutex avoids the deadlock between the two driver threads."
)


def _chunks(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=KnowledgeChunk(
                content=f"Kernel scheduling and spinlocks, chunk {i}.",
                source_topic="Operating Systems",
                source_field="computer_science",
                chunk_index=i,
            ),
            score=0.8 - i * 0.1,
            rank=i,
        )
        for i in range(n)
    ]


def _profile_matching(response: str) -> StyleProfile:
    """Profile whose vector equals the response's feature vector → style ~1.0."""
    feats = extract_features(
        EmailMessage(
            sender="x",
            subject="",
            body=response,
            timestamp=__import__("datetime").datetime(2000, 1, 1),
            message_id="x",
            quote_ratio=0.0,
        )
    )
    return StyleProfile(
        leader_name="Test",
        features=feats,
        style_vector=feats.to_vector(),
        email_count=10,
    )


def _mock_hhem_scorer(groundedness: float = 0.85) -> HHEMGroundednessScorer:
    """Return a pre-constructed mock HHEMGroundednessScorer."""
    scorer = MagicMock(spec=HHEMGroundednessScorer)
    scorer.score.return_value = groundedness
    return scorer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_returns_three_floats_in_range():
    engine = ScoringEngine(groundedness_scorer=_mock_hhem_scorer())
    scores = engine.score("kernel scheduling?", _TECHY, _profile_matching(_TECHY), _chunks())

    assert isinstance(scores, Scores)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_style_score_high_for_matching_profile():
    engine = ScoringEngine(groundedness_scorer=_mock_hhem_scorer())
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), _chunks())
    assert scores.style_score > 0.99


def test_groundedness_delegated_to_hhem_scorer():
    """ScoringEngine passes response and chunks to HHEMGroundednessScorer."""
    mock_scorer = _mock_hhem_scorer(groundedness=0.77)
    engine = ScoringEngine(groundedness_scorer=mock_scorer)
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), _chunks())

    assert scores.groundedness_score == 0.77
    mock_scorer.score.assert_called_once_with(_TECHY, _chunks())


def test_groundedness_empty_response_is_zero():
    """Empty response returns 0.0 without calling the scorer."""
    mock_scorer = _mock_hhem_scorer(groundedness=0.0)
    mock_scorer.score.return_value = 0.0
    engine = ScoringEngine(groundedness_scorer=mock_scorer)
    scores = engine.score("q", "", _profile_matching(_TECHY), _chunks())
    assert scores.groundedness_score == 0.0


def test_groundedness_no_chunks_is_zero():
    """No chunks returns 0.0 without calling scorer with real inference."""
    mock_scorer = _mock_hhem_scorer(groundedness=0.0)
    mock_scorer.score.return_value = 0.0
    engine = ScoringEngine(groundedness_scorer=mock_scorer)
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), [])
    assert scores.groundedness_score == 0.0


def test_confidence_in_range_with_hedging():
    engine = ScoringEngine(groundedness_scorer=_mock_hhem_scorer())
    hedged = "I think maybe the kernel possibly uses a spinlock, but I'm not certain."
    scores = engine.score("kernel spinlock?", hedged, _profile_matching(hedged), _chunks())
    assert 0.0 <= scores.confidence_score <= 1.0


def test_score_latency_under_500ms():
    engine = ScoringEngine(groundedness_scorer=_mock_hhem_scorer())
    profile, chunks = _profile_matching(_TECHY), _chunks()
    start = time.perf_counter()
    engine.score("kernel scheduling?", _TECHY, profile, chunks)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"scoring took {elapsed:.3f}s (budget 500ms; HHEM mocked)"


def test_hhem_scorer_loaded_at_construction():
    """ScoringEngine with no injected scorer constructs HHEMGroundednessScorer."""
    with patch(
        "src.components.scoring_engine.HHEMGroundednessScorer",
        return_value=_mock_hhem_scorer(),
    ) as MockScorer:
        engine = ScoringEngine()
        MockScorer.assert_called_once()
        assert engine._gscorer is MockScorer.return_value
