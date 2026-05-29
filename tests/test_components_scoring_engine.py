"""Tests for src/components/scoring_engine.py.

embed_openai (groundedness's only network dependency) is mocked — the
deterministic scoring math runs offline.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from src.components.scoring_engine import ScoringEngine, Scores
from src.schemas import EmailMessage, KnowledgeChunk, RetrievalResult, StyleProfile
from src.style.feature_extractor import extract_features

_TECHY = (
    "The kernel scheduler uses a spinlock to protect the run queue. Because the "
    "interrupt handler can preempt the context switch, we must disable IRQs. "
    "The mutex avoids the deadlock between the two driver threads."
)


def _chunks(n: int = 3, embedding=None) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=KnowledgeChunk(
                content=f"Kernel scheduling and spinlocks, chunk {i}.",
                source_topic="Operating Systems",
                source_field="computer_science",
                chunk_index=i,
                embedding=embedding,
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


def _fixed_embedding(dim: int = 16):
    vec = np.ones(dim, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _mock_embed_openai(texts, *args, **kwargs):
    return [_fixed_embedding() for _ in texts]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=_mock_embed_openai)
def test_score_returns_three_floats_in_range(_mock):
    engine = ScoringEngine()
    scores = engine.score("kernel scheduling?", _TECHY, _profile_matching(_TECHY), _chunks())

    assert isinstance(scores, Scores)
    for s in scores:
        assert 0.0 <= s <= 1.0


@patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=_mock_embed_openai)
def test_style_score_high_for_matching_profile(_mock):
    engine = ScoringEngine()
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), _chunks())
    # Profile vector == response feature vector → cosine ≈ 1.0
    assert scores.style_score > 0.99


@patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=_mock_embed_openai)
def test_groundedness_all_chunks_aligned(_mock):
    """All sentence/chunk embeddings identical → groundedness ≈ 1.0."""
    engine = ScoringEngine()
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), _chunks())
    assert scores.groundedness_score > 0.99


def test_groundedness_empty_response_is_zero():
    engine = ScoringEngine()
    scores = engine.score("q", "", _profile_matching(_TECHY), _chunks())
    assert scores.groundedness_score == 0.0


def test_groundedness_no_chunks_is_zero():
    engine = ScoringEngine()
    scores = engine.score("q", _TECHY, _profile_matching(_TECHY), [])
    assert scores.groundedness_score == 0.0


@patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=_mock_embed_openai)
def test_confidence_in_range_with_hedging(_mock):
    engine = ScoringEngine()
    hedged = "I think maybe the kernel possibly uses a spinlock, but I'm not certain."
    scores = engine.score("kernel spinlock?", hedged, _profile_matching(hedged), _chunks())
    assert 0.0 <= scores.confidence_score <= 1.0


@patch("src.evaluation.groundedness_scorer.embed_openai", side_effect=_mock_embed_openai)
def test_score_latency_under_500ms(_mock):
    engine = ScoringEngine()
    profile, chunks = _profile_matching(_TECHY), _chunks()
    start = time.perf_counter()
    engine.score("kernel scheduling?", _TECHY, profile, chunks)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"scoring took {elapsed:.3f}s (budget 500ms; embeddings mocked)"
