"""Tests for all 11 Pydantic schemas in src/schemas.py.

Coverage target: >= 90% of schemas.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from src.schemas import (
    Citation,
    CloneState,
    EmailMessage,
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    LeaderComparison,
    RetrievalResult,
    StyleFeatures,
    StyleProfile,
    StyledResponse,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_email(**kwargs) -> EmailMessage:
    defaults = dict(
        sender="torvalds@linux-foundation.org",
        recipients=["linux-kernel@vger.kernel.org"],
        subject="Re: [PATCH] Fix memory leak",
        body="The problem here is that we never free the buffer. " * 5,
        timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
        message_id="<abc123@kernel.org>",
        is_patch=False,
    )
    return EmailMessage(**(defaults | kwargs))


def _make_style_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.4,
        greeting_patterns={"hi": 0.1},
        punctuation_patterns={"?": 0.03, "!": 0.01},
        capitalization_ratio=0.05,
        question_frequency=0.1,
        vocabulary_richness=0.6,
        common_phrases=["the thing is", "the point is"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.3, "negative": 0.1},
        formality_level=0.3,
        technical_terminology=0.8,
        code_snippet_freq=0.2,
        quote_reply_ratio=0.15,
        patch_language={"applied": 0.4},
        technical_depth=0.7,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_style_profile(**kwargs) -> StyleProfile:
    features = _make_style_features()
    defaults = dict(
        leader_name="Linus Torvalds",
        features=features,
        style_vector=np.array([0.1] * 15, dtype=np.float64),
        email_count=250,
        last_updated=datetime(2023, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )
    return StyleProfile(**(defaults | kwargs))


def _make_chunk(**kwargs) -> KnowledgeChunk:
    defaults = dict(
        content="TCP/IP is a fundamental networking protocol.",
        source_topic="Networking",
        source_field="computer science",
        chunk_index=0,
    )
    return KnowledgeChunk(**(defaults | kwargs))


def _make_citation(**kwargs) -> Citation:
    defaults = dict(
        chunk_id="chunk_042",
        source_topic="Networking",
        text_snippet="TCP/IP is fundamental...",
        relevance_score=0.85,
    )
    return Citation(**(defaults | kwargs))


def _make_eval_result(**kwargs) -> EvaluationResult:
    defaults = dict(
        style_score=0.92,
        groundedness_score=0.75,
        confidence_score=0.80,
        final_score=round(0.4 * 0.92 + 0.4 * 0.75 + 0.2 * 0.80, 10),
        explanation="Style matches well. Response grounded in 3 chunks.",
        decision="deliver",
    )
    return EvaluationResult(**(defaults | kwargs))


def _make_fallback(**kwargs) -> FallbackResponse:
    defaults = dict(
        trigger_reason="final_score=0.60 < 0.75",
        context_summary="Query about memory management in Linux kernel.",
        calendar_link="https://cal.com/torvalds/book",
        available_slots=["2026-04-10T10:00", "2026-04-10T14:00"],
        unstyled_response=None,
    )
    return FallbackResponse(**(defaults | kwargs))


def _make_styled_response(**kwargs) -> StyledResponse:
    defaults = dict(
        query="What is TCP/IP?",
        leader="torvalds",
        response="Look, TCP/IP is basically...",
        evaluation=_make_eval_result(),
        citations=[_make_citation()],
        fallback=None,
    )
    return StyledResponse(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# EmailMessage
# ---------------------------------------------------------------------------


def test_email_message_valid():
    email = _make_email()
    assert email.sender == "torvalds@linux-foundation.org"
    assert email.is_patch is False


def test_email_message_minimal():
    email = EmailMessage(
        sender="test@example.com",
        subject="Test",
        body="This is a test email with enough words here.",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        message_id="<test@example.com>",
    )
    assert email.recipients == []
    assert email.is_patch is False


def test_email_message_is_patch_true():
    email = _make_email(is_patch=True, subject="[PATCH v2] Add feature")
    assert email.is_patch is True


# ---------------------------------------------------------------------------
# StyleFeatures
# ---------------------------------------------------------------------------


def test_style_features_valid():
    features = _make_style_features()
    assert features.avg_message_length == 0.4
    assert features.technical_terminology == 0.8


def test_style_features_out_of_range_raises():
    with pytest.raises(ValidationError):
        _make_style_features(capitalization_ratio=1.5)


def test_style_features_negative_raises():
    with pytest.raises(ValidationError):
        _make_style_features(question_frequency=-0.1)


def test_style_features_to_vector_length():
    features = _make_style_features()
    vec = features.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (15,)


def test_style_features_to_vector_all_in_range():
    features = _make_style_features()
    vec = features.to_vector()
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_style_features_to_vector_empty_dicts():
    features = _make_style_features(
        greeting_patterns={},
        punctuation_patterns={},
        reasoning_patterns={},
        sentiment_distribution={},
        patch_language={},
    )
    vec = features.to_vector()
    assert vec.shape == (15,)
    # Dict-mean for empty dict should be 0.0
    assert vec[1] == 0.0  # greeting_patterns


def test_style_features_phrase_diversity_capped_at_one():
    features = _make_style_features(common_phrases=["a"] * 30)
    vec = features.to_vector()
    assert vec[14] == 1.0  # capped at 1.0 (30/20 > 1)


# ---------------------------------------------------------------------------
# StyleProfile
# ---------------------------------------------------------------------------


def test_style_profile_valid():
    profile = _make_style_profile()
    assert profile.leader_name == "Linus Torvalds"
    assert profile.email_count == 250
    assert isinstance(profile.style_vector, np.ndarray)


def test_style_profile_alpha_out_of_range():
    with pytest.raises(ValidationError):
        _make_style_profile(alpha=1.5)


def test_style_profile_alpha_negative():
    with pytest.raises(ValidationError):
        _make_style_profile(alpha=-0.1)


def test_style_profile_serialization_roundtrip():
    profile = _make_style_profile()
    data = profile.model_dump()
    # style_vector should serialize to list[float]
    assert isinstance(data["style_vector"], list)
    assert all(isinstance(v, float) for v in data["style_vector"])
    # Deserialize back
    profile2 = StyleProfile.model_validate(data)
    assert isinstance(profile2.style_vector, np.ndarray)
    np.testing.assert_array_almost_equal(profile.style_vector, profile2.style_vector)


def test_style_profile_list_input_coerced_to_ndarray():
    profile = StyleProfile(
        leader_name="Test",
        features=_make_style_features(),
        style_vector=[0.1] * 15,  # list input
        email_count=10,
        last_updated=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(profile.style_vector, np.ndarray)


# ---------------------------------------------------------------------------
# KnowledgeChunk
# ---------------------------------------------------------------------------


def test_knowledge_chunk_without_embedding():
    chunk = _make_chunk()
    assert chunk.embedding is None


def test_knowledge_chunk_with_embedding():
    emb = np.random.rand(1536).astype(np.float64)
    chunk = _make_chunk(embedding=emb)
    assert isinstance(chunk.embedding, np.ndarray)
    assert chunk.embedding.shape == (1536,)


def test_knowledge_chunk_embedding_serialization():
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    chunk = _make_chunk(embedding=emb)
    data = chunk.model_dump()
    assert isinstance(data["embedding"], list)
    chunk2 = KnowledgeChunk.model_validate(data)
    assert isinstance(chunk2.embedding, np.ndarray)
    np.testing.assert_array_almost_equal(emb, chunk2.embedding)


def test_knowledge_chunk_embedding_none_serialization():
    chunk = _make_chunk(embedding=None)
    data = chunk.model_dump()
    assert data["embedding"] is None


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


def test_retrieval_result_valid():
    rr = RetrievalResult(chunk=_make_chunk(), score=0.87, rank=0)
    assert rr.score == 0.87
    assert rr.rank == 0


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def test_citation_valid():
    c = _make_citation()
    assert c.relevance_score == 0.85


def test_citation_relevance_out_of_range():
    with pytest.raises(ValidationError):
        Citation(
            chunk_id="x",
            source_topic="y",
            text_snippet="z",
            relevance_score=1.5,
        )


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


def test_evaluation_result_deliver():
    ev = _make_eval_result(decision="deliver")
    assert ev.decision == "deliver"


def test_evaluation_result_fallback():
    style, ground, conf = 0.5, 0.5, 0.5
    final = round(0.4 * style + 0.4 * ground + 0.2 * conf, 10)
    ev = EvaluationResult(
        style_score=style,
        groundedness_score=ground,
        confidence_score=conf,
        final_score=final,
        explanation="Low scores across all dimensions.",
        decision="fallback",
    )
    assert ev.decision == "fallback"


def test_evaluation_result_invalid_decision():
    with pytest.raises(ValidationError):
        EvaluationResult(
            style_score=0.5,
            groundedness_score=0.5,
            confidence_score=0.5,
            final_score=0.5,
            explanation="test",
            decision="maybe",  # not in Literal["deliver", "fallback"]
        )


def test_evaluation_result_formula_mismatch_raises():
    with pytest.raises(ValidationError):
        EvaluationResult(
            style_score=0.9,
            groundedness_score=0.8,
            confidence_score=0.7,
            final_score=0.5,  # wrong — should be 0.4*0.9+0.4*0.8+0.2*0.7 = 0.82
            explanation="test",
            decision="deliver",
        )


def test_evaluation_result_formula_valid():
    style, ground, conf = 0.92, 0.75, 0.80
    final = round(0.4 * style + 0.4 * ground + 0.2 * conf, 10)
    ev = _make_eval_result(
        style_score=style,
        groundedness_score=ground,
        confidence_score=conf,
        final_score=final,
    )
    assert abs(ev.final_score - 0.828) < 0.001


# ---------------------------------------------------------------------------
# FallbackResponse
# ---------------------------------------------------------------------------


def test_fallback_response_valid():
    fb = _make_fallback()
    assert fb.unstyled_response is None
    assert len(fb.available_slots) == 2


def test_fallback_response_with_unstyled():
    fb = _make_fallback(unstyled_response="Here is a grounded answer without style.")
    assert fb.unstyled_response is not None


# ---------------------------------------------------------------------------
# StyledResponse
# ---------------------------------------------------------------------------


def test_styled_response_valid():
    sr = _make_styled_response()
    assert sr.leader == "torvalds"
    assert sr.fallback is None


def test_styled_response_with_fallback():
    sr = _make_styled_response(fallback=_make_fallback())
    assert sr.fallback is not None


# ---------------------------------------------------------------------------
# LeaderComparison
# ---------------------------------------------------------------------------


def test_leader_comparison_valid():
    torvalds_sr = _make_styled_response(leader="torvalds")
    gkh_sr = _make_styled_response(leader="kroah_hartman")
    lc = LeaderComparison(
        query="What is TCP/IP?",
        torvalds=torvalds_sr,
        kroah_hartman=gkh_sr,
    )
    assert lc.query == "What is TCP/IP?"
    assert lc.torvalds.leader == "torvalds"
    assert lc.kroah_hartman.leader == "kroah_hartman"


# ---------------------------------------------------------------------------
# CloneState
# ---------------------------------------------------------------------------


def test_clone_state_defaults():
    state = CloneState()
    assert state.query == ""
    assert state.leader == ""
    assert state.retrieved_chunks == []
    assert state.styled_response == ""
    assert state.evaluation is None
    assert state.final_output is None


def test_clone_state_incremental_population():
    """Simulate how DigitalCloneFlow populates state step-by-step."""
    state = CloneState()

    # Step 1: start
    state.query = "What is a kernel?"
    state.leader = "torvalds"

    # Step 2: retrieve_knowledge
    state.retrieved_chunks = [RetrievalResult(chunk=_make_chunk(), score=0.9, rank=0)]

    # Step 3: apply_style
    state.styled_response = "Look, the kernel is basically the core of the OS."

    # Step 4: evaluate_response
    state.evaluation = _make_eval_result()

    # Step 5: deliver_response
    state.final_output = _make_styled_response()

    assert state.query == "What is a kernel?"
    assert len(state.retrieved_chunks) == 1
    assert state.evaluation.decision == "deliver"
    assert isinstance(state.final_output, StyledResponse)
