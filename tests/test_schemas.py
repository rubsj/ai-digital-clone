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
    CloneResponse,
    CloneState,
    EmailMessage,
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    LeaderComparison,
    RetrievalResult,
    RoutingDecision,
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
        explanation="Style matches well. Response grounded in 3 chunks.",
        flags=[],
    )
    return EvaluationResult(**(defaults | kwargs))


def _make_clone_response(**kwargs) -> CloneResponse:
    defaults = dict(
        response_text="Look, TCP/IP is basically how machines talk.",
        citations=[_make_citation()],
    )
    return CloneResponse(**(defaults | kwargs))


def _make_fallback(**kwargs) -> FallbackResponse:
    defaults = dict(
        acknowledgment="That's outside what I can answer well from the retrieved material.",
        suggested_redirections=["How does the buddy allocator work?", "What is slab allocation?"],
        calendar_link="https://cal.com/torvalds/book",
        available_slots=["2026-04-10T10:00", "2026-04-10T14:00", "2026-04-11T09:00"],
        unstyled_response="The kernel memory subsystem handles physical pages via the buddy allocator.",
    )
    return FallbackResponse(**(defaults | kwargs))


def _make_routing_decision(**kwargs) -> RoutingDecision:
    defaults = dict(
        decision="deliver",
        reasoning="Style score 0.91 and groundedness 0.68 both above targets; no flags raised.",
        trigger_reason=None,
    )
    return RoutingDecision(**(defaults | kwargs))


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


def test_email_message_quote_ratio_default():
    email = _make_email()
    assert email.quote_ratio == 0.0


def test_email_message_quote_ratio_valid():
    email = _make_email(quote_ratio=0.35)
    assert email.quote_ratio == 0.35


def test_email_message_quote_ratio_below_zero_raises():
    with pytest.raises(ValidationError):
        _make_email(quote_ratio=-0.1)


def test_email_message_quote_ratio_above_one_raises():
    with pytest.raises(ValidationError):
        _make_email(quote_ratio=1.1)


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


def test_style_profile_sample_emails_default_empty():
    profile = _make_style_profile()
    assert profile.sample_emails == []


def test_style_profile_sample_emails_populated():
    profile = _make_style_profile(sample_emails=["cleaned email one", "cleaned email two"])
    assert len(profile.sample_emails) == 2


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


def test_evaluation_result_five_fields():
    ev = _make_eval_result(flags=["low_groundedness"])
    assert set(EvaluationResult.model_fields) == {
        "style_score",
        "groundedness_score",
        "confidence_score",
        "explanation",
        "flags",
    }
    assert ev.flags == ["low_groundedness"]


def test_evaluation_result_flags_default_empty():
    ev = _make_eval_result()
    assert ev.flags == []


def test_evaluation_result_rejects_final_score():
    """extra='forbid' — a v1 caller passing final_score fails loudly (ADR-010/011)."""
    with pytest.raises(ValidationError):
        EvaluationResult(
            style_score=0.9,
            groundedness_score=0.8,
            confidence_score=0.7,
            explanation="test",
            final_score=0.82,
        )


def test_evaluation_result_rejects_decision():
    with pytest.raises(ValidationError):
        EvaluationResult(
            style_score=0.9,
            groundedness_score=0.8,
            confidence_score=0.7,
            explanation="test",
            decision="deliver",
        )


def test_evaluation_result_score_out_of_range_raises():
    with pytest.raises(ValidationError):
        _make_eval_result(groundedness_score=1.5)


# ---------------------------------------------------------------------------
# CloneResponse
# ---------------------------------------------------------------------------


def test_clone_response_valid():
    cr = _make_clone_response()
    assert cr.response_text.startswith("Look")
    assert len(cr.citations) == 1


def test_clone_response_citations_default_empty():
    cr = CloneResponse(response_text="hi")
    assert cr.citations == []


def test_clone_response_roundtrip():
    cr = _make_clone_response()
    cr2 = CloneResponse.model_validate_json(cr.model_dump_json())
    assert cr2.response_text == cr.response_text
    assert cr2.citations[0].chunk_id == cr.citations[0].chunk_id


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


def test_routing_decision_deliver():
    rd = _make_routing_decision()
    assert rd.decision == "deliver"
    assert rd.reasoning
    assert rd.trigger_reason is None


def test_routing_decision_fallback():
    rd = _make_routing_decision(
        decision="fallback",
        reasoning="Groundedness 0.21 far below target; flag: low_groundedness.",
        trigger_reason="low_groundedness",
    )
    assert rd.decision == "fallback"
    assert rd.trigger_reason == "low_groundedness"


def test_routing_decision_invalid_literal_raises():
    with pytest.raises(ValidationError):
        RoutingDecision(decision="maybe", reasoning="uncertain")


def test_routing_decision_empty_reasoning_raises():
    with pytest.raises(ValidationError):
        RoutingDecision(decision="deliver", reasoning="")


def test_routing_decision_trigger_category_defaults_none():
    rd = _make_routing_decision()
    assert rd.trigger_category is None


def test_routing_decision_trigger_category_valid_literals():
    for cat in ("low_groundedness", "off_domain", "hallucination_risk", "chunk_mismatch", "empty_retrieval"):
        rd = _make_routing_decision(
            decision="fallback",
            reasoning="scores below target.",
            trigger_reason="score too low",
            trigger_category=cat,
        )
        assert rd.trigger_category == cat


def test_routing_decision_trigger_category_invalid_raises():
    with pytest.raises(ValidationError):
        RoutingDecision(
            decision="fallback",
            reasoning="scores below target.",
            trigger_category="weird_reason",
        )


# ---------------------------------------------------------------------------
# FallbackResponse
# ---------------------------------------------------------------------------


def test_fallback_response_valid():
    fb = _make_fallback()
    assert fb.acknowledgment
    assert isinstance(fb.suggested_redirections, list)
    assert fb.calendar_link
    assert len(fb.available_slots) == 3
    assert fb.unstyled_response


def test_fallback_response_empty_redirections():
    fb = _make_fallback(suggested_redirections=[])
    assert fb.suggested_redirections == []


def test_fallback_response_v1_fields_rejected():
    """v1 fields trigger_reason / context_summary must not be silently accepted."""
    with pytest.raises((ValidationError, TypeError)):
        FallbackResponse(
            trigger_reason="low score",
            context_summary="some context",
            calendar_link="https://cal.com/x",
            available_slots=[],
            unstyled_response="",
        )


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
    assert state.chunks == []
    assert state.style_profile is None
    assert state.response_text is None
    assert state.citations == []
    assert state.evaluation is None
    assert state.routing_decision is None
    assert state.styled_response is None
    assert state.fallback_response is None


def test_clone_state_routing_decision_populated():
    state = CloneState()
    state.routing_decision = _make_routing_decision()
    assert state.routing_decision.decision == "deliver"


def test_clone_state_routing_decision_fallback():
    state = CloneState()
    state.routing_decision = _make_routing_decision(
        decision="fallback",
        reasoning="Low groundedness detected.",
        trigger_reason="low_groundedness",
    )
    assert state.routing_decision.decision == "fallback"
    assert state.routing_decision.trigger_reason == "low_groundedness"


def test_clone_state_incremental_population():
    """Simulate how DigitalCloneFlow populates state step-by-step (v2 pipeline)."""
    state = CloneState()

    # Step 1: retrieve
    state.query = "What is a kernel?"
    state.leader = "torvalds"
    state.chunks = [RetrievalResult(chunk=_make_chunk(), score=0.9, rank=0)]

    # Step 2: clone
    state.response_text = "Look, the kernel is basically the core of the OS."

    # Step 3: evaluate
    state.evaluation = _make_eval_result()

    # Step 4: route
    state.routing_decision = _make_routing_decision()

    # Step 5: finalize (deliver)
    state.styled_response = _make_styled_response()

    assert state.query == "What is a kernel?"
    assert len(state.chunks) == 1
    assert state.evaluation.style_score == 0.92
    assert isinstance(state.styled_response, StyledResponse)
