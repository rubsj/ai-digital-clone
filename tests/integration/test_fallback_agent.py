"""Contract tests for FallbackAgent (ADR-012).

These are plumbing tests — they verify that inputs reach the built prompt and
that the mocked LLM output parses to a valid FallbackResponse. Leader-voice
quality, redirection relevance, and trigger-appropriate tone are real-LLM
behavior measured on Day 12, not provable under a mock.

The one real-behavior test is the failsafe: mocking the LLM to raise and
asserting the templated FallbackResponse is returned (no exception escapes).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.fallback_agent import (
    FallbackAgent,
    _FallbackDraft,
    _build_backstory,
    _build_goal,
    _build_role,
    _build_task_description,
)
from src.schemas import FallbackResponse, KnowledgeChunk, RetrievalResult, StyleFeatures, StyleProfile


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")


def _make_features() -> StyleFeatures:
    return StyleFeatures(
        avg_message_length=0.5,
        greeting_patterns={"hi": 0.3},
        punctuation_patterns={"dash": 0.2},
        capitalization_ratio=0.1,
        question_frequency=0.15,
        vocabulary_richness=0.6,
        common_phrases=["good point"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.6},
        formality_level=0.4,
        technical_terminology=0.5,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.5,
    )


def _make_profile(leader_name: str = "torvalds") -> StyleProfile:
    f = _make_features()
    return StyleProfile(
        leader_name=leader_name,
        features=f,
        style_vector=f.to_vector(),
        email_count=50,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )


def _make_chunk(content: str = "buddy allocator manages physical pages", topic: str = "Memory Management") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic=topic,
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


# ---------------------------------------------------------------------------
# Prompt builders — plumbing assertions
# ---------------------------------------------------------------------------


def test_role_mentions_leader():
    assert "torvalds" in _build_role("torvalds").lower()


def test_goal_mentions_leader():
    assert "torvalds" in _build_goal("torvalds").lower()


def test_backstory_non_empty():
    assert _build_backstory("torvalds")
    assert _build_backstory("kroah_hartman")


def test_task_description_contains_query():
    desc = _build_task_description(
        query="How does virtual memory work?",
        trigger_reason="low_groundedness",
        chunks=[_make_chunk()],
        trigger_category="low_groundedness",
        groundedness_score=0.45,
        style_score=0.75,
        confidence_score=0.82,
        style_profile=None,
    )
    assert "How does virtual memory work?" in desc
    # ADR-018: trigger_category and scores are passed to FallbackAgent and written into the task.
    assert "Failure category: low_groundedness" in desc
    assert "groundedness=0.450" in desc


def test_task_description_contains_trigger_reason():
    desc = _build_task_description(
        query="q",
        trigger_reason="chunk_mismatch: response discussed slab but chunks cover buddy",
        chunks=[_make_chunk()],
        trigger_category=None,
        groundedness_score=None,
        style_score=None,
        confidence_score=None,
        style_profile=None,
    )
    assert "chunk_mismatch" in desc


def test_task_description_contains_chunk_content():
    profile = _make_profile()
    profile = profile.model_copy(update={"sample_emails": ["Be direct. Patch looks wrong."]})
    desc = _build_task_description(
        query="q",
        trigger_reason="low_groundedness",
        chunks=[_make_chunk("buddy allocator manages physical pages", "Memory Management")],
        trigger_category=None,
        groundedness_score=None,
        style_score=None,
        confidence_score=None,
        style_profile=profile,
    )
    assert "buddy allocator" in desc
    # ADR-018: style_profile is wired into the task description so the redirect is in the leader's voice.
    assert "Style examples from your own emails" in desc


def test_task_description_no_chunks():
    desc = _build_task_description(
        query="q",
        trigger_reason="r",
        chunks=[],
        trigger_category=None,
        groundedness_score=None,
        style_score=None,
        confidence_score=None,
        style_profile=None,
    )
    assert "no source material" in desc.lower()


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------


def test_crew_has_one_agent_one_task():
    crew = FallbackAgent()._build_crew(
        "q", "torvalds", "low_groundedness", [_make_chunk()],
        "low_groundedness", 0.45, 0.75, 0.82, _make_profile(),
    )
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1


def test_crew_agent_has_role_goal_backstory():
    agent = FallbackAgent()._build_crew(
        "q", "torvalds", "low_groundedness", [_make_chunk()],
        "low_groundedness", 0.45, 0.75, 0.82, _make_profile(),
    ).agents[0]
    assert agent.role and agent.goal and agent.backstory


# ---------------------------------------------------------------------------
# run — contract: output shape (LLM mocked)
# ---------------------------------------------------------------------------


def _mock_run(draft: _FallbackDraft):
    """Return patches for Crew.kickoff and instructor.from_litellm."""
    kickoff_output = MagicMock()
    kickoff_output.raw = "I can't answer this well. Try: How does the buddy allocator work?"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = draft
    return (
        patch("src.agents.fallback_agent.Crew.kickoff", return_value=kickoff_output),
        patch("src.agents.fallback_agent.instructor.from_litellm", return_value=mock_client),
    )


def test_run_returns_valid_fallback_response():
    draft = _FallbackDraft(
        acknowledgment="I can't give you a good answer on this.",
        suggested_redirections=["How does the buddy allocator work?", "What is slab allocation?"],
    )
    kickoff_p, instr_p = _mock_run(draft)
    with kickoff_p, instr_p:
        result = FallbackAgent().run(
            query="How does virtual memory work?",
            leader="torvalds",
            trigger_reason="low_groundedness",
            style_profile=_make_profile(),
            chunks=[_make_chunk()],
        )

    assert isinstance(result, FallbackResponse)
    assert result.acknowledgment  # non-empty
    assert isinstance(result.suggested_redirections, list)
    assert result.calendar_link
    # calendar mock is the real deterministic helper — 3 slots always generated
    assert len(result.available_slots) == 3
    assert result.unstyled_response


def test_run_parse_uses_temperature_zero():
    draft = _FallbackDraft(acknowledgment="Sorry.", suggested_redirections=[])
    kickoff_p, instr_p = _mock_run(draft)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = draft

    with kickoff_p, patch("src.agents.fallback_agent.instructor.from_litellm", return_value=mock_client):
        FallbackAgent().run("q", "torvalds", "low", _make_profile(), [_make_chunk()])

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is _FallbackDraft
    assert kwargs["temperature"] == 0


# ---------------------------------------------------------------------------
# Failsafe test — real behavior: LLM raises → templated FallbackResponse
# ---------------------------------------------------------------------------


def test_run_failsafe_on_llm_raise():
    """When Crew.kickoff raises, no exception escapes and a templated FallbackResponse is returned."""
    with patch("src.agents.fallback_agent.Crew.kickoff", side_effect=RuntimeError("LLM timeout")):
        result = FallbackAgent().run(
            query="How does virtual memory work?",
            leader="torvalds",
            trigger_reason="low_groundedness",
            style_profile=_make_profile(),
            chunks=[_make_chunk()],
        )

    assert isinstance(result, FallbackResponse)
    # Leader name substituted in the failsafe acknowledgment
    assert "torvalds" in result.acknowledgment.lower()
    # Calendar mock intact
    assert result.calendar_link
    assert len(result.available_slots) == 3
    # System always returns usable prose
    assert result.unstyled_response


def test_run_failsafe_on_instructor_raise():
    """When the Instructor parse raises after a successful kickoff, failsafe activates."""
    kickoff_output = MagicMock()
    kickoff_output.raw = "Some raw text."
    with (
        patch("src.agents.fallback_agent.Crew.kickoff", return_value=kickoff_output),
        patch(
            "src.agents.fallback_agent.instructor.from_litellm",
            side_effect=Exception("instructor parse error"),
        ),
    ):
        result = FallbackAgent().run(
            query="q",
            leader="kroah_hartman",
            trigger_reason="low_groundedness",
            style_profile=_make_profile("kroah_hartman"),
            chunks=[_make_chunk()],
        )

    assert isinstance(result, FallbackResponse)
    assert "kroah_hartman" in result.acknowledgment.lower()
    assert len(result.available_slots) == 3
