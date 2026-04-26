"""Tests for src/agents/style_crew.py.

LLM creation requires OPENAI_API_KEY (set to 'dummy' via autouse fixture).
Crew.kickoff is patched for all tests that invoke generate_styled_response.
Helper functions (_build_role, _build_goal, _build_backstory) are tested
directly — they are pure functions with no I/O.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.style_crew import (
    _build_backstory,
    _build_goal,
    _build_role,
    build_style_crew,
    generate_styled_response,
)
from src.schemas import KnowledgeChunk, RetrievalResult, StyleFeatures, StyleProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch):
    """CrewAI LLM validation requires OPENAI_API_KEY at construction time."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.5,
        greeting_patterns={"hi": 0.3},
        punctuation_patterns={"dash": 0.2},
        capitalization_ratio=0.1,
        question_frequency=0.15,
        vocabulary_richness=0.6,
        common_phrases=["good point", "in the"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.6},
        formality_level=0.4,
        technical_terminology=0.5,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.5,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_profile(name: str = "Test Leader", **feature_kwargs) -> StyleProfile:
    f = _make_features(**feature_kwargs)
    return StyleProfile(
        leader_name=name,
        features=f,
        style_vector=f.to_vector(),
        email_count=50,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )


def _make_result(content: str = "kernel memory management details") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic="Linux Kernel",
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


# ---------------------------------------------------------------------------
# _build_role
# ---------------------------------------------------------------------------


def test_role_contains_leader_name():
    profile = _make_profile("Linus Torvalds")
    role = _build_role(profile)
    assert "Linus Torvalds" in role


def test_role_mentions_lkml():
    role = _build_role(_make_profile())
    assert "LKML" in role or "Linux Kernel Mailing List" in role


# ---------------------------------------------------------------------------
# _build_goal
# ---------------------------------------------------------------------------


def test_goal_contains_avg_message_length():
    profile = _make_profile(avg_message_length=0.42)
    goal = _build_goal(profile)
    assert "0.420" in goal


def test_goal_contains_formality_level():
    profile = _make_profile(formality_level=0.77)
    goal = _build_goal(profile)
    assert "0.770" in goal


def test_goal_contains_technical_depth():
    profile = _make_profile(technical_depth=0.33)
    goal = _build_goal(profile)
    assert "0.330" in goal


def test_goal_contains_vocabulary_richness():
    profile = _make_profile(vocabulary_richness=0.88)
    goal = _build_goal(profile)
    assert "0.880" in goal


def test_goal_contains_common_phrases():
    profile = _make_profile(common_phrases=["unique-phrase-xyz"])
    goal = _build_goal(profile)
    assert "unique-phrase-xyz" in goal


# ---------------------------------------------------------------------------
# _build_backstory
# ---------------------------------------------------------------------------


def test_backstory_contains_code_snippet_freq():
    profile = _make_profile(code_snippet_freq=0.25)
    backstory = _build_backstory(profile)
    assert "0.250" in backstory


def test_backstory_contains_question_frequency():
    profile = _make_profile(question_frequency=0.18)
    backstory = _build_backstory(profile)
    assert "0.180" in backstory


def test_backstory_tone_blunt_when_low_formality():
    profile = _make_profile(formality_level=0.3)
    assert "direct and blunt" in _build_backstory(profile)


def test_backstory_tone_structured_when_high_formality():
    profile = _make_profile(formality_level=0.7)
    assert "clear and structured" in _build_backstory(profile)


# ---------------------------------------------------------------------------
# Differentiation test (core contract)
# ---------------------------------------------------------------------------


def test_differentiation_goals_differ_between_leaders():
    """
    Torvalds and Kroah-Hartman profiles must produce distinct goal strings,
    and the distinguishing numerical feature (avg_message_length) must appear
    in each leader's goal with its own specific value.
    """
    torvalds = _make_profile("Linus Torvalds", avg_message_length=0.340)
    kh = _make_profile("Greg Kroah-Hartman", avg_message_length=0.166)

    t_goal = _build_goal(torvalds)
    k_goal = _build_goal(kh)

    # (a) baseline sanity: goals are distinct
    assert t_goal != k_goal

    # (b) Torvalds-specific value appears in Torvalds goal, not in KH goal
    assert "0.340" in t_goal
    assert "0.340" not in k_goal

    # (b) KH-specific value appears in KH goal, not in Torvalds goal
    assert "0.166" in k_goal
    assert "0.166" not in t_goal


# ---------------------------------------------------------------------------
# build_style_crew
# ---------------------------------------------------------------------------


def test_crew_has_one_agent():
    profile = _make_profile()
    crew = build_style_crew(profile, [_make_result()], "How does memory work?")
    assert len(crew.agents) == 1


def test_crew_has_one_task():
    profile = _make_profile()
    crew = build_style_crew(profile, [_make_result()], "How does memory work?")
    assert len(crew.tasks) == 1


def test_task_description_contains_query():
    profile = _make_profile()
    query = "What is the buddy allocator?"
    crew = build_style_crew(profile, [_make_result()], query)
    assert query in crew.tasks[0].description


def test_task_description_contains_chunk_content():
    profile = _make_profile()
    rr = _make_result("slab allocator details here")
    crew = build_style_crew(profile, [rr], "test query")
    assert "slab allocator details here" in crew.tasks[0].description


def test_crew_caps_chunks_at_max():
    profile = _make_profile()
    chunks = [_make_result(f"chunk {i}") for i in range(10)]
    crew = build_style_crew(profile, chunks, "test query")
    # Only first 5 chunks should appear
    desc = crew.tasks[0].description
    assert "chunk 4" in desc
    assert "chunk 5" not in desc


def test_agent_goal_embedded_in_crew():
    profile = _make_profile("Linus Torvalds", avg_message_length=0.34)
    crew = build_style_crew(profile, [_make_result()], "test")
    assert "Linus Torvalds" in crew.agents[0].goal
    assert "0.340" in crew.agents[0].goal


# ---------------------------------------------------------------------------
# generate_styled_response
# ---------------------------------------------------------------------------


def test_generate_styled_response_returns_raw_string():
    profile = _make_profile()
    mock_output = MagicMock()
    mock_output.raw = "The memory subsystem manages page frames."

    with patch("src.agents.style_crew.Crew.kickoff", return_value=mock_output):
        result = generate_styled_response(profile, [_make_result()], "memory question")

    assert result == "The memory subsystem manages page frames."


def test_generate_styled_response_non_empty():
    profile = _make_profile()
    mock_output = MagicMock()
    mock_output.raw = "Some non-empty styled response."

    with patch("src.agents.style_crew.Crew.kickoff", return_value=mock_output):
        result = generate_styled_response(profile, [_make_result()], "query")

    assert len(result) > 0


def test_generate_styled_response_calls_kickoff_once():
    profile = _make_profile()
    mock_output = MagicMock()
    mock_output.raw = "response"

    with patch("src.agents.style_crew.Crew.kickoff", return_value=mock_output) as mock_kickoff:
        generate_styled_response(profile, [_make_result()], "query")

    mock_kickoff.assert_called_once()
