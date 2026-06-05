"""Tests for src/agents/evaluator_agent.py.

ScoringEngine, Crew.kickoff, and the Instructor client are mocked. CrewAI LLM
construction requires OPENAI_API_KEY (set to a dummy via autouse fixture).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.evaluator_agent import (
    EvaluatorAgent,
    _ReviewDraft,
    _build_backstory,
    _build_goal,
    _build_role,
    _build_task_description,
)
from src.components.scoring_engine import Scores
from src.schemas import EvaluationResult, KnowledgeChunk, RetrievalResult, StyleFeatures, StyleProfile


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
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
    return StyleFeatures(**(defaults | kwargs))


def _make_profile() -> StyleProfile:
    f = _make_features()
    return StyleProfile(
        leader_name="Test Leader",
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
# Prompt builders
# ---------------------------------------------------------------------------


def test_role_mentions_reviewer():
    assert "reviewer" in _build_role().lower()


def test_goal_and_backstory_non_empty():
    assert _build_goal()
    assert _build_backstory()


def test_task_description_contains_scores_and_inputs():
    scores = Scores(style_score=0.91, groundedness_score=0.62, confidence_score=0.83)
    desc = _build_task_description("How does memory work?", "A styled response.", scores,
                                   [_make_result("slab allocator facts")])
    assert "How does memory work?" in desc
    assert "A styled response." in desc
    assert "slab allocator facts" in desc
    assert "0.910" in desc
    assert "0.620" in desc
    assert "0.830" in desc


def test_task_description_handles_no_chunks():
    scores = Scores(0.5, 0.5, 0.5)
    desc = _build_task_description("q", "r", scores, [])
    assert "no source chunks" in desc.lower()


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------


def test_crew_has_one_agent_one_task():
    crew = EvaluatorAgent()._build_crew("q", "r", Scores(0.9, 0.6, 0.8), [_make_result()])
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1


def test_crew_agent_has_role_goal_backstory():
    agent = EvaluatorAgent()._build_crew("q", "r", Scores(0.9, 0.6, 0.8), [_make_result()]).agents[0]
    assert agent.role and agent.goal and agent.backstory


# ---------------------------------------------------------------------------
# run — full hybrid path
# ---------------------------------------------------------------------------


def _patch_run(scores: Scores, raw: str, draft: _ReviewDraft):
    kickoff_output = MagicMock()
    kickoff_output.raw = raw
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = draft
    mock_scoring = MagicMock()
    mock_scoring.score.return_value = scores
    return (
        mock_scoring,
        patch("src.agents.evaluator_agent.Crew.kickoff", return_value=kickoff_output),
        patch("src.agents.evaluator_agent.instructor.from_litellm", return_value=mock_client),
        mock_client,
    )


def test_run_assembles_evaluation_result_from_scores_and_llm():
    scores = Scores(style_score=0.91, groundedness_score=0.62, confidence_score=0.83)
    draft = _ReviewDraft(explanation="Well grounded and styled.", flags=[])
    mock_scoring, kickoff_p, instr_p, _ = _patch_run(scores, "verdict", draft)

    with kickoff_p, instr_p:
        result = EvaluatorAgent(scoring_engine=mock_scoring).run(
            "query", "response", _make_profile(), [_make_result()]
        )

    assert isinstance(result, EvaluationResult)
    assert result.style_score == 0.91
    assert result.groundedness_score == 0.62
    assert result.confidence_score == 0.83
    assert result.explanation == "Well grounded and styled."
    assert result.flags == []


def test_run_calls_scoring_engine_once():
    scores = Scores(0.9, 0.6, 0.8)
    draft = _ReviewDraft(explanation="ok", flags=[])
    mock_scoring, kickoff_p, instr_p, _ = _patch_run(scores, "verdict", draft)

    with kickoff_p, instr_p:
        EvaluatorAgent(scoring_engine=mock_scoring).run("q", "r", _make_profile(), [_make_result()])

    mock_scoring.score.assert_called_once()


def test_run_llm_parse_uses_temperature_zero():
    scores = Scores(0.9, 0.6, 0.8)
    draft = _ReviewDraft(explanation="ok", flags=["low_confidence"])
    mock_scoring, kickoff_p, instr_p, mock_client = _patch_run(scores, "verdict", draft)

    with kickoff_p, instr_p:
        EvaluatorAgent(scoring_engine=mock_scoring).run("q", "r", _make_profile(), [_make_result()])

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is _ReviewDraft
    assert kwargs["temperature"] == 0


def test_run_propagates_flags():
    scores = Scores(0.4, 0.3, 0.5)
    draft = _ReviewDraft(explanation="Weak grounding.")
    mock_scoring, kickoff_p, instr_p, _ = _patch_run(scores, "verdict", draft)

    with kickoff_p, instr_p:
        result = EvaluatorAgent(scoring_engine=mock_scoring).run(
            "q", "r", _make_profile(), [_make_result()]
        )

    # ADR-017 RC-1: groundedness is the safety-critical flag and is emitted first.
    # All three scores (style=0.4, groundedness=0.3, confidence=0.5) are below their floors.
    assert result.flags == ["low_groundedness", "low_style", "low_confidence"]


def test_run_result_has_no_final_score():
    scores = Scores(0.9, 0.6, 0.8)
    draft = _ReviewDraft(explanation="ok", flags=[])
    mock_scoring, kickoff_p, instr_p, _ = _patch_run(scores, "verdict", draft)

    with kickoff_p, instr_p:
        result = EvaluatorAgent(scoring_engine=mock_scoring).run("q", "r", _make_profile(), [_make_result()])

    assert not hasattr(result, "final_score")
    assert "final_score" not in result.model_dump()
