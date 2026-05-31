"""Contract tests for GatekeeperAgent (ADR-010).

These are plumbing tests — they verify that inputs reach the built prompt and
that the mocked LLM output parses to a valid RoutingDecision. Routing correctness
(right decision for given scores/flags), determinism at temperature=0, and
trigger_category accuracy are real-LLM behavior measured on Day 12, not provable
under a mock.

A green suite here proves plumbing, not routing behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.gatekeeper_agent import (
    GatekeeperAgent,
    _build_backstory,
    _build_goal,
    _build_role,
    _build_task_description,
)
from src.schemas import (
    EvaluationResult,
    KnowledgeChunk,
    RetrievalResult,
    RoutingDecision,
)

_TRIGGER_CATEGORIES = (
    "low_groundedness",
    "off_domain",
    "hallucination_risk",
    "chunk_mismatch",
    "empty_retrieval",
)


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")


def _make_eval(
    style_score: float = 0.92,
    groundedness_score: float = 0.75,
    confidence_score: float = 0.80,
    explanation: str = "Style matches well. Response grounded in 3 chunks.",
    flags: list[str] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        style_score=style_score,
        groundedness_score=groundedness_score,
        confidence_score=confidence_score,
        explanation=explanation,
        flags=flags or [],
    )


def _make_chunk(
    content: str = "buddy allocator manages physical pages",
    topic: str = "Memory Management",
) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic=topic,
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


def _mock_run(routing_decision: RoutingDecision):
    """Return patches for Crew.kickoff and instructor.from_litellm."""
    kickoff_output = MagicMock()
    kickoff_output.raw = "Route to deliver — all scores above target."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = routing_decision
    return (
        patch("src.agents.gatekeeper_agent.Crew.kickoff", return_value=kickoff_output),
        patch("src.agents.gatekeeper_agent.instructor.from_litellm", return_value=mock_client),
    )


# ---------------------------------------------------------------------------
# Prompt builders — plumbing assertions
# ---------------------------------------------------------------------------


def test_role_non_empty():
    assert _build_role()


def test_goal_non_empty():
    assert _build_goal()


def test_backstory_non_empty():
    assert _build_backstory()


def test_task_description_contains_style_score():
    desc = _build_task_description("q", "resp", [], _make_eval(style_score=0.92))
    assert "0.920" in desc


def test_task_description_contains_groundedness_score():
    desc = _build_task_description("q", "resp", [], _make_eval(groundedness_score=0.75))
    assert "0.750" in desc


def test_task_description_contains_confidence_score():
    desc = _build_task_description("q", "resp", [], _make_eval(confidence_score=0.80))
    assert "0.800" in desc


def test_task_description_contains_flags():
    desc = _build_task_description(
        "q", "resp", [], _make_eval(flags=["low_groundedness", "chunk_mismatch"])
    )
    assert "low_groundedness" in desc
    assert "chunk_mismatch" in desc


def test_task_description_no_flags_label():
    desc = _build_task_description("q", "resp", [], _make_eval(flags=[]))
    assert "(none)" in desc


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------


def test_crew_has_one_agent_one_task():
    crew = GatekeeperAgent()._build_crew(
        "q", "resp", [_make_chunk()], _make_eval(), "torvalds"
    )
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1


def test_crew_agent_has_role_goal_backstory():
    agent = GatekeeperAgent()._build_crew(
        "q", "resp", [_make_chunk()], _make_eval(), "torvalds"
    ).agents[0]
    assert agent.role and agent.goal and agent.backstory


# ---------------------------------------------------------------------------
# run() contract — output shape (LLM mocked)
# ---------------------------------------------------------------------------


def test_run_deliver_returns_valid_routing_decision():
    rd = RoutingDecision(
        decision="deliver",
        reasoning="Style 0.920 and groundedness 0.750 both above targets; no flags.",
        trigger_reason=None,
        trigger_category=None,
    )
    kickoff_p, instr_p = _mock_run(rd)
    with kickoff_p, instr_p:
        result = GatekeeperAgent().run(
            query="How does virtual memory work?",
            response_text="The buddy allocator manages physical pages...",
            chunks=[_make_chunk()],
            evaluation=_make_eval(),
            leader="torvalds",
        )

    assert isinstance(result, RoutingDecision)
    assert result.decision == "deliver"
    assert result.reasoning
    assert result.trigger_category is None
    assert result.trigger_reason is None


def test_run_fallback_returns_valid_routing_decision():
    rd = RoutingDecision(
        decision="fallback",
        reasoning="Groundedness 0.210 far below target; flag low_groundedness raised.",
        trigger_reason="groundedness_score 0.210 is too low to trust this response.",
        trigger_category="low_groundedness",
    )
    kickoff_p, instr_p = _mock_run(rd)
    with kickoff_p, instr_p:
        result = GatekeeperAgent().run(
            query="q",
            response_text="resp",
            chunks=[_make_chunk()],
            evaluation=_make_eval(groundedness_score=0.21, flags=["low_groundedness"]),
            leader="torvalds",
        )

    assert isinstance(result, RoutingDecision)
    assert result.decision == "fallback"
    assert result.reasoning
    assert result.trigger_reason


# ---------------------------------------------------------------------------
# trigger_category iff contract
# ---------------------------------------------------------------------------


def test_trigger_category_none_on_deliver():
    rd = RoutingDecision(
        decision="deliver",
        reasoning="All scores above target.",
        trigger_reason=None,
        trigger_category=None,
    )
    kickoff_p, instr_p = _mock_run(rd)
    with kickoff_p, instr_p:
        result = GatekeeperAgent().run("q", "resp", [], _make_eval(), "torvalds")
    assert result.trigger_category is None


def test_trigger_category_set_on_fallback():
    rd = RoutingDecision(
        decision="fallback",
        reasoning="Groundedness 0.210 too low.",
        trigger_reason="score too low",
        trigger_category="low_groundedness",
    )
    kickoff_p, instr_p = _mock_run(rd)
    with kickoff_p, instr_p:
        result = GatekeeperAgent().run("q", "resp", [], _make_eval(), "torvalds")
    assert result.trigger_category in _TRIGGER_CATEGORIES


def test_trigger_category_each_literal_parses():
    """Each of the 5 bounded literals is a valid RoutingDecision trigger_category."""
    for cat in _TRIGGER_CATEGORIES:
        rd = RoutingDecision(
            decision="fallback",
            reasoning="scores below target.",
            trigger_reason="score too low",
            trigger_category=cat,
        )
        assert rd.trigger_category == cat


# ---------------------------------------------------------------------------
# temperature=0 config assertion
# ---------------------------------------------------------------------------


def test_parse_uses_temperature_zero():
    rd = RoutingDecision(
        decision="deliver",
        reasoning="All scores above target.",
    )
    kickoff_output = MagicMock()
    kickoff_output.raw = "Deliver — scores fine."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = rd

    with (
        patch("src.agents.gatekeeper_agent.Crew.kickoff", return_value=kickoff_output),
        patch("src.agents.gatekeeper_agent.instructor.from_litellm", return_value=mock_client),
    ):
        GatekeeperAgent().run("q", "resp", [], _make_eval(), "torvalds")

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is RoutingDecision
    assert kwargs["temperature"] == 0
