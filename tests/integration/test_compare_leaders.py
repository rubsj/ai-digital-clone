"""Integration test for compare_leaders() — ADR-005 shared retrieval gate.

Gate: Retriever.run() is called exactly once across both Flow runs.
Leader 2's retrieve step early-exits on state.chunks already populated.

IMPORTANT distinction: this asserts Retriever-Component-count (the RAG
pipeline call), NOT zero embedding calls. The per-leader evaluate stage
still re-embeds chunk text for groundedness scoring (B0 confirmed path) —
those embed_openai calls are expected and are NOT counted here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from src.flow import compare_leaders
from src.schemas import (
    Citation,
    CloneResponse,
    EvaluationResult,
    KnowledgeChunk,
    LeaderComparison,
    RetrievalResult,
    RoutingDecision,
    StyledResponse,
    StyleFeatures,
    StyleProfile,
)


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_flow.py — integration tests are self-contained)
# ---------------------------------------------------------------------------


def _make_profile(name: str = "Linus Torvalds") -> StyleProfile:
    f = StyleFeatures()
    return StyleProfile(
        leader_name=name,
        features=f,
        style_vector=f.to_vector(),
        email_count=50,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _make_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        content="buddy allocator manages pages",
        source_topic="Memory Management",
        source_field="cs",
        chunk_index=0,
        embedding=None,
    )


def _make_retrieval_result() -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(), score=0.85, rank=0)


def _make_clone_response() -> CloneResponse:
    return CloneResponse(
        response_text="The buddy allocator splits pages in power-of-two blocks.",
        citations=[
            Citation(
                chunk_id="0",
                source_topic="Memory Management",
                text_snippet="buddy allocator",
                relevance_score=0.85,
            )
        ],
    )


def _make_eval() -> EvaluationResult:
    return EvaluationResult(
        style_score=0.80,
        groundedness_score=0.75,
        confidence_score=0.72,
        explanation="Grounded in retrieved chunks.",
        flags=[],
    )


def _make_routing_decision(decision: str = "deliver") -> RoutingDecision:
    return RoutingDecision(
        decision=decision,
        reasoning="Scores above target." if decision == "deliver" else "Low groundedness.",
    )


def _make_mock_config():
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.leaders = {
        "torvalds": MagicMock(profile_path="data/models/torvalds_profile.json"),
        "kroah_hartman": MagicMock(profile_path="data/models/kroah_hartman_profile.json"),
    }
    return cfg


# ---------------------------------------------------------------------------
# Gate test: Retriever-Component-call-count
# ---------------------------------------------------------------------------


def test_compare_leaders_retriever_called_once():
    """Retriever.run() is called exactly once across both Flow runs.

    Leader 2's retrieve step early-exits on state.chunks already populated
    (ADR-005). This asserts Retriever-Component-count, NOT zero embedding
    calls — the per-leader evaluate stage still re-embeds chunk text for
    groundedness (B0 confirmed path; those embed_openai calls are expected).
    """
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.load_profile", return_value=_make_profile()),
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.run.return_value = [_make_retrieval_result()]

        compare_leaders("How does virtual memory work?")

    mock_retriever_instance.run.assert_called_once(), (
        "Retriever-Component-call-count must be 1: leader 1 retrieves; "
        "leader 2's retrieve step early-exits on pre-populated state.chunks. "
        "(Not zero embedding calls — evaluate still re-embeds for groundedness.)"
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_compare_leaders_returns_leader_comparison():
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.load_profile", return_value=_make_profile()),
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        MockRetriever.return_value.run.return_value = [_make_retrieval_result()]
        result = compare_leaders("What is a scheduler?")
    assert isinstance(result, LeaderComparison)


def test_compare_leaders_query_propagated():
    q = "What is the buddy allocator?"
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.load_profile", return_value=_make_profile()),
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        MockRetriever.return_value.run.return_value = [_make_retrieval_result()]
        result = compare_leaders(q)
    assert result.query == q


def test_compare_leaders_both_arms_are_styled_response():
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.load_profile", return_value=_make_profile()),
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        MockRetriever.return_value.run.return_value = [_make_retrieval_result()]
        result = compare_leaders("What is a scheduler?")
    assert isinstance(result.torvalds, StyledResponse)
    assert isinstance(result.kroah_hartman, StyledResponse)
