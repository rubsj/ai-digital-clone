"""Tests for src/flow.py v2 — five-step pipeline with Gatekeeper routing.

All LLM-using steps are mocked (Retriever, CloneAgent, EvaluatorAgent,
Gatekeeper, FallbackAgent). These tests prove wiring only — routing
correctness is measured with real LLM on Day 12. The known pre-existing
failure test_query_loader.py::test_load_queries_canonical_file is unrelated
to this module.
"""

from __future__ import annotations

from unittest.mock import patch

from src.flow import DigitalCloneFlow
from src.schemas import (
    Citation,
    CloneResponse,
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    RetrievalResult,
    RoutingDecision,
    StyledResponse,
    StyleFeatures,
    StyleProfile,
)

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.34,
        greeting_patterns={},
        punctuation_patterns={"dash": 0.2},
        capitalization_ratio=0.05,
        question_frequency=0.1,
        vocabulary_richness=0.6,
        common_phrases=["good point"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.3},
        formality_level=0.5,
        technical_terminology=0.4,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.12,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_profile(name: str = "Linus Torvalds") -> StyleProfile:
    f = _make_features()
    return StyleProfile(
        leader_name=name,
        features=f,
        style_vector=f.to_vector(),
        email_count=100,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )


def _make_chunk(content: str = "kernel memory details") -> KnowledgeChunk:
    return KnowledgeChunk(
        content=content,
        source_topic="Linux Kernel",
        source_field="cs",
        chunk_index=0,
        embedding=None,
    )


def _make_retrieval_result(content: str = "kernel memory details") -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(content), score=0.85, rank=0)


def _make_clone_response() -> CloneResponse:
    return CloneResponse(
        response_text="The buddy allocator manages physical pages in a power-of-two hierarchy.",
        citations=[
            Citation(
                chunk_id="0",
                source_topic="Linux Kernel",
                text_snippet="kernel memory",
                relevance_score=0.85,
            )
        ],
    )


def _make_eval(flags: list[str] | None = None) -> EvaluationResult:
    return EvaluationResult(
        style_score=0.82,
        groundedness_score=0.78,
        confidence_score=0.75,
        explanation="Well-styled and grounded in the retrieved chunks.",
        flags=flags or [],
    )


def _make_routing_decision(decision: str = "deliver") -> RoutingDecision:
    if decision == "deliver":
        return RoutingDecision(
            decision="deliver",
            reasoning="Scores above target; no flags raised.",
        )
    return RoutingDecision(
        decision="fallback",
        reasoning="Groundedness 0.210 below target.",
        trigger_reason="groundedness_score too low",
        trigger_category="low_groundedness",
    )


def _make_fallback_response() -> FallbackResponse:
    return FallbackResponse(
        acknowledgment="That question is outside what I can confidently answer.",
        suggested_redirections=["How does the buddy allocator work?"],
        calendar_link="https://cal.com/placeholder",
        available_slots=["2024-02-01 10:00", "2024-02-02 14:00", "2024-02-03 09:00"],
        unstyled_response="Here is a plain answer.",
    )


# ---------------------------------------------------------------------------
# Central mock runner helpers
# ---------------------------------------------------------------------------


def _run_deliver(
    query: str = "How does virtual memory work?",
    leader: str = "Linus Torvalds",
    chunks_preloaded: list[RetrievalResult] | None = None,
) -> DigitalCloneFlow:
    """Run the full v2 flow with all agents mocked, routing to deliver."""
    profile = _make_profile(leader)
    inputs: dict = {"query": query, "leader": leader, "style_profile": profile}
    if chunks_preloaded is not None:
        inputs["chunks"] = chunks_preloaded

    with (
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        MockRetriever.return_value.run.return_value = [_make_retrieval_result()]
        flow = DigitalCloneFlow()
        flow.kickoff(inputs=inputs)
    return flow


def _run_fallback(
    query: str = "Who will win the next election?",
    leader: str = "Linus Torvalds",
) -> DigitalCloneFlow:
    """Run the full v2 flow with all agents mocked, routing to fallback."""
    profile = _make_profile(leader)
    with (
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("fallback")),
        patch("src.flow.FallbackAgent.run", return_value=_make_fallback_response()),
    ):
        MockRetriever.return_value.run.return_value = [_make_retrieval_result()]
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": query, "leader": leader, "style_profile": profile})
    return flow


# ---------------------------------------------------------------------------
# Step 1: retrieve — early-exit guard (ADR-005)
# ---------------------------------------------------------------------------


def test_retrieve_skipped_when_chunks_prepopulated():
    """retrieve() must not call Retriever.run when state.chunks already present."""
    with (
        patch("src.flow.Retriever") as MockRetriever,
        patch("src.flow.CloneAgent.run", return_value=_make_clone_response()),
        patch("src.flow.EvaluatorAgent.run", return_value=_make_eval()),
        patch("src.flow.Gatekeeper.run", return_value=_make_routing_decision("deliver")),
    ):
        mock_run = MockRetriever.return_value.run
        mock_run.return_value = [_make_retrieval_result()]
        pre_populated = [_make_retrieval_result("pre-populated chunk")]
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={
            "query": "test",
            "leader": "Linus Torvalds",
            "style_profile": _make_profile(),
            "chunks": pre_populated,
        })
    mock_run.assert_not_called()


def test_retrieve_populates_state_chunks():
    flow = _run_deliver()
    assert isinstance(flow.state.chunks, list)
    assert len(flow.state.chunks) > 0
    assert isinstance(flow.state.chunks[0], RetrievalResult)


# ---------------------------------------------------------------------------
# Deliver path — state typing and output shape
# ---------------------------------------------------------------------------


def test_deliver_path_styled_response_is_styled_response():
    """Deliver arm must produce a StyledResponse in state.styled_response."""
    flow = _run_deliver()
    assert isinstance(flow.state.styled_response, StyledResponse)


def test_deliver_path_fallback_response_is_none():
    flow = _run_deliver()
    assert flow.state.fallback_response is None


def test_deliver_path_response_text_populated():
    flow = _run_deliver()
    assert flow.state.response_text is not None
    assert len(flow.state.response_text) > 0


def test_deliver_path_evaluation_is_evaluation_result():
    flow = _run_deliver()
    assert isinstance(flow.state.evaluation, EvaluationResult)


def test_deliver_path_routing_decision_is_routing_decision():
    flow = _run_deliver()
    assert isinstance(flow.state.routing_decision, RoutingDecision)
    assert flow.state.routing_decision.decision == "deliver"


def test_deliver_path_styled_response_query_matches():
    q = "What is the buddy allocator?"
    flow = _run_deliver(query=q)
    assert flow.state.styled_response.query == q


def test_deliver_path_styled_response_leader_matches():
    flow = _run_deliver(leader="Linus Torvalds")
    assert flow.state.styled_response.leader == "Linus Torvalds"


def test_deliver_path_five_step_trace():
    """All five v2 pipeline fields must be populated on the deliver path."""
    flow = _run_deliver()
    assert flow.state.chunks            # step 1: retrieve
    assert flow.state.response_text     # step 2: clone
    assert flow.state.evaluation        # step 3: evaluate
    assert flow.state.routing_decision  # step 4: route
    assert flow.state.styled_response   # step 5: finalize (deliver arm)


# ---------------------------------------------------------------------------
# Fallback path — state typing and output shape
# ---------------------------------------------------------------------------


def test_fallback_path_fallback_response_is_fallback_response():
    """Fallback arm must produce a FallbackResponse in state.fallback_response."""
    flow = _run_fallback()
    assert isinstance(flow.state.fallback_response, FallbackResponse)


def test_fallback_path_styled_response_is_none():
    flow = _run_fallback()
    assert flow.state.styled_response is None


def test_fallback_path_routing_decision_is_fallback():
    flow = _run_fallback()
    assert isinstance(flow.state.routing_decision, RoutingDecision)
    assert flow.state.routing_decision.decision == "fallback"


def test_fallback_path_five_step_trace():
    """All five v2 pipeline fields must be populated on the fallback path."""
    flow = _run_fallback()
    assert flow.state.chunks             # step 1: retrieve
    assert flow.state.response_text      # step 2: clone
    assert flow.state.evaluation         # step 3: evaluate
    assert flow.state.routing_decision   # step 4: route
    assert flow.state.fallback_response  # step 5: handle_fallback (fallback arm)


# ---------------------------------------------------------------------------
# Kroah-Hartman leader path
# ---------------------------------------------------------------------------


def test_kroah_hartman_leader_deliver_path():
    flow = _run_deliver(leader="Greg Kroah-Hartman")
    assert isinstance(flow.state.styled_response, StyledResponse)
    assert flow.state.styled_response.leader == "Greg Kroah-Hartman"


# ---------------------------------------------------------------------------
# B2 latency — timings dict populated (not asserted for values — Day 12)
# ---------------------------------------------------------------------------


def test_timings_dict_has_retrieve_key():
    flow = _run_deliver()
    assert "retrieve_ms" in flow.timings


def test_timings_dict_has_clone_keys():
    flow = _run_deliver()
    assert "clone_ms" in flow.timings
    assert "clone_generate_ms" in flow.timings
    assert "clone_parse_ms" in flow.timings


def test_timings_dict_has_evaluate_keys():
    flow = _run_deliver()
    assert "evaluate_ms" in flow.timings
    assert "evaluate_generate_ms" in flow.timings
    assert "evaluate_parse_ms" in flow.timings


def test_timings_dict_has_route_keys():
    flow = _run_deliver()
    assert "route_ms" in flow.timings
    assert "route_generate_ms" in flow.timings
    assert "route_parse_ms" in flow.timings


def test_timings_dict_has_deliver_key():
    flow = _run_deliver()
    assert "deliver_ms" in flow.timings


def test_timings_dict_has_fallback_keys():
    flow = _run_fallback()
    assert "fallback_ms" in flow.timings
    assert "fallback_generate_ms" in flow.timings
    assert "fallback_parse_ms" in flow.timings


def test_timings_not_retrieved_when_early_exit():
    """Timings dict must not have retrieve_ms when retrieve step early-exits."""
    flow = _run_deliver(chunks_preloaded=[_make_retrieval_result()])
    assert "retrieve_ms" not in flow.timings
