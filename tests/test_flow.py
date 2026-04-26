"""Tests for src/flow.py — Phase 2 happy path.

All LLM-using steps (generate_styled_response, EvaluatorAgent.evaluate) are mocked.
RAGAgent.retrieve and load_profile are also mocked to avoid I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.flow import DigitalCloneFlow, compare_leaders
from src.schemas import (
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    LeaderComparison,
    RetrievalResult,
    StyledResponse,
    StyleFeatures,
    StyleProfile,
)

_MOCK_FALLBACK = FallbackResponse(
    trigger_reason="low score",
    context_summary="kernel memory",
    calendar_link="https://cal.com/placeholder",
    available_slots=["2024-02-01 10:00", "2024-02-02 14:00", "2024-02-03 09:00"],
    unstyled_response="Here is an unstyled answer.",
)


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


def _make_retrieval_result(content: str = "kernel memory details") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic="Linux Kernel",
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


def _make_evaluation(decision: str = "deliver") -> EvaluationResult:
    return EvaluationResult(
        style_score=0.8,
        groundedness_score=0.85,
        confidence_score=0.75,
        final_score=round(0.4 * 0.8 + 0.4 * 0.85 + 0.2 * 0.75, 6),
        explanation="Well-styled and grounded.",
        decision=decision,
    )


def _make_mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.leaders = {
        "torvalds": MagicMock(profile_path="data/models/torvalds_profile.json"),
        "kroah_hartman": MagicMock(profile_path="data/models/kroah_hartman_profile.json"),
    }
    return cfg


# ---------------------------------------------------------------------------
# Shared patch context for all happy-path tests
# ---------------------------------------------------------------------------


def _run_happy_path(
    query: str = "How does memory management work?",
    leader: str = "Linus Torvalds",
    styled_text: str = "The kernel uses slab allocators.",
    decision: str = "deliver",
) -> DigitalCloneFlow:
    """Run the flow end-to-end with all I/O mocked."""
    mock_eval = _make_evaluation(decision=decision)
    mock_profile = _make_profile(leader)

    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", return_value=[_make_retrieval_result()]),
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", return_value=styled_text),
        patch("src.flow.EvaluatorAgent.evaluate", return_value=mock_eval),
    ):
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": query, "leader": leader})

    return flow


# ---------------------------------------------------------------------------
# Happy path — state population
# ---------------------------------------------------------------------------


def test_happy_path():
    flow = _run_happy_path()
    assert len(flow.state.retrieved_chunks) == 1
    assert flow.state.styled_response == "The kernel uses slab allocators."
    assert flow.state.evaluation is not None
    assert 0.0 <= flow.state.evaluation.final_score <= 1.0
    assert isinstance(flow.state.final_output, StyledResponse)


def test_happy_path_retrieved_chunks_populated():
    flow = _run_happy_path()
    assert len(flow.state.retrieved_chunks) > 0


def test_happy_path_styled_response_non_empty():
    flow = _run_happy_path(styled_text="The buddy allocator splits pages.")
    assert len(flow.state.styled_response) > 0


def test_happy_path_evaluation_score_in_range():
    flow = _run_happy_path()
    assert 0.0 <= flow.state.evaluation.final_score <= 1.0


def test_happy_path_final_output_is_styled_response():
    flow = _run_happy_path()
    assert isinstance(flow.state.final_output, StyledResponse)


def test_happy_path_final_output_query_matches():
    q = "What is the buddy allocator?"
    flow = _run_happy_path(query=q)
    assert flow.state.final_output.query == q


def test_happy_path_final_output_leader_matches():
    flow = _run_happy_path(leader="Linus Torvalds")
    assert flow.state.final_output.leader == "Linus Torvalds"


def test_happy_path_final_output_response_matches_styled():
    flow = _run_happy_path(styled_text="Slab allocator manages kernel objects.")
    assert flow.state.final_output.response == "Slab allocator manages kernel objects."


def test_happy_path_evaluation_embedded_in_final_output():
    flow = _run_happy_path()
    assert flow.state.final_output.evaluation == flow.state.evaluation


def test_happy_path_final_output_never_none():
    flow = _run_happy_path()
    assert flow.state.final_output is not None


# ---------------------------------------------------------------------------
# Retrieve step — early-exit guard (Phase 4 hook)
# ---------------------------------------------------------------------------


def test_retrieve_skipped_when_chunks_pre_populated():
    """retrieve() must not call RAGAgent.retrieve when chunks already present."""
    mock_eval = _make_evaluation()
    mock_profile = _make_profile()

    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", return_value=[_make_retrieval_result()]) as mock_retrieve,
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", return_value="response"),
        patch("src.flow.EvaluatorAgent.evaluate", return_value=mock_eval),
    ):
        flow = DigitalCloneFlow()
        pre_populated = [_make_retrieval_result("pre-populated chunk")]
        flow.kickoff(inputs={
            "query": "test",
            "leader": "Linus Torvalds",
            "retrieved_chunks": pre_populated,
        })

    mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# KH leader path
# ---------------------------------------------------------------------------


def test_happy_path_kroah_hartman():
    flow = _run_happy_path(leader="Greg Kroah-Hartman")
    assert flow.state.final_output.leader == "Greg Kroah-Hartman"


# ---------------------------------------------------------------------------
# Phase 3 — Router boundary tests
# ---------------------------------------------------------------------------


def _run_with_score(final_score: float) -> DigitalCloneFlow:
    """Run the flow with a manually constructed EvaluationResult at a given score."""
    style_score = final_score
    groundedness_score = final_score
    confidence_score = final_score
    decision = "deliver" if final_score >= 0.75 else "fallback"
    mock_eval = EvaluationResult(
        style_score=style_score,
        groundedness_score=groundedness_score,
        confidence_score=confidence_score,
        final_score=final_score,
        explanation="boundary test",
        decision=decision,
    )
    mock_profile = _make_profile()

    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", return_value=[_make_retrieval_result()]),
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", return_value="styled text"),
        patch("src.flow.EvaluatorAgent.evaluate", return_value=mock_eval),
        patch("src.flow.build_fallback_response", return_value=_MOCK_FALLBACK),
    ):
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": "test", "leader": "Linus Torvalds"})

    return flow


def test_router_below_threshold_routes_to_fallback():
    """Score 0.7499 must produce a FallbackResponse."""
    flow = _run_with_score(0.7499)
    assert isinstance(flow.state.final_output, FallbackResponse)


def test_router_at_threshold_routes_to_deliver():
    """Score 0.7500 must produce a StyledResponse."""
    flow = _run_with_score(0.7500)
    assert isinstance(flow.state.final_output, StyledResponse)


def test_router_fallback_output_never_none():
    flow = _run_with_score(0.7499)
    assert flow.state.final_output is not None


def test_router_deliver_output_never_none():
    flow = _run_with_score(0.7500)
    assert flow.state.final_output is not None


# ---------------------------------------------------------------------------
# Phase 3 — Error-injection tests
# ---------------------------------------------------------------------------


def _run_with_retrieve_error() -> DigitalCloneFlow:
    mock_profile = _make_profile()
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", side_effect=RuntimeError("FAISS index missing")),
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", return_value="styled text"),
        patch("src.flow.EvaluatorAgent.evaluate", return_value=_make_evaluation()),
        patch("src.flow.build_fallback_response", return_value=_MOCK_FALLBACK),
    ):
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": "test", "leader": "Linus Torvalds"})
    return flow


def _run_with_style_error() -> DigitalCloneFlow:
    mock_profile = _make_profile()
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", return_value=[_make_retrieval_result()]),
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", side_effect=RuntimeError("LLM timeout")),
        patch("src.flow.EvaluatorAgent.evaluate", return_value=_make_evaluation()),
        patch("src.flow.build_fallback_response", return_value=_MOCK_FALLBACK),
    ):
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": "test", "leader": "Linus Torvalds"})
    return flow


def _run_with_evaluate_error() -> DigitalCloneFlow:
    mock_profile = _make_profile()
    with (
        patch("src.flow.load_config", return_value=_make_mock_config()),
        patch("src.flow.RAGAgent.__init__", return_value=None),
        patch("src.flow.RAGAgent.retrieve", return_value=[_make_retrieval_result()]),
        patch("src.flow.load_profile", return_value=mock_profile),
        patch("src.flow.generate_styled_response", return_value="styled text"),
        patch("src.flow.EvaluatorAgent.evaluate", side_effect=RuntimeError("Instructor parse error")),
        patch("src.flow.build_fallback_response", return_value=_MOCK_FALLBACK),
    ):
        flow = DigitalCloneFlow()
        flow.kickoff(inputs={"query": "test", "leader": "Linus Torvalds"})
    return flow


def test_retrieve_error_routes_to_fallback():
    """RAGAgent.retrieve failure → FallbackResponse in final_output."""
    flow = _run_with_retrieve_error()
    assert isinstance(flow.state.final_output, FallbackResponse)


def test_style_error_routes_to_fallback():
    """generate_styled_response failure → FallbackResponse in final_output."""
    flow = _run_with_style_error()
    assert isinstance(flow.state.final_output, FallbackResponse)


def test_evaluate_error_routes_to_fallback():
    """EvaluatorAgent.evaluate failure → FallbackResponse in final_output."""
    flow = _run_with_evaluate_error()
    assert isinstance(flow.state.final_output, FallbackResponse)


def test_retrieve_error_final_output_never_none():
    flow = _run_with_retrieve_error()
    assert flow.state.final_output is not None


def test_style_error_final_output_never_none():
    flow = _run_with_style_error()
    assert flow.state.final_output is not None


def test_evaluate_error_final_output_never_none():
    flow = _run_with_evaluate_error()
    assert flow.state.final_output is not None


def test_retrieve_error_trigger_reason_mentions_step():
    """trigger_reason on retrieve failure must mention 'retrieve'."""
    flow = _run_with_retrieve_error()
    assert "retrieve" in flow.state.trigger_reason.lower()


# ---------------------------------------------------------------------------
# Phase 4 — Dual-leader comparison
# ---------------------------------------------------------------------------

_SHARED_PATCHES = dict(
    load_config="src.flow.load_config",
    rag_init="src.flow.RAGAgent.__init__",
    rag_retrieve="src.flow.RAGAgent.retrieve",
    load_profile="src.flow.load_profile",
    generate="src.flow.generate_styled_response",
    evaluate="src.flow.EvaluatorAgent.evaluate",
    fallback="src.flow.build_fallback_response",
)


def _run_compare_leaders(
    query: str = "How does memory management work?",
    styled_text: str = "The kernel uses slab allocators.",
) -> LeaderComparison:
    mock_eval = _make_evaluation(decision="deliver")
    mock_profile = _make_profile()
    with (
        patch(_SHARED_PATCHES["load_config"], return_value=_make_mock_config()),
        patch(_SHARED_PATCHES["rag_init"], return_value=None),
        patch(_SHARED_PATCHES["rag_retrieve"], return_value=[_make_retrieval_result()]),
        patch(_SHARED_PATCHES["load_profile"], return_value=mock_profile),
        patch(_SHARED_PATCHES["generate"], return_value=styled_text),
        patch(_SHARED_PATCHES["evaluate"], return_value=mock_eval),
        patch(_SHARED_PATCHES["fallback"], return_value=_MOCK_FALLBACK),
    ):
        return compare_leaders(query)


def test_dual_leader_returns_leader_comparison():
    result = _run_compare_leaders()
    assert isinstance(result, LeaderComparison)


def test_dual_leader_torvalds_field_is_styled_response():
    result = _run_compare_leaders()
    assert isinstance(result.torvalds, StyledResponse)


def test_dual_leader_kroah_hartman_field_is_styled_response():
    result = _run_compare_leaders()
    assert isinstance(result.kroah_hartman, StyledResponse)


def test_dual_leader_query_propagated():
    q = "What is the buddy allocator?"
    result = _run_compare_leaders(query=q)
    assert result.query == q
    assert result.torvalds.query == q
    assert result.kroah_hartman.query == q


def test_dual_leader_leaders_differ():
    result = _run_compare_leaders()
    assert result.torvalds.leader != result.kroah_hartman.leader


def test_dual_leader_rag_retrieve_called_once():
    """RAGAgent.retrieve must be called exactly once across both Flow runs."""
    mock_eval = _make_evaluation(decision="deliver")
    mock_profile = _make_profile()
    with (
        patch(_SHARED_PATCHES["load_config"], return_value=_make_mock_config()),
        patch(_SHARED_PATCHES["rag_init"], return_value=None),
        patch(_SHARED_PATCHES["rag_retrieve"], return_value=[_make_retrieval_result()]) as mock_retrieve,
        patch(_SHARED_PATCHES["load_profile"], return_value=mock_profile),
        patch(_SHARED_PATCHES["generate"], return_value="response"),
        patch(_SHARED_PATCHES["evaluate"], return_value=mock_eval),
    ):
        compare_leaders("test query")

    mock_retrieve.assert_called_once()


def test_dual_leader_leader_a_failure_does_not_block_leader_b():
    """If leader A's style step fails, leader B must still produce a StyledResponse."""
    mock_eval = _make_evaluation(decision="deliver")
    mock_profile = _make_profile()
    call_count = {"n": 0}

    def generate_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("LLM timeout for leader A")
        return "KH response"

    with (
        patch(_SHARED_PATCHES["load_config"], return_value=_make_mock_config()),
        patch(_SHARED_PATCHES["rag_init"], return_value=None),
        patch(_SHARED_PATCHES["rag_retrieve"], return_value=[_make_retrieval_result()]),
        patch(_SHARED_PATCHES["load_profile"], return_value=mock_profile),
        patch(_SHARED_PATCHES["generate"], side_effect=generate_side_effect),
        patch(_SHARED_PATCHES["evaluate"], return_value=mock_eval),
        patch(_SHARED_PATCHES["fallback"], return_value=_MOCK_FALLBACK),
    ):
        flow_t = DigitalCloneFlow()
        flow_t.kickoff(inputs={"query": "test", "leader": "Linus Torvalds"})
        shared_chunks = list(flow_t.state.retrieved_chunks)

        flow_kh = DigitalCloneFlow()
        flow_kh.kickoff(inputs={
            "query": "test",
            "leader": "Greg Kroah-Hartman",
            "retrieved_chunks": shared_chunks,
        })

    # Leader A failed → FallbackResponse; leader B must have still run and produced output
    assert isinstance(flow_t.state.final_output, FallbackResponse)
    assert isinstance(flow_kh.state.final_output, StyledResponse)
    assert flow_kh.state.final_output.response == "KH response"
