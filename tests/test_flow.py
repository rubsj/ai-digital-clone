"""Tests for src/flow.py — Phase 2 happy path.

All LLM-using steps (generate_styled_response, EvaluatorAgent.evaluate) are mocked.
RAGAgent.retrieve and load_profile are also mocked to avoid I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.flow import DigitalCloneFlow
from src.schemas import (
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    RetrievalResult,
    StyledResponse,
    StyleFeatures,
    StyleProfile,
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
