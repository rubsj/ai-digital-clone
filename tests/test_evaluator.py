"""Tests for src/evaluation/evaluator.py.

All three scorers and the Instructor LLM call are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.evaluator import _build_explanation_prompt, evaluate
from src.schemas import EvaluationResult, KnowledgeChunk, RetrievalResult, StyleFeatures, StyleProfile


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.5,
        greeting_patterns={"hi": 0.5},
        punctuation_patterns={"exclamation": 0.1},
        capitalization_ratio=0.1,
        question_frequency=0.1,
        vocabulary_richness=0.6,
        common_phrases=["good point"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.6, "neutral": 0.4},
        formality_level=0.4,
        technical_terminology=0.5,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.5,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_profile(features: StyleFeatures | None = None) -> StyleProfile:
    f = features or _make_features()
    return StyleProfile(
        leader_name="Test Leader",
        features=f,
        style_vector=f.to_vector(),
        email_count=50,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )


def _make_result(score: float = 0.8) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content="kernel memory management details",
        source_topic="Linux Kernel",
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=score, rank=0)


def _mock_instructor_client(explanation: str = "Response is well-grounded and styled.") -> MagicMock:
    """Build a mock instructor client that returns a fixed explanation."""
    mock_result = MagicMock()
    mock_result.explanation = explanation

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_result
    return mock_client


# ---------------------------------------------------------------------------
# _build_explanation_prompt
# ---------------------------------------------------------------------------


def test_build_explanation_prompt_contains_scores():
    prompt = _build_explanation_prompt(0.9, 0.7, 0.8, 0.82, "deliver")
    assert "0.900" in prompt
    assert "0.700" in prompt
    assert "0.800" in prompt
    assert "deliver" in prompt


def test_build_explanation_prompt_mentions_threshold():
    prompt = _build_explanation_prompt(0.5, 0.4, 0.5, 0.47, "fallback")
    assert "0.75" in prompt
    assert "fallback" in prompt


# ---------------------------------------------------------------------------
# evaluate — weighted formula
# ---------------------------------------------------------------------------


def _run_evaluate(style: float, groundedness: float, confidence: float) -> EvaluationResult:
    """Helper: mock all scorers with fixed values and run evaluate()."""
    features = _make_features()
    profile = _make_profile(features)
    results = [_make_result()]

    with (
        patch("src.evaluation.evaluator.score_style", return_value=style),
        patch("src.evaluation.evaluator.score_groundedness", return_value=groundedness),
        patch("src.evaluation.evaluator.score_confidence", return_value=confidence),
        patch("src.evaluation.evaluator.instructor") as mock_instructor,
    ):
        mock_instructor.from_litellm.return_value = _mock_instructor_client()
        return evaluate(
            query="How does memory management work?",
            response="The kernel uses slab allocators for efficient memory management.",
            response_features=features,
            profile=profile,
            retrieval_results=results,
        )


def test_evaluate_formula_exact():
    result = _run_evaluate(style=0.9, groundedness=0.8, confidence=0.7)
    expected = round(0.4 * 0.9 + 0.4 * 0.8 + 0.2 * 0.7, 6)
    assert result.final_score == pytest.approx(expected, abs=0.02)


def test_evaluate_decision_deliver_at_threshold():
    # Exactly 0.75 → deliver
    # 0.4*s + 0.4*g + 0.2*c = 0.75 → s=g=c=0.75
    result = _run_evaluate(style=0.75, groundedness=0.75, confidence=0.75)
    assert result.decision == "deliver"
    assert result.final_score == pytest.approx(0.75, abs=0.02)


def test_evaluate_decision_fallback_below_threshold():
    # 0.4*0.7 + 0.4*0.7 + 0.2*0.7 = 0.70 < 0.75
    result = _run_evaluate(style=0.70, groundedness=0.70, confidence=0.70)
    assert result.decision == "fallback"


def test_evaluate_decision_deliver_above_threshold():
    result = _run_evaluate(style=0.9, groundedness=0.9, confidence=0.9)
    assert result.decision == "deliver"


def test_evaluate_boundary_just_below():
    # Produce final ≈ 0.7499: 0.4*0.75 + 0.4*0.75 + 0.2*0.7495 = 0.7499
    result = _run_evaluate(style=0.75, groundedness=0.75, confidence=0.7495)
    assert result.decision == "fallback"


def test_evaluate_explanation_non_empty():
    result = _run_evaluate(style=0.8, groundedness=0.7, confidence=0.6)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0


def test_evaluate_returns_evaluation_result():
    result = _run_evaluate(style=0.8, groundedness=0.7, confidence=0.6)
    assert isinstance(result, EvaluationResult)


def test_evaluate_scores_in_range():
    result = _run_evaluate(style=0.8, groundedness=0.7, confidence=0.6)
    assert 0.0 <= result.style_score <= 1.0
    assert 0.0 <= result.groundedness_score <= 1.0
    assert 0.0 <= result.confidence_score <= 1.0
    assert 0.0 <= result.final_score <= 1.0
