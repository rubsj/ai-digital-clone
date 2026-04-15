"""Evaluator: combines style, groundedness, and confidence into EvaluationResult.

Formula:  final_score = 0.4 × style + 0.4 × groundedness + 0.2 × confidence
Decision: "deliver" if final_score >= 0.75, else "fallback".

One Instructor + LiteLLM call generates the human-readable explanation string.
This is the ONLY LLM call in the evaluation pipeline — not per-scorer.
"""

from __future__ import annotations

import litellm
import instructor
from pydantic import BaseModel

from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import score_groundedness
from src.schemas import EvaluationResult, RetrievalResult, StyleFeatures, StyleProfile
from src.style.style_scorer import score_style

_LLM_MODEL = "gpt-4o-mini"
_LLM_MAX_RETRIES = 3

# Tolerance already enforced by EvaluationResult model_validator (0.02),
# so we round to 6 places to avoid floating-point drift.
_FORMULA_WEIGHTS = (0.4, 0.4, 0.2)


class _ExplanationModel(BaseModel):
    """Structured output for the explanation LLM call."""

    explanation: str


def _build_explanation_prompt(
    style: float,
    groundedness: float,
    confidence: float,
    final: float,
    decision: str,
) -> str:
    return (
        f"You are a quality reviewer for AI-generated leadership coaching responses.\n\n"
        f"Evaluation scores:\n"
        f"  Style score:        {style:.3f} (target > 0.90)\n"
        f"  Groundedness score: {groundedness:.3f} (target > 0.60)\n"
        f"  Confidence score:   {confidence:.3f} (target > 0.80)\n"
        f"  Final score:        {final:.3f} (threshold 0.75)\n"
        f"  Decision:           {decision}\n\n"
        f"Write ONE concise sentence (≤ 25 words) explaining the decision in plain English. "
        f"Focus on the weakest dimension if decision is 'fallback'. "
        f"Be direct; no hedging."
    )


def _generate_explanation(
    style: float,
    groundedness: float,
    confidence: float,
    final: float,
    decision: str,
) -> str:
    """Single Instructor call → validated explanation string."""
    client = instructor.from_litellm(litellm.completion)
    prompt = _build_explanation_prompt(style, groundedness, confidence, final, decision)
    result: _ExplanationModel = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=_ExplanationModel,
        max_retries=_LLM_MAX_RETRIES,
    )
    return result.explanation


def evaluate(
    query: str,
    response: str,
    response_features: StyleFeatures,
    profile: StyleProfile,
    retrieval_results: list[RetrievalResult],
) -> EvaluationResult:
    """Score a response across three dimensions and route to deliver or fallback.

    Args:
        query:             Original user query.
        response:          Generated response text.
        response_features: 15-feature vector extracted from the response.
        profile:           Leader style profile (Day 2).
        retrieval_results: Top-k results from the RAG pipeline.

    Returns:
        EvaluationResult with final_score, decision, and explanation.
    """
    style = score_style(profile, response_features)
    groundedness = score_groundedness(response, retrieval_results)
    confidence = score_confidence(query, response, retrieval_results)

    final = round(
        _FORMULA_WEIGHTS[0] * style
        + _FORMULA_WEIGHTS[1] * groundedness
        + _FORMULA_WEIGHTS[2] * confidence,
        6,
    )
    decision = "deliver" if final >= 0.75 else "fallback"

    explanation = _generate_explanation(style, groundedness, confidence, final, decision)

    return EvaluationResult(
        style_score=style,
        groundedness_score=groundedness,
        confidence_score=confidence,
        final_score=final,
        explanation=explanation,
        decision=decision,
    )
