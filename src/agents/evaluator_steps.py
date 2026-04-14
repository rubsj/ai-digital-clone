"""EvaluatorAgent: facade over the evaluation pipeline.

Wraps groundedness_scorer + confidence_scorer + evaluator into a single
callable class that the Day 5 flow can call without knowing the internals.
"""

from __future__ import annotations

from src.evaluation.evaluator import evaluate
from src.schemas import EvaluationResult, RetrievalResult, StyleFeatures, StyleProfile


class EvaluatorAgent:
    """Facade for the full evaluation pipeline.

    Usage:
        agent = EvaluatorAgent()
        result = agent.evaluate(response, chunks, profile, query, response_features)
    """

    def evaluate(
        self,
        response: str,
        chunks: list[RetrievalResult],
        profile: StyleProfile,
        query: str,
        response_features: StyleFeatures,
    ) -> EvaluationResult:
        """Score a response and decide deliver vs. fallback.

        Args:
            response:          Generated response text.
            chunks:            Top-k retrieved chunks from RAG pipeline.
            profile:           Leader style profile (Day 2).
            query:             Original user query.
            response_features: 15-feature vector extracted from the response.

        Returns:
            EvaluationResult with style/groundedness/confidence scores,
            final_score, decision, and explanation.
        """
        return evaluate(
            query=query,
            response=response,
            response_features=response_features,
            profile=profile,
            retrieval_results=chunks,
        )
