"""ScoringEngine Component: three deterministic quality scores (ADR-007, ADR-011).

LLM-free. Wraps the three v1 sub-scorers — style (cosine on 15-dim feature
vectors), groundedness (sentence-level semantic similarity), and confidence
(multi-factor heuristic) — into one score() call. No combined score and no
routing decision: the EvaluatorAgent consumes these scores and the
GatekeeperAgent owns routing. The scoring math (ADR-003/004) is unchanged.

Note: groundedness uses embeddings via src/rag/embedder (transitively LiteLLM).
Embeddings are vector math on the frozen scoring path, not LLM reasoning; this
module imports no litellm/openai/cohere/instructor directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import score_groundedness
from src.schemas import EmailMessage, RetrievalResult, StyleProfile
from src.style.feature_extractor import extract_features
from src.style.style_scorer import score_style

# Fixed epoch — extract_features only reads body + quote_ratio, but EmailMessage
# requires a timestamp. Constant keeps response feature-extraction deterministic.
_RESPONSE_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class Scores(NamedTuple):
    """The three deterministic dimension scores. Not a schema — a return type."""

    style_score: float
    groundedness_score: float
    confidence_score: float


class ScoringEngine:
    """Deterministic three-dimension scorer."""

    def score(
        self,
        query: str,
        response: str,
        profile: StyleProfile,
        chunks: list[RetrievalResult],
    ) -> Scores:
        """Score a response on style, groundedness, and confidence.

        Style features are extracted from the response text via the frozen
        feature extractor (response wrapped in a minimal EmailMessage).
        """
        response_features = extract_features(
            EmailMessage(
                sender="response",
                subject="",
                body=response,
                timestamp=_RESPONSE_EPOCH,
                message_id="response",
                quote_ratio=0.0,
            )
        )
        return Scores(
            style_score=score_style(profile, response_features),
            groundedness_score=score_groundedness(response, chunks),
            confidence_score=score_confidence(query, response, chunks),
        )
