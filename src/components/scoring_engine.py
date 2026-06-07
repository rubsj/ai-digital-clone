"""ScoringEngine Component: three deterministic quality scores (ADR-007, ADR-011).

LLM-free. Wraps three sub-scorers — style (cosine on 15-dim feature vectors),
groundedness (HHEM-2.1-Open local entailment; ADR-020), and confidence (multi-
factor heuristic) — into one score() call. No combined score and no routing
decision: the EvaluatorAgent consumes these scores and the Gatekeeper owns
routing. The scoring math (ADR-003/004) is unchanged.

The HHEM model is loaded once in __init__ and held resident (same lifecycle as
the FAISS index). Zero paid API calls; all groundedness inference local and
in-process.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import HHEMGroundednessScorer
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
    """Deterministic three-dimension scorer.

    The HHEM groundedness model is loaded at construction and held resident.
    Pass a pre-constructed HHEMGroundednessScorer to share an already-loaded
    instance (e.g. in tests or when multiple engines run in the same process).
    """

    def __init__(
        self,
        groundedness_scorer: HHEMGroundednessScorer | None = None,
    ) -> None:
        self._gscorer = groundedness_scorer or HHEMGroundednessScorer()

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
            groundedness_score=self._gscorer.score(response, chunks),
            confidence_score=score_confidence(query, response, chunks),
        )
