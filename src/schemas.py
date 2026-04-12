"""All Pydantic v2 data models for P6 Torvalds Digital Clone.

Includes CloneState (Flow state) and all domain models from PRD Section 5a.
Models with np.ndarray fields use ConfigDict(arbitrary_types_allowed=True) +
field_serializer/field_validator for JSON roundtrip compatibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


# ---------------------------------------------------------------------------
# Email models
# ---------------------------------------------------------------------------


class EmailMessage(BaseModel):
    """Parsed and cleaned LKML email from an mbox archive."""

    sender: str
    recipients: list[str] = Field(default_factory=list)
    subject: str
    body: str
    timestamp: datetime
    message_id: str
    is_patch: bool = False
    quote_ratio: float = Field(default=0.0, ge=0.0, le=1.0)  # quoted lines / total lines, computed pre-cleaning


# ---------------------------------------------------------------------------
# Style models
# ---------------------------------------------------------------------------


class StyleFeatures(BaseModel):
    """15 numerical style features extracted from one email or an aggregate.

    All float fields are normalized to [0, 1]. Dict fields hold frequency
    distributions keyed by pattern name (e.g. {"?": 0.03, "!": 0.01}).
    """

    # --- 11 base features ---
    avg_message_length: float = Field(default=0.0, ge=0.0, le=1.0)
    greeting_patterns: dict[str, float] = Field(default_factory=dict)
    punctuation_patterns: dict[str, float] = Field(default_factory=dict)
    capitalization_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    question_frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    vocabulary_richness: float = Field(default=0.0, ge=0.0, le=1.0)
    common_phrases: list[str] = Field(default_factory=list)
    reasoning_patterns: dict[str, float] = Field(default_factory=dict)
    sentiment_distribution: dict[str, float] = Field(default_factory=dict)
    formality_level: float = Field(default=0.0, ge=0.0, le=1.0)
    technical_terminology: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- 4 LKML-specific features ---
    code_snippet_freq: float = Field(default=0.0, ge=0.0, le=1.0)
    quote_reply_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_language: dict[str, float] = Field(default_factory=dict)
    technical_depth: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_vector(self) -> np.ndarray:
        """Concatenate all 15 features into a length-15 float64 array.

        For dict fields (frequency distributions), use the mean of values so
        the vector stays fixed-length regardless of which patterns appear.
        """
        dict_mean = lambda d: float(np.mean(list(d.values()))) if d else 0.0  # noqa: E731

        return np.array(
            [
                self.avg_message_length,
                dict_mean(self.greeting_patterns),
                dict_mean(self.punctuation_patterns),
                self.capitalization_ratio,
                self.question_frequency,
                self.vocabulary_richness,
                dict_mean(self.reasoning_patterns),
                dict_mean(self.sentiment_distribution),
                self.formality_level,
                self.technical_terminology,
                self.code_snippet_freq,
                self.quote_reply_ratio,
                dict_mean(self.patch_language),
                self.technical_depth,
                # 15th: phrase diversity (unique phrases / max_phrases cap 20)
                min(len(self.common_phrases) / 20.0, 1.0),
            ],
            dtype=np.float64,
        )


class StyleProfile(BaseModel):
    """Aggregate learned style profile for one leader.

    style_vector is a length-15 np.ndarray computed from StyleFeatures.to_vector().
    Serializes to list[float] for JSON caching; deserializes back to np.ndarray.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    leader_name: str
    features: StyleFeatures
    style_vector: np.ndarray = Field(
        ..., description="Length-15 feature vector, all values in [0, 1]"
    )
    email_count: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_serializer("style_vector")
    def serialize_vector(self, v: np.ndarray) -> list[float]:
        return v.tolist()

    @field_validator("style_vector", mode="before")
    @classmethod
    def coerce_vector(cls, v: object) -> np.ndarray:
        if isinstance(v, list):
            return np.array(v, dtype=np.float64)
        return v  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# RAG / knowledge models
# ---------------------------------------------------------------------------


class KnowledgeChunk(BaseModel):
    """One chunk of textbook content, optionally with its embedding vector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    source_topic: str
    source_field: str
    chunk_index: int = Field(ge=0)
    embedding: Optional[np.ndarray] = None

    @field_serializer("embedding")
    def serialize_embedding(self, v: Optional[np.ndarray]) -> Optional[list[float]]:
        return v.tolist() if v is not None else None

    @field_validator("embedding", mode="before")
    @classmethod
    def coerce_embedding(cls, v: object) -> Optional[np.ndarray]:
        if isinstance(v, list):
            return np.array(v, dtype=np.float64)
        return v  # type: ignore[return-value]


class RetrievalResult(BaseModel):
    """One retrieved chunk with its similarity score and reranking position."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: KnowledgeChunk
    score: float
    rank: int = Field(ge=0)


class Citation(BaseModel):
    """Source reference extracted from a generated response."""

    chunk_id: str
    source_topic: str
    text_snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Quality scores for one generated response across three dimensions.

    final_score = 0.4 * style_score + 0.4 * groundedness_score + 0.2 * confidence_score
    """

    style_score: float = Field(ge=0.0, le=1.0)
    groundedness_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    decision: Literal["deliver", "fallback"]

    @model_validator(mode="after")
    def validate_formula(self) -> EvaluationResult:
        expected = (
            0.4 * self.style_score
            + 0.4 * self.groundedness_score
            + 0.2 * self.confidence_score
        )
        if abs(self.final_score - expected) > 0.02:
            raise ValueError(
                f"final_score {self.final_score:.3f} doesn't match formula "
                f"(expected {expected:.3f} = 0.4*{self.style_score} + "
                f"0.4*{self.groundedness_score} + 0.2*{self.confidence_score})"
            )
        return self


# ---------------------------------------------------------------------------
# Fallback / output models
# ---------------------------------------------------------------------------


class FallbackResponse(BaseModel):
    """Output when EvaluatorAgent routes to fallback (final_score < 0.75)."""

    trigger_reason: str
    context_summary: str
    calendar_link: str
    available_slots: list[str] = Field(default_factory=list)
    unstyled_response: Optional[str] = None


class StyledResponse(BaseModel):
    """Final output for a single leader query — scored, cited, optionally fallback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    leader: str
    response: str
    evaluation: EvaluationResult
    citations: list[Citation] = Field(default_factory=list)
    fallback: Optional[FallbackResponse] = None


class LeaderComparison(BaseModel):
    """Side-by-side dual-leader comparison for the same query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    torvalds: StyledResponse
    kroah_hartman: StyledResponse


# ---------------------------------------------------------------------------
# Flow state
# ---------------------------------------------------------------------------


class CloneState(BaseModel):
    """Typed state passed between all DigitalCloneFlow steps.

    All fields have defaults — CrewAI Flow populates them incrementally as each
    step completes. Needs arbitrary_types_allowed because RetrievalResult
    contains KnowledgeChunk which contains np.ndarray.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = ""
    leader: str = ""
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    styled_response: str = ""
    evaluation: Optional[EvaluationResult] = None
    final_output: Optional[Union[StyledResponse, FallbackResponse]] = None
