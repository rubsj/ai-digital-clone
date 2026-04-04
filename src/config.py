"""YAML config loading with Pydantic v2 validation.

Always uses yaml.safe_load() — never yaml.load() (arbitrary code execution risk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    primary_model: str
    baseline_model: str
    dimension: int = Field(ge=1)


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(ge=50, le=5000)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_lt_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self


class RerankerConfig(BaseModel):
    provider: str
    model: str
    top_n_initial: int = Field(ge=1)
    top_n_final: int = Field(ge=1)


class ScoringConfig(BaseModel):
    style_weight: float = Field(ge=0.0, le=1.0)
    groundedness_weight: float = Field(ge=0.0, le=1.0)
    confidence_weight: float = Field(ge=0.0, le=1.0)
    fallback_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScoringConfig:
        total = self.style_weight + self.groundedness_weight + self.confidence_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"style_weight + groundedness_weight + confidence_weight must sum to 1.0, got {total:.3f}"
            )
        return self


class LLMConfig(BaseModel):
    model: str
    max_retries: int = Field(default=3, ge=1)


class DateRangeConfig(BaseModel):
    start: str
    end: str


class StyleConfig(BaseModel):
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    min_email_words: int = Field(default=20, ge=1)
    date_range: DateRangeConfig


class LeaderConfig(BaseModel):
    name: str
    email_filter: str
    mbox_path: str
    profile_path: str


class AppConfig(BaseModel):
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    reranker: RerankerConfig
    scoring: ScoringConfig
    llm: LLMConfig
    leaders: dict[str, LeaderConfig]
    style: StyleConfig


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path: Optional[Path | str] = None) -> AppConfig:
    """Load and validate YAML config. Raises ValidationError on bad values.

    Defaults to configs/default.yaml relative to the project root.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
