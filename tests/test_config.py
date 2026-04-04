"""Tests for src/config.py YAML config loading and validation.

Coverage target: >= 90% of config.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    AppConfig,
    ChunkingConfig,
    ScoringConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# load_config (default)
# ---------------------------------------------------------------------------


def test_load_default_config():
    config = load_config()
    assert isinstance(config, AppConfig)


def test_config_scoring_weights_sum_to_one():
    config = load_config()
    total = (
        config.scoring.style_weight
        + config.scoring.groundedness_weight
        + config.scoring.confidence_weight
    )
    assert abs(total - 1.0) < 0.001


def test_config_fallback_threshold():
    config = load_config()
    assert config.scoring.fallback_threshold == 0.75


def test_config_alpha_in_range():
    config = load_config()
    assert 0.0 <= config.style.alpha <= 1.0


def test_config_both_leaders_present():
    config = load_config()
    assert "torvalds" in config.leaders
    assert "kroah_hartman" in config.leaders


def test_config_leader_fields():
    config = load_config()
    torvalds = config.leaders["torvalds"]
    assert "torvalds@" in torvalds.email_filter
    assert torvalds.mbox_path.endswith(".mbox")


# ---------------------------------------------------------------------------
# ScoringConfig validation
# ---------------------------------------------------------------------------


def test_scoring_weights_not_summing_to_one_raises():
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        ScoringConfig(
            style_weight=0.5,
            groundedness_weight=0.5,
            confidence_weight=0.5,  # total = 1.5
            fallback_threshold=0.75,
        )


def test_scoring_config_valid():
    sc = ScoringConfig(
        style_weight=0.4,
        groundedness_weight=0.4,
        confidence_weight=0.2,
        fallback_threshold=0.75,
    )
    assert sc.style_weight == 0.4


# ---------------------------------------------------------------------------
# ChunkingConfig validation
# ---------------------------------------------------------------------------


def test_chunking_overlap_gte_size_raises():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_chunking_overlap_less_than_size_valid():
    cc = ChunkingConfig(chunk_size=500, chunk_overlap=50)
    assert cc.chunk_size == 500


# ---------------------------------------------------------------------------
# load_config with custom YAML
# ---------------------------------------------------------------------------


def test_load_custom_config(tmp_path):
    yaml_content = textwrap.dedent("""\
        embedding:
          primary_model: "text-embedding-3-small"
          baseline_model: "all-MiniLM-L6-v2"
          dimension: 1536
        chunking:
          chunk_size: 500
          chunk_overlap: 50
        reranker:
          provider: "cohere"
          model: "rerank-english-v3.0"
          top_n_initial: 20
          top_n_final: 5
        scoring:
          style_weight: 0.4
          groundedness_weight: 0.4
          confidence_weight: 0.2
          fallback_threshold: 0.80
        llm:
          model: "gpt-4o-mini"
          max_retries: 3
        leaders:
          torvalds:
            name: "Linus Torvalds"
            email_filter: "torvalds@"
            mbox_path: "data/emails/torvalds.mbox"
            profile_path: "data/models/torvalds_profile.json"
          kroah_hartman:
            name: "Greg Kroah-Hartman"
            email_filter: "gregkh@"
            mbox_path: "data/emails/kroah_hartman.mbox"
            profile_path: "data/models/kroah_hartman_profile.json"
        style:
          alpha: 0.3
          min_email_words: 20
          date_range:
            start: "2015-01-01"
            end: "2023-12-31"
    """)
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_config(config_file)
    assert config.scoring.fallback_threshold == 0.80
