"""Tests for src/style/style_scorer.py.

Coverage target: >= 90% of style_scorer.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.schemas import StyleFeatures, StyleProfile
from src.style.style_scorer import cosine_similarity, score_style


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.4,
        greeting_patterns={"hi": 0.0, "none": 1.0},
        punctuation_patterns={"exclamation": 0.1, "question": 0.05, "ellipsis": 0.0,
                               "dash": 0.02, "semicolon": 0.01, "colon": 0.03},
        capitalization_ratio=0.08,
        question_frequency=0.15,
        vocabulary_richness=0.65,
        common_phrases=["memory leak", "race condition"],
        reasoning_patterns={"because": 0.2, "but": 0.3},
        sentiment_distribution={"positive": 0.3, "negative": 0.4, "neutral": 0.3},
        formality_level=0.35,
        technical_terminology=0.6,
        code_snippet_freq=0.15,
        quote_reply_ratio=0.2,
        patch_language={"nak": 1.0, "please_fix": 1.0},
        technical_depth=0.55,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_profile(features: StyleFeatures | None = None, **kwargs) -> StyleProfile:
    f = features or _make_features()
    defaults = dict(
        leader_name="Linus Torvalds",
        features=f,
        style_vector=f.to_vector(),
        email_count=100,
        last_updated=datetime(2023, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )
    return StyleProfile(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# cosine_similarity — edge cases
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors():
    v = np.array([0.3, 0.5, 0.2, 0.8, 0.1], dtype=np.float64)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_zero_norm_a():
    a = np.zeros(5, dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_zero_norm_b():
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_both_zero():
    a = np.zeros(5, dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_orthogonal_non_negative():
    # With non-negative vectors, true orthogonality requires one to be zero
    # Test near-orthogonal case with minimal overlap
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_returns_float():
    v = np.array([0.5, 0.5], dtype=np.float64)
    result = cosine_similarity(v, v)
    assert isinstance(result, float)


def test_cosine_result_in_range():
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.random(15)
        b = rng.random(15)
        result = cosine_similarity(a, b)
        assert 0.0 <= result <= 1.0


def test_cosine_similar_vectors_high_score():
    a = np.array([0.5, 0.6, 0.4, 0.7, 0.3], dtype=np.float64)
    b = a + np.array([0.01, -0.01, 0.02, -0.02, 0.01], dtype=np.float64)
    assert cosine_similarity(a, b) > 0.99


def test_cosine_dissimilar_vectors_lower_score():
    a = np.array([0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_style
# ---------------------------------------------------------------------------


def test_score_style_same_features_near_one():
    f = _make_features()
    profile = _make_profile(features=f)
    score = score_style(profile, f)
    assert score > 0.99


def test_score_style_returns_float():
    profile = _make_profile()
    score = score_style(profile, _make_features())
    assert isinstance(score, float)


def test_score_style_in_range():
    profile = _make_profile()
    score = score_style(profile, _make_features())
    assert 0.0 <= score <= 1.0


def test_score_style_different_features_lower():
    # Technical profile vs casual features
    technical = _make_features(
        technical_terminology=0.9,
        technical_depth=0.9,
        capitalization_ratio=0.2,
        formality_level=0.2,
    )
    casual = _make_features(
        technical_terminology=0.0,
        technical_depth=0.0,
        capitalization_ratio=0.0,
        formality_level=0.9,
    )
    profile = _make_profile(features=technical)
    score_same = score_style(profile, technical)
    score_diff = score_style(profile, casual)
    assert score_same > score_diff


def test_score_style_zero_vector_returns_zero():
    # Profile with zero vector (all features zero)
    zero_features = StyleFeatures()  # all defaults = 0.0, empty dicts
    profile = _make_profile(
        features=zero_features,
        style_vector=np.zeros(15, dtype=np.float64),
    )
    score = score_style(profile, _make_features())
    assert score == 0.0
