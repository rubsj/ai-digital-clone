"""Tests for src/style/profile_builder.py.

Coverage target: >= 90% of profile_builder.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.schemas import StyleFeatures, StyleProfile
from src.style.profile_builder import (
    build_profile_batch,
    load_profile,
    save_profile,
    update_profile_incremental,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.4,
        greeting_patterns={"hi": 0.0, "hello": 0.0, "hey": 0.0, "dear": 0.0, "none": 1.0},
        punctuation_patterns={"exclamation": 0.1, "question": 0.05, "ellipsis": 0.0,
                               "dash": 0.02, "semicolon": 0.01, "colon": 0.03},
        capitalization_ratio=0.08,
        question_frequency=0.15,
        vocabulary_richness=0.65,
        common_phrases=["memory leak", "race condition", "the kernel"],
        reasoning_patterns={"because": 0.2, "therefore": 0.05, "however": 0.1,
                             "but": 0.3, "so": 0.1, "if_then": 0.0, "the_thing_is": 0.05},
        sentiment_distribution={"positive": 0.3, "negative": 0.4, "neutral": 0.3},
        formality_level=0.35,
        technical_terminology=0.6,
        code_snippet_freq=0.15,
        quote_reply_ratio=0.2,
        patch_language={"applied": 0.0, "nak": 1.0, "acked_by": 0.0, "reviewed_by": 0.0,
                        "looks_good": 0.0, "please_fix": 1.0, "resubmit": 0.0},
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
# build_profile_batch — single email
# ---------------------------------------------------------------------------


def test_batch_single_email_returns_profile():
    f = _make_features()
    profile = build_profile_batch("Linus Torvalds", [f])
    assert isinstance(profile, StyleProfile)
    assert profile.leader_name == "Linus Torvalds"
    assert profile.email_count == 1


def test_batch_single_email_vector_matches_features():
    f = _make_features()
    profile = build_profile_batch("Linus Torvalds", [f])
    np.testing.assert_array_almost_equal(profile.style_vector, f.to_vector())


def test_batch_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        build_profile_batch("Linus Torvalds", [])


# ---------------------------------------------------------------------------
# build_profile_batch — multi-email averaging
# ---------------------------------------------------------------------------


def test_batch_multi_email_count():
    features = [_make_features() for _ in range(5)]
    profile = build_profile_batch("Linus Torvalds", features)
    assert profile.email_count == 5


def test_batch_scalar_averaging():
    f1 = _make_features(avg_message_length=0.2)
    f2 = _make_features(avg_message_length=0.8)
    profile = build_profile_batch("test", [f1, f2])
    assert abs(profile.features.avg_message_length - 0.5) < 1e-9


def test_batch_dict_key_merging():
    # f1 has key "applied", f2 has key "nak" — both should appear in result
    f1 = _make_features(patch_language={"applied": 1.0, "nak": 0.0})
    f2 = _make_features(patch_language={"applied": 0.0, "nak": 1.0})
    profile = build_profile_batch("test", [f1, f2])
    assert "applied" in profile.features.patch_language
    assert "nak" in profile.features.patch_language
    assert abs(profile.features.patch_language["applied"] - 0.5) < 1e-9
    assert abs(profile.features.patch_language["nak"] - 0.5) < 1e-9


def test_batch_dict_key_partial_coverage():
    # Key only present in one of two dicts — mean over emails that contain it
    f1 = _make_features(reasoning_patterns={"because": 0.6})
    f2 = _make_features(reasoning_patterns={})
    profile = build_profile_batch("test", [f1, f2])
    # "because" appears in only f1, so mean = 0.6 / 1 = 0.6
    assert abs(profile.features.reasoning_patterns["because"] - 0.6) < 1e-9


def test_batch_phrase_top_20():
    # 25 unique phrases across emails — result must be <= 20
    phrases_a = [f"phrase_{i}" for i in range(15)]
    phrases_b = [f"phrase_{i}" for i in range(10, 25)]
    f1 = _make_features(common_phrases=phrases_a)
    f2 = _make_features(common_phrases=phrases_b)
    profile = build_profile_batch("test", [f1, f2])
    assert len(profile.features.common_phrases) <= 20


def test_batch_phrase_most_frequent_kept():
    # "memory leak" appears in both — should be in top-20
    f1 = _make_features(common_phrases=["memory leak", "race condition"])
    f2 = _make_features(common_phrases=["memory leak", "kernel bug"])
    profile = build_profile_batch("test", [f1, f2])
    assert "memory leak" in profile.features.common_phrases


def test_batch_vector_shape():
    features = [_make_features() for _ in range(3)]
    profile = build_profile_batch("test", features)
    assert profile.style_vector.shape == (15,)


def test_batch_vector_all_in_range():
    features = [_make_features() for _ in range(3)]
    profile = build_profile_batch("test", features)
    assert np.all(profile.style_vector >= 0.0)
    assert np.all(profile.style_vector <= 1.0)


def test_batch_alpha_stored():
    features = [_make_features()]
    profile = build_profile_batch("test", features, alpha=0.5)
    assert profile.alpha == 0.5


# ---------------------------------------------------------------------------
# update_profile_incremental — EMA behaviour
# ---------------------------------------------------------------------------


def test_incremental_vector_moves_by_alpha():
    profile = _make_profile()
    old_vec = profile.style_vector.copy()
    new_features = _make_features(avg_message_length=1.0)  # push one dimension high
    updated = update_profile_incremental(profile, new_features)
    new_vec = new_features.to_vector()
    expected = np.clip(0.7 * old_vec + 0.3 * new_vec, 0.0, 1.0)
    np.testing.assert_array_almost_equal(updated.style_vector, expected)


def test_incremental_email_count_increments():
    profile = _make_profile()
    updated = update_profile_incremental(profile, _make_features())
    assert updated.email_count == profile.email_count + 1


def test_incremental_vector_clipped_to_range():
    # Force a vector beyond [0,1] through EMA — result must be clipped
    profile = _make_profile()
    # Use features that will push vector values high
    extreme_features = _make_features(
        avg_message_length=1.0,
        capitalization_ratio=1.0,
        technical_terminology=1.0,
    )
    updated = update_profile_incremental(profile, extreme_features)
    assert np.all(updated.style_vector >= 0.0)
    assert np.all(updated.style_vector <= 1.0)


def test_incremental_new_dict_key_initialized():
    # Profile has no "resubmit" in patch_language; new email does
    profile = _make_profile(
        features=_make_features(patch_language={"applied": 0.5})
    )
    # Rebuild profile vector to match modified features
    profile = _make_profile(
        features=_make_features(patch_language={"applied": 0.5}),
    )
    new_f = _make_features(patch_language={"applied": 0.5, "resubmit": 1.0})
    updated = update_profile_incremental(profile, new_f)
    assert "resubmit" in updated.features.patch_language
    # New key: alpha * 1.0 = 0.3
    assert abs(updated.features.patch_language["resubmit"] - 0.3) < 1e-6


def test_incremental_absent_dict_key_decays():
    # Profile has "nak"; new email has no "nak" — should decay
    profile = _make_profile(
        features=_make_features(patch_language={"nak": 1.0}),
    )
    new_f = _make_features(patch_language={})
    updated = update_profile_incremental(profile, new_f)
    assert "nak" in updated.features.patch_language
    # Absent in new: (1 - 0.3) * 1.0 = 0.7
    assert abs(updated.features.patch_language["nak"] - 0.7) < 1e-6


def test_incremental_returns_new_profile():
    profile = _make_profile()
    updated = update_profile_incremental(profile, _make_features())
    assert updated is not profile


def test_incremental_leader_name_preserved():
    profile = _make_profile()
    updated = update_profile_incremental(profile, _make_features())
    assert updated.leader_name == profile.leader_name


def test_incremental_alpha_preserved():
    profile = _make_profile()
    updated = update_profile_incremental(profile, _make_features())
    assert updated.alpha == profile.alpha


# ---------------------------------------------------------------------------
# save_profile / load_profile
# ---------------------------------------------------------------------------


def test_save_creates_file(tmp_path):
    profile = _make_profile()
    path = tmp_path / "torvalds_profile.json"
    save_profile(profile, path)
    assert path.exists()


def test_save_creates_parent_dirs(tmp_path):
    profile = _make_profile()
    path = tmp_path / "nested" / "deep" / "profile.json"
    save_profile(profile, path)
    assert path.exists()


def test_save_load_roundtrip(tmp_path):
    profile = _make_profile()
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.leader_name == profile.leader_name
    assert loaded.email_count == profile.email_count
    assert isinstance(loaded.style_vector, np.ndarray)
    np.testing.assert_array_almost_equal(loaded.style_vector, profile.style_vector)


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_profile(Path("/nonexistent/profile.json"))


def test_save_load_features_preserved(tmp_path):
    profile = _make_profile()
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert abs(loaded.features.capitalization_ratio - profile.features.capitalization_ratio) < 1e-6
    assert loaded.features.common_phrases == profile.features.common_phrases
