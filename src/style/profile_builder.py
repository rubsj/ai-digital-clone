"""Style profile builder — batch aggregation and incremental EMA updates.

Public API:
    build_profile_batch(leader_name, features_list, alpha) -> StyleProfile
    update_profile_incremental(profile, new_features) -> StyleProfile
    save_profile(profile, path) -> None
    load_profile(path) -> StyleProfile

Batch aggregation: element-wise mean across all per-email feature vectors.
Incremental update: exponential moving average with profile.alpha.

Callers must supply the save/load path — get it from config.leaders[name].profile_path,
never hardcode it here.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.schemas import StyleFeatures, StyleProfile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_profile_batch(
    leader_name: str,
    features_list: list[StyleFeatures],
    alpha: float = 0.3,
) -> StyleProfile:
    """Build a StyleProfile by averaging features across all emails.

    Scalar fields: mean across all emails.
    Dict fields: union all keys, mean per key across emails that contain it.
    common_phrases: flatten + Counter, top-20 by frequency.
    style_vector: computed from the aggregated StyleFeatures via to_vector().

    Raises ValueError if features_list is empty.
    """
    if not features_list:
        raise ValueError("features_list must not be empty")

    agg = _aggregate_features(features_list)
    return StyleProfile(
        leader_name=leader_name,
        features=agg,
        style_vector=agg.to_vector(),
        email_count=len(features_list),
        last_updated=datetime.now(tz=timezone.utc),
        alpha=alpha,
    )


def update_profile_incremental(
    profile: StyleProfile,
    new_features: StyleFeatures,
) -> StyleProfile:
    """Return a NEW StyleProfile updated with one new email via EMA.

    Vector update: updated = (1 - alpha) * current + alpha * new, clipped to [0, 1].
    StyleFeatures scalars and dicts are also EMA-updated for JSON introspection.
    New dict keys: initialized at alpha * new_value.
    Absent dict keys in new: decayed at (1 - alpha) * old_value.
    """
    alpha = profile.alpha
    new_vec = new_features.to_vector()
    updated_vec = np.clip(
        (1.0 - alpha) * profile.style_vector + alpha * new_vec,
        0.0,
        1.0,
    )
    updated_features = _ema_features(profile.features, new_features, alpha)
    return StyleProfile(
        leader_name=profile.leader_name,
        features=updated_features,
        style_vector=updated_vec,
        email_count=profile.email_count + 1,
        last_updated=datetime.now(tz=timezone.utc),
        alpha=alpha,
    )


def save_profile(profile: StyleProfile, path: Path) -> None:
    """Write profile to JSON. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2))


def load_profile(path: Path) -> StyleProfile:
    """Load a StyleProfile from a JSON file written by save_profile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    return StyleProfile.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_features(features_list: list[StyleFeatures]) -> StyleFeatures:
    """Element-wise mean of all features across the list."""
    n = len(features_list)

    # --- Scalar fields ---
    def _mean_scalar(attr: str) -> float:
        return statistics.mean(getattr(f, attr) for f in features_list)

    avg_message_length = _mean_scalar("avg_message_length")
    capitalization_ratio = _mean_scalar("capitalization_ratio")
    question_frequency = _mean_scalar("question_frequency")
    vocabulary_richness = _mean_scalar("vocabulary_richness")
    formality_level = _mean_scalar("formality_level")
    technical_terminology = _mean_scalar("technical_terminology")
    code_snippet_freq = _mean_scalar("code_snippet_freq")
    quote_reply_ratio = _mean_scalar("quote_reply_ratio")
    technical_depth = _mean_scalar("technical_depth")

    # --- Dict fields: union keys, mean per key over emails that contain it ---
    greeting_patterns = _aggregate_dict([f.greeting_patterns for f in features_list])
    punctuation_patterns = _aggregate_dict([f.punctuation_patterns for f in features_list])
    reasoning_patterns = _aggregate_dict([f.reasoning_patterns for f in features_list])
    sentiment_distribution = _aggregate_dict([f.sentiment_distribution for f in features_list])
    patch_language = _aggregate_dict([f.patch_language for f in features_list])

    # --- common_phrases: flatten all lists, top-20 by frequency ---
    all_phrases: list[str] = []
    for f in features_list:
        all_phrases.extend(f.common_phrases)
    phrase_counts = Counter(all_phrases)
    common_phrases = [p for p, _ in phrase_counts.most_common(20)]

    return StyleFeatures(
        avg_message_length=avg_message_length,
        greeting_patterns=greeting_patterns,
        punctuation_patterns=punctuation_patterns,
        capitalization_ratio=capitalization_ratio,
        question_frequency=question_frequency,
        vocabulary_richness=vocabulary_richness,
        common_phrases=common_phrases,
        reasoning_patterns=reasoning_patterns,
        sentiment_distribution=sentiment_distribution,
        formality_level=formality_level,
        technical_terminology=technical_terminology,
        code_snippet_freq=code_snippet_freq,
        quote_reply_ratio=quote_reply_ratio,
        patch_language=patch_language,
        technical_depth=technical_depth,
    )


def _aggregate_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Union all keys; value = mean over ALL dicts (absent key contributes 0.0).

    Treating absent keys as 0.0 produces the correct frequency semantics for
    sparse features like greeting_patterns (most emails return {}) and
    sentiment_distribution (emails with no emotional content return {}).
    The old "mean over emails that contain the key" behaviour inflated values
    for sparse features: a key appearing in 1 of 6000 emails would still get
    value 1.0 in the profile, matching no individual email at cosine time.
    """
    all_keys: set[str] = set()
    for d in dicts:
        all_keys.update(d.keys())

    result: dict[str, float] = {}
    n = len(dicts)
    for key in all_keys:
        # Include 0.0 for every dict that doesn't contain the key
        total = sum(d.get(key, 0.0) for d in dicts)
        result[key] = total / n
    return result


def _ema_features(
    old: StyleFeatures,
    new: StyleFeatures,
    alpha: float,
) -> StyleFeatures:
    """EMA update of StyleFeatures fields individually."""

    def _ema_scalar(old_val: float, new_val: float) -> float:
        return float(np.clip((1.0 - alpha) * old_val + alpha * new_val, 0.0, 1.0))

    def _ema_dict(old_d: dict[str, float], new_d: dict[str, float]) -> dict[str, float]:
        all_keys = set(old_d) | set(new_d)
        result: dict[str, float] = {}
        for k in all_keys:
            if k in old_d and k in new_d:
                result[k] = float(np.clip(
                    (1.0 - alpha) * old_d[k] + alpha * new_d[k], 0.0, 1.0
                ))
            elif k in old_d:
                # Key absent in new: decay
                result[k] = float(np.clip((1.0 - alpha) * old_d[k], 0.0, 1.0))
            else:
                # New key: initialize at alpha * new_value
                result[k] = float(np.clip(alpha * new_d[k], 0.0, 1.0))
        return result

    # common_phrases: merge old + new, keep top-20 by combined frequency
    phrase_counts = Counter(old.common_phrases) + Counter(new.common_phrases)
    updated_phrases = [p for p, _ in phrase_counts.most_common(20)]

    return StyleFeatures(
        avg_message_length=_ema_scalar(old.avg_message_length, new.avg_message_length),
        greeting_patterns=_ema_dict(old.greeting_patterns, new.greeting_patterns),
        punctuation_patterns=_ema_dict(old.punctuation_patterns, new.punctuation_patterns),
        capitalization_ratio=_ema_scalar(old.capitalization_ratio, new.capitalization_ratio),
        question_frequency=_ema_scalar(old.question_frequency, new.question_frequency),
        vocabulary_richness=_ema_scalar(old.vocabulary_richness, new.vocabulary_richness),
        common_phrases=updated_phrases,
        reasoning_patterns=_ema_dict(old.reasoning_patterns, new.reasoning_patterns),
        sentiment_distribution=_ema_dict(old.sentiment_distribution, new.sentiment_distribution),
        formality_level=_ema_scalar(old.formality_level, new.formality_level),
        technical_terminology=_ema_scalar(old.technical_terminology, new.technical_terminology),
        code_snippet_freq=_ema_scalar(old.code_snippet_freq, new.code_snippet_freq),
        quote_reply_ratio=_ema_scalar(old.quote_reply_ratio, new.quote_reply_ratio),
        patch_language=_ema_dict(old.patch_language, new.patch_language),
        technical_depth=_ema_scalar(old.technical_depth, new.technical_depth),
    )
