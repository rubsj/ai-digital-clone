"""Style similarity scorer using cosine distance on 15-dim feature vectors.

Public API:
    cosine_similarity(a, b) -> float   — clamped to [0, 1], 0.0 for zero-norm
    score_style(profile, features) -> float
"""

from __future__ import annotations

import numpy as np

from src.schemas import StyleFeatures, StyleProfile


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two non-negative vectors, clamped to [0, 1].

    Returns 0.0 if either vector has zero norm (avoids division by zero).
    Since all feature values are in [0, 1], the result is naturally in [0, 1].
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(similarity, 0.0, 1.0))


def score_style(profile: StyleProfile, response_features: StyleFeatures) -> float:
    """Cosine similarity between a leader's profile vector and a response's feature vector.

    Returns a float in [0, 1]. A score >= 0.90 indicates strong style match.
    """
    return cosine_similarity(profile.style_vector, response_features.to_vector())
