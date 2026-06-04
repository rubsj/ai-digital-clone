"""Groundedness scorer using HHEM-2.1-Open (ADR-020).

V0 aggregation (same pipeline position as the replaced cosine scorer):
  1. Split response into sentences (regex, no nltk).
  2. For each sentence: HHEM entailment score against each of the top-k chunks
     (chunk = premise, sentence = hypothesis).
  3. Max over the k entailment scores → per-sentence groundedness.
  4. Mean over sentences → groundedness_score ∈ [0, 1].

The model is loaded once at HHEMGroundednessScorer construction and held
resident (same lifecycle as the FAISS index). Zero paid API calls; all
inference local and in-process.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from src.evaluation.hhem.configuration_hhem_v2 import HHEMv2Config  # noqa: F401 — ensures config class is importable before from_pretrained
from src.evaluation.hhem.modeling_hhem_v2 import HHEMv2ForSequenceClassification
from src.schemas import RetrievalResult

_MIN_SENTENCE_CHARS = 10
_HHEM_HUB_ID = "vectara/hallucination_evaluation_model"
_HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"

# Module-level singleton — set by _get_singleton(), never by callers directly.
_singleton: Optional["HHEMGroundednessScorer"] = None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences via punctuation look-behind. No nltk."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_CHARS]


class HHEMGroundednessScorer:
    """HHEM-2.1-Open groundedness scorer — load once, hold resident."""

    def __init__(self) -> None:
        self._model = HHEMv2ForSequenceClassification.from_pretrained(
            _HHEM_HUB_ID,
            revision=_HHEM_REVISION,
            local_files_only=True,
        )
        self._model.eval()

    def score(
        self,
        response: str,
        chunks: list[RetrievalResult],
        top_k: int = 5,
    ) -> float:
        """V0 aggregation: per-sentence max over top-k chunks, mean over sentences."""
        if not response or not chunks:
            return 0.0

        sentences = _split_sentences(response)
        if not sentences:
            return 0.0

        top_chunks = chunks[:top_k]
        per_sentence_max: list[float] = []

        for sentence in sentences:
            pairs = [(chunk.chunk.content, sentence) for chunk in top_chunks]
            with torch.no_grad():
                raw_scores = self._model.predict(pairs)
            per_sentence_max.append(float(raw_scores.max().item()))

        return float(sum(per_sentence_max) / len(per_sentence_max))


def _get_singleton() -> HHEMGroundednessScorer:
    global _singleton
    if _singleton is None:
        _singleton = HHEMGroundednessScorer()
    return _singleton


def score_groundedness(
    response: str,
    chunks: list[RetrievalResult],
    top_k: int = 5,
    *,
    scorer: Optional[HHEMGroundednessScorer] = None,
) -> float:
    """Score groundedness via HHEM-2.1-Open (ADR-020, V0 aggregation)."""
    return (scorer or _get_singleton()).score(response, chunks, top_k)
