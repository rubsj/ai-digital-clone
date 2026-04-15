"""Groundedness scorer: sentence-level semantic similarity against retrieved chunks.

Algorithm:
  1. Split response into sentences (regex, no nltk).
  2. Batch-embed all sentences in ONE embed_openai() call (MD5-cached).
  3. Reuse chunk.embedding from RAG pipeline; only re-embed missing ones.
  4. For each sentence: max cosine-sim across top-k chunk embeddings.
  5. Average the per-sentence maxima → groundedness_score ∈ [0, 1].

Target: > 0.60 on in-domain queries (per PRD quality table).
"""

from __future__ import annotations

import re

import numpy as np

from src.rag.embedder import embed_openai
from src.schemas import RetrievalResult

# Sentences shorter than this (in chars) are skipped to avoid noise from
# fragment phrases like "So." or "Yes."
_MIN_SENTENCE_CHARS = 10


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences via punctuation look-behind. No nltk."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_CHARS]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, returns 0.0 for zero-norm vectors."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def score_groundedness(
    response: str,
    chunks: list[RetrievalResult],
    top_k: int = 5,
) -> float:
    """Average of per-sentence max cosine similarity against top-k chunks.

    Args:
        response: The generated response text.
        chunks:   Retrieved chunks from RAG pipeline (may already have .embedding).
        top_k:    Maximum number of chunks to compare against.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 for empty response or no chunks.
    """
    if not response or not chunks:
        return 0.0

    sentences = _split_sentences(response)
    if not sentences:
        return 0.0

    # --- 1. Batch-embed all sentences in ONE API call ---
    sentence_vecs = embed_openai(sentences)

    # --- 2. Gather chunk embeddings; re-embed missing ones in ONE batch call ---
    top_chunks = chunks[:top_k]
    chunk_vecs: list[np.ndarray] = []
    missing_indices: list[int] = []
    missing_texts: list[str] = []

    for i, rr in enumerate(top_chunks):
        if rr.chunk.embedding is not None:
            chunk_vecs.append(rr.chunk.embedding)
        else:
            chunk_vecs.append(None)  # type: ignore[arg-type]
            missing_indices.append(i)
            missing_texts.append(rr.chunk.content)

    if missing_texts:
        embedded_missing = embed_openai(missing_texts)
        for idx, vec in zip(missing_indices, embedded_missing):
            chunk_vecs[idx] = vec

    # --- 3. Per-sentence max cosine similarity ---
    per_sentence_max: list[float] = []
    for s_vec in sentence_vecs:
        max_sim = max(_cosine(s_vec, c_vec) for c_vec in chunk_vecs)
        per_sentence_max.append(max_sim)

    return float(np.mean(per_sentence_max))
