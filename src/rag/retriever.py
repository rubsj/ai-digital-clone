"""FAISS-based retrieval for KnowledgeChunks.

retrieve() embeds the query, searches the IndexFlatIP, and returns top-N
RetrievalResult objects sorted by descending dot-product score.
"""

from __future__ import annotations

from typing import Literal

import faiss
import numpy as np

from src.rag.embedder import embed_query
from src.schemas import KnowledgeChunk, RetrievalResult


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    top_n: int = 20,
    provider: Literal["openai", "minilm"] = "openai",
) -> list[RetrievalResult]:
    """Embed query → FAISS search → return top-N RetrievalResults.

    Steps:
      1. embed_query() returns a normalized 1-D vector
      2. Reshape to (1, dim) float32 and normalize again (belt-and-suspenders)
      3. index.search() → (scores, indices) both shape (1, k)
      4. Filter -1 padding indices (FAISS returns -1 when ntotal < k)
      5. Reconstruct KnowledgeChunk from metadata dict, rank by position
    """
    if index.ntotal == 0:
        return []

    effective_k = min(top_n, index.ntotal)

    query_vec = embed_query(query, provider=provider)
    query_2d = query_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_2d)

    scores, indices = index.search(query_2d, effective_k)

    results: list[RetrievalResult] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        chunk = KnowledgeChunk(**metadata[idx])
        results.append(RetrievalResult(chunk=chunk, score=float(score), rank=rank))

    return results
