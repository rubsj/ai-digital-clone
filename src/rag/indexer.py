"""Build and persist a FAISS IndexFlatIP for KnowledgeChunk embeddings.

All embeddings must be L2-normalized before indexing (IndexFlatIP computes
dot-product similarity, which equals cosine similarity for unit vectors).
faiss.normalize_L2() is applied before index.add() as a belt-and-suspenders
safeguard even when embed_chunks() has already normalized.

Sidecar file:
  {index_dir}/index.faiss — the FAISS binary
  {index_dir}/metadata.json — list of chunk dicts (embedding excluded)
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.schemas import KnowledgeChunk


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_norms(embeddings: np.ndarray, tol: float = 1e-5) -> None:
    """Assert all row L2 norms are ≈ 1.0. Raises ValueError if not."""
    norms = np.linalg.norm(embeddings, axis=1)
    bad = np.where(np.abs(norms - 1.0) > tol)[0]
    if bad.size > 0:
        raise ValueError(
            f"Embeddings at indices {bad.tolist()} are not L2-normalized "
            f"(norms: {norms[bad].tolist()})"
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    chunks: list[KnowledgeChunk],
    dimension: int = 1536,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Build a FAISS IndexFlatIP from embedded KnowledgeChunks.

    Validates embeddings are present, L2-normalizes (via faiss.normalize_L2),
    then adds to index. Returns (index, metadata_list) where metadata_list
    excludes the embedding field.
    """
    if not chunks:
        raise ValueError("Cannot build index from empty chunk list.")

    embeddings = np.array(
        [c.embedding for c in chunks], dtype=np.float32
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        raise ValueError(
            f"Expected embeddings of shape (N, {dimension}), got {embeddings.shape}"
        )

    # Normalize in-place (FAISS mutates the array)
    faiss.normalize_L2(embeddings)
    _validate_norms(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    metadata = [c.model_dump(exclude={"embedding"}) for c in chunks]
    return index, metadata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    index_dir: Path = Path("data/rag/faiss_index"),
) -> None:
    """Write index.faiss and metadata.json to index_dir."""
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def load_index(
    index_dir: Path = Path("data/rag/faiss_index"),
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Read index.faiss and metadata.json from index_dir."""
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.json") as f:
        metadata = json.load(f)
    return index, metadata
