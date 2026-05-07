"""Embed KnowledgeChunk content using OpenAI (via LiteLLM) or MiniLM.

Two embedding providers:
  - openai: text-embedding-3-small (1536d) via LiteLLM, npz cache, batch 100
  - minilm: all-MiniLM-L6-v2 (384d) via SentenceTransformers, npz cache

All vectors are L2-normalized before caching and returning.
Cache keys are MD5 hashes of the input text.
Cache format: numpy .npz (keys array + vectors matrix) — ~13x smaller than JSON.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from rich.progress import Progress

import litellm
from sentence_transformers import SentenceTransformer

from src.schemas import KnowledgeChunk

# ---------------------------------------------------------------------------
# Module-level MiniLM model (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_MINILM_MODEL: Optional[SentenceTransformer] = None


def _get_minilm() -> SentenceTransformer:
    global _MINILM_MODEL
    if _MINILM_MODEL is None:
        _MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MINILM_MODEL


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _md5(text: str) -> str:
    """MD5 hash of text for use as cache key."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    """Load npz cache. Returns empty dict if file missing or corrupt."""
    if not cache_path.exists():
        return {}
    try:
        data = np.load(str(cache_path), allow_pickle=False)
        keys = data["keys"].tolist()
        vectors = data["vectors"]
        return {k: vectors[i].tolist() for i, k in enumerate(keys)}
    except Exception:
        return {}


def _save_cache(cache: dict[str, list[float]], cache_path: Path) -> None:
    """Persist cache to numpy npz, creating parent dirs as needed.

    np.savez_compressed appends .npz automatically, so strip it first.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stem = str(cache_path)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    if not cache:
        keys_arr: np.ndarray = np.array([], dtype="U1")
        vectors_arr = np.zeros((0, 0), dtype=np.float32)
    else:
        keys_arr = np.array(list(cache.keys()), dtype=str)
        vectors_arr = np.array(list(cache.values()), dtype=np.float32)
    np.savez_compressed(stem, keys=keys_arr, vectors=vectors_arr)


# ---------------------------------------------------------------------------
# OpenAI embeddings via LiteLLM
# ---------------------------------------------------------------------------


def embed_openai(
    texts: list[str],
    cache_path: Path = Path("data/cache/embeddings_openai.npz"),
    batch_size: int = 100,
) -> list[np.ndarray]:
    """Embed texts with text-embedding-3-small via LiteLLM.

    Caches by MD5. L2-normalizes before caching. Returns list of 1536-d arrays.
    """
    if not texts:
        return []

    cache = _load_cache(cache_path)
    keys = [_md5(t) for t in texts]

    # Separate cached from uncached
    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]
    uncached_texts = [texts[i] for i in uncached_indices]

    if uncached_texts:
        with Progress() as progress:
            task = progress.add_task("Embedding (OpenAI)...", total=len(uncached_texts))
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[batch_start : batch_start + batch_size]
                response = litellm.embedding(model="text-embedding-3-small", input=batch)
                for j, item in enumerate(response.data):
                    raw = item["embedding"] if isinstance(item, dict) else item.embedding
                    vec = np.array(raw, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    global_i = uncached_indices[batch_start + j]
                    cache[keys[global_i]] = vec.tolist()
                progress.advance(task, len(batch))

        _save_cache(cache, cache_path)

    return [np.array(cache[k], dtype=np.float32) for k in keys]


# ---------------------------------------------------------------------------
# MiniLM embeddings via SentenceTransformers
# ---------------------------------------------------------------------------


def embed_minilm(
    texts: list[str],
    cache_path: Path = Path("data/cache/embeddings_minilm.npz"),
) -> list[np.ndarray]:
    """Embed texts with all-MiniLM-L6-v2.

    Caches by MD5. normalize_embeddings=True (L2-normalized by ST). Returns 384-d arrays.
    """
    if not texts:
        return []

    cache = _load_cache(cache_path)
    keys = [_md5(t) for t in texts]

    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]
    uncached_texts = [texts[i] for i in uncached_indices]

    if uncached_texts:
        model = _get_minilm()
        vecs = model.encode(uncached_texts, normalize_embeddings=True)
        for j, idx in enumerate(uncached_indices):
            cache[keys[idx]] = vecs[j].tolist()
        _save_cache(cache, cache_path)

    return [np.array(cache[k], dtype=np.float32) for k in keys]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[KnowledgeChunk],
    provider: Literal["openai", "minilm"] = "openai",
    cache_path: Optional[Path] = None,
) -> list[KnowledgeChunk]:
    """Embed all chunks, returning new KnowledgeChunk objects with embedding set.

    Uses model_copy(update={"embedding": vec}) — immutable Pydantic v2 pattern.
    """
    if not chunks:
        return []

    texts = [c.content for c in chunks]

    if provider == "minilm":
        path = cache_path or Path("data/cache/embeddings_minilm.npz")
        vecs = embed_minilm(texts, cache_path=path)
    else:
        path = cache_path or Path("data/cache/embeddings_openai.npz")
        vecs = embed_openai(texts, cache_path=path)

    return [c.model_copy(update={"embedding": v}) for c, v in zip(chunks, vecs)]


def embed_query(
    query: str,
    provider: Literal["openai", "minilm"] = "openai",
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """Embed a single query string. Returns a normalized vector."""
    if provider == "minilm":
        path = cache_path or Path("data/cache/embeddings_minilm.npz")
        vecs = embed_minilm([query], cache_path=path)
    else:
        path = cache_path or Path("data/cache/embeddings_openai.npz")
        vecs = embed_openai([query], cache_path=path)
    return vecs[0]
