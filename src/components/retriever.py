"""Retriever Component: FAISS + Cohere two-stage retrieval (ADR-002, ADR-009).

LLM-free. Wraps the v1 src/rag/ pipeline (embed → FAISS top-20 → Cohere
rerank top-5). The Cohere client lives in src/rag/reranker.py and is invoked
through rerank_with_status() — this module never imports cohere directly.

Cohere fail-loud: run() records last_rerank_ran and logs an error when
reranking did not actually execute, so the original silent-failure bug (an
unset COHERE_API_KEY quietly degrading precision) cannot recur. ADR-002
graceful fallback to FAISS top-5 is preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import faiss

from src.config import AppConfig, load_config
from src.rag.embedder import embed_chunks
from src.rag.indexer import build_index, load_index, save_index
from src.rag.reranker import rerank_with_status
from src.rag.retriever import retrieve
from src.schemas import KnowledgeChunk, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_DIR = Path("data/rag/faiss_index")


class Retriever:
    """Two-stage retrieval Component: embed → FAISS top-20 → Cohere rerank top-5.

    The lifecycle has two distinct phases:

      Write-time — run rarely, only when the knowledge base changes:
          r = Retriever()
          r.build(chunks)   # embed all chunks, build the FAISS index, persist to disk

      Query-time — run per user query, fast, no rebuild:
          r = Retriever()              # auto-loads the persisted index from disk
          results = r.run("some query")

    build() is the expensive one-time setup; run() is the per-query path. They are
    separated so the corpus is never re-embedded on every query. When a persisted
    index already exists at index_dir, a freshly-constructed Retriever loads it and
    is ready to run() immediately — no build() needed.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        index_dir: Path = _DEFAULT_INDEX_DIR,
    ) -> None:
        self._config = config or load_config()
        self._index_dir = index_dir
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: list[dict] = []
        self.last_rerank_ran: bool = False

        if (index_dir / "index.faiss").exists():
            try:
                self._index, self._metadata = load_index(index_dir)
                logger.info(
                    "Loaded FAISS index from %s (%d vectors).",
                    index_dir,
                    self._index.ntotal,
                )
            except Exception as exc:
                logger.warning("Failed to load index from %s: %s", index_dir, exc)

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        """Embed chunks, build the FAISS index, and persist to disk."""
        dimension = self._config.embedding.dimension
        logger.info("Embedding %d chunks (provider=openai) …", len(chunks))
        embedded = embed_chunks(chunks, provider="openai")
        logger.info("Building FAISS IndexFlatIP (dim=%d) …", dimension)
        self._index, self._metadata = build_index(embedded, dimension=dimension)
        save_index(self._index, self._metadata, index_dir=self._index_dir)
        logger.info("Index saved to %s (%d vectors).", self._index_dir, self._index.ntotal)

    def run(self, query: str) -> list[RetrievalResult]:
        """Full pipeline: embed → FAISS top-20 → Cohere rerank → top-5.

        Sets self.last_rerank_ran to whether Cohere actually executed. Logs an
        error (not a swallowed warning) when it did not, so a silent precision
        regression is impossible. Raises RuntimeError if no index is available.
        """
        if self._index is None:
            raise RuntimeError(
                "Retriever.run() called before an index is available. "
                "Run Retriever.build(chunks) first, or inject an index."
            )

        candidates = retrieve(
            query,
            self._index,
            self._metadata,
            top_n=self._config.reranker.top_n_initial,
        )

        # Corpus has duplicate-content entries; dedup before rerank so Cohere selects top-5 from distinct passages.
        seen_content: set[str] = set()
        deduped: list[RetrievalResult] = []
        for r in candidates:
            if r.chunk.content not in seen_content:
                seen_content.add(r.chunk.content)
                deduped.append(r)
        candidates = deduped

        reranked, ran = rerank_with_status(
            query,
            candidates,
            model=self._config.reranker.model,
            top_n=self._config.reranker.top_n_final,
        )
        self.last_rerank_ran = ran
        if not ran and candidates:
            logger.error(
                "Retriever returned FAISS-only results for query %r — Cohere "
                "reranking did NOT run. Precision is degraded (ADR-002).",
                query,
            )
        return reranked
