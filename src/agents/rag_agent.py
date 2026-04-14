"""RAGAgent: facade for the full RAG retrieval pipeline.

NOT a CrewAI Crew — the Flow calls agent.retrieve(query) directly.
Pipeline: embed chunks → build FAISS index → save to disk (build time)
          embed query  → FAISS top-20     → Cohere rerank top-5 (query time)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import AppConfig, load_config
from src.rag.embedder import embed_chunks
from src.rag.indexer import build_index, load_index, save_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve
from src.schemas import KnowledgeChunk, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_DIR = Path("data/rag/faiss_index")


class RAGAgent:
    """Facade for the full RAG pipeline.

    Usage:
        agent = RAGAgent()
        agent.build(chunks)          # one-time: embed + index + persist
        results = agent.retrieve(q)  # query time: embed + FAISS + Cohere
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        index_dir: Path = _DEFAULT_INDEX_DIR,
    ) -> None:
        self._config = config or load_config()
        self._index_dir = index_dir
        self._index = None
        self._metadata: list[dict] = []

        # Try loading a pre-built index from disk
        index_path = index_dir / "index.faiss"
        if index_path.exists():
            try:
                self._index, self._metadata = load_index(index_dir)
                logger.info(
                    "Loaded FAISS index from %s (%d vectors).",
                    index_dir,
                    self._index.ntotal,
                )
            except Exception as exc:
                logger.warning("Failed to load index from %s: %s", index_dir, exc)

    # ------------------------------------------------------------------
    # Build (one-time setup)
    # ------------------------------------------------------------------

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        """Embed chunks, build FAISS index, and persist to disk.

        Overwrites any existing index at self._index_dir.
        """
        provider = "openai"  # primary per ADR-002
        dimension = self._config.embedding.dimension

        logger.info("Embedding %d chunks with provider=%s …", len(chunks), provider)
        embedded = embed_chunks(chunks, provider=provider)

        logger.info("Building FAISS IndexFlatIP (dim=%d) …", dimension)
        self._index, self._metadata = build_index(embedded, dimension=dimension)

        save_index(self._index, self._metadata, index_dir=self._index_dir)
        logger.info(
            "Index saved to %s (%d vectors).", self._index_dir, self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Retrieve (query time)
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Full pipeline: embed → FAISS top-20 → Cohere rerank → top-5.

        Raises RuntimeError if the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError(
                "RAGAgent.retrieve() called before index is built. "
                "Run RAGAgent.build(chunks) first."
            )

        top_n_initial = self._config.reranker.top_n_initial  # 20
        top_n_final = self._config.reranker.top_n_final       # 5
        model = self._config.reranker.model

        candidates = retrieve(
            query,
            self._index,
            self._metadata,
            top_n=top_n_initial,
        )

        return rerank(query, candidates, model=model, top_n=top_n_final)
