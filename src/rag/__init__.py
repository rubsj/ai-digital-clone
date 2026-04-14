"""Public re-exports for the src.rag package."""

from src.rag.chunker import chunk_baseline, chunk_documents, chunk_semantic
from src.rag.citation_extractor import extract_citations
from src.rag.corpus_loader import RawDocument, load_corpus
from src.rag.embedder import embed_chunks, embed_query
from src.rag.indexer import build_index, load_index, save_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve

__all__ = [
    "RawDocument",
    "load_corpus",
    "chunk_baseline",
    "chunk_semantic",
    "chunk_documents",
    "embed_chunks",
    "embed_query",
    "build_index",
    "save_index",
    "load_index",
    "retrieve",
    "rerank",
    "extract_citations",
]
