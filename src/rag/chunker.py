"""Split RawDocuments into KnowledgeChunk objects using two strategies.

Baseline: RecursiveCharacterTextSplitter (500 chars / 50 overlap).
Experiment: MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter.
chunk_index is globally sequential across all documents.
"""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import AppConfig
from src.rag.corpus_loader import RawDocument
from src.schemas import KnowledgeChunk

_MARKDOWN_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3")]


def chunk_baseline(
    documents: list[RawDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    """RecursiveCharacterTextSplitter with global sequential chunk_index.

    source_topic comes from RawDocument.topic. embedding left as None.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[KnowledgeChunk] = []
    idx = 0
    for doc in documents:
        if not doc.text.strip():
            continue
        for piece in splitter.split_text(doc.text):
            chunks.append(
                KnowledgeChunk(
                    content=piece,
                    source_topic=doc.topic,
                    source_field=doc.field,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_semantic(
    documents: list[RawDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    """MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter.

    Split on headers first to preserve semantic section boundaries, then
    size-split any section that exceeds chunk_size. source_topic comes from
    the header metadata when present, falling back to RawDocument.topic.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[KnowledgeChunk] = []
    idx = 0
    for doc in documents:
        if not doc.text.strip():
            continue
        try:
            sections = header_splitter.split_text(doc.text)
        except Exception:
            # Fall back to pure size splitting if markdown parsing fails
            sections = []

        if not sections:
            # No headers found — treat whole doc as one section
            pieces = size_splitter.split_text(doc.text)
            for piece in pieces:
                chunks.append(
                    KnowledgeChunk(
                        content=piece,
                        source_topic=doc.topic,
                        source_field=doc.field,
                        chunk_index=idx,
                    )
                )
                idx += 1
            continue

        for section in sections:
            # Extract topic from header metadata (H1 > H2 > H3 > doc.topic)
            meta = section.metadata if hasattr(section, "metadata") else {}
            topic = meta.get("H1") or meta.get("H2") or meta.get("H3") or doc.topic

            text = section.page_content if hasattr(section, "page_content") else str(section)
            if not text.strip():
                continue

            # Size-split sections that exceed chunk_size
            if len(text) > chunk_size:
                pieces = size_splitter.split_text(text)
            else:
                pieces = [text]

            for piece in pieces:
                if not piece.strip():
                    continue
                chunks.append(
                    KnowledgeChunk(
                        content=piece,
                        source_topic=topic,
                        source_field=doc.field,
                        chunk_index=idx,
                    )
                )
                idx += 1

    return chunks


def chunk_documents(
    documents: list[RawDocument],
    config: AppConfig,
    strategy: str = "baseline",
) -> list[KnowledgeChunk]:
    """Dispatch to baseline or semantic strategy using config chunk sizes."""
    size = config.chunking.chunk_size
    overlap = config.chunking.chunk_overlap
    if strategy == "semantic":
        return chunk_semantic(documents, chunk_size=size, chunk_overlap=overlap)
    return chunk_baseline(documents, chunk_size=size, chunk_overlap=overlap)
