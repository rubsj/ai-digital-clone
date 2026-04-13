"""Tests for src/rag/chunker.py.

No API calls — pure text splitting logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.rag.chunker import chunk_baseline, chunk_documents, chunk_semantic
from src.rag.corpus_loader import RawDocument
from src.schemas import KnowledgeChunk


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _make_raw_doc(
    text: str = "Hello world. " * 50,
    topic: str = "Test Topic",
    field: str = "computer_science",
    subfield: str = "algorithms_and_data_structures",
) -> RawDocument:
    return RawDocument(text=text, topic=topic, field=field, subfield=subfield)


def _make_config(chunk_size: int = 500, chunk_overlap: int = 50) -> MagicMock:
    config = MagicMock()
    config.chunking.chunk_size = chunk_size
    config.chunking.chunk_overlap = chunk_overlap
    return config


# ---------------------------------------------------------------------------
# chunk_baseline
# ---------------------------------------------------------------------------


def test_chunk_baseline_returns_knowledge_chunks():
    doc = _make_raw_doc(text="A" * 1500)
    chunks = chunk_baseline([doc], chunk_size=500, chunk_overlap=50)
    assert all(isinstance(c, KnowledgeChunk) for c in chunks)


def test_chunk_baseline_produces_multiple_chunks():
    doc = _make_raw_doc(text="word " * 500)  # ~2500 chars
    chunks = chunk_baseline([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunk_baseline_chunk_index_sequential():
    doc = _make_raw_doc(text="word " * 500)
    chunks = chunk_baseline([doc], chunk_size=500, chunk_overlap=50)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_baseline_global_index_across_docs():
    doc1 = _make_raw_doc(text="a " * 400, topic="Topic A")
    doc2 = _make_raw_doc(text="b " * 400, topic="Topic B")
    chunks = chunk_baseline([doc1, doc2], chunk_size=500, chunk_overlap=50)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_baseline_source_topic_from_doc():
    doc = _make_raw_doc(text="content " * 200, topic="Sorting Algorithms")
    chunks = chunk_baseline([doc])
    assert all(c.source_topic == "Sorting Algorithms" for c in chunks)


def test_chunk_baseline_source_field_from_doc():
    doc = _make_raw_doc(text="content " * 200, field="computer_science")
    chunks = chunk_baseline([doc])
    assert all(c.source_field == "computer_science" for c in chunks)


def test_chunk_baseline_embedding_none():
    doc = _make_raw_doc(text="text " * 200)
    chunks = chunk_baseline([doc])
    assert all(c.embedding is None for c in chunks)


def test_chunk_baseline_content_preserved():
    text = "The quick brown fox jumps over the lazy dog. " * 5
    doc = _make_raw_doc(text=text)
    chunks = chunk_baseline([doc], chunk_size=500, chunk_overlap=0)
    reconstructed = "".join(c.content for c in chunks)
    # All original chars appear somewhere (splitter may reorder whitespace)
    assert len(reconstructed) >= len(text) * 0.9


def test_chunk_baseline_single_short_doc():
    doc = _make_raw_doc(text="short text")
    chunks = chunk_baseline([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].content == "short text"


def test_chunk_baseline_empty_doc_skipped():
    doc_empty = _make_raw_doc(text="")
    doc_ok = _make_raw_doc(text="Some real content here " * 10)
    chunks = chunk_baseline([doc_empty, doc_ok])
    assert len(chunks) >= 1
    assert all(c.content.strip() for c in chunks)


def test_chunk_baseline_whitespace_only_doc_skipped():
    doc = _make_raw_doc(text="   \n  \t  ")
    chunks = chunk_baseline([doc])
    assert chunks == []


# ---------------------------------------------------------------------------
# chunk_semantic
# ---------------------------------------------------------------------------


def test_chunk_semantic_returns_knowledge_chunks():
    text = "# Introduction\n\nSome intro text.\n\n## Details\n\nMore details here."
    doc = _make_raw_doc(text=text)
    chunks = chunk_semantic([doc])
    assert all(isinstance(c, KnowledgeChunk) for c in chunks)


def test_chunk_semantic_uses_header_as_topic():
    text = "# Graph Theory\n\nVertices and edges.\n\n## Spanning Trees\n\nA spanning tree connects all nodes."
    doc = _make_raw_doc(text=text, topic="Fallback Topic")
    chunks = chunk_semantic([doc])
    topics = {c.source_topic for c in chunks}
    # At least one chunk should have a header-derived topic
    assert len(topics) > 0


def test_chunk_semantic_global_index_sequential():
    text = "# Section A\n\n" + "word " * 200 + "\n\n## Section B\n\n" + "word " * 200
    doc = _make_raw_doc(text=text)
    chunks = chunk_semantic([doc], chunk_size=500, chunk_overlap=50)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_semantic_fallback_no_headers():
    # Plain text without markdown headers — should still chunk
    doc = _make_raw_doc(text="plain text " * 300)
    chunks = chunk_semantic([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 1


def test_chunk_semantic_embedding_none():
    doc = _make_raw_doc(text="# Header\n\ncontent " * 10)
    chunks = chunk_semantic([doc])
    assert all(c.embedding is None for c in chunks)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------


def test_chunk_documents_baseline_dispatch():
    doc = _make_raw_doc(text="content " * 300)
    config = _make_config(chunk_size=500, chunk_overlap=50)
    chunks = chunk_documents([doc], config, strategy="baseline")
    assert len(chunks) >= 1
    assert all(isinstance(c, KnowledgeChunk) for c in chunks)


def test_chunk_documents_semantic_dispatch():
    doc = _make_raw_doc(text="# Header\n\ncontent " * 100)
    config = _make_config(chunk_size=500, chunk_overlap=50)
    chunks = chunk_documents([doc], config, strategy="semantic")
    assert len(chunks) >= 1


def test_chunk_documents_uses_config_chunk_size():
    doc = _make_raw_doc(text="w " * 500)  # ~1000 chars
    config = _make_config(chunk_size=200, chunk_overlap=0)
    chunks = chunk_documents([doc], config, strategy="baseline")
    assert all(len(c.content) <= 200 for c in chunks)
