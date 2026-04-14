"""Tests for src/rag/citation_extractor.py.

Pure function — no API calls, no mocking needed.
"""

from __future__ import annotations

import pytest

from src.rag.citation_extractor import extract_citations
from src.schemas import Citation, KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_result(idx: int, score: float = 0.8, topic: str = "Test Topic") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=f"This is the content of chunk number {idx}. " * 5,
        source_topic=topic,
        source_field="computer_science",
        chunk_index=idx,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_single_citation():
    results = [_make_result(0), _make_result(1), _make_result(2)]
    citations = extract_citations("See [1] for details.", results)
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_0"


def test_extract_multiple_citations():
    results = [_make_result(i) for i in range(5)]
    citations = extract_citations("See [1] and [3] and [5].", results)
    assert len(citations) == 3
    chunk_ids = {c.chunk_id for c in citations}
    assert chunk_ids == {"chunk_0", "chunk_2", "chunk_4"}


def test_out_of_range_citation_skipped():
    results = [_make_result(0), _make_result(1)]
    citations = extract_citations("See [99] for details.", results)
    assert citations == []


def test_zero_citation_skipped():
    """[0] is invalid (1-based indexing); should be skipped."""
    results = [_make_result(0)]
    citations = extract_citations("See [0] for more.", results)
    assert citations == []


def test_duplicate_citations_deduplicated():
    results = [_make_result(0), _make_result(1)]
    citations = extract_citations("[1] and [1] and [1].", results)
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_0"


def test_no_citations_empty_list():
    results = [_make_result(0)]
    citations = extract_citations("No refs here at all.", results)
    assert citations == []


def test_empty_text_returns_empty():
    results = [_make_result(0)]
    citations = extract_citations("", results)
    assert citations == []


def test_empty_results_returns_empty():
    citations = extract_citations("[1] ref here.", [])
    assert citations == []


def test_citation_fields_correct():
    results = [_make_result(0, score=0.75, topic="Graph Theory")]
    citations = extract_citations("Per [1], graphs are useful.", results)
    c = citations[0]
    assert c.chunk_id == "chunk_0"
    assert c.source_topic == "Graph Theory"
    assert isinstance(c.text_snippet, str)
    assert len(c.text_snippet) <= 100
    assert isinstance(c, Citation)


def test_text_snippet_truncated_to_100():
    long_content = "x" * 200
    chunk = KnowledgeChunk(
        content=long_content,
        source_topic="Topic",
        source_field="computer_science",
        chunk_index=0,
    )
    result = RetrievalResult(chunk=chunk, score=0.9, rank=0)
    citations = extract_citations("[1]", [result])
    assert len(citations[0].text_snippet) == 100


def test_relevance_score_clamped_high():
    """Score > 1.0 should be clamped to 1.0."""
    chunk = KnowledgeChunk(
        content="content", source_topic="T", source_field="f", chunk_index=0
    )
    result = RetrievalResult(chunk=chunk, score=1.5, rank=0)
    citations = extract_citations("[1]", [result])
    assert citations[0].relevance_score == 1.0


def test_relevance_score_clamped_low():
    """Score < 0.0 should be clamped to 0.0."""
    chunk = KnowledgeChunk(
        content="content", source_topic="T", source_field="f", chunk_index=0
    )
    result = RetrievalResult(chunk=chunk, score=-0.5, rank=0)
    citations = extract_citations("[1]", [result])
    assert citations[0].relevance_score == 0.0


def test_mixed_valid_and_invalid_refs():
    results = [_make_result(0), _make_result(1)]
    # [2] valid, [99] invalid, [0] invalid
    citations = extract_citations("[2] is good. [99] is bad. [0] is bad.", results)
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_1"
