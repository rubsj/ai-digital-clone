"""Tests for src/fallback/context_summarizer.py.

Deterministic string composition — no API calls.
"""

from __future__ import annotations

import pytest

from src.fallback.context_summarizer import summarize_context
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_result(topic: str = "Memory Management", rank: int = 0) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content="some content",
        source_topic=topic,
        source_field="cs",
        chunk_index=rank,
    )
    return RetrievalResult(chunk=chunk, score=0.8, rank=rank)


# ---------------------------------------------------------------------------
# summarize_context
# ---------------------------------------------------------------------------


def test_summarize_empty_chunks():
    result = summarize_context("How does memory work?", [])
    assert "How does memory work?" in result
    assert "more depth" in result


def test_summarize_single_topic():
    result = summarize_context("How does memory work?", [_make_result("Memory Management")])
    assert "Memory Management" in result
    assert "more depth" in result


def test_summarize_two_topics():
    results = [_make_result("Memory Management"), _make_result("Scheduler", rank=1)]
    result = summarize_context("How do things work?", results)
    assert "Memory Management" in result
    assert "Scheduler" in result
    assert "and" in result


def test_summarize_three_topics():
    results = [
        _make_result("Memory Management"),
        _make_result("Scheduler"),
        _make_result("File System", rank=2),
    ]
    result = summarize_context("query", results)
    assert "Memory Management" in result
    assert "Scheduler" in result
    assert "File System" in result


def test_summarize_deduplicates_topics():
    # Same topic repeated three times → appears only once in output
    results = [_make_result("Memory Management") for _ in range(3)]
    result = summarize_context("How does memory work?", results)
    assert result.count("Memory Management") == 1


def test_summarize_returns_string():
    result = summarize_context("query", [_make_result()])
    assert isinstance(result, str)


def test_summarize_non_empty():
    result = summarize_context("query", [_make_result()])
    assert len(result) > 0


def test_summarize_long_query_truncated():
    long_query = "How does the Linux kernel manage memory using slab allocators efficiently"
    result = summarize_context(long_query, [_make_result()])
    # Query is truncated to first 6 words + ellipsis
    assert "…" in result


def test_summarize_short_query_no_truncation():
    short_query = "memory management"
    result = summarize_context(short_query, [_make_result()])
    assert "…" not in result


def test_summarize_preserves_topic_order():
    results = [
        _make_result("Alpha"),
        _make_result("Beta", rank=1),
        _make_result("Gamma", rank=2),
    ]
    result = summarize_context("query", results)
    alpha_pos = result.index("Alpha")
    beta_pos = result.index("Beta")
    gamma_pos = result.index("Gamma")
    assert alpha_pos < beta_pos < gamma_pos
