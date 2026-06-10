"""Tests for src/rag/corpus_loader.py.

All HuggingFace calls are mocked — never calls the real API in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.corpus_loader import (
    RawDocument,
    _EVALUATED_TOPICS,
    _extract_topic,
    load_corpus,
)


# ---------------------------------------------------------------------------
# _extract_topic
# ---------------------------------------------------------------------------


def test_extract_topic_h1():
    outline = "# Introduction to Algorithms\n\n## Overview"
    assert _extract_topic(outline) == "Introduction to Algorithms"


def test_extract_topic_h2_only():
    outline = "## Data Structures"
    assert _extract_topic(outline) == "Data Structures"


def test_extract_topic_no_heading_returns_first_line():
    outline = "Networks and Protocols\nSome more text"
    assert _extract_topic(outline) == "Networks and Protocols"


def test_extract_topic_empty_string():
    assert _extract_topic("") == ""


def test_extract_topic_whitespace_only():
    assert _extract_topic("   \n  ") == ""


# ---------------------------------------------------------------------------
# Helpers for mocking the dataset
# ---------------------------------------------------------------------------


def _make_row(
    topic: str = "Algorithms",
    field: str = "computer_science",
    subfield: str = "algorithms_and_data_structures",
    markdown: str = "# Algorithms\n\nSome content about sorting.",
    outline: str = "# Algorithms",
) -> dict:
    return {
        "topic": topic,
        "field": field,
        "subfield": subfield,
        "markdown": markdown,
        "outline": outline,
        "model": "gpt-4",
        "concepts": "",
        "rag": "",
    }


def _mock_dataset(rows: list[dict]) -> MagicMock:
    """Build a minimal mock that behaves like a HuggingFace dataset."""
    mock_ds = MagicMock()
    mock_ds.num_rows = len(rows)
    mock_ds.__iter__ = MagicMock(return_value=iter(rows))

    filtered = MagicMock()
    filtered.num_rows = len(rows)
    filtered.__iter__ = MagicMock(return_value=iter(rows))
    filtered.select = MagicMock(side_effect=lambda r: _mock_slice(rows, r))

    mock_ds.filter = MagicMock(return_value=filtered)
    return mock_ds


def _mock_slice(rows: list[dict], indices: range) -> MagicMock:
    sliced = list(rows)[indices.start : indices.stop]
    m = MagicMock()
    m.num_rows = len(sliced)
    m.__iter__ = MagicMock(return_value=iter(sliced))
    m.select = MagicMock(side_effect=lambda r: _mock_slice(sliced, r))
    return m


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_returns_raw_documents(mock_ld):
    rows = [_make_row(), _make_row(topic="Networks", subfield="computer_networks")]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_uses_topic_column(mock_ld):
    rows = [_make_row(topic="Graph Theory")]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert docs[0].topic == "Graph Theory"


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_falls_back_to_outline(mock_ld):
    rows = [_make_row(topic="", outline="# Operating Systems\n## Scheduling")]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert docs[0].topic == "Operating Systems"


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_falls_back_to_subfield(mock_ld):
    rows = [_make_row(topic="", outline="", subfield="computer_networks")]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert docs[0].topic == "computer networks"


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_max_docs_caps_output(mock_ld):
    rows = [_make_row() for _ in range(10)]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None, max_docs=3)

    assert len(docs) == 3


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_skips_empty_markdown(mock_ld):
    rows = [_make_row(markdown=""), _make_row(markdown="  "), _make_row()]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert len(docs) == 1


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_empty_dataset(mock_ld):
    mock_ds = MagicMock()
    mock_ds.num_rows = 0
    mock_ds.__iter__ = MagicMock(return_value=iter([]))
    filtered = MagicMock()
    filtered.num_rows = 0
    filtered.__iter__ = MagicMock(return_value=iter([]))
    filtered.select = MagicMock(return_value=filtered)
    mock_ds.filter = MagicMock(return_value=filtered)
    mock_ld.return_value = mock_ds

    docs = load_corpus(topic_filter=None)

    assert docs == []


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_field_and_subfield_propagated(mock_ld):
    rows = [_make_row(field="computer_science", subfield="artificial_intelligence")]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus(topic_filter=None)

    assert docs[0].field == "computer_science"
    assert docs[0].subfield == "artificial_intelligence"


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_default_topic_filter_accepts_evaluated(mock_ld):
    # One row matching an evaluated topic, one that doesn't — only the match returns.
    evaluated_topic = next(iter(_EVALUATED_TOPICS))
    rows = [
        _make_row(topic=evaluated_topic, markdown="Some content."),
        _make_row(topic="Unrelated Topic", markdown="Other content."),
    ]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus()  # default topic_filter = _EVALUATED_TOPICS

    assert len(docs) == 1
    assert docs[0].topic == evaluated_topic


@patch("src.rag.corpus_loader.load_dataset")
def test_load_corpus_duplicate_topic_skips_second(mock_ld):
    # Dataset has two rows for the same evaluated topic — first wins, second is warned and skipped.
    evaluated_topic = next(iter(_EVALUATED_TOPICS))
    rows = [
        _make_row(topic=evaluated_topic, markdown="Version A."),
        _make_row(topic=evaluated_topic, markdown="Version B."),
    ]
    mock_ld.return_value = _mock_dataset(rows)

    docs = load_corpus()  # default topic_filter active — no exception

    assert len(docs) == 1
    assert docs[0].topic == evaluated_topic
