"""Tests for src/fallback/unstyled_responder.py.

Instructor + LiteLLM are mocked — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.fallback.unstyled_responder import _build_user_prompt, generate_unstyled_response
from src.schemas import KnowledgeChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_result(content: str = "The kernel manages memory via slab allocators.", topic: str = "Linux Kernel") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic=topic,
        source_field="cs",
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=0.9, rank=0)


def _mock_instructor_client(answer: str = "The kernel uses slab allocators.") -> MagicMock:
    mock_result = MagicMock()
    mock_result.answer = answer
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_result
    return mock_client


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_contains_query():
    prompt = _build_user_prompt("How does memory work?", [_make_result()])
    assert "How does memory work?" in prompt


def test_build_user_prompt_contains_chunk_content():
    prompt = _build_user_prompt("query", [_make_result("slab allocator details")])
    assert "slab allocator details" in prompt


def test_build_user_prompt_contains_topic():
    prompt = _build_user_prompt("query", [_make_result(topic="Memory Management")])
    assert "Memory Management" in prompt


def test_build_user_prompt_numbers_excerpts():
    results = [_make_result() for _ in range(3)]
    prompt = _build_user_prompt("query", results)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" in prompt


def test_build_user_prompt_empty_chunks():
    prompt = _build_user_prompt("How does memory work?", [])
    assert "How does memory work?" in prompt


# ---------------------------------------------------------------------------
# generate_unstyled_response
# ---------------------------------------------------------------------------


def test_generate_unstyled_response_returns_string():
    with patch("src.fallback.unstyled_responder.instructor") as mock_instructor:
        mock_instructor.from_litellm.return_value = _mock_instructor_client("Plain factual answer.")
        result = generate_unstyled_response("How does memory work?", [_make_result()])
    assert isinstance(result, str)


def test_generate_unstyled_response_non_empty():
    with patch("src.fallback.unstyled_responder.instructor") as mock_instructor:
        mock_instructor.from_litellm.return_value = _mock_instructor_client("Answer here.")
        result = generate_unstyled_response("query", [_make_result()])
    assert len(result) > 0


def test_generate_unstyled_response_uses_system_prompt_with_no_style():
    """Verify the system prompt forbids rhetorical style — check it's passed to create()."""
    captured_kwargs: dict = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        mock_result = MagicMock()
        mock_result.answer = "Factual answer."
        return mock_result

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("src.fallback.unstyled_responder.instructor") as mock_instructor:
        mock_instructor.from_litellm.return_value = mock_client
        generate_unstyled_response("query", [_make_result()])

    messages = captured_kwargs.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert system_messages, "No system message found"
    system_content = system_messages[0]["content"]
    assert "no" in system_content.lower() or "not" in system_content.lower()
    # Key requirement: system prompt must mention style restriction
    assert any(
        phrase in system_content.lower()
        for phrase in ["no rhetorical style", "no personality", "plain", "factual"]
    )


def test_generate_unstyled_response_returns_answer_field():
    expected = "This is the plain factual answer."
    with patch("src.fallback.unstyled_responder.instructor") as mock_instructor:
        mock_instructor.from_litellm.return_value = _mock_instructor_client(expected)
        result = generate_unstyled_response("How does memory work?", [_make_result()])
    assert result == expected
