"""Tests for src/agents/clone_agent.py.

CrewAI LLM construction requires OPENAI_API_KEY (set to a dummy via autouse
fixture). Crew.kickoff and the Instructor client are mocked; helper functions
are pure and tested directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.clone_agent import (
    CloneAgent,
    _CloneDraft,
    _build_backstory,
    _build_goal,
    _build_role,
    _format_chunks,
    _format_style_examples,
)
from src.schemas import Citation, CloneResponse, KnowledgeChunk, RetrievalResult, StyleFeatures, StyleProfile


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")


def _make_features(**kwargs) -> StyleFeatures:
    defaults = dict(
        avg_message_length=0.5,
        greeting_patterns={"hi": 0.3},
        punctuation_patterns={"dash": 0.2},
        capitalization_ratio=0.1,
        question_frequency=0.15,
        vocabulary_richness=0.6,
        common_phrases=["good point", "in the"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.6},
        formality_level=0.4,
        technical_terminology=0.5,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.5,
    )
    return StyleFeatures(**(defaults | kwargs))


def _make_profile(name: str = "Test Leader", sample_emails=None, **feature_kwargs) -> StyleProfile:
    f = _make_features(**feature_kwargs)
    return StyleProfile(
        leader_name=name,
        features=f,
        style_vector=f.to_vector(),
        email_count=50,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
        sample_emails=sample_emails or [],
    )


def _make_result(content: str = "kernel memory management details", index: int = 0,
                 topic: str = "Linux Kernel", score: float = 0.85) -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic=topic,
        source_field="cs",
        chunk_index=index,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=score, rank=index)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_role_contains_leader_name():
    assert "Linus Torvalds" in _build_role("Linus Torvalds")


def test_role_mentions_lkml():
    role = _build_role("Test Leader")
    assert "LKML" in role or "Linux Kernel Mailing List" in role


def test_goal_contains_feature_values():
    goal = _build_goal("Test Leader", _make_profile(formality_level=0.77, technical_depth=0.33))
    assert "0.770" in goal
    assert "0.330" in goal


def test_goal_contains_common_phrases():
    goal = _build_goal("Test Leader", _make_profile(common_phrases=["unique-phrase-xyz"]))
    assert "unique-phrase-xyz" in goal


def test_backstory_tone_blunt_when_low_formality():
    assert "direct and blunt" in _build_backstory("L", _make_profile(formality_level=0.3))


def test_backstory_tone_structured_when_high_formality():
    assert "clear and structured" in _build_backstory("L", _make_profile(formality_level=0.7))


# ---------------------------------------------------------------------------
# _format_chunks / _format_style_examples
# ---------------------------------------------------------------------------


def test_format_chunks_zero_based_numbering():
    text = _format_chunks([_make_result("alpha"), _make_result("beta")])
    assert "[0]" in text and "alpha" in text
    assert "[1]" in text and "beta" in text


def test_format_chunks_caps_at_max():
    chunks = [_make_result(f"chunk {i}", index=i) for i in range(10)]
    text = _format_chunks(chunks)
    assert "chunk 4" in text
    assert "chunk 5" not in text


def test_format_style_examples_empty_when_no_samples():
    assert _format_style_examples(_make_profile()) == ""


def test_format_style_examples_includes_and_caps_samples():
    profile = _make_profile(sample_emails=["email-a", "email-b", "email-c", "email-d"])
    text = _format_style_examples(profile)
    assert "email-a" in text and "email-c" in text
    assert "email-d" not in text  # capped at 3


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------


def test_crew_has_one_agent_one_task():
    crew = CloneAgent()._build_crew("How does memory work?", "Test Leader", _make_profile(), [_make_result()])
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1


def test_crew_agent_has_role_goal_backstory():
    crew = CloneAgent()._build_crew("q", "Linus Torvalds", _make_profile("Linus Torvalds"), [_make_result()])
    agent = crew.agents[0]
    assert "Linus Torvalds" in agent.role
    assert "Linus Torvalds" in agent.goal
    assert agent.backstory


def test_task_description_contains_query_and_chunk():
    crew = CloneAgent()._build_crew("What is the buddy allocator?", "L", _make_profile(),
                                    [_make_result("slab allocator details here")])
    desc = crew.tasks[0].description
    assert "What is the buddy allocator?" in desc
    assert "slab allocator details here" in desc


def test_task_description_includes_style_examples():
    profile = _make_profile(sample_emails=["a sample email body"])
    crew = CloneAgent()._build_crew("q", "L", profile, [_make_result()])
    assert "a sample email body" in crew.tasks[0].description


# ---------------------------------------------------------------------------
# _reconcile
# ---------------------------------------------------------------------------


def test_reconcile_maps_index_to_full_citation():
    chunks = [_make_result("content here", index=7, topic="Memory", score=0.9)]
    citations = CloneAgent()._reconcile([0], chunks)
    assert len(citations) == 1
    c = citations[0]
    assert c.chunk_id == "chunk_7"
    assert c.source_topic == "Memory"
    assert c.text_snippet == "content here"
    assert c.relevance_score == pytest.approx(0.9)


def test_reconcile_drops_out_of_range():
    chunks = [_make_result(index=0)]
    assert CloneAgent()._reconcile([5, -1], chunks) == []


def test_reconcile_dedups():
    chunks = [_make_result(index=0), _make_result(index=1)]
    citations = CloneAgent()._reconcile([0, 0, 1], chunks)
    assert [c.chunk_id for c in citations] == ["chunk_0", "chunk_1"]


def test_reconcile_clamps_relevance_score():
    chunks = [_make_result(score=1.5)]
    assert CloneAgent()._reconcile([0], chunks)[0].relevance_score == 1.0


def test_reconcile_snippet_truncated_to_100():
    chunks = [_make_result("x" * 250)]
    assert len(CloneAgent()._reconcile([0], chunks)[0].text_snippet) == 100


# ---------------------------------------------------------------------------
# run — full path with mocked Crew + Instructor
# ---------------------------------------------------------------------------


def _patch_run(raw: str, draft: _CloneDraft):
    """Context managers patching Crew.kickoff and the Instructor client."""
    kickoff_output = MagicMock()
    kickoff_output.raw = raw
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = draft
    return (
        patch("src.agents.clone_agent.Crew.kickoff", return_value=kickoff_output),
        patch("src.agents.clone_agent.instructor.from_litellm", return_value=mock_client),
        mock_client,
    )


def test_run_returns_clone_response_with_citations():
    chunks = [_make_result("memory facts", index=0), _make_result("scheduler facts", index=1)]
    draft = _CloneDraft(response_text="The kernel manages memory.", cited_chunk_indices=[0])
    kickoff_p, instr_p, _ = _patch_run("The kernel manages memory.", draft)
    with kickoff_p, instr_p:
        result = CloneAgent().run("How does memory work?", "Test Leader", _make_profile(), chunks)

    assert isinstance(result, CloneResponse)
    assert result.response_text == "The kernel manages memory."
    assert len(result.citations) == 1
    assert isinstance(result.citations[0], Citation)
    assert result.citations[0].chunk_id == "chunk_0"


def test_run_invokes_instructor_parse():
    chunks = [_make_result(index=0)]
    draft = _CloneDraft(response_text="resp", cited_chunk_indices=[])
    kickoff_p, instr_p, mock_client = _patch_run("resp", draft)
    with kickoff_p, instr_p:
        CloneAgent().run("q", "L", _make_profile(), chunks)

    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is _CloneDraft
    assert kwargs["temperature"] == 0


def test_run_drops_hallucinated_citation_index():
    chunks = [_make_result(index=0)]
    draft = _CloneDraft(response_text="resp", cited_chunk_indices=[0, 9])
    kickoff_p, instr_p, _ = _patch_run("resp", draft)
    with kickoff_p, instr_p:
        result = CloneAgent().run("q", "L", _make_profile(), chunks)

    assert [c.chunk_id for c in result.citations] == ["chunk_0"]
