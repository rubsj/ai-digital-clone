"""Contract tests for Gatekeeper (ADR-018).

Tests verify the deterministic deliver-or-fallback logic. No LLM calls; no mocks.
_compute_flags and Gatekeeper.run() are pure functions; every assertion is a
straightforward arithmetic check. A green suite here proves routing correctness
(the real-LLM plumbing tests from the old LLM-based router file are obsolete —
the Gatekeeper has no LLM path to plumb).
"""

from __future__ import annotations

import pytest

from src.components.gatekeeper import Gatekeeper, _compute_flags
from src.schemas import EvaluationResult, KnowledgeChunk, RetrievalResult


def _make_eval(
    groundedness_score: float = 0.75,
    style_score: float = 0.90,
    confidence_score: float = 0.85,
    flags: list[str] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        groundedness_score=groundedness_score,
        style_score=style_score,
        confidence_score=confidence_score,
        explanation="test",
        flags=flags or [],
    )


def _make_chunk(content: str = "buddy allocator manages physical pages") -> RetrievalResult:
    chunk = KnowledgeChunk(
        content=content,
        source_topic="kernel_memory",
        source_field="cs",
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


# ---------------------------------------------------------------------------
# _compute_flags
# ---------------------------------------------------------------------------


def test_compute_flags_all_clear():
    flags = _compute_flags(_make_eval(groundedness_score=0.75, style_score=0.90, confidence_score=0.85))
    assert flags == []


def test_compute_flags_low_groundedness():
    flags = _compute_flags(_make_eval(groundedness_score=0.55))
    assert "low_groundedness" in flags


def test_compute_flags_at_floor_is_clear():
    # 0.60 is the floor; exactly at floor = NOT low_groundedness.
    flags = _compute_flags(_make_eval(groundedness_score=0.60))
    assert "low_groundedness" not in flags


def test_compute_flags_low_style():
    flags = _compute_flags(_make_eval(style_score=0.65))
    assert "low_style" in flags


def test_compute_flags_low_confidence():
    flags = _compute_flags(_make_eval(confidence_score=0.75))
    assert "low_confidence" in flags


def test_compute_flags_multiple():
    flags = _compute_flags(_make_eval(groundedness_score=0.50, style_score=0.60, confidence_score=0.70))
    assert set(flags) == {"low_groundedness", "low_style", "low_confidence"}


# ---------------------------------------------------------------------------
# Gatekeeper.run() — deliver path
# ---------------------------------------------------------------------------


def test_run_delivers_when_groundedness_at_floor():
    rd = Gatekeeper().run("q", "resp", [_make_chunk()], _make_eval(groundedness_score=0.60), "torvalds")
    assert rd.decision == "deliver"
    assert rd.trigger_category is None
    assert rd.trigger_reason is None


def test_run_delivers_when_groundedness_above_floor():
    rd = Gatekeeper().run("q", "resp", [_make_chunk()], _make_eval(groundedness_score=0.80), "torvalds")
    assert rd.decision == "deliver"


def test_run_deliver_quality_flags_no_blocking():
    # low_style fires (ss=0.65) but should NOT block delivery; appears in quality_flags only.
    rd = Gatekeeper().run(
        "q", "resp", [_make_chunk()],
        _make_eval(groundedness_score=0.70, style_score=0.65, confidence_score=0.85),
        "torvalds",
    )
    assert rd.decision == "deliver"
    assert "low_style" in rd.quality_flags
    assert "low_groundedness" not in rd.quality_flags


def test_run_deliver_quality_flags_low_confidence():
    # low_confidence fires (cs=0.75) but should NOT block delivery.
    rd = Gatekeeper().run(
        "q", "resp", [_make_chunk()],
        _make_eval(groundedness_score=0.70, style_score=0.85, confidence_score=0.75),
        "torvalds",
    )
    assert rd.decision == "deliver"
    assert "low_confidence" in rd.quality_flags
    assert "low_groundedness" not in rd.quality_flags


def test_run_deliver_quality_flags_empty_when_all_clear():
    rd = Gatekeeper().run("q", "resp", [_make_chunk()], _make_eval(), "torvalds")
    assert rd.quality_flags == []


# ---------------------------------------------------------------------------
# Gatekeeper.run() — fallback path (low_groundedness)
# ---------------------------------------------------------------------------


def test_run_falls_back_when_groundedness_below_floor():
    rd = Gatekeeper().run(
        "q", "resp", [_make_chunk()],
        _make_eval(groundedness_score=0.55),
        "torvalds",
    )
    assert rd.decision == "fallback"
    assert rd.trigger_category == "low_groundedness"
    assert rd.trigger_reason is not None
    assert "0.55" in rd.trigger_reason


def test_run_fallback_low_groundedness_quality_flags_no_blocking():
    # quality_flags must NOT contain low_groundedness (blocking flag travels in trigger_category).
    rd = Gatekeeper().run(
        "q", "resp", [_make_chunk()],
        _make_eval(groundedness_score=0.55),
        "torvalds",
    )
    assert "low_groundedness" not in rd.quality_flags


def test_run_fallback_low_groundedness_with_low_confidence():
    # If gs < floor AND cs < CONFIDENCE_MIN, quality_flags carries low_confidence only.
    rd = Gatekeeper().run(
        "q", "resp", [_make_chunk()],
        _make_eval(groundedness_score=0.55, confidence_score=0.70),
        "torvalds",
    )
    assert rd.decision == "fallback"
    assert rd.trigger_category == "low_groundedness"
    assert "low_confidence" in rd.quality_flags
    assert "low_groundedness" not in rd.quality_flags


# ---------------------------------------------------------------------------
# Gatekeeper.run() — empty_retrieval (ordered BEFORE low_groundedness)
# ---------------------------------------------------------------------------


def test_run_falls_back_empty_retrieval_on_zero_chunks():
    # gs=0.0 on zero chunks; router must emit empty_retrieval, not low_groundedness.
    rd = Gatekeeper().run(
        "q", "resp", [],
        _make_eval(groundedness_score=0.0),
        "torvalds",
    )
    assert rd.decision == "fallback"
    assert rd.trigger_category == "empty_retrieval"
    assert rd.trigger_reason == "empty_retrieval: 0 chunks retrieved"


def test_run_empty_retrieval_quality_flags_empty():
    # low_groundedness is a blocking flag (excluded from quality_flags);
    # even when gs=0.0, quality_flags must be empty on the empty_retrieval path.
    rd = Gatekeeper().run(
        "q", "resp", [],
        _make_eval(groundedness_score=0.0),
        "torvalds",
    )
    assert rd.quality_flags == []


# ---------------------------------------------------------------------------
# last_run_timings
# ---------------------------------------------------------------------------


def test_last_run_timings_populated():
    g = Gatekeeper()
    g.run("q", "resp", [_make_chunk()], _make_eval(), "torvalds")
    assert hasattr(g, "last_run_timings")
    assert g.last_run_timings["generate_ms"] == 0.0
    assert g.last_run_timings["parse_ms"] == 0.0
