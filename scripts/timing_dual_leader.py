"""Timing harness: shared-retrieval vs independent-pipeline latency.

Measures wall-clock time for:
  A) compare_leaders() — one RAG call, two style+evaluate passes
  B) Two independent DigitalCloneFlow runs — two RAG calls

All LLM calls are mocked with a fixed 50ms sleep to simulate real latency.
RAGAgent.retrieve is also mocked with a fixed 100ms sleep to simulate
embed + FAISS + Cohere rerank.

Run from the repo root:
    source .venv/bin/activate
    python scripts/timing_dual_leader.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from src.flow import DigitalCloneFlow, compare_leaders
from src.schemas import (
    EvaluationResult,
    FallbackResponse,
    KnowledgeChunk,
    RetrievalResult,
    StyleFeatures,
    StyleProfile,
)

_RAG_SLEEP_MS = 100
_LLM_SLEEP_MS = 50
_QUERY = "How does memory management work in the Linux kernel?"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features() -> StyleFeatures:
    return StyleFeatures(
        avg_message_length=0.34,
        greeting_patterns={},
        punctuation_patterns={"dash": 0.2},
        capitalization_ratio=0.05,
        question_frequency=0.1,
        vocabulary_richness=0.6,
        common_phrases=["good point"],
        reasoning_patterns={"because": 0.2},
        sentiment_distribution={"positive": 0.3},
        formality_level=0.5,
        technical_terminology=0.4,
        code_snippet_freq=0.1,
        quote_reply_ratio=0.2,
        patch_language={"nak": 0.5},
        technical_depth=0.12,
    )


def _make_profile(name: str) -> StyleProfile:
    f = _make_features()
    return StyleProfile(
        leader_name=name,
        features=f,
        style_vector=f.to_vector(),
        email_count=100,
        last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        alpha=0.3,
    )


def _make_chunk() -> RetrievalResult:
    chunk = KnowledgeChunk(
        content="kernel memory details",
        source_topic="Linux Kernel",
        source_field="cs",
        chunk_index=0,
        embedding=np.ones(1536, dtype=np.float32) / np.sqrt(1536),
    )
    return RetrievalResult(chunk=chunk, score=0.85, rank=0)


def _make_eval() -> EvaluationResult:
    return EvaluationResult(
        style_score=0.8,
        groundedness_score=0.85,
        confidence_score=0.75,
        final_score=0.81,
        explanation="Good.",
        decision="deliver",
    )


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.leaders = {
        "torvalds": MagicMock(profile_path="data/models/torvalds_profile.json"),
        "kroah_hartman": MagicMock(profile_path="data/models/kroah_hartman_profile.json"),
    }
    return cfg


def _slow_retrieve(*args, **kwargs) -> list[RetrievalResult]:
    time.sleep(_RAG_SLEEP_MS / 1000)
    return [_make_chunk()]


def _slow_generate(*args, **kwargs) -> str:
    time.sleep(_LLM_SLEEP_MS / 1000)
    return "The kernel uses slab allocators."


def _slow_evaluate(*args, **kwargs) -> EvaluationResult:
    time.sleep(_LLM_SLEEP_MS / 1000)
    return _make_eval()


# ---------------------------------------------------------------------------
# Timing runs
# ---------------------------------------------------------------------------

_PATCHES = [
    ("src.flow.load_config", dict(return_value=_mock_config())),
    ("src.flow.RAGAgent.__init__", dict(return_value=None)),
    ("src.flow.RAGAgent.retrieve", dict(side_effect=_slow_retrieve)),
    ("src.flow.load_profile", dict(side_effect=lambda p: _make_profile("Test Leader"))),
    ("src.flow.generate_styled_response", dict(side_effect=_slow_generate)),
    ("src.flow.EvaluatorAgent.evaluate", dict(side_effect=_slow_evaluate)),
    ("src.flow.build_fallback_response", dict(return_value=FallbackResponse(
        trigger_reason="test", context_summary="", calendar_link="", available_slots=[], unstyled_response=""
    ))),
]


def _apply_patches():
    """Return a list of active patch objects (must be started/stopped by caller)."""
    return [patch(target, **kwargs) for target, kwargs in _PATCHES]


def time_shared_retrieval(runs: int = 5) -> float:
    """Wall-clock time for compare_leaders() averaged over N runs (ms)."""
    patches = _apply_patches()
    for p in patches:
        p.start()
    try:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            compare_leaders(_QUERY)
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)
    finally:
        for p in patches:
            p.stop()


def time_independent_pipelines(runs: int = 5) -> float:
    """Wall-clock time for two independent DigitalCloneFlow runs averaged over N runs (ms)."""
    patches = _apply_patches()
    for p in patches:
        p.start()
    try:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            flow_t = DigitalCloneFlow()
            flow_t.kickoff(inputs={"query": _QUERY, "leader": "Linus Torvalds"})
            flow_kh = DigitalCloneFlow()
            flow_kh.kickoff(inputs={"query": _QUERY, "leader": "Greg Kroah-Hartman"})
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)
    finally:
        for p in patches:
            p.stop()


if __name__ == "__main__":
    runs = 5
    print(f"Timing harness — {runs} runs each, mocked RAG ({_RAG_SLEEP_MS}ms) + LLM ({_LLM_SLEEP_MS}ms)")
    print(f"Query: {_QUERY!r}\n")

    shared_ms = time_shared_retrieval(runs)
    independent_ms = time_independent_pipelines(runs)
    saved_ms = independent_ms - shared_ms
    speedup_pct = (saved_ms / independent_ms) * 100

    print(f"shared-retrieval (compare_leaders):      {shared_ms:.1f} ms")
    print(f"independent pipelines (two full flows):  {independent_ms:.1f} ms")
    print(f"savings:                                 {saved_ms:.1f} ms  ({speedup_pct:.1f}%)")
    print(f"\nExpected savings ≈ {_RAG_SLEEP_MS} ms (one avoided RAG call)")
