"""Pure-function router assertion for STOP GATE 1.6a (no LLM calls).

Checks four things:
  (1) Over the 84 stored re-eval records: the new deterministic GatekeeperAgent
      delivers iff groundedness_score >= 0.60, for every record with non-empty chunks.
      quality_flags on all 84 records contains no blocking flags.
  (2) Constructed zero-chunk unit case: confirms the router emits
      trigger_category='empty_retrieval' (not 'low_groundedness') and that
      quality_flags is empty (low_groundedness is a blocking flag, excluded from
      quality_flags even when it fires due to gs=0.0 on empty chunks).
  (3) Constructed deliver-low-style unit case: confirms a delivered record with
      low style score carries quality_flags=['low_style'] (non-blocking flag rides
      through on the deliver path).
  (4) Evaluation-is-None unit case: exercises the emergency guard in flow.py route(),
      confirms trigger_category='evaluation_error' is emitted, not null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.gatekeeper_agent import GatekeeperAgent
from src.schemas import EvaluationResult, KnowledgeChunk, RetrievalResult

REEVAL_PATH = Path("results/evaluation_day12_reeval.json")
GROUNDEDNESS_MIN = 0.60
_BLOCKING_FLAGS = frozenset({"low_groundedness"})


def make_evaluation(lr: dict) -> EvaluationResult:
    return EvaluationResult(
        groundedness_score=lr["groundedness_score"],
        style_score=lr["style_score"],
        confidence_score=lr["confidence_score"],
        explanation="stored record",
        flags=lr.get("flags", []),
    )


def make_chunks(chunk_contents: list) -> list[RetrievalResult]:
    result = []
    for c in chunk_contents:
        chunk = KnowledgeChunk(
            content=c["content"],
            source_topic=c["source_topic"],
            source_field="stored",
            chunk_index=c["rank"],
        )
        result.append(RetrievalResult(chunk=chunk, score=c["score"], rank=c["rank"]))
    return result


def main() -> None:
    # ── (1) 84-record stored assertion ──────────────────────────────────────────
    with open(REEVAL_PATH) as f:
        records = json.load(f)

    router = GatekeeperAgent()
    violations: list[dict] = []
    blocking_in_quality_flags: list[dict] = []
    total = 0

    for rec in records:
        for leader_key in ("torvalds", "kroah_hartman"):
            lr = rec[leader_key]
            evaluation = make_evaluation(lr)
            chunks = make_chunks(lr.get("chunk_contents", []))
            result = router.run(
                query=rec["query"],
                response_text=lr.get("clone_response_text", ""),
                chunks=chunks,
                evaluation=evaluation,
                leader=lr.get("leader", leader_key),
            )
            gs = lr["groundedness_score"]
            n_chunks = len(chunks)
            expected = "fallback" if (n_chunks == 0 or gs < GROUNDEDNESS_MIN) else "deliver"
            total += 1
            if result.decision != expected:
                violations.append({
                    "pass": rec["pass"],
                    "query_id": rec["query_id"],
                    "leader": leader_key,
                    "gs": gs,
                    "n_chunks": n_chunks,
                    "expected": expected,
                    "got": result.decision,
                    "trigger_category": result.trigger_category,
                })
            # quality_flags must not contain any blocking flag.
            bad_flags = [f for f in result.quality_flags if f in _BLOCKING_FLAGS]
            if bad_flags:
                blocking_in_quality_flags.append({
                    "query_id": rec["query_id"],
                    "leader": leader_key,
                    "decision": result.decision,
                    "bad_quality_flags": bad_flags,
                })

    print(f"Checked {total} records from {REEVAL_PATH}")
    if violations:
        print(f"FAIL: {len(violations)} decision violations")
        for v in violations[:10]:
            print(f"  {v}")
        sys.exit(1)
    else:
        print(
            f"PASS: delivers iff groundedness >= {GROUNDEDNESS_MIN} "
            f"— all {total} records correct"
        )

    if blocking_in_quality_flags:
        print(f"FAIL: {len(blocking_in_quality_flags)} records have blocking flags in quality_flags")
        for v in blocking_in_quality_flags[:10]:
            print(f"  {v}")
        sys.exit(1)
    else:
        print("PASS: no blocking flags in quality_flags across all 84 records")

    # ── (2) Zero-chunk unit case (empty_retrieval branch) ───────────────────────
    # Exercises the empty_retrieval branch (unreachable in stored data — all 84
    # records have 5 chunks). A zero-chunk case also scores gs=0.0 (below the floor);
    # the router must label it empty_retrieval, not low_groundedness, AND quality_flags
    # must be empty — low_groundedness is a blocking flag excluded from quality_flags.
    print("\n── Zero-chunk unit case (empty_retrieval branch) ──")
    eval_zero = EvaluationResult(
        groundedness_score=0.0,   # scorer returns 0.0 on empty chunks (groundedness_scorer.py:56)
        style_score=0.85,
        confidence_score=0.90,
        explanation="zero chunks retrieved; scorer returns 0.0 immediately",
        flags=["low_groundedness"],
    )
    result_zero = router.run(
        query="Why is the sky blue?",
        response_text="The sky is blue because of Rayleigh scattering.",
        chunks=[],
        evaluation=eval_zero,
        leader="torvalds",
    )
    assert result_zero.decision == "fallback", (
        f"Expected fallback, got {result_zero.decision}"
    )
    assert result_zero.trigger_category == "empty_retrieval", (
        f"Expected empty_retrieval, got {result_zero.trigger_category!r} — "
        "empty_retrieval must be checked BEFORE low_groundedness"
    )
    assert result_zero.trigger_reason == "empty_retrieval: 0 chunks retrieved", (
        f"Unexpected trigger_reason: {result_zero.trigger_reason!r}"
    )
    assert result_zero.quality_flags == [], (
        f"Expected empty quality_flags (low_groundedness is blocking, excluded), "
        f"got {result_zero.quality_flags}"
    )
    print(f"  decision:         {result_zero.decision}")
    print(f"  trigger_category: {result_zero.trigger_category}")
    print(f"  trigger_reason:   {result_zero.trigger_reason}")
    print(f"  quality_flags:    {result_zero.quality_flags}")
    print("PASS: zero-chunk unit case → trigger_category='empty_retrieval', quality_flags=[]")

    # ── (3) Deliver-low-style unit case (non-blocking flag on deliver path) ─────
    # gs=0.65 (>= 0.60 floor → deliver), ss=0.65 (< 0.70 → low_style fires),
    # cs=0.85 (>= 0.80 → low_confidence does not fire).
    # Expected: decision=deliver, quality_flags=['low_style'].
    print("\n── Deliver-low-style unit case (non-blocking flag on deliver path) ──")
    eval_low_style = EvaluationResult(
        groundedness_score=0.65,
        style_score=0.65,
        confidence_score=0.85,
        explanation="in-domain, grounded but off-style",
        flags=["low_style"],
    )
    result_low_style = router.run(
        query="What is memory management?",
        response_text="Memory management handles allocation and deallocation of RAM.",
        chunks=[
            RetrievalResult(
                chunk=KnowledgeChunk(
                    content="Memory management is critical.",
                    source_topic="kernel_memory",
                    source_field="synthetic",
                    chunk_index=0,
                ),
                score=0.8,
                rank=0,
            )
        ],
        evaluation=eval_low_style,
        leader="torvalds",
    )
    assert result_low_style.decision == "deliver", (
        f"Expected deliver (gs=0.65 >= 0.60), got {result_low_style.decision}"
    )
    assert result_low_style.trigger_category is None, (
        f"Expected trigger_category=None on deliver, got {result_low_style.trigger_category!r}"
    )
    assert result_low_style.quality_flags == ["low_style"], (
        f"Expected quality_flags=['low_style'], got {result_low_style.quality_flags}"
    )
    print(f"  decision:         {result_low_style.decision}")
    print(f"  trigger_category: {result_low_style.trigger_category}")
    print(f"  quality_flags:    {result_low_style.quality_flags}")
    print("PASS: deliver-low-style → quality_flags=['low_style'], no blocking flag")

    # ── (4) Evaluation-is-None unit case (evaluation_error guard in flow.py) ────
    # Exercises the emergency guard in route() when self.state.evaluation is None.
    # The guard must emit trigger_category='evaluation_error', not leave it null.
    # This imports DigitalCloneFlow but calls route() directly — no LLM call is made
    # because the guard fires before GatekeeperAgent.run().
    print("\n── Evaluation-is-None unit case (evaluation_error guard in flow.py) ──")
    from src.flow import DigitalCloneFlow
    flow = DigitalCloneFlow()
    flow.state.query = "Why is the sky blue?"
    flow.state.response_text = "The sky is blue because of Rayleigh scattering."
    flow.state.chunks = []
    flow.state.evaluation = None   # simulate evaluate step returning None
    route_result = flow.route()
    assert route_result == "fallback", (
        f"Expected 'fallback', got {route_result!r}"
    )
    rd = flow.state.routing_decision
    assert rd is not None, "routing_decision not set by guard"
    assert rd.trigger_category == "evaluation_error", (
        f"Expected 'evaluation_error', got {rd.trigger_category!r} — "
        "guard must emit trigger_category='evaluation_error', not null"
    )
    assert rd.trigger_reason == "evaluation_error: evaluate step returned None", (
        f"Unexpected trigger_reason: {rd.trigger_reason!r}"
    )
    assert rd.decision == "fallback", (
        f"Guard must set decision='fallback', got {rd.decision!r}"
    )
    print(f"  decision:         {rd.decision}")
    print(f"  trigger_category: {rd.trigger_category}")
    print(f"  trigger_reason:   {rd.trigger_reason}")
    print(f"  quality_flags:    {rd.quality_flags}")
    print("PASS: evaluation-is-None → trigger_category='evaluation_error'")


if __name__ == "__main__":
    main()
