"""GatekeeperAgent: deterministic deliver-or-fallback routing (ADR-018).

Replaces the LLM decision (ADR-010) with arithmetic. The LLM was ignoring
the computed flags and applying its own unconstrained groundedness judgment
(RC-2: non-monotonic, flipping deliver/fallback/deliver across identical passes)
and mislabeling trigger_category regardless of actual flags (RC-3). Routing is
purely arithmetic — no LLM involved. See ADR-018.

Same run() signature and RoutingDecision return type as the old LLM-based
GatekeeperAgent. The flow contract is unchanged; only the internals are replaced.
"""

from __future__ import annotations

import logging

# Import thresholds from the evaluator so the router and evaluator stay in sync.
# Any future recalibration of a threshold automatically propagates here.
from src.agents.evaluator_agent import CONFIDENCE_MIN, GROUNDEDNESS_MIN, STYLE_MIN
from src.schemas import EvaluationResult, RetrievalResult, RoutingDecision

logger = logging.getLogger(__name__)

# Flags that promote to trigger_category and must NOT appear in quality_flags.
# quality_flags carries only non-blocking annotations (low_style, low_confidence).
_BLOCKING_FLAGS: frozenset[str] = frozenset({"low_groundedness"})


def _compute_flags(evaluation: EvaluationResult) -> list[str]:
    """Reproduce the deterministic flag set from scores (mirrors evaluator_agent.py)."""
    flags: list[str] = []
    if evaluation.groundedness_score < GROUNDEDNESS_MIN:
        flags.append("low_groundedness")
    if evaluation.style_score < STYLE_MIN:
        flags.append("low_style")
    if evaluation.confidence_score < CONFIDENCE_MIN:
        flags.append("low_confidence")
    return flags


class GatekeeperAgent:
    """Deterministic routing; retains the flow's run() → RoutingDecision contract (ADR-018)."""

    def __init__(self, model: str = "") -> None:
        # model param kept so GatekeeperAgent() call sites in flow.py need no change.
        pass

    def run(
        self,
        query: str,
        response_text: str,
        chunks: list[RetrievalResult],
        evaluation: EvaluationResult,
        leader: str,
    ) -> RoutingDecision:
        """Deterministic deliver-or-fallback in three steps (ADR-018).

        (a) Compute all flags from scores — same thresholds as evaluator_agent.py.
        (b) Label trigger_category; empty_retrieval MUST precede low_groundedness
            because a zero-chunk retrieval also scores gs=0.0 (below the floor) and
            would mislabel as low_groundedness without the chunk-count check first.
        (c) Fallback iff a blocking category was set. quality_flags carries
            non-blocking flags only (_BLOCKING_FLAGS excluded) on both paths.
            The blocking flag travels in trigger_category, not quality_flags.
        """
        # Step (a): recompute flags independently from scores.
        flags = _compute_flags(evaluation)
        # Non-blocking flags only — blocking flags are in trigger_category.
        quality_flags = [f for f in flags if f not in _BLOCKING_FLAGS]

        # Step (b): ordered category tree.
        # empty_retrieval is checked FIRST — chunk count is the only discriminant;
        # groundedness alone cannot distinguish zero-chunk from low-groundedness-by-content.
        trigger_category = None
        if len(chunks) == 0:
            trigger_category = "empty_retrieval"
        elif "low_groundedness" in flags:
            trigger_category = "low_groundedness"

        # Step (c): route and assemble RoutingDecision.
        # trigger_reason is a factual templated string; no LLM, no prose judgment.
        self.last_run_timings: dict[str, float] = {"generate_ms": 0.0, "parse_ms": 0.0}

        if trigger_category is not None:
            if trigger_category == "empty_retrieval":
                trigger_reason = "empty_retrieval: 0 chunks retrieved"
            else:
                trigger_reason = (
                    f"low_groundedness: groundedness "
                    f"{evaluation.groundedness_score:.2f} below "
                    f"{GROUNDEDNESS_MIN:.2f} floor"
                )
            reasoning = (
                f"Deterministic routing: {trigger_reason}. "
                f"gs={evaluation.groundedness_score:.3f} "
                f"ss={evaluation.style_score:.3f} "
                f"cs={evaluation.confidence_score:.3f} "
                f"flags={flags}"
            )
            return RoutingDecision(
                decision="fallback",
                reasoning=reasoning,
                trigger_category=trigger_category,
                trigger_reason=trigger_reason,
                quality_flags=quality_flags,
            )

        # Deliver: trigger_category and trigger_reason null.
        # quality_flags carries any non-blocking flags (low_style, low_confidence).
        reasoning = (
            f"Deterministic routing: deliver. "
            f"gs={evaluation.groundedness_score:.3f} >= {GROUNDEDNESS_MIN:.2f} floor. "
            f"quality_flags={flags}"
        )
        return RoutingDecision(
            decision="deliver",
            reasoning=reasoning,
            trigger_category=None,
            trigger_reason=None,
            quality_flags=quality_flags,
        )
