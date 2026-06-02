"""DigitalCloneFlow: CrewAI Flow orchestrating the full query pipeline.

v2 pipeline (Day 11, ADR-010/012): retrieve → clone → evaluate → route →
deliver|fallback. Each step calls a real CrewAI Agent. Retrieval is shared
across leaders in compare_leaders() (ADR-005). Profile is caller-injected via
kickoff(inputs={"style_profile": ...}); the Flow has no profile-load step.

Public API:
    DigitalCloneFlow().kickoff(inputs={"query": ..., "leader": ..., "style_profile": ...})
    compare_leaders(query) -> LeaderComparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from crewai.flow.flow import Flow, listen, router, start
from pydantic import PrivateAttr

from src.agents.clone_agent import CloneAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.fallback_agent import FallbackAgent
from src.agents.gatekeeper_agent import GatekeeperAgent
from src.components.retriever import Retriever
from src.config import load_config
from src.schemas import (
    CloneState,
    LeaderComparison,
    RoutingDecision,
    StyledResponse,
)
from src.style.profile_builder import load_profile

logger = logging.getLogger(__name__)

_LEADERS = ("Linus Torvalds", "Greg Kroah-Hartman")
_LEADER_KEY_MAP: dict[str, str] = {
    "Linus Torvalds": "torvalds",
    "Greg Kroah-Hartman": "kroah_hartman",
}


class DigitalCloneFlow(Flow[CloneState]):
    """End-to-end query pipeline for a single-leader styled response."""

    _timings: dict = PrivateAttr(default_factory=dict)

    @property
    def timings(self) -> dict[str, float]:
        """Per-stage wall-clock timings (ms). Not asserted — observability only."""
        return dict(self._timings)

    # ------------------------------------------------------------------
    # Step 1: retrieve
    # ------------------------------------------------------------------

    @start()
    def retrieve(self) -> None:
        """Embed query → FAISS top-20 → Cohere rerank → top-5 chunks.

        Early-exits when state.chunks is already populated (ADR-005 dual-leader
        retrieve-once optimization).
        """
        if self.state.chunks:
            return
        t0 = perf_counter()
        self.state.chunks = Retriever().run(self.state.query)
        self._timings["retrieve_ms"] = (perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # Step 2: clone
    # ------------------------------------------------------------------

    @listen(retrieve)
    def clone(self) -> None:
        """Call CloneAgent to generate a leader-styled response and citations."""
        agent = CloneAgent()
        t0 = perf_counter()
        result = agent.run(
            query=self.state.query,
            leader=self.state.leader,
            style_profile=self.state.style_profile,
            chunks=self.state.chunks,
        )
        self._timings["clone_ms"] = (perf_counter() - t0) * 1000
        tims = getattr(agent, "last_run_timings", {})
        self._timings["clone_generate_ms"] = tims.get("generate_ms", 0.0)
        self._timings["clone_parse_ms"] = tims.get("parse_ms", 0.0)
        self.state.response_text = result.response_text
        self.state.citations = list(result.citations)

    # ------------------------------------------------------------------
    # Step 3: evaluate
    # ------------------------------------------------------------------

    @listen(clone)
    def evaluate(self) -> None:
        """EvaluatorAgent: deterministic scores + LLM explanation/flags (ADR-011)."""
        agent = EvaluatorAgent()
        t0 = perf_counter()
        self.state.evaluation = agent.run(
            query=self.state.query,
            response=self.state.response_text or "",
            profile=self.state.style_profile,
            chunks=self.state.chunks,
        )
        self._timings["evaluate_ms"] = (perf_counter() - t0) * 1000
        tims = getattr(agent, "last_run_timings", {})
        self._timings["evaluate_score_ms"] = tims.get("score_ms", 0.0)
        self._timings["evaluate_generate_ms"] = tims.get("generate_ms", 0.0)
        self._timings["evaluate_parse_ms"] = tims.get("parse_ms", 0.0)

    # ------------------------------------------------------------------
    # Step 4: route (@router)
    # ------------------------------------------------------------------

    @router(evaluate)
    def route(self) -> str:
        """GatekeeperAgent decides deliver or fallback (ADR-010)."""
        if self.state.evaluation is None:
            self.state.routing_decision = RoutingDecision(
                decision="fallback",
                reasoning="evaluate step produced no result — emergency fallback",
                trigger_category="evaluation_error",
                trigger_reason="evaluation_error: evaluate step returned None",
            )
            return "fallback"
        agent = GatekeeperAgent()
        t0 = perf_counter()
        self.state.routing_decision = agent.run(
            query=self.state.query,
            response_text=self.state.response_text or "",
            chunks=self.state.chunks,
            evaluation=self.state.evaluation,
            leader=self.state.leader,
        )
        self._timings["route_ms"] = (perf_counter() - t0) * 1000
        tims = getattr(agent, "last_run_timings", {})
        self._timings["route_generate_ms"] = tims.get("generate_ms", 0.0)
        self._timings["route_parse_ms"] = tims.get("parse_ms", 0.0)
        return self.state.routing_decision.decision

    # ------------------------------------------------------------------
    # Step 5a: finalize (deliver arm)
    # ------------------------------------------------------------------

    @listen("deliver")
    def finalize(self) -> None:
        """Assemble the final StyledResponse into state."""
        t0 = perf_counter()
        self.state.styled_response = StyledResponse(
            query=self.state.query,
            leader=self.state.leader,
            response=self.state.response_text or "",
            evaluation=self.state.evaluation,
            citations=self.state.citations,
        )
        self._timings["deliver_ms"] = (perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # Step 5b: handle_fallback (fallback arm)
    # ------------------------------------------------------------------

    @listen("fallback")
    def handle_fallback(self) -> None:
        """FallbackAgent generates a leader-voiced fallback response (ADR-012/018)."""
        trigger = ""
        trigger_category = None
        if self.state.routing_decision:
            trigger = self.state.routing_decision.trigger_reason or ""
            trigger_category = self.state.routing_decision.trigger_category
        agent = FallbackAgent()
        t0 = perf_counter()
        self.state.fallback_response = agent.run(
            query=self.state.query,
            leader=self.state.leader,
            trigger_reason=trigger,
            style_profile=self.state.style_profile,
            chunks=self.state.chunks,
            trigger_category=trigger_category,
            groundedness_score=(
                self.state.evaluation.groundedness_score if self.state.evaluation else None
            ),
            style_score=(
                self.state.evaluation.style_score if self.state.evaluation else None
            ),
            confidence_score=(
                self.state.evaluation.confidence_score if self.state.evaluation else None
            ),
        )
        self._timings["fallback_ms"] = (perf_counter() - t0) * 1000
        tims = getattr(agent, "last_run_timings", {})
        self._timings["fallback_generate_ms"] = tims.get("generate_ms", 0.0)
        self._timings["fallback_parse_ms"] = tims.get("parse_ms", 0.0)


# ---------------------------------------------------------------------------
# Dual-leader comparison wrapper (ADR-005)
# ---------------------------------------------------------------------------


def compare_leaders(query: str) -> LeaderComparison:
    """Run the flow for both leaders, sharing retrieved chunks across runs.

    The first run (Torvalds) performs the RAG retrieval. The second run
    (Kroah-Hartman) receives those chunks pre-populated so its retrieve step
    early-exits — one embed + FAISS + rerank call instead of two (ADR-005).

    Both flows execute regardless of each other's outcome; asymmetric
    deliver/fallback outcomes are expected and surfaced faithfully.
    """
    config = load_config()

    profile_t = load_profile(Path(config.leaders["torvalds"].profile_path))
    flow_t = DigitalCloneFlow()
    flow_t.kickoff(inputs={
        "query": query,
        "leader": _LEADERS[0],
        "style_profile": profile_t,
    })

    shared_chunks = list(flow_t.state.chunks)

    profile_kh = load_profile(Path(config.leaders["kroah_hartman"].profile_path))
    flow_kh = DigitalCloneFlow()
    flow_kh.kickoff(inputs={
        "query": query,
        "leader": _LEADERS[1],
        "style_profile": profile_kh,
        "chunks": shared_chunks,
    })

    t_out = flow_t.state.styled_response or flow_t.state.fallback_response
    kh_out = flow_kh.state.styled_response or flow_kh.state.fallback_response

    if t_out is None or kh_out is None:
        raise ValueError(f"Pipeline produced no output — t={type(t_out)}, kh={type(kh_out)}")

    return LeaderComparison(query=query, torvalds=t_out, kroah_hartman=kh_out)
