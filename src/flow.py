"""DigitalCloneFlow: CrewAI Flow orchestrating the full query pipeline.

Steps (Phase 2, happy path): retrieve → style_response → evaluate_response → deliver
Router + fallback branch added in Phase 3.

Public API:
    DigitalCloneFlow().kickoff(inputs={"query": ..., "leader": ...})
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pydantic
from crewai.flow.flow import Flow, listen, router, start

from src.agents.evaluator_steps import EvaluatorAgent
from src.agents.fallback_steps import build_fallback_response
from src.agents.rag_agent import RAGAgent
from src.agents.style_crew import generate_styled_response
from src.config import load_config
from src.schemas import CloneState, EmailMessage, FallbackResponse, StyledResponse
from src.style.feature_extractor import extract_features
from src.style.profile_builder import load_profile

_LEADER_KEY_MAP: dict[str, str] = {
    "Linus Torvalds": "torvalds",
    "Greg Kroah-Hartman": "kroah_hartman",
}


class DigitalCloneFlow(Flow[CloneState]):
    """End-to-end query pipeline for a single-leader styled response."""

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._rag = RAGAgent(config=self._config)
        self._evaluator = EvaluatorAgent()

    # ------------------------------------------------------------------
    # Step 1: retrieve
    # ------------------------------------------------------------------

    @start()
    def retrieve(self) -> None:
        """Embed query → FAISS top-20 → Cohere rerank → top-5 chunks.

        Early-exits when retrieved_chunks is already populated (dual-leader
        retrieve-once optimization, Phase 4).
        """
        if self.state.retrieved_chunks:
            return
        try:
            self.state.retrieved_chunks = self._rag.retrieve(self.state.query)
        except (pydantic.ValidationError, AssertionError):
            raise
        except Exception as exc:
            self.state.trigger_reason = f"retrieve failed: {exc}"
            self.state.retrieved_chunks = []

    # ------------------------------------------------------------------
    # Step 2: style_response
    # ------------------------------------------------------------------

    @listen(retrieve)
    def style_response(self) -> None:
        """Load leader profile and invoke the single-agent style Crew."""
        if self.state.trigger_reason:
            return
        try:
            leader_key = _LEADER_KEY_MAP[self.state.leader]
            profile_path = Path(self._config.leaders[leader_key].profile_path)
            profile = load_profile(profile_path)
            self.state.styled_response = generate_styled_response(
                profile=profile,
                chunks=self.state.retrieved_chunks,
                query=self.state.query,
            )
        except (pydantic.ValidationError, AssertionError):
            raise
        except Exception as exc:
            self.state.trigger_reason = f"style_response failed: {exc}"

    # ------------------------------------------------------------------
    # Step 3: evaluate_response (router)
    # ------------------------------------------------------------------

    @router(style_response)
    def evaluate_response(self) -> str:
        """Score the styled response; return routing decision string."""
        if self.state.trigger_reason:
            return "fallback"
        try:
            leader_key = _LEADER_KEY_MAP[self.state.leader]
            profile_path = Path(self._config.leaders[leader_key].profile_path)
            profile = load_profile(profile_path)

            fake_email = EmailMessage(
                sender="generated",
                subject="response",
                body=self.state.styled_response,
                timestamp=datetime.now(tz=timezone.utc),
                message_id="generated-response",
            )
            response_features = extract_features(fake_email)

            self.state.evaluation = self._evaluator.evaluate(
                response=self.state.styled_response,
                chunks=self.state.retrieved_chunks,
                profile=profile,
                query=self.state.query,
                response_features=response_features,
            )
            return self.state.evaluation.decision
        except (pydantic.ValidationError, AssertionError):
            raise
        except Exception as exc:
            self.state.trigger_reason = f"evaluate_response failed: {exc}"
            return "fallback"

    # ------------------------------------------------------------------
    # Step 4a: finalize (happy path, route="deliver")
    # ------------------------------------------------------------------

    @listen("deliver")
    def finalize(self) -> None:
        """Assemble the final StyledResponse into state."""
        self.state.final_output = StyledResponse(
            query=self.state.query,
            leader=self.state.leader,
            response=self.state.styled_response,
            evaluation=self.state.evaluation,
        )

    # ------------------------------------------------------------------
    # Step 4b: handle_fallback (route="fallback")
    # ------------------------------------------------------------------

    @listen("fallback")
    def handle_fallback(self) -> None:
        """Build a FallbackResponse when evaluation score is below threshold."""
        trigger = self.state.trigger_reason or (
            f"final_score {self.state.evaluation.final_score:.4f} < threshold 0.75"
            if self.state.evaluation
            else "evaluation not completed"
        )
        try:
            self.state.final_output = build_fallback_response(
                query=self.state.query,
                chunks=self.state.retrieved_chunks,
                trigger_reason=trigger,
            )
        except (pydantic.ValidationError, AssertionError):
            raise
        except Exception as exc:
            self.state.final_output = FallbackResponse(
                trigger_reason=f"fallback itself failed: {exc}",
                context_summary="",
                calendar_link="",
                available_slots=[],
                unstyled_response="",
            )
