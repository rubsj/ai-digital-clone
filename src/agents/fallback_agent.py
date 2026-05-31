"""FallbackAgent: real CrewAI Agent for graceful redirection (ADR-012).

When GatekeeperAgent routes to fallback, this Agent generates a leader-voiced
acknowledgment and 2-3 in-domain redirections inferred from the retrieved chunks.
Calendar mock is generated deterministically (not by the LLM). A try/except
wraps the LLM path; any failure activates the templated failsafe so the system
always returns a usable FallbackResponse.
"""

from __future__ import annotations

import logging

import instructor
import litellm
from crewai import LLM, Agent, Crew, Task
from pydantic import BaseModel, Field

from src.fallback.calendar_mock import generate_available_slots
from src.schemas import FallbackResponse, RetrievalResult, StyleProfile

logger = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o-mini"
_MAX_CHUNKS = 5
_GEN_TEMPERATURE = 0.3
_PARSE_TEMPERATURE = 0
_LLM_MAX_RETRIES = 2
_CALENDAR_LINK = "https://cal.com/placeholder"

# Deterministic RNG seed for failsafe slots so tests can assert exact values.
_FAILSAFE_SEED = 42


class _FallbackDraft(BaseModel):
    """Instructor parse target: the LLM's acknowledgment and suggested redirections."""

    acknowledgment: str
    suggested_redirections: list[str] = Field(default_factory=list)


def _build_role(leader: str) -> str:
    return (
        f"You are {leader}, a Linux kernel maintainer, responding gracefully when "
        "you cannot give a well-grounded answer to a developer question."
    )


def _build_goal(leader: str) -> str:
    return (
        f"Write an honest, {leader}-voiced acknowledgment of why this question is "
        "outside what you can answer well right now, and suggest 2-3 adjacent "
        "questions you actually can answer based on the available material."
    )


def _build_backstory(leader: str) -> str:
    if leader.lower() in ("torvalds", "linus torvalds"):
        tone = "direct and terse — say what you mean in as few words as needed"
    else:
        tone = "measured and constructive — acknowledge the gap, then point to something useful"
    return (
        f"Your style is {tone}. Honesty about limitations builds trust. "
        "Redirect to topics the retrieved material actually covers."
    )


def _format_chunk_topics(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return "(no source material retrieved)"
    topics = [f"- {rr.chunk.source_topic}: {rr.chunk.content[:80]}" for rr in chunks[:_MAX_CHUNKS]]
    return "\n".join(topics)


def _build_task_description(
    query: str,
    trigger_reason: str,
    chunks: list[RetrievalResult],
) -> str:
    return (
        f"Original question: {query}\n\n"
        f"Why this is a fallback: {trigger_reason}\n\n"
        f"Available source material (use for redirection ideas):\n{_format_chunk_topics(chunks)}\n\n"
        "Write a brief acknowledgment in your voice, then list 2-3 specific questions "
        "that the source material above can actually answer."
    )


def _make_failsafe_response(leader: str) -> FallbackResponse:
    """Deterministic fallback when the LLM call fails."""
    slots = generate_available_slots(n=3, seed=_FAILSAFE_SEED)
    return FallbackResponse(
        acknowledgment=(
            f"Look, I'm {leader} and I can't give you a useful answer to this right now. "
            "The retrieved material doesn't cover it well enough."
        ),
        suggested_redirections=[],
        calendar_link=_CALENDAR_LINK,
        available_slots=slots,
        unstyled_response="The system was unable to generate a response at this time.",
    )


class FallbackAgent:
    """Generates a leader-voiced fallback acknowledgment with redirections and calendar mock."""

    def __init__(self, model: str = _LLM_MODEL) -> None:
        self._model = model

    def _build_crew(
        self,
        query: str,
        leader: str,
        trigger_reason: str,
        chunks: list[RetrievalResult],
    ) -> Crew:
        llm = LLM(model=self._model, temperature=_GEN_TEMPERATURE)
        agent = Agent(
            role=_build_role(leader),
            goal=_build_goal(leader),
            backstory=_build_backstory(leader),
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=_build_task_description(query, trigger_reason, chunks),
            expected_output="A brief acknowledgment followed by 2-3 specific redirections.",
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def _parse_draft(self, raw: str) -> _FallbackDraft:
        client = instructor.from_litellm(litellm.completion)
        prompt = (
            "Below is a fallback acknowledgment written by a Linux kernel maintainer.\n\n"
            f"Text:\n{raw}\n\n"
            "Extract the acknowledgment as a single string and the suggested redirections "
            "as a list of short question strings. If no redirections are present, return an "
            "empty list."
        )
        return client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_model=_FallbackDraft,
            temperature=_PARSE_TEMPERATURE,
            max_retries=_LLM_MAX_RETRIES,
        )

    def run(
        self,
        query: str,
        leader: str,
        trigger_reason: str,
        style_profile: StyleProfile,
        chunks: list[RetrievalResult],
    ) -> FallbackResponse:
        """Generate a leader-voiced fallback response, or return the templated failsafe on LLM failure."""
        slots = generate_available_slots(n=3)
        try:
            crew = self._build_crew(query, leader, trigger_reason, chunks)
            raw = crew.kickoff().raw
            draft = self._parse_draft(raw)
            return FallbackResponse(
                acknowledgment=draft.acknowledgment,
                suggested_redirections=draft.suggested_redirections,
                calendar_link=_CALENDAR_LINK,
                available_slots=slots,
                unstyled_response=raw,
            )
        except Exception as exc:
            logger.error(
                "FallbackAgent LLM call failed for leader=%r; activating templated failsafe. "
                "Error: %s",
                leader,
                exc,
            )
            failsafe = _make_failsafe_response(leader)
            # Replace slots with the already-generated real slots (same calendar mock data).
            return failsafe.model_copy(update={"available_slots": slots})
