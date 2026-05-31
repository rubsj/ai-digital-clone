"""GatekeeperAgent: real CrewAI Agent for deliver-or-fallback routing (ADR-010).

Inputs: query, response_text, chunks, evaluation (EvaluationResult), leader.
Output: RoutingDecision. Runs at temperature=0. On fallback, the prompt demands
trigger_category from the bounded 5-literal set and trigger_reason as free text
referencing specific scores/flags. On deliver, both are null. Routing correctness
is measured with real LLM on Day 12 — a green suite here proves plumbing only.
"""

from __future__ import annotations

import logging

import instructor
import litellm
from crewai import LLM, Agent, Crew, Task

from src.schemas import EvaluationResult, RetrievalResult, RoutingDecision

logger = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o-mini"
_MAX_CHUNKS = 5
_TEMPERATURE = 0
_LLM_MAX_RETRIES = 2

_TRIGGER_CATEGORIES = (
    "low_groundedness",
    "off_domain",
    "hallucination_risk",
    "chunk_mismatch",
    "empty_retrieval",
)


def _build_role() -> str:
    return (
        "You are a routing decision-maker for an AI digital clone system that "
        "generates responses in the voice of a Linux kernel maintainer."
    )


def _build_goal() -> str:
    return (
        "Decide whether to DELIVER a generated response or route to FALLBACK, "
        "based solely on the three measured quality scores and evaluation flags "
        "provided. Your reasoning must cite the specific score values and flags "
        "you received — not general impressions."
    )


def _build_backstory() -> str:
    return (
        "You are conservative about hallucination risk: a confident-sounding "
        "response that is poorly grounded is worse than an honest fallback. "
        "Route to fallback when groundedness is low, when flags signal chunk "
        "mismatch or off-domain content, or when no usable evidence was retrieved. "
        "Default to deliver when scores are reasonable and no flags are raised."
    )


def _format_chunks_summary(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    topics = [f"- {rr.chunk.source_topic}" for rr in chunks[:_MAX_CHUNKS]]
    return "\n".join(topics)


def _build_task_description(
    query: str,
    response_text: str,
    chunks: list[RetrievalResult],
    evaluation: EvaluationResult,
) -> str:
    flags_str = ", ".join(evaluation.flags) if evaluation.flags else "(none)"
    categories_str = ", ".join(f'"{c}"' for c in _TRIGGER_CATEGORIES)
    return (
        f"Query: {query}\n\n"
        f"Generated response:\n{response_text}\n\n"
        f"Retrieved chunk topics:\n{_format_chunks_summary(chunks)}\n\n"
        "Measured quality scores (0-1):\n"
        f"  style_score:        {evaluation.style_score:.3f}\n"
        f"  groundedness_score: {evaluation.groundedness_score:.3f}\n"
        f"  confidence_score:   {evaluation.confidence_score:.3f}\n\n"
        f"Evaluation flags: {flags_str}\n\n"
        f"Evaluator explanation: {evaluation.explanation}\n\n"
        "ROUTING RULES:\n"
        "- Default: DELIVER. Route to FALLBACK only when a specific score or flag "
        "warrants it.\n"
        "- Your reasoning MUST cite the specific score values and flags listed above.\n"
        "- If decision is 'fallback': set trigger_reason (free-text explanation "
        "citing specific scores/flags) and trigger_category (one of: "
        f"{categories_str}).\n"
        "- If decision is 'deliver': trigger_reason and trigger_category must both "
        "be null.\n"
    )


class GatekeeperAgent:
    """Routes each response to deliver or fallback based on evaluated quality."""

    def __init__(self, model: str = _LLM_MODEL) -> None:
        self._model = model

    def _build_crew(
        self,
        query: str,
        response_text: str,
        chunks: list[RetrievalResult],
        evaluation: EvaluationResult,
        leader: str,
    ) -> Crew:
        llm = LLM(model=self._model, temperature=_TEMPERATURE)
        agent = Agent(
            role=_build_role(),
            goal=_build_goal(),
            backstory=_build_backstory(),
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=_build_task_description(query, response_text, chunks, evaluation),
            expected_output=(
                "A routing decision: decision ('deliver' or 'fallback'), reasoning "
                "citing the specific scores and flags, trigger_category and "
                "trigger_reason set only on fallback, null on deliver."
            ),
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def _parse_decision(self, raw: str) -> RoutingDecision:
        client = instructor.from_litellm(litellm.completion)
        categories_str = ", ".join(_TRIGGER_CATEGORIES)
        prompt = (
            "Below is a routing verdict for an AI-generated response.\n\n"
            f"Verdict:\n{raw}\n\n"
            "Extract the routing decision as structured output. Set decision to "
            "'deliver' or 'fallback', reasoning to the rationale text. If decision "
            f"is 'fallback', set trigger_category to one of: {categories_str}, and "
            "trigger_reason to the free-text explanation. If decision is 'deliver', "
            "both trigger_category and trigger_reason must be null."
        )
        return client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_model=RoutingDecision,
            temperature=_TEMPERATURE,
            max_retries=_LLM_MAX_RETRIES,
        )

    def run(
        self,
        query: str,
        response_text: str,
        chunks: list[RetrievalResult],
        evaluation: EvaluationResult,
        leader: str,
    ) -> RoutingDecision:
        """Decide deliver or fallback; on fallback set trigger_reason + trigger_category."""
        crew = self._build_crew(query, response_text, chunks, evaluation, leader)
        raw = crew.kickoff().raw
        return self._parse_decision(raw)
