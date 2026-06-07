"""EvaluatorAgent: hybrid quality reviewer (ADR-011).

Three deterministic scores come from ScoringEngine (no LLM). A real CrewAI
reviewer Agent then reasons about those scores against the response and its
sources; one Instructor parse structures that reasoning into an explanation.
Flags are raised deterministically in code from the ScoringEngine scores
(ADR-017 RC-1 fix: was previously LLM judgment, which drifted above threshold).
The three scores, explanation, and flags assemble into an EvaluationResult.
There is no combined score and no routing decision here: the Gatekeeper
owns routing (ADR-018). Both LLM steps run at temperature 0 for
deterministic review.
"""

from __future__ import annotations

import logging
from time import perf_counter

import instructor
import litellm
from crewai import LLM, Agent, Crew, Task
from pydantic import BaseModel

from src.components.scoring_engine import ScoringEngine, Scores
from src.schemas import EvaluationResult, RetrievalResult, StyleProfile

logger = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o-mini"
_MAX_CHUNKS = 5
_TEMPERATURE = 0
_LLM_MAX_RETRIES = 2

# ADR-017 RC-1 fix: flag thresholds as named constants, not LLM judgment.
# Previously the thresholds lived only as f-string literals in natural-language
# prose and the LLM drifted to flagging at ~0.70-0.75 instead of 0.60.
#
# W1b.2 operating point (ADR-020): 0.40 on HHEM's entailment scale, derived by a
# pre-registered safety-asymmetric rule against the Day-13 oracle (query-level
# train/held-out split). The cosine-era 0.60 does not transfer — HHEM's entailment
# scores run lower than cosine's echo-inflated scores, and carrying 0.60 would
# route ~42 % of oracle-grounded content to fallback. Per-leader floor (W3c,
# ADR-021) is handled separately and does not change this value.
GROUNDEDNESS_MIN: float = 0.40
# ADR-017 Amendment 1: corrected from 0.90 (synthetic-data calibration target per
# ADR-003) to 0.70 (ADR-003 full-corpus self-similarity benchmark). The 0.90 was
# never validated as a per-response flag threshold; 0.70 is the cosine proximity
# the leaders' own LKML corpus clears against the style profile.
STYLE_MIN: float = 0.70
CONFIDENCE_MIN: float = 0.80


class _ReviewDraft(BaseModel):
    """Instructor parse target for the reviewer's explanation prose."""

    explanation: str


def _build_role() -> str:
    return (
        "You are a quality reviewer for AI-generated responses written in the voice "
        "of a Linux kernel maintainer."
    )


def _build_goal() -> str:
    return (
        "Judge whether a response is well-styled, grounded in its sources, and "
        "confident, using the three measured scores. Explain the verdict in plain "
        "language, focusing on the weakest dimension."
    )


def _compute_flags(scores: Scores) -> list[str]:
    # ADR-017 RC-1 fix: threshold comparison is arithmetic, was previously
    # delegated to LLM judgment which drifted to flagging at ~0.70-0.75 in practice.
    flags = []
    if scores.groundedness_score < GROUNDEDNESS_MIN:
        flags.append("low_groundedness")
    if scores.style_score < STYLE_MIN:
        flags.append("low_style")
    if scores.confidence_score < CONFIDENCE_MIN:
        flags.append("low_confidence")
    return flags


def _build_backstory() -> str:
    return (
        "You have reviewed thousands of mailing-list replies. You are direct, you "
        "trust the measured scores over impressions, and you name concrete problems "
        "rather than hedging."
    )


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return "(no source chunks retrieved)"
    return "\n\n---\n\n".join(
        f"({rr.chunk.source_topic})\n{rr.chunk.content}" for rr in chunks[:_MAX_CHUNKS]
    )


def _build_task_description(query: str, response: str, scores: Scores,
                            chunks: list[RetrievalResult]) -> str:
    return (
        f"Query: {query}\n\n"
        f"Response under review:\n{response}\n\n"
        f"Source chunks the response should be grounded in:\n{_format_chunks(chunks)}\n\n"
        f"Measured scores (0-1):\n"
        f"  Style:        {scores.style_score:.3f} (stylistic match to the leader's voice)\n"
        f"  Groundedness: {scores.groundedness_score:.3f} (HHEM entailment; well-grounded in the retrieved context)\n"
        f"  Confidence:   {scores.confidence_score:.3f} (model's expressed certainty in the response)\n\n"
        "Write a concise explanation of the response's quality that references these "
        "scores and focuses on the weakest dimension."
    )


class EvaluatorAgent:
    """Hybrid reviewer: deterministic scores plus an LLM-reasoned explanation and flags."""

    def __init__(self, scoring_engine: ScoringEngine | None = None,
                 model: str = _LLM_MODEL) -> None:
        self._scoring = scoring_engine or ScoringEngine()
        self._model = model

    def _build_crew(self, query: str, response: str, scores: Scores,
                    chunks: list[RetrievalResult]) -> Crew:
        llm = LLM(model=self._model, temperature=_TEMPERATURE)
        agent = Agent(
            role=_build_role(),
            goal=_build_goal(),
            backstory=_build_backstory(),
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=_build_task_description(query, response, scores, chunks),
            expected_output="A short quality explanation of the response.",
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def _parse_review(self, raw: str) -> _ReviewDraft:
        client = instructor.from_litellm(litellm.completion)
        prompt = (
            "Below is a reviewer's verdict on a generated response.\n\n"
            f"Verdict:\n{raw}\n\n"
            "Return only the explanation text (the quality assessment prose). "
            "Do not include any flag labels or appended content."
        )
        return client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_model=_ReviewDraft,
            temperature=_TEMPERATURE,
            max_retries=_LLM_MAX_RETRIES,
        )

    def run(
        self,
        query: str,
        response: str,
        profile: StyleProfile,
        chunks: list[RetrievalResult],
    ) -> EvaluationResult:
        """Score deterministically, reason about the scores, assemble EvaluationResult."""
        t_score = perf_counter()
        scores = self._scoring.score(query, response, profile, chunks)
        t_gen = perf_counter()
        crew = self._build_crew(query, response, scores, chunks)
        raw = crew.kickoff().raw
        t_parse = perf_counter()
        draft = self._parse_review(raw)
        t_done = perf_counter()
        self.last_run_timings: dict[str, float] = {
            "score_ms": (t_gen - t_score) * 1000,
            "generate_ms": (t_parse - t_gen) * 1000,
            "parse_ms": (t_done - t_parse) * 1000,
        }
        return EvaluationResult(
            style_score=scores.style_score,
            groundedness_score=scores.groundedness_score,
            confidence_score=scores.confidence_score,
            explanation=draft.explanation,
            flags=_compute_flags(scores),  # ADR-017: deterministic, not from LLM
        )
