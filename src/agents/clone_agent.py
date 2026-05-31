"""CloneAgent: real CrewAI Agent that writes a leader-styled, grounded response.

A class wrapping a CrewAI Agent (role/goal/backstory) + a single Task run via a
single-agent Crew. The Crew kickoff produces the styled prose; one Instructor
call then parses that prose into a typed CloneResponse (ADR-009, Rule 4). The
parse emits the chunk indices the response drew on; run() reconciles each index
to a full Citation from the input chunks. Generation runs at temperature 0.3 for
voice variation; the parse runs at temperature 0 for deterministic structuring.
"""

from __future__ import annotations

import logging
from time import perf_counter

import instructor
import litellm
from crewai import LLM, Agent, Crew, Task
from pydantic import BaseModel

from src.schemas import Citation, CloneResponse, RetrievalResult, StyleProfile

logger = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o-mini"
_MAX_CHUNKS = 5
_MAX_SAMPLE_EMAILS = 3
_GEN_TEMPERATURE = 0.3
_PARSE_TEMPERATURE = 0
_LLM_MAX_RETRIES = 2
_SNIPPET_LEN = 100


class _CloneDraft(BaseModel):
    """Instructor parse target. cited_chunk_indices are 0-based into the input chunks."""

    response_text: str
    cited_chunk_indices: list[int]


def _build_role(leader: str) -> str:
    return (
        f"You are {leader}, a Linux kernel maintainer writing a response "
        "to a developer question on the Linux Kernel Mailing List (LKML)."
    )


def _build_goal(leader: str, profile: StyleProfile) -> str:
    f = profile.features
    phrases = ", ".join(f.common_phrases[:3]) if f.common_phrases else "none"
    return (
        f"Produce a response that precisely mirrors {leader}'s measurable style: "
        f"avg_message_length={f.avg_message_length:.3f} (normalized 0-1), "
        f"formality_level={f.formality_level:.3f}, "
        f"technical_depth={f.technical_depth:.3f}, "
        f"vocabulary_richness={f.vocabulary_richness:.3f}. "
        f"Characteristic phrases include: {phrases}. "
        "Write in the first person. Ground every factual claim in the provided context."
    )


def _build_backstory(leader: str, profile: StyleProfile) -> str:
    f = profile.features
    tone = "direct and blunt" if f.formality_level < 0.55 else "clear and structured"
    return (
        f"{leader} writes with code_snippet_freq={f.code_snippet_freq:.3f} "
        f"and question_frequency={f.question_frequency:.3f}. "
        f"The style tends toward {tone}, "
        f"with technical_terminology={f.technical_terminology:.3f}."
    )


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    """Number chunks 0-based so the parse can cite them by index."""
    return "\n\n---\n\n".join(
        f"[{i}] ({rr.chunk.source_topic})\n{rr.chunk.content}"
        for i, rr in enumerate(chunks[:_MAX_CHUNKS])
    )


def _format_style_examples(profile: StyleProfile) -> str:
    samples = profile.sample_emails[:_MAX_SAMPLE_EMAILS]
    if not samples:
        return ""
    joined = "\n\n---\n\n".join(samples)
    return f"\n\nStyle examples ({profile.leader_name}'s own emails):\n{joined}"


class CloneAgent:
    """Generates a leader-styled, context-grounded response with reconciled citations."""

    def __init__(self, model: str = _LLM_MODEL) -> None:
        self._model = model

    def _build_crew(self, query: str, leader: str, profile: StyleProfile,
                    chunks: list[RetrievalResult]) -> Crew:
        llm = LLM(model=self._model, temperature=_GEN_TEMPERATURE)
        agent = Agent(
            role=_build_role(leader),
            goal=_build_goal(leader, profile),
            backstory=_build_backstory(leader, profile),
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=(
                f"Query: {query}\n\n"
                f"Context from retrieved knowledge:\n{_format_chunks(chunks)}"
                f"{_format_style_examples(profile)}\n\n"
                f"Write a response to the query in {leader}'s style, grounded in the "
                "above context. Do not introduce facts not present in the context."
            ),
            expected_output=f"A response in {leader}'s voice, 1-3 paragraphs.",
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def _parse_citations(self, raw: str, chunks: list[RetrievalResult]) -> _CloneDraft:
        client = instructor.from_litellm(litellm.completion)
        prompt = (
            "Below is a response and the numbered source chunks it was written from.\n\n"
            f"Response:\n{raw}\n\n"
            f"Source chunks:\n{_format_chunks(chunks)}\n\n"
            "Return the response text exactly as given, and the 0-based indices of the "
            "source chunks the response actually draws on. Only use indices that appear "
            "in the list above."
        )
        return client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_model=_CloneDraft,
            temperature=_PARSE_TEMPERATURE,
            max_retries=_LLM_MAX_RETRIES,
        )

    def _reconcile(self, indices: list[int],
                   chunks: list[RetrievalResult]) -> list[Citation]:
        """Map each cited chunk index to a full Citation; drop out-of-range and dupes."""
        citations: list[Citation] = []
        seen: set[int] = set()
        for idx in indices:
            if idx < 0 or idx >= len(chunks):
                logger.warning(
                    "CloneAgent cited out-of-range chunk index %d (have %d chunks); dropping.",
                    idx,
                    len(chunks),
                )
                continue
            if idx in seen:
                continue
            seen.add(idx)
            rr = chunks[idx]
            citations.append(
                Citation(
                    chunk_id=f"chunk_{rr.chunk.chunk_index}",
                    source_topic=rr.chunk.source_topic,
                    text_snippet=rr.chunk.content[:_SNIPPET_LEN],
                    relevance_score=min(max(rr.score, 0.0), 1.0),
                )
            )
        return citations

    def run(
        self,
        query: str,
        leader: str,
        style_profile: StyleProfile,
        chunks: list[RetrievalResult],
    ) -> CloneResponse:
        """Generate a styled response and reconcile its citations to input chunks."""
        crew = self._build_crew(query, leader, style_profile, chunks)
        t_gen = perf_counter()
        raw = crew.kickoff().raw
        t_parse = perf_counter()
        draft = self._parse_citations(raw, chunks)
        t_done = perf_counter()
        self.last_run_timings: dict[str, float] = {
            "generate_ms": (t_parse - t_gen) * 1000,
            "parse_ms": (t_done - t_parse) * 1000,
        }
        return CloneResponse(
            response_text=draft.response_text,
            citations=self._reconcile(draft.cited_chunk_indices, chunks),
        )
