"""ChatStyleAgent: single-agent CrewAI Crew for style-conditioned response generation.

This is the only place in the codebase that uses CrewAI Agent/Crew/Task.
All other pipeline steps are plain Python calls.

Public API:
    build_style_crew(profile, chunks, query) -> Crew
    generate_styled_response(profile, chunks, query) -> str
"""

from __future__ import annotations

from crewai import Agent, Crew, LLM, Task

from src.schemas import RetrievalResult, StyleProfile

_LLM_MODEL = "gpt-4o-mini"
_MAX_CHUNKS = 5


def _build_role(profile: StyleProfile) -> str:
    return (
        f"You are {profile.leader_name}, a Linux kernel maintainer writing a response "
        "to a developer question on the Linux Kernel Mailing List (LKML)."
    )


def _build_goal(profile: StyleProfile) -> str:
    f = profile.features
    phrases = ", ".join(f.common_phrases[:3]) if f.common_phrases else "none"
    return (
        f"Produce a response that precisely mirrors {profile.leader_name}'s measurable style: "
        f"avg_message_length={f.avg_message_length:.3f} (normalized 0-1), "
        f"formality_level={f.formality_level:.3f}, "
        f"technical_depth={f.technical_depth:.3f}, "
        f"vocabulary_richness={f.vocabulary_richness:.3f}. "
        f"Characteristic phrases include: {phrases}. "
        "Write in the first person. Ground every factual claim in the provided context."
    )


def _build_backstory(profile: StyleProfile) -> str:
    f = profile.features
    tone = "direct and blunt" if f.formality_level < 0.55 else "clear and structured"
    return (
        f"{profile.leader_name} writes with code_snippet_freq={f.code_snippet_freq:.3f} "
        f"and question_frequency={f.question_frequency:.3f}. "
        f"The style tends toward {tone}, "
        f"with technical_terminology={f.technical_terminology:.3f}."
    )


def build_style_crew(
    profile: StyleProfile,
    chunks: list[RetrievalResult],
    query: str,
) -> Crew:
    """Build a single-agent Crew bound to a specific leader's StyleProfile."""
    llm = LLM(model=_LLM_MODEL)

    agent = Agent(
        role=_build_role(profile),
        goal=_build_goal(profile),
        backstory=_build_backstory(profile),
        llm=llm,
        verbose=False,
    )

    chunk_texts = "\n\n---\n\n".join(
        f"[{rr.chunk.source_topic}]\n{rr.chunk.content}" for rr in chunks[:_MAX_CHUNKS]
    )

    task = Task(
        description=(
            f"Query: {query}\n\n"
            f"Context from retrieved knowledge:\n{chunk_texts}\n\n"
            f"Write a response to the query in {profile.leader_name}'s style, "
            "grounded in the above context. Do not introduce facts not present in the context."
        ),
        expected_output=f"A response in {profile.leader_name}'s voice, 1-3 paragraphs.",
        agent=agent,
    )

    return Crew(agents=[agent], tasks=[task], verbose=False)


def generate_styled_response(
    profile: StyleProfile,
    chunks: list[RetrievalResult],
    query: str,
) -> str:
    """Build and kick off the style Crew; return the raw response string."""
    crew = build_style_crew(profile=profile, chunks=chunks, query=query)
    result = crew.kickoff()
    return result.raw
