"""Unstyled responder: factual LLM response from retrieved chunks, no style applied.

Uses Instructor + LiteLLM (gpt-4o-mini) to generate a plain, factual answer.
The system prompt explicitly forbids rhetorical style and personality cues so
this response is clearly distinct from the styled RAG output.
"""

from __future__ import annotations

import litellm
import instructor
from pydantic import BaseModel

from src.schemas import RetrievalResult

_LLM_MODEL = "gpt-4o-mini"
_LLM_MAX_RETRIES = 3

_SYSTEM_PROMPT = (
    "You are a factual assistant. Answer the user's question using ONLY the "
    "provided source excerpts. Do not add rhetorical style, personality cues, "
    "metaphors, or leadership coaching framing. Write plainly and concisely. "
    "If the excerpts do not contain enough information, say so directly."
)


class UnstyledAnswer(BaseModel):
    """Structured output for the unstyled responder."""

    answer: str


def _build_user_prompt(query: str, chunks: list[RetrievalResult]) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] ({rr.chunk.source_topic}) {rr.chunk.content[:400]}"
        for i, rr in enumerate(chunks)
    )
    return f"Question: {query}\n\nSource excerpts:\n{excerpts}"


def generate_unstyled_response(
    query: str,
    chunks: list[RetrievalResult],
) -> str:
    """Generate a plain factual answer from retrieved chunks via Instructor + LiteLLM.

    Args:
        query:  Original user query.
        chunks: Retrieved chunks to ground the answer.

    Returns:
        Plain-text factual answer string.
    """
    client = instructor.from_litellm(litellm.completion)
    result: UnstyledAnswer = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, chunks)},
        ],
        response_model=UnstyledAnswer,
        max_retries=_LLM_MAX_RETRIES,
    )
    return result.answer
