"""FallbackSteps: facade that composes all fallback modules into a FallbackResponse.

Called by the Day 5 flow when EvaluatorAgent returns decision="fallback".
"""

from __future__ import annotations

from src.fallback.calendar_mock import generate_available_slots
from src.fallback.context_summarizer import summarize_context
from src.fallback.unstyled_responder import generate_unstyled_response
from src.schemas import FallbackResponse, RetrievalResult

# Placeholder link — no real calendar integration in this project scope.
_CALENDAR_LINK = "https://cal.com/placeholder"


def build_fallback_response(
    query: str,
    chunks: list[RetrievalResult],
    trigger_reason: str,
) -> FallbackResponse:
    """Build a complete FallbackResponse from retrieved chunks.

    Composes:
      - generate_available_slots()   → 3 business-day slots
      - summarize_context()          → 1–2 sentence topic summary
      - generate_unstyled_response() → factual LLM answer (no style)

    Args:
        query:          Original user query.
        chunks:         Retrieved chunks from the RAG pipeline.
        trigger_reason: Human-readable reason the fallback was triggered
                        (e.g. "final_score 0.62 < threshold 0.75").

    Returns:
        FallbackResponse with all fields populated.
    """
    slots = generate_available_slots(n=3)
    context_summary = summarize_context(query, chunks)
    unstyled = generate_unstyled_response(query, chunks)

    return FallbackResponse(
        trigger_reason=trigger_reason,
        context_summary=context_summary,
        calendar_link=_CALENDAR_LINK,
        available_slots=slots,
        unstyled_response=unstyled,
    )
