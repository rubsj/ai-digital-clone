"""Public re-exports for the src.fallback package."""

from src.fallback.calendar_mock import generate_available_slots
from src.fallback.context_summarizer import summarize_context
from src.fallback.unstyled_responder import generate_unstyled_response

__all__ = ["generate_available_slots", "summarize_context", "generate_unstyled_response"]
