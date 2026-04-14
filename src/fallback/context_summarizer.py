"""Context summarizer: deterministic string composition — no LLM call.

Extracts unique source_topic values from retrieved chunks and builds a
1–2 sentence context string for the fallback response.
"""

from __future__ import annotations

from src.schemas import RetrievalResult


def summarize_context(query: str, chunks: list[RetrievalResult]) -> str:
    """Summarize query + retrieved chunk topics into a short context string.

    Args:
        query:  Original user query.
        chunks: Retrieved chunks from the RAG pipeline.

    Returns:
        A 1–2 sentence string suitable for the fallback response.
        Falls back gracefully when chunks is empty.
    """
    if not chunks:
        return (
            f'Your question about "{query}" requires more context than I '
            "currently have available. I'd like to discuss this in more depth."
        )

    # Deduplicate topics while preserving order.
    seen: set[str] = set()
    topics: list[str] = []
    for rr in chunks:
        t = rr.chunk.source_topic.strip()
        if t and t not in seen:
            seen.add(t)
            topics.append(t)

    # Build query topic from first 6 words of query to keep it brief.
    query_words = query.strip().split()
    query_topic = " ".join(query_words[:6])
    if len(query_words) > 6:
        query_topic += "…"

    if len(topics) == 1:
        topics_joined = topics[0]
    elif len(topics) == 2:
        topics_joined = f"{topics[0]} and {topics[1]}"
    else:
        topics_joined = ", ".join(topics[:-1]) + f", and {topics[-1]}"

    return (
        f'Your question about "{query_topic}" touches on {topics_joined}. '
        "I'd like to discuss this in more depth."
    )
