"""Parse [N] citation markers from generated text and map to RetrievalResults.

Citations use 1-based indexing in the text ([1] → results[0]).
Out-of-range refs are silently skipped. Duplicates are deduplicated.
"""

from __future__ import annotations

import re

from src.schemas import Citation, RetrievalResult

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(
    text: str,
    retrieved: list[RetrievalResult],
) -> list[Citation]:
    """Parse [N] refs from text, map to retrieved chunks.

    - 1-indexed: [1] → retrieved[0]
    - Skips refs where index < 0 or index >= len(retrieved)
    - Deduplicates by source index (first occurrence wins)
    - relevance_score clamped to [0.0, 1.0] for Pydantic validation
    """
    if not text or not retrieved:
        return []

    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _CITATION_PATTERN.finditer(text):
        n = int(match.group(1))
        idx = n - 1  # convert 1-based to 0-based
        if idx < 0 or idx >= len(retrieved):
            continue
        if idx in seen:
            continue
        seen.add(idx)

        result = retrieved[idx]
        chunk = result.chunk
        citations.append(
            Citation(
                chunk_id=f"chunk_{chunk.chunk_index}",
                source_topic=chunk.source_topic,
                text_snippet=chunk.content[:100],
                relevance_score=min(max(result.score, 0.0), 1.0),
            )
        )

    return citations
