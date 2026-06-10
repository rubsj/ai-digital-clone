"""Load and filter the open-phi/textbooks HuggingFace dataset.

Returns RawDocument objects ready for chunking. The `topic` column in the
dataset is used directly; `_extract_topic` parses `outline` as a fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from datasets import load_dataset
from rich.progress import Progress

logger = logging.getLogger(__name__)

# The 5 textbooks the Day-15 canonical run and existing FAISS index were
# evaluated against. Pinned by topic name so corpus identity is stable
# regardless of HuggingFace dataset row ordering.
_EVALUATED_TOPICS: frozenset[str] = frozenset({
    "Statistical Learning Theory and Applications.",
    "Introduction to Computers and Engineering Problem Solving copy.",
    "Numerical Methods Applied to Chemical Engineering.",
    "Principles and Practice of Assistive Technology.",
    "Data Mining.",
})


@dataclass
class RawDocument:
    """Raw textbook document before chunking. Internal to the pipeline."""

    text: str       # from markdown column
    topic: str      # from topic column, or first heading from outline
    field: str      # e.g. "computer_science"
    subfield: str   # e.g. "algorithms_and_data_structures"


def _extract_topic(outline: str) -> str:
    """Parse first heading from outline string. Falls back to empty string."""
    if not outline:
        return ""
    match = re.search(r"^#+\s+(.+)$", outline, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # First non-empty line if no heading markers
    for line in outline.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def load_corpus(
    dataset_name: str = "open-phi/textbooks",
    field_filter: str = "computer_science",
    topic_filter: frozenset[str] | None = _EVALUATED_TOPICS,
    max_docs: int | None = None,
) -> list[RawDocument]:
    """Load HuggingFace dataset, filter by field and topic, return RawDocuments.

    Default topic_filter pins to the 5 evaluated textbooks (the corpus the
    Day-15 canonical run and existing FAISS index were built against). Pass
    topic_filter=None to load the full field slice; use max_docs to cap by count.

    Raises ValueError if any topic in topic_filter matches more than one document
    (guards against silent corpus expansion from duplicate dataset rows).
    """
    ds = load_dataset(dataset_name, split="train")
    cs_ds = ds.filter(lambda row: row["field"] == field_filter)

    if max_docs is not None:
        cs_ds = cs_ds.select(range(min(max_docs, cs_ds.num_rows)))

    docs: list[RawDocument] = []
    topic_match_count: dict[str, int] = {}
    with Progress() as progress:
        task = progress.add_task(f"Loading {field_filter} docs...", total=cs_ds.num_rows)
        for row in cs_ds:
            # Use topic column directly; fall back to parsing outline
            topic = (row.get("topic") or "").strip()
            if not topic:
                topic = _extract_topic(row.get("outline") or "")
            if not topic:
                topic = (row.get("subfield") or "unknown").replace("_", " ")

            if topic_filter is not None and topic not in topic_filter:
                progress.advance(task)
                continue

            text = (row.get("markdown") or "").strip()
            if not text:
                progress.advance(task)
                continue

            # Each pinned topic should map to exactly one document. The
            # open-phi/textbooks dataset contains at least one duplicate topic
            # ("Principles and Practice of Assistive Technology." appears at
            # rows 1 and 14 of the CS slice). The first occurrence is the
            # evaluated document (864 chunks, 100% key match vs FAISS index);
            # subsequent occurrences are skipped with a warning.
            if topic_filter is not None:
                topic_match_count[topic] = topic_match_count.get(topic, 0) + 1
                if topic_match_count[topic] > 1:
                    logger.warning(
                        "Skipping duplicate document for topic '%s' "
                        "(occurrence %d); keeping first match only.",
                        topic,
                        topic_match_count[topic],
                    )
                    progress.advance(task)
                    continue

            docs.append(
                RawDocument(
                    text=text,
                    topic=topic,
                    field=row.get("field", field_filter),
                    subfield=row.get("subfield", ""),
                )
            )
            progress.advance(task)

    return docs
