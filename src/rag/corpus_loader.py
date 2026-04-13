"""Load and filter the open-phi/textbooks HuggingFace dataset.

Returns RawDocument objects ready for chunking. The `topic` column in the
dataset is used directly; `_extract_topic` parses `outline` as a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset
from rich.progress import Progress


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
    max_docs: int | None = None,
) -> list[RawDocument]:
    """Load HuggingFace dataset, filter by field, return RawDocuments.

    max_docs caps the result count — essential for tests (5) and fast dev (20).
    Prefer the dataset's `topic` column; fall back to parsing `outline`.
    """
    ds = load_dataset(dataset_name, split="train")
    cs_ds = ds.filter(lambda row: row["field"] == field_filter)

    if max_docs is not None:
        cs_ds = cs_ds.select(range(min(max_docs, cs_ds.num_rows)))

    docs: list[RawDocument] = []
    with Progress() as progress:
        task = progress.add_task(f"Loading {field_filter} docs...", total=cs_ds.num_rows)
        for row in cs_ds:
            # Use topic column directly; fall back to parsing outline
            topic = (row.get("topic") or "").strip()
            if not topic:
                topic = _extract_topic(row.get("outline") or "")
            if not topic:
                topic = (row.get("subfield") or "unknown").replace("_", " ")

            text = (row.get("markdown") or "").strip()
            if not text:
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
