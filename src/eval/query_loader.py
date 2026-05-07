"""Loads the shared experiment query set from a versioned JSON file."""

import json
from pathlib import Path


def load_queries(path: str | Path) -> list[dict]:
    """Return all query records from a queries_v*.json file.

    Each record has keys: id, query, topic, expected_groundedness_band.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array, got {type(records).__name__}")
    return records
