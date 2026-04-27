"""Tests for src/eval/query_loader.py."""

import json
import pytest
from pathlib import Path
from src.eval.query_loader import load_queries


@pytest.fixture
def query_file(tmp_path: Path) -> Path:
    records = [
        {"id": "q01", "query": "What is TCP?", "topic": "networking", "expected_groundedness_band": "high"},
        {"id": "q02", "query": "What is virtual memory?", "topic": "operating_systems", "expected_groundedness_band": "medium"},
    ]
    p = tmp_path / "queries_test.json"
    p.write_text(json.dumps(records))
    return p


def test_load_queries_returns_list(query_file: Path) -> None:
    result = load_queries(query_file)
    assert isinstance(result, list)
    assert len(result) == 2


def test_load_queries_record_shape(query_file: Path) -> None:
    result = load_queries(query_file)
    for record in result:
        assert "id" in record
        assert "query" in record
        assert "topic" in record
        assert "expected_groundedness_band" in record


def test_load_queries_raises_on_non_array(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"id": "q01"}))
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_queries(p)


def test_load_queries_canonical_file() -> None:
    canonical = Path("data/eval/queries_v2.json")
    records = load_queries(canonical)
    assert len(records) == 10
    bands = {r["expected_groundedness_band"] for r in records}
    assert bands <= {"high", "medium", "low"}
