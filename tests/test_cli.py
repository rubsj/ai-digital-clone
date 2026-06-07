"""Tests for Click CLI commands in src/cli.py.

All heavy I/O (LLM, FAISS, Cohere, corpus load, mbox parse) is mocked at
the src.cli module boundary via unittest.mock.patch. No real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.config import (
    AppConfig,
    ChunkingConfig,
    DateRangeConfig,
    EmbeddingConfig,
    LeaderConfig,
    LLMConfig,
    RerankerConfig,
    ScoringConfig,
    StyleConfig,
)
from src.schemas import EvaluationResult, FallbackResponse, LeaderComparison, StyledResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(
            primary_model="text-embedding-3-small",
            baseline_model="all-MiniLM-L6-v2",
            dimension=1536,
        ),
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
        reranker=RerankerConfig(
            provider="cohere",
            model="rerank-english-v3.0",
            top_n_initial=20,
            top_n_final=5,
        ),
        scoring=ScoringConfig(
            style_weight=0.4,
            groundedness_weight=0.4,
            confidence_weight=0.2,
            fallback_threshold=0.75,
        ),
        llm=LLMConfig(model="gpt-4o-mini"),
        leaders={
            "torvalds": LeaderConfig(
                name="Linus Torvalds",
                email_filter="torvalds@",
                mbox_path="data/emails/torvalds.mbox",
                profile_path="data/models/torvalds_profile.json",
            ),
            "kroah_hartman": LeaderConfig(
                name="Greg Kroah-Hartman",
                email_filter="gregkh@",
                mbox_path="data/emails/kroah_hartman.mbox",
                profile_path="data/models/kroah_hartman_profile.json",
            ),
        },
        style=StyleConfig(
            alpha=0.3,
            min_email_words=20,
            date_range=DateRangeConfig(start="2015-01-01", end="2023-12-31"),
        ),
    )


def _make_evaluation() -> MagicMock:
    ev = MagicMock(spec=EvaluationResult)
    ev.style_score = 0.80
    ev.groundedness_score = 0.90
    ev.confidence_score = 0.70
    ev.explanation = "Well grounded response."
    ev.flags = []
    return ev


def _make_styled_response() -> MagicMock:
    r = MagicMock(spec=StyledResponse)
    r.response = "Here is the styled response."
    r.evaluation = _make_evaluation()
    return r


def _make_fallback_response() -> MagicMock:
    fb = MagicMock(spec=FallbackResponse)
    fb.acknowledgment = "This topic is outside my grounded knowledge base."
    fb.suggested_redirections = ["See the kernel documentation."]
    fb.unstyled_response = "Here is a plain answer."
    fb.calendar_link = "https://calendar.example.com"
    fb.available_slots = []
    return fb


def _make_leader_record(decision: str = "deliver", leader_key: str = "torvalds") -> dict:
    """Harness-schema per-leader record for mocking _run_leader_pair."""
    if decision == "deliver":
        return {
            "leader": "Linus Torvalds" if leader_key == "torvalds" else "Greg Kroah-Hartman",
            "leader_key": leader_key,
            "decision": "deliver",
            "trigger_category": None,
            "trigger_reason": None,
            "routing_reasoning": "groundedness 0.75 >= 0.40",
            "clone_response_text": "TCP is a transport protocol.",
            "delivered_text": "TCP is a transport protocol.",
            "output_type": "StyledResponse",
            "style_score": 0.80,
            "groundedness_score": 0.75,
            "confidence_score": 0.70,
            "flags": [],
            "chunk_contents": [],
            "timings": {},
        }
    return {
        "leader": "Linus Torvalds" if leader_key == "torvalds" else "Greg Kroah-Hartman",
        "leader_key": leader_key,
        "decision": "fallback",
        "trigger_category": "low_groundedness",
        "trigger_reason": "groundedness 0.30 < 0.40",
        "routing_reasoning": "groundedness below floor",
        "clone_response_text": "Some response.",
        "delivered_text": "This is outside my expertise.",
        "output_type": "FallbackResponse",
        "style_score": 0.70,
        "groundedness_score": 0.30,
        "confidence_score": 0.60,
        "flags": ["low_groundedness"],
        "chunk_contents": [],
        "timings": {},
    }


def _make_pair_result(t_decision: str = "deliver", kh_decision: str = "deliver") -> dict:
    """Full harness run_leader_pair result for one query."""
    return {
        "query": "What is TCP?",
        "retriever_call_count": 1,
        "torvalds": _make_leader_record(t_decision, "torvalds"),
        "kroah_hartman": _make_leader_record(kh_decision, "kroah_hartman"),
    }


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------


class TestLearnCommand:
    def test_success_with_mbox_override(self, runner: CliRunner, tmp_path: Path) -> None:
        mbox = tmp_path / "test.mbox"
        mbox.touch()
        profile = MagicMock()
        profile.email_count = 42

        with (
            patch("src.cli.load_config", return_value=_make_config()),
            patch("src.cli.parse_mbox", return_value=[MagicMock()]) as mock_parse,
            patch("src.cli.extract_features", return_value=MagicMock()),
            patch("src.cli.build_profile_batch", return_value=profile) as mock_build,
            patch("src.cli.save_profile") as mock_save,
        ):
            result = runner.invoke(
                cli, ["learn", "--leader", "torvalds", "--mbox", str(mbox)]
            )

        assert result.exit_code == 0, result.output
        assert "42 emails" in result.output
        mock_parse.assert_called_once()
        mock_build.assert_called_once()
        mock_save.assert_called_once()

    def test_missing_leader_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["learn"])
        assert result.exit_code == 2
        assert "Missing option" in result.output


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


class TestIndexCommand:
    def test_success(self, runner: CliRunner) -> None:
        mock_retriever = MagicMock()
        sentinel_chunks = [MagicMock()]

        with (
            patch("src.cli.load_config", return_value=_make_config()),
            patch("src.cli.load_corpus", return_value=[MagicMock()]) as mock_corpus,
            patch("src.cli.chunk_documents", return_value=sentinel_chunks) as mock_chunk,
            patch("src.cli.Retriever", return_value=mock_retriever) as mock_retriever_cls,
        ):
            result = runner.invoke(cli, ["index"])

        assert result.exit_code == 0, result.output
        assert "FAISS index saved" in result.output
        mock_corpus.assert_called_once()
        mock_chunk.assert_called_once()
        mock_retriever_cls.assert_called_once()
        mock_retriever.build.assert_called_once_with(sentinel_chunks)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQueryCommand:
    def test_styled_response(self, runner: CliRunner) -> None:
        mock_flow = MagicMock()
        mock_flow.state.styled_response = _make_styled_response()
        mock_flow.state.fallback_response = None

        with patch("src.cli.DigitalCloneFlow", return_value=mock_flow):
            result = runner.invoke(cli, ["query", "What is TCP?", "--leader", "torvalds"])

        assert result.exit_code == 0, result.output
        assert "Linus Torvalds" in result.output
        assert "0.80" in result.output          # style_score
        assert "0.90" in result.output          # groundedness_score
        assert "HHEM" in result.output          # metric label
        mock_flow.kickoff.assert_called_once_with(
            inputs={"query": "What is TCP?", "leader": "Linus Torvalds"}
        )

    def test_fallback_response(self, runner: CliRunner) -> None:
        mock_flow = MagicMock()
        mock_flow.state.styled_response = None
        mock_flow.state.fallback_response = _make_fallback_response()

        with patch("src.cli.DigitalCloneFlow", return_value=mock_flow):
            result = runner.invoke(
                cli, ["query", "Hard question", "--leader", "kroah-hartman"]
            )

        assert result.exit_code == 0, result.output
        assert "FALLBACK" in result.output
        assert "outside my grounded knowledge" in result.output

    def test_missing_leader_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["query", "What is TCP?"])
        assert result.exit_code == 2

    def test_invalid_leader_choice(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["query", "What is TCP?", "--leader", "gates"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareCommand:
    def test_success(self, runner: CliRunner) -> None:
        comparison = MagicMock(spec=LeaderComparison)
        comparison.torvalds = _make_styled_response()
        comparison.kroah_hartman = _make_styled_response()

        with patch("src.cli._compare_leaders", return_value=comparison) as mock_cmp:
            result = runner.invoke(cli, ["compare", "What is TCP?"])

        assert result.exit_code == 0, result.output
        assert "Linus Torvalds" in result.output
        assert "Greg Kroah-Hartman" in result.output
        mock_cmp.assert_called_once_with("What is TCP?")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluateCommand:
    def test_success_writes_json(self, runner: CliRunner, tmp_path: Path) -> None:
        queries_file = tmp_path / "queries.json"
        queries_file.write_text(
            '[{"id": "q01", "query": "What is TCP?", "category": "statistical_learning_ml",'
            ' "expected_behavior": "deliver"}]'
        )
        out_dir = tmp_path / "results"

        with patch("src.cli._run_leader_pair", return_value=_make_pair_result()):
            result = runner.invoke(
                cli,
                ["evaluate", "--queries", str(queries_file), "--output-dir", str(out_dir)],
            )

        assert result.exit_code == 0, result.output
        assert "Results written" in result.output
        assert "2×2 deliver" in result.output  # 2×2 grid header

        json_files = list(out_dir.glob("evaluation_*.json"))
        assert len(json_files) == 1
        records = json.loads(json_files[0].read_text())
        assert len(records) == 1                              # one record per query pair
        assert records[0]["query_id"] == "q01"
        assert records[0]["torvalds"]["decision"] == "deliver"
        assert records[0]["kroah_hartman"]["decision"] == "deliver"
        assert "groundedness_score" in records[0]["torvalds"]
        assert "retriever_call_count" in records[0]

    def test_fallback_recorded(self, runner: CliRunner, tmp_path: Path) -> None:
        queries_file = tmp_path / "queries.json"
        queries_file.write_text(
            '[{"id": "q01", "query": "Hard question", "category": "off_topic_technical",'
            ' "expected_behavior": "fallback"}]'
        )
        out_dir = tmp_path / "results"

        with patch(
            "src.cli._run_leader_pair",
            return_value=_make_pair_result(t_decision="fallback", kh_decision="fallback"),
        ):
            result = runner.invoke(
                cli,
                ["evaluate", "--queries", str(queries_file), "--output-dir", str(out_dir)],
            )

        assert result.exit_code == 0, result.output
        records = json.loads(list(out_dir.glob("evaluation_*.json"))[0].read_text())
        assert records[0]["torvalds"]["decision"] == "fallback"
        assert records[0]["kroah_hartman"]["decision"] == "fallback"

    def test_default_queries_path_is_v2(self) -> None:
        """Default --queries points to data/eval/queries.json, not queries_v1.json."""
        from src.cli import evaluate as evaluate_cmd
        defaults = {p.name: p.default for p in evaluate_cmd.params}
        assert "queries_v1" not in str(defaults["queries"])
        assert "queries.json" in str(defaults["queries"])

    def test_missing_queries_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["evaluate", "--queries", "/nonexistent/queries.json"])
        assert result.exit_code == 2
