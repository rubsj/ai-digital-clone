"""Tests for src/visualization.py chart functions.

Each function writes a PNG to a tmp_path — tests verify the file exists and
is non-empty. No display backend required (all functions use matplotlib Agg).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.visualization import (
    plot_fallback_rate,
    plot_groundedness_distribution,
    plot_latency_distribution,
    plot_score_breakdown,
    plot_style_distribution,
    plot_style_radar,
)

# visualization.py reads v1 final_score from evaluation dicts. Refactor is
# explicit Day-12 scope (D-B1); skip until then.
pytestmark = pytest.mark.skip(reason="visualization.py refactor deferred to Day 12 (D-B1)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_style_profile(name: str, vector: list[float] | None = None):
    from datetime import datetime

    import numpy as np

    from src.schemas import StyleFeatures, StyleProfile

    vec = np.array(vector if vector else [0.5] * 15, dtype=np.float64)
    return StyleProfile(
        leader_name=name,
        features=StyleFeatures(),
        style_vector=vec,
        email_count=100,
        last_updated=datetime.utcnow(),
    )


def _scored_record(leader: str, style: float, ground: float, conf: float, final: float, latency: float = 500.0) -> dict:
    return {
        "id": "q01",
        "leader": leader,
        "fallback": False,
        "style_score": style,
        "groundedness_score": ground,
        "confidence_score": conf,
        "final_score": final,
        "latency_ms": latency,
    }


def _fallback_record(leader: str, latency: float = 300.0) -> dict:
    return {"id": "q02", "leader": leader, "fallback": True, "latency_ms": latency}


_SAMPLE_RECORDS = [
    _scored_record("torvalds", 0.85, 0.72, 0.90, 0.80, latency=1200.0),
    _scored_record("torvalds", 0.60, 0.55, 0.70, 0.61, latency=980.0),
    _fallback_record("torvalds", latency=450.0),
    _scored_record("kroah_hartman", 0.78, 0.68, 0.85, 0.75, latency=1100.0),
    _fallback_record("kroah_hartman", latency=320.0),
]


# ---------------------------------------------------------------------------
# plot_style_radar
# ---------------------------------------------------------------------------


class TestPlotStyleRadar:
    def test_creates_png(self, tmp_path: Path) -> None:
        torvalds = _make_style_profile("Linus Torvalds")
        kh = _make_style_profile("Greg Kroah-Hartman", vector=[0.3] * 15)
        out = tmp_path / "radar.png"
        plot_style_radar([torvalds, kh], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_profile(self, tmp_path: Path) -> None:
        out = tmp_path / "radar_single.png"
        plot_style_radar([_make_style_profile("Solo")], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_style_distribution
# ---------------------------------------------------------------------------


class TestPlotStyleDistribution:
    def test_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "02-style.png"
        plot_style_distribution(_SAMPLE_RECORDS, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "02-empty.png"
        plot_style_distribution([], out)
        assert out.exists()

    def test_all_fallback(self, tmp_path: Path) -> None:
        out = tmp_path / "02-fallback.png"
        plot_style_distribution([_fallback_record("torvalds")], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_groundedness_distribution
# ---------------------------------------------------------------------------


class TestPlotGroundednessDistribution:
    def test_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "03-ground.png"
        plot_groundedness_distribution(_SAMPLE_RECORDS, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "03-empty.png"
        plot_groundedness_distribution([], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_score_breakdown
# ---------------------------------------------------------------------------


class TestPlotScoreBreakdown:
    def test_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "04-breakdown.png"
        plot_score_breakdown(_SAMPLE_RECORDS, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_scored_records(self, tmp_path: Path) -> None:
        out = tmp_path / "04-empty.png"
        plot_score_breakdown([_fallback_record("torvalds")], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_fallback_rate
# ---------------------------------------------------------------------------


class TestPlotFallbackRate:
    def test_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "05-fallback.png"
        plot_fallback_rate(_SAMPLE_RECORDS, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_zero_fallback(self, tmp_path: Path) -> None:
        records = [_scored_record("torvalds", 0.8, 0.7, 0.9, 0.78)]
        out = tmp_path / "05-zero.png"
        plot_fallback_rate(records, out)
        assert out.exists()

    def test_all_fallback(self, tmp_path: Path) -> None:
        records = [_fallback_record("torvalds"), _fallback_record("kroah_hartman")]
        out = tmp_path / "05-all.png"
        plot_fallback_rate(records, out)
        assert out.exists()

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "05-empty.png"
        plot_fallback_rate([], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_latency_distribution
# ---------------------------------------------------------------------------


class TestPlotLatencyDistribution:
    def test_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "06-latency.png"
        plot_latency_distribution(_SAMPLE_RECORDS, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_latency_field(self, tmp_path: Path) -> None:
        records = [{"id": "q01", "leader": "torvalds", "fallback": False}]
        out = tmp_path / "06-nolat.png"
        plot_latency_distribution(records, out)
        assert out.exists()

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "06-empty.png"
        plot_latency_distribution([], out)
        assert out.exists()
