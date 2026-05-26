"""Visualization utilities for P6 Torvalds Digital Clone.

Implemented charts per phase:
  Day 2 — plot_style_radar: style profile comparison radar chart
  Day 7 — style histogram, groundedness histogram, final score breakdown,
           fallback rate, latency distribution, style evolution
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import StyleProfile

# Matches StyleFeatures.to_vector() field order exactly (15 elements)
_RADAR_LABELS = [
    "Msg Length",
    "Greetings",
    "Punctuation",
    "Caps Ratio",
    "Question Freq",
    "Vocab Richness",
    "Reasoning",
    "Sentiment",
    "Formality",
    "Tech Terms",
    "Code Snippets",
    "Quote Reply",
    "Patch Lang",
    "Tech Depth",
    "Phrase Diversity",
]

# Blue for Torvalds, orange for Kroah-Hartman (+ fallback palette for extras)
_COLORS = ["#2563EB", "#EA580C", "#16A34A", "#9333EA"]


def plot_style_radar(
    profiles: list[StyleProfile],
    output_path: Path | str,
    *,
    dpi: int = 150,
) -> None:
    """Save a radar (spider) chart comparing style profiles to output_path.

    Each profile becomes one overlaid polygon on 15 axes (one per feature
    dimension). Uses matplotlib's polar projection; Agg backend so it works
    without a display.

    Args:
        profiles: List of StyleProfile objects to compare (typically 2).
        output_path: Destination path for the PNG. Parent dirs must exist.
        dpi: Output resolution (default 150).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    N = len(_RADAR_LABELS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, profile in enumerate(profiles):
        values = profile.style_vector.tolist()
        values_closed = values + values[:1]
        color = _COLORS[i % len(_COLORS)]
        ax.plot(angles_closed, values_closed, color=color, linewidth=2, label=profile.leader_name)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_LABELS, size=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="grey")
    ax.set_title("Style Profile Comparison", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_style_distribution(records: list[dict], output_path: Path) -> None:
    """Histogram of style_score across all non-fallback evaluation records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [r["style_score"] for r in records if not r.get("fallback", True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if scores:
        ax.hist(scores, bins=10, range=(0, 1), color=_COLORS[0], edgecolor="white", alpha=0.85)
    ax.axvline(0.75, color="red", linestyle="--", linewidth=1.2, label="threshold 0.75")
    ax.set_xlabel("Style Score")
    ax.set_ylabel("Count")
    ax.set_title("Style Score Distribution")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_groundedness_distribution(records: list[dict], output_path: Path) -> None:
    """Histogram of groundedness_score across all non-fallback evaluation records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [r["groundedness_score"] for r in records if not r.get("fallback", True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if scores:
        ax.hist(scores, bins=10, range=(0, 1), color=_COLORS[1], edgecolor="white", alpha=0.85)
    ax.axvline(0.60, color="red", linestyle="--", linewidth=1.2, label="target 0.60")
    ax.set_xlabel("Groundedness Score")
    ax.set_ylabel("Count")
    ax.set_title("Groundedness Score Distribution")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_breakdown(records: list[dict], output_path: Path) -> None:
    """Grouped bar chart: mean style / groundedness / confidence / final per leader."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    leaders = ["torvalds", "kroah_hartman"]
    labels = ["Linus Torvalds", "Greg Kroah-Hartman"]
    score_keys = ["style_score", "groundedness_score", "confidence_score", "final_score"]
    score_labels = ["Style", "Groundedness", "Confidence", "Final"]

    means: dict[str, list[float]] = {}
    for ldr in leaders:
        scored = [r for r in records if r.get("leader") == ldr and not r.get("fallback", True)]
        means[ldr] = [
            float(sum(r[k] for r in scored) / len(scored)) if scored else 0.0
            for k in score_keys
        ]

    x = np.arange(len(score_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, means[leaders[0]], width, label=labels[0], color=_COLORS[0], alpha=0.85)
    ax.bar(x + width / 2, means[leaders[1]], width, label=labels[1], color=_COLORS[1], alpha=0.85)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.2, label="threshold 0.75")
    ax.set_xticks(x)
    ax.set_xticklabels(score_labels)
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)
    ax.set_title("Score Breakdown by Leader")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fallback_rate(records: list[dict], output_path: Path) -> None:
    """Bar chart of fallback rate (fraction of queries that triggered fallback) per leader."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leaders = ["torvalds", "kroah_hartman"]
    labels = ["Linus Torvalds", "Greg Kroah-Hartman"]
    rates: list[float] = []
    for ldr in leaders:
        ldr_recs = [r for r in records if r.get("leader") == ldr]
        rate = sum(1 for r in ldr_recs if r.get("fallback", False)) / len(ldr_recs) if ldr_recs else 0.0
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rates, color=[_COLORS[0], _COLORS[1]], alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Fallback Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Fallback Rate by Leader")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latency_distribution(records: list[dict], output_path: Path) -> None:
    """Histogram of latency_ms across all evaluation records (fallback + scored)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
    fig, ax = plt.subplots(figsize=(8, 5))
    if latencies:
        ax.hist(latencies, bins=15, color=_COLORS[2], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Query Latency Distribution")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
