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
