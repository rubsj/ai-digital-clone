#!/usr/bin/env python3
"""Build and save style profiles for both leaders from real mbox data.

Usage:
    uv run python scripts/build_profiles.py

Outputs:
    data/models/torvalds_profile.json
    data/models/kroah_hartman_profile.json
    results/charts/style_radar.png
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure project root is on the Python path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.style.email_parser import parse_mbox
from src.style.feature_extractor import extract_features
from src.style.profile_builder import build_profile_batch, save_profile
from src.style.style_scorer import cosine_similarity, score_style
from src.visualization import plot_style_radar

# Matches to_vector() field order exactly
_FEATURE_LABELS = [
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

console = Console()


def _print_variance_table(leader_name: str, vectors: list[np.ndarray]) -> None:
    """Rich table: mean/std/min/max per feature dimension across all emails."""
    arr = np.array(vectors)  # (n_emails, 15)
    table = Table(
        title=f"[bold]{leader_name}[/bold] — Feature Diagnostics ({len(vectors)} emails)"
    )
    table.add_column("Feature", style="cyan", min_width=16)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    for i, label in enumerate(_FEATURE_LABELS):
        col = arr[:, i]
        low_std = col.std() < 0.05
        row_style = "yellow" if low_std else ""
        table.add_row(
            label,
            f"{col.mean():.3f}",
            f"[yellow]{col.std():.3f}[/yellow]" if low_std else f"{col.std():.3f}",
            f"{col.min():.3f}",
            f"{col.max():.3f}",
            style=row_style,
        )
    console.print(table)
    console.print("  [dim]Yellow rows: std < 0.05 — low discriminative power[/dim]")


def _self_similarity_check(profile, feature_list, n_sample: int = 20) -> float:
    """Sample n_sample emails, compute score_style for each.

    Threshold 0.70: cosine similarity on raw (unweighted) feature vectors is
    naturally depressed by high-variance sparse features (greeting, sentiment)
    and by non-discriminative high-magnitude features (vocab richness, formality)
    that dominate the L2 norm but don't differentiate leaders. A real LKML
    corpus of diverse email lengths produces self-similarity in the 0.60-0.80
    range; 0.70 is the practical pass gate for this feature design.
    """
    sample = random.sample(feature_list, min(n_sample, len(feature_list)))
    scores = [score_style(profile, f) for f in sample]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    status = "[green]PASS ✓[/green]" if mean_score >= 0.70 else "[red]WARN ✗[/red]"
    console.print(
        f"  Self-similarity ({len(sample)} emails): "
        f"{mean_score:.4f} ± {std_score:.4f}  {status}"
    )
    if mean_score < 0.70:
        console.print(
            "  [yellow]  → Below 0.70 threshold. "
            "Check feature normalization or corpus size.[/yellow]"
        )
    return mean_score


def main() -> None:
    config = load_config()
    project_root = Path(__file__).parent.parent

    profiles: dict[str, object] = {}

    for key, leader_cfg in config.leaders.items():
        console.rule(f"[bold blue]{leader_cfg.name}")

        mbox_path = project_root / leader_cfg.mbox_path
        if not mbox_path.exists():
            console.print(f"[red]mbox not found: {mbox_path}[/red]")
            sys.exit(1)

        console.print(f"  Parsing {mbox_path.name} …")
        emails = parse_mbox(str(mbox_path), leader_cfg.email_filter)

        # Exclude patch emails — they share structure across leaders, collapsing profiles
        emails = [e for e in emails if not e.is_patch]
        console.print(f"  {len(emails)} non-patch emails after filtering")

        if len(emails) < 5:
            console.print(
                f"[red]Too few emails ({len(emails)}); "
                f"need at least 5 to build a meaningful profile.[/red]"
            )
            sys.exit(1)

        console.print("  Extracting features …")
        features = []
        skipped = 0
        for email in emails:
            try:
                features.append(extract_features(email))
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                console.print(f"  [dim]Skipped one email: {exc}[/dim]")
        console.print(f"  {len(features)} features extracted  (skipped: {skipped})")

        # Per-feature variance diagnostics
        vectors = [f.to_vector() for f in features]
        _print_variance_table(leader_cfg.name, vectors)

        # Build and save profile
        profile = build_profile_batch(
            leader_cfg.name, features, alpha=config.style.alpha
        )
        profile_path = project_root / leader_cfg.profile_path
        save_profile(profile, profile_path)
        console.print(f"  Saved  → {profile_path}")

        # Self-similarity check
        _self_similarity_check(profile, features)

        profiles[key] = profile

    # Cross-leader cosine similarity + per-feature delta table
    if "torvalds" in profiles and "kroah_hartman" in profiles:
        t_vec = profiles["torvalds"].style_vector
        g_vec = profiles["kroah_hartman"].style_vector
        cross = cosine_similarity(t_vec, g_vec)
        console.rule("[bold]Cross-Leader Similarity")
        console.print(f"  Torvalds ↔ Kroah-Hartman cosine: {cross:.4f}")
        if cross > 0.95:
            console.print(
                "  [yellow]Note: cosine similarity is dominated by high-magnitude "
                "non-discriminative features (Vocab Richness, Formality). "
                "See per-feature deltas below.[/yellow]"
            )

        # Per-feature absolute delta — shows where leaders actually differ
        delta_table = Table(title="Per-Feature Absolute Delta (Torvalds − Kroah-Hartman)")
        delta_table.add_column("Feature", style="cyan", min_width=16)
        delta_table.add_column("Torvalds", justify="right")
        delta_table.add_column("KH", justify="right")
        delta_table.add_column("|Delta|", justify="right")
        for i, label in enumerate(_FEATURE_LABELS):
            t_val = float(t_vec[i])
            g_val = float(g_vec[i])
            delta = abs(t_val - g_val)
            row_style = "green" if delta > 0.05 else ""
            delta_table.add_row(
                label,
                f"{t_val:.3f}",
                f"{g_val:.3f}",
                f"[green]{delta:.3f}[/green]" if delta > 0.05 else f"{delta:.3f}",
                style=row_style,
            )
        console.print(delta_table)
        console.print("  [dim]Green rows: |delta| > 0.05 — meaningful separation[/dim]")

    # Radar chart
    chart_path = project_root / "results" / "charts" / "style_radar.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plot_style_radar(list(profiles.values()), chart_path)
    console.print(f"\n  Radar chart saved → {chart_path}")


if __name__ == "__main__":
    main()
