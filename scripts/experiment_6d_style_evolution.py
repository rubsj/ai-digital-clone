"""Experiment 6d — Pre/post-2018 Torvalds style evolution.

Partition: emails with timestamp < 2018-09-01 (pre) vs >= 2018-09-01 (post).
  Pre:  emails from 2015-01-01 to 2018-08-31
  Post: emails from 2018-09-01 to 2023-12-31

Four tracked features:
  1. sentiment         — dict_mean(sentiment_distribution), [0, 1]
  2. capitalization    — capitalization_ratio, [0, 1]
  3. exclamations      — punctuation_patterns["exclamation"], [0, 1]
  4. formality         — formality_level, [0, 1]

These are taken directly from extract_features() without modification to
feature_extractor.py or profile_builder.py (constraint per day6-plan.md §Phase 5).

Significance criterion (stated before measuring):
  A per-feature delta is a measurable shift only if:
    |pre_mean - post_mean| > 2 × std(feature on larger partition)
  where std is computed on the POST partition (6,661 emails — larger of the two).
  Anything below this threshold is reported as "within noise" in the chart
  legend and iteration-log Delta field.

Chart: 2×2 grid of monthly-bucketed time-series panels, one per feature.
  Each panel: x-axis = year-month, y-axis = feature mean per month,
  vertical dashed line at 2018-09-01, horizontal lines for pre/post partition means.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.style.email_parser import parse_mbox
from src.style.feature_extractor import extract_features
from src.schemas import StyleFeatures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MBOX_PATH = Path("data/emails/torvalds.mbox")
CHART_PATH = Path("docs/images/6d-style-evolution.png")  # output moved to results/charts/07-style-evolution.png (Day 7 gallery split)

CUTOFF = datetime(2018, 9, 1, tzinfo=timezone.utc)
SENDER_FILTER = "torvalds@"
MIN_PARTITION_SIZE = 30

FEATURES = ["sentiment", "capitalization", "exclamations", "formality"]
FEATURE_LABELS = {
    "sentiment": "Sentiment (positive/negative word rate)",
    "capitalization": "Capitalization ratio (ALL-CAPS words)",
    "exclamations": "Exclamation frequency",
    "formality": "Formality level",
}

sep = "=" * 90
sep2 = "-" * 80


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _tz_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _extract_four(features: StyleFeatures) -> dict[str, float]:
    """Pull the four tracked features from a StyleFeatures object."""
    sentiment_dict = features.sentiment_distribution
    sentiment = float(np.mean(list(sentiment_dict.values()))) if sentiment_dict else 0.0
    excl = features.punctuation_patterns.get("exclamation", 0.0)
    return {
        "sentiment": sentiment,
        "capitalization": features.capitalization_ratio,
        "exclamations": excl,
        "formality": features.formality_level,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(sep)
    print("Experiment 6d — Pre/post-2018 Torvalds style evolution")
    print(sep)

    # --- Parse mbox ---
    print(f"\nParsing {MBOX_PATH} ...")
    emails = parse_mbox(MBOX_PATH, SENDER_FILTER)
    print(f"  {len(emails)} clean emails loaded")

    # --- Partition ---
    pre_emails = [e for e in emails if _tz_aware(e.timestamp) < CUTOFF]
    post_emails = [e for e in emails if _tz_aware(e.timestamp) >= CUTOFF]

    print(f"\nPartition at 2018-09-01:")
    print(f"  Pre:  {len(pre_emails)} emails ({_tz_aware(min(e.timestamp for e in pre_emails)).date()} — {_tz_aware(max(e.timestamp for e in pre_emails)).date()})")
    print(f"  Post: {len(post_emails)} emails ({_tz_aware(min(e.timestamp for e in post_emails)).date()} — {_tz_aware(max(e.timestamp for e in post_emails)).date()})")

    larger_label = "post" if len(post_emails) >= len(pre_emails) else "pre"
    larger_emails = post_emails if larger_label == "post" else pre_emails

    if len(pre_emails) < MIN_PARTITION_SIZE or len(post_emails) < MIN_PARTITION_SIZE:
        print(f"\nWARNING: One partition has < {MIN_PARTITION_SIZE} emails. "
              f"Pre={len(pre_emails)}, Post={len(post_emails)}. "
              "Consider year-bucketing fallback.")

    # --- Extract features for all emails ---
    print(f"\nExtracting features from {len(emails)} emails (no API calls) ...")
    all_data: list[tuple[datetime, dict[str, float]]] = []
    for i, email in enumerate(emails):
        feats = extract_features(email)
        vals = _extract_four(feats)
        all_data.append((_tz_aware(email.timestamp), vals))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(emails)}")
    print(f"  Done. {len(all_data)} feature rows computed.")

    # --- Per-partition stats ---
    pre_data = [vals for ts, vals in all_data if ts < CUTOFF]
    post_data = [vals for ts, vals in all_data if ts >= CUTOFF]

    pre_means: dict[str, float] = {}
    post_means: dict[str, float] = {}
    larger_stds: dict[str, float] = {}

    std_col_header = f"Std ({larger_label})"
    print(f"\n{'Feature':<16} {'Pre mean':>10} {'Post mean':>10} {'Δ(post−pre)':>12} "
          f"{std_col_header:>16} {'2σ threshold':>13} {'Verdict':>14}")
    print(sep2)

    for feat in FEATURES:
        pre_vals = [d[feat] for d in pre_data]
        post_vals = [d[feat] for d in post_data]
        larger_vals = post_vals if larger_label == "post" else pre_vals

        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        larger_std = float(np.std(larger_vals))
        threshold_2sigma = 2.0 * larger_std
        delta = post_mean - pre_mean
        significant = abs(delta) > threshold_2sigma
        verdict = "MEASURABLE SHIFT" if significant else "within noise"

        pre_means[feat] = pre_mean
        post_means[feat] = post_mean
        larger_stds[feat] = larger_std

        print(f"{feat:<16} {pre_mean:>10.5f} {post_mean:>10.5f} {delta:>+12.5f} "
              f"{larger_std:>14.5f} {threshold_2sigma:>13.5f}  {verdict}")

    print(sep2)

    # --- Monthly bucketing for chart ---
    monthly: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ts, vals in all_data:
        key = f"{ts.year}-{ts.month:02d}"
        for feat in FEATURES:
            monthly[key][feat].append(vals[feat])

    sorted_months = sorted(monthly.keys())
    month_means: dict[str, list[float]] = {feat: [] for feat in FEATURES}
    month_dates: list[datetime] = []
    for month_key in sorted_months:
        year, mon = int(month_key[:4]), int(month_key[5:])
        month_dates.append(datetime(year, mon, 1, tzinfo=timezone.utc))
        for feat in FEATURES:
            vals_list = monthly[month_key][feat]
            month_means[feat].append(float(np.mean(vals_list)) if vals_list else 0.0)

    # --- Chart ---
    _save_chart(month_dates, month_means, pre_means, post_means, larger_stds, larger_label)
    print(f"\nChart saved: {CHART_PATH}")

    # --- PRD exit criterion ---
    significant_features = [
        f for f in FEATURES if abs(post_means[f] - pre_means[f]) > 2.0 * larger_stds[f]
    ]
    print(f"\n--- PRD §8 exit criterion ---")
    print(f"  Features with measurable shift (|Δ| > 2σ): {significant_features if significant_features else 'NONE'}")
    if significant_features:
        print(f"  EXIT CRITERION MET: style evolution chart shows measurable shift on: {', '.join(significant_features)}")
    else:
        print(f"  EXIT CRITERION STATUS: No feature cleared the 2σ threshold.")
        print(f"  Iteration-log Keep? field will read: n/a — no measurable shift detected at the 2σ threshold")
    print(sep)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_chart(
    month_dates: list[datetime],
    month_means: dict[str, list[float]],
    pre_means: dict[str, float],
    post_means: dict[str, float],
    larger_stds: dict[str, float],
    larger_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle(
        "Experiment 6d — Torvalds Style Evolution (Pre/Post 2018-09-01)",
        fontsize=13,
        fontweight="bold",
    )

    colors = {
        "sentiment": "#1f77b4",
        "capitalization": "#ff7f0e",
        "exclamations": "#2ca02c",
        "formality": "#d62728",
    }

    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, FEATURES):
        x = month_dates
        y = month_means[feat]

        ax.plot(x, y, color=colors[feat], linewidth=1.0, alpha=0.6, label="Monthly mean")

        # Smoothed trend (12-month rolling mean)
        if len(y) >= 12:
            window = 12
            smoothed = np.convolve(y, np.ones(window) / window, mode="same")
            ax.plot(x, smoothed, color=colors[feat], linewidth=2.0, label="12-month rolling mean")

        # Pre/post partition means
        ax.axhline(pre_means[feat], color="steelblue", linestyle="--", linewidth=1.0,
                   label=f"Pre mean={pre_means[feat]:.4f}")
        ax.axhline(post_means[feat], color="darkorange", linestyle="--", linewidth=1.0,
                   label=f"Post mean={post_means[feat]:.4f}")

        # 2σ significance band around pre mean
        sigma2 = 2.0 * larger_stds[feat]
        ax.axhspan(pre_means[feat] - sigma2, pre_means[feat] + sigma2,
                   alpha=0.07, color="steelblue", label=f"±2σ ({larger_label} std={larger_stds[feat]:.4f})")

        # Partition boundary
        cutoff_dt = datetime(2018, 9, 1, tzinfo=timezone.utc)
        ax.axvline(cutoff_dt, color="black", linestyle=":", linewidth=1.5, label="2018-09-01")

        # Significance verdict
        delta = post_means[feat] - pre_means[feat]
        verdict = "SHIFT" if abs(delta) > sigma2 else "noise"
        ax.set_title(f"{FEATURE_LABELS[feat]}\nΔ={delta:+.4f} ({verdict})", fontsize=9)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel(feat.capitalize(), fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
