"""Visualization utilities for P6 Torvalds Digital Clone.

Implemented charts per phase:
  Day 2 — plot_style_radar: style profile comparison radar chart
  Day 7 — style histogram, groundedness histogram, score breakdown,
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
    ax.set_xlabel("Style Score")
    ax.set_ylabel("Count")
    ax.set_title("Style Score Distribution")
    ax.set_xlim(0, 1)
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
    ax.axvline(0.40, color="red", linestyle="--", linewidth=1.2, label="GROUNDEDNESS_MIN 0.40 (HHEM)")
    ax.set_xlabel("Groundedness Score (HHEM entailment)")
    ax.set_ylabel("Count")
    ax.set_title("Groundedness Score Distribution (HHEM entailment)")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_breakdown(records: list[dict], output_path: Path) -> None:
    """Grouped bar chart: mean style / groundedness / confidence per leader."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    leaders = ["torvalds", "kroah_hartman"]
    labels = ["Linus Torvalds", "Greg Kroah-Hartman"]
    score_keys = ["style_score", "groundedness_score", "confidence_score"]
    score_labels = ["Style", "Groundedness", "Confidence"]

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


# ---------------------------------------------------------------------------
# Day 15 P3a — functions for the P2 nested-record schema
# Pair records have shape: {pass, query_id, axis, expected_behavior,
#   torvalds: {decision, groundedness_score, style_score, ...timings, chunk_contents},
#   kroah_hartman: {...}}
# ---------------------------------------------------------------------------

_LEADER_KEYS = ("torvalds", "kroah_hartman")
_LEADER_LABELS = {"torvalds": "Linus Torvalds", "kroah_hartman": "Greg Kroah-Hartman"}


def _iter_leader_records(pair_records: list[dict], passes: set | None = None) -> list[dict]:
    """Flatten nested pair records to per-leader dicts, optionally filtered by pass."""
    out = []
    for r in pair_records:
        if passes is not None and r.get("pass") not in passes:
            continue
        for lk in _LEADER_KEYS:
            lr = r[lk]
            out.append({
                "leader_key": lk,
                "leader": lr["leader"],
                "pass": r["pass"],
                "query_id": r["query_id"],
                "axis": r["axis"],
                "expected_behavior": r["expected_behavior"],
                "decision": lr["decision"],
                "fallback": lr["decision"] == "fallback",
                "groundedness_score": lr.get("groundedness_score"),
                "style_score": lr.get("style_score"),
                "confidence_score": lr.get("confidence_score"),
                "trigger_reason": lr.get("trigger_reason"),
                "timings": lr.get("timings", {}),
                "chunk_contents": lr.get("chunk_contents", []),
            })
    return out


def plot_routing_correctness_grid(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Per-query × leader routing correctness heatmap (pass 1 only).

    Rows: q01–q20 (in-domain then OOD). Columns: Torvalds, KH.
    Green = correct routing, red = incorrect.
    OOD section separated by a horizontal rule.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pass1 = [r for r in pair_records if r.get("pass") == 1]
    pass1_by_qid = {r["query_id"]: r for r in pass1}
    query_ids = sorted(pass1_by_qid.keys(), key=lambda q: int(q[1:]))

    # Build correctness matrix: 1 = correct, 0 = incorrect
    n_q = len(query_ids)
    matrix = np.zeros((n_q, 2), dtype=float)
    gs_matrix: list[list[str]] = [["", ""] for _ in range(n_q)]
    axes_list = []

    for row_i, qid in enumerate(query_ids):
        r = pass1_by_qid[qid]
        axes_list.append(r["axis"])
        for col_j, lk in enumerate(_LEADER_KEYS):
            expected = r["expected_behavior"]
            actual = r[lk]["decision"]
            gs = r[lk].get("groundedness_score") or 0.0
            matrix[row_i, col_j] = 1.0 if expected == actual else 0.0
            gs_matrix[row_i][col_j] = f"{gs:.2f}"

    fig, ax = plt.subplots(figsize=(5, 9))
    cmap = plt.cm.RdYlGn
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Torvalds", "KH"], fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(query_ids, fontsize=9)

    # Annotate cells with gs score; mark off-expectation cells
    for row_i in range(n_q):
        for col_j in range(2):
            correct = matrix[row_i, col_j] == 1.0
            qid = query_ids[row_i]
            lk = _LEADER_KEYS[col_j]
            r = pass1_by_qid[qid]
            decision = r[lk]["decision"]
            gs = r[lk].get("groundedness_score") or 0.0
            text = f"{decision[0].upper()}\n{gs:.2f}"
            color = "white" if not correct else "black"
            ax.text(col_j, row_i, text, ha="center", va="center", fontsize=7, color=color)

    # Horizontal rule between in-domain and OOD
    n_in_domain = sum(1 for a in axes_list if a == "in_domain")
    ax.axhline(n_in_domain - 0.5, color="black", linewidth=2.0)

    # Section labels
    ax.text(-0.7, n_in_domain / 2 - 0.5, "In-domain", fontsize=9, va="center",
            color="#1a5276", rotation=90, fontweight="bold")
    n_ood = sum(1 for a in axes_list if a == "ood")
    ax.text(-0.7, n_in_domain + n_ood / 2 - 0.5, "OOD", fontsize=9, va="center",
            color="#922b21", rotation=90, fontweight="bold")

    total = n_q * 2
    n_correct = int(matrix.sum())
    ax.set_title(
        f"Routing Correctness — Pass 1 ({n_correct}/{total} = {n_correct/total:.0%})\n"
        "Green=correct, Red=incorrect. G/F = deliver/fallback decision.",
        fontsize=10,
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_style_score_distribution_per_leader(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Overlaid histograms of style_score per leader (in-domain deliver records, all passes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = _iter_leader_records(pair_records)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lk in enumerate(_LEADER_KEYS):
        scores = [r["style_score"] for r in flat
                  if r["leader_key"] == lk and r["axis"] == "in_domain"
                  and r["decision"] == "deliver" and r["style_score"] is not None]
        if scores:
            ax.hist(scores, bins=15, range=(0, 1), color=_COLORS[i], edgecolor="white",
                    alpha=0.65, label=_LEADER_LABELS[lk])
    ax.set_xlabel("Style Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Style Score Distribution per Leader\n(in-domain deliver records, all passes)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_groundedness_from_eval(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Histogram of groundedness score (HHEM entailment) across all records.

    Uses 25 bins over the actual data range; marks the 0.40 gate.
    Separate colors for deliver vs fallback records.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = _iter_leader_records(pair_records)
    deliver_gs = [r["groundedness_score"] for r in flat
                  if r["decision"] == "deliver" and r["groundedness_score"] is not None]
    fallback_gs = [r["groundedness_score"] for r in flat
                   if r["decision"] == "fallback" and r["groundedness_score"] is not None]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 25
    rng = (0.0, 1.0)
    if deliver_gs:
        ax.hist(deliver_gs, bins=bins, range=rng, color=_COLORS[0], edgecolor="white",
                alpha=0.75, label="Deliver")
    if fallback_gs:
        ax.hist(fallback_gs, bins=bins, range=rng, color=_COLORS[1], edgecolor="white",
                alpha=0.75, label="Fallback")
    ax.axvline(0.40, color="red", linestyle="--", linewidth=1.5,
               label="GROUNDEDNESS_MIN 0.40 (HHEM gate)")
    ax.set_xlabel("Groundedness Score (HHEM entailment)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Groundedness Score Distribution (HHEM entailment)\n"
        "All records, all passes. Red dashed = 0.40 routing gate.",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_deliver_rate_distribution(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Per-leader in-domain deliver rate across passes 1/2/3.

    Grouped bars (one group per pass), with ADR-015 floors and PRD §2.1
    E1/E2 reference lines. Honesty: OOD fallback is 91.7%, not annotated here
    (see routing grid for OOD outcome).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pass_nums = [1, 2, 3]
    rates: dict[str, list[float]] = {lk: [] for lk in _LEADER_KEYS}

    for p in pass_nums:
        p_records = [r for r in pair_records if r.get("pass") == p]
        in_domain = [r for r in p_records if r.get("axis") == "in_domain"]
        n = len(in_domain)
        for lk in _LEADER_KEYS:
            n_deliver = sum(1 for r in in_domain if r[lk]["decision"] == "deliver")
            rates[lk].append(n_deliver / n if n else 0.0)

    x = np.arange(len(pass_nums))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))

    bars_t = ax.bar(x - width / 2, rates["torvalds"], width,
                    label="Linus Torvalds", color=_COLORS[0], alpha=0.85)
    bars_kh = ax.bar(x + width / 2, rates["kroah_hartman"], width,
                     label="Greg Kroah-Hartman", color=_COLORS[1], alpha=0.85)

    # Annotate bars
    for bar in list(bars_t) + list(bars_kh):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.0%}", ha="center", va="bottom", fontsize=9)

    # Reference lines
    ref_lines = [
        (0.55, "E2 ≥55% (PRD §2.1)", "#0d6efd", "--"),
        (0.39, "E1 ≥39% (PRD §2.1)", "#6f42c1", "--"),
        (0.429, "Torvalds floor 42.9% (ADR-015)", _COLORS[0], ":"),
        (0.357, "KH floor 35.7% (ADR-015)", _COLORS[1], ":"),
    ]
    for y_val, label, color, ls in ref_lines:
        ax.axhline(y_val, color=color, linestyle=ls, linewidth=1.2, alpha=0.8, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Pass {p}" for p in pass_nums], fontsize=11)
    ax.set_ylabel("In-domain Deliver Rate", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "In-domain Deliver Rate per Leader × Pass\n"
        "Reference lines: PRD §2.1 (E1/E2) and ADR-015 floors",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fallback_trigger_distribution(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Bar chart of fallback trigger categories across all records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = _iter_leader_records(pair_records)
    n_deliver = sum(1 for r in flat if r["decision"] == "deliver")
    n_low_gs = sum(
        1 for r in flat
        if r["trigger_reason"] and r["trigger_reason"].startswith("low_groundedness")
    )
    n_other = sum(
        1 for r in flat
        if r["trigger_reason"] and not r["trigger_reason"].startswith("low_groundedness")
    )

    labels = ["Delivered\n(no trigger)", "Low groundedness\n(HHEM < 0.40)"]
    counts = [n_deliver, n_low_gs]
    colors = [_COLORS[0], _COLORS[1]]
    if n_other:
        labels.append("Other trigger")
        counts.append(n_other)
        colors.append(_COLORS[2])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Record Count", fontsize=11)
    ax.set_title(
        "Fallback Trigger Distribution\n(all records, all passes, both leaders)",
        fontsize=11,
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latency_by_path(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Histogram of end-to-end latency (s) per leader, deliver vs fallback.

    Latency computed from stage timings: clone_ms + evaluate_ms + route_ms
    + deliver_ms (deliver path) or fallback_ms (fallback path).
    Torvalds records include retrieve_ms; KH uses shared retrieval (0ms).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = _iter_leader_records(pair_records)
    deliver_lat = []
    fallback_lat = []

    for r in flat:
        t = r["timings"]
        stage_ms = (
            t.get("clone_ms", 0)
            + t.get("evaluate_ms", 0)
            + t.get("route_ms", 0)
            + t.get("deliver_ms", 0)
            + t.get("fallback_ms", 0)
        )
        lat_s = stage_ms / 1000.0
        if r["decision"] == "deliver":
            deliver_lat.append(lat_s)
        else:
            fallback_lat.append(lat_s)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 20
    if deliver_lat:
        ax.hist(deliver_lat, bins=bins, color=_COLORS[0], edgecolor="white",
                alpha=0.7, label=f"Deliver (n={len(deliver_lat)})")
    if fallback_lat:
        ax.hist(fallback_lat, bins=bins, color=_COLORS[1], edgecolor="white",
                alpha=0.7, label=f"Fallback (n={len(fallback_lat)})")
    ax.set_xlabel("End-to-end Latency (seconds)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Query Latency Distribution by Routing Path\n"
        "(clone + evaluate + route + deliver/fallback stage times)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_style_evolution(
    mbox_path: Path | str,
    output_path: Path | str,
) -> None:
    """Monthly time-series of 4 Torvalds style features with 2018-09-01 marker.

    Features: sentiment, capitalization, exclamation frequency, formality.
    Methodology from experiment_6d_style_evolution.py — same 4 features,
    same significance criterion (|Δ| > 2σ of larger partition).
    Null result expected: no feature cleared the 2σ threshold in Day 6.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    from datetime import timezone
    from src.style.email_parser import parse_mbox
    from src.style.feature_extractor import extract_features

    mbox_path = Path(mbox_path)
    cutoff_dt = _EVOLUTION_CUTOFF

    emails = parse_mbox(mbox_path, "torvalds@")

    all_data: list[tuple] = []
    for email in emails:
        ts = email.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        feats = extract_features(email)
        sentiment_dict = feats.sentiment_distribution
        sentiment = float(np.mean(list(sentiment_dict.values()))) if sentiment_dict else 0.0
        excl = feats.punctuation_patterns.get("exclamation", 0.0)
        vals = {
            "sentiment": sentiment,
            "capitalization": feats.capitalization_ratio,
            "exclamations": excl,
            "formality": feats.formality_level,
        }
        all_data.append((ts, vals))

    features = ["sentiment", "capitalization", "exclamations", "formality"]
    feature_labels = {
        "sentiment": "Sentiment",
        "capitalization": "Capitalization ratio",
        "exclamations": "Exclamation frequency",
        "formality": "Formality level",
    }

    pre_data = [vals for ts, vals in all_data if ts < cutoff_dt]
    post_data = [vals for ts, vals in all_data if ts >= cutoff_dt]
    larger_data = post_data if len(post_data) >= len(pre_data) else pre_data

    pre_means = {f: float(np.mean([d[f] for d in pre_data])) for f in features}
    post_means = {f: float(np.mean([d[f] for d in post_data])) for f in features}
    larger_stds = {f: float(np.std([d[f] for d in larger_data])) for f in features}

    monthly: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ts, vals in all_data:
        key = f"{ts.year}-{ts.month:02d}"
        for feat in features:
            monthly[key][feat].append(vals[feat])

    sorted_months = sorted(monthly.keys())
    from datetime import datetime
    month_dates = [datetime(int(k[:4]), int(k[5:]), 1, tzinfo=timezone.utc) for k in sorted_months]
    month_means: dict[str, list[float]] = {f: [] for f in features}
    for mk in sorted_months:
        for feat in features:
            vs = monthly[mk][feat]
            month_means[feat].append(float(np.mean(vs)) if vs else 0.0)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Torvalds Style Evolution — Pre/Post 2018-09-01 (C6 measurement)",
        fontsize=12, fontweight="bold",
    )

    for ax, feat, color in zip(axes.flatten(), features, colors):
        y = month_means[feat]
        ax.plot(month_dates, y, color=color, linewidth=1.0, alpha=0.55, label="Monthly mean")
        if len(y) >= 12:
            # mode="valid" only returns points where the full 12-month window fits,
            # avoiding partial-window edge dives at both ends of the series.
            sm = np.convolve(y, np.ones(12) / 12, mode="valid")
            ax.plot(month_dates[11:], sm, color=color, linewidth=2.0, label="12-mo rolling mean")
        ax.axhline(pre_means[feat], color="steelblue", linestyle="--", linewidth=1.0,
                   label=f"Pre mean {pre_means[feat]:.4f}")
        ax.axhline(post_means[feat], color="darkorange", linestyle="--", linewidth=1.0,
                   label=f"Post mean {post_means[feat]:.4f}")
        sigma2 = 2.0 * larger_stds[feat]
        ax.axhspan(pre_means[feat] - sigma2, pre_means[feat] + sigma2,
                   alpha=0.07, color="steelblue")
        ax.axvline(cutoff_dt, color="black", linestyle=":", linewidth=1.5,
                   label="2018-09-01")
        delta = post_means[feat] - pre_means[feat]
        verdict = "SHIFT" if abs(delta) > sigma2 else "noise"
        ax.set_title(f"{feature_labels[feat]}\nΔ={delta:+.4f} ({verdict})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_relevance_contrast(
    pair_records: list[dict],
    output_path: Path | str,
) -> None:
    """Log-scale dot plot: top-chunk retrieval score for in-domain vs OOD queries.

    Uses pass 1 records. In-domain scores cluster 0.32–0.9999; OOD scores
    cluster 0.0–0.0013. q20/Torvalds annotated: it delivers despite top-chunk
    score 0.0013, while all in-domain top scores are ≥0.32.
    Key message: groundedness cannot distinguish q20 from in-domain, but
    retrieval relevance separates them by ~3 orders of magnitude.
    §7.6 addition — no slot in the 8-chart spec; flagged for PRD reconciliation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pass1 = [r for r in pair_records if r.get("pass") == 1]

    in_domain_points: list[tuple[str, float]] = []
    ood_points: list[tuple[str, float]] = []

    for r in pass1:
        chunks = r["torvalds"].get("chunk_contents", [])
        if chunks:
            top_score = max(c["score"] for c in chunks)
        else:
            top_score = 0.0
        qid = r["query_id"]
        if r["axis"] == "in_domain":
            in_domain_points.append((qid, top_score))
        else:
            ood_points.append((qid, top_score))

    fig, ax = plt.subplots(figsize=(8, 6))

    rng = np.random.default_rng(42)
    jitter_scale = 0.08

    in_domain_scores = [s for _, s in in_domain_points]
    ood_scores = [s for _, s in ood_points]
    jitter_id = rng.uniform(-jitter_scale, jitter_scale, len(in_domain_scores))
    jitter_ood = rng.uniform(-jitter_scale, jitter_scale, len(ood_scores))

    ax.scatter(0 + jitter_id, in_domain_scores, color=_COLORS[0], alpha=0.8,
               s=60, label=f"In-domain (n={len(in_domain_scores)}, median={np.median(in_domain_scores):.3f})",
               zorder=3)
    ax.scatter(1 + jitter_ood, ood_scores, color=_COLORS[1], alpha=0.8,
               s=60, label=f"OOD (n={len(ood_scores)}, median={np.median(ood_scores):.5f})",
               zorder=3)

    # Annotate q20 (the OOD deliver)
    for qid, score in ood_points:
        if qid == "q20":
            j = jitter_ood[next(i for i, (q, _) in enumerate(ood_points) if q == "q20")]
            ax.annotate(
                "q20: delivers\n(gs=0.422, top_chunk=0.0013)",
                xy=(1 + j, score),
                xytext=(1.25, score * 3),
                fontsize=8,
                color="#922b21",
                arrowprops=dict(arrowstyle="->", color="#922b21", lw=1.2),
            )

    ax.set_yscale("log")
    ax.set_xlim(-0.5, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["In-domain", "OOD"], fontsize=12)
    ax.set_ylabel("Top-chunk Retrieval Score (log scale)", fontsize=11)
    ax.set_title(
        "Retrieval Relevance: In-domain vs OOD (Pass 1, Torvalds)\n"
        "~3 orders of magnitude separation. q20 delivers despite OOD-level retrieval.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# Sentinel used by plot_style_evolution — defined at module level so tests can patch it
from datetime import datetime as _dt, timezone as _tz
_EVOLUTION_CUTOFF = _dt(2018, 9, 1, tzinfo=_tz.utc)
