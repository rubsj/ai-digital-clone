"""Experiment 6c — Scoring weight sensitivity (3 configs × 10 queries).

Design: RETRIEVE ONCE, RESCORE THREE TIMES.
  - Retrieve and Cohere-rerank ONCE per query (10 Cohere calls + 1 preflight = 11 total).
  - Compute (style, groundedness, confidence) component scores once per query.
  - Apply three weight configurations to the same component scores.
  - No additional Cohere or LLM calls during the weight-sensitivity sweep.

Optimal-config criterion (stated before running, not after):
  "Optimal" = the config whose mean final score across the 10 queries is highest
  AND whose per-query fallback rate is closest to the PRD §2d target of 30-40%.
  If those two criteria disagree, both are reported and the iteration-log entry
  describes the tradeoff; no automatic winner is declared.

Component score proxies (same as Experiments 6a/6b):
  - groundedness: score_groundedness(query, top5) — query as response proxy.
    Uses OpenAI embed; MD5-cached, so chunk embeddings are cache-hits.
  - confidence: score_confidence(query, query, top5) — query as response proxy.
    As documented in the iteration log, completeness=1.0 and uncertainty=1.0 for
    all queries; only retrieval_relevance (mean Cohere score) varies.
  - style: extract_features on a minimal EmailMessage wrapping the query text,
    then score_style(torvalds_profile, query_features). Queries are not
    Torvalds-style emails so style scores will be uniformly low (~0.1-0.3).
    Limitation acknowledged; consistent across all three configs so the
    weight-sensitivity signal is still interpretable.

ADR-006 trigger check (per day6-plan.md §Phase 7):
  Flag if optimal ≠ default AND (delta mean_final > 0.05 OR fallback rate
  moves outside the 30-40% band in a way default does not).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("CO_API_KEY") and os.environ.get("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]

import cohere

from src.eval.query_loader import load_queries
from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import score_groundedness
from src.rag.indexer import load_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve
from src.schemas import EmailMessage
from src.style.feature_extractor import extract_features
from src.style.profile_builder import load_profile
from src.style.style_scorer import score_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERIES_PATH = Path("data/eval/queries_v1.json")
INDEX_DIR = Path("data/rag/faiss_index")
PROFILE_PATH = Path("data/models/torvalds_profile.json")
CHART_PATH = Path("docs/images/6c-weight-sensitivity.png")

RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5
THRESHOLD = 0.75
TARGET_FALLBACK_LOW = 0.30
TARGET_FALLBACK_HIGH = 0.40
COHERE_INTER_QUERY_SLEEP = 10.0
COHERE_POST_PREFLIGHT_SLEEP = 10.0

WEIGHT_CONFIGS: dict[str, tuple[float, float, float]] = {
    "default":      (0.4, 0.4, 0.2),  # style, groundedness, confidence
    "style_heavy":  (0.5, 0.3, 0.2),
    "ground_heavy": (0.3, 0.5, 0.2),
}

sep = "=" * 90
sep2 = "-" * 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_as_email(qid: str, query: str) -> EmailMessage:
    """Wrap a query string in a minimal EmailMessage for style feature extraction."""
    return EmailMessage(
        sender="query-proxy",
        recipients=[],
        subject="",
        body=query,
        timestamp=datetime.now(tz=timezone.utc),
        message_id=qid,
        is_patch=False,
        quote_ratio=0.0,
    )


def _fallback_rate(finals: list[float]) -> float:
    """Fraction of queries below the 0.75 threshold."""
    return sum(1 for f in finals if f < THRESHOLD) / len(finals)


def _in_target_band(rate: float) -> bool:
    return TARGET_FALLBACK_LOW <= rate <= TARGET_FALLBACK_HIGH


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _cohere_preflight() -> None:
    client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY", ""))
    resp = client.rerank(
        model="rerank-english-v3.0",
        query="health check",
        documents=["foo", "bar"],
        top_n=1,
    )
    if not resp.results:
        raise RuntimeError("Cohere preflight returned empty results — quota may be exhausted.")
    print("cohere quota pre-check OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(sep)
    print("Experiment 6c — Scoring weight sensitivity (3 configs × 10 queries)")
    print(sep)

    # --- Load assets ---
    print("\nLoading FAISS index, Torvalds profile, and queries...")
    index, metadata = load_index(INDEX_DIR)
    print(f"  Index: {index.ntotal} vectors from {INDEX_DIR}")

    profile = load_profile(PROFILE_PATH)
    print(f"  Profile: {profile.leader_name} ({profile.email_count} emails)")

    queries = load_queries(QUERIES_PATH)
    print(f"  Queries: {len(queries)} loaded from {QUERIES_PATH}")

    # --- Cohere preflight ---
    print()
    _cohere_preflight()
    print(f"Sleeping {COHERE_POST_PREFLIGHT_SLEEP}s after preflight to clear rate-limit window...")
    time.sleep(COHERE_POST_PREFLIGHT_SLEEP)

    # --- Per-query computation ---
    print(f"\n{'Query':<8} {'Style':>7} {'G(post)':>8} {'Conf':>7}", end="")
    for cfg_name in WEIGHT_CONFIGS:
        print(f" | {'Final[' + cfg_name + ']':>18}", end="")
    print()
    print(sep2)

    # Store per-query component scores and finals
    per_query: list[dict] = []

    for q_idx, qr in enumerate(queries):
        qid = qr["id"]
        query = qr["query"]

        # --- Retrieve + rerank (1 Cohere call) ---
        candidates = retrieve(query, index, metadata, top_n=RETRIEVAL_TOP_N, provider="openai")
        top5 = rerank(query, candidates, top_n=RERANK_TOP_N)

        # --- Component scores (once per query) ---
        fake_email = _query_as_email(qid, query)
        query_features = extract_features(fake_email)
        style = score_style(profile, query_features)

        groundedness = score_groundedness(query, top5)
        confidence = score_confidence(query, query, top5)

        # --- Apply three weight configs ---
        finals: dict[str, float] = {}
        for cfg_name, (w_s, w_g, w_c) in WEIGHT_CONFIGS.items():
            finals[cfg_name] = round(w_s * style + w_g * groundedness + w_c * confidence, 6)

        per_query.append({
            "id": qid,
            "style": style,
            "groundedness": groundedness,
            "confidence": confidence,
            "finals": finals,
        })

        print(f"{qid:<8} {style:>7.4f} {groundedness:>8.4f} {confidence:>7.4f}", end="")
        for cfg_name in WEIGHT_CONFIGS:
            f = finals[cfg_name]
            marker = "↑" if f >= THRESHOLD else "↓"
            print(f" | {f:>16.4f} {marker}", end="")
        print()

        if q_idx < len(queries) - 1:
            time.sleep(COHERE_INTER_QUERY_SLEEP)

    print(sep2)

    # --- Aggregate stats per config ---
    print("\n--- Aggregate (all 10 queries) ---")
    agg: dict[str, dict] = {}
    for cfg_name in WEIGHT_CONFIGS:
        finals_list = [row["finals"][cfg_name] for row in per_query]
        fr = _fallback_rate(finals_list)
        agg[cfg_name] = {
            "mean_final": float(np.mean(finals_list)),
            "std_final": float(np.std(finals_list)),
            "fallback_rate": fr,
            "in_band": _in_target_band(fr),
            "finals": finals_list,
        }
        band_marker = "✓" if agg[cfg_name]["in_band"] else "✗"
        print(
            f"  {cfg_name:<14} mean_final={agg[cfg_name]['mean_final']:.4f}±{agg[cfg_name]['std_final']:.4f}"
            f"  fallback={fr:.0%} {band_marker} ({'in' if agg[cfg_name]['in_band'] else 'outside'} 30-40% target)"
        )

    # --- Component score summary ---
    styles = [row["style"] for row in per_query]
    grounds = [row["groundedness"] for row in per_query]
    confs = [row["confidence"] for row in per_query]
    print(f"\n--- Component score summary ---")
    print(f"  style:        mean={np.mean(styles):.4f}  std={np.std(styles):.4f}  min={np.min(styles):.4f}  max={np.max(styles):.4f}")
    print(f"  groundedness: mean={np.mean(grounds):.4f}  std={np.std(grounds):.4f}  min={np.min(grounds):.4f}  max={np.max(grounds):.4f}")
    print(f"  confidence:   mean={np.mean(confs):.4f}  std={np.std(confs):.4f}  min={np.min(confs):.4f}  max={np.max(confs):.4f}")

    # --- Optimal config decision ---
    print(f"\n--- Optimal config decision (criterion: highest mean_final AND nearest 30-40% fallback band) ---")
    best_mean = max(WEIGHT_CONFIGS, key=lambda k: agg[k]["mean_final"])
    in_band_configs = [k for k in WEIGHT_CONFIGS if agg[k]["in_band"]]
    if in_band_configs:
        best_band = max(in_band_configs, key=lambda k: agg[k]["mean_final"])
    else:
        best_band = min(WEIGHT_CONFIGS, key=lambda k: abs(agg[k]["fallback_rate"] - 0.35))
        print(f"  NOTE: No config falls within the 30-40% fallback band.")
        print(f"  Nearest to band: {best_band} (fallback={agg[best_band]['fallback_rate']:.0%})")

    agree = best_mean == best_band
    print(f"  Highest mean_final:   {best_mean} ({agg[best_mean]['mean_final']:.4f})")
    print(f"  Best fallback-band:   {best_band} (fallback={agg[best_band]['fallback_rate']:.0%})")
    print(f"  Criteria agree:       {'YES → ' + best_mean + ' is optimal' if agree else 'NO → report both, data decides'}")

    # ADR-006 trigger check
    default_mean = agg["default"]["mean_final"]
    optimal_mean = agg[best_mean]["mean_final"]
    delta = optimal_mean - default_mean
    default_in_band = agg["default"]["in_band"]
    optimal_in_band = agg[best_mean]["in_band"]
    trigger = (abs(delta) > 0.05) or (not default_in_band and optimal_in_band) or (default_in_band and not optimal_in_band)
    print(f"\n--- ADR-006 trigger check ---")
    print(f"  Δ(optimal − default) mean_final: {delta:+.4f}  ({'> 0.05 → TRIGGER' if abs(delta) > 0.05 else '≤ 0.05'})")
    print(f"  Default in 30-40% band: {default_in_band}  |  Optimal in band: {optimal_in_band}")
    print(f"  ADR-006 TRIGGER: {'YES — weight config change is materially beneficial' if trigger else 'NO — default is already optimal or delta is noise-level'}")

    # --- Chart ---
    _save_chart(queries, per_query, agg)
    print(f"\nChart saved: {CHART_PATH}")
    print(sep)


def _save_chart(
    queries: list[dict],
    per_query: list[dict],
    agg: dict,
) -> None:
    qids = [row["id"] for row in per_query]
    n = len(qids)
    x = np.arange(n)

    colors = {"default": "#1f77b4", "style_heavy": "#ff7f0e", "ground_heavy": "#2ca02c"}
    markers = {"default": "o", "style_heavy": "s", "ground_heavy": "^"}

    fig, (ax_main, ax_comp) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle("Experiment 6c — Scoring Weight Sensitivity", fontsize=13, fontweight="bold")

    # Main panel: final scores per config
    for cfg_name in WEIGHT_CONFIGS:
        finals_list = agg[cfg_name]["finals"]
        fr = agg[cfg_name]["fallback_rate"]
        label = f"{cfg_name} (fallback={fr:.0%})"
        ax_main.plot(x, finals_list, marker=markers[cfg_name], color=colors[cfg_name], label=label, linewidth=1.5)

    ax_main.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.2, label=f"threshold {THRESHOLD}")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(qids, rotation=45, ha="right", fontsize=9)
    ax_main.set_ylabel("Final Score")
    ax_main.set_xlabel("Query")
    ax_main.set_title("Final score per query by weight config")
    ax_main.legend(fontsize=8)
    ax_main.set_ylim(0, 1.0)
    ax_main.grid(axis="y", alpha=0.3)

    # Component panel: style / groundedness / confidence
    grounds = [row["groundedness"] for row in per_query]
    styles_list = [row["style"] for row in per_query]
    confs = [row["confidence"] for row in per_query]

    ax_comp.plot(x, grounds, marker="o", color="#1f77b4", label=f"groundedness (mean={np.mean(grounds):.3f})", linewidth=1.5)
    ax_comp.plot(x, confs, marker="s", color="#ff7f0e", label=f"confidence (mean={np.mean(confs):.3f})", linewidth=1.5)
    ax_comp.plot(x, styles_list, marker="^", color="#2ca02c", label=f"style (mean={np.mean(styles_list):.3f})", linewidth=1.5)
    ax_comp.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.0, label=f"threshold {THRESHOLD}")
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(qids, rotation=45, ha="right", fontsize=9)
    ax_comp.set_ylabel("Component Score")
    ax_comp.set_xlabel("Query")
    ax_comp.set_title("Component scores driving the weight sensitivity")
    ax_comp.legend(fontsize=8)
    ax_comp.set_ylim(0, 1.0)
    ax_comp.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
