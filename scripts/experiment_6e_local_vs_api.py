"""Experiment 6e — GPT-4o-mini vs local Ollama qwen3:8b for evaluation scoring.

Design: RETRIEVE ONCE, EXPLAIN TWICE.
  - Retrieve and Cohere-rerank ONCE per query (10 Cohere calls + 1 preflight = 11 total).
  - Compute (style, groundedness, confidence) component scores ONCE per query.
  - Apply the deterministic final-score formula once (same for both configs).
  - Generate the explanation string TWICE per query: once via GPT-4o-mini and once via
    Ollama qwen3:8b, both using instructor.from_litellm(litellm.completion).
  - Time each explanation call.

Pearson note (stated before running): final scores are computed from the same component
scores via the same deterministic formula, so Pearson(GPT-scores, Ollama-scores) = 1.0
trivially. The meaningful metrics are:
  (a) Explanation validity — structured-output success rate + 5-query spot check.
  (b) Latency per explanation call — GPT-4o-mini vs Ollama.

Decision outcomes (stated before running, not after):
  1. Quality parity   — all Ollama explanations valid AND qualitatively equivalent
                        on 5-query spot check → recommend Ollama for dev, GPT-4o-mini
                        for prod.
  2. Quality drift    — any Ollama structured-output failure OR hallucinated reasoning
                        on spot check → recommend GPT-4o-mini for both; document gap.
  3. Latency tradeoff — parity on quality but Ollama latency meaningfully different
                        → state tradeoff; recommend per-environment.

Constraint: src/evaluation/evaluator.py is NOT modified. The Ollama Instructor client
and the explanation helper live in this script only.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import pearsonr

load_dotenv()

if not os.environ.get("CO_API_KEY") and os.environ.get("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]

import cohere
import instructor
import litellm

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
CHART_PATH = Path("docs/images/6e-local-vs-api.png")  # output moved to docs/experiments/charts/6e-local-vs-api.png (Day 7 gallery split)

RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5
THRESHOLD = 0.75
FORMULA_WEIGHTS = (0.4, 0.4, 0.2)  # style, groundedness, confidence

GPT_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "ollama/qwen3:8b"
OLLAMA_DAEMON_URL = "http://localhost:11434"

COHERE_INTER_QUERY_SLEEP = 10.0
COHERE_POST_PREFLIGHT_SLEEP = 10.0

# Pearson ≥ 0.90 on final scores AND all Ollama explanations structurally valid.
# Final scores are deterministic from component scores, so Pearson = 1.0 trivially.
# Parity verdict is driven by explanation validity + spot check.
PARITY_PEARSON_THRESHOLD = 0.90
SPOT_CHECK_N = 5

sep = "=" * 90
sep2 = "-" * 80


# ---------------------------------------------------------------------------
# Explanation model (mirrors evaluator.py's _ExplanationModel)
# ---------------------------------------------------------------------------

class _ExplanationModel(BaseModel):
    explanation: str


# ---------------------------------------------------------------------------
# Script-level Instructor clients (NOT modifying evaluator.py)
# ---------------------------------------------------------------------------

_gpt_client = instructor.from_litellm(litellm.completion)
_ollama_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)


def _build_prompt(style: float, groundedness: float, confidence: float,
                  final: float, decision: str) -> str:
    """Mirrors evaluator.py:_build_explanation_prompt verbatim."""
    return (
        f"You are a quality reviewer for AI-generated leadership coaching responses.\n\n"
        f"Evaluation scores:\n"
        f"  Style score:        {style:.3f} (target > 0.90)\n"
        f"  Groundedness score: {groundedness:.3f} (target > 0.60)\n"
        f"  Confidence score:   {confidence:.3f} (target > 0.80)\n"
        f"  Final score:        {final:.3f} (threshold 0.75)\n"
        f"  Decision:           {decision}\n\n"
        f"Write ONE concise sentence (≤ 25 words) explaining the decision in plain English. "
        f"Focus on the weakest dimension if decision is 'fallback'. "
        f"Be direct; no hedging."
    )


def _explain(client: instructor.Instructor, model: str,
             style: float, groundedness: float, confidence: float,
             final: float, decision: str) -> tuple[str, float, bool]:
    """Call the explanation LLM. Returns (explanation, elapsed_s, success)."""
    prompt = _build_prompt(style, groundedness, confidence, final, decision)
    t0 = time.perf_counter()
    try:
        result: _ExplanationModel = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=_ExplanationModel,
            max_retries=2,
        )
        elapsed = time.perf_counter() - t0
        return result.explanation, elapsed, True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return f"[FAILED: {type(exc).__name__}: {exc}]", elapsed, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_as_email(qid: str, query: str) -> EmailMessage:
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


def _daemon_reachable() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_DAEMON_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _model_present(model_name: str) -> bool:
    """Check `ollama list` output for the bare model name."""
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        return model_name in out
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cohere preflight
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
        raise RuntimeError("Cohere preflight returned empty results.")
    print("cohere quota pre-check OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(sep)
    print("Experiment 6e — GPT-4o-mini vs local Ollama qwen3:8b for evaluation scoring")
    print(sep)

    # --- Script-level Ollama assertions (seatbelt, not airbag) ---
    print("\n[Pre-flight assertions]")
    if not _daemon_reachable():
        raise RuntimeError("Ollama daemon not reachable at localhost:11434 — run `ollama serve`.")
    print("  Ollama daemon: reachable")

    bare_model = OLLAMA_MODEL.replace("ollama/", "")
    if not _model_present(bare_model):
        raise RuntimeError(f"Model '{bare_model}' not found in `ollama list`. Run `ollama pull {bare_model}`.")
    print(f"  Ollama model:  {bare_model} present")

    # --- Load assets ---
    print(f"\nLoading FAISS index, Torvalds profile, and queries...")
    index, metadata = load_index(INDEX_DIR)
    print(f"  Index: {index.ntotal} vectors from {INDEX_DIR}")

    profile = load_profile(PROFILE_PATH)
    print(f"  Profile: {profile.leader_name} ({profile.email_count} emails)")

    queries = load_queries(QUERIES_PATH)
    print(f"  Queries: {len(queries)} loaded from {QUERIES_PATH}")

    # --- Cohere preflight ---
    print()
    _cohere_preflight()
    print(f"Sleeping {COHERE_POST_PREFLIGHT_SLEEP}s after preflight ...")
    time.sleep(COHERE_POST_PREFLIGHT_SLEEP)

    # --- Pearson note (stated before run) ---
    print(f"\n[Pearson note] Final scores are deterministic from component scores.")
    print(f"  Pearson(GPT-scores, Ollama-scores) will be 1.0 trivially — both use")
    print(f"  the same formula on the same components. Meaningful metrics: (a)")
    print(f"  explanation validity, (b) latency per explanation call.")

    # --- Per-query header ---
    print(f"\n{'Query':<8} {'Style':>6} {'G':>6} {'Conf':>6} {'Final':>6} {'Dec':<8}"
          f" {'GPT_ms':>7} {'OL_ms':>7} {'OL_ok':>6}")
    print(sep2)

    per_query: list[dict] = []

    for q_idx, qr in enumerate(queries):
        qid = qr["id"]
        query = qr["query"]

        # --- Retrieve + rerank ---
        candidates = retrieve(query, index, metadata, top_n=RETRIEVAL_TOP_N, provider="openai")
        top5 = rerank(query, candidates, top_n=RERANK_TOP_N)

        # --- Component scores (once per query) ---
        fake_email = _query_as_email(qid, query)
        query_features = extract_features(fake_email)
        style = score_style(profile, query_features)
        groundedness = score_groundedness(query, top5)
        confidence = score_confidence(query, query, top5)

        final = round(
            FORMULA_WEIGHTS[0] * style + FORMULA_WEIGHTS[1] * groundedness + FORMULA_WEIGHTS[2] * confidence,
            6,
        )
        decision = "deliver" if final >= THRESHOLD else "fallback"

        # --- Explanation × 2 ---
        gpt_expl, gpt_ms, gpt_ok = _explain(
            _gpt_client, GPT_MODEL, style, groundedness, confidence, final, decision
        )
        ollama_expl, ollama_ms, ollama_ok = _explain(
            _ollama_client, OLLAMA_MODEL, style, groundedness, confidence, final, decision
        )

        per_query.append({
            "id": qid,
            "query": query,
            "style": style,
            "groundedness": groundedness,
            "confidence": confidence,
            "final": final,
            "decision": decision,
            "gpt_expl": gpt_expl,
            "gpt_ms": gpt_ms * 1000,
            "gpt_ok": gpt_ok,
            "ollama_expl": ollama_expl,
            "ollama_ms": ollama_ms * 1000,
            "ollama_ok": ollama_ok,
        })

        print(f"{qid:<8} {style:>6.4f} {groundedness:>6.4f} {confidence:>6.4f} "
              f"{final:>6.4f} {decision:<8} "
              f"{gpt_ms*1000:>7.0f} {ollama_ms*1000:>7.0f} {'✓' if ollama_ok else '✗':>6}")

        if q_idx < len(queries) - 1:
            time.sleep(COHERE_INTER_QUERY_SLEEP)

    print(sep2)

    # --- Aggregate stats ---
    finals = [r["final"] for r in per_query]
    gpt_latencies = [r["gpt_ms"] for r in per_query]
    ollama_latencies = [r["ollama_ms"] for r in per_query]
    ollama_success_rate = sum(1 for r in per_query if r["ollama_ok"]) / len(per_query)

    print(f"\n--- Aggregate stats ---")
    print(f"  Final scores (identical for both): mean={np.mean(finals):.4f}  "
          f"std={np.std(finals):.4f}  min={np.min(finals):.4f}  max={np.max(finals):.4f}")
    print(f"  GPT-4o-mini latency (ms): mean={np.mean(gpt_latencies):.0f}  "
          f"std={np.std(gpt_latencies):.0f}  min={np.min(gpt_latencies):.0f}  "
          f"max={np.max(gpt_latencies):.0f}")
    print(f"  Ollama qwen3:8b latency (ms): mean={np.mean(ollama_latencies):.0f}  "
          f"std={np.std(ollama_latencies):.0f}  min={np.min(ollama_latencies):.0f}  "
          f"max={np.max(ollama_latencies):.0f}")
    print(f"  Ollama structured-output success rate: {ollama_success_rate:.0%} "
          f"({sum(1 for r in per_query if r['ollama_ok'])}/{len(per_query)})")

    # --- Pearson (trivially 1.0) ---
    # Both configs use the same deterministic formula from the same component scores.
    pearson_r, pearson_p = pearsonr(finals, finals)
    print(f"\n  Pearson(GPT-scores, Ollama-scores) = {pearson_r:.6f} (trivially 1.0 by design)")
    print(f"  Note: Pearson is 1.0 because final = deterministic(style, groundedness, confidence)")
    print(f"  and both configs share the same component scores. The experiment measures")
    print(f"  explanation quality and latency, not score divergence.")

    # --- 5-query spot check ---
    print(f"\n--- Spot check: first {SPOT_CHECK_N} queries — explanation comparison ---")
    for r in per_query[:SPOT_CHECK_N]:
        print(f"\n  {r['id']}: {r['query'][:60]}...")
        print(f"  GPT: {r['gpt_expl']}")
        print(f"  OL:  {r['ollama_expl']}")

    # --- Decision ---
    latency_ratio = np.mean(ollama_latencies) / max(np.mean(gpt_latencies), 1.0)
    latency_meaningful = latency_ratio > 1.5 or latency_ratio < 0.67
    quality_parity = (ollama_success_rate == 1.0)

    print(f"\n--- Decision ---")
    print(f"  Ollama success rate: {ollama_success_rate:.0%}  |  "
          f"Latency ratio (Ollama/GPT): {latency_ratio:.2f}x  |  "
          f"Latency meaningfully different: {latency_meaningful}")

    if not quality_parity:
        decision_str = (
            f"QUALITY DRIFT: Ollama structured-output success rate = {ollama_success_rate:.0%} "
            f"(< 100%). Recommend GPT-4o-mini for both dev and prod. "
            f"Document the Ollama failure rate in ADR-006."
        )
        outcome_tag = "quality_drift"
    elif quality_parity and latency_meaningful:
        if latency_ratio < 1.0:
            latency_desc = f"Ollama takes {latency_ratio:.2f}x GPT-4o-mini latency ({1/latency_ratio:.1f}x faster)"
        else:
            latency_desc = f"Ollama takes {latency_ratio:.2f}x GPT-4o-mini latency ({latency_ratio:.1f}x slower)"
        decision_str = (
            f"LATENCY-QUALITY TRADEOFF: Ollama explanations structurally valid (100% success). "
            f"{latency_desc}. "
            f"Recommend Ollama for dev (zero API cost), GPT-4o-mini for prod. "
            f"Verify spot-check quality above before accepting dev recommendation."
        )
        outcome_tag = "latency_tradeoff"
    else:
        decision_str = (
            f"QUALITY PARITY: Ollama explanations structurally valid (100% success) and "
            f"latency within 1.5x of GPT-4o-mini ({latency_ratio:.2f}x). "
            f"Recommend Ollama for dev, GPT-4o-mini for prod. "
            f"Verify spot-check quality above before accepting dev recommendation."
        )
        outcome_tag = "quality_parity"

    print(f"\n  OUTCOME: {outcome_tag.upper()}")
    print(f"  {decision_str}")

    # --- Chart ---
    _save_chart(per_query, gpt_latencies, ollama_latencies, pearson_r, outcome_tag)
    print(f"\nChart saved: {CHART_PATH}")
    print(sep)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_chart(
    per_query: list[dict],
    gpt_latencies: list[float],
    ollama_latencies: list[float],
    pearson_r: float,
    outcome_tag: str,
) -> None:
    qids = [r["id"] for r in per_query]
    finals = [r["final"] for r in per_query]
    ollama_ok = [r["ollama_ok"] for r in per_query]
    n = len(qids)
    x = np.arange(n)

    fig, (ax_scatter, ax_lat) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        f"Experiment 6e — GPT-4o-mini vs Ollama qwen3:8b  |  Outcome: {outcome_tag}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: scatter — GPT final score vs Ollama final score
    # Since finals are identical, all points lie on y=x. Color by ollama_ok.
    colors = ["#2ca02c" if ok else "#d62728" for ok in ollama_ok]
    ax_scatter.scatter(finals, finals, c=colors, s=60, zorder=3, label="queries")
    lo, hi = min(finals) - 0.05, max(finals) + 0.05
    lo, hi = max(0.0, lo), min(1.0, hi)
    ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y=x reference")
    ax_scatter.axhline(0.75, color="red", linestyle=":", linewidth=0.8, label="threshold 0.75")
    ax_scatter.axvline(0.75, color="red", linestyle=":", linewidth=0.8)
    ax_scatter.set_xlabel("GPT-4o-mini final score")
    ax_scatter.set_ylabel("Ollama qwen3:8b final score")
    ax_scatter.set_title(
        f"Final score parity\nPearson r={pearson_r:.4f} (trivially 1.0 — same formula)",
        fontsize=9,
    )
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # Annotate with query ids
    for i, qid in enumerate(qids):
        ax_scatter.annotate(qid, (finals[i], finals[i]),
                            textcoords="offset points", xytext=(4, 4), fontsize=6)

    # Panel 2: latency bar chart
    bar_w = 0.35
    ax_lat.bar(x - bar_w / 2, gpt_latencies, bar_w, label="GPT-4o-mini", color="#1f77b4", alpha=0.8)
    ax_lat.bar(x + bar_w / 2, ollama_latencies, bar_w, label="Ollama qwen3:8b", color="#ff7f0e", alpha=0.8)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(qids, rotation=45, ha="right", fontsize=8)
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xlabel("Query")

    gpt_mean = np.mean(gpt_latencies)
    ol_mean = np.mean(ollama_latencies)
    ratio = ol_mean / max(gpt_mean, 1.0)
    ax_lat.set_title(
        f"Explanation latency per query\n"
        f"GPT mean={gpt_mean:.0f}ms  Ollama mean={ol_mean:.0f}ms  ratio={ratio:.2f}x",
        fontsize=9,
    )
    ax_lat.legend(fontsize=8)
    ax_lat.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
