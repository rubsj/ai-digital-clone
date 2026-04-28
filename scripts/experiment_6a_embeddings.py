"""Day 6 Experiment 6a (Run 2) — Embedding comparison: OpenAI vs MiniLM

Hypothesis (logged BEFORE running, per Day 6 protocol):
  H1. q01 (TCP, ~9% corpus coverage) and q03 (binary search, ~28%) may top out near
      1.0 groundedness for both configs. If both land >=0.95 they are non-differentiating;
      report them separately rather than including in the unweighted mean.
  H2. The 98% programming-subfield corpus concentration likely compresses the delta below
      P5 RAG-eval's +26% Recall@5. Both embeddings can retrieve "some programming book"
      for almost any CS query. Expected Δmean_groundedness: 10-18%. Direction should hold
      (OpenAI > MiniLM); magnitude may shrink.

Run 1 rejection findings (now fixed):
  F1. Single textbook (max_docs=1, 1476 chunks) too small; candidate pool 60% identical
      for q01 → reranker collapses embedding difference.
  F2. 7/10 queries had bit-identical post-rerank groundedness because top-3 reranked
      chunks were identical (same candidates in). Collapse site: Cohere rerank step.
  F3. Confidence 0.6667 floor = query-as-proxy making completeness=1.0, penalty=1.0
      always. Documented as ADR-006 candidate. Not fixed (out of Day 6 scope).
  F4. 921MB JSON cache (30K chunks) → segfault. Fixed by switching to numpy npz cache.

Config (all knobs in source):
  corpus   : open-phi/textbooks, field=computer_science, max_docs=5 (~6713 chunks)
  chunking : RecursiveCharacterTextSplitter, size=500, overlap=50
  reranker : Cohere rerank-english-v3.0, top-20->top-20 (for distribution), then top-5
  variable : embedding model only (text-embedding-3-small 1536d vs all-MiniLM-L6-v2 384d)

New metrics in Run 2:
  pre_rerank_groundedness  — top-5 by raw FAISS score, NO Cohere. Isolates embedding.
  post_rerank_groundedness — top-5 selected by Cohere. Production metric.
  cohere_dist              — mean/std/min/max of Cohere scores across all top-20 candidates.

Confidence limitation note:
  score_confidence() with query-as-proxy: completeness=1.0, uncertainty_penalty=1.0
  always. Only retrieval_relevance (1/3 weight) varies. Confidence ~= 0.667 + relevance/3.
  ADR-006 candidate. Reported for completeness; not used in primary comparison.

Scoring note:
  style_score = STYLE_SCORE_FIXED (0.75) — held constant; this experiment isolates
  retrieval quality, not style. Response proxy = query text itself (reproducible, no LLM
  stochasticity across configs). score_groundedness() always uses embed_openai() for
  response-sentence embedding regardless of config; the variable is WHICH chunks are
  retrieved. final = 0.4*style + 0.4*groundedness + 0.2*confidence.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("CO_API_KEY") and os.environ.get("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]

import cohere
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.query_loader import load_queries
from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import score_groundedness
from src.rag.chunker import chunk_baseline
from src.rag.corpus_loader import load_corpus
from src.rag.embedder import embed_chunks
from src.rag.indexer import build_index, load_index, save_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_MAX_DOCS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5
STYLE_SCORE_FIXED = 0.75
FORMULA_WEIGHTS = (0.4, 0.4, 0.2)

EMBED_CONFIGS: dict[str, dict] = {
    "openai": {
        "provider": "openai",
        "index_dir": Path("data/rag/faiss_index"),
        "dimension": 1536,
    },
    "minilm": {
        "provider": "minilm",
        "index_dir": Path("data/rag/faiss_index_minilm"),
        "dimension": 384,
    },
}

QUERIES_PATH = Path("data/eval/queries_v1.json")
CHART_PATH = Path("docs/images/6a-embeddings-run2.png")


# ---------------------------------------------------------------------------
# Cohere quota pre-flight
# ---------------------------------------------------------------------------

def _cohere_preflight() -> None:
    client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY", ""))
    try:
        client.rerank(
            model="rerank-english-v3.0",
            query="health check",
            documents=["foo", "bar"],
            top_n=1,
        )
    except Exception as exc:
        raise SystemExit(f"[ABORT] Cohere pre-check failed: {exc}") from exc
    print("cohere quota pre-check OK")


# ---------------------------------------------------------------------------
# Index build/load
# ---------------------------------------------------------------------------

def _build_or_load(
    provider: str,
    index_dir: Path,
    dimension: int,
    chunks_raw: list,
) -> tuple:
    index_path = index_dir / "index.faiss"
    if index_path.exists():
        print(f"  [{provider}] Loading cached index from {index_dir} ...")
        return load_index(index_dir)
    print(f"  [{provider}] Building {dimension}d index from {len(chunks_raw)} chunks ...")
    t0 = time.perf_counter()
    embedded = embed_chunks(chunks_raw, provider=provider)
    index, metadata = build_index(embedded, dimension=dimension)
    save_index(index, metadata, index_dir=index_dir)
    elapsed = time.perf_counter() - t0
    print(f"  [{provider}] Built and saved: {index.ntotal} vectors in {elapsed:.1f}s -> {index_dir}")
    return index, metadata


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_chart(results: dict[str, list[dict]]) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    ax1, ax2, ax3 = axes

    query_ids = [r["id"] for r in results["openai"]]
    x = np.arange(len(query_ids))
    width = 0.35

    oa_post = [r["post_rerank_groundedness"] for r in results["openai"]]
    ml_post = [r["post_rerank_groundedness"] for r in results["minilm"]]
    oa_pre = [r["pre_rerank_groundedness"] for r in results["openai"]]
    ml_pre = [r["pre_rerank_groundedness"] for r in results["minilm"]]

    # Panel 1: post-rerank groundedness per query
    ax1.bar(x - width / 2, oa_post, width, label="OpenAI post-rerank", color="#2563EB", alpha=0.85)
    ax1.bar(x + width / 2, ml_post, width, label="MiniLM post-rerank", color="#EA580C", alpha=0.85)
    ax1.set_xlabel("Query ID")
    ax1.set_ylabel("Groundedness")
    ax1.set_title("Post-Rerank Groundedness (top-5 Cohere)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)

    # Panel 2: pre-rerank groundedness per query
    ax2.bar(x - width / 2, oa_pre, width, label="OpenAI pre-rerank", color="#1D4ED8", alpha=0.85)
    ax2.bar(x + width / 2, ml_pre, width, label="MiniLM pre-rerank", color="#C2410C", alpha=0.85)
    ax2.set_xlabel("Query ID")
    ax2.set_ylabel("Groundedness")
    ax2.set_title("Pre-Rerank Groundedness (top-5 by FAISS score)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)

    # Panel 3: mean groundedness comparison (post vs pre, with stdev)
    configs_labels = ["OA-post", "ML-post", "OA-pre", "ML-pre"]
    means_all = [
        float(np.mean(oa_post)), float(np.mean(ml_post)),
        float(np.mean(oa_pre)), float(np.mean(ml_pre)),
    ]
    stds_all = [
        float(np.std(oa_post)), float(np.std(ml_post)),
        float(np.std(oa_pre)), float(np.std(ml_pre)),
    ]
    colors_all = ["#2563EB", "#EA580C", "#1D4ED8", "#C2410C"]

    bars = ax3.bar(configs_labels, means_all, color=colors_all, alpha=0.85)
    ax3.errorbar(configs_labels, means_all, yerr=stds_all, fmt="none", color="black", capsize=5)
    for bar, m in zip(bars, means_all):
        ax3.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_ylabel("Mean Groundedness")
    ax3.set_title("Mean ± Stdev: Post vs Pre-Rerank")
    ax3.set_ylim(0, 1.1)

    fig.suptitle("6a Run 2: OpenAI vs MiniLM (5-doc corpus, npz cache)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {CHART_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _cohere_preflight()

    print("\n--- Corpus + Chunking ---")
    docs = load_corpus(max_docs=CORPUS_MAX_DOCS)
    from collections import Counter
    subfields = Counter(d.subfield for d in docs)
    print(f"  {len(docs)} documents | subfields: {dict(subfields)}")
    chunks_raw = chunk_baseline(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks_raw)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("\n--- Index Build / Load ---")
    indices: dict[str, tuple] = {}
    for name, cfg in EMBED_CONFIGS.items():
        idx, meta = _build_or_load(cfg["provider"], cfg["index_dir"], cfg["dimension"], chunks_raw)
        indices[name] = (idx, meta)

    print("\n--- Queries ---")
    queries = load_queries(QUERIES_PATH)
    print(f"  {len(queries)} queries from {QUERIES_PATH}")

    print("\nPre-run hypotheses:")
    print("  H1: q01(TCP) and q03(binary search) may top out >=0.95 groundedness.")
    print("  H2: Programming-heavy corpus may compress OA-vs-MiniLM delta to 10-18%.")

    results: dict[str, list[dict]] = {name: [] for name in EMBED_CONFIGS}

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"{'Query':<8} {'Config':<8} {'Post-G':>8} {'Pre-G':>8} {'CohMean':>9} {'CohStd':>8} {'CohMax':>8} {'Conf':>8} {'Lat(ms)':>10}")
    print(sep)

    # Trial key: 10 Cohere calls/min. We make 2 per query (one per embedding) = 20 total.
    # Sleep 7s after both calls per query to stay under the rate limit (2 calls per 7s = ~17/min
    # but we process both sequentially within ~1s, so sleeping 7s after the pair clears the window).
    COHERE_INTER_QUERY_SLEEP = 7.0

    for q_idx, qr in enumerate(queries):
        qid = qr["id"]
        query = qr["query"]

        for name, cfg in EMBED_CONFIGS.items():
            idx, meta = indices[name]

            t0 = time.perf_counter()

            # Retrieve top-20 by FAISS (raw embedding score)
            candidates = retrieve(query, idx, meta, top_n=RETRIEVAL_TOP_N, provider=cfg["provider"])

            # Pre-rerank top-5: first 5 by FAISS score (no Cohere)
            pre_top5 = candidates[:RERANK_TOP_N]

            # Rerank top-20 → top-20 to capture full Cohere score distribution
            all_reranked = rerank(query, candidates, top_n=RETRIEVAL_TOP_N)
            post_top5 = all_reranked[:RERANK_TOP_N]

            latency_ms = (time.perf_counter() - t0) * 1000

            # Groundedness: post-rerank (primary) and pre-rerank (embedding isolation)
            post_g = score_groundedness(query, post_top5)
            pre_g = score_groundedness(query, pre_top5)

            confidence = score_confidence(query, query, post_top5)
            final = round(
                FORMULA_WEIGHTS[0] * STYLE_SCORE_FIXED
                + FORMULA_WEIGHTS[1] * post_g
                + FORMULA_WEIGHTS[2] * confidence,
                4,
            )

            # Cohere score distribution across all 20 reranked candidates
            cohere_scores = [r.score for r in all_reranked]
            coh_mean = float(np.mean(cohere_scores))
            coh_std = float(np.std(cohere_scores))
            coh_max = float(np.max(cohere_scores))

            results[name].append({
                "id": qid,
                "topic": qr["topic"],
                "band": qr["expected_groundedness_band"],
                "post_rerank_groundedness": post_g,
                "pre_rerank_groundedness": pre_g,
                "cohere_mean": coh_mean,
                "cohere_std": coh_std,
                "cohere_max": coh_max,
                "confidence": confidence,
                "final": final,
                "latency_ms": latency_ms,
            })

            print(
                f"{qid:<8} {name:<8} {post_g:>8.4f} {pre_g:>8.4f} "
                f"{coh_mean:>9.4f} {coh_std:>8.4f} {coh_max:>8.4f} "
                f"{confidence:>8.4f} {latency_ms:>10.1f}"
            )

        # Throttle: trial key allows 10 Cohere calls/min; 2 calls per query pair.
        # Skip sleep after last query.
        if q_idx < len(queries) - 1:
            time.sleep(COHERE_INTER_QUERY_SLEEP)

    print(sep)

    # Aggregate
    print("\n--- Aggregate Metrics ---")
    agg: dict[str, dict] = {}
    for name in EMBED_CONFIGS:
        rows = results[name]
        post_gs = [r["post_rerank_groundedness"] for r in rows]
        pre_gs = [r["pre_rerank_groundedness"] for r in rows]
        coh_means = [r["cohere_mean"] for r in rows]
        agg[name] = {
            "mean_post_groundedness": float(np.mean(post_gs)),
            "std_post_groundedness": float(np.std(post_gs)),
            "mean_pre_groundedness": float(np.mean(pre_gs)),
            "std_pre_groundedness": float(np.std(pre_gs)),
            "mean_cohere_score": float(np.mean(coh_means)),
        }
        print(
            f"  {name}: post_G={agg[name]['mean_post_groundedness']:.4f}±{agg[name]['std_post_groundedness']:.4f}  "
            f"pre_G={agg[name]['mean_pre_groundedness']:.4f}±{agg[name]['std_pre_groundedness']:.4f}  "
            f"mean_cohere={agg[name]['mean_cohere_score']:.4f}"
        )

    # H1 check
    print("\n--- H1 Check: near-ceiling queries ---")
    ceiling_found = False
    for qr, r_oa, r_ml in zip(queries, results["openai"], results["minilm"]):
        if r_oa["post_rerank_groundedness"] >= 0.95 and r_ml["post_rerank_groundedness"] >= 0.95:
            print(f"  {qr['id']} ({qr['topic']}): OA={r_oa['post_rerank_groundedness']:.4f}, "
                  f"ML={r_ml['post_rerank_groundedness']:.4f} — near-ceiling")
            ceiling_found = True
    if not ceiling_found:
        print("  No queries hit >=0.95 ceiling (H1 not triggered).")

    # H2 check
    pg_oa = agg["openai"]["mean_post_groundedness"]
    pg_ml = agg["minilm"]["mean_post_groundedness"]
    abs_delta_post = pg_oa - pg_ml
    pct_delta_post = abs_delta_post / max(pg_ml, 1e-9) * 100
    preg_oa = agg["openai"]["mean_pre_groundedness"]
    preg_ml = agg["minilm"]["mean_pre_groundedness"]
    abs_delta_pre = preg_oa - preg_ml
    pct_delta_pre = abs_delta_pre / max(preg_ml, 1e-9) * 100
    print(f"\n--- H2 Check: embedding delta vs P5 prior ---")
    print(f"  Post-rerank Δ (OA - ML): {abs_delta_post:+.4f} ({pct_delta_post:+.1f}%)")
    print(f"  Pre-rerank  Δ (OA - ML): {abs_delta_pre:+.4f} ({pct_delta_pre:+.1f}%)")
    print(f"  P5 prior: +26%. H2 predicted: 10-18%.")

    # Cohere reranker behavior summary
    print(f"\n--- Cohere Reranker Behavior (top-20 score distributions) ---")
    for name in EMBED_CONFIGS:
        rows = results[name]
        all_maxes = [r["cohere_max"] for r in rows]
        all_means = [r["cohere_mean"] for r in rows]
        all_stds = [r["cohere_std"] for r in rows]
        print(f"  {name}: per-query Cohere max: mean={np.mean(all_maxes):.4f}, "
              f"range=[{min(all_maxes):.4f}, {max(all_maxes):.4f}]")
        print(f"         per-query Cohere std: mean={np.mean(all_stds):.4f} "
              f"(low std = uniform/non-differentiating reranker)")

    # Decision
    keep = "openai" if pg_oa >= pg_ml else "minilm"
    print(f"\nDecision: keep {keep.upper()} (post-rerank: OA={pg_oa:.4f}, ML={pg_ml:.4f})")

    _save_chart(results)


if __name__ == "__main__":
    main()
