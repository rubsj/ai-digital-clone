"""Day 6 Experiment 6a — Embedding comparison: OpenAI text-embedding-3-small vs all-MiniLM-L6-v2

Hypothesis (logged BEFORE running, per Day 6 protocol):
  H1. q01 (TCP, ~9% corpus coverage) and q03 (binary search, ~28%) may top out near
      1.0 groundedness for both configs. If both land >=0.95 they are non-differentiating;
      report them separately rather than including in the unweighted mean.
  H2. The 98% programming-subfield corpus concentration likely compresses the delta below
      P5 RAG-eval's +26% Recall@5. Both embeddings can retrieve "some programming book"
      for almost any CS query. Expected Δmean_groundedness: 10-18%. Direction should hold
      (OpenAI > MiniLM); magnitude may shrink.

Config (all knobs in source, not env-vars):
  corpus   : open-phi/textbooks, field=computer_science, max_docs=20 (~900 chunks)
  chunking : RecursiveCharacterTextSplitter, size=500, overlap=50
  reranker : Cohere rerank-english-v3.0, top-20->top-5
  variable : embedding model only (text-embedding-3-small 1536d vs all-MiniLM-L6-v2 384d)

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

# reranker.py reads CO_API_KEY; .env uses COHERE_API_KEY — align at script startup
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

CORPUS_MAX_DOCS = 1  # 1 CS textbook ≈ 1500 chunks; 20 docs = 30K chunks → 921MB JSON cache OOM
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5
STYLE_SCORE_FIXED = 0.75  # held constant; see module docstring
FORMULA_WEIGHTS = (0.4, 0.4, 0.2)  # style, groundedness, confidence

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
CHART_PATH = Path("docs/images/6a-embeddings.png")


# ---------------------------------------------------------------------------
# Cohere quota pre-flight
# ---------------------------------------------------------------------------

def _cohere_preflight() -> None:
    """Issue a synthetic rerank call. Aborts (non-zero exit) on any failure."""
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
    """Load FAISS index if cached; build, embed, and save otherwise."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    query_ids = [r["id"] for r in results["openai"]]
    openai_finals = [r["final"] for r in results["openai"]]
    minilm_finals = [r["final"] for r in results["minilm"]]
    openai_grounds = [r["groundedness"] for r in results["openai"]]
    minilm_grounds = [r["groundedness"] for r in results["minilm"]]

    x = np.arange(len(query_ids))
    width = 0.35

    # Left panel: per-query final score (grouped bar)
    ax1.bar(x - width / 2, openai_finals, width, label="OpenAI", color="#2563EB", alpha=0.85)
    ax1.bar(x + width / 2, minilm_finals, width, label="MiniLM", color="#EA580C", alpha=0.85)
    ax1.axhline(0.75, color="black", linestyle="--", linewidth=0.9, label="threshold=0.75")
    ax1.set_xlabel("Query ID")
    ax1.set_ylabel("Final Score")
    ax1.set_title("Per-query Final Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)

    # Right panel: groundedness mean ± stdev with individual dots
    configs = ["openai", "minilm"]
    means = [float(np.mean(openai_grounds)), float(np.mean(minilm_grounds))]
    stds = [float(np.std(openai_grounds)), float(np.std(minilm_grounds))]
    colors = ["#2563EB", "#EA580C"]

    bars = ax2.bar(configs, means, color=colors, alpha=0.85)
    ax2.errorbar(configs, means, yerr=stds, fmt="none", color="black", capsize=5, linewidth=1.5)

    rng = np.random.RandomState(42)
    for i, grounds in enumerate([openai_grounds, minilm_grounds]):
        jitter = rng.uniform(-0.12, 0.12, len(grounds))
        ax2.scatter(
            [i + j for j in jitter],
            grounds,
            color="black",
            alpha=0.5,
            s=18,
            zorder=3,
        )

    ax2.set_ylabel("Groundedness Score")
    ax2.set_title("Groundedness Mean ± Stdev (10 queries)")
    ax2.set_ylim(0, 1.1)

    for bar, mean_val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + 0.03,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle("6a: OpenAI vs MiniLM Embedding Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Cohere pre-check (must pass before any query loop)
    _cohere_preflight()

    print("\n--- Corpus + Chunking ---")
    docs = load_corpus(max_docs=CORPUS_MAX_DOCS)
    print(f"  {len(docs)} documents loaded.")
    chunks_raw = chunk_baseline(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks_raw)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("\n--- Index Build / Load ---")
    indices: dict[str, tuple] = {}
    for name, cfg in EMBED_CONFIGS.items():
        idx, meta = _build_or_load(cfg["provider"], cfg["index_dir"], cfg["dimension"], chunks_raw)
        indices[name] = (idx, meta)

    print("\n--- Queries ---")
    queries = load_queries(QUERIES_PATH)
    print(f"  {len(queries)} queries loaded from {QUERIES_PATH}")

    print("\nPre-run hypotheses:")
    print("  H1: q01(TCP) and q03(binary search) may top out >=0.95 groundedness — non-differentiating.")
    print("  H2: 98% programming-subfield may compress delta to 10-18% (vs P5 prior: +26%).")

    # 2. Run queries
    results: dict[str, list[dict]] = {name: [] for name in EMBED_CONFIGS}

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"{'Query':<8} {'Config':<8} {'Ground':>8} {'Conf':>8} {'Final':>8} {'Lat(ms)':>10}")
    print(sep)

    for qr in queries:
        qid = qr["id"]
        query = qr["query"]

        for name, cfg in EMBED_CONFIGS.items():
            idx, meta = indices[name]

            t0 = time.perf_counter()
            candidates = retrieve(query, idx, meta, top_n=RETRIEVAL_TOP_N, provider=cfg["provider"])
            reranked = rerank(query, candidates, top_n=RERANK_TOP_N)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Query text as response proxy: reproducible, config-agnostic
            groundedness = score_groundedness(query, reranked)
            confidence = score_confidence(query, query, reranked)
            final = round(
                FORMULA_WEIGHTS[0] * STYLE_SCORE_FIXED
                + FORMULA_WEIGHTS[1] * groundedness
                + FORMULA_WEIGHTS[2] * confidence,
                4,
            )

            results[name].append(
                {
                    "id": qid,
                    "topic": qr["topic"],
                    "band": qr["expected_groundedness_band"],
                    "groundedness": groundedness,
                    "confidence": confidence,
                    "final": final,
                    "latency_ms": latency_ms,
                }
            )

            print(
                f"{qid:<8} {name:<8} {groundedness:>8.4f} {confidence:>8.4f} "
                f"{final:>8.4f} {latency_ms:>10.1f}"
            )

    print(sep)

    # 3. Aggregate
    print("\n--- Aggregate Metrics ---")
    agg: dict[str, dict] = {}
    for name in EMBED_CONFIGS:
        rows = results[name]
        grounds = [r["groundedness"] for r in rows]
        finals = [r["final"] for r in rows]
        lats = [r["latency_ms"] for r in rows]
        agg[name] = {
            "mean_groundedness": float(np.mean(grounds)),
            "std_groundedness": float(np.std(grounds)),
            "mean_final": float(np.mean(finals)),
            "mean_latency_ms": float(np.mean(lats)),
        }
        print(
            f"  {name}: mean_groundedness={agg[name]['mean_groundedness']:.4f}  "
            f"mean_final={agg[name]['mean_final']:.4f}  "
            f"mean_latency={agg[name]['mean_latency_ms']:.1f}ms"
        )

    # H1 check: flag non-differentiating queries
    print("\n--- H1 Check: near-ceiling queries ---")
    for qr, r_oa, r_ml in zip(queries, results["openai"], results["minilm"]):
        if r_oa["groundedness"] >= 0.95 and r_ml["groundedness"] >= 0.95:
            print(f"  {qr['id']} ({qr['topic']}): openai={r_oa['groundedness']:.4f}, "
                  f"minilm={r_ml['groundedness']:.4f} — near-ceiling, non-differentiating")

    # H2 check: delta vs P5 prior
    g_oa = agg["openai"]["mean_groundedness"]
    g_ml = agg["minilm"]["mean_groundedness"]
    abs_delta = g_oa - g_ml
    pct_delta = abs_delta / max(g_ml, 1e-9) * 100
    print(f"\n--- H2 Check: delta vs P5 prior ---")
    print(f"  Δmean_groundedness (openai - minilm): {abs_delta:+.4f} ({pct_delta:+.1f}%)")
    print(f"  P5 prior: +26%. H2 predicted: 10-18%. Actual: {pct_delta:+.1f}%.")

    # 4. Chart
    _save_chart(results)
    print(f"\nChart saved -> {CHART_PATH}")

    # 5. Decision summary
    keep = "openai" if g_oa >= g_ml else "minilm"
    print(f"\nDecision: keep {keep.upper()} embeddings "
          f"(mean groundedness: openai={g_oa:.4f}, minilm={g_ml:.4f}).")


if __name__ == "__main__":
    main()
