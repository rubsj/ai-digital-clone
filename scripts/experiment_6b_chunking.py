"""Day 6 Experiment 6b — Chunking comparison: fixed 500/50 vs semantic markdown

Goal: same 10 queries through two configurations that differ ONLY in chunking.
Same OpenAI embedding, same Cohere reranker, same retriever, same evaluation.

Chunk-relevance metric (defined before measuring):
  Mean of the top-5 Cohere reranker relevance scores per query,
  averaged across all 10 queries. Higher = reranker finds the retrieved chunks
  more relevant. This isolates chunking quality from embedding/reranker choice.

Pre-run hypotheses (logged before running):
  H1: On the 6 queries where Phase 2 saw near-zero Cohere signal (q01, q05, q07,
      q08, q09, q10), chunking changes will NOT recover Cohere relevance. The
      corpus lacks the relevant content; chunking cannot create it.
  H2: On the 4 queries where Phase 2 saw real Cohere signal (q02, q03, q04, q06),
      semantic chunking may shift WHICH chunks rank highest but Cohere max scores
      should remain in their current band (>0.15).
  H3: The open-phi/textbooks documents are loaded from a 'markdown' column, so
      chunk_semantic() may find section headers and produce meaningfully different
      chunks from chunk_baseline(). If the documents are prose-heavy with few
      headers, semantic ≈ baseline (chunk count similar, same fallback path fires).
  H4: Net aggregate groundedness delta will be smaller than the per-query delta on
      high-Cohere queries. Plan to report both aggregate and high-Cohere-subset.

Config (all knobs in source):
  corpus      : open-phi/textbooks, field=computer_science, max_docs=5
  chunking    : (A) RecursiveCharacterTextSplitter, size=500, overlap=50
                (B) MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter, size=500, overlap=50
  embedding   : text-embedding-3-small (1536d) — same for both configs
  reranker    : Cohere rerank-english-v3.0, top-20 → top-5
  variable    : chunking strategy only

Scoring note:
  style_score = STYLE_SCORE_FIXED (0.75) — held constant.
  Response proxy = query text itself (reproducible, no LLM stochasticity).
  score_groundedness() always uses embed_openai() regardless of chunking config.
  final = 0.4*style + 0.4*groundedness + 0.2*confidence.
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
from src.rag.chunker import chunk_baseline, chunk_semantic
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

CHUNK_CONFIGS: dict[str, dict] = {
    "baseline": {
        "index_dir": Path("data/rag/faiss_index"),
        "embed_cache": Path("data/cache/embeddings_openai.npz"),
        "description": "RecursiveCharacterTextSplitter 500/50",
    },
    "semantic": {
        "index_dir": Path("data/rag/faiss_index_semantic"),
        "embed_cache": Path("data/cache/embeddings_openai_semantic.npz"),
        "description": "MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter 500/50",
    },
}

QUERIES_PATH = Path("data/eval/queries_v1.json")
CHART_PATH = Path("docs/images/6b-chunking.png")

# Cohere trial key: 10 calls/min. 21 calls (1 pre-check + 20 per query×config).
COHERE_INTER_QUERY_SLEEP = 7.0


# ---------------------------------------------------------------------------
# Cohere preflight
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
    name: str,
    cfg: dict,
    chunks_by_strategy: dict[str, list],
) -> tuple:
    index_path = cfg["index_dir"] / "index.faiss"
    if index_path.exists():
        print(f"  [{name}] Loading cached index from {cfg['index_dir']} ...")
        return load_index(cfg["index_dir"])

    chunks = chunks_by_strategy[name]
    print(f"  [{name}] Building 1536d index from {len(chunks)} chunks ...")
    t0 = time.perf_counter()
    embedded = embed_chunks(chunks, provider="openai", cache_path=cfg["embed_cache"])
    index, metadata = build_index(embedded, dimension=1536)
    save_index(index, metadata, index_dir=cfg["index_dir"])
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] Built and saved: {index.ntotal} vectors in {elapsed:.1f}s -> {cfg['index_dir']}")
    return index, metadata


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_chart(results: dict[str, list[dict]]) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    ax1, ax2, ax3 = axes

    query_ids = [r["id"] for r in results["baseline"]]
    x = np.arange(len(query_ids))
    width = 0.35

    bl_post = [r["post_g"] for r in results["baseline"]]
    sm_post = [r["post_g"] for r in results["semantic"]]
    bl_rel = [r["chunk_relevance"] for r in results["baseline"]]
    sm_rel = [r["chunk_relevance"] for r in results["semantic"]]

    # Panel 1: post-rerank groundedness per query
    ax1.bar(x - width / 2, bl_post, width, label="baseline post-G", color="#2563EB", alpha=0.85)
    ax1.bar(x + width / 2, sm_post, width, label="semantic post-G", color="#EA580C", alpha=0.85)
    ax1.set_xlabel("Query ID")
    ax1.set_ylabel("Groundedness")
    ax1.set_title("Post-Rerank Groundedness (top-5 Cohere)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)

    # Panel 2: chunk-relevance per query
    ax2.bar(x - width / 2, bl_rel, width, label="baseline chunk-rel", color="#1D4ED8", alpha=0.85)
    ax2.bar(x + width / 2, sm_rel, width, label="semantic chunk-rel", color="#C2410C", alpha=0.85)
    ax2.set_xlabel("Query ID")
    ax2.set_ylabel("Chunk Relevance (mean top-5 Cohere score)")
    ax2.set_title("Chunk Relevance (mean top-5 Cohere)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)

    # Panel 3: aggregate comparison (groundedness + chunk-relevance, mean±stdev)
    configs_labels = ["BL-post-G", "SM-post-G", "BL-chunk-rel", "SM-chunk-rel"]
    means_all = [
        float(np.mean(bl_post)), float(np.mean(sm_post)),
        float(np.mean(bl_rel)), float(np.mean(sm_rel)),
    ]
    stds_all = [
        float(np.std(bl_post)), float(np.std(sm_post)),
        float(np.std(bl_rel)), float(np.std(sm_rel)),
    ]
    colors_all = ["#2563EB", "#EA580C", "#1D4ED8", "#C2410C"]

    bars = ax3.bar(configs_labels, means_all, color=colors_all, alpha=0.85)
    ax3.errorbar(configs_labels, means_all, yerr=stds_all, fmt="none", color="black", capsize=5)
    for bar, m in zip(bars, means_all):
        ax3.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.set_title("Aggregate: Groundedness + Chunk Relevance (mean ± stdev)")
    ax3.set_ylim(0, 1.1)

    fig.suptitle("6b: Chunking — Baseline 500/50 vs Semantic Markdown (5-doc corpus)", fontsize=12, fontweight="bold")
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

    chunks_baseline = chunk_baseline(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_semantic = chunk_semantic(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  baseline chunks: {len(chunks_baseline)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  semantic chunks: {len(chunks_semantic)} (MarkdownHeader → RecursiveChar)")

    chunks_by_strategy = {"baseline": chunks_baseline, "semantic": chunks_semantic}

    # H3 check: chunk count similarity
    delta_pct = abs(len(chunks_semantic) - len(chunks_baseline)) / max(len(chunks_baseline), 1) * 100
    if delta_pct < 5:
        print(f"  H3 note: chunk counts within {delta_pct:.1f}% — markdown headers rare in corpus; "
              f"semantic ≈ baseline for most documents.")
    else:
        print(f"  H3 note: semantic produces {delta_pct:.1f}% different chunk count — "
              f"markdown headers found and acted on.")

    print("\n--- Index Build / Load ---")
    indices: dict[str, tuple] = {}
    for name, cfg in CHUNK_CONFIGS.items():
        idx, meta = _build_or_load(name, cfg, chunks_by_strategy)
        indices[name] = (idx, meta)

    print("\n--- Queries ---")
    queries = load_queries(QUERIES_PATH)
    print(f"  {len(queries)} queries from {QUERIES_PATH}")

    print("\nPre-run hypotheses:")
    print("  H1: Near-zero Cohere queries (q01,q05,q07,q08,q09,q10) — chunking won't recover signal.")
    print("  H2: High-Cohere queries (q02,q03,q04,q06) — semantic may shift chunks, not ceiling.")
    print("  H3: Semantic ≈ baseline if docs are prose-heavy (few markdown headers).")
    print("  H4: Aggregate delta < per-query delta on high-Cohere queries.")

    results: dict[str, list[dict]] = {name: [] for name in CHUNK_CONFIGS}

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"{'Query':<8} {'Config':<10} {'Post-G':>8} {'Pre-G':>8} {'ChunkRel':>10} {'CohMax':>8} {'Conf':>8} {'Lat(ms)':>10}")
    print(sep)

    for q_idx, qr in enumerate(queries):
        qid = qr["id"]
        query = qr["query"]

        for name in CHUNK_CONFIGS:
            idx, meta = indices[name]

            t0 = time.perf_counter()

            # Retrieve top-20 by FAISS
            candidates = retrieve(query, idx, meta, top_n=RETRIEVAL_TOP_N, provider="openai")
            pre_top5 = candidates[:RERANK_TOP_N]

            # Rerank top-20 → top-20 (capture full Cohere distribution)
            all_reranked = rerank(query, candidates, top_n=RETRIEVAL_TOP_N)
            post_top5 = all_reranked[:RERANK_TOP_N]

            latency_ms = (time.perf_counter() - t0) * 1000

            post_g = score_groundedness(query, post_top5)
            pre_g = score_groundedness(query, pre_top5)
            confidence = score_confidence(query, query, post_top5)

            # Chunk-relevance = mean Cohere score of the top-5 selected chunks
            # (scores of the post_top5 results)
            post_top5_scores = [r.score for r in all_reranked[:RERANK_TOP_N]]
            chunk_relevance = float(np.mean(post_top5_scores))
            coh_max = float(max(r.score for r in all_reranked))

            final = round(
                FORMULA_WEIGHTS[0] * STYLE_SCORE_FIXED
                + FORMULA_WEIGHTS[1] * post_g
                + FORMULA_WEIGHTS[2] * confidence,
                4,
            )

            results[name].append({
                "id": qid,
                "topic": qr["topic"],
                "band": qr["expected_groundedness_band"],
                "post_g": post_g,
                "pre_g": pre_g,
                "chunk_relevance": chunk_relevance,
                "cohere_max": coh_max,
                "confidence": confidence,
                "final": final,
                "latency_ms": latency_ms,
            })

            print(
                f"{qid:<8} {name:<10} {post_g:>8.4f} {pre_g:>8.4f} "
                f"{chunk_relevance:>10.4f} {coh_max:>8.4f} {confidence:>8.4f} {latency_ms:>10.1f}"
            )

        if q_idx < len(queries) - 1:
            time.sleep(COHERE_INTER_QUERY_SLEEP)

    print(sep)

    # Aggregate
    print("\n--- Aggregate Metrics ---")
    agg: dict[str, dict] = {}
    for name in CHUNK_CONFIGS:
        rows = results[name]
        post_gs = [r["post_g"] for r in rows]
        pre_gs = [r["pre_g"] for r in rows]
        chunk_rels = [r["chunk_relevance"] for r in rows]
        agg[name] = {
            "mean_post_g": float(np.mean(post_gs)),
            "std_post_g": float(np.std(post_gs)),
            "mean_pre_g": float(np.mean(pre_gs)),
            "std_pre_g": float(np.std(pre_gs)),
            "mean_chunk_rel": float(np.mean(chunk_rels)),
            "std_chunk_rel": float(np.std(chunk_rels)),
        }
        print(
            f"  {name}: post_G={agg[name]['mean_post_g']:.4f}±{agg[name]['std_post_g']:.4f}  "
            f"pre_G={agg[name]['mean_pre_g']:.4f}±{agg[name]['std_pre_g']:.4f}  "
            f"chunk_rel={agg[name]['mean_chunk_rel']:.4f}±{agg[name]['std_chunk_rel']:.4f}"
        )

    # Delta
    post_delta = agg["baseline"]["mean_post_g"] - agg["semantic"]["mean_post_g"]
    pre_delta = agg["baseline"]["mean_pre_g"] - agg["semantic"]["mean_pre_g"]
    rel_delta = agg["baseline"]["mean_chunk_rel"] - agg["semantic"]["mean_chunk_rel"]
    winner = "baseline" if post_delta >= 0 else "semantic"
    print(f"\n  Post-G Δ (baseline − semantic): {post_delta:+.4f} ({post_delta/max(agg['semantic']['mean_post_g'],1e-9)*100:+.1f}%)")
    print(f"  Pre-G  Δ (baseline − semantic): {pre_delta:+.4f} ({pre_delta/max(agg['semantic']['mean_pre_g'],1e-9)*100:+.1f}%)")
    print(f"  ChunkRel Δ (baseline − semantic): {rel_delta:+.4f}")
    print(f"  Winner (post-rerank groundedness): {winner.upper()}")

    # H1/H2 subset check — high-Cohere queries only
    HIGH_COHERE_QS = {"q02", "q03", "q04", "q06"}
    print(f"\n--- H2 Check: high-Cohere queries ({', '.join(sorted(HIGH_COHERE_QS))}) ---")
    for name in CHUNK_CONFIGS:
        rows = [r for r in results[name] if r["id"] in HIGH_COHERE_QS]
        if rows:
            mean_g = float(np.mean([r["post_g"] for r in rows]))
            mean_rel = float(np.mean([r["chunk_relevance"] for r in rows]))
            print(f"  {name}: mean_post_G={mean_g:.4f}  mean_chunk_rel={mean_rel:.4f}")
    print("  (For high-Cohere queries, chunking changes are expected to shift chunk selection")
    print("   but not dramatically move the Cohere max ceiling per H2.)")

    # H1 check — near-zero Cohere queries
    NEAR_ZERO_QS = {"q01", "q05", "q07", "q08", "q09", "q10"}
    print(f"\n--- H1 Check: near-zero Cohere queries ({', '.join(sorted(NEAR_ZERO_QS))}) ---")
    for name in CHUNK_CONFIGS:
        rows = [r for r in results[name] if r["id"] in NEAR_ZERO_QS]
        if rows:
            mean_g = float(np.mean([r["post_g"] for r in rows]))
            mean_rel = float(np.mean([r["chunk_relevance"] for r in rows]))
            max_rel = float(max(r["chunk_relevance"] for r in rows))
            print(f"  {name}: mean_post_G={mean_g:.4f}  mean_chunk_rel={mean_rel:.4f}  max_chunk_rel={max_rel:.4f}")

    # Decision
    keep = winner
    post_bl = agg["baseline"]["mean_post_g"]
    post_sm = agg["semantic"]["mean_post_g"]
    print(f"\nDecision: keep {keep.upper()} "
          f"(post-rerank: baseline={post_bl:.4f}, semantic={post_sm:.4f})")

    _save_chart(results)


if __name__ == "__main__":
    main()
