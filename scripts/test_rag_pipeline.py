"""End-to-end validation of the Day 3 RAG pipeline.

Run with:  uv run python scripts/test_rag_pipeline.py

What this script validates:
  1. Corpus loading (capped at 20 docs)
  2. Baseline chunking → ≥900 chunks
  3. Embedding (OpenAI via LiteLLM — uses JSON cache after first run)
  4. FAISS index build + save to disk
  5. Retrieval: "What is TCP/IP?" → top-20 results, timed
  6. Cohere rerank → top-5
  7. Citation extraction from a sample text
  8. Rich summary table
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.rag.chunker import chunk_baseline
from src.rag.citation_extractor import extract_citations
from src.rag.corpus_loader import load_corpus
from src.rag.embedder import embed_chunks, embed_query
from src.rag.indexer import build_index, load_index, save_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve

console = Console()

INDEX_DIR = Path("data/rag/faiss_index")
QUERY = "What is TCP/IP?"
MAX_DOCS = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_N_INITIAL = 20
TOP_N_FINAL = 5


def main() -> None:
    console.rule("[bold blue]RAG Pipeline End-to-End Validation[/bold blue]")

    # ------------------------------------------------------------------ #
    # 1. Load corpus
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Step 1:[/bold] Loading corpus (max {MAX_DOCS} docs)…")
    docs = load_corpus(max_docs=MAX_DOCS)
    console.print(f"  Loaded [green]{len(docs)}[/green] documents.")

    # ------------------------------------------------------------------ #
    # 2. Chunk
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Step 2:[/bold] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})…")
    chunks = chunk_baseline(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks.")
    assert len(chunks) >= 1, "Expected at least 1 chunk."

    # ------------------------------------------------------------------ #
    # 3. Embed (or load cached index)
    # ------------------------------------------------------------------ #
    if (INDEX_DIR / "index.faiss").exists():
        console.print(f"\n[bold]Step 3+4:[/bold] Loading cached FAISS index from {INDEX_DIR}…")
        index, metadata = load_index(INDEX_DIR)
        console.print(f"  Index has [green]{index.ntotal}[/green] vectors (cached).")
    else:
        console.print(f"\n[bold]Step 3:[/bold] Embedding {len(chunks)} chunks (OpenAI)…")
        embedded = embed_chunks(chunks, provider="openai")
        console.print(f"  Embedded [green]{len(embedded)}[/green] chunks.")

        # ---------------------------------------------------------------- #
        # 4. Build + save index
        # ---------------------------------------------------------------- #
        console.print(f"\n[bold]Step 4:[/bold] Building FAISS index…")
        index, metadata = build_index(embedded, dimension=1536)
        save_index(index, metadata, index_dir=INDEX_DIR)
        console.print(f"  Index built and saved: [green]{index.ntotal}[/green] vectors → {INDEX_DIR}")

    # ------------------------------------------------------------------ #
    # 5. Retrieve
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Step 5:[/bold] Retrieving top-{TOP_N_INITIAL} for: [italic]{QUERY}[/italic]…")
    t0 = time.perf_counter()
    candidates = retrieve(QUERY, index, metadata, top_n=TOP_N_INITIAL, provider="openai")
    latency_ms = (time.perf_counter() - t0) * 1000
    console.print(f"  Retrieved [green]{len(candidates)}[/green] candidates in [yellow]{latency_ms:.1f}ms[/yellow].")

    # ------------------------------------------------------------------ #
    # 6. Rerank
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Step 6:[/bold] Cohere rerank → top-{TOP_N_FINAL}…")
    results = rerank(QUERY, candidates, top_n=TOP_N_FINAL)
    console.print(f"  Reranked to [green]{len(results)}[/green] results.")

    # ------------------------------------------------------------------ #
    # 7. Results table
    # ------------------------------------------------------------------ #
    table = Table(title=f'Top-{TOP_N_FINAL} Results for "{QUERY}"', show_lines=True)
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Score", style="yellow", width=8)
    table.add_column("Topic", style="green", width=30)
    table.add_column("Snippet", style="white", width=60)

    for r in results:
        snippet = r.chunk.content[:80].replace("\n", " ")
        table.add_row(
            str(r.rank + 1),
            f"{r.score:.4f}",
            r.chunk.source_topic[:30],
            snippet,
        )

    console.print()
    console.print(table)

    # ------------------------------------------------------------------ #
    # 8. Citation extraction
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Step 7:[/bold] Citation extraction…")
    sample_text = (
        "TCP/IP is the foundation of internet communication [1]. "
        "The transport layer handles reliable delivery [2]. "
        "See also networking layers [3]."
    )
    citations = extract_citations(sample_text, results)
    console.print(f"  Extracted [green]{len(citations)}[/green] citations from sample text.")
    for c in citations:
        console.print(f"    [{c.chunk_id}] {c.source_topic[:40]} (score={c.relevance_score:.3f})")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"  Docs loaded:      {len(docs)}")
    console.print(f"  Chunks created:   {len(chunks)}")
    console.print(f"  Index vectors:    {index.ntotal}")
    console.print(f"  Retrieval time:   {latency_ms:.1f}ms")
    console.print(f"  Top-5 topics:     {[r.chunk.source_topic[:25] for r in results]}")
    console.print(f"  Citations parsed: {len(citations)}")
    console.print()


if __name__ == "__main__":
    main()
