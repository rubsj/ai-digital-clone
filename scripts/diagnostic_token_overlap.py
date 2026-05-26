"""Diagnostic 2a + 2c: token-overlap groundedness and retrieval-gap test.

Reads scripts/diagnostic_floor.jsonl. Computes per-record:
  - production cosine groundedness (already in record)
  - token-overlap groundedness (Jaccard, sentence-vs-chunk max, mean over sentences)
Reports Pearson correlation between the two.

Then loads the production FAISS index (data/rag/faiss_index/), embeds each response,
fetches top-10 nearest chunks, computes overlap (Jaccard on chunk-content) with the
query-retrieved chunks. Reports per-record overlap percentage.
"""
from __future__ import annotations

import json
import re
import statistics as s
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import faiss  # noqa: E402

from src.rag.embedder import embed_openai  # noqa: E402

JSONL = Path("scripts/diagnostic_floor.jsonl")
INDEX_PATH = Path("data/rag/faiss_index/index.faiss")
CHUNK_META_PATH = Path("data/rag/faiss_index/metadata.json")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = {
    "the","a","an","of","to","in","is","it","and","or","for","on","with","that","this",
    "as","by","be","are","was","were","at","from","which","what","how","does","do","i",
    "you","he","she","they","we","its","their","but","not","can","will","would","should",
    "if","then","than","so","also","because","when","while","where","there","these","those",
    "more","most","much","into","over","such","one","two","both","each","other","some","any",
}


def tokenize(text: str) -> set[str]:
    toks = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in toks if t not in _STOP and len(t) > 2}


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 10]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_overlap_groundedness(response: str, chunks: list[dict]) -> float:
    sents = split_sentences(response)
    if not sents or not chunks:
        return 0.0
    chunk_tokens = [tokenize(c["content"]) for c in chunks[:5]]
    per_sent = []
    for sent in sents:
        st = tokenize(sent)
        max_j = max((jaccard(st, ct) for ct in chunk_tokens), default=0.0)
        per_sent.append(max_j)
    return float(np.mean(per_sent))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = s.mean(xs), s.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> None:
    records = [json.loads(line) for line in JSONL.read_text().splitlines() if line]
    records = [r for r in records if r.get("scores")]

    # --- 2a token-overlap ---
    print("=" * 90)
    print("STEP 2a — Token-overlap groundedness vs production cosine groundedness")
    print("=" * 90)
    print(f"{'id':4s} {'leader':14s} {'cosine':>8s} {'token':>8s} {'delta':>8s}  query topic")
    cosines, tokens = [], []
    for r in records:
        cos = r["scores"]["groundedness"]
        tok = token_overlap_groundedness(r["styled_response"], r["retrieved_chunks"])
        cosines.append(cos)
        tokens.append(tok)
        print(f"{r['id']:4s} {r['leader']:14s} {cos:8.4f} {tok:8.4f} {tok-cos:+8.4f}  {r.get('topic','')}")
    pr = pearson(cosines, tokens)
    print()
    print(f"Pearson(cosine, token-overlap) = {pr:.4f}")
    print(f"Mean cosine = {s.mean(cosines):.4f}   Mean token-overlap = {s.mean(tokens):.4f}")
    high_tok_low_cos = sum(1 for c, t in zip(cosines, tokens) if t > 0.6 and c < 0.50)
    print(f"Records with token>0.6 AND cosine<0.50: {high_tok_low_cos}/{len(records)}")

    # --- 2c retrieval-gap ---
    print()
    print("=" * 90)
    print("STEP 2c — Retrieval-gap test: response-nearest chunks vs query-retrieved chunks")
    print("=" * 90)

    index = faiss.read_index(str(INDEX_PATH))
    # The chunks.json should map FAISS rows → chunk content
    if CHUNK_META_PATH.exists():
        chunk_meta = json.loads(CHUNK_META_PATH.read_text())
    else:
        # Find an alternative metadata path
        cand = list(Path("data/rag/faiss_index").glob("*.json"))
        print(f"chunks.json not found; alternative json files in index dir: {cand}")
        return
    print(f"Index ntotal={index.ntotal} dim={index.d}; metadata entries={len(chunk_meta)}")

    overlaps = []
    for r in records:
        response = r["styled_response"]
        if not response:
            continue
        # Embed response in one call, get top-10
        vec = embed_openai([response])
        D, I = index.search(np.asarray(vec, dtype=np.float32), 10)
        nearest_content = []
        for idx in I[0]:
            if 0 <= idx < len(chunk_meta):
                nearest_content.append(chunk_meta[idx].get("content","")[:400])
        # Compare via content-prefix match (rough but stable across chunk reordering)
        query_contents = [c["content"][:400] for c in r["retrieved_chunks"][:5]]
        overlap = sum(1 for nc in nearest_content[:5] if any(nc == qc for qc in query_contents))
        overlaps.append(overlap)
        print(f"{r['id']:4s} {r['leader']:14s} response-nearest-top5 ∩ query-retrieved-top5 = {overlap}/5")
    if overlaps:
        print()
        print(f"Mean overlap (top-5 ∩ top-5): {s.mean(overlaps):.2f}/5")
        print(f"Records with overlap >= 3/5: {sum(1 for o in overlaps if o >= 3)}/{len(overlaps)}")


if __name__ == "__main__":
    main()
