"""Smoke test the COHERE_THROTTLE_SECONDS env-var gate on 3 queries."""
import os, time, logging
os.environ["COHERE_THROTTLE_SECONDS"] = "7"
from dotenv import load_dotenv
load_dotenv()
from src.schemas import RetrievalResult, KnowledgeChunk
from src.rag.reranker import rerank

chunks = [KnowledgeChunk(content=f"Content about topic {i}", source_topic="t", source_field="f", chunk_index=i) for i in range(5)]
results = [RetrievalResult(chunk=c, score=1.0-i*0.1, rank=i) for i, c in enumerate(chunks)]

logging.basicConfig(level=logging.WARNING)

queries = [
    "What is binary search?",
    "How do confusion matrices work?",
    "What is L2 regularization?",
]
t0 = time.monotonic()
for q in queries:
    s = time.monotonic()
    out = rerank(query=q, results=results, top_n=3)
    e = time.monotonic()
    print(f"  '{q}' took {e-s:.2f}s, returned {len(out)} top_score={out[0].score:.4f}")
total = time.monotonic() - t0
print(f"\nTotal wall-clock: {total:.2f}s (3 calls × 7s throttle = expected ~14s on cold start, ~14+API on warm)")
