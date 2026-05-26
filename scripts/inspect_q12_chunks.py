"""Verification 2: retrieve+rerank the binary-search query and dump top-5 chunks.

Used to confirm whether Cohere surfaces the same chunks for q03 (v1) and q12 (v2)
— same query text, same index, so chunks should be identical or near-identical.
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from src.rag.retriever import retrieve
from src.rag.reranker import rerank
from src.rag.indexer import load_index

os.environ["COHERE_THROTTLE_SECONDS"] = "7"

index, metadata = load_index(Path('data/rag/faiss_index'))
query = "How does binary search work and what is its time complexity?"

retrieved = retrieve(query, index, metadata, top_n=20)
reranked = rerank(query=query, results=retrieved, top_n=5)

print(f"Query: {query}\n")
print("Top-5 Cohere-reranked chunks:")
for r in reranked:
    snippet = r.chunk.content[:300].replace("\n", " ")
    print(f"  rank={r.rank}  score={r.score:.4f}  [{r.chunk.source_topic[:60]}]")
    print(f"     > {snippet}")
    print()
