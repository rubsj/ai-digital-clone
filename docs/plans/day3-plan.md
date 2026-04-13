# Day 3 Plan: RAGAgent — Knowledge Retrieval Pipeline

## Context

Day 1 (foundation: schemas, email parser, config — 78 tests) and Day 2 (style pipeline: feature extractor, profile builder, scorer — 207 tests, PR #2 merged) are complete. Day 3 builds the RAG pipeline: corpus loading, chunking, embedding, FAISS indexing, retrieval, Cohere reranking, and citation extraction. This is the second of five "agents" — a deterministic pipeline (no CrewAI Crew, just function calls) that the Flow orchestrates on Day 5. The `RAGAgent.retrieve(query)` method is the single entry point the Flow calls.

**Pre-flight:** Pull main (PR #2 merged), create `feat/day3-rag-pipeline` branch, verify 207 tests pass.

---

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/rag/__init__.py` | Modify stub (add re-exports) | 4 |
| `src/rag/corpus_loader.py` | Fill stub | 1 |
| `src/rag/chunker.py` | Fill stub | 1 |
| `tests/test_corpus_loader.py` | New | 1 |
| `tests/test_chunker.py` | New | 1 |
| `src/rag/embedder.py` | Fill stub | 2 |
| `src/rag/indexer.py` | Fill stub | 2 |
| `tests/test_embedder.py` | New | 2 |
| `tests/test_indexer.py` | New | 2 |
| `src/rag/retriever.py` | Fill stub | 3 |
| `src/rag/reranker.py` | **New file** (no stub) | 3 |
| `src/rag/citation_extractor.py` | Fill stub | 3 |
| `tests/test_retriever.py` | New | 3 |
| `tests/test_reranker.py` | New | 3 |
| `tests/test_citation_extractor.py` | New | 3 |
| `src/agents/rag_agent.py` | **New file** | 4 |
| `scripts/test_rag_pipeline.py` | New | 4 |
| `docs/adr/ADR-002-rag-config-embeddings-reranking-chunking.md` | New | 4 |

Note: `src/agents/rag_steps.py` stub exists but stays empty — Day 5 may use it for Flow step wrappers.

---

## Phase 1: Corpus Loader + Chunker (No API Calls)

### Pre-flight: Verify dataset schema

Before writing any code, run this to confirm column names, field filter values, and CS doc count:
```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('open-phi/textbooks', split='train')
print('Columns:', ds.column_names)
print('Fields:', set(ds['field']))
cs = ds.filter(lambda x: x['field'] == 'computer_science')
print(f'CS docs: {cs.num_rows}')
print('Sample subfields:', set(list(cs['subfield'])[:20]))
"
```

**Verified output** (2026-04-13):
```
Columns: ['topic', 'model', 'concepts', 'outline', 'markdown', 'field', 'subfield', 'rag']
Fields: {'chemistry', 'mechanical_engineering', 'physics', ..., 'computer_science', ...}
CS docs: 1511
Sample subfields: {'human-computer_interfaces', 'software_design_and_engineering',
  'computer_networks', 'algorithms_and_data_structures', 'artificial_intelligence',
  'programming_languages', 'graphics_and_visualization', 'computer_design_and_engineering',
  'data_mining'}
```
**Key differences from plan assumptions:**
- Filter `"computer_science"` ✅ confirmed correct
- `topic` column exists directly — use it instead of parsing `outline`. `_extract_topic` still parses `outline` as fallback if `topic` is empty.
- 1511 CS docs → well above 900-chunk target

### `src/rag/corpus_loader.py`

**Imports:** `datasets.load_dataset`, `rich.progress.Progress`

**Internal dataclass** (not Pydantic — internal pipeline only):
```python
@dataclass
class RawDocument:
    text: str          # from markdown field
    topic: str         # from outline (first heading) or subfield
    field: str         # "computer_science"
    subfield: str      # e.g. "algorithms_and_data_structures"
```

**Functions:**
```python
def load_corpus(
    dataset_name: str = "open-phi/textbooks",
    field_filter: str = "computer_science",
    max_docs: int | None = None,
) -> list[RawDocument]:
    """Load HuggingFace dataset, filter by field. Rich progress bar."""

def _extract_topic(outline: str) -> str:
    """Parse first heading from outline string."""
```

**Key details:**
- Dataset has columns: `model`, `concepts`, `outline`, `markdown`, `field`, `subfield`, `rag`
- Filter: `field == "computer_science"` (underscore, not space)
- Text comes from `markdown` column (full textbook content, ~129K chars avg)
- Topic from `outline` column (first heading), fallback to `subfield`
- `max_docs` parameter: essential for tests (5 docs) and fast dev (20 docs)
- Rich progress bar over iteration

### `src/rag/chunker.py`

**Imports:** `langchain_text_splitters.RecursiveCharacterTextSplitter`, `langchain_text_splitters.MarkdownHeaderTextSplitter`, `src.schemas.KnowledgeChunk`, `src.config.AppConfig`

**Functions:**
```python
def chunk_baseline(
    documents: list[RawDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    """RecursiveCharacterTextSplitter. Global sequential chunk_index."""

def chunk_semantic(
    documents: list[RawDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    """MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter.
    Split on headers first, then size-split oversized sections."""

def chunk_documents(
    documents: list[RawDocument],
    config: AppConfig,
    strategy: str = "baseline",
) -> list[KnowledgeChunk]:
    """Dispatch to baseline or semantic. Reads chunk_size/overlap from config."""
```

**Key details:**
- `chunk_index` is globally sequential (0, 1, 2, ..., N) across all documents
- `source_topic` from `RawDocument.topic` (baseline) or header metadata (semantic)
- `source_field` always `RawDocument.field`
- `embedding` left as `None` (Phase 2 sets it)
- MarkdownHeaderTextSplitter: `headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]`
- Target: >900 chunks (20 docs at 500-char chunks will produce thousands)

### Tests

**`tests/test_corpus_loader.py`** (~8 tests):
- Mock `datasets.load_dataset` — never call HuggingFace in unit tests
- Test filtering returns only CS docs
- Test `_extract_topic` parses headings correctly
- Test `max_docs` caps output
- Test empty dataset returns empty list

**`tests/test_chunker.py`** (~15 tests):
- `_make_raw_doc()` builder helper
- Test `chunk_baseline`: known text → expected chunk count, content preserved
- Test `chunk_semantic`: markdown with headers → topics from headers
- Test `chunk_index` is globally sequential across multiple docs
- Test all chunks are `KnowledgeChunk` with `embedding is None`
- Test chunk_overlap: adjacent chunks share content
- Test empty doc, single-word doc, doc shorter than chunk_size
- Test `chunk_documents` dispatches correctly

### STOP Gate 1
```
pytest tests/test_corpus_loader.py tests/test_chunker.py -v  # all green, no API calls
```

---

## Phase 2: Embedder + Indexer (OpenAI API, Mocked in Tests)

### `src/rag/embedder.py`

**Imports:** `hashlib`, `json`, `pathlib.Path`, `numpy`, `litellm`, `sentence_transformers.SentenceTransformer`, `rich.progress.Progress`, `src.schemas.KnowledgeChunk`

**Functions:**
```python
def _md5(text: str) -> str:
    """MD5 hash for cache key."""

def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    """Load JSON cache. Empty dict if missing."""

def _save_cache(cache: dict[str, list[float]], cache_path: Path) -> None:
    """Persist cache to JSON."""

def embed_openai(
    texts: list[str],
    cache_path: Path = Path("data/cache/embeddings_openai.json"),
    batch_size: int = 100,
) -> list[np.ndarray]:
    """LiteLLM text-embedding-3-small. Cached by MD5. L2-normalized. Rich progress bar."""

def embed_minilm(
    texts: list[str],
    cache_path: Path = Path("data/cache/embeddings_minilm.json"),
) -> list[np.ndarray]:
    """SentenceTransformers all-MiniLM-L6-v2. Cached by MD5. L2-normalized."""

def embed_chunks(
    chunks: list[KnowledgeChunk],
    provider: Literal["openai", "minilm"] = "openai",
    cache_path: Path | None = None,
) -> list[KnowledgeChunk]:
    """Return NEW KnowledgeChunk instances with embedding set.
    Uses chunk.model_copy(update={"embedding": vec}) — no in-place mutation."""

def embed_query(
    query: str,
    provider: Literal["openai", "minilm"] = "openai",
    cache_path: Path | None = None,
) -> np.ndarray:
    """Embed single query. Returns normalized vector."""
```

**Cache flow:**
1. Load cache from JSON
2. Compute MD5 for each text
3. Separate cached vs uncached
4. Batch uncached (groups of 100 for OpenAI)
5. `litellm.embedding(model="text-embedding-3-small", input=batch)` → `response.data[i].embedding`
6. L2-normalize: `vec / np.linalg.norm(vec)` — normalize BEFORE caching
7. Store in cache dict, save to disk
8. Return all embeddings in original order

**MiniLM:** `SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)` — loaded once at module level.

Two cache files: `data/cache/embeddings_openai.json`, `data/cache/embeddings_minilm.json`.

### `src/rag/indexer.py`

**Imports:** `faiss`, `json`, `numpy`, `pathlib.Path`, `src.schemas.KnowledgeChunk`

**Functions:**
```python
def build_index(
    chunks: list[KnowledgeChunk],
    dimension: int = 1536,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Build IndexFlatIP. L2-normalize with faiss.normalize_L2() before add().
    Validate norms ≈ 1.0. Returns (index, metadata_list)."""

def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    index_dir: Path = Path("data/rag/faiss_index"),
) -> None:
    """faiss.write_index() + metadata.json sidecar."""

def load_index(
    index_dir: Path = Path("data/rag/faiss_index"),
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """faiss.read_index() + load metadata.json."""

def _validate_norms(embeddings: np.ndarray, tol: float = 1e-5) -> None:
    """Assert all L2 norms ≈ 1.0. Raises ValueError if not."""
```

**Key details:**
- Extract embeddings: `np.array([c.embedding for c in chunks], dtype=np.float32)` — FAISS uses float32
- `faiss.normalize_L2(embeddings)` BEFORE `index.add(embeddings)` — CRITICAL
- Metadata sidecar: `[chunk.model_dump(exclude={"embedding"}) for chunk in chunks]`
- Save: `faiss.write_index(index, str(path / "index.faiss"))` + `json.dump(metadata)`
- `index_dir.mkdir(parents=True, exist_ok=True)` on save

### Tests

**`tests/test_embedder.py`** (~15 tests):
- `@patch("src.rag.embedder.litellm")` — mock all API calls
- `@patch("src.rag.embedder.SentenceTransformer")` for MiniLM
- Test cache: call twice with same text → API called once (mock call count)
- Test cache file creation/loading with `tmp_path`
- Test L2 normalization: `np.linalg.norm(result) ≈ 1.0`
- Test batch splitting: 250 texts / batch_size=100 → 3 API calls
- Test `embed_chunks` returns new chunks with embedding set (originals unchanged)
- Test `embed_query` returns single normalized vector
- Test empty input returns empty list

**`tests/test_indexer.py`** (~12 tests):
- Create chunks with random normalized 1536-d embeddings
- Test `build_index`: `index.ntotal == len(chunks)`
- Test `_validate_norms`: passes for normalized, raises for unnormalized
- Test `save_index` + `load_index` roundtrip with `tmp_path`
- Test search: query with known vector, verify correct results returned
- Test metadata sidecar: contains chunk fields, no embedding

### STOP Gate 2
```
pytest tests/test_embedder.py tests/test_indexer.py -v  # all green, APIs mocked
```

---

## Phase 3: Retriever + Reranker + Citation Extractor

### `src/rag/retriever.py`

**Functions:**
```python
def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    top_n: int = 20,
    provider: Literal["openai", "minilm"] = "openai",
) -> list[RetrievalResult]:
    """Embed query → FAISS search → return top-N RetrievalResults."""
```

**Key details:**
- `query_vec = embed_query(query, provider=provider)` — already normalized
- `query_vec.reshape(1, -1).astype(np.float32)` — FAISS expects 2D float32
- `faiss.normalize_L2(query_vec_2d)` — belt-and-suspenders
- `scores, indices = index.search(query_vec_2d, top_n)`
- Reconstruct `KnowledgeChunk` from `metadata[idx]` (without embedding)
- Filter out -1 indices (FAISS padding)
- `top_n = min(top_n, index.ntotal)`

### `src/rag/reranker.py` (NEW FILE)

**Functions:**
```python
def rerank(
    query: str,
    results: list[RetrievalResult],
    model: str = "rerank-english-v3.0",
    top_n: int = 5,
) -> list[RetrievalResult]:
    """Cohere Rerank API: top-20 → top-5.
    Graceful fallback: if API fails, return original top-5 with warning."""
```

**Key details:**
- `client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY", ""))` — or check if `cohere.Client` is the right class for installed version
- `documents = [r.chunk.content for r in results]`
- `response = client.rerank(model=model, query=query, documents=documents, top_n=top_n)`
- `response.results` → list with `.index` (int) and `.relevance_score` (float)
- Map back: build new `RetrievalResult(chunk=results[item.index].chunk, score=item.relevance_score, rank=i)`
- Fallback: `try/except Exception` → log warning → return `results[:top_n]`

### `src/rag/citation_extractor.py`

**Functions:**
```python
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citations(
    text: str,
    retrieved: list[RetrievalResult],
) -> list[Citation]:
    """Parse [N] refs (1-indexed in text), map to retrieved chunks.
    Deduplicates. Skips out-of-range refs."""
```

**Key details:**
- `re.findall(r"\[(\d+)\]", text)` → list of number strings
- 1-indexed: `[1]` → `retrieved[0]`
- Skip if `idx < 0 or idx >= len(retrieved)`
- Deduplicate by tracking seen indices
- `chunk_id = f"chunk_{chunk.chunk_index}"`
- `text_snippet = chunk.content[:100]`
- `relevance_score = min(max(result.score, 0.0), 1.0)` — clamp for Pydantic

### Tests

**`tests/test_retriever.py`** (~10 tests):
- Build small FAISS index (10 chunks, random vectors) in fixture
- Mock `embed_query` to return known vector
- Test returns correct count, sorted by score, RetrievalResult fields populated
- Test `top_n > index.ntotal` doesn't crash

**`tests/test_reranker.py`** (~10 tests):
- Mock `cohere.ClientV2` entirely
- Mock response with known indices and scores
- Test reduces 20 results to 5
- Test rank reassignment (0-4)
- Test fallback: mock Cohere raising exception → returns original top-5
- Test empty results list

**`tests/test_citation_extractor.py`** (~10 tests):
- Test `[1]`, `[2]`, `[3]` → correct chunks mapped
- Test out-of-range `[99]` skipped
- Test no citations → empty list
- Test duplicate `[1] ... [1]` → deduplicated
- Test Citation fields match schema
- Test text with no brackets → empty list

### STOP Gate 3
```
pytest tests/test_retriever.py tests/test_reranker.py tests/test_citation_extractor.py -v  # all green
```

---

## Phase 4: RAGAgent Facade + E2E Script + ADR-002

### `src/agents/rag_agent.py` (NEW FILE)

```python
class RAGAgent:
    """Facade for the full RAG pipeline. NOT a CrewAI Crew.
    The Flow calls agent.retrieve(query) → list[RetrievalResult]."""

    def __init__(
        self,
        config: AppConfig | None = None,
        index_dir: Path = Path("data/rag/faiss_index"),
    ) -> None:
        """Load config. Try loading index from disk (None if not built yet)."""

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Full pipeline: embed → FAISS top-20 → Cohere rerank → top-5."""

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        """Embed chunks, build FAISS index, save to disk. One-time setup."""
```

- `retrieve()` calls `faiss_retrieve()` then `rerank()`
- `build()` calls `embedded = embed_chunks(chunks)` (returns new list) then `build_index(embedded)` then `save_index()`
- Raises `RuntimeError` if `retrieve()` called before index is built

### `src/rag/__init__.py` (Modify existing stub)

Re-export the public API so consumers can `from src.rag import retrieve, rerank, ...`:
```python
from src.rag.corpus_loader import load_corpus
from src.rag.chunker import chunk_documents
from src.rag.embedder import embed_chunks, embed_query
from src.rag.indexer import build_index, save_index, load_index
from src.rag.retriever import retrieve
from src.rag.reranker import rerank
from src.rag.citation_extractor import extract_citations
```

### `scripts/test_rag_pipeline.py`

End-to-end validation (not pytest — a runnable script):
1. Load corpus (cap at ~20 docs for speed)
2. Chunk with baseline strategy → verify ≥900 chunks
3. Embed chunks (OpenAI via LiteLLM — uses cache after first run)
4. Build FAISS index, save to disk
5. Query "What is TCP/IP?" — time the retrieval
6. Rerank top-20 → top-5
7. Print results with Rich table (rank, score, topic, snippet)
8. Test citation extraction with sample text containing `[1]`, `[2]` refs
9. Print summary: chunk count, retrieval latency, top-5 topics

### `docs/adr/ADR-002-rag-config-embeddings-reranking-chunking.md`

Follow ADR-001 template (8 sections). Content:
- **Context:** RAG pipeline needs three config decisions: embedding model, chunking strategy, reranking
- **Decision:** OpenAI text-embedding-3-small primary (1536d), 500/50 baseline + semantic experiment, Cohere rerank-english-v3.0 top-20→top-5
- **Alternatives Considered:** MiniLM as primary (26% worse Recall@5 from P2 grid search), no reranking (20% worse from P2), larger chunks (1000/100)
- **Quantified Validation:** P2 data — OpenAI 26% better, Cohere 20% lift, cost analysis
- **Consequences:** API cost (~$0.02/1M tokens), Cohere free tier limit (1K/month), cache mitigates

### STOP Gate 4 (Final)
```bash
# E2E script
uv run python scripts/test_rag_pipeline.py
# → ≥900 chunks, retrieval <1s, citations parse

# RAGAgent facade exists
grep -n "def retrieve" src/agents/rag_agent.py | head -5

# All tests pass (207 existing + new)
python -m pytest tests/ -x --tb=short 2>&1 | tail -10

# Coverage on RAG modules ≥90%
python -m pytest --cov=src/rag --cov-report=term-missing 2>&1 | tail -20
```

---

## Verification Contract

Before reporting Day 3 complete, paste ACTUAL terminal output for:
- A. `python scripts/test_rag_pipeline.py` — chunk count, query results, latency
- B. `grep -n "def retrieve" src/agents/rag_agent.py | head -5` — facade exists
- C. `python -m pytest tests/ -x --tb=short 2>&1 | tail -10` — all tests pass
- D. `python -m pytest --cov=src/rag --cov-report=term-missing 2>&1 | tail -20` — ≥90% coverage

---

## Session End Protocol

1. Git add, commit, push on `feat/day3-rag-pipeline` branch
2. Open PR to main
3. Update CLAUDE.md "Current State" section
4. Write handoff: branch/commit, file:line refs, metrics, what's next (Day 4: EvaluatorAgent + FallbackAgent)
