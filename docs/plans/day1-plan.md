# Day 1 Implementation Plan: P6 Torvalds Digital Clone — Foundation

## Context

P6 is a greenfield multi-agent system that creates digital clones of Linus Torvalds and Greg Kroah-Hartman. No code exists yet — only CLAUDE.md and docs/PRD.md. Day 1 establishes the foundation: project scaffolding, all data models, config loading, email parsing pipeline, and real LKML data. This is the "learn day" — CrewAI Flows understanding is validated via a proof-of-concept before any production code depends on it.

---

## Phase 1: Scaffolding + Schemas + Config (5 items)

### 1A. `scratch/flow_poc.py` — CrewAI Flows Learning Artifact
- Demonstrates `@start`, `@listen`, `@router`, `FlowState` in a self-contained script
- Mimics P6's deliver/fallback routing pattern
- **Verify:** `uv run python scratch/flow_poc.py` prints all step outputs

### 1B. `pyproject.toml` + `.env.example` + `.gitignore`
- Dependencies: crewai, pydantic, instructor, litellm, faiss-cpu, cohere, numpy, pyyaml, python-dotenv, click, rich, matplotlib, seaborn, streamlit, sentence-transformers, datasets, langchain-text-splitters
- Dev: pytest, pytest-cov, ruff
- `.env.example`: OPENAI_API_KEY, COHERE_API_KEY placeholders
- `.gitignore`: data/emails/*.mbox, data/models/, data/rag/, data/cache/, .env, __pycache__, .venv/

### 1C. Directory Structure (PRD Section 9)
```
src/{__init__,schemas,config,flow}.py
src/agents/{__init__,style_crew,rag_steps,evaluator_steps,fallback_steps}.py
src/style/{__init__,email_parser,feature_extractor,profile_builder,style_scorer}.py
src/rag/{__init__,corpus_loader,chunker,embedder,indexer,retriever,citation_extractor}.py
src/evaluation/{__init__,groundedness_scorer,confidence_scorer,evaluator}.py
src/fallback/{__init__,calendar_mock,context_summarizer,unstyled_responder}.py
src/visualization.py (empty placeholder)
src/cli.py (empty placeholder)
streamlit_app.py (project root, empty placeholder)
tests/, configs/, data/{emails,models,rag/faiss_index,evaluations,fallback_logs,cache}/
results/charts/, docs/{adr,architecture,plans}/, scratch/, scripts/
```

### 1D. `src/schemas.py` — All 11 Pydantic Models
Models from PRD Section 5a:
1. **EmailMessage** — sender, recipients, subject, body, timestamp, message_id, is_patch
2. **StyleFeatures** — 15 fields (11 base + 4 LKML-specific), float fields constrained to [0,1], `to_vector() -> np.ndarray` method
3. **StyleProfile** — leader_name, features, style_vector (np.ndarray), email_count, last_updated, alpha=0.3. Needs `ConfigDict(arbitrary_types_allowed=True)` + `field_serializer`/`field_validator` for ndarray
4. **KnowledgeChunk** — content, source_topic, source_field, chunk_index, embedding (Optional[np.ndarray])
5. **RetrievalResult** — chunk, score, rank
6. **Citation** — chunk_id, source_topic, text_snippet, relevance_score
7. **EvaluationResult** — style/groundedness/confidence/final scores, explanation, decision (Literal["deliver","fallback"]), model_validator for formula check
8. **FallbackResponse** — trigger_reason, context_summary, calendar_link, available_slots, unstyled_response (Optional)
9. **StyledResponse** — query, leader, response, evaluation, citations, fallback (Optional)
10. **LeaderComparison** — query, torvalds, kroah_hartman (both StyledResponse)
11. **CloneState** — Flow state, all fields with defaults for incremental population

### 1E. `configs/default.yaml` + `src/config.py`
- YAML config with: embedding, chunking, reranker, scoring (0.4/0.4/0.2), llm, leaders (torvalds/kroah_hartman), style settings
- `src/config.py`: Pydantic validation models (EmbeddingConfig, ChunkingConfig, ScoringConfig, etc.) + `load_config()` using `yaml.safe_load()`
- Validators: weights sum to 1.0, overlap < chunk_size, alpha in [0,1]

### Phase 1 STOP Gate
- [ ] `uv sync` succeeds
- [ ] `uv run python scratch/flow_poc.py` runs successfully
- [ ] `from src.schemas import *` works
- [ ] `from src.config import load_config; load_config()` works

---

## Phase 2: Email Parser + LKML Data + ADR-001 (3 items)

### 2A. `src/style/email_parser.py` — Mbox Parser + Cleaner
**Public API:** `parse_mbox(mbox_path, sender_filter) -> list[EmailMessage]`

**Cleaning pipeline (PRD Decision 5):**
1. Parse mbox with `mailbox.mbox()`
2. Filter by `From:` containing sender_filter (e.g. `torvalds@`)
3. `_strip_quoted_text()` — remove lines starting with `>`
4. `_remove_signatures()` — split on `\n-- \n`, handle common sign-off patterns
5. `_remove_patches()` — remove contiguous `+`/`-`/`@@`/`diff --git` blocks
6. `_remove_footers()` — mailing list footers, unsubscribe links
7. Filter < 20 words after cleaning
8. Validate: sender, date, >= 20 words body

**Encoding fallback:** UTF-8 -> Latin-1 -> `errors='replace'` (per CLAUDE.md troubleshooting)

**Key functions:**
- `parse_mbox(mbox_path, sender_filter) -> list[EmailMessage]`
- `_extract_body(msg) -> str` (encoding fallback chain)
- `_clean_body(body) -> str` (applies all cleaning steps)
- `_strip_quoted_text(text) -> str`
- `_remove_signatures(text) -> str`
- `_remove_patches(text) -> str`
- `_remove_footers(text) -> str`
- `_parse_timestamp(msg) -> datetime | None`
- `_detect_patch(subject, body) -> bool`

Uses Rich progress bar for parsing progress.

### 2B. LKML mbox Validation (files pre-downloaded by Ruby)
Ruby downloads mbox files manually via browser before Phase 2 begins. Files expected at:
- `data/emails/torvalds.mbox`
- `data/emails/kroah_hartman.mbox`

**Sonnet's job:** Validate files exist, parse them, confirm >= 200 clean emails per leader.

**Validation script** (`scripts/validate_emails.py`): check files exist, parse both, print counts, spot-check 3 samples per leader

### 2C. `docs/adr/ADR-001-crewai-flow-pattern.md`
Uses the exact P1-P5 ADR template (Title + Metadata header, then 5 content sections). First-person practitioner voice per CLAUDE.md writing rules.

**Structure:**
- **Title + Metadata** — `# ADR-001: CrewAI Flow vs Sequential vs Hierarchical`, Date, Status: Accepted, Project: P6, Category: Orchestration
- **Context** — P6 needs conditional branching (deliver vs fallback), typed state across 5 agent steps, dual-leader mode with shared retrieval. Which CrewAI orchestration pattern fits?
- **Decision** — Flow + single-agent Crew for style generation step only. Other agents are direct function calls in Flow steps. The Flow IS the PlannerAgent.
- **Alternatives Considered** — Sequential Crew (no branching), Hierarchical Crew (documented broken in prod), raw Python orchestration (lose state mgmt + @router)
- **Quantified Validation** — Lead Score Flow example from CrewAI docs, DocuSign case study (Dec 2025), Towards Data Science Nov 2025 article on hierarchical failures
- **Consequences** — lighter CrewAI footprint than expected, ChatStyleAgent is the only Crew, 4 other agents are plain functions. Risk: CrewAI Flows API is new (2024), mitigated by pinning version.

### Phase 2 STOP Gate
- [ ] `parse_mbox()` runs on both pre-downloaded mbox files
- [ ] >= 200 clean emails per leader
- [ ] Spot-check: 3 samples per leader show clean text (no quotes, patches, signatures)
- [ ] ADR-001 follows P1-P5 template (Title+Metadata, Context, Decision, Alternatives, Quantified Validation, Consequences)

---

## Phase 3: Tests + Final Verification (3 items)

### 3A. `tests/test_schemas.py` (~25 tests)
- Happy path for all 11 models
- Validation errors: out-of-range floats, invalid Literal values, formula mismatch
- `np.ndarray` serialization roundtrip
- CloneState incremental population (simulates Flow)
- Uses inline `_make_*()` builder helpers

### 3B. `tests/test_email_parser.py` (~15 tests)
- Uses synthetic mbox files created in-memory (no dependency on real data)
- Tests: sender filtering, short email exclusion, quote stripping, nested quotes, signature removal, patch removal (preserves normal text), footer removal, combined cleaning, encoding fallbacks (UTF-8, Latin-1, multipart), patch detection, timestamp parsing

### 3C. `tests/test_config.py` (~6 tests)
- Default config loads, weights sum to 1.0, invalid weights rejected, overlap >= size rejected, both leaders present, alpha in range

### Phase 3 STOP Gate (Final)
- [ ] `uv run pytest tests/ -v` — all pass
- [ ] `uv run pytest --cov=src/schemas --cov=src/style --cov=src/config --cov-report=term-missing` — >= 90%
- [ ] Commit on `feat/day1-foundation` branch
- [ ] CLAUDE.md "Current State" updated

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| np.ndarray in Pydantic | Serialization breaks | Proven P5 pattern: `ConfigDict(arbitrary_types_allowed=True)` + field_serializer/validator |
| CrewAI Flows API mismatch | POC fails | flow_poc.py validates API before production code depends on it |
| Cleaning too aggressive | < 200 emails | Lower min_words from 20 to 10, or include short acknowledgments |
| mbox encoding chaos | Parser crashes | Try/except per message, skip and log bad entries |

---

## Files Modified/Created

| File | Action |
|------|--------|
| `pyproject.toml` | Create |
| `.env.example` | Create |
| `.gitignore` | Create |
| `scratch/flow_poc.py` | Create |
| `src/schemas.py` | Create |
| `src/config.py` | Create |
| `configs/default.yaml` | Create |
| `src/style/email_parser.py` | Create |
| `src/visualization.py` | Create (empty placeholder) |
| `src/cli.py` | Create (empty placeholder) |
| `streamlit_app.py` (project root) | Create (empty placeholder) |
| `scripts/validate_emails.py` | Create |
| `docs/adr/ADR-001-crewai-flow-pattern.md` | Create |
| `tests/test_schemas.py` | Create |
| `tests/test_email_parser.py` | Create |
| `tests/test_config.py` | Create |
| `CLAUDE.md` | Update Current State |
| All `__init__.py` files | Create (empty) |
| All directories per PRD Section 9 | Create |

## Verification Contract
Before reporting Day 1 complete, run and paste actual terminal output:
1. `uv run python -c "from src.schemas import *; print('OK')"`
2. `uv run python -c "from src.config import load_config; c = load_config(); print(c.scoring.fallback_threshold)"`
3. Email counts per leader
4. `uv run pytest tests/ -x --tb=short` — all pass
5. `uv run pytest --cov=src/schemas --cov=src/style --cov=src/config --cov-report=term-missing` — >= 90%
