# Codebase Audit Checklist

Reusable six-category verification template. Per PRD §12.5 and §7.5.1.

**How to use.** Copy this file to your project's `docs/` directory. Replace every `{PLACEHOLDER}` with project-specific vocabulary before running. Capture your run's findings in your project's `docs/session-notes/dayN.md` using the output format at the bottom of this file. The template stays vocabulary-agnostic; the session notes carry the project-specific results.

**Scope.** Verification, not cleanup. Each finding resolves to one of three actions: delete, fix, or defer. Refactoring is out of scope — if dead code is entangled, delete it, not the live code around it. Time budget: 1–1.5 hours.

---

## Vocabulary parameters

Fill these in before running. They define the project-specific patterns the greps search for.

```
V1_CLASS_NAMES   = {comma-separated old class/file names, e.g. ChatStyleAgent, RAGAgent, PlannerAgent}
V1_FILE_STEMS    = {comma-separated old module stems, e.g. rag_agent, style_crew, evaluator_steps}
V1_THRESHOLD     = {the old numeric threshold, e.g. 0.75}
V1_SCORE_FIELD   = {the old computed field name, e.g. final_score}
V1_FORMULA_WORDS = {words identifying the old routing formula, e.g. weighted.*formula}
LOCK_TOKENS      = {locked-decision identifiers that must STOP rather than be fixed, e.g. HHEM,0.40,floors}
```

---

## Category 1: Dead code from old architecture

### 1a. vulture — unused functions, classes, imports

```bash
uv tool run vulture src/ --min-confidence 80
```

Decision rule: if a Python file in `src/` has zero importers elsewhere, delete it. If a function or class is defined but never called, delete it. If vulture flags a language-required parameter (e.g. `cls` in `@classmethod`) that is unused by the body, that is a false positive — record it, no action.

False-positive watchlist (common patterns vulture misidentifies):
- `@classmethod` / `@field_validator` `cls` parameters (Pydantic v2)
- `@staticmethod` first params
- `__all__` export lists
- Protocol/ABC abstract method stubs

### 1b. pytest coverage — uncovered branches as dead-code signal

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Decision rule: uncovered lines are a signal, not a verdict. For each module with coverage below 80%, determine whether the gap is: (a) a dead architectural remnant → delete; (b) an error/edge-case branch → acceptable; (c) vendored/locked code → document-and-defer.

---

## Category 2: Dead documentation references

### 2a. References to retired v1 file paths

```bash
grep -rn "{V1_FILE_STEMS}" docs/ --include="*.md"
```

Decision rule: if a doc reference points to a v1 file that no longer exists in `src/`, or names a v1 concept that the v2 architecture replaced, fix the doc. Historical plan files and session-note audit trails are exempt — they are the record of the retirement, not broken references.

Exemption pattern: a match inside a plan file (`docs/plans/`), session note (`docs/session-notes/`), or a §12.2-style v1-to-v2 mapping table is expected historical documentation. Only matches in current-facing docs (README, evaluation reports, ADRs) require fixes.

### 2b. References to obsoleted ADRs or superseded thresholds

```bash
grep -rn "{OBSOLETED_ADR_REFS}" docs/
```

Decision rule: same as 2a.

### 2c. Architecture docs — old agent names

```bash
ls docs/architecture/ 2>/dev/null || echo "directory does not exist"
grep -rn "{V1_CLASS_NAMES}" docs/architecture/ 2>/dev/null
```

Decision rule: any diagram or companion doc naming a v1 class/agent that has been reclassified must be updated or deleted.

---

## Category 3: v1 vocabulary leaks

### 3a. Old class and file names in source

```bash
grep -rn "{V1_CLASS_NAMES}\|{V1_FILE_STEMS}" src/ --include="*.py"
```

Decision rule: all findings are bugs. Fix them.

### 3b. Old computed-field names

```bash
grep -rn "{V1_SCORE_FIELD}" src/ --include="*.py"
```

Decision rule: a match in a docstring that *explains why the field was removed* is not a leak — it is anti-v1 documentation. A match in live code (assignment, comparison, schema field) is a bug. Fix live-code matches.

### 3c. Old threshold values

```bash
grep -rn "{V1_THRESHOLD}" src/ --include="*.py"
```

Decision rule: review each match. False-positive patterns: a comment explaining the historical drift toward the old threshold (e.g. "the LLM drifted to X in practice") is load-bearing context, not a vocabulary leak. A live numeric literal used in a comparison or configuration is a bug.

### 3d. Old routing formula vocabulary

```bash
grep -rn "{V1_FORMULA_WORDS}" src/ --include="*.py"
```

Decision rule: any grep match that appears in an anti-v1 explanation ("routing is NOT a weighted formula") is not a leak. A match in a live conditional or config is a bug.

---

## Category 4: Orphaned data files

### 4a. Embedding and model caches

```bash
ls -la data/cache/
```

For each file, run a zero-reader check:

```bash
grep -rn "{CACHE_FILENAME}" src/ tests/ scripts/
```

Decision rule: if the pipeline never reads or writes a data file and the zero-reader grep returns nothing, delete it. For large files (>50MB), note the size in the session note.

### 4b. Evaluation results

```bash
ls -la results/
```

Decision rule: keep files cited by any ADR evidence claim. Keep files produced by runs that answer a question no other file answers (especially isolated-variable measurements). Archive rather than delete audit-trail files. See the project's `results/MANIFEST.md` for the question-to-file mapping.

### 4c. Pipeline-adjacent data directories

```bash
find data/ -type d
```

For each directory not referenced in `src/` or `tests/`, determine whether it is: (a) empty → safe to delete; (b) an experiment artifact → document-and-defer; (c) live pipeline input → keep.

---

## Category 5: Stale comments and docstrings

### 5a. Comments referencing v1 concepts

```bash
grep -rn "# .*{V1_SCORE_FIELD}\|# .*{V1_THRESHOLD}\|# .*{V1_FORMULA_WORDS}" src/
```

Decision rule: fix the comment to match the current code, or delete if no longer relevant. Exception: a comment that documents *why a v1 approach was abandoned* is load-bearing (WHY comment) — do not delete it.

### 5b. Docstrings referencing v1 concepts

```bash
grep -rn '""".*{V1_SCORE_FIELD}\|""".*{V1_THRESHOLD}' src/
```

Decision rule: same as 5a.

### 5c. Outstanding TODOs and FIXMEs

```bash
grep -rn "TODO\|FIXME\|XXX" src/
```

Decision rule: for each hit, decide one of: resolved (delete the comment), kept with date (annotate with today's date and a one-line status), or moved to Post-Portfolio Followups. TODOs in vendored or locked-decision files are document-and-defer — route to the project owner, do not touch vendor code.

---

## Category 6: Unused dependencies

### 6a. Inspect declared dependencies

```bash
cat pyproject.toml
```

### 6b. Verify each dependency is imported

For each package in `[project] dependencies`, run:

```bash
grep -rn "^import {pkg_import_name}\|^from {pkg_import_name}" src/ tests/
```

Note: some packages have import names that differ from their install names (e.g. `python-dotenv` imports as `dotenv`; `pyyaml` imports as `yaml`; `faiss-cpu` imports as `faiss`; `sentence-transformers` imports as `sentence_transformers`). Also check for lazy imports inside function bodies with `grep -rn "import {pkg_import_name}"` (without the `^` anchor).

Decision rule: if a package is in `pyproject.toml` but not imported anywhere in `src/` or `tests/`, remove it from `pyproject.toml` and run `uv sync` to verify the project still builds. Exceptions:
- Indirect / transitive dependencies explicitly listed for version pinning — document-and-defer rather than blindly remove; test removal first.
- Packages used only in `scripts/` (not `src/` or `tests/`) — decide whether scripts warrant a separate optional group or whether the dep should remain.
- Packages required by locked-decision code (e.g. a vendored model's tokenizer) — document-and-defer; route to project owner.

---

## Audit output format

For each category, record in `docs/session-notes/dayN.md`:

1. The exact command run (copy-paste from above with placeholders filled in)
2. The raw output captured (verbatim; "no output" is recorded as "no findings" with the command that produced it — clean output is the proof)
3. For each finding: the decision (delete / fix / defer) and a one-line basis
4. For any proposed delete: the zero-importer grep command and its output confirming zero readers

Empty output is data. Recording "no findings" with the command establishes that the check ran and passed, not that it was skipped.

---

## Stop-gate rules

- **Destructive deletes** (files, dependencies): stop-gated. Paste the zero-reader grep before any deletion; suite must be green after each.
- **Locked-decision findings**: do not fix. Surface to the project owner and document-and-defer. A finding that implicates a locked design decision (identified in {LOCK_TOKENS}) is a STOP-and-surface, not a fix license.
- **Non-destructive fixes** (removing unused imports, fixing comments, updating stale default arguments): may proceed without a stop gate, but record them in the session notes.
