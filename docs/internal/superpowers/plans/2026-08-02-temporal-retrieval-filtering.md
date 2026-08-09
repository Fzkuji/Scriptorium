# Temporal Retrieval Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the retrieval Agent optionally provide `date_from` and `date_to` in an existing BM25 or Embedding call while Runtime validates partial dates and filters evidence-level candidates by interval overlap.

**Architecture:** Topic Markdown remains unchanged. The lexical parser emits one retrieval event per evidence-supported claim and retains all dates attached to that claim. Shared stdlib-only temporal helpers normalize `YYYY`, `YYYY-MM`, and `YYYY-MM-DD` into half-open intervals; BM25 and Embedding use the same overlap predicate before ranking.

**Tech Stack:** Python stdlib `datetime`, existing `rank_bm25`, existing NumPy embedding index, pytest.

## Global Constraints

- Do not add a database, dependency, background service, or additional LLM call.
- `date_from` and `date_to` are optional arguments in the same retrieval call.
- Omitted temporal arguments preserve the existing candidate set.
- A hard temporal filter excludes undated claims and keeps a claim when any of its evidence intervals overlaps the query interval.
- Do not collapse several discrete evidence dates into one minimum-to-maximum interval.
- Existing V8 and older Topic formats remain readable.
- Do not stage or commit because the target files already contain overlapping uncommitted work; inspect only the task-specific diff.

### Task 1: Evidence-level temporal filtering for BM25

**Files:**
- Modify: `code/tests/test_memory_bm25.py`
- Modify: `code/src/memory_bm25.py`

**Interfaces:**
- Produces: `temporal_bounds(value: str) -> tuple[date, date]`
- Produces: `event_matches_time_window(event: MemoryEvent, date_from: str | None, date_to: str | None) -> bool`
- Extends: `MemoryEvent.dates: list[str]`

- [x] Add failing tests proving that year/month/day query precision uses inclusive user-facing bounds, malformed or reversed ranges fail, undated claims are excluded only under a hard filter, and two discrete dates do not match a year between them.
- [x] Add a failing Topic fixture containing two claims with different evidence footnotes in one paragraph; assert that parsing emits separate claim events with their own dates and sources.
- [x] Run `pytest -q code/tests/test_memory_bm25.py` and confirm the new assertions fail for the old first-date/string-comparison behavior.
- [x] Implement stdlib-only half-open interval normalization and overlap filtering, bump the disposable BM25 cache schema, and parse paragraph citation groups as evidence-supported claims.
- [x] Run `pytest -q code/tests/test_memory_bm25.py` and confirm it passes.

### Task 2: Reuse temporal filtering in Embedding retrieval

**Files:**
- Modify: `code/tests/test_memory_embedding.py`
- Modify: `code/src/memory_embedding.py`

**Interfaces:**
- Consumes: `event_matches_time_window` from `src.memory_bm25`
- Extends: `MemoryEmbeddingIndex.search(query, *, top_k=10, date_from=None, date_to=None)`

- [x] Add a failing test with identical semantic vectors in different years and assert that an optional partial-year window retains only the overlapping event.
- [x] Run `pytest -q code/tests/test_memory_embedding.py` and confirm `search` rejects the new keyword arguments.
- [x] Filter event/vector pairs with the shared predicate before sorting; preserve vector reuse and the no-filter behavior.
- [x] Run `pytest -q code/tests/test_memory_embedding.py` and confirm it passes.

### Task 3: Expose optional time bounds to the retrieval Agent

**Files:**
- Modify: `code/tests/test_reanswer_longmemeval_existing_memory.py`
- Modify: `code/src/adapters/run_nativemem.py`
- Modify: `code/scripts/reanswer_longmemeval_existing_memory.py`

**Interfaces:**
- `bm25_search` and `embedding_search` accept optional `date_from` and `date_to` values in `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` format.
- The existing retrieval model call supplies those values; no time-extraction model call is added.

- [x] Add a failing integration test proving that `embedding_search` forwards both optional bounds and returns only the matching year.
- [x] Add a tool-schema behavior assertion that both retrieval tools expose the two optional parameters.
- [x] Run the focused reanswer tests and confirm the embedding path fails before implementation.
- [x] Forward both parameters to Embedding search, clarify both tool schemas, and tell the retrieval Agent to use hard dates only when the calendar range is explicit or already resolved.
- [x] Update the V8 BM25 schema and prompt with the same partial-date contract.
- [x] Run `pytest -q code/tests/test_reanswer_longmemeval_existing_memory.py code/tests/test_v8_prompt.py code/tests/test_v8_readtool.py`.

### Task 4: Method documentation and verification

**Files:**
- Modify: `docs/method/nativemem-method.html`
- Verify: `docs/method/designs/file_native_multiview_design.md`

**Interfaces:**
- Documents the exact Agent/Runtime boundary and overlap semantics implemented above.

- [x] Add the optional temporal arguments, precision rules, hard-filter guard, and evidence-level indexing to the Query-Time Access section.
- [x] Run the repository's document-page tests that cover method HTML.
- [x] Run the focused BM25, Embedding, retrieval-runner, and Topic parser tests together.
- [x] Run `git diff --check` and inspect the task-specific diff without staging unrelated changes.
