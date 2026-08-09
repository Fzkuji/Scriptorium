# NativeMem Target Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved file-native NativeMem method in the V11 runtime while preserving existing benchmark entry points.

**Architecture:** `MemoryWorkspace` remains the transaction coordinator. A focused Markdown module parses and renders authoritative footnote memory units; a reconciliation module classifies Topic diffs and validates semantic reconciliation output; a runtime-state module persists provider cursors and decides incremental, local, idle, and daily triggers. Timeline, Recent, and Relations remain deterministic derivatives.

**Tech Stack:** Python 3.12, standard library, pytest, existing V11 LLM client contract, Git CLI.

## Global Constraints

- Do not modify `scripts/eval_full.py` or its locked LoCoMo protocol.
- Preserve the current `save_memory`, `write_sessions`, `manage_memory`, and query entry points.
- Source files are append-only; Topic and Core are authoritative semantic state; Timeline, Recent, and Relations are rebuildable.
- New production behavior must be introduced by a failing test first.
- Existing uncommitted user changes must be preserved.

### Task 1: Authoritative Topic Markdown

**Files:**
- Create: `code/src/nativemem_versions/v11/topic_markdown.py`
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_topic_markdown.py`

**Interfaces:**
- Produces `MemoryUnit`, `parse_topic_tree(topics: Path)`, `append_memory_unit(path, unit)`, and `render_footnote(unit, topic_path)`.
- `MemoryWorkspace` consumes parsed units instead of Timeline JSON comments.

- [ ] Write failing tests for one/multiple adjacent IDs, free-form paragraphs, stable headings/path extraction, unknown dates, duplicate IDs, and invalid/missing source definitions.
- [ ] Run the new test file and confirm failures are caused by missing parser behavior.
- [ ] Implement the minimal parser and renderer using the standard library.
- [ ] Convert `save_memory` output to footnote Markdown while keeping its structured input.
- [ ] Run the new tests and existing V11 memory tests.

### Task 2: Deterministic Derived Views

**Files:**
- Create: `code/src/nativemem_versions/v11/derived_views.py`
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_derived_views.py`

**Interfaces:**
- Consumes parsed `MemoryUnit` values.
- Produces Timeline files, Recent FIFO state, link/backlink data, and compact structure maps.

- [ ] Write failing tests for Timeline date/source/ID ordering, `undated`, Recent creation-order stability, moved Topic links, and full rebuild from Topic.
- [ ] Confirm the tests fail against the current Timeline catalog implementation.
- [ ] Implement derived-view generation and replace Timeline JSON as the catalog authority.
- [ ] Run new and existing V11 tests.

### Task 3: Topic Diff and Automatic Reconciliation

**Files:**
- Create: `code/src/nativemem_versions/v11/reconciliation.py`
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_reconciliation.py`

**Interfaces:**
- Produces `TopicDiff`, `ReconciliationRequest`, `ReconciliationResult`, `classify_topic_diff(before, after)`, and `apply_reconciliation(...)`.
- `MemoryWorkspace` accepts an injectable reconciler callable for production LLM use and deterministic tests.

- [ ] Write failing tests for structural-only edits, content rewrites, new text, split/merge, missing IDs, invalid source handles, ambiguous quotes, explicit correction, one retry, and rollback.
- [ ] Confirm each test fails for absent behavior.
- [ ] Implement deterministic diff classification and validated reconciliation application.
- [ ] Invoke the Reconciler only for semantic changes; preserve IDs for structural edits.
- [ ] Run reconciliation and V11 regression tests.

### Task 4: Atomic Workspace and Git Transaction

**Files:**
- Create: `code/src/nativemem_versions/v11/runtime_state.py`
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_runtime_state.py`

**Interfaces:**
- Produces `RuntimeStateStore`, `MemoryCursor`, and `MemoryTransaction`.
- State is stored outside memory正文 under `.nativemem/runtime.json`.

- [ ] Write failing tests for atomic Topic/Core/derived installation, Source immutability, failure rollback, Git commit after installation, and cursor advancement only after commit.
- [ ] Confirm failures.
- [ ] Implement state persistence and transaction coordination with atomic file replacement and optional Git commits.
- [ ] Run runtime-state and V11 regression tests.

### Task 5: Provider Sources and Scheduling

**Files:**
- Modify: `code/src/nativemem_versions/v11/runtime_state.py`
- Modify: `code/src/nativemem_versions/v11/adapter.py`
- Test: `code/tests/test_v11_runtime_state.py`
- Test: `code/tests/test_v11_memory.py`

**Interfaces:**
- Produces provider-neutral `SourceRecord`, `should_incremental_write`, `should_local_reorganize`, and `should_global_manage`.

- [ ] Write failing tests for opaque provider IDs, message ordering, token threshold, one-hour idle flush, local batch/token thresholds, daily management with/without new commits, and messages arriving during a write.
- [ ] Confirm failures.
- [ ] Implement provider adapters and deterministic trigger decisions; leave newly arrived messages after the captured cursor for the next transaction.
- [ ] Run adapter, runtime-state, and V11 tests.

### Task 6: Integration, Documentation, and Verification

**Files:**
- Modify: `docs/method/nativemem-method.html`
- Modify: `docs/method/README.md`
- Test: `code/tests/test_document_pages.py`

**Interfaces:**
- Documents the exact implemented/experimental boundary after the code is verified.

- [ ] Run targeted V11 tests and the repository portable-layout check.
- [ ] Run the full test suite and record any pre-existing unrelated failures separately.
- [ ] Update the method implementation table only for verified behavior.
- [ ] Run HTML/link checks and `git diff --check`.
- [ ] Commit only the implementation, tests, and corresponding method documentation.
