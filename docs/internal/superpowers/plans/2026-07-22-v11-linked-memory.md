# V11 Linked Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make V11 write linked topics, timeline, sources, and a bounded recent-event view with code-owned synchronization.

**Architecture:** Extend the existing `MemoryWorkspace` instead of adding a new subsystem. Agent mutations occur in staged topics through `save_memory`, shell, and `commit`; one reconciliation path validates stable event IDs and renders every derived link before publication.

**Tech Stack:** Python standard library, existing OpenAI-compatible tool loop, pytest.

## Global Constraints

- Preserve unrestricted topic organization and Markdown heading depths `#` through `######`.
- Do not add embeddings or dependencies.
- Keep currently running processes untouched; behavior changes only for new Python processes.
- Default recent-event capacity is 100 and is configurable with `NATIVEMEM_V11_RECENT_LIMIT`.
- Code is the only writer of sources, timeline, recent events, and managed links.

---

### Task 1: Deterministic linked storage

**Files:**
- Modify: `src/v11_memory.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Produces: `MemoryWorkspace.archive_sessions(sessions)`, `MemoryWorkspace.save_memory(events)`, and `MemoryWorkspace.commit()`.
- Produces: stable topic markers and generated topic, timeline, source, and recent-event links.

- [ ] Write a failing test that archives `D52:7`, saves one event, commits it, and asserts the topic, timeline, source, and recent rows share one event ID and resolve to existing targets.
- [ ] Run `pytest tests/test_v11_memory.py -q` and confirm the new test fails because linked storage is absent.
- [ ] Implement source rendering, stable event IDs, staged topic insertion, reconciliation, a 100-row recent limit, link validation, and atomic publication using `pathlib`, `tempfile`, `shutil`, `hashlib`, and `os.replace`.
- [ ] Run `pytest tests/test_v11_memory.py -q` and confirm the linked-storage test passes.
- [ ] Commit only `src/v11_memory.py` and `tests/test_v11_memory.py` with `feat: link v11 memory views`.

### Task 2: Agent tool protocol

**Files:**
- Modify: `src/v11_memory.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Consumes: `MemoryWorkspace.save_memory(events)` and `MemoryWorkspace.commit()` from Task 1.
- Produces: tool definitions `shell`, `save_memory`, and `commit`; `_run_agent` requires a successful commit before accepting completion.

- [ ] Write failing tests showing one `save_memory` call accepts multiple events, shell operates only on staged topics, and a failed commit returns a tool error that the model can correct.
- [ ] Run the three focused tests and confirm they fail for the missing protocol.
- [ ] Replace `save_events` with `save_memory`, add `commit`, update the English prompts, and route shell to staged topics while preserving the existing retry and usage logging paths.
- [ ] Run `pytest tests/test_v11_memory.py -q` and confirm all V11 tests pass.
- [ ] Commit only `src/v11_memory.py` and `tests/test_v11_memory.py` with `feat: add transactional v11 writer tools`.

### Task 3: Build and retrieval integration

**Files:**
- Modify: `src/adapters/run_nativemem.py`
- Modify: `scripts/reanswer_longmemeval_existing_memory.py`
- Test: `tests/test_v11_memory.py`
- Test: `tests/test_reanswer_longmemeval_existing_memory.py`

**Interfaces:**
- Consumes: the linked V11 workspace from Tasks 1 and 2.
- Produces: source archival before writer calls and retrieval inventory covering topics, timeline, and recent events while hiding direct source search.

- [ ] Write failing adapter tests asserting sessions are archived before grouped writer calls and build event counts come from successful `save_memory` records.
- [ ] Write a failing retrieval test asserting source files are absent from general inventory while topic, timeline, and recent-event files are visible.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Update `_build_memory_v11` to archive prepared sessions and count saved events; update V11 retrieval inventory and read restrictions for the linked views.
- [ ] Run `pytest tests/test_v11_memory.py tests/test_reanswer_longmemeval_existing_memory.py -q` and confirm all tests pass.
- [ ] Commit only the four task files with `feat: integrate linked v11 memory`.

### Task 4: Regression verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a reproducible local verification result without API calls.

- [ ] Run `pytest tests/test_v11_memory.py tests/test_reanswer_longmemeval_existing_memory.py tests/test_run_v88_gpt55_longmemeval.py -q`.
- [ ] Run `git diff --check`.
- [ ] Inspect `git status --short` and verify no unrelated user files were staged or committed.
