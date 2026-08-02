# V11 Retrieval-Aligned Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an independent query-driven verification and optional repair pass after every V11 session write.

**Architecture:** Add probe, retrieval, answer-check, and repair functions beside the existing V11 writer and manager. Reuse the current agent runner and workspace synchronization; retrieval operates on a disposable workspace copy so it cannot mutate the real memory, while repair uses the normal mutable V11 workspace.

**Tech Stack:** Python 3.12, existing OpenAI-compatible client, pytest, Markdown memory workspace.

## Global Constraints

- Verification is an outer Python stage, not a model tool.
- Run verification once after every written session.
- Do not expose the expected answer to the retrieval agent.
- Do not add embeddings, a fixed taxonomy, benchmark-specific rules, or a separate verification tool.
- Reuse `shell`, `save_memory`, source archives, and V11 automatic synchronization.
- Provider failures must remain visible.

---

### Task 1: Verification agent flow

**Files:**
- Modify: `src/nativemem_versions/v11/memory.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Produces: `verify_session(memory_dir, *, client, model, observation_date, turns, refs, usage_logger=None) -> dict[str, Any]`
- Uses: `_run_agent(...)`, `MemoryWorkspace`, and `_chat_completion_with_retry(...)`

- [x] **Step 1: Write failing tests**

Add tests proving that the probe result is withheld from retrieval, a successful
retrieval causes no mutation, a failed retrieval invokes repair, repair uses the
normal V11 tools, post-repair retrieval runs, and provider errors propagate.

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_v11_memory.py -k 'verify_session'
```

Expected: collection or assertion failure because `verify_session` does not
exist.

- [x] **Step 3: Implement the minimal verification flow**

Add:

```python
def verify_session(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    observation_date: str,
    turns: list[tuple[str, str]],
    refs: list[str],
    usage_logger: Any | None = None,
) -> dict[str, Any]:
    ...
```

The function generates one JSON probe, runs retrieval against a disposable
copy, checks the retrieved answer with a JSON model call, runs `_run_agent` with
a repair task only when needed, and repeats retrieval once after repair.

- [x] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_v11_memory.py -k 'verify_session'
```

Expected: all selected tests pass.

- [x] **Step 5: Run the complete V11 memory tests**

Run:

```bash
pytest -q tests/test_v11_memory.py
```

Expected: all tests pass.

### Task 2: Per-session builder integration

**Files:**
- Modify: `src/nativemem_versions/v11/adapter.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Consumes: `memory.verify_session(...)`
- Produces: V11 build order `write session -> verify session -> next session -> final manager`

- [x] **Step 1: Write a failing invocation-order test**

Replace the batch-only assertion with a test that records calls and expects:

```python
[
    ("write", session_1),
    ("verify", session_1),
    ("write", session_2),
    ("verify", session_2),
    ("manage", None),
]
```

- [x] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/test_v11_memory.py -k 'verify_each_session'
```

Expected: failure because the adapter currently batches writes and never calls
`verify_session`.

- [x] **Step 3: Integrate verification**

Change the V11 build loop to write and verify each session in order. Preserve
event counting and the final manager call. Record verification usage through
the existing usage logger.

- [x] **Step 4: Run focused and regression tests**

Run:

```bash
pytest -q tests/test_v11_memory.py
pytest -q tests/test_reanswer_longmemeval_existing_memory.py tests/test_run_v88_gpt55_longmemeval.py
```

Expected: all tests pass.

### Task 3: Verify the implementation

**Files:**
- Modify only if failures require it: `src/nativemem_versions/v11/memory.py`
- Modify only if failures require it: `src/nativemem_versions/v11/adapter.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Consumes: completed Tasks 1 and 2
- Produces: verified V11 behavior and clean diff

- [x] **Step 1: Run syntax and diff checks**

```bash
python -m py_compile src/nativemem_versions/v11/memory.py src/nativemem_versions/v11/adapter.py
git diff --check
```

Expected: both commands exit zero.

- [x] **Step 2: Run the V11 test set**

```bash
pytest -q tests/test_v11_memory.py tests/test_reanswer_longmemeval_existing_memory.py
```

Expected: all tests pass.

- [x] **Step 3: Inspect the final diff**

Confirm that verification adds no embedding dependency, no benchmark-specific
prompt, no new model tool, and no changes outside V11 implementation, V11
tests, and the approved documentation.
