# Resumable Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing LongMemEval re-answer runner survive transient API failures, stop promptly, and continue automatically from durable per-item results.

**Architecture:** Keep the existing per-item atomic JSON layout. Add one reusable request retry policy, validate existing item records before treating them as complete, and make the executor stop submitting work after an interruption while preserving completed futures.

**Tech Stack:** Python standard library, OpenAI-compatible client exceptions, pytest.

## Global Constraints

- Do not add dependencies or a second checkpoint format.
- Retry only transient connection, timeout, HTTP 429, and HTTP 5xx failures.
- Keep permanent HTTP 4xx failures visible.
- Preserve completed item files atomically and derive `results.json` from them.

### Task 1: Request retry policy

**Files:**
- Modify: `src/v11_memory.py`
- Test: `tests/test_v11_memory.py`

**Interfaces:**
- Produces: `_chat_completion_with_retry(create, **kwargs)` that retries transient errors until success and propagates permanent errors.

- [ ] Add tests using a callable that raises connection errors before succeeding and an exception carrying HTTP 400.
- [ ] Run `pytest tests/test_v11_memory.py -q` and verify the new tests fail for the current policy.
- [ ] Extend `_chat_completion_with_retry` with transient-error classification and interruptible bounded exponential delays.
- [ ] Run `pytest tests/test_v11_memory.py -q` and verify all tests pass.

### Task 2: Durable continuation and result validation

**Files:**
- Modify: `scripts/reanswer_longmemeval_existing_memory.py`
- Test: `tests/test_reanswer_longmemeval_existing_memory.py`

**Interfaces:**
- Produces: `load_completed_results(output_dir, dataset, sources)` returning only identity-valid completed records.
- Produces: aggregate `results.json` reconstructed atomically from validated item records.

- [ ] Add tests proving valid existing items are skipped and malformed or conflicting records are rejected.
- [ ] Run the focused tests and verify they fail because the validation helper does not exist.
- [ ] Implement the smallest loader and replace the unvalidated directory scan.
- [ ] Run the focused tests and verify they pass.

### Task 3: Prompt interruption

**Files:**
- Modify: `scripts/reanswer_longmemeval_existing_memory.py`
- Test: `tests/test_reanswer_longmemeval_existing_memory.py`

**Interfaces:**
- Produces: `run_pending(...)` that bounds submitted futures, stops submission on `KeyboardInterrupt`, cancels pending work, and returns completed records plus interruption state.

- [ ] Add a deterministic test where one item completes and interruption occurs before later work is submitted.
- [ ] Run the focused test and verify it fails because the execution helper does not exist.
- [ ] Implement incremental future submission and executor shutdown with `cancel_futures=True`.
- [ ] Make `main()` persist the aggregate after every completion and return exit code 130 when interrupted.
- [ ] Run `pytest tests/test_reanswer_longmemeval_existing_memory.py -q` and verify all tests pass.

### Task 4: Regression verification

**Files:**
- Verify: `src/v11_memory.py`
- Verify: `scripts/reanswer_longmemeval_existing_memory.py`

- [ ] Run `pytest tests/test_v11_memory.py tests/test_reanswer_longmemeval_existing_memory.py tests/test_run_v88_gpt55_longmemeval.py -q`.
- [ ] Run `python scripts/reanswer_longmemeval_existing_memory.py --help`.
- [ ] Run `git diff --check` on the modified implementation and test files.
