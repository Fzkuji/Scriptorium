# NativeMem Version Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put V8, V10, and V11 implementation code in separate packages while preserving the current adapter and benchmark commands.

**Architecture:** `src/nativemem_versions/` owns one package per version. `src/adapters/run_nativemem.py` selects a version adapter and retains temporary compatibility aliases; benchmark persistence remains shared.

**Tech Stack:** Python standard library, pytest.

## Global Constraints

- Do not change memory prompts or benchmark semantics.
- Do not move result directories, checkpoints, or running processes.
- Existing imports and CLI commands must continue working.
- Version packages must not import another version, except the explicit V10-to-V8.8 compatibility dependency.

### Task 1: Add version package contracts

**Files:**
- Create: `tests/test_nativemem_version_isolation.py`
- Create: `src/nativemem_versions/__init__.py`

**Interfaces:**
- Produces: `load_version(name: str)` returning the selected adapter module.

- [ ] Write a test importing `v8`, `v10`, and `v11`, rejecting an unknown version, and verifying that each module exposes `build_memory`.
- [ ] Run the test and verify failure because the package does not exist.
- [ ] Add the minimum package dispatcher.
- [ ] Run the test and verify it passes.

### Task 2: Isolate V11

**Files:**
- Create: `src/nativemem_versions/v11/__init__.py`
- Move: `src/v11_memory.py` to `src/nativemem_versions/v11/memory.py`
- Create: `src/nativemem_versions/v11/adapter.py`
- Modify: `src/v11_memory.py`
- Modify: `src/adapters/run_nativemem.py`
- Test: `tests/test_nativemem_version_isolation.py`

**Interfaces:**
- Produces: `v11.adapter.build_memory(conv, memory_dir, max_sessions=None)`.
- Preserves: `import src.v11_memory` through a compatibility re-export.

- [ ] Add a failing test proving V11 dispatch calls only the V11 adapter.
- [ ] Move the V11 memory implementation and add the compatibility module.
- [ ] Move `_build_memory_v11` into the V11 adapter.
- [ ] Run V11, reanswer, and LongMemEval tests.

### Task 3: Isolate V10 and V8

**Files:**
- Create: `src/nativemem_versions/v10/{__init__.py,memory.py,adapter.py}`
- Create: `src/nativemem_versions/v8/{__init__.py,memory.py,adapter.py}`
- Modify: `src/v10_memory.py`
- Modify: `src/v8_memory.py`
- Modify: `src/adapters/run_nativemem.py`
- Test: `tests/test_nativemem_version_isolation.py`

**Interfaces:**
- Produces: `v8.adapter.build_memory(...)` and `v10.adapter.build_memory(...)`.
- Preserves: historical `src.v8_memory` and `src.v10_memory` imports.

- [ ] Add failing tests proving V8 and V10 dispatch select distinct adapters.
- [ ] Move each memory implementation into its version package and add compatibility re-exports.
- [ ] Move each builder into its version adapter.
- [ ] Reduce `run_nativemem.build_memory` to version selection and delegation.
- [ ] Run all NativeMem version tests and `git diff --check`.
