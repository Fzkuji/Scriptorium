# NativeMem V11 Memory Organizer Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy LongMemEval item 83, let `gpt-4o-mini` organize its `topics/` tree without numeric organization rules, validate lossless provenance, and rerun the item against the organized copy.

**Architecture:** A standalone pilot script copies the existing memory, records a lossless content snapshot, and exposes constrained file tools to `gpt-4o-mini`. The model chooses semantic merges, renames, and entry moves; Python performs them inside `topics/` and rejects any final state that changes the multiset of non-heading memory lines or source references. Existing reanswer code performs the one-item comparison.

**Tech Stack:** Python standard library, installed OpenAI client, existing NativeMem LongMemEval adapter, pytest.

## Global Constraints

- Do not modify the source memory or existing evaluation artifacts.
- Use `gpt-4o-mini` through the existing local OpenRouter cost gateway.
- Do not impose a topic count, file-size target, minimum entry count, fixed taxonomy, or merge threshold.
- The organizer may edit only `topics/`; `timeline/` remains unchanged.
- Preserve every non-heading memory line and every `[Dx:y]` source reference.

---

### Task 1: Constrained organizer and validation

**Files:**
- Create: `scripts/run_v11_memory_organizer_pilot.py`
- Create: `tests/test_run_v11_memory_organizer_pilot.py`

**Interfaces:**
- Produces: `snapshot_topics(root: Path) -> TopicSnapshot`, `OrganizerWorkspace(root: Path)`, and `validate_topics(before: TopicSnapshot, root: Path) -> dict`.
- `OrganizerWorkspace` provides `list_tree`, `read_file`, `search_memory`, `merge_files`, `rename_file`, `move_entries`, and `delete_empty_file`; every path resolves below `root/topics`.

- [ ] **Step 1: Write failing path and preservation tests**

Create tests that build a temporary `topics/` tree, assert path traversal is rejected, merge two related files, move one exact non-heading entry, and verify `validate_topics` reports identical line and reference multisets. Add a negative test that deletes one entry and expects validation failure.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `/opt/miniconda3/bin/python3 -m pytest -q tests/test_run_v11_memory_organizer_pilot.py`

Expected: collection fails because `scripts.run_v11_memory_organizer_pilot` does not exist.

- [ ] **Step 3: Implement the minimum constrained workspace**

Use `Path.resolve()` plus `relative_to(topics.resolve())` for every tool path. Treat non-empty lines not beginning with `#` as memory entries. `merge_files` and `move_entries` move exact existing lines without rewriting them; destination writes are atomic with `tempfile.mkstemp` and `os.replace`. `snapshot_topics` stores `Counter` values for entries and `D\d+:\d+` references.

- [ ] **Step 4: Add the model loop**

Configure the installed OpenAI client with model `openai/gpt-4o-mini`, the local gateway URL, and the seven constrained tools. The system instruction asks the model to inspect and organize related content, contains no numeric organization rules, and requires `finish` when satisfied. Persist every requested tool call and result summary to `organizer_trace.jsonl`; stop safely if validation fails.

- [ ] **Step 5: Run focused tests**

Run: `/opt/miniconda3/bin/python3 -m pytest -q tests/test_run_v11_memory_organizer_pilot.py`

Expected: all tests pass.

- [ ] **Step 6: Commit the organizer**

Run: `git add scripts/run_v11_memory_organizer_pilot.py tests/test_run_v11_memory_organizer_pilot.py && git commit -m 'feat: add v11 memory organizer pilot'`

### Task 2: Item 83 pilot and single-item comparison

**Files:**
- Create: `results/formal/v11-organizer-item83-gpt4omini-20260721-r1/`
- Reuse: `scripts/reanswer_longmemeval_existing_memory.py`
- Reuse: `scripts/run_v88_gpt55_longmemeval.py`

**Interfaces:**
- Consumes the item 83 source checkpoint and copied memory.
- Produces `before_structure.json`, `after_structure.json`, `validation.json`, `organizer_trace.jsonl`, and a one-item reanswer result.

- [ ] **Step 1: Run the organizer pilot**

Run the new script with the item 83 checkpoint, the result directory, `http://127.0.0.1:63059/v1`, model `openai/gpt-4o-mini`, and the gateway placeholder API key. The script copies the memory before making any model request.

- [ ] **Step 2: Verify structural preservation**

Run the script's offline `--validate-only` mode against the completed result directory.

Expected: `valid=true`, zero missing entries, zero added entries, and zero missing source references.

- [ ] **Step 3: Rerun item 83**

Create a one-record source analysis pointing to the organized memory and invoke the existing reanswer runner with the same LongMemEval prompt, `gpt-4o-mini`, map mode `files`, map inline `128`, read context `1`, and the local gateway.

- [ ] **Step 4: Compare the answer and structure**

Report the original and organized answers, file counts, entry-per-file distribution, duplicate count, validation result, model-call usage, and all organizer operations. State explicitly that one item cannot establish benchmark-level improvement.

- [ ] **Step 5: Run regression tests**

Run: `/opt/miniconda3/bin/python3 -m pytest -q tests/test_run_v11_memory_organizer_pilot.py tests/test_reanswer_longmemeval_existing_memory.py tests/test_run_v88_gpt55_longmemeval.py`

Expected: all tests pass.
