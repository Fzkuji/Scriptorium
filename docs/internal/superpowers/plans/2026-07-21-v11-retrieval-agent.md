# V11 Retrieval Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-answer LongMemEval questions from arbitrary V11 memory trees with mandatory evidence reads.

**Architecture:** Extend the existing re-answer runner with one V11 retrieval loop. Reuse the backend client and source turn index; expose only read-only file inventory, Markdown read, literal search, and source-resolution tools. Reject final answers until the agent has read non-empty memory evidence.

**Tech Stack:** Python standard library, existing OpenAI-compatible client, pytest.

## Global Constraints

- Do not stop or modify the active eight V11 builders.
- Do not rebuild memory for re-answering.
- Do not assume fixed directory names or heading depths.
- Keep every tool read-only and confined to the sample memory root.

### Task 1: Add and validate the V11 retrieval agent

**Files:**
- Modify: `scripts/reanswer_longmemeval_existing_memory.py`
- Modify: `tests/test_reanswer_longmemeval_existing_memory.py`

**Interfaces:**
- Consumes: `backend.client`, `backend.ALIYUN_MODEL`, `turn_index`, V11 memory directory.
- Produces: `collect_answer(..., prompt="v11-agent") -> (memories, steps, answer, trace)`.

- [ ] Add a failing test where the first model response tries to answer without tools and the second response reads the correct Markdown file before answering.
- [ ] Run the focused test and confirm failure because `v11-agent` is not implemented.
- [ ] Implement safe file listing, file reading, literal search, source resolution, and the mandatory-read agent loop.
- [ ] Add `v11-agent` to the CLI prompt choices.
- [ ] Run focused and existing re-answer tests.
- [ ] Re-answer item 0 with `files + inline128 + context8`; require `Business Administration` and a non-empty tool trace.
