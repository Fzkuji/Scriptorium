# LLM Self-Review Retrieval Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure two generic LLM-directed retrieval changes on existing LongMemEval memories.

**Architecture:** Add optional prompt and same-model review hooks to the existing LongMemEval retrieval loop while preserving its default behavior. A small experiment entrypoint selects the frozen 15-item pilot and delegates storage, concurrency, and resume behavior to the existing re-answer runner.

**Tech Stack:** Python 3, existing OpenAI-compatible client, pytest, existing unified LongMemEval evaluator.

## Global Constraints

- Do not rebuild memory.
- Do not use embeddings or task-specific retrieval rules.
- Keep file map mode `files`, inline limit `128`, and read context `1`.
- Use GPT-4o-mini for retrieval and the official LongMemEval judge.
- Run variants in separate processes so prompt state cannot cross variants.

### Task 1: Optional same-model review hook

**Files:**
- Modify: `scripts/run_v88_gpt55_longmemeval.py`
- Modify: `tests/test_run_v88_gpt55_longmemeval.py`

**Interfaces:**
- Consumes: existing `collect_and_answer_longmemeval(backend, item, memory_dir, turn_index)` callers.
- Produces: optional keyword arguments `prompt_template: str | None` and `review_prompt: str | None`; defaults preserve current behavior.

- [ ] Add a failing unit test using a fake client that first emits a draft and then emits a reviewed answer.
- [ ] Run `pytest -q tests/test_run_v88_gpt55_longmemeval.py` and confirm the new test fails.
- [ ] Add optional prompt selection and one review turn. Append the first draft as an assistant message and the review request as a user message; keep tools enabled for the review turn.
- [ ] Run `pytest -q tests/test_run_v88_gpt55_longmemeval.py` and confirm all tests pass.

### Task 2: Pilot runner and online evaluation

**Files:**
- Create: `scripts/experiment_llm_self_review_longmemeval.py`
- Create: `tests/test_experiment_llm_self_review_longmemeval.py`
- Create at runtime: `results/formal/gpt4omini-longmemeval-self-review-pilot-20260721-r1/`

**Interfaces:**
- Consumes: existing `scripts.reanswer_longmemeval_existing_memory` runner and the optional hooks from Task 1.
- Produces: per-variant `results.json`, official judge JSON, and a comparison JSON.

- [ ] Add a failing test that verifies the exact 15-item selection and that each variant installs only its declared prompt/review behavior.
- [ ] Run `pytest -q tests/test_experiment_llm_self_review_longmemeval.py` and confirm failure before the script exists.
- [ ] Implement the entrypoint by patching the optional hooks in-process and delegating all item loading, memory validation, concurrency, checkpoint writing, and resume handling to the existing re-answer runner.
- [ ] Run the focused tests and confirm they pass.
- [ ] Launch `reflective-prompt` and `same-model-review` as separate concurrent processes with 15 workers each.
- [ ] Score both `results.json` files using `python -m src.evaluation.evaluate --benchmark longmemeval --metrics judge --skip-answerer` through the existing OpenRouter gateway.
- [ ] Compare each variant against the baseline on wrong-item recovery and six-control retention. Run the better safe variant on all 30 items only if the pilot decision rule passes.
