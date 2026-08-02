# GPT-5.6 QA Continuation Recovery Implementation Plan

**Goal:** Resume the interrupted Sol and Terra W=32 QA runs without recomputing completed questions, while preserving auditable provenance across the original and continuation generations.

**Constraints:** Keep `scripts/eval_full.py`, `AGENTS.md`, and `src/nativemem.py` unchanged. Use `/opt/miniconda3/bin/python`. Start a new parent proxy and new result roots after any source change. Treat the existing `sol-r5` and `terra-r5` roots as immutable parents.

## Task 1: Recover output present only in `response.completed`

**Files:**
- Modify: `src/chatgpt_proxy.py`
- Test: `tests/test_chatgpt_proxy.py`

1. Add failing parser tests for final-event-only `output_text` and `function_call` items.
2. Confirm that a genuinely empty completed response is still rejected.
3. Add a fallback that reads `response.output` only when streamed text/tool-call events produced no output.
4. Run `tests/test_chatgpt_proxy.py`, Ruff, and `py_compile`.

## Task 2: Preserve provider-attempt evidence on completion-cap rejection

**Files:**
- Modify: `scripts/controlled_subscription_qa_proxy.py`
- Modify: `scripts/run_controlled_locomo_answers.py`
- Modify: `scripts/audit_gpt56_chunk_curve_qa.py`
- Test: `tests/test_controlled_subscription_qa_proxy.py`
- Test: `tests/test_controlled_locomo_answers.py`
- Test: `tests/test_audit_gpt56_chunk_curve_qa.py`

1. Change the existing cap-rejection test to require preserved upstream attempt count, response ID, model, usage, and proxy metadata.
2. Add an answer-client test showing a cap rejection with known provider attempts receives the existing bounded physical retry.
3. Parse auditable upstream metadata before applying the completion-token cap check; return a machine-readable cap-rejection error with `error_code=completion_token_cap_exceeded`, `requested_completion_token_cap`, `observed_completion_tokens`, `upstream_http_attempts`, `upstream_response_id`, `upstream_response_model`, `upstream_usage`, `upstream_response_sha256`, and `ignored_client_parameters`.
4. Validate this error evidence in the answer client and formal auditor. Count the provider attempts and let only the existing answer-level physical retry continue; do not repeat retrieval or add a question-level retry.
5. Keep the cap fail-closed and keep the retry bound unchanged.
6. Run all three targeted test files, Ruff, and `py_compile`.

## Task 3: Add an auditable continuation generation

**Files:**
- Create: `scripts/gpt56_chunk_curve_qa_continuation.py`
- Modify: `scripts/run_gpt56_chunk_curve_qa.py`
- Modify: `scripts/audit_gpt56_chunk_curve_qa.py`
- Modify: `scripts/gpt56_chunk_curve_qa_contract.py`
- Test: `tests/test_gpt56_chunk_curve_qa_continuation.py`
- Test: `tests/test_run_gpt56_chunk_curve_qa.py`
- Test: `tests/test_audit_gpt56_chunk_curve_qa.py`

1. Add failing tests for importing only completed parent question directories into a fresh child root.
2. Add `--continue-from PATH` and `--expected-imported-questions N`. Record a continuation manifest that pins the parent root, parent preregistration hash, imported question IDs, parent proxy snapshots, and imported artifact tree hashes.
3. Give newly executed questions a new child `qa_run_id`; retain the parent `qa_run_id` in imported attempt artifacts. Select the expected run ID by question provenance.
4. Deterministically import every valid parent `record.json`, but do not import incomplete terminal question directories. Fail on invalid completed artifacts, links, hardlinks, parent snapshot changes, or copied-artifact tampering.
5. Extend the auditor to validate parent and child proxy events separately, classify pending-question parent events as superseded, then combine only the pinned imported parent questions with newly executed child questions.
6. Add CLI support for the explicit parent root and make the completion record report imported and newly executed counts.
7. Run continuation unit tests, the full QA contract test set, Ruff, and `py_compile`.

## Task 4: Verify and resume only missing questions

1. Verify locked evaluator hash and all modified source hashes.
2. Run the full targeted test suite and confirm the only known unrelated failures remain the frozen `src/nativemem.py` hash checks, if the repository-wide suite is run.
3. Start a fresh parent proxy on a new port with a fresh request log; verify `/healthz` with local proxy bypass.
4. Create fresh continuation roots for Sol and Terra from `sol-r5` and `terra-r5`.
5. Run preflight and verify imported counts are 332 and 324, leaving exactly 24 and 32 new questions.
6. Start both continuation runs, monitor at low frequency, and write no completion claim until each has `completion.json` and passes the formal auditor.

No commits are included in this plan because the repository contains unrelated user changes and several experiment files are currently untracked; validation will be recorded through hashes, preregistration artifacts, logs, and audit output.
