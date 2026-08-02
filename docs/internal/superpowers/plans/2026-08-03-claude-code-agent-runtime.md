# Claude Code Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NativeMem's hand-written Writer, Manager, verification-retrieval, and query agent loops with the Claude Agent SDK while preserving the existing transactional memory semantics, retrieval choices, result schema, and cost accounting.

**Architecture:** A small `src/agent_runtime/` adapter starts one isolated Claude Code subprocess per agent trajectory. NativeMem exposes its existing write and retrieval operations as in-process MCP tools; Claude Code owns model turns, tool-call protocol, malformed-call handling, provider errors, and tool-result context management. Source archiving, Topic validation, atomic installation, Timeline/Recent/Relations rebuilding, BM25, Embedding, and benchmark evaluation remain NativeMem responsibilities.

**Tech Stack:** Python 3.12, `claude-agent-sdk`, Claude Code CLI, SDK MCP tools, pytest.

## Global Constraints

- All API settings enter through Python function arguments; no shell profile, global process environment, `~/.claude.json`, or `~/.claude/settings.json` is modified.
- Each Claude Code subprocess uses `--bare`, `--no-session-persistence`, `--strict-mcp-config`, and a fresh `CLAUDE_CONFIG_DIR`.
- The configured endpoint must implement the Anthropic Messages API expected by Claude Code.
- Claude Code receives no built-in tools. Only NativeMem's phase-specific MCP tools are available.
- Writer and Manager may modify only the staged workspace through the existing transactional `MemoryWorkspace.shell` operation. Source Memory remains append-only.
- Retrieval remains read-only and preserves the existing condition-specific visible-file rules.
- Native retrieval continues to offer shell/grep, BM25, and Embedding; the model chooses which operation to invoke.
- Tool output is returned directly to Claude Code. NativeMem does not compact, summarize, or parse prior tool messages.
- The default safety limit is 20 Claude Code turns per trajectory. The framework reports actual turns, tokens, duration, stop reason, and cost.
- Do not edit `code/scripts/eval_full.py`; its SHA-256 must remain `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`.
- Preserve unrelated and untracked files, especially `profile_output/`.

### Task 1: Isolated Claude Code Adapter

**Files:**
- Create: `code/src/agent_runtime/__init__.py`
- Create: `code/src/agent_runtime/claude_code.py`
- Modify: `requirements.txt`
- Test: `code/tests/agent_runtime/test_claude_code.py`

**Interfaces:**
- Produces: `ClaudeCodeConfig`, `AgentResult`, `ClaudeCodeAgent.run(...)`.
- `ClaudeCodeAgent.run` accepts the prompt, system prompt, working directory, MCP tools, maximum turns, optional JSON schema, and optional budget.

- [ ] **Step 1: Write failing tests for isolated options and result parsing**

```python
def test_agent_uses_isolated_bare_nonpersistent_claude_code(tmp_path, monkeypatch):
    result = agent.run(prompt="answer", system_prompt="system", cwd=tmp_path)
    assert result.text == "done"
    assert captured.options.extra_args == {
        "bare": None,
        "no-session-persistence": None,
        "strict-mcp-config": None,
    }
    assert captured.options.env["CLAUDE_CONFIG_DIR"] != str(Path.home() / ".claude")
    assert captured.options.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
```

- [ ] **Step 2: Run the focused test and verify it fails because the adapter does not exist**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/agent_runtime/test_claude_code.py -q`

- [ ] **Step 3: Implement the minimal synchronous adapter over SDK `query()`**

The implementation must normalize `AssistantMessage`/`ResultMessage`, raise one `AgentExecutionError` for framework-reported failure, clean the temporary configuration directory in `finally`, and never expose the API key in traces or exceptions.

- [ ] **Step 4: Run the focused test and dependency import check**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/agent_runtime/test_claude_code.py -q`

Run: `.venv/bin/python -c 'import claude_agent_sdk'`

### Task 2: Management MCP Tools and Writer Migration

**Files:**
- Create: `code/src/management/tools.py`
- Modify: `code/src/management/agent.py`
- Modify: `code/src/management/api.py`
- Modify: `code/src/management/config.py`
- Modify: `code/src/management/model_reconciliation.py`
- Modify: `code/src/management/verification.py`
- Delete: `code/src/management/provider.py`
- Test: `code/tests/management/test_memory.py`

**Interfaces:**
- Consumes: `ClaudeCodeAgent.run` and existing `MemoryWorkspace` methods.
- Produces: the existing `write_sessions`, `manage_memory`, `organize_topics`, and `verify_session` public results without an OpenAI chat-completions loop.

- [ ] **Step 1: Write failing tests proving a scripted agent can call the real staged shell tool, receive tool errors, and finish with one framework result**

```python
def test_writer_delegates_turns_and_tool_errors_to_agent(tmp_path):
    audit = write_sessions(tmp_path, agent=scripted_agent, sessions=sessions)
    assert scripted_agent.run_count == 1
    assert audit[-1]["tool"] == "agent"
    assert audit[-1]["status"] == "ok"
```

- [ ] **Step 2: Run the focused tests and verify failure against the manual loop**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/management/test_memory.py -q -k 'agent or writer or manager or verification'`

- [ ] **Step 3: Expose `MemoryWorkspace.shell` as one SDK MCP tool and replace `_run_agent`**

The handler records command, return code, changed Topic paths, and created-block count. It returns MCP `is_error=true` for rejected edits or command failures. `_run_agent` performs no JSON parsing, retry loop, message-history mutation, DSML detection, or tool-output compaction.

- [ ] **Step 4: Route structured verification and reconciliation through Claude Code structured output**

Use `output_format={"type":"json_schema","schema":...}` and consume `AgentResult.structured_output`; do not parse JSON from free-form model text.

- [ ] **Step 5: Run management tests**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/management/test_memory.py -q`

### Task 3: Retrieval MCP Tools and Query Migration

**Files:**
- Create: `code/src/retrieval/tool_server.py`
- Modify: `code/src/retrieval/agent.py`
- Modify: `code/src/retrieval/config.py`
- Modify: `code/src/retrieval/context.py`
- Modify: `code/src/retrieval/tools.py`
- Modify: `code/src/retrieval/runtime.py`
- Delete: `code/src/runtime/tool_output.py`
- Test: `code/tests/retrieval/test_longmemeval_reanswer.py`

**Interfaces:**
- Consumes: `ClaudeCodeAgent.run`, `execute_tool_call`, and condition-specific visible files.
- Produces: existing `(memories, steps, answer, trace)` return shape, with `steps` equal to Claude Code turns and trace reporting actual MCP calls and framework termination.

- [ ] **Step 1: Write failing tests for one Claude Code trajectory choosing shell, BM25, and Embedding and returning `<answer>`**

```python
def test_query_agent_uses_framework_tools_and_reports_usage(tmp_path):
    memories, steps, answer, trace = collect_answer(runtime, item, tmp_path, {})
    assert answer == "Shanghai"
    assert steps == 3
    assert [row["type"] for row in trace if row.get("type") != "termination"] == ["bash", "bm25_search"]
    assert trace[-1]["tool_calls"] == 2
```

- [ ] **Step 2: Run focused retrieval tests and verify failure against the hand-written loop**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/retrieval/test_longmemeval_reanswer.py -q -k 'agent or tool or answer'`

- [ ] **Step 3: Build one phase-specific in-process MCP server**

Each handler calls the existing real retrieval implementation and records its output. The handler returns an MCP error without adding any retry or alternate parsing logic. Claude Code owns continuation after errors.

- [ ] **Step 4: Replace `collect_answer` with one `ClaudeCodeAgent.run` call**

Remove the manual round loop, `no_new_evidence`, forced finalization request, custom visible-output truncation, and `ToolOutputStore`. Preserve initial Core/Recent/Inventory context, source-verification prompt switch, `<answer>` extraction, and read-only workspace hash checks in callers.

- [ ] **Step 5: Run retrieval tests**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/retrieval/test_longmemeval_reanswer.py -q`

### Task 4: Runtime, CLI, Accounting, and Documentation

**Files:**
- Modify: `code/src/retrieval/runtime.py`
- Modify: `code/src/build.py`
- Modify: `code/scripts/nativemem/locomo/config.py`
- Modify: `code/scripts/nativemem/locomo/runner.py`
- Modify: `code/scripts/nativemem/run_longmemeval.py`
- Modify: `code/scripts/nativemem/longmemeval/cli.py`
- Modify: `code/scripts/model_capacity/calibrate_writer.py`
- Modify: `code/scripts/nativemem/locomo/metrics.py`
- Modify: `code/scripts/nativemem/locomo/results.py`
- Modify: `docs/method/README.md`
- Modify: `docs/method/designs/file_native_multiview_design.md`
- Modify: `docs/method/nativemem-method.html`
- Test: `code/tests/scripts/test_current_runtime.py`
- Test: `code/tests/runtime/test_writer_capacity.py`

**Interfaces:**
- Produces: a single `create_runtime(base_url, api_key=..., model=...)` path that constructs an isolated Claude Code agent and records framework-native usage.

- [ ] **Step 1: Write failing tests for CLI propagation and aggregated framework usage**

```python
def test_runtime_records_framework_turns_as_calls():
    runtime.log_agent_result(result, phase="writer")
    assert runtime.call_log[-1]["calls"] == result.num_turns
    assert runtime.call_log[-1]["total_cost_usd"] == result.total_cost_usd
```

- [ ] **Step 2: Run focused runtime/script tests and verify failure**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/runtime/test_writer_capacity.py code/tests/scripts/test_current_runtime.py -q`

- [ ] **Step 3: Wire explicit Claude Code arguments and update accounting**

Remove `api_format`, OpenAI-client retry settings, and independent Writer/retrieval tool-call limits from the primary NativeMem runners. Retain one explicit `max_turns=20`, optional framework budget, endpoint, API key, model, and optional CLI path. Usage summaries sum `record.get("calls", 1)` and prefer framework-reported total cost when available while retaining price-based estimates.

- [ ] **Step 4: Update method documentation to distinguish framework control from NativeMem semantics**

Document that Claude Code handles trajectory mechanics while NativeMem contributes the memory representation, transactional writer tools, deterministic derived views, and retrieval operators. Mark Anthropic Messages compatibility and configuration isolation explicitly.

- [ ] **Step 5: Run the focused and complete current-source suites**

Run: `PYTHONPATH=code .venv/bin/pytest code/tests/agent_runtime code/tests/management code/tests/retrieval code/tests/runtime code/tests/scripts/test_current_runtime.py -q`

Run: `PYTHONPATH=code .venv/bin/python -m compileall -q code/src code/scripts/nativemem code/scripts/model_capacity`

Run: `sha256sum code/scripts/eval_full.py`

### Task 5: Framework Smoke Test and Conv-50 Gate

**Files:**
- Create only run artifacts under: `code/results/formal/<new-run-id>/`

**Interfaces:**
- Consumes: the completed Claude Code runtime and an Anthropic-Messages-compatible API endpoint.
- Produces: one no-cost/local protocol smoke result, followed by a fresh Conv-50 build/query result only after the smoke passes.

- [ ] **Step 1: Run a local adapter smoke that verifies CLI startup and MCP registration without touching the default Claude configuration**

Record the temporary configuration root, final framework status, tool count, and absence of new files under the default Claude session directory.

- [ ] **Step 2: Run one small real-API Writer/query smoke**

Use a fresh result directory and explicit function/CLI arguments. Stop before Conv-50 if the provider does not implement the Anthropic Messages API or tool calls are unsupported.

- [ ] **Step 3: Run Conv-50 in a fresh result directory**

Do not overwrite historical runs. Record build/query calls, input/output tokens, framework-reported cost, estimated price-based cost, latency, memory inventory, answer accuracy, and tool trace.

- [ ] **Step 4: Verify LoCoMo evaluator lock before scoring**

Run: `shasum -a 256 code/scripts/eval_full.py`

Expected: `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`

