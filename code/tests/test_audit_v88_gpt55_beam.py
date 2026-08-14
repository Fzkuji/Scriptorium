import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_v88_gpt55_beam.py"
SPEC = importlib.util.spec_from_file_location("audit_v88_gpt55_beam", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
RUNNER_SCRIPT = ROOT / "scripts" / "run_v88_gpt55_beam.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_v88_gpt55_beam_for_audit_test", RUNNER_SCRIPT
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def formal_read_original_tool():
    return {
        "type": "function",
        "function": {
            "name": "read_original",
            "description": (
                "读事件对应的原始对话。dia_ids 从事件行末尾的 [D1:3, D1:5] "
                "里取；锚可能是区间/列举形式，原样传即可。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dia_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "要回原文的 dia_id 列表，元素可含区间/列举写法，如 "
                            '["D1:3"]、["D2:3-15"]、["D2:2,10"]'
                        ),
                    }
                },
                "required": ["dia_ids"],
            },
        },
    }


def formal_tools():
    return [*RUNNER.BEAM_MEMORY_TOOLS, formal_read_original_tool()]


def make_direct_trace(question, answer, response_id, memory_dir):
    tokenizer, tokenizer_identity = MOD._load_context_tokenizer()
    structure = MOD._memory_structure_map(memory_dir)
    initial_prompt = MOD.BEAM_SINGLE_PROMPT.format(
        structure=structure, question=question
    )
    tools = formal_tools()
    descriptor = MOD._request_descriptor(
        tokenizer, [{"role": "user", "content": initial_prompt}], tools
    )
    observation = {
        "sequence": 1,
        "phase": "retrieve",
        "attempt": 1,
        "sent": True,
        "outcome": "response_received",
        "reason": None,
        "tools_enabled": True,
        "message_count": 1,
        "output_reservation_tokens": MOD.CONTEXT_OUTPUT_RESERVATION_TOKENS,
        "safety_margin_tokens": MOD.CONTEXT_SAFETY_MARGIN_TOKENS,
        **descriptor,
    }
    interaction = {
        "schema_version": MOD.INTERACTION_TRACE_SCHEMA_VERSION,
        "initial_prompt": initial_prompt,
        "structure": structure,
        "tools": tools,
        "finalize_instruction": MOD.FINALIZE_INSTRUCTION,
        "responses": [{
            "step": 1,
            "phase": "beam_v8_single",
            "request_sequence": 1,
            "response_id": response_id,
            "model": "gpt-5.5",
            "finish_reason": "stop",
            "refusal": "",
            "content": f"<answer>{answer}</answer>",
            "tool_calls": [],
        }],
        "termination_reason": "model_answer",
    }
    context = {
        "policy_version": MOD.CONTEXT_POLICY_VERSION,
        "trace_schema_version": MOD.CONTEXT_TRACE_SCHEMA_VERSION,
        "tokenizer": tokenizer_identity,
        "local_request_token_limit": MOD.LOCAL_REQUEST_TOKEN_LIMIT,
        "total_tool_content_token_limit": MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
        "per_tool_content_token_limit": MOD.PER_TOOL_CONTENT_TOKEN_LIMIT,
        "output_reservation_tokens": MOD.CONTEXT_OUTPUT_RESERVATION_TOKENS,
        "safety_margin_tokens": MOD.CONTEXT_SAFETY_MARGIN_TOKENS,
        "raw_tool_tokens": 0,
        "delivered_tool_tokens": 0,
        "truncated_tool_results": 0,
        "tool_budget_exhausted": False,
        "request_observations": [observation],
        "request_local_tokens": [descriptor["local_tokens"]],
        "max_request_local_tokens": descriptor["local_tokens"],
        "max_sent_request_local_tokens": descriptor["local_tokens"],
        "rejected_request_candidates": 0,
        "context_compaction_used": False,
        "compact_finalization": None,
    }
    return interaction, context


def make_formal_run(tmp_path):
    run_dir = tmp_path / "beam-run"
    proxy_log = tmp_path / "proxy.jsonl"
    conv_dir = run_dir / "100K" / "conversation_000"
    memory_dir = conv_dir / "memory"
    memory_dir.mkdir(parents=True)
    (run_dir / MOD.RUNNER_LOCK_FILENAME).touch()
    (memory_dir / "topic.md").write_text("Fact [D1:1]\n", encoding="utf-8")
    markdown_count, memory_hash = MOD.memory_tree_sha256(memory_dir)

    source_hashes = {
        relative: MOD.sha256_file(ROOT / relative)
        for relative in MOD.REQUIRED_SOURCE_FILES
    }
    method = {
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "base_url": "http://127.0.0.1:8199/v1",
        "single_model_retrieve_answer": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "tidy": True,
        "sections": True,
        "article": False,
        "tidy_combined": False,
        "verify": True,
        "merge_lines": True,
        "max_topics": 30,
        "map": "dir",
        "map_inline": 8,
        "max_rounds": 12,
        "max_tokens": 1200,
        "read_context": 1,
        "rewrite_min": 8,
        "max_depth": 4,
        "top_k": 20,
        "memory_tool_input_error_policy": "return-validated-error-to-model-v2",
        "context_safety": dict(MOD.EXPECTED_CONTEXT_CONFIG),
        "trust_system_proxy": False,
        "request_concurrency": 2,
        "dataset": MOD.EXPECTED_DATASET,
        "dataset_revision": MOD.EXPECTED_REVISION,
        "calendar": True,
        "build_import": None,
        "proxy_request_log": str(proxy_log),
        "source_sha256": source_hashes,
    }
    source = {
        "kind": "huggingface",
        "dataset": MOD.EXPECTED_DATASET,
        "config": "default",
        "revision": MOD.EXPECTED_REVISION,
        "split": "100K",
        "rows": 20,
        "fingerprint": "fixture-fingerprint",
    }
    mapping = {"raw-turn-1": ["D1:1"]}
    config = {
        **method,
        "chat_size": "100K",
        "conversation_index": 0,
        "conversation_id": "conv-0",
        "conversation_seed": {},
        "input": {
            "sessions": 1,
            "turns": 1,
            "source_id_count": 1,
            "source_id_occurrences": 1,
            "duplicate_source_id_count": 0,
            "source_id_map_sha256": MOD.stable_hash(mapping),
        },
    }
    config_hash = MOD.stable_hash(config)
    build_stats = {
        "status": "complete",
        "build_time_s": 1.0,
        "events": 1,
        "markdown_files": markdown_count,
        "memory_sha256": memory_hash,
        "expected_dia_ids": 1,
        "stored_dia_ids": 1,
        "dia_id_coverage": 1.0,
        "chunks_total": 1,
        "chunks_with_events": 1,
        "chunk_audit": [{"turns": 1, "events": 1, "referenced_dia_ids": 1}],
        "requested_models": ["gpt-5.5"],
        "response_models": ["gpt-5.5"],
        "finish_reasons": ["stop"],
        "calls": 1,
        "tokens_in": 10,
        "tokens_out": 5,
        "llm_time_s": 0.5,
        "reused_existing_build": False,
    }
    write_json(
        memory_dir / MOD.BUILD_MARKER,
        {
            "schema_version": 1,
            "input_hash": "input-hash",
            "config_hash": config_hash,
            "completed_at": "2026-07-14T00:10:00+00:00",
            "stats": build_stats,
        },
    )
    write_json(
        conv_dir / "source_id_map.json",
        {
            "schema_version": 1,
            "description": "test",
            "sha256": MOD.stable_hash(mapping),
            "mapping": mapping,
        },
    )

    questions = {}
    response_ids = []
    question_index = 0
    for question_type in MOD.EXPECTED_QUESTION_TYPES:
        for within_type in range(2):
            question_id = f"100K_0_q{question_index}_{question_type}"
            response_id = f"resp-q-{question_index}"
            response_ids.append(response_id)
            questions[question_id] = {
                "question_id": question_id,
                "question_index": question_index,
                "chat_size": "100K",
                "conversation_index": 0,
                "conversation_id": "conv-0",
                "question_type": question_type,
                "difficulty": "easy",
                "question": f"Question {question_index}?",
                "gold_field": "answer",
                "gold": f"Gold {question_index}",
                "ideal_response": f"Gold {question_index}",
                "rubric": [f"Criterion {question_index}"],
                "plan_reference": None,
                "source_chat_ids": [],
                "abstention_type": None,
                "why_unanswerable": None,
                "memory_sha256": memory_hash,
                "status": "complete",
                "completed_at": "2026-07-14T00:20:00+00:00",
                "answer": f"Answer {question_index}",
                "answer_format": "answer_tag",
                "memories": [],
                "retrieval": {
                    "latency_s": 0.1,
                    "steps": 1,
                    "calls": 1,
                    "tokens_in": 5,
                    "tokens_out": 2,
                    "llm_time_s": 0.1,
                    "response_models": ["gpt-5.5"],
                    "response_ids": [response_id],
                    "tool_input_errors": [],
                    "tool_trace": [],
                },
            }
            interaction, context = make_direct_trace(
                f"Question {question_index}?",
                f"Answer {question_index}",
                response_id,
                memory_dir,
            )
            questions[question_id]["retrieval"].update({
                "interaction_trace": interaction,
                "context_safety": context,
            })
            question_index += 1
    checkpoint = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "method": "NativeMem-v8.8+calendar",
        "chat_size": "100K",
        "conversation_index": 0,
        "conversation_id": "conv-0",
        "status": "complete",
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:30:00+00:00",
        "completed_at": "2026-07-14T00:30:00+00:00",
        "input_hash": "input-hash",
        "config_hash": config_hash,
        "config": config,
        "source": source,
        "question_count": 20,
        "build": {
            **build_stats,
            "completed_at": "2026-07-14T00:10:00+00:00",
            "memory_dir": "memory",
        },
        "questions": questions,
        "errors": [],
    }
    write_json(conv_dir / "checkpoint.json", checkpoint)
    write_json(conv_dir / "results.json", MOD.expected_result_payload(checkpoint))

    selection = {
        "chat_sizes": ["100K"],
        "conversations": "all",
        "limit_per_split": 1,
    }
    manifest = {
        "schema_version": 2,
        "benchmark": "BEAM",
        "method": method,
        "run_fingerprint": MOD.stable_hash(
            {"benchmark": "BEAM", "method": method, "fixture": None}
        ),
        "git_commit": "test",
        "created_at": "2026-07-14T00:00:00+00:00",
        "finished_at": "2026-07-14T01:00:00+00:00",
        "status": "complete",
        "selection": selection,
        "selection_history": [selection],
        "build_import": None,
        "sources": {"100K": source},
        "conversations": {
            "100K:0": {
                "status": "complete",
                "detail": "complete",
                "checkpoint": str(conv_dir / "checkpoint.json"),
            }
        },
        "errors": [],
        "failed_conversations": [],
    }
    write_json(run_dir / "run_manifest.json", manifest)

    entries = [
        {
            "timestamp": "2026-07-14T00:05:00+00:00",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
            "response_id": "resp-build-0",
            "attempts": 2,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    ]
    entries.extend(
        {
            "timestamp": "2026-07-14T00:20:00+00:00",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
            "response_id": response_id,
            "attempts": 2,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        for response_id in response_ids
    )
    proxy_log.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    return run_dir, proxy_log


def add_tool_trace_to_first_question(run_dir, proxy_log):
    checkpoint_path = run_dir / "100K/conversation_000/checkpoint.json"
    results_path = checkpoint_path.with_name("results.json")
    checkpoint = json.loads(checkpoint_path.read_text())
    question = min(
        checkpoint["questions"].values(),
        key=lambda item: item["question_index"],
    )
    retrieval = question["retrieval"]
    old_response_id = retrieval["response_ids"][0]
    final_response_id = f"{old_response_id}-final"
    tokenizer, tokenizer_identity = MOD._load_context_tokenizer()
    structure = MOD._memory_structure_map(checkpoint_path.parent / "memory")
    initial_prompt = MOD.BEAM_SINGLE_PROMPT.format(
        structure=structure, question=question["question"]
    )
    tools = formal_tools()
    call = {
        "id": "tool-list-1",
        "type": "function",
        "function": {
            "name": "list_memory",
            "arguments": '{"path":".","recursive":true}',
        },
    }
    arguments = {"path": ".", "recursive": True}
    memory_dir = checkpoint_path.parent / "memory"
    raw, error = MOD._execute_memory_tool_for_audit(
        "list_memory", arguments, memory_dir
    )
    assert error is None
    delivered, raw_tokens, delivered_tokens, truncated = (
        MOD._truncate_tool_content(
            raw,
            tokenizer=tokenizer,
            limit=MOD.PER_TOOL_CONTENT_TOKEN_LIMIT,
        )
    )
    tool_trace = [{
        "step": 1,
        "tool_call_id": call["id"],
        "tool": "list_memory",
        "arguments": arguments,
        "raw_sha256": MOD.hashlib.sha256(raw.encode()).hexdigest(),
        "delivered_sha256": MOD.hashlib.sha256(delivered.encode()).hexdigest(),
        "raw_tokens": raw_tokens,
        "delivered_tokens": delivered_tokens,
        "truncated": truncated,
        "remaining_tokens_before": MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
        "applied_token_limit": MOD.PER_TOOL_CONTENT_TOKEN_LIMIT,
        "truncation_algorithm": MOD.TRUNCATION_ALGORITHM,
        "cumulative_delivered_tokens": delivered_tokens,
        "delivered_text": delivered,
    }]
    messages = [{"role": "user", "content": initial_prompt}]
    descriptor_1 = MOD._request_descriptor(tokenizer, messages, tools)
    messages.extend([
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": delivered},
    ])
    descriptor_2 = MOD._request_descriptor(tokenizer, messages, tools)

    def observation(sequence, descriptor):
        return {
            "sequence": sequence,
            "phase": "retrieve",
            "attempt": 1,
            "sent": True,
            "outcome": "response_received",
            "reason": None,
            "tools_enabled": True,
            "message_count": 1 if sequence == 1 else 3,
            "output_reservation_tokens": MOD.CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": MOD.CONTEXT_SAFETY_MARGIN_TOKENS,
            **descriptor,
        }

    observations = [
        observation(1, descriptor_1),
        observation(2, descriptor_2),
    ]
    retrieval.update({
        "steps": 2,
        "calls": 2,
        "response_ids": [old_response_id, final_response_id],
        "tool_trace": tool_trace,
        "interaction_trace": {
            "schema_version": MOD.INTERACTION_TRACE_SCHEMA_VERSION,
            "initial_prompt": initial_prompt,
            "structure": structure,
            "tools": tools,
            "finalize_instruction": MOD.FINALIZE_INSTRUCTION,
            "responses": [
                {
                    "step": 1,
                    "phase": "beam_v8_single",
                    "request_sequence": 1,
                    "response_id": old_response_id,
                    "model": "gpt-5.5",
                    "finish_reason": "tool_calls",
                    "refusal": "",
                    "content": "",
                    "tool_calls": [call],
                },
                {
                    "step": 2,
                    "phase": "beam_v8_single",
                    "request_sequence": 2,
                    "response_id": final_response_id,
                    "model": "gpt-5.5",
                    "finish_reason": "stop",
                    "refusal": "",
                    "content": f"<answer>{question['answer']}</answer>",
                    "tool_calls": [],
                },
            ],
            "termination_reason": "model_answer",
        },
        "context_safety": {
            "policy_version": MOD.CONTEXT_POLICY_VERSION,
            "trace_schema_version": MOD.CONTEXT_TRACE_SCHEMA_VERSION,
            "tokenizer": tokenizer_identity,
            "local_request_token_limit": MOD.LOCAL_REQUEST_TOKEN_LIMIT,
            "total_tool_content_token_limit": MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
            "per_tool_content_token_limit": MOD.PER_TOOL_CONTENT_TOKEN_LIMIT,
            "output_reservation_tokens": MOD.CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": MOD.CONTEXT_SAFETY_MARGIN_TOKENS,
            "raw_tool_tokens": raw_tokens,
            "delivered_tool_tokens": delivered_tokens,
            "truncated_tool_results": int(truncated),
            "tool_budget_exhausted": False,
            "request_observations": observations,
            "request_local_tokens": [
                descriptor_1["local_tokens"], descriptor_2["local_tokens"]
            ],
            "max_request_local_tokens": max(
                descriptor_1["local_tokens"], descriptor_2["local_tokens"]
            ),
            "max_sent_request_local_tokens": max(
                descriptor_1["local_tokens"], descriptor_2["local_tokens"]
            ),
            "rejected_request_candidates": 0,
            "context_compaction_used": False,
            "compact_finalization": None,
        },
    })
    write_json(checkpoint_path, checkpoint)
    write_json(results_path, MOD.expected_result_payload(checkpoint))
    entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    entries.append({
        "timestamp": "2026-07-14T00:21:00+00:00",
        "status": "success",
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": final_response_id,
        "attempts": 2,
        "unsupported_parameters": ["max_output_tokens"],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    })
    proxy_log.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return checkpoint_path, question["question_id"]


def add_compact_trace_to_first_question(run_dir, proxy_log):
    checkpoint_path = run_dir / "100K/conversation_000/checkpoint.json"
    results_path = checkpoint_path.with_name("results.json")
    checkpoint = json.loads(checkpoint_path.read_text())
    question = min(
        checkpoint["questions"].values(),
        key=lambda item: item["question_index"],
    )
    first_response_id = question["retrieval"]["response_ids"][0]
    final_response_id = f"{first_response_id}-compact-final"
    assistant_padding = "padding " * 100_000
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                function = SimpleNamespace(
                    name="list_memory",
                    arguments='{"path":".","recursive":true}',
                )
                tool_call = SimpleNamespace(
                    id="compact-list",
                    type="function",
                    function=function,
                )
                message = SimpleNamespace(
                    content=assistant_padding,
                    tool_calls=[tool_call],
                    refusal="",
                )
                finish_reason = "tool_calls"
                response_id = first_response_id
            else:
                message = SimpleNamespace(
                    content=f"<answer>{question['answer']}</answer>",
                    tool_calls=None,
                    refusal="",
                )
                finish_reason = "stop"
                response_id = final_response_id
            return SimpleNamespace(
                id=response_id,
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=message,
                    finish_reason=finish_reason,
                )],
            )

    read_tool = formal_read_original_tool()
    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL=read_tool,
        _v8_structure_map=lambda path: MOD._memory_structure_map(Path(path)),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    result = RUNNER.answer_question(
        native,
        question["question"],
        checkpoint_path.parent / "memory",
        {},
    )
    assert len(calls) == 2
    assert result["context_safety"]["context_compaction_used"] is True
    question["answer"] = result["answer"]
    question["memories"] = result["memories"]
    question["retrieval"].update({
        "steps": result["steps"],
        "calls": result["steps"],
        "response_models": result["response_models"],
        "response_ids": result["response_ids"],
        "tool_input_errors": result["tool_input_errors"],
        "tool_trace": result["tool_trace"],
        "interaction_trace": result["interaction_trace"],
        "context_safety": result["context_safety"],
    })
    write_json(checkpoint_path, checkpoint)
    write_json(results_path, MOD.expected_result_payload(checkpoint))
    entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    entries.append({
        "timestamp": "2026-07-14T00:21:00+00:00",
        "status": "success",
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": final_response_id,
        "attempts": 2,
        "unsupported_parameters": ["max_output_tokens"],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    })
    proxy_log.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return checkpoint_path, question["question_id"]


def test_strict_audit_collects_true_runner_schema(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    evaluation_input, report = MOD.audit(
        run_dir, proxy_log=proxy_log, allow_fixture=True
    )
    assert evaluation_input["question_count"] == 20
    assert evaluation_input["rubric_nugget_count"] == 20
    assert set(evaluation_input["question_type_counts"]) == set(
        MOD.EXPECTED_QUESTION_TYPES
    )
    assert report["model_evidence"]["qa_response_ids_matched_in_proxy_log"] == 20
    assert report["scoring_scope"]["official_tau_b_times_f1"] == "not_computed"
    assert not report["formal_scope_verified"]
    assert report["run_dir"] == str(run_dir)
    assert report["run_manifest_path"] == str((run_dir / "run_manifest.json").resolve())
    assert report["run_manifest_sha256"] == MOD.sha256_file(
        run_dir / "run_manifest.json"
    )
    assert report["evaluation_input_path"] == str(
        (run_dir / "evaluation_input.json").resolve()
    )


def test_audit_lock_is_exclusive_and_holds_runner_snapshot_lock(tmp_path):
    run_dir, _ = make_formal_run(tmp_path)
    first = MOD.acquire_audit_locks(run_dir)
    try:
        with pytest.raises(MOD.AuditError, match="another BEAM audit"):
            MOD.acquire_audit_locks(run_dir)

        runner_probe = (run_dir / MOD.RUNNER_LOCK_FILENAME).open("r+")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(runner_probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            runner_probe.close()
    finally:
        MOD.release_audit_locks(first)


def test_auditor_rejects_active_runner_lock(tmp_path):
    run_dir, _ = make_formal_run(tmp_path)
    runner = (run_dir / MOD.RUNNER_LOCK_FILENAME).open("r+")
    fcntl.flock(runner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(MOD.AuditError, match="runner is active"):
            MOD.acquire_audit_locks(run_dir)
    finally:
        fcntl.flock(runner.fileno(), fcntl.LOCK_UN)
        runner.close()

    locks = MOD.acquire_audit_locks(run_dir)
    MOD.release_audit_locks(locks)


def test_formal_audit_rejects_partial_selection(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    with pytest.raises(MOD.AuditError, match="formal BEAM scope"):
        MOD.audit(run_dir, proxy_log=proxy_log)


def test_audit_rejects_missing_proxy_response_id(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    lines = proxy_log.read_text(encoding="utf-8").splitlines()
    proxy_log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(MOD.AuditError, match="absent from the proxy log"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_audit_rejects_changed_memory(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    (run_dir / "100K" / "conversation_000" / "memory" / "topic.md").write_text(
        "changed [D1:1]\n", encoding="utf-8"
    )
    with pytest.raises(MOD.AuditError, match="memory hash mismatch"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_audit_ignores_unrelated_proxy_requests_and_does_not_substitute_build(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    entries = entries[1:]
    entries.append(
        {
            "timestamp": "2026-07-14T00:25:00+00:00",
            "status": "success",
            "requested_model": "openai/gpt-4o-mini",
            "actual_model": "openai/gpt-4o-mini",
            "response_id": "unrelated-primary-judge",
            "attempts": 1,
            "unsupported_parameters": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )
    proxy_log.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    _, report = MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)
    evidence = report["model_evidence"]
    assert evidence["recorded_build_calls"] == 1
    assert evidence["build"]["response_id_linkage"].startswith("unavailable")
    assert evidence["build"]["proxy_calls_used_as_substitute"] == 0
    assert evidence["proxy_window"]["matched_qa_successes"] == 20


def test_audit_rejects_memory_and_checkpoint_symlinks(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    conv_dir = run_dir / "100K/conversation_000"
    memory_dir = conv_dir / "memory"
    outside = tmp_path / "outside-memory"
    memory_dir.rename(outside)
    memory_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(MOD.AuditError, match="symbolic link"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)

    memory_dir.unlink()
    outside.rename(memory_dir)
    checkpoint = conv_dir / "checkpoint.json"
    real_checkpoint = conv_dir / "checkpoint-real.json"
    checkpoint.rename(real_checkpoint)
    checkpoint.symlink_to(real_checkpoint)
    with pytest.raises(MOD.AuditError, match="symbolic link"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_failed_beam_cli_removes_stale_passed_audit(tmp_path):
    run_dir = tmp_path / "invalid-run"
    run_dir.mkdir()
    (run_dir / MOD.RUNNER_LOCK_FILENAME).touch()
    audit_path = run_dir / "audit.json"
    audit_path.write_text('{"status":"passed"}', encoding="utf-8")

    with pytest.raises(SystemExit):
        MOD.main([str(run_dir), "--proxy-log", str(tmp_path / "missing.jsonl")])
    assert not audit_path.exists()


def test_audit_rejects_proxy_path_not_frozen_in_method(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    other_log = tmp_path / "other-proxy.jsonl"
    other_log.write_text(proxy_log.read_text(), encoding="utf-8")
    with pytest.raises(MOD.AuditError, match="differs from the path frozen"):
        MOD.audit(run_dir, proxy_log=other_log, allow_fixture=True)


def test_audit_rejects_marker_checkpoint_build_model_disagreement(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    marker_path = run_dir / "100K" / "conversation_000" / "memory" / MOD.BUILD_MARKER
    marker = json.loads(marker_path.read_text())
    marker["stats"]["response_models"] = ["wrong-model"]
    write_json(marker_path, marker)
    with pytest.raises(MOD.AuditError, match="marker/checkpoint response_models"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


@pytest.mark.parametrize(
    "tamper",
    [
        "delivered_text",
        "delivered_sha256",
        "delivered_tokens",
        "cumulative_delivered_tokens",
        "context_delivered_tokens",
        "request_payload_sha256",
        "request_local_tokens",
    ],
)
def test_auditor_rejects_context_trace_tampering(tmp_path, tamper):
    run_dir, proxy_log = make_formal_run(tmp_path)
    checkpoint_path, question_id = add_tool_trace_to_first_question(
        run_dir, proxy_log
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    retrieval = checkpoint["questions"][question_id]["retrieval"]
    trace = retrieval["tool_trace"][0]
    context = retrieval["context_safety"]
    if tamper == "delivered_text":
        trace["delivered_text"] += "tampered"
    elif tamper == "delivered_sha256":
        trace["delivered_sha256"] = "0" * 64
    elif tamper == "delivered_tokens":
        trace["delivered_tokens"] += 1
    elif tamper == "cumulative_delivered_tokens":
        trace["cumulative_delivered_tokens"] += 1
    elif tamper == "context_delivered_tokens":
        context["delivered_tool_tokens"] += 1
    elif tamper == "request_payload_sha256":
        context["request_observations"][0]["payload_sha256"] = "0" * 64
    elif tamper == "request_local_tokens":
        context["request_observations"][0]["local_tokens"] += 1
    write_json(checkpoint_path, checkpoint)
    with pytest.raises(MOD.AuditError, match="mismatch"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_auditor_requires_exact_context_safety_method_config(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["method"]["context_safety"]["truncation_algorithm"]
    write_json(manifest_path, manifest)
    with pytest.raises(MOD.AuditError, match="method configuration mismatch"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_auditor_fails_closed_when_tiktoken_is_unavailable(
        tmp_path, monkeypatch):
    run_dir, proxy_log = make_formal_run(tmp_path)

    def missing(_name):
        raise MOD.importlib_metadata.PackageNotFoundError("tiktoken")

    monkeypatch.setattr(MOD.importlib_metadata, "version", missing)
    with pytest.raises(MOD.AuditError, match="no tokenizer fallback"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)


def test_auditor_replays_compact_finalization_and_rejects_tamper(tmp_path):
    run_dir, proxy_log = make_formal_run(tmp_path)
    checkpoint_path, question_id = add_compact_trace_to_first_question(
        run_dir, proxy_log
    )
    _, report = MOD.audit(
        run_dir, proxy_log=proxy_log, allow_fixture=True
    )
    assert report["status"] == "passed"

    checkpoint = json.loads(checkpoint_path.read_text())
    context = checkpoint["questions"][question_id]["retrieval"][
        "context_safety"
    ]
    context["compact_finalization"]["delivered_sha256"] = "0" * 64
    write_json(checkpoint_path, checkpoint)
    with pytest.raises(MOD.AuditError, match="compact-finalization record mismatch"):
        MOD.audit(run_dir, proxy_log=proxy_log, allow_fixture=True)
