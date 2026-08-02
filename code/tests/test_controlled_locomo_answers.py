from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import controlled_locomo_answer_contract as contract  # noqa: E402
import audit_controlled_locomo_answers as answer_auditor  # noqa: E402
import freeze_controlled_answer_protocol as freezer  # noqa: E402
import run_controlled_locomo_answer_sanity as sanity  # noqa: E402
import run_controlled_locomo_answers as runner  # noqa: E402
from scripts.evaluation.answerer import format_memories  # noqa: E402
from scripts.evaluation.visible_token_budget import TokenCounter  # noqa: E402


def _flex_health() -> dict[str, object]:
    return {
        "status": "ok",
        "schema": "openai-gpt55-flex-health/v1",
        "requested_model": "gpt-5.5",
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "budget": {
            "max_cost_usd": "100",
            "committed_cost_usd": "1",
            "reserved_cost_usd": "2",
            "remaining_cost_usd": "97",
            "in_flight": 1,
        },
    }


def _add_flex_success_evidence(
    response: dict[str, object], *, attempts: int = 1
) -> dict[str, object]:
    response.update(
        {
            "service_tier": "flex",
            "flex_gateway_meta": {
                "request_id": "gateway-request-1",
                "request_sha256": "a" * 64,
                "provider_request_sha256": "b" * 64,
                "provider_actual_model": "gpt-5.5-2026-04-23",
                "returned_alias": "gpt-5.5",
                "service_tier": "flex",
                "physical_attempt_count": attempts,
                "physical_retry_count": attempts - 1,
            },
            "proxy_meta": {
                "attempts": attempts,
                "http_request_id": "provider-request-1",
                "unsupported_parameters": [],
                "provider_actual_model": "gpt-5.5-2026-04-23",
                "service_tier": "flex",
            },
        }
    )
    return response


def test_individual_renderer_is_format_memories_equivalent_and_stable():
    memories = [
        {"text": "late-a", "date": "2025-02-01"},
        {"text": "early", "date": "2025-01-01"},
        {"text": "late-b", "date": "2025-02-01"},
    ]
    rendered = contract.render_memories_individually(memories)
    assert "".join(item.rendered_text for item in rendered) == format_memories(memories)
    assert [item.input_index for item in rendered] == [1, 0, 2]


def test_prompt_boundary_excludes_gold_evidence_category_and_raw_overflow(tmp_path):
    record = {
        "question_id": "s0_q0",
        "question": "What remains?",
        "gold": sanity.GOLD_CANARY,
        "category": sanity.CATEGORY_CANARY,
        "evidence": [sanity.EVIDENCE_CANARY],
        "memories": [
            {
                "text": "visible " * 20 + sanity.RAW_OVERFLOW_CANARY,
                "date": None,
            },
            {"text": sanity.POST_EXHAUSTION_CANARY, "date": None},
        ],
    }
    fake = runner.FakeAnswerClient(
        proxy_log=tmp_path / "proxy.jsonl", run_id="canary-run"
    )
    (tmp_path / "attempt").mkdir()
    result = runner.answer_one(
        run_id="canary-run",
        method="bm25",
        question_id=record["question_id"],
        question=record["question"],
        memories=record["memories"],
        input_record_sha256=contract.canonical_hash(record),
        attempt_dir=tmp_path / "attempt",
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        budget_policy=contract.HARD_CAP_POLICY,
        budget_tokens=16,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="a" * 64,
        client=fake,
    )
    prompt = fake.prompts[0]
    assert "visible" in prompt
    assert sanity.GOLD_CANARY not in prompt
    assert sanity.EVIDENCE_CANARY not in prompt
    assert sanity.CATEGORY_CANARY not in prompt
    assert sanity.RAW_OVERFLOW_CANARY not in prompt
    assert sanity.POST_EXHAUSTION_CANARY not in prompt
    assert result["budget"]["visible_tokens"] <= 16
    assert result["budget"]["source_resolution_tokens"] == 0
    trace = [
        json.loads(line)
        for line in (tmp_path / "attempt/visible_tokens.jsonl").read_text().splitlines()
    ]
    deliveries = trace[1:-1]
    assert len(deliveries) == 2
    assert deliveries[0]["decision"] == "truncated"
    assert deliveries[1]["decision"] == "rejected_budget_exhausted"
    assert all(item["cumulative_source_resolution_tokens"] == 0 for item in deliveries)


def test_full_context_unbounded_accounted_never_truncates(tmp_path):
    fake = runner.FakeAnswerClient(
        proxy_log=tmp_path / "proxy.jsonl", run_id="full-run"
    )
    (tmp_path / "attempt").mkdir()
    result = runner.answer_one(
        run_id="full-run",
        method="full_context",
        question_id="s0_q0",
        question="What is present?",
        memories=[
            {"text": "first", "date": "2025-01-01"},
            {"text": sanity.FULL_CONTEXT_CANARY, "date": "2025-01-02"},
        ],
        input_record_sha256="b" * 64,
        attempt_dir=tmp_path / "attempt",
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        budget_policy=contract.FULL_CONTEXT_POLICY,
        budget_tokens=None,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="c" * 64,
        client=fake,
    )
    assert result["budget"]["declared_tokens"] == "unbounded"
    assert result["budget"]["full_context_matched_cap_claimed"] is False
    assert sanity.FULL_CONTEXT_CANARY in fake.prompts[0]
    trace = [
        json.loads(line)
        for line in (tmp_path / "attempt/visible_tokens.jsonl").read_text().splitlines()
    ]
    assert all(item["decision"] == "delivered" for item in trace[1:-1])


def test_full_context_refuses_context_overflow_without_renaming(tmp_path):
    fake = runner.FakeAnswerClient(
        proxy_log=tmp_path / "proxy.jsonl", run_id="full-overflow"
    )
    (tmp_path / "attempt").mkdir()
    with pytest.raises(contract.ControlledAnswerError, match="context limit"):
        runner.answer_one(
            run_id="full-overflow",
            method="full_context",
            question_id="s0_q0",
            question="Question?",
            memories=[{"text": "x" * 100, "date": None}],
            input_record_sha256="d" * 64,
            attempt_dir=tmp_path / "attempt",
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            budget_policy=contract.FULL_CONTEXT_POLICY,
            budget_tokens=None,
            model_context_limit_tokens=20,
            answer_max_tokens=10,
            preregistration_sha256="e" * 64,
            client=fake,
        )
    assert not fake.prompts
    assert not (tmp_path / "attempt/visible_tokens.manifest.json").exists()


def test_cat5_canonical_gold_rebuilt_from_raw_dataset_and_distractor_never_gold():
    dataset = json.loads((ROOT / "benchmarks/locomo/data/locomo10.json").read_text())
    references = []
    for sample, item in enumerate(dataset):
        for index, qa in enumerate(item["qa"]):
            if qa["category"] == 5:
                references.append(
                    contract.canonical_scoring_reference(dataset, f"s{sample}_q{index}")
                )
    assert len(references) == 446
    explicit = [
        reference
        for reference in references
        if reference["canonical_gold_source"] == "raw_dataset.answer"
    ]
    abstention = [
        reference
        for reference in references
        if reference["canonical_gold_source"] == "cat5_canonical_abstention"
    ]
    assert len(explicit) == 2
    assert len(abstention) == 444
    assert all(
        reference["canonical_gold"] == contract.CAT5_CANONICAL_ABSTENTION
        for reference in abstention
    )
    assert all(reference["adversarial_distractor"] for reference in references)
    assert all(
        reference["canonical_gold"] != reference["adversarial_distractor"]
        for reference in abstention
    )


def test_independent_question_auditor_rebuilds_prompt_and_cat5_labels(tmp_path):
    output = tmp_path / "output"
    attempt = output / "questions/s0_q0/attempts/attempt-0001"
    attempt.mkdir(parents=True)
    record = {
        "question_id": "s0_q0",
        "question": "What was never stated?",
        "gold": "DISTRACTOR_CANARY_NOT_GOLD",
        "category": 5,
        "evidence": ["D1:1"],
        "memories": [{"text": "ordinary memory", "date": None}],
        "retrieval": {},
    }
    raw_dataset = [
        {
            "qa": [
                {
                    "question": record["question"],
                    "category": 5,
                    "evidence": record["evidence"],
                    "adversarial_answer": record["gold"],
                }
            ]
        }
    ]
    fake = runner.FakeAnswerClient(
        proxy_log=output / "proxy.jsonl", run_id="audit-run"
    )
    result = runner.answer_one(
        run_id="audit-run",
        method="bm25",
        question_id="s0_q0",
        question=record["question"],
        memories=record["memories"],
        input_record_sha256=contract.canonical_hash(record),
        attempt_dir=attempt,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        budget_policy=contract.HARD_CAP_POLICY,
        budget_tokens=20_000,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="f" * 64,
        client=fake,
    )
    reference = contract.canonical_scoring_reference(raw_dataset, "s0_q0")
    result["scoring_reference"] = {
        "category": reference["category"],
        "canonical_gold": reference["canonical_gold"],
        "canonical_gold_source": reference["canonical_gold_source"],
        "adversarial_distractor": reference["adversarial_distractor"],
        "evidence": reference["evidence"],
        "attached_after_answer": True,
        "baseline_input_gold_authoritative": False,
    }
    checkpoint = runner._write_completion(
        output_dir=output,
        question_id="s0_q0",
        attempt_dir=attempt,
        result=result,
    )
    proxy_entries = [json.loads(line) for line in (output / "proxy.jsonl").read_text().splitlines()]
    report = answer_auditor.audit_question(
        output_dir=output,
        input_record=record,
        raw_dataset=raw_dataset,
        run_manifest={
            "run_id": "audit-run",
            "config": {
                "method": "bm25",
                "budget_policy": contract.HARD_CAP_POLICY,
                "preregistration_sha256": "f" * 64,
            },
        },
        protocol={},
        checkpoint_path=checkpoint,
        proxy_entries=proxy_entries,
        require_proxy_log=True,
    )
    assert report["category"] == 5
    assert result["scoring_reference"]["canonical_gold"] == contract.CAT5_CANONICAL_ABSTENTION
    assert result["scoring_reference"]["adversarial_distractor"] == record["gold"]
    assert record["gold"] not in fake.prompts[0]


def test_cli_requires_explicit_20k_or_unbounded_policy(fake_flex_provider):
    gateway_args = [
        "--allow-model-requests",
        "--gateway-root",
        str(fake_flex_provider.root),
    ]
    assert runner.parse_args(
        [
            "--method",
            "bm25",
            "--input-dir",
            "/tmp/in",
            "--output-dir",
            "/tmp/out",
            "--preregistration",
            "/tmp/pre.json",
            "--preregistration-sha256",
            "f" * 64,
            "--budget-policy",
            "hard_cap",
            "--budget-tokens",
            "20000",
            "--model-context-limit-tokens",
            "100000",
            *gateway_args,
        ]
    ).budget_tokens == 20_000
    assert runner.parse_args(
        [
            "--method",
            "full_context",
            "--input-dir",
            "/tmp/in",
            "--output-dir",
            "/tmp/out",
            "--preregistration",
            "/tmp/pre.json",
            "--preregistration-sha256",
            "f" * 64,
            "--budget-policy",
            "full_context_unbounded_accounted",
            "--budget-tokens",
            "unbounded",
            "--model-context-limit-tokens",
            "100000",
            *gateway_args,
        ]
    ).budget_tokens is None
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--method",
                "bm25",
                "--input-dir",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--preregistration",
                "/tmp/pre.json",
                "--preregistration-sha256",
                "f" * 64,
                "--budget-policy",
                "hard_cap",
                "--model-context-limit-tokens",
                "100000",
                *gateway_args,
            ]
        )


def test_protocol_freeze_rows_are_20k_and_full_context_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        freezer,
        "_input_binding",
        lambda method, run_dir: {
            "run_dir": str(run_dir),
            "method": method,
            "dataset_sha256": "d" * 64,
        },
    )
    payload = freezer.create_protocol(
        input_dirs={method: tmp_path / method for method in contract.FORMAL_METHODS},
        model_context_limit_tokens=100_000,
        answer_max_tokens_requested=512,
    )
    rows = {row["method"]: row for row in payload["rows"]}
    assert rows["full_context"]["budget_policy"] == contract.FULL_CONTEXT_POLICY
    assert rows["full_context"]["budget_tokens"] == "unbounded"
    assert rows["full_context"]["matched_cap_claim_allowed"] is False
    assert all(
        rows[method]["budget_tokens"] == 20_000
        for method in ("bm25", "mem0", "zep")
    )
    assert payload["claims"]["legacy_6000_token_claim_retained"] is False
    assert payload["scoring_labels"]["adversarial_answer_role"] == "distractor_not_gold"
    path = tmp_path / "protocol.json"
    contract.atomic_json_no_clobber(path, payload)
    assert contract.validate_preregistration(
        path, expected_file_sha256=contract.sha256_file(path)
    ) == payload
    tampered = json.loads(json.dumps(payload))
    tampered["claims"]["legacy_6000_token_claim_retained"] = True
    tampered["protocol_content_sha256"] = contract.protocol_content_hash(tampered)
    tampered_path = tmp_path / "tampered-protocol.json"
    contract.atomic_json_no_clobber(tampered_path, tampered)
    with pytest.raises(contract.ControlledAnswerError, match="claims"):
        contract.validate_preregistration(
            tampered_path,
            expected_file_sha256=contract.sha256_file(tampered_path),
        )


def test_hash_chained_ledger_rejects_tampering(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with contract.DurableLedger(path, run_id="r") as ledger:
        ledger.append("one", {"value": 1})
        ledger.append("two", {"value": 2})
    assert len(contract.audit_ledger(path, expected_run_id="r")) == 2
    text = path.read_text().replace('"value":2', '"value":3')
    path.write_text(text)
    with pytest.raises(contract.ControlledAnswerError, match="record hash"):
        contract.audit_ledger(path, expected_run_id="r")


def test_path_guards_reject_symlink_hardlink_and_unicode_case_collisions(tmp_path):
    original = tmp_path / "artifact.json"
    original.write_text("{}")
    hardlink = tmp_path / "hard.json"
    hardlink.hardlink_to(original)
    with pytest.raises(contract.ControlledAnswerError, match="inode collision"):
        contract.ensure_distinct_paths({"left": original, "right": hardlink})
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(original)
    with pytest.raises(contract.ControlledAnswerError, match="symlink"):
        contract.ensure_distinct_paths({"linked": symlink, "other": tmp_path / "x"})
    composed = tmp_path / "\u00e9.json"
    decomposed = tmp_path / "e\u0301.json"
    with pytest.raises(contract.ControlledAnswerError, match="path collision"):
        contract.ensure_distinct_paths({"composed": composed, "decomposed": decomposed})
    with pytest.raises(contract.ControlledAnswerError, match="path collision"):
        contract.ensure_distinct_paths(
            {"upper": tmp_path / "Answer.JSON", "lower": tmp_path / "answer.json"}
        )


def test_no_network_sanity_uses_fake_proxy_and_formal_tokenizer(tmp_path):
    report = sanity.run_sanity(tmp_path / "sanity", budget_tokens=16)
    assert report["status"] == "passed"
    assert report["network_requests"] == 0
    assert report["fake_proxy_entries"] == 2
    assert report["tokenizer"]["implementation_version"] == "0.12.0"
    assert report["tokenizer"]["encoding_name"] == "o200k_base"


def test_exclusive_proxy_logs_question_usage_upstream_attempts_and_hashes(tmp_path):
    class FakeUpstream(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            payload = json.dumps(_flex_health()).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            payload = json.dumps(
                _add_flex_success_evidence({
                    "id": "resp-1",
                    "model": request["model"],
                    "choices": [
                        {
                            "message": {"content": "<answer>x</answer>"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }, attempts=2)
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            del format, args

    upstream = HTTPServer(("127.0.0.1", 0), FakeUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    log = tmp_path / "requests.jsonl"
    ready = tmp_path / "ready.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/controlled_gpt55_run_proxy.py"),
            "--upstream",
            f"http://127.0.0.1:{upstream.server_port}",
            "--log",
            str(log),
            "--ready",
            str(ready),
            "--run-id",
            "proxy-test",
        ],
        cwd=ROOT,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        metadata = json.loads(ready.read_text())
        body = json.dumps(
            {"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            metadata["base_url"] + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Controlled-Question-ID": "s0_q0",
                "X-Controlled-Logical-Call-ID": "run:s0_q0:attempt-0001",
            },
            method="POST",
        )
        response = json.loads(
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            .open(request, timeout=5)
            .read()
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
    entry = json.loads(log.read_text())
    assert entry["question_id"] == "s0_q0"
    assert entry["logical_call_id"] == "run:s0_q0:attempt-0001"
    assert entry["request_sha256"] == contract.sha256_bytes(body)
    assert entry["response_sha256"]
    assert entry["usage"]["total_tokens"] == 12
    assert entry["upstream_http_attempts"] == 2
    assert entry["unsupported_parameters"] == []
    assert entry["provider_actual_model"] == "gpt-5.5-2026-04-23"
    assert entry["service_tier"] == "flex"
    assert entry["gateway_request_id"] == "gateway-request-1"
    assert entry["gateway_request_sha256"] == "a" * 64
    assert entry["provider_request_sha256"] == "b" * 64
    assert response["exclusive_proxy_meta"]["event_id"] == entry["event_id"]
    assert response["exclusive_proxy_meta"]["request_sha256"] == entry["request_sha256"]


def test_http_client_accounts_for_failed_then_successful_proxy_attempts(tmp_path):
    class RetryUpstream(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):  # noqa: N802
            type(self).calls += 1
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if type(self).calls == 1:
                status = 429
                response = {
                    "error": "retryable",
                    "proxy_meta": {
                        "attempts": 1,
                        "unsupported_parameters": ["first_attempt_marker"],
                    },
                }
            else:
                status = 200
                response = _add_flex_success_evidence({
                    "id": "resp-retry-success",
                    "model": request["model"],
                    "choices": [
                        {
                            "message": {"content": "<answer>x</answer>"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }, attempts=2)
            payload = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            del format, args

    upstream = HTTPServer(("127.0.0.1", 0), RetryUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    log = tmp_path / "requests.jsonl"
    ready = tmp_path / "ready.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/controlled_gpt55_run_proxy.py"),
            "--upstream",
            f"http://127.0.0.1:{upstream.server_port}",
            "--log",
            str(log),
            "--ready",
            str(ready),
            "--run-id",
            "retry-proxy-test",
        ],
        cwd=ROOT,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        metadata = json.loads(ready.read_text())
        logical_call_id = "retry-proxy-test:s0_q0:attempt-0001"
        with contract.DurableLedger(
            tmp_path / "question_ledger.jsonl", run_id=logical_call_id
        ) as ledger:
            call = runner.HttpAnswerClient(
                base_url=metadata["base_url"], retries=2, answer_max_tokens=512
            ).complete(
                prompt="question",
                question_id="s0_q0",
                logical_call_id=logical_call_id,
                ledger=ledger,
            )
    finally:
        process.terminate()
        process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["error", "success"]
    assert call.client_http_attempts == 2
    assert call.upstream_http_attempts == 3
    assert list(call.proxy_event_ids) == [entry["event_id"] for entry in entries]
    assert list(call.response_sha256s) == [
        entry["response_sha256"] for entry in entries
    ]
    assert set(call.unsupported_parameters) == {
        "first_attempt_marker",
    }


def test_http_client_escalates_empty_output_with_unknown_provider_attempts(
    tmp_path,
):
    class EmptyOutputUpstream(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):  # noqa: N802
            type(self).calls += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = {
                "error": (
                    "upstream failed after retries: "
                    "upstream completed with empty output"
                ),
                "exclusive_proxy_meta": {
                    "event_id": "empty-output-event",
                    "question_id": self.headers["X-Controlled-Question-ID"],
                    "logical_call_id": self.headers[
                        "X-Controlled-Logical-Call-ID"
                    ],
                    "request_sha256": contract.sha256_bytes(body),
                    "client_http_attempts": 1,
                    "upstream_http_attempts": None,
                    "unsupported_parameters": [],
                },
            }
            encoded = json.dumps(payload).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            del format, args

    upstream = HTTPServer(("127.0.0.1", 0), EmptyOutputUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    logical_call_id = "qa-test:attempt-0001:q0:dual_source:answer"
    try:
        with contract.DurableLedger(
            tmp_path / "answer_ledger.jsonl", run_id=logical_call_id
        ) as ledger:
            with pytest.raises(runner.RetryableEmptyOutputError) as exc_info:
                runner.HttpAnswerClient(
                    base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
                    retries=3,
                    answer_max_tokens=4096,
                ).complete(
                    prompt="question",
                    question_id="q0",
                    logical_call_id=logical_call_id,
                    ledger=ledger,
                )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert exc_info.value.status_code == 500
    assert "upstream completed with empty output" in str(exc_info.value)
    assert EmptyOutputUpstream.calls == 1
    ledger_events = [
        json.loads(line)
        for line in (tmp_path / "answer_ledger.jsonl").read_text().splitlines()
    ]
    finished = [
        event
        for event in ledger_events
        if event.get("event") == "physical_http_attempt_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["payload"]["status"] == "retryable_empty_output"
    assert finished[0]["payload"]["upstream_http_attempts"] is None


def test_http_client_does_not_retry_non_500_empty_output_marker(tmp_path):
    class Non500Upstream(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):  # noqa: N802
            type(self).calls += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = {
                "error": "upstream completed with empty output",
                "exclusive_proxy_meta": {
                    "event_id": "non-500-event",
                    "question_id": self.headers["X-Controlled-Question-ID"],
                    "logical_call_id": self.headers[
                        "X-Controlled-Logical-Call-ID"
                    ],
                    "request_sha256": contract.sha256_bytes(body),
                    "client_http_attempts": 1,
                    "upstream_http_attempts": None,
                    "unsupported_parameters": [],
                },
            }
            encoded = json.dumps(payload).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            del format, args

    upstream = HTTPServer(("127.0.0.1", 0), Non500Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    logical_call_id = "qa-test:attempt-0001:q0:dual_source:answer"
    try:
        with contract.DurableLedger(
            tmp_path / "answer_ledger.jsonl", run_id=logical_call_id
        ) as ledger:
            with pytest.raises(contract.ControlledAnswerError) as exc_info:
                runner.HttpAnswerClient(
                    base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
                    retries=3,
                    answer_max_tokens=4096,
                ).complete(
                    prompt="question",
                    question_id="q0",
                    logical_call_id=logical_call_id,
                    ledger=ledger,
                )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert not isinstance(exc_info.value, runner.RetryableEmptyOutputError)
    assert Non500Upstream.calls == 1
    ledger_events = [
        json.loads(line)
        for line in (tmp_path / "answer_ledger.jsonl").read_text().splitlines()
    ]
    finished = [
        event
        for event in ledger_events
        if event.get("event") == "physical_http_attempt_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["payload"]["status"] == "unaccountable_error"


def test_interrupted_proxy_recovery_is_immutable_and_auditable(
    tmp_path, fake_flex_provider
):
    output = tmp_path / "output"
    proxy_dir = output / "proxy/invocation-dead"
    proxy_dir.mkdir(parents=True)
    run_id = "recover-run"
    ready_path = proxy_dir / "ready.json"
    log_path = proxy_dir / "requests.jsonl"
    process_log = proxy_dir / "process.log"
    start_path = proxy_dir / "start.json"
    base_sha = contract.sha256_file(ROOT / "scripts/gpt55_run_proxy.py")
    controlled_sha = contract.sha256_file(
        ROOT / "scripts/controlled_gpt55_run_proxy.py"
    )
    provider_window = runner.flex_evidence.capture_start(fake_flex_provider.root)
    provider_contract = provider_window["contract"]
    upstream = provider_contract["origin"]
    ready = {
        "run_id": run_id,
        "pid": 999_999_999,
        "port": 32123,
        "base_url": "http://127.0.0.1:32123/v1",
        "upstream": upstream,
        "log": str(log_path),
        "base_wrapper_sha256": base_sha,
        "controlled_wrapper_sha256": controlled_sha,
    }
    health = {
        "status": "ok",
        "run_id": run_id,
        "exclusive_log": str(log_path),
        "upstream": upstream,
        "base_wrapper_sha256": base_sha,
        "controlled_wrapper_sha256": controlled_sha,
        "upstream_health": {
            "status": "ok",
            "schema": "openai-gpt55-flex-health/v1",
            "requested_model": "gpt-5.5",
            "provider_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
            "budget": {"max_cost_usd": provider_contract["max_cost_usd"]},
        },
    }
    ready_path.write_text(json.dumps(ready))
    log_path.write_text("")
    process_log.write_text("")
    start = {
        "run_id": run_id,
        "invocation_id": proxy_dir.name,
        "invocation_dir": str(proxy_dir.relative_to(output)),
        "started_at": contract.utc_now(),
        "pid": ready["pid"],
        "base_url": ready["base_url"],
        "ready": str(ready_path.relative_to(output)),
        "log": str(log_path.relative_to(output)),
        "process_log": str(process_log.relative_to(output)),
        "health": health,
        "upstream": upstream,
        "gateway_result_root": provider_contract["result_root"],
        "gateway_root_marker_sha256": provider_contract["root_marker_sha256"],
        "gateway_ready_sha256": provider_contract["ready_sha256"],
        "gateway_max_cost_usd": provider_contract["max_cost_usd"],
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "provider_window": provider_window,
    }
    contract.atomic_json_no_clobber(start_path, start)

    runner._recover_interrupted_proxies(output, run_id)
    manifest_path = proxy_dir / "manifest.json"
    manifest = contract.read_json(manifest_path)
    assert manifest["status"] == "interrupted_recovered"
    assert manifest["recovery_action"] == "original_process_not_running"
    assert manifest["start_sha256"] == contract.sha256_file(start_path)
    entries, report = answer_auditor._load_proxy_entries(output, run_id)
    assert entries == []
    assert report["invocations"][0]["status"] == "interrupted_recovered"

    original_manifest = manifest_path.read_bytes()
    runner._recover_interrupted_proxies(output, run_id)
    assert manifest_path.read_bytes() == original_manifest


def test_full_run_auditor_checks_inventory_proxy_and_complete_manifest(
    tmp_path, monkeypatch, fake_flex_provider
):
    monkeypatch.setattr(contract, "EXPECTED_QUESTIONS", 1)
    monkeypatch.setattr(contract, "EXPECTED_PRIMARY", 0)
    monkeypatch.setattr(contract, "EXPECTED_ADVERSARIAL", 1)
    monkeypatch.setattr(contract, "EXPECTED_CATEGORIES", {5: 1})
    monkeypatch.setattr(answer_auditor.contract, "EXPECTED_QUESTIONS", 1)
    monkeypatch.setattr(answer_auditor.contract, "EXPECTED_PRIMARY", 0)
    monkeypatch.setattr(answer_auditor.contract, "EXPECTED_ADVERSARIAL", 1)
    monkeypatch.setattr(answer_auditor.contract, "EXPECTED_CATEGORIES", {5: 1})

    input_dir = tmp_path / "input"
    output = tmp_path / "output"
    input_dir.mkdir()
    output.mkdir()
    raw_dataset = [
        {
            "qa": [
                {
                    "question": "What was not stated?",
                    "category": 5,
                    "evidence": ["D1:1"],
                    "adversarial_answer": "DISTRACTOR",
                }
            ]
        }
    ]
    dataset_path = input_dir / "dataset.json"
    dataset_path.write_text(json.dumps(raw_dataset))
    record = {
        "question_id": "s0_q0",
        "question": "What was not stated?",
        "gold": "DISTRACTOR",
        "category": 5,
        "evidence": ["D1:1"],
        "memories": [{"text": "ordinary memory", "date": None}],
        "retrieval": {},
    }
    (input_dir / "questions.json").write_text(json.dumps([record]))
    (input_dir / "run_manifest.json").write_text(
        json.dumps({"config": {"dataset": str(dataset_path)}})
    )
    prereg = tmp_path / "protocol.json"
    prereg.write_text("{}")
    protocol = {
        "protocol_content_sha256": "p" * 64,
        "answerer": {"prompt_source_hashes": contract.prompt_source_hashes(ROOT)},
    }
    monkeypatch.setattr(
        answer_auditor.contract,
        "validate_preregistration",
        lambda path, expected_file_sha256=None: protocol,
    )
    monkeypatch.setattr(
        answer_auditor.baseline_auditor,
        "audit",
        lambda path: {"status": "passed", "method": "bm25"},
    )

    run_id = "one-question-audit"
    proxy_dir = output / "proxy/invocation-test"
    proxy_dir.mkdir(parents=True)
    provider = fake_flex_provider.close_window(run_id)
    provider_contract = provider["contract"]
    provider_response = provider["response"]
    assert provider_response is not None
    monkeypatch.setattr(
        runner.uuid,
        "uuid4",
        lambda: SimpleNamespace(
            hex=str(provider_response["id"]).removeprefix("fake-")
        ),
    )
    fake = runner.FakeAnswerClient(
        proxy_log=proxy_dir / "requests.jsonl", run_id=run_id
    )
    attempt = output / "questions/s0_q0/attempts/attempt-0001"
    attempt.mkdir(parents=True)
    result = runner.answer_one(
        run_id=run_id,
        method="bm25",
        question_id="s0_q0",
        question=record["question"],
        memories=record["memories"],
        input_record_sha256=contract.canonical_hash(record),
        attempt_dir=attempt,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        budget_policy=contract.HARD_CAP_POLICY,
        budget_tokens=20_000,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="q" * 64,
        client=fake,
    )
    reference = contract.canonical_scoring_reference(raw_dataset, "s0_q0")
    result["scoring_reference"] = {
        "category": 5,
        "canonical_gold": reference["canonical_gold"],
        "canonical_gold_source": reference["canonical_gold_source"],
        "adversarial_distractor": reference["adversarial_distractor"],
        "evidence": reference["evidence"],
        "attached_after_answer": True,
        "baseline_input_gold_authoritative": False,
    }
    runner._write_completion(
        output_dir=output,
        question_id="s0_q0",
        attempt_dir=attempt,
        result=result,
    )
    with contract.DurableLedger(output / "run_ledger.jsonl", run_id=run_id) as ledger:
        ledger.append("synthetic_run", {})

    log_path = proxy_dir / "requests.jsonl"
    proxy_event = json.loads(log_path.read_text(encoding="utf-8"))
    proxy_event.update(provider["consumer_records"][0])
    log_path.write_text(contract.canonical_json(proxy_event) + "\n", encoding="utf-8")

    ready_path = proxy_dir / "ready.json"
    start_path = proxy_dir / "start.json"
    process_log = proxy_dir / "process.log"
    process_log.write_text("")
    base_sha = contract.sha256_file(ROOT / "scripts/gpt55_run_proxy.py")
    controlled_sha = contract.sha256_file(ROOT / "scripts/controlled_gpt55_run_proxy.py")
    upstream_sha = contract.sha256_file(
        ROOT / "scripts/gateways/openai_gpt55_flex_gateway.py"
    )
    flex_evidence_sha = contract.sha256_file(
        ROOT / "scripts/gateways/openai_gpt55_flex_gateway_evidence.py"
    )
    upstream = provider_contract["origin"]
    ready = {
        "run_id": run_id,
        "pid": 123,
        "port": 32123,
        "base_url": "http://127.0.0.1:32123/v1",
        "upstream": upstream,
        "log": str(log_path),
        "base_wrapper_sha256": base_sha,
        "controlled_wrapper_sha256": controlled_sha,
    }
    ready_path.write_text(json.dumps(ready))
    health = {
        "status": "ok",
        "run_id": run_id,
        "exclusive_log": str(log_path),
        "upstream": upstream,
        "base_wrapper_sha256": base_sha,
        "controlled_wrapper_sha256": controlled_sha,
        "upstream_health": {
            "status": "ok",
            "schema": "openai-gpt55-flex-health/v1",
            "requested_model": "gpt-5.5",
            "provider_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
            "budget": {"max_cost_usd": provider_contract["max_cost_usd"]},
        },
    }
    started_at = contract.utc_now()
    start = {
        "run_id": run_id,
        "invocation_id": proxy_dir.name,
        "invocation_dir": str(proxy_dir.relative_to(output)),
        "started_at": started_at,
        "pid": 123,
        "base_url": ready["base_url"],
        "ready": str(ready_path.relative_to(output)),
        "log": str(log_path.relative_to(output)),
        "process_log": str(process_log.relative_to(output)),
        "health": health,
        "upstream": upstream,
        "gateway_result_root": provider_contract["result_root"],
        "gateway_root_marker_sha256": provider_contract["root_marker_sha256"],
        "gateway_ready_sha256": provider_contract["ready_sha256"],
        "gateway_max_cost_usd": provider_contract["max_cost_usd"],
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "provider_window": provider["window"],
    }
    contract.atomic_json_no_clobber(start_path, start)
    proxy_manifest = {
        "run_id": run_id,
        "invocation_id": proxy_dir.name,
        "invocation_dir": str(proxy_dir.relative_to(output)),
        "started_at": started_at,
        "pid": 123,
        "base_url": ready["base_url"],
        "start": str(start_path.relative_to(output)),
        "ready": str(ready_path.relative_to(output)),
        "log": str(log_path.relative_to(output)),
        "process_log": str(process_log.relative_to(output)),
        "health": health,
        "wrapper_sha256": controlled_sha,
        "base_wrapper_sha256": base_sha,
        "upstream_proxy_sha256": upstream_sha,
        "flex_evidence_sha256": flex_evidence_sha,
        "gateway_result_root": provider_contract["result_root"],
        "gateway_root_marker_sha256": provider_contract["root_marker_sha256"],
        "gateway_ready_sha256": provider_contract["ready_sha256"],
        "gateway_max_cost_usd": provider_contract["max_cost_usd"],
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "provider_window": provider["window"],
    }
    for key, path in (
        ("start", start_path),
        ("ready", ready_path),
        ("log", log_path),
        ("process_log", process_log),
    ):
        proxy_manifest[f"{key}_sha256"] = contract.sha256_file(path)
        proxy_manifest[f"{key}_bytes"] = path.stat().st_size
    (proxy_dir / "manifest.json").write_text(json.dumps(proxy_manifest))

    python = Path(sys.executable).resolve()
    config = {
        "method": "bm25",
        "budget_policy": contract.HARD_CAP_POLICY,
        "input_dir": str(input_dir),
        "preregistration": str(prereg),
        "preregistration_sha256": "q" * 64,
        "protocol_content_sha256": protocol["protocol_content_sha256"],
        "python": str(python),
        "python_resolved": str(python),
        "python_sha256": contract.sha256_file(python),
        "runner_python_resolved": str(python),
        "runner_python_sha256": contract.sha256_file(python),
        "gateway_contract": provider_contract,
        "model_requests_authorized": True,
    }
    run_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "run_id": run_id,
        "config": config,
        "source_hashes": runner._source_hashes(),
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest))
    report = answer_auditor.audit_run(output, require_complete_manifest=False)
    assert report["inventory"]["questions"] == 1
    assert report["inventory"]["adversarial_cat5"] == 1
    assert report["calls"]["successful_logical_answer_calls"] == 1
    assert report["calls"]["client_http_attempts"] == 1
    complete = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "completed_at": contract.utc_now(),
        "inventory": report["inventory"],
        "calls": report["calls"],
        "proxy": report["proxy"],
        "run_ledger_sha256": report["run_ledger_sha256"],
        "preregistration_sha256": "q" * 64,
    }
    (output / "complete.json").write_text(json.dumps(complete))
    assert answer_auditor.audit_run(output)["status"] == "passed"
