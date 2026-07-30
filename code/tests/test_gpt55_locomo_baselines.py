from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_gpt55_locomo_baselines as auditor  # noqa: E402
import run_gpt55_locomo_baselines as runner  # noqa: E402
from src import openai_gpt55_flex_gateway as flex_gateway  # noqa: E402


EXPECTED_REGISTRY = {
    "full_context": ("run_naive.py", ("--method", "full_context")),
    "bm25": ("run_naive.py", ("--method", "bm25")),
    "mem0": ("run_mem0.py", ()),
    "amem": ("run_amem.py", ()),
    "lightmem": ("run_lightmem.py", ("--k", "20")),
    "nemori": ("run_nemori.py", ()),
    "zep": ("run_zep.py", ()),
    "memoryos": ("run_memoryos.py", ()),
    "memos": ("run_memos.py", ()),
    "simplemem": ("run_simplemem.py", ()),
    "hindsight": ("run_hindsight.py", ()),
    "memmachine": ("run_memmachine.py", ()),
    "mirix": ("run_mirix.py", ()),
    "emem": ("run_emem.py", ()),
    "evermemos": ("run_evermemos.py", ()),
}


class FakeFlexTransport:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, payload: bytes) -> flex_gateway.UpstreamResponse:
        self.calls += 1
        request = json.loads(payload)
        assert request["model"] == flex_gateway.PROVIDER_MODEL
        assert request["service_tier"] == flex_gateway.SERVICE_TIER
        body = {
            "id": f"flex-provider-response-{self.calls}",
            "model": flex_gateway.PROVIDER_MODEL,
            "service_tier": flex_gateway.SERVICE_TIER,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return flex_gateway.UpstreamResponse(
            200,
            {"x-request-id": f"physical-{self.calls}"},
            json.dumps(body).encode(),
        )


def make_active_flex_root(path: Path) -> tuple[object, dict]:
    gateway = flex_gateway.GPT55FlexGateway(
        result_root=path,
        max_cost_usd="1000",
        transport=FakeFlexTransport(),
    )
    ready = {
        "schema": "openai-gpt55-flex-ready/v1",
        "pid": 1234,
        "base_url": "http://127.0.0.1:38200/v1",
        "health_url": "http://127.0.0.1:38200/healthz",
        "requested_model": flex_gateway.REQUESTED_MODEL,
        "provider_model": flex_gateway.PROVIDER_MODEL,
        "service_tier": flex_gateway.SERVICE_TIER,
        "max_cost_usd": "1000",
        "started_at": "2026-07-13T00:00:00+00:00",
    }
    (path / "gateway_ready.json").write_text(
        json.dumps(ready, indent=2) + "\n", encoding="utf-8"
    )
    return gateway, runner.load_flex_gateway_contract(path)


def build_record(
    notes: str = "fake baseline; failed_sessions=[]",
    *,
    builder_model: str = "not_applicable",
    base_url: str = "not_applicable",
    usage_tracking: str = "not_applicable",
) -> dict:
    return {
        "question_id": "_build_stats",
        "build_time_s": 0.1,
        "num_memories": 1,
        "build_calls": 0,
        "build_tokens_in": 0,
        "build_tokens_out": 0,
        "build_llm_time_s": 0.0,
        "notes": notes,
        "builder": {
            "requested_model": builder_model,
            "base_url": base_url,
            "usage_tracking": usage_tracking,
        },
    }


def question_record() -> dict:
    return {
        "question_id": "s0_q0",
        "question": "Question?",
        "gold": "Answer",
        "category": 1,
        "evidence": ["D1:1"],
        "memories": [{"text": "Retrieved memory", "date": "2025-01-01"}],
        "retrieval": {
            "latency_s": 0.01,
            "k": 20,
            "requested_k": 20,
            "returned_count": 1,
            "calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        },
    }


def dataset_item() -> dict:
    return {
        "qa": [
            {
                "question": "Question?",
                "answer": "Answer",
                "category": 1,
                "evidence": ["D1:1"],
            }
        ]
    }


def write_fake_adapter(path: Path, *, build_calls: int = 0) -> None:
    source = (
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=int, required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--fixed')
parser.add_argument('--max-sessions')
parser.add_argument('--questions-limit', type=int)
args, unknown = parser.parse_known_args()
if unknown:
    raise SystemExit(f'unexpected args: {unknown}')
if args.fixed != '20':
    raise SystemExit('fixed adapter arguments were not supplied')
dataset = json.loads(Path(os.environ['LOCOMO_BASELINE_DATASET']).read_text())
item = dataset[args.sample]
qa_rows = item['qa'][:args.questions_limit] if args.questions_limit else item['qa']
rows = [{
    'question_id': '_build_stats',
    'build_time_s': 0.0,
    'num_memories': 1,
    'build_calls': __BUILD_CALLS__,
    'build_tokens_in': 0,
    'build_tokens_out': 0,
    'build_llm_time_s': 0.0,
    'notes': 'offline fake adapter; failed_sessions=[]',
}]
for index, qa in enumerate(qa_rows):
    gold = qa.get('answer', qa.get('adversarial_answer', ''))
    rows.append({
        'question_id': f's{args.sample}_q{index}',
        'question': qa['question'],
        'gold': str(gold),
        'category': qa['category'],
        'memories': [{'text': f'retrieved sample {args.sample}', 'date': None}],
        'retrieval': {
            'latency_s': 0.0,
            'k': 20,
            'calls': 0,
            'tokens_in': 0,
            'tokens_out': 0,
        },
    })
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(rows, ensure_ascii=False))
counter = Path(args.output).parents[1] / 'counter.txt'
with counter.open('a') as handle:
    handle.write(f'{args.sample}\\n')
"""
    ).replace("__BUILD_CALLS__", str(build_calls))
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_registry_and_default_python_are_frozen():
    assert runner.DEFAULT_PYTHON == "/opt/miniconda3/bin/python3"
    assert set(runner.METHOD_REGISTRY) == set(EXPECTED_REGISTRY)
    for method, (filename, extra_args) in EXPECTED_REGISTRY.items():
        spec = runner.METHOD_REGISTRY[method]
        assert spec.adapter.name == filename
        assert spec.extra_args == extra_args


def test_python_resolution_preserves_venv_launcher_path(tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    assert runner.resolve_python(str(venv_python)) == venv_python.absolute()


def test_formal_matrix_freezes_graphiti_as_structured_method(tmp_path):
    assert runner.FORMAL_METHODS == ("full_context", "bm25", "mem0", "zep")
    assert runner.STRUCTURED_PREREGISTRATION["method"] == "zep"
    assert "Graphiti OSS" in runner.STRUCTURED_PREREGISTRATION["implementation"]
    with pytest.raises(SystemExit):
        runner.parse_args(["--method", "simplemem", "--output-dir", str(tmp_path)])
    smoke = runner.parse_args(
        [
            "--method",
            "simplemem",
            "--scope",
            "smoke",
            "--output-dir",
            str(tmp_path),
            "--gateway-root",
            str(tmp_path / "gateway"),
            "--allow-model-requests",
        ]
    )
    assert smoke.scope == "smoke"


def test_independent_preregistration_is_frozen_and_fingerprinted(tmp_path):
    payload = runner.load_formal_preregistration()
    assert payload["formal_methods"] == ["full_context", "bm25", "mem0", "zep"]
    assert payload["structured_method"] == runner.STRUCTURED_PREREGISTRATION
    assert payload["selection"]["selection_used_test_answers"] is False
    assert payload["downstream_gates"] == {
        "R002_source_mapping": "required_not_yet_audited",
        "R004_visible_token_gate": "required_not_yet_audited",
        "formal_answer_generation": "not_started",
        "judging": "not_started",
    }
    assert runner.PREREGISTRATION.resolve() in runner.required_source_paths(
        runner.METHOD_REGISTRY["zep"]
    )
    record = runner.preregistration_record(payload)
    assert record["sha256"] == runner.sha256_file(runner.PREREGISTRATION)

    tampered_path = tmp_path / "preregistration.json"
    tampered = json.loads(json.dumps(payload))
    tampered["formal_methods"].append("simplemem")
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.BaselineRunError, match="frozen contract"):
        runner.load_formal_preregistration(tampered_path)


def test_sample_validator_rejects_forgery_empty_memory_and_failure_markers():
    rows = [build_record(), question_record()]
    assert runner.validate_sample_records(rows, dataset_item(), 0)["questions"] == 1

    forged = json.loads(json.dumps(rows))
    forged[1]["gold"] = "forged"
    with pytest.raises(runner.BaselineRunError, match="gold differs"):
        runner.validate_sample_records(forged, dataset_item(), 0)

    forged_evidence = json.loads(json.dumps(rows))
    forged_evidence[1]["evidence"] = ["D9:9"]
    with pytest.raises(runner.BaselineRunError, match="evidence differs"):
        runner.validate_sample_records(forged_evidence, dataset_item(), 0)

    wrong_count = json.loads(json.dumps(rows))
    wrong_count[1]["retrieval"]["returned_count"] = 2
    with pytest.raises(runner.BaselineRunError, match="returned_count differs"):
        runner.validate_sample_records(wrong_count, dataset_item(), 0)

    empty = json.loads(json.dumps(rows))
    empty[1]["memories"] = []
    with pytest.raises(runner.BaselineRunError, match="empty memories"):
        runner.validate_sample_records(empty, dataset_item(), 0)

    failed_session = [build_record("failed_sessions=[3]"), question_record()]
    with pytest.raises(runner.BaselineRunError, match="failed_sessions"):
        runner.validate_sample_records(failed_session, dataset_item(), 0)

    build_error = [build_record("build_error=connection refused"), question_record()]
    with pytest.raises(runner.BaselineRunError, match="build_error"):
        runner.validate_sample_records(build_error, dataset_item(), 0)

    with pytest.raises(runner.BaselineRunError, match="fallback"):
        runner.validate_sample_records(
            rows, dataset_item(), 0, log_text="LLM failed; using fallback"
        )


def exclusive_proxy_manifest(
    tmp_path: Path,
    *,
    actual_model: str = "gpt-5.5",
    request_count: int = 1,
) -> dict:
    run_dir = tmp_path / "run"
    proxy_dir = run_dir / "proxy"
    proxy_dir.mkdir(parents=True)
    gateway, provider_contract = make_active_flex_root(tmp_path / "gateway")
    provider_window = runner.flex_evidence.capture_start(tmp_path / "gateway")
    run_id = "exclusive-test-run"
    log_path = proxy_dir / "requests.jsonl"
    process_log = proxy_dir / "process.log"
    ready_path = proxy_dir / "ready.json"
    requests = []
    for index in range(1, request_count + 1):
        status, response = gateway.handle(
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": f"request {index}"}],
                "max_completion_tokens": 8,
            }
        )
        assert status == 200
        meta = response["flex_gateway_meta"]
        requests.append({
            "run_id": run_id,
            "started_at": "2026-07-13T00:00:02+00:00",
            "finished_at": "2026-07-13T00:00:03+00:00",
            "status": "success",
            "http_status": 200,
            "requested_model": "gpt-5.5",
            "requested_service_tier": None,
            "actual_model": actual_model,
            "provider_actual_model": meta["provider_actual_model"],
            "service_tier": meta["service_tier"],
            "gateway_request_id": meta["request_id"],
            "gateway_request_sha256": meta["request_sha256"],
            "provider_request_sha256": meta["provider_request_sha256"],
            "response_id": response["id"],
            "request_sha256": f"{index:064x}",
            "error": None,
        })
    provider_window = runner.flex_evidence.capture_end(provider_window)
    log_path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    process_log.write_text("")
    upstream = provider_contract["origin"]
    ready = {
        "run_id": run_id,
        "pid": 1234,
        "port": 32123,
        "base_url": "http://127.0.0.1:32123/v1",
        "upstream": upstream,
        "log": str(log_path),
        "started_at": "2026-07-13T00:00:00+00:00",
    }
    ready_path.write_text(json.dumps(ready))
    invocation = {
        "run_id": run_id,
        "status": "captured",
        "started_at": "2026-07-13T00:00:01+00:00",
        "finished_at": "2026-07-13T00:00:04+00:00",
        "base_url": ready["base_url"],
        "upstream": upstream,
        "log": str(log_path.relative_to(run_dir)),
        "ready": str(ready_path.relative_to(run_dir)),
        "process_log": str(process_log.relative_to(run_dir)),
        "pid": ready["pid"],
        "returncode": -15,
        "health": {
            "status": "ok",
            "run_id": run_id,
            "exclusive_log": str(log_path),
            "upstream": upstream,
            "upstream_health": gateway.health(),
        },
        "wrapper_sha256": runner.sha256_file(runner.RUN_PROXY),
        "gateway_source_sha256": runner.sha256_file(runner.FLEX_GATEWAY),
        "gateway_root_marker_sha256": provider_contract["root_marker_sha256"],
        "gateway_result_root": provider_contract["result_root"],
        "provider_model": runner.FLEX_PROVIDER_MODEL,
        "service_tier": runner.FLEX_SERVICE_TIER,
        "provider_window": provider_window,
        "exact_run_linkage": True,
        "log_sha256": runner.sha256_file(log_path),
        "log_bytes": log_path.stat().st_size,
        "process_log_sha256": runner.sha256_file(process_log),
        "process_log_bytes": process_log.stat().st_size,
        "ready_sha256": runner.sha256_file(ready_path),
        "ready_bytes": ready_path.stat().st_size,
    }
    return {
        "output_dir": str(run_dir),
        "provider_contract": provider_contract,
        "proxy_evidence": {
            "mode": "managed_exclusive",
            "exact_run_linkage": True,
            "invocations": [invocation],
        },
    }


def test_proxy_audit_requires_exact_per_run_gpt55_evidence(tmp_path):
    manifest = exclusive_proxy_manifest(tmp_path)
    spec = runner.MethodSpec(tmp_path / "fake.py", expects_llm=True)
    report = auditor.audit_proxy_evidence(manifest, spec)
    assert report["actual_model_verification"] == "per_request"
    assert report["exact_run_linkage"] is True
    assert report["successes"] == 1

    bad_manifest = exclusive_proxy_manifest(tmp_path / "bad", actual_model="gpt-5.4")
    with pytest.raises(auditor.BaselineAuditError, match="GPT-5.5"):
        auditor.audit_proxy_evidence(bad_manifest, spec)

    deterministic = {
        "proxy_evidence": {
            "mode": "not_applicable",
            "exact_run_linkage": False,
            "invocations": [],
        }
    }
    deterministic_report = auditor.audit_proxy_evidence(
        deterministic,
        runner.MethodSpec(tmp_path / "deterministic.py", expects_llm=False),
    )
    assert deterministic_report["mode"] == "not_applicable"


def test_interrupted_exclusive_proxy_is_captured_for_resume(tmp_path):
    manifest = exclusive_proxy_manifest(tmp_path)
    invocation = manifest["proxy_evidence"]["invocations"][0]
    invocation["status"] = "running"
    invocation["pid"] = 999_999_999
    ready_path = Path(manifest["output_dir"]) / invocation["ready"]
    ready = json.loads(ready_path.read_text())
    ready["pid"] = invocation["pid"]
    ready_path.write_text(json.dumps(ready))
    invocation.pop("finished_at")
    invocation.pop("returncode")
    runner.recover_interrupted_managed_proxies(
        manifest,
        Path(manifest["output_dir"]),
    )
    assert invocation["status"] == "interrupted_captured"
    assert invocation["recovery_action"] == "original_process_not_running"
    report = auditor.audit_proxy_evidence(
        manifest,
        runner.MethodSpec(tmp_path / "adapter.py", expects_llm=True),
    )
    assert report["successes"] == 1


def test_exclusive_proxy_forwards_only_gpt55_to_local_upstream(tmp_path):
    class FakeUpstream(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            payload = json.dumps(
                {
                    "status": "ok",
                    "schema": runner.FLEX_HEALTH_SCHEMA,
                    "requested_model": runner.FLEX_REQUESTED_MODEL,
                    "provider_model": runner.FLEX_PROVIDER_MODEL,
                    "service_tier": runner.FLEX_SERVICE_TIER,
                    "budget": {"max_cost_usd": "1000"},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            payload = json.dumps(
                {
                    "id": "fake-response-id",
                    "model": request["model"],
                    "service_tier": runner.FLEX_SERVICE_TIER,
                    "choices": [],
                    "flex_gateway_meta": {
                        "request_id": "fake-gateway-request-id",
                        "request_sha256": "a" * 64,
                        "provider_request_sha256": "b" * 64,
                        "provider_actual_model": runner.FLEX_PROVIDER_MODEL,
                        "service_tier": runner.FLEX_SERVICE_TIER,
                    },
                }
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
    log_path = tmp_path / "exclusive.jsonl"
    ready_path = tmp_path / "ready.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(runner.RUN_PROXY),
            "--upstream",
            f"http://127.0.0.1:{upstream.server_port}",
            "--log",
            str(log_path),
            "--ready",
            str(ready_path),
            "--run-id",
            "wrapper-unit-test",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 10
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_path.is_file()
        ready = json.loads(ready_path.read_text())
        health = json.loads(
            opener.open(
                f"http://127.0.0.1:{ready['port']}/healthz", timeout=2
            ).read()
        )
        assert health["status"] == "ok"
        request = urllib.request.Request(
            ready["base_url"] + "/chat/completions",
            data=json.dumps({"model": "gpt-5.5", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = json.loads(opener.open(request, timeout=2).read())
        assert response["model"] == "gpt-5.5"
        rejected = urllib.request.Request(
            ready["base_url"] + "/chat/completions",
            data=json.dumps({"model": "gpt-4.1", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            opener.open(rejected, timeout=2)
        assert exc_info.value.code == 400
    finally:
        process.terminate()
        process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["success", "error"]
    assert entries[0]["actual_model"] == "gpt-5.5"
    assert all(entry["run_id"] == "wrapper-unit-test" for entry in entries)


def test_launcher_resume_scope_stale_hash_lock_and_manifest_tamper(
    tmp_path, monkeypatch
):
    fake_adapter = tmp_path / "fake_adapter.py"
    write_fake_adapter(fake_adapter)
    fake_spec = runner.MethodSpec(
        fake_adapter,
        extra_args=("--fixed", "20"),
        expects_llm=False,
        usage_tracking="not_applicable",
        max_sample_workers=2,
    )
    monkeypatch.setitem(runner.METHOD_REGISTRY, "full_context", fake_spec)
    run_dir = tmp_path / "run"
    counter = run_dir / "counter.txt"
    args = [
        "--method",
        "full_context",
        "--output-dir",
        str(run_dir),
        "--python",
        sys.executable,
        "--sample-workers",
        "2",
        "--retries",
        "1",
    ]

    assert runner.main(args) == 0
    assert len(counter.read_text().splitlines()) == 10
    questions = json.loads((run_dir / "questions.json").read_text())
    assert len(questions) == 10 + 1986
    assert sum(record.get("category") in {1, 2, 3, 4} for record in questions[10:]) == 1540
    assert sum(record.get("category") == 5 for record in questions[10:]) == 446
    assert all(isinstance(record["evidence"], list) for record in questions[10:])
    assert all(
        record["retrieval"]["returned_count"] == len(record["memories"])
        for record in questions[10:]
    )
    report = auditor.audit(run_dir)
    assert report["status"] == "passed"
    assert report["scope"]["questions"] == 1986
    assert report["scope"]["sample_count"] == 10
    assert report["evaluation"]["answerer"] == "not_run"

    # A second identical invocation validates checkpoints and invokes no adapter.
    assert runner.main(args) == 0
    assert len(counter.read_text().splitlines()) == 10
    assert auditor.audit(run_dir)["status"] == "passed"

    # The auditor must reject an active launcher lock.
    lock_handle = (run_dir / ".launcher.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(auditor.BaselineAuditError, match="launcher is active"):
            auditor.audit(run_dir)
    finally:
        lock_handle.close()

    # Tampering with the assembler manifest is detected independently.
    assembly_path = run_dir / "inputs_manifest.json"
    original_assembly = assembly_path.read_bytes()
    assembly = json.loads(original_assembly)
    assembly["output"]["question_records"] = 1
    assembly_path.write_text(json.dumps(assembly))
    with pytest.raises(auditor.BaselineAuditError, match="output record differs"):
        auditor.audit(run_dir)
    assembly_path.write_bytes(original_assembly)
    assert auditor.audit(run_dir)["status"] == "passed"

    # Modifying the adapter changes the source fingerprint.  Resume must refuse
    # without deleting the previously complete assembled input.
    previous_questions_hash = runner.sha256_file(run_dir / "questions.json")
    fake_adapter.write_text(fake_adapter.read_text() + "\n# changed\n")
    assert runner.main(args) == 1
    assert len(counter.read_text().splitlines()) == 10
    assert runner.sha256_file(run_dir / "questions.json") == previous_questions_hash


def test_llm_launcher_records_exclusive_proxy_and_builder_contract(
    tmp_path, monkeypatch
):
    fake_adapter = tmp_path / "fake_llm_adapter.py"
    write_fake_adapter(fake_adapter, build_calls=1)
    fake_spec = runner.MethodSpec(
        fake_adapter,
        extra_args=("--fixed", "20"),
        expects_llm=True,
        usage_tracking="in_process",
        max_sample_workers=2,
    )
    monkeypatch.setitem(runner.METHOD_REGISTRY, "mem0", fake_spec)
    run_dir = tmp_path / "run"
    fixture = exclusive_proxy_manifest(tmp_path, request_count=10)

    class FinishedProxy:
        def poll(self):
            return -15

    def fake_start(args, out_dir):
        del args
        assert out_dir == run_dir
        return FinishedProxy(), fixture["proxy_evidence"]["invocations"][0]

    monkeypatch.setattr(runner, "start_managed_proxy", fake_start)
    monkeypatch.setattr(
        runner,
        "finish_managed_proxy",
        lambda process, record, out_dir: None,
    )
    assert (
        runner.main(
            [
                "--method",
                "mem0",
                    "--output-dir",
                    str(run_dir),
                    "--gateway-root",
                    str(tmp_path / "gateway"),
                    "--allow-model-requests",
                "--python",
                sys.executable,
                "--sample-workers",
                "2",
                "--retries",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["builder_contract"] == {
        "requested_model": "gpt-5.5",
        "base_url": (
            "managed-exclusive-proxy->openai-api-flex-gateway->"
            "http://127.0.0.1:38200/v1"
        ),
        "usage_tracking": "in_process",
    }
    report = auditor.audit(run_dir)
    assert report["proxy_evidence"]["exact_run_linkage"] is True
    assert report["proxy_evidence"]["successes"] == 10
    assert report["proxy_evidence"]["call_count_comparison"]["exact_match"] is True


def test_smoke_scope_is_fingerprinted_and_has_no_formal_assembly(
    tmp_path, monkeypatch
):
    fake_adapter = tmp_path / "fake_smoke_adapter.py"
    write_fake_adapter(fake_adapter)
    fake_spec = runner.MethodSpec(
        fake_adapter,
        extra_args=("--fixed", "20"),
        expects_llm=False,
        usage_tracking="not_applicable",
    )
    monkeypatch.setitem(runner.METHOD_REGISTRY, "full_context", fake_spec)
    run_dir = tmp_path / "smoke"
    assert (
        runner.main(
            [
                "--method",
                "full_context",
                "--scope",
                "smoke",
                "--samples",
                "2",
                "--max-sessions",
                "1",
                "--questions-limit",
                "2",
                "--output-dir",
                str(run_dir),
                "--python",
                sys.executable,
                "--retries",
                "1",
            ]
        )
        == 0
    )
    report = auditor.audit(run_dir)
    assert report["scope"]["kind"] == "smoke"
    assert report["scope"]["samples"] == [2]
    assert report["scope"]["sample_count"] == 1
    assert report["scope"]["questions"] == 2
    assert sum(report["scope"]["categories"].values()) == 2
    assert report["assembly"] == {"status": "not_applicable_smoke_scope"}
    assert not (run_dir / "questions.json").exists()
    assert not (run_dir / "inputs_manifest.json").exists()
    sample_rows = json.loads((run_dir / "sample2_questions.json").read_text())
    assert len(sample_rows) == 3
    assert all(row["evidence"] for row in sample_rows[1:])


def test_environment_is_scrubbed_and_runtime_route_is_exact(tmp_path, monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "JUDGE_MODEL",
        "JUDGE_KEY",
        "ANSWERER_MODEL",
        "CHATGPT_PROXY_LOG",
        "UNRELATED_SECRET",
    ):
        monkeypatch.setenv(key, "must-not-pass")
    args = argparse.Namespace(
        api_key="x",
        base_url="http://127.0.0.1:32123/v1",
        python=sys.executable,
        runtime_python=Path(sys.executable).resolve(),
    )
    spec = runner.MethodSpec(
        tmp_path / "adapter.py",
        runtime_python_env="MIRIX_PY",
    )
    diagnostics = tmp_path / "diagnostics"
    env = runner.experiment_env(args, runner.DATASET, spec, diagnostics)
    assert not {
        "OPENROUTER_API_KEY",
        "JUDGE_MODEL",
        "JUDGE_KEY",
        "ANSWERER_MODEL",
        "CHATGPT_PROXY_LOG",
        "UNRELATED_SECRET",
    } & set(env)
    assert env["OPENAI_BASE_URL"] == args.base_url
    assert env["BUILDER_MODEL"] == "gpt-5.5"
    assert env["MIRIX_PY"] == str(args.runtime_python)


def test_path_archive_and_diagnostic_guards(tmp_path):
    symlink = tmp_path / "repo-link"
    symlink.symlink_to(runner.ROOT, target_is_directory=True)
    args = argparse.Namespace(
        output_dir=symlink,
        dataset=runner.DATASET,
        method="full_context",
        proxy_log=None,
    )
    primary = Path(sys.executable).resolve()
    with pytest.raises(runner.BaselineRunError, match="output directory"):
        runner.validate_run_paths(args, primary, primary)

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    partials = tmp_path / "partials"
    runner.archive_path(artifact, partials, "../../outside/attempt")
    archived = list(partials.iterdir())
    assert len(archived) == 1
    assert archived[0].parent == partials
    assert not (tmp_path / "outside").exists()

    diagnostic = tmp_path / "run" / "diagnostics" / "sample0" / "token" / "server.log"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("server started\n")
    records = [
        {
            "path": str(diagnostic),
            "sha256": runner.sha256_file(diagnostic),
            "bytes": diagnostic.stat().st_size,
        }
    ]
    runner.validate_diagnostic_records(records, tmp_path / "run", required=True)
    diagnostic.write_text("Traceback (most recent call last): failure\n")
    records[0].update(
        sha256=runner.sha256_file(diagnostic),
        bytes=diagnostic.stat().st_size,
    )
    with pytest.raises(runner.BaselineRunError, match="fatal marker"):
        runner.validate_diagnostic_records(records, tmp_path / "run", required=True)


def test_failed_run_has_no_assembled_complete_marker(tmp_path, monkeypatch):
    fake_adapter = tmp_path / "empty_adapter.py"
    fake_adapter.write_text(
        """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--sample'); p.add_argument('--output'); p.add_argument('--fixed'); a = p.parse_args()
Path(a.output).write_text(json.dumps([]))
"""
    )
    fake_spec = runner.MethodSpec(
        fake_adapter, extra_args=("--fixed", "20"), expects_llm=False
    )
    monkeypatch.setitem(runner.METHOD_REGISTRY, "full_context", fake_spec)
    run_dir = tmp_path / "failed"
    code = runner.main(
        [
            "--method",
            "full_context",
            "--output-dir",
            str(run_dir),
            "--python",
            sys.executable,
            "--retries",
            "1",
        ]
    )
    assert code == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["assembly"]["status"] != "inputs_complete"
    assert not (run_dir / "questions.json").exists()
    assert not (run_dir / "inputs_manifest.json").exists()


def test_standalone_assembler_rejects_collisions_and_removes_stale_marker(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# adapter\n")
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    sample0 = input_dir / "sample0_questions.json"
    sample0.write_text("[]")
    common = [
        sys.executable,
        str(runner.ASSEMBLER),
        "--method",
        "test",
        "--input-dir",
        str(input_dir),
        "--dataset",
        str(runner.DATASET),
        "--adapter",
        str(adapter),
        "--builder-model",
        "not_applicable",
        "--builder-base-url",
        "not_applicable",
        "--usage-tracking",
        "not_applicable",
    ]
    collision_manifest = tmp_path / "collision-manifest.json"
    collision = subprocess.run(
        [
            *common,
            "--output",
            str(sample0),
            "--manifest",
            str(collision_manifest),
        ],
        cwd=runner.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert collision.returncode != 0
    assert "distinct paths" in collision.stderr

    output = tmp_path / "questions.json"
    manifest = tmp_path / "inputs_manifest.json"
    output.write_text('[{"stale": true}]')
    manifest.write_text('{"status": "inputs_complete"}')
    malformed = subprocess.run(
        [
            *common,
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=runner.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert malformed.returncode != 0
    assert not manifest.exists()


def test_launcher_rejects_non_pinned_dataset(tmp_path):
    custom_dataset = tmp_path / "locomo10.json"
    custom_dataset.write_bytes(runner.DATASET.read_bytes())
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--method",
                "full_context",
                "--dataset",
                str(custom_dataset),
            ]
        )


def test_auditor_failure_removes_old_pass_marker(tmp_path, monkeypatch):
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / "audit.json").write_text('{"status":"passed"}')
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_gpt55_locomo_baselines.py", str(run_dir)],
    )
    with pytest.raises(SystemExit):
        auditor.main()
    assert not (run_dir / "audit.json").exists()
