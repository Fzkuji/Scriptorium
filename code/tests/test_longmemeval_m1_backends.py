import importlib.metadata
import json
from contextlib import nullcontext
from pathlib import Path

import pytest


try:
    importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    pytest.skip(
        "formal LongMemEval backend tests require the pinned M1 environment",
        allow_module_level=True,
    )

from baselines.longmemeval_m1 import audit_longmemeval_m1_backends as auditor
from baselines.longmemeval_m1 import longmemeval_m1_backend_contract as backend_contract
from baselines.longmemeval_m1 import longmemeval_m1_contract as contract
from baselines.longmemeval_m1 import longmemeval_shared_answer_contract as shared_contract
from baselines.longmemeval_m1 import run_longmemeval_m1_backends as runner
from baselines.longmemeval_m1 import run_longmemeval_m1_baselines as base_runner
from scripts.evaluation import durable_model_ledger as durable


PREREGISTRATION = (
    Path(__file__).resolve().parents[1]
    / "results/gpt55-longmemeval-m1-baselines-20260714/"
    "formal_matrix_preregistration.json"
)


class FakeBackendAdapter:
    """Deterministic no-network adapter used to test execution semantics only."""

    observer = None

    def __init__(
        self,
        *,
        workspace,
        item,
        item_index,
        proxy_base_url,
        proxy_log,
        ledger,
        artifact_root,
        tokenizer,
        formal,
    ):
        del proxy_base_url, proxy_log, ledger, artifact_root, tokenizer, formal
        self.method = "fake"
        self.workspace = Path(workspace["workspace"])
        self.item = item
        self.item_index = item_index
        self.sessions = []
        self.closed = False

    def operation(self, operation_id):
        del operation_id
        return nullcontext()

    def ingest(self, session):
        self.sessions.append(dict(session))
        contract.atomic_json_replace(
            self.workspace / "fake_store.json",
            {
                "item_index": self.item_index,
                "source_session_ids": [
                    value["source_session_id"] for value in self.sessions
                ],
            },
        )
        return {
            "backend": self.method,
            "source_session_id": session["source_session_id"],
            "dataset_session_index": session["dataset_session_index"],
            "memory_ids": [f"memory-{session['dataset_session_index']:04d}"],
        }

    def retrieve(self, question):
        del question
        selected = self.sessions[:2]
        records = []
        for rank, session in enumerate(selected):
            text = (
                "literal <|endoftext|> source fact"
                if rank == 0
                else "second source fact " * 30_000
            )
            records.append(
                {
                    "rank": rank,
                    "backend_record_id": f"record-{rank:04d}",
                    "text": text,
                    "score": 1.0 - rank / 10,
                    "source_session_ids": [session["source_session_id"]],
                    "source_dataset_session_indices": [
                        session["dataset_session_index"]
                    ],
                    "source_document_sha256s": [session["text_sha256"]],
                    "backend_trace": {"synthetic": True},
                }
            )
        return records

    def close(self):
        if not self.closed:
            contract.atomic_json_replace(
                self.workspace / "fake_store.json",
                {
                    "item_index": self.item_index,
                    "source_session_ids": [
                        value["source_session_id"] for value in self.sessions
                    ],
                    "closed": True,
                },
            )
            self.closed = True


def fake_factory(method, **kwargs):
    adapter = FakeBackendAdapter(**kwargs)
    adapter.method = method
    return adapter


def make_plan(tmp_path, method="mem0", scope="smoke", item_index=0):
    output_root = tmp_path / f"{method}-{scope}-output"
    plan_path = tmp_path / f"{method}-{scope}-plan.json"
    base_runner.create_backend_plan(
        method=method,
        scope=scope,
        item_index=item_index if scope == "smoke" else None,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=PREREGISTRATION,
        output_root=output_root,
        output_path=plan_path,
    )
    return plan_path, output_root


def execute_synthetic(plan_path):
    return runner.execute_backend_plan(
        plan_path=plan_path,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=PREREGISTRATION,
        adapter_factory=fake_factory,
        proxy_base_url="http://127.0.0.1:9/v1",
        proxy_log=None,
        formal=False,
    )


@pytest.mark.parametrize("method", ["mem0", "graphiti"])
def test_one_item_execution_gate_shared_interface_and_independent_audit(
    tmp_path, method
):
    plan_path, output_root = make_plan(tmp_path, method=method)

    manifest = execute_synthetic(plan_path)

    assert manifest["status"] == "inputs_complete"
    assert manifest["completed_items"] == manifest["expected_items"] == 1
    assert manifest["model_calls"] == manifest["network_calls"] == 0
    item_record = manifest["items"][0]
    item_dir = output_root / method / "items" / item_record["item_dir"]
    answer = contract.read_json(item_dir / "answer_input.json")
    tokenizer = contract.FormalTokenizer.resolve()
    validated = shared_contract.validate_frozen_answer_input(
        answer,
        method=method,
        tokenizer=tokenizer,
        expected_index=0,
        expected_question_id="e47becba",
    )
    assert validated["context_visible_tokens"] == 20_000
    assert answer["context"]["budget"]["truncated_events"] == 1
    assert len(answer["context"]["source_session_ids"]) == 2
    assert (item_dir / "backend/fake_store.json").is_file()
    assert (item_dir / "source_trace.json").is_file()
    assert (item_dir / "retrieval_budget.manifest.json").is_file()

    report = auditor.audit_backend_run(
        plan_path=plan_path,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=PREREGISTRATION,
        formal=False,
    )
    assert report["status"] == "passed"
    assert report["scope"] == "smoke"
    assert report["items"] == 1
    assert report["unique_workspace_paths"] == 1


def test_resume_rejects_orphan_attempt_before_adapter_creation(tmp_path):
    plan_path, output_root = make_plan(tmp_path)
    plan = contract.read_json(plan_path)
    record = plan["items"][0]
    item_dir = (
        output_root
        / "mem0/items"
        / contract.item_directory_name(record["dataset_index"], record["question_id"])
    )
    item_dir.mkdir(parents=True)
    contract.append_jsonl_fsync(
        item_dir / "attempts.jsonl",
        {
            "event": "attempt_started",
            "attempt_id": "orphan-attempt",
            "method": "mem0",
            "dataset_index": 0,
            "question_id": record["question_id"],
        },
    )

    with pytest.raises(runner.BackendExecutionError, match="unresolved attempt"):
        execute_synthetic(plan_path)

    assert not Path(record["workspace"]["workspace"]).exists()


def test_no_clobber_existing_workspace_is_rejected_before_attempt(tmp_path):
    plan_path, _ = make_plan(tmp_path)
    plan = contract.read_json(plan_path)
    record = plan["items"][0]
    workspace = Path(record["workspace"]["workspace"])
    workspace.mkdir(parents=True)
    (workspace / "foreign.txt").write_text("not this run", encoding="utf-8")

    with pytest.raises(
        runner.BackendExecutionError, match="artifacts without an attempt ledger"
    ):
        execute_synthetic(plan_path)

    item_dir = workspace.parent
    assert not (item_dir / "attempts.jsonl").exists()
    with pytest.raises(
        runner.BackendExecutionError, match="artifacts without an attempt ledger"
    ):
        execute_synthetic(plan_path)
    assert (workspace / "foreign.txt").read_text(encoding="utf-8") == "not this run"


def test_auditor_rejects_retrieval_tamper(tmp_path):
    plan_path, output_root = make_plan(tmp_path)
    manifest = execute_synthetic(plan_path)
    item_dir = output_root / "mem0/items" / manifest["items"][0]["item_dir"]
    retrieval_path = item_dir / "retrieval_raw.json"
    value = contract.read_json(retrieval_path)
    value["records"][0]["text"] = "tampered"
    contract.atomic_json_replace(retrieval_path, value)

    with pytest.raises(auditor.BackendAuditError, match="retrieval trace differs"):
        auditor.audit_backend_run(
            plan_path=plan_path,
            dataset_path=contract.DEFAULT_DATASET,
            preregistration_path=PREREGISTRATION,
            formal=False,
        )


def test_formal_model_summary_rejects_wrong_actual_model(tmp_path):
    artifact_root = tmp_path / "model_evidence"
    ledger = durable.HashChainLedger(
        artifact_root / "ledger.jsonl", run_id="wrong-model-test"
    )
    operation_artifact = artifact_root / "operations/ingest.json"
    request_path = artifact_root / "calls/request.json"
    response_path = artifact_root / "calls/response.json"
    request_sha = durable.atomic_json(request_path, {"request": True})
    response_sha = durable.atomic_json(response_path, {"response": True})
    ledger.append(
        "operation_started",
        operation_id="ingest-0000",
        kind="history_ingest",
        operation_input_sha256="a" * 64,
    )
    ledger.append(
        "model_call_started",
        operation_id="ingest-0000",
        logical_call_id="wrong-model-test:ingest-0000:call-0001",
        request_path="calls/request.json",
        request_sha256=request_sha,
        requested_model="gpt-5.5",
    )
    ledger.append(
        "model_call_finished",
        operation_id="ingest-0000",
        logical_call_id="wrong-model-test:ingest-0000:call-0001",
        response_path="calls/response.json",
        response_sha256=response_sha,
        response_id="response-1",
        response_model="gpt-5.4",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        proxy_evidence={
            "mode": "exclusive_proxy",
            "events": [{"status": "success"}],
            "upstream_http_attempts": 1,
        },
    )
    operation_sha = durable.atomic_json(operation_artifact, {"complete": True})
    ledger.append(
        "operation_committed",
        operation_id="ingest-0000",
        kind="history_ingest",
        operation_input_sha256="a" * 64,
        artifact_path="operations/ingest.json",
        artifact_sha256=operation_sha,
    )

    with pytest.raises(
        backend_contract.BackendContractError, match="actual model differs"
    ):
        backend_contract.model_ledger_summary(
            ledger_path=artifact_root / "ledger.jsonl",
            artifact_root=artifact_root,
            formal=True,
        )


def test_exclusive_proxy_slice_rejects_unrelated_event(tmp_path):
    log = tmp_path / "exclusive.jsonl"
    log.write_bytes(b"")
    start = durable.proxy_prefix(log)
    events = [
        {
            "event_id": "one",
            "logical_call_id": "call-1",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
        },
        {
            "event_id": "foreign",
            "logical_call_id": "foreign-call",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
        },
    ]
    with log.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    with pytest.raises(
        backend_contract.BackendContractError, match="missing or unrelated"
    ):
        backend_contract.proxy_slice(
            path=log,
            start=start,
            logical_call_ids=["call-1"],
            formal=True,
        )


def test_proxy_slice_allows_retry_and_survives_later_item_append(tmp_path):
    log = tmp_path / "exclusive.jsonl"
    log.write_bytes(b"")
    start = durable.proxy_prefix(log)
    retry_events = [
        {
            "event_id": "retry-error",
            "logical_call_id": "call-1",
            "status": "error",
            "requested_model": "gpt-5.5",
            "actual_model": None,
        },
        {
            "event_id": "retry-success",
            "logical_call_id": "call-1",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
        },
    ]
    with log.open("a", encoding="utf-8") as handle:
        for event in retry_events:
            handle.write(json.dumps(event) + "\n")
    value = backend_contract.proxy_slice(
        path=log,
        start=start,
        logical_call_ids=["call-1"],
        formal=True,
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_id": "later-item",
                    "logical_call_id": "call-2",
                    "status": "success",
                    "requested_model": "gpt-5.5",
                    "actual_model": "gpt-5.5",
                }
            )
            + "\n"
        )

    backend_contract.validate_proxy_slice(value, formal=True)
    assert value["event_count"] == 2


def test_existing_formal_plans_have_500_distinct_item_workspaces():
    root = PREREGISTRATION.parent / "backend_plans"
    for method in ("mem0", "graphiti"):
        plan, _, _ = backend_contract.validate_plan(
            plan_path=root / f"{method}-formal.json",
            dataset_path=contract.DEFAULT_DATASET,
            preregistration_path=PREREGISTRATION,
        )
        workspaces = [record["workspace"]["workspace"] for record in plan["items"]]
        assert plan["scope"] == "formal"
        assert plan["item_count"] == len(workspaces) == 500
        assert len(set(workspaces)) == 500


def test_existing_smoke_and_formal_item_zero_collision_is_detectable():
    root = PREREGISTRATION.parent / "backend_plans"
    for method in ("mem0", "graphiti"):
        smoke = contract.read_json(root / f"{method}-smoke-item0000.json")
        formal = contract.read_json(root / f"{method}-formal.json")
        assert (
            smoke["items"][0]["workspace"]["workspace"]
            == formal["items"][0]["workspace"]["workspace"]
        )
