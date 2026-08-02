from __future__ import annotations

import copy
import json
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import audit_r115_beam_controls as auditor
from scripts import r115_beam_control_contract as contract
from scripts import run_r115_beam_controls as runner
from scripts.gateways import openai_gpt55_flex_gateway as flex_gateway
from scripts.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence
from scripts.evaluation import durable_model_ledger as durable
from scripts.evaluation import visible_token_budget as visible


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/beam_hf_fixture.json"


class FakeResponse:
    def __init__(self, response_id: str, answer: str = "fixture answer") -> None:
        self.id = response_id
        self.model = "gpt-5.5"
        self.usage = {
            "prompt_tokens": 101,
            "completion_tokens": 7,
            "total_tokens": 108,
        }
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=f"<answer>{answer}</answer>")
            )
        ]

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return {
            "id": self.id,
            "model": self.model,
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 7,
                "total_tokens": 108,
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": self.choices[0].message.content,
                    }
                }
            ],
        }


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(copy.deepcopy(kwargs))
        return FakeResponse(f"fixture-response-{len(self.calls):04d}")


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        return None


class FakeFlexTransport:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.lock = threading.Lock()

    def send(self, payload: bytes) -> flex_gateway.UpstreamResponse:
        with self.lock:
            self.payloads.append(payload)
            index = len(self.payloads)
        response = {
            "id": f"chatcmpl-r115-{index}",
            "object": "chat.completion",
            "created": 1,
            "model": flex_gateway.PROVIDER_MODEL,
            "service_tier": flex_gateway.SERVICE_TIER,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<answer>fake flex answer</answer>",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 8,
                "total_tokens": 108,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
        return flex_gateway.UpstreamResponse(
            200,
            {"x-request-id": f"provider-r115-{index}"},
            json.dumps(response).encode(),
        )


class GatewayResponse:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value
        self.id = value["id"]
        self.model = value["model"]
        self.usage = value["usage"]
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=value["choices"][0]["message"]["content"]
                )
            )
        ]

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return copy.deepcopy(self.value)


class GatewayCompletions:
    def __init__(self, gateway: flex_gateway.GPT55FlexGateway) -> None:
        self.gateway = gateway

    def create(self, **kwargs: Any) -> GatewayResponse:
        status, value = self.gateway.handle(kwargs)
        assert status == 200
        return GatewayResponse(value)


class GatewayClient:
    def __init__(self, gateway: flex_gateway.GPT55FlexGateway) -> None:
        self.chat = SimpleNamespace(completions=GatewayCompletions(gateway))


class FakeMem0Backend:
    def __init__(
        self,
        *,
        workspace: Path,
        conversation: dict[str, Any],
        ledger: durable.HashChainLedger,
        model_root: Path,
        tokenizer: visible.TokenCounter,
        fake_client: FakeClient,
        **_: Any,
    ) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "state.json").write_text(
            json.dumps(
                {
                    "chat_size": conversation["chat_size"],
                    "conversation_index": conversation["conversation_index"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.conversation = conversation
        self.observer = durable.DurableModelObserver(
            ledger=ledger,
            artifact_root=model_root,
            token_counter=tokenizer,
            expected_model=contract.REQUESTED_MODEL,
            proxy_log=None,
            formal=False,
        )
        self.observer.install(fake_client.chat.completions)

    def operation(self, operation_id: str):
        return self.observer.operation(operation_id)

    def ingest(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_index": session["session_index"],
            "source_chat_ids": session["source_chat_ids"],
            "source_document_sha256": session["document_sha256"],
            "result_count": 1,
            "memory_ids": [f"memory-{session['session_index']}"]
        }

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        del question
        session = self.conversation["sessions"][0]
        return [
            {
                "rank": 0,
                "record_id": "fake-memory-0",
                "text": "The selected session contains the relevant fact.",
                "score": 0.9,
                "source_chat_ids": session["source_chat_ids"],
                "session_index": session["session_index"],
                "source_document_sha256": session["document_sha256"],
                "backend_metadata_sha256": contract.canonical_hash(
                    {"fixture": True}
                ),
            }
        ]

    def close(self) -> None:
        self.observer.restore()


def preregistration(tmp_path: Path) -> Path:
    path = tmp_path / "preregistration.json"
    runner.create_preregistration(path)
    return path


def run_args(
    tmp_path: Path,
    *,
    condition: str,
    preregistration_path: Path,
    client: FakeClient,
    backend_factory=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        condition=condition,
        output_root=(tmp_path / "run").resolve(),
        preregistration=preregistration_path.resolve(),
        dataset_cache_dir=contract.HF_CACHE.resolve(),
        gateway_root=None,
        resume=False,
        allow_model_requests=False,
        fixture=FIXTURE.resolve(),
        client_factory=lambda: client,
        backend_factory=backend_factory,
    )


def test_exact_shared_prefix_counter_matches_full_request() -> None:
    tokenizer = visible.TokenCounter.resolve(
        requested_model=contract.REQUESTED_MODEL,
        fallback_encoding=contract.TOKEN_ENCODING,
        allow_byte_fallback=False,
    )
    evidence = "多语言 evidence with JSON escapes: \\\" and newline\n" * 200
    counter, descriptor = runner.exact_prompt_token_counter(tokenizer, evidence)
    assert descriptor["version"] == contract.EXACT_PROMPT_COUNT_VERSION
    for question in ("plain question?", "引号 \\\" 与换行\n", "emoji: 🧪"):
        assert counter(question) == runner.rendered_request_tokens(
            tokenizer, runner.answer_messages(question, evidence)
        )


def test_frozen_bm25_index_is_score_and_rank_equivalent() -> None:
    rng = random.Random(115)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    documents = [
        " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 20)))
        for _ in range(30)
    ]
    chunks = [
        {"chunk_id": f"D{index:03d}", "text": text}
        for index, text in enumerate(documents)
    ]
    index = runner.build_bm25_index(chunks)
    for question in ("alpha beta", "gamma gamma delta", "absent", "epsilon"):
        old_scores = contract.bm25_scores(question, documents)
        assert index.scores(question) == old_scores
        old_ranking = sorted(
            zip(chunks, old_scores),
            key=lambda value: (-value[1], value[0]["chunk_id"]),
        )[: contract.BM25_TOP_K]
        assert [row["chunk_id"] for row in runner.rank_bm25(question, chunks, index)] == [
            chunk["chunk_id"] for chunk, _ in old_ranking
        ]


def test_model_request_acknowledgement_is_formal_only(tmp_path: Path) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()
    fixture_args = run_args(
        tmp_path,
        condition="bm25",
        preregistration_path=prereg,
        client=client,
    )
    fixture_args.allow_model_requests = True
    with pytest.raises(runner.RunError, match="synthetic.*forbids"):
        runner.run_condition(fixture_args)

    formal_args = copy.copy(fixture_args)
    formal_args.fixture = None
    formal_args.allow_model_requests = False
    with pytest.raises(runner.RunError, match="requires --allow-model-requests"):
        runner.run_condition(formal_args)
    with pytest.raises(SystemExit):
        runner.parse_args(["preflight", "--allow-model-requests"])


def test_full_context_fixture_and_independent_audit(tmp_path: Path) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()
    args = run_args(
        tmp_path,
        condition="full_context",
        preregistration_path=prereg,
        client=client,
    )
    manifest = runner.run_condition(args)
    assert manifest["status"] == "complete"
    assert manifest["completed_question_count"] == 3
    assert len(client.completions.calls) == 3
    report = auditor.audit(
        condition_dir=args.output_root / "full_context",
        fixture=FIXTURE,
        cache_dir=contract.HF_CACHE,
        allow_fixture=True,
    )
    assert report["status"] == "passed"
    assert report["answered_count"] == 3
    assert report["blocked_over_context_count"] == 0


def test_full_context_overflow_is_blocked_without_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()
    monkeypatch.setattr(contract, "MAX_RENDERED_PROMPT_TOKENS", 1)
    args = run_args(
        tmp_path,
        condition="full_context",
        preregistration_path=prereg,
        client=client,
    )
    runner.run_condition(args)
    assert not client.completions.calls
    report = auditor.audit(
        condition_dir=args.output_root / "full_context",
        fixture=FIXTURE,
        cache_dir=contract.HF_CACHE,
        allow_fixture=True,
    )
    assert report["blocked_over_context_count"] == 3
    record = contract.read_json(
        args.output_root / "full_context/100K/conversation_000/questions/00.json"
    )
    assert record["status"] == contract.BLOCKED_STATUS
    assert record["answer"] is None
    assert record["delivered_evidence"] is None


def test_bm25_fixture_budget_trace_and_tamper_rejection(tmp_path: Path) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()
    args = run_args(
        tmp_path,
        condition="bm25",
        preregistration_path=prereg,
        client=client,
    )
    runner.run_condition(args)
    condition_dir = args.output_root / "bm25"
    report = auditor.audit(
        condition_dir=condition_dir,
        fixture=FIXTURE,
        cache_dir=contract.HF_CACHE,
        allow_fixture=True,
    )
    assert report["status"] == "passed"
    (condition_dir / "audit.json").unlink()
    question_path = condition_dir / "100K/conversation_000/questions/00.json"
    question = contract.read_json(question_path)
    question["retrieval_records"][0]["record_id"] = "tampered"
    contract.atomic_json_replace(question_path, question)
    with pytest.raises(auditor.AuditError, match="content hash|BM25"):
        auditor.audit(
            condition_dir=condition_dir,
            fixture=FIXTURE,
            cache_dir=contract.HF_CACHE,
            allow_fixture=True,
        )


def test_mem0_fixture_source_trace_and_workspace_hash(tmp_path: Path) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()

    def backend_factory(**kwargs: Any) -> FakeMem0Backend:
        return FakeMem0Backend(fake_client=client, **kwargs)

    args = run_args(
        tmp_path,
        condition="mem0",
        preregistration_path=prereg,
        client=client,
        backend_factory=backend_factory,
    )
    runner.run_condition(args)
    report = auditor.audit(
        condition_dir=args.output_root / "mem0",
        fixture=FIXTURE,
        cache_dir=contract.HF_CACHE,
        allow_fixture=True,
    )
    assert report["status"] == "passed"
    assert report["answered_count"] == 3


def test_budget_truncation_is_durable_and_reconstructable(tmp_path: Path) -> None:
    tokenizer = visible.TokenCounter.resolve(
        requested_model="gpt-5.5", fallback_encoding="o200k_base"
    )
    conversation = runner.normalize_conversation(
        contract.read_json(FIXTURE)["100K"][0],
        chat_size="100K",
        conversation_index=0,
        require_formal_inventory=False,
    )
    item = runner.build_inventory(
        splits={"100K": [contract.read_json(FIXTURE)["100K"][0]], "1M": []},
        selections={"100K": [0]},
        mapping=None,
    )[0][0]
    before = visible.snapshot_memory_bytes(b"fixed", label="fixture")
    evidence, accounting = runner.prepare_capped_evidence(
        method="bm25",
        question_record=item,
        conversation=conversation,
        retrieval_records=[
            {
                "rank": 0,
                "record_id": "oversized",
                "text": "relevant " * 40_000,
                "source_chat_ids": ["0"],
            }
        ],
        item_dir=tmp_path,
        tokenizer=tokenizer,
        memory_before=before,
        memory_after=before,
        actual_model_usage={"calls": 0},
    )
    assert accounting["visible_tokens"] <= 20_000
    assert accounting["exhausted"] is True
    assert tokenizer.count(evidence) <= 20_000
    report = auditor.visible_token_audit.audit_visible_token_trace(
        tmp_path / accounting["trace_path"],
        manifest_path=tmp_path / accounting["manifest_path"],
    )
    assert report["audit_status"] == "pass"


def test_resume_and_conflicting_identity_fail_closed(tmp_path: Path) -> None:
    prereg = preregistration(tmp_path)
    client = FakeClient()
    args = run_args(
        tmp_path,
        condition="full_context",
        preregistration_path=prereg,
        client=client,
    )
    runner.run_condition(args)
    first_call_count = len(client.completions.calls)
    args.resume = True
    runner.run_condition(args)
    assert len(client.completions.calls) == first_call_count
    manifest_path = args.output_root / "full_context/run_manifest.json"
    manifest = contract.read_json(manifest_path)
    manifest["provider_contract"] = {"status": "conflict"}
    contract.atomic_json_replace(manifest_path, manifest)
    with pytest.raises(runner.RunError, match="identity differs"):
        runner.run_condition(args)


def test_orphan_model_call_rejects_resume(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    ledger = durable.HashChainLedger(model_root / "ledger.jsonl", run_id="orphan")
    ledger.append("operation_started", operation_id="answer-q00")
    ledger.append(
        "model_call_started",
        operation_id="answer-q00",
        logical_call_id="orphan:answer-q00:call-0001",
        request_path="calls/request.json",
        request_sha256="0" * 64,
        requested_model="gpt-5.5",
        local_visible_tokens=1,
        tokenizer={},
        proxy_log_start=None,
    )
    with pytest.raises(durable.DurableLedgerError, match="unresolved ledger state"):
        durable.ledger_state(durable.read_ledger(model_root / "ledger.jsonl"))


def test_graphiti_exclusion_is_explicit_and_pre_answer(tmp_path: Path) -> None:
    prereg = contract.read_json(preregistration(tmp_path))
    decision = prereg["structured_control_decision"]
    assert decision["status"] == "excluded_before_formal_answers"
    assert "turn-level" in decision["reason"]
    assert prereg["model_calls"] == prereg["network_calls"] == 0


def test_flex_window_binds_durable_consumer_without_network(tmp_path: Path) -> None:
    provider_root = tmp_path / "provider"
    transport = FakeFlexTransport()
    gateway = flex_gateway.GPT55FlexGateway(
        result_root=provider_root,
        max_cost_usd="100",
        transport=transport,
        max_physical_attempts=1,
        initial_backoff_seconds=0,
    )
    contract.atomic_json_no_clobber(
        provider_root / "gateway_ready.json",
        {
            "schema": "openai-gpt55-flex-ready/v1",
            "pid": 1,
            "base_url": "http://127.0.0.1:8200/v1",
            "health_url": "http://127.0.0.1:8200/healthz",
            "requested_model": flex_gateway.REQUESTED_MODEL,
            "provider_model": flex_gateway.PROVIDER_MODEL,
            "service_tier": flex_gateway.SERVICE_TIER,
            "max_cost_usd": "100",
            "started_at": "synthetic",
        },
    )
    window_start = flex_evidence.capture_start(provider_root)
    model_root = tmp_path / "consumer"
    ledger = durable.HashChainLedger(model_root / "ledger.jsonl", run_id="r115-flex")
    tokenizer = visible.TokenCounter.resolve(
        requested_model="gpt-5.5", fallback_encoding="o200k_base"
    )
    client = GatewayClient(gateway)
    observer = durable.DurableModelObserver(
        ledger=ledger,
        artifact_root=model_root,
        token_counter=tokenizer,
        expected_model="gpt-5.5",
        proxy_log=None,
        formal=False,
    )
    observer.install(client.chat.completions)
    ledger.append("operation_started", operation_id="answer-q00")
    answer = runner.call_answer(
        client=client,
        observer=observer,
        operation_id="answer-q00",
        question="What is the answer?",
        evidence="A fixed fixture fact.",
    )
    ledger.append(
        "operation_committed",
        operation_id="answer-q00",
        result_sha256=contract.canonical_hash(answer),
    )
    observer.restore()
    window = flex_evidence.capture_end(window_start)
    consumers = runner.flex_consumer_records(
        ledger_records=durable.read_ledger(model_root / "ledger.jsonl"),
        model_root=model_root,
    )
    report = flex_evidence.audit_window(window, consumer_records=consumers)
    assert report["status"] == "passed"
    assert report["requests"] == 1
    assert consumers[0]["provider_actual_model"] == "gpt-5.5-2026-04-23"
    assert len(transport.payloads) == 1
