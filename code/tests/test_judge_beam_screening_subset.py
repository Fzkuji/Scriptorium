from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

GOLD_FIELD_BY_TYPE = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": "expected_compliance",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": "expected_compliance",
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}

RUBRIC_LENGTHS = (
    (1, 1, 4, 4, 3, 5, 1, 6, 1, 1, 1, 1, 2, 4, 2, 2, 5, 5, 2, 2),
    (1, 1, 4, 4, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 5, 5, 2, 2),
)


def _module():
    name = "scripts.judge_beam_screening_subset"
    assert importlib.util.find_spec(name) is not None, "semantic judge module missing"
    return importlib.import_module(name)


def _offline_module():
    return importlib.import_module("scripts.eval_beam_screening_subset")


def _write_source_report(tmp_path: Path) -> Path:
    offline = _offline_module()
    gold_paths: list[Path] = []
    predictions: list[dict[str, str]] = []
    for conversation_index, conversation_id in enumerate(("1", "2")):
        questions: list[dict[str, object]] = []
        question_index = 0
        for question_type in QUESTION_TYPES:
            for _ in range(2):
                question_id = f"{conversation_id}-q{question_index}"
                gold_field = GOLD_FIELD_BY_TYPE[question_type]
                gold = f"Gold {question_id}"
                rubric = [
                    f"Rubric {question_id} nugget {index}"
                    for index in range(
                        RUBRIC_LENGTHS[conversation_index][question_index]
                    )
                ]
                questions.append(
                    {
                        "question_id": question_id,
                        "question_type": question_type,
                        "question_text": f"Question {question_id}?",
                        "gold_field": gold_field,
                        "gold": gold,
                        gold_field: gold,
                        "rubric_nuggets": rubric,
                    }
                )
                predictions.append({"question_id": question_id, "answer": gold})
                question_index += 1
        unit = {
            "benchmark": "beam-100k",
            "unit_id": f"100K-conv-{conversation_id}",
            "selection": {
                "split": "100K",
                "conversation_index": conversation_index,
                "conversation_id": conversation_id,
            },
            "questions": questions,
        }
        gold_path = tmp_path / f"unit-{conversation_id}.json"
        gold_path.write_text(json.dumps(unit), encoding="utf-8")
        gold_paths.append(gold_path)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in predictions),
        encoding="utf-8",
    )
    report = offline.evaluate(
        gold_units=gold_paths,
        predictions_path=predictions_path,
    )
    report_path = tmp_path / "screening-report.json"
    offline.write_report_no_clobber(report_path, report)
    return report_path


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rewrite_report(path: Path, mutate) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    report.pop("report_content_sha256", None)
    report["report_content_sha256"] = _canonical_hash(report)
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


class FakeCompletions:
    def __init__(self, responses: list[dict[str, object]], on_response=None) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []
        self.on_response = on_response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if self.on_response is not None:
            self.on_response(response)
        usage = response.get(
            "usage",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        return SimpleNamespace(
            id=response["id"],
            model=response.get("model", "openai/gpt-4o-mini"),
            choices=[
                SimpleNamespace(
                    finish_reason=response.get("finish_reason", "stop"),
                    message=SimpleNamespace(
                        content=response.get(
                            "content", '{"score": 1, "reason": "supported"}'
                        ),
                        refusal=response.get("refusal"),
                    ),
                )
            ],
            usage=SimpleNamespace(**usage) if usage is not None else None,
        )


def _fake_client(count: int, *, start: int = 0, on_response=None):
    completions = FakeCompletions(
        [
            {"id": f"screening-response-{index}"}
            for index in range(start, start + count)
        ],
        on_response=on_response,
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_log=completions,
    )


def _write_marked_gateway(tmp_path: Path, mod) -> tuple[Path, str]:
    evidence = mod.FULL_EVALUATOR.openrouter_gateway_evidence
    root = tmp_path / "openrouter-gateway"
    root.mkdir()
    base_url = "http://127.0.0.1:8765/v1"
    marker_path = root / evidence.ROOT_MARKER_NAME
    state_path = root / evidence.STATE_NAME
    log_path = root / evidence.REQUEST_LOG_NAME
    marker_path.write_text(
        json.dumps(
            {
                "schema": evidence.ROOT_SCHEMA,
                "provider": "OpenRouter",
                "api": "chat_completions",
                "upstream_url": "https://openrouter.ai/api/v1/chat/completions",
                "provider_model": evidence.PROVIDER_MODEL,
                "returned_alias": evidence.REQUESTED_MODEL,
                "cross_model_fallbacks": False,
                "billing": "openrouter_api",
                "routing_policy": "exact_model_no_fallback",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "schema": evidence.STATE_SCHEMA,
        "provider": "OpenRouter",
        "provider_model": evidence.PROVIDER_MODEL,
        "returned_alias": evidence.REQUESTED_MODEL,
        "cross_model_fallbacks": False,
        "max_cost_usd": 5.0,
        "committed_cost_nanos": 0,
        "reserved_cost_nanos": 0,
        "billable_request_count": 0,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    (root / evidence.READY_NAME).write_text(
        json.dumps(
            {
                "schema": "openrouter-gpt4o-mini-ready/v1",
                "base_url": base_url,
                "result_root": str(root.resolve()),
                "root_marker": str(marker_path.resolve()),
                "state": str(state_path.resolve()),
                "request_log": str(log_path.resolve()),
                "requested_model": evidence.REQUESTED_MODEL,
                "provider_model": evidence.PROVIDER_MODEL,
                "cross_model_fallbacks": False,
                "max_cost_usd": 5.0,
            }
        ),
        encoding="utf-8",
    )
    return root, base_url


def _append_gateway_response(root: Path, mod, response: dict[str, object]) -> None:
    evidence = mod.FULL_EVALUATOR.openrouter_gateway_evidence
    with (root / evidence.REQUEST_LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": evidence.REQUEST_LOG_SCHEMA,
                    "status": "success",
                    "response_id": response["id"],
                    "requested_model": evidence.REQUESTED_MODEL,
                    "provider_actual_model": evidence.PROVIDER_MODEL,
                    "returned_alias": evidence.REQUESTED_MODEL,
                    "billable": True,
                }
            )
            + "\n"
        )


def test_load_report_builds_frozen_judge_input(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)

    loaded = mod.load_screening_report(report_path)

    assert loaded["schema_version"] == 1
    assert loaded["protocol_class"] == "screening_subset"
    assert loaded["formal_scope_verified"] is False
    assert loaded["selected_conversations"] == {"100K": [0, 1]}
    assert loaded["question_count"] == 40
    assert loaded["rubric_nugget_count"] == 103
    assert loaded["question_type_counts"] == {
        question_type: 4 for question_type in QUESTION_TYPES
    }
    assert loaded["records"][0]["question_id"] == "100K_0_q0_abstention"
    assert loaded["records"][0]["source_question_id"] == "1-q0"
    assert loaded["records"][20]["question_id"] == "100K_1_q0_abstention"
    assert loaded["records"][20]["source_question_id"] == "2-q0"
    assert loaded["source_report_path"] == str(report_path.resolve())
    assert (
        loaded["source_report_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert (
        loaded["source_report_content_sha256"]
        == json.loads(report_path.read_text(encoding="utf-8"))["report_content_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.__setitem__("protocol_class", "formal"), "protocol"),
        (
            lambda report: report.__setitem__("formal_scope_verified", True),
            "formal_scope_verified",
        ),
        (lambda report: report.__setitem__("question_count", 39), "40 questions"),
        (
            lambda report: report["records"][0]["rubric"].pop(),
            "103 rubric nuggets",
        ),
        (
            lambda report: report["records"][20].__setitem__("conversation_index", 2),
            "conversation selection",
        ),
        (
            lambda report: report["records"][0].__setitem__(
                "question_type", "temporal_reasoning"
            ),
            "four questions per type",
        ),
        (
            lambda report: report["records"][0].__setitem__("question_index", 1),
            "question indices",
        ),
    ),
)
def test_load_report_rejects_out_of_scope_inventory(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    _rewrite_report(report_path, mutation)

    with pytest.raises(mod.ScreeningJudgeError, match=message):
        mod.load_screening_report(report_path)


def test_load_report_rejects_stale_content_hash(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["records"][0]["answer"] = "tampered"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(mod.ScreeningJudgeError, match="content hash"):
        mod.load_screening_report(report_path)


def test_load_report_rejects_non_list_rubric_even_when_total_is_103(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)

    def mutate(report):
        report["records"][0]["rubric"] = "x"
        report["records"][1]["rubric"].append("replacement nugget")

    _rewrite_report(report_path, mutate)

    with pytest.raises(mod.ScreeningJudgeError, match="rubric must be a list"):
        mod.load_screening_report(report_path)


def test_load_report_rejects_boolean_conversation_indices(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)

    def mutate(report):
        for record in report["records"][20:]:
            record["conversation_index"] = True

    _rewrite_report(report_path, mutate)

    with pytest.raises(mod.ScreeningJudgeError, match="conversation index"):
        mod.load_screening_report(report_path)


def test_load_report_rejects_invalid_json_as_contract_error(tmp_path: Path) -> None:
    mod = _module()
    report_path = tmp_path / "screening-report.json"
    report_path.write_text("{", encoding="utf-8")

    with pytest.raises(mod.ScreeningJudgeError, match="valid JSON"):
        mod.load_screening_report(report_path)


def test_screening_config_is_primary_hash_bound_and_nonformal(tmp_path: Path) -> None:
    mod = _module()
    loaded = mod.load_screening_report(_write_source_report(tmp_path))

    config = mod.build_screening_config(
        loaded,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    mod.validate_screening_config(config, loaded)

    assert config["formal"] is False
    assert config["protocol_class"] == "screening_subset"
    assert config["tier"] == "terra"
    assert config["profile"] == "primary"
    assert config["provider"] == "OpenRouter"
    assert config["model"] == "openai/gpt-4o-mini"
    assert config["expected_response_model"] == "openai/gpt-4o-mini"
    assert config["transport_contract"] == "injected_test_judge"
    assert config["max_retries"] == 3
    assert config["source_report_sha256"] == loaded["source_report_sha256"]
    assert config["paired_screening_inventory_sha256"] == _canonical_hash(
        [
            {
                key: record[key]
                for key in (
                    "question_id",
                    "source_question_id",
                    "chat_size",
                    "conversation_index",
                    "question_index",
                    "question_type",
                    "question",
                    "rubric",
                )
            }
            for record in loaded["records"]
        ]
    )
    assert config["judge_procedure_sha256"] == _canonical_hash(
        {
            "profile": "primary",
            "provider": "OpenRouter",
            "model": "openai/gpt-4o-mini",
            "expected_response_model": "openai/gpt-4o-mini",
            "prompt_source_sha256": config["prompt_source_sha256"],
            "system_prompt_sha256": config["system_prompt_sha256"],
            "temperature": 0.0,
            "max_tokens": 400,
            "max_retries": 3,
            "scoring": "question_mean_over_rubric_nuggets",
            "valid_scores": [0.0, 0.5, 1.0],
            "transport_contract": "injected_test_judge",
        }
    )
    assert (
        config["evaluator_source_sha256"]
        == hashlib.sha256(mod.SCRIPT_PATH.read_bytes()).hexdigest()
    )
    assert (
        config["reused_evaluator_source_sha256"]
        == hashlib.sha256(mod.FULL_EVALUATOR.SCRIPT_PATH.read_bytes()).hexdigest()
    )
    assert (
        config["prompt_source_sha256"]
        == hashlib.sha256(mod.FULL_EVALUATOR.PROMPTS_PATH.read_bytes()).hexdigest()
    )
    assert config["judge_procedure_comparable_to_project_unified_protocol"] is True
    assert config["comparable_to_project_unified_protocol"] is False
    assert config["full_benchmark_comparable"] is False
    assert config["comparable_to_published_beam_official"] is False
    assert config["official_or_formal_score"] is False
    assert config["claim_scope"] == "two-conversation paired screening only"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("formal", True),
        ("tier", None),
        ("profile", "secondary"),
        ("model", "gpt-5.5"),
        ("provider", "OpenAI"),
        ("max_retries", 2),
        ("full_benchmark_comparable", True),
        ("comparable_to_published_beam_official", True),
        ("official_or_formal_score", True),
        ("source_report_sha256", "0" * 64),
        ("prompt_source_sha256", "0" * 64),
    ),
)
def test_screening_config_rejects_scope_model_or_hash_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    mod = _module()
    loaded = mod.load_screening_report(_write_source_report(tmp_path))
    config = mod.build_screening_config(
        loaded,
        tier="sol",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    config[field] = value

    with pytest.raises(mod.ScreeningJudgeError, match=field):
        mod.validate_screening_config(config, loaded)


def test_screening_config_rejects_unknown_tier(tmp_path: Path) -> None:
    mod = _module()
    loaded = mod.load_screening_report(_write_source_report(tmp_path))

    with pytest.raises(mod.ScreeningJudgeError, match="tier"):
        mod.build_screening_config(
            loaded,
            tier="unknown",
            base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
            max_tokens=400,
        )


def test_run_writes_103_auditable_judgments_and_screening_only_results(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    client = _fake_client(103)
    output_root = tmp_path / "semantic-judge"

    results = mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=client,
        resume=False,
    )

    tier_dir = output_root / "terra"
    assert len(client.completions_log.calls) == 103
    assert len(list((tier_dir / "judgments").glob("*.json"))) == 103
    assert len(list((tier_dir / "attempt_ledgers").glob("*.jsonl"))) == 103
    first_judgment = json.loads(
        (tier_dir / "judgments" / "100K_0_q0_abstention__n000.json").read_text()
    )
    assert first_judgment["response_id"] == "screening-response-0"
    assert first_judgment["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    first_ledger = tier_dir / "attempt_ledgers" / "100K_0_q0_abstention__n000.jsonl"
    assert [
        json.loads(line)["event"] for line in first_ledger.read_text().splitlines()
    ] == ["attempt_started", "attempt_finished"]
    assert results["status"] == "complete"
    assert results["protocol_class"] == "screening_subset"
    assert results["tier"] == "terra"
    assert results["formal_scope_verified"] is False
    assert results["question_count"] == 40
    assert results["rubric_nugget_count"] == 103
    assert results["judge_response_ids"] == 103
    assert results["full_benchmark_comparable"] is False
    assert results["comparable_to_published_beam_official"] is False
    assert results["official_or_formal_score"] is False
    assert results["claim_scope"] == "two-conversation paired screening only"
    assert results["paired_screening_comparison_allowed"] is True
    assert (
        results["paired_screening_inventory_sha256"]
        == config["paired_screening_inventory_sha256"]
    )
    assert results["judge_procedure_sha256"] == config["judge_procedure_sha256"]
    assert results["paired_screening_requirement"] == (
        "compare tiers only when this inventory hash and judge procedure match"
    )
    assert results["evaluations"][0]["source_question_id"] == "1-q0"
    assert results["metrics"]["event_ordering"]["official_tau_b_times_f1"] is None
    assert (tier_dir / "results.json").is_file()
    assert (tier_dir / "metrics.json").is_file()
    assert (tier_dir / "evaluation_audit.json").is_file()
    state = json.loads((tier_dir / "run_state.json").read_text())
    assert state["status"] == "complete"
    assert state["completed_nuggets"] == 103
    assert state["failed_nuggets"] == 0
    assert (
        state["evaluation_audit_sha256"]
        == hashlib.sha256((tier_dir / "evaluation_audit.json").read_bytes()).hexdigest()
    )
    assert not (output_root / "results.json").exists()


def test_fresh_rerun_is_no_clobber_and_makes_no_calls(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="sol",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="sol",
        config=config,
        client=_fake_client(103),
        resume=False,
    )
    tier_dir = output_root / "sol"
    before = {path: path.read_bytes() for path in tier_dir.rglob("*") if path.is_file()}
    unused = _fake_client(1)

    with pytest.raises(mod.ScreeningJudgeError, match="--resume"):
        mod.run_screening_judge(
            report_path=report_path,
            output_root=output_root,
            tier="sol",
            config=config,
            client=unused,
            resume=False,
        )

    assert unused.completions_log.calls == []
    assert {path: path.read_bytes() for path in before} == before


def test_resume_skips_completed_nuggets_and_finishes_remaining_jobs(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    output_dir = output_root / "terra"
    output_dir.mkdir(parents=True)
    state = mod.FULL_EVALUATOR.load_or_create_state(
        output_dir,
        input_path=report_path.resolve(),
        input_sha256=evaluation_input["source_report_sha256"],
        config=config,
        total_jobs=103,
        resume=False,
    )
    first = evaluation_input["records"][0]
    mod.FULL_EVALUATOR.judge_one_job(
        _fake_client(1),
        output_dir=output_dir,
        record=first,
        nugget_index=0,
        nugget=first["rubric"][0],
        input_sha256=evaluation_input["source_report_sha256"],
        config_hash=state["config_hash"],
        config=config,
        sleep_fn=lambda _: None,
    )
    remaining = _fake_client(102, start=1)

    results = mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=remaining,
        resume=True,
    )

    assert len(remaining.completions_log.calls) == 102
    assert results["status"] == "complete"
    assert results["judge_response_ids"] == 103
    assert len(list((output_dir / "attempt_ledgers").glob("*.jsonl"))) == 103


def test_transient_judge_failure_retries_and_preserves_attempt_ledger(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    record = evaluation_input["records"][0]
    completions = FakeCompletions(
        [RuntimeError("temporary transport failure"), {"id": "retry-success"}]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    judgment = mod.FULL_EVALUATOR.judge_one_job(
        client,
        output_dir=tmp_path / "judge",
        record=record,
        nugget_index=0,
        nugget=record["rubric"][0],
        input_sha256=evaluation_input["source_report_sha256"],
        config_hash=mod.FULL_EVALUATOR.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )

    assert judgment["status"] == "complete"
    assert judgment["response_id"] == "retry-success"
    assert len(completions.calls) == 2
    assert [attempt["status"] for attempt in judgment["attempts"]] == [
        "failed",
        "complete",
    ]


def test_main_defaults_to_audit_only_and_creates_no_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    output_root = tmp_path / "semantic-judge"
    monkeypatch.setattr(
        mod,
        "create_openai_client",
        lambda **_: pytest.fail("audit-only mode created a model client"),
    )

    returncode = mod.main(
        [
            str(report_path),
            "--output-root",
            str(output_root),
            "--tier",
            "terra",
        ]
    )

    assert returncode == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "audit_only"
    assert printed["would_request_model"] is False
    assert printed["question_count"] == 40
    assert printed["rubric_nugget_count"] == 103
    assert not output_root.exists()


@pytest.mark.parametrize(
    "single_flag",
    ("--allow-model-requests", "--confirm-screening-subset-requests"),
)
def test_main_requires_both_model_request_flags(
    tmp_path: Path,
    single_flag: str,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)

    with pytest.raises(SystemExit) as caught:
        mod.main(
            [
                str(report_path),
                "--output-root",
                str(tmp_path / "semantic-judge"),
                "--tier",
                "terra",
                single_flag,
            ]
        )

    assert caught.value.code == 2


def test_main_dual_flags_bind_all_fake_response_ids_to_marked_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    gateway_root, base_url = _write_marked_gateway(tmp_path, mod)
    client = _fake_client(
        103,
        on_response=lambda response: _append_gateway_response(
            gateway_root, mod, response
        ),
    )
    monkeypatch.setattr(mod, "create_openai_client", lambda **_: client)
    output_root = tmp_path / "semantic-judge"

    returncode = mod.main(
        [
            str(report_path),
            "--output-root",
            str(output_root),
            "--tier",
            "sol",
            "--openrouter-gateway-root",
            str(gateway_root),
            "--base-url",
            base_url,
            "--allow-model-requests",
            "--confirm-screening-subset-requests",
        ]
    )

    assert returncode == 0
    state = json.loads((output_root / "sol" / "run_state.json").read_text())
    assert state["config"]["transport_contract"] == "marked_openrouter_gateway"
    assert state["openrouter_gateway_evidence"] == {
        "status": "passed",
        "bound_response_ids": 103,
        "interval_entries": 103,
        "interval_successes": 103,
    }
    audit = json.loads((output_root / "sol" / "evaluation_audit.json").read_text())
    assert audit["gateway_evidence"]["bound_response_ids"] == 103


def test_missing_gateway_response_ids_fail_closed(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    gateway_root, base_url = _write_marked_gateway(tmp_path, mod)
    binding = mod.FULL_EVALUATOR.openrouter_gateway_evidence.capture_binding(
        gateway_root,
        base_url=base_url,
    )
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=base_url,
        max_tokens=400,
        openrouter_gateway=binding,
    )
    output_root = tmp_path / "semantic-judge"

    with pytest.raises(mod.ScreeningJudgeError, match="absent from gateway"):
        mod.run_screening_judge(
            report_path=report_path,
            output_root=output_root,
            tier="terra",
            config=config,
            client=_fake_client(103),
            resume=False,
        )

    tier_dir = output_root / "terra"
    state = json.loads((tier_dir / "run_state.json").read_text())
    assert state["status"] == "failed"
    assert "absent from gateway" in state["failure"]
    assert not (tier_dir / "results.json").exists()
    assert not (tier_dir / "metrics.json").exists()


def test_main_resume_reuses_frozen_marked_gateway_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    gateway_root, base_url = _write_marked_gateway(tmp_path, mod)
    binding = mod.FULL_EVALUATOR.openrouter_gateway_evidence.capture_binding(
        gateway_root,
        base_url=base_url,
    )
    config = mod.build_screening_config(
        evaluation_input,
        tier="sol",
        base_url=base_url,
        max_tokens=400,
        openrouter_gateway=binding,
    )
    output_root = tmp_path / "semantic-judge"
    output_dir = output_root / "sol"
    output_dir.mkdir(parents=True)
    state = mod.FULL_EVALUATOR.load_or_create_state(
        output_dir,
        input_path=report_path.resolve(),
        input_sha256=evaluation_input["source_report_sha256"],
        config=config,
        total_jobs=103,
        resume=False,
    )
    first_record = evaluation_input["records"][0]
    mod.FULL_EVALUATOR.judge_one_job(
        _fake_client(
            1,
            on_response=lambda response: _append_gateway_response(
                gateway_root, mod, response
            ),
        ),
        output_dir=output_dir,
        record=first_record,
        nugget_index=0,
        nugget=first_record["rubric"][0],
        input_sha256=evaluation_input["source_report_sha256"],
        config_hash=state["config_hash"],
        config=config,
        sleep_fn=lambda _: None,
    )
    remaining = _fake_client(
        102,
        start=1,
        on_response=lambda response: _append_gateway_response(
            gateway_root, mod, response
        ),
    )
    monkeypatch.setattr(mod, "create_openai_client", lambda **_: remaining)

    assert (
        mod.main(
            [
                str(report_path),
                "--output-root",
                str(output_root),
                "--tier",
                "sol",
                "--resume",
                "--openrouter-gateway-root",
                str(gateway_root),
                "--base-url",
                base_url,
                "--allow-model-requests",
                "--confirm-screening-subset-requests",
            ]
        )
        == 0
    )

    assert len(remaining.completions_log.calls) == 102
    completed = json.loads((output_dir / "run_state.json").read_text())
    assert completed["status"] == "complete"
    assert completed["openrouter_gateway_evidence"]["bound_response_ids"] == 103
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_audit_existing_run_recomputes_results_and_ledgers(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=_fake_client(103),
        resume=False,
    )

    audit = mod.audit_screening_run(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
    )

    assert audit["status"] == "passed"
    assert audit["rubric_nugget_count"] == 103
    assert audit["judge_response_ids"] == 103
    assert audit["judge_usage"]["total_tokens"] == 1545


def test_main_default_audits_existing_completed_tier_without_requests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=_fake_client(103),
        resume=False,
    )
    monkeypatch.setattr(
        mod,
        "create_openai_client",
        lambda **_: pytest.fail("audit-only mode created a model client"),
    )

    assert (
        mod.main(
            [
                str(report_path),
                "--output-root",
                str(output_root),
                "--tier",
                "terra",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "audit_only"
    assert printed["status"] == "passed"
    assert printed["judge_response_ids"] == 103


def test_audit_existing_run_rejects_tampered_append_only_ledger(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="sol",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="sol",
        config=config,
        client=_fake_client(103),
        resume=False,
    )
    ledger = next((output_root / "sol" / "attempt_ledgers").glob("*.jsonl"))
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(mod.ScreeningJudgeError, match="attempt ledger"):
        mod.audit_screening_run(
            report_path=report_path,
            output_root=output_root,
            tier="sol",
        )


def test_complete_resume_audits_before_returning_existing_results(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    output_root = tmp_path / "semantic-judge"
    mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=_fake_client(103),
        resume=False,
    )
    results_path = output_root / "terra" / "results.json"
    results = json.loads(results_path.read_text())
    results["claim_scope"] = "full BEAM"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    unused = _fake_client(1)

    with pytest.raises(mod.ScreeningJudgeError, match="stored results"):
        mod.run_screening_judge(
            report_path=report_path,
            output_root=output_root,
            tier="terra",
            config=config,
            client=unused,
            resume=True,
        )

    assert unused.completions_log.calls == []


def test_duplicate_response_ids_fail_closed_without_numeric_results(
    tmp_path: Path,
) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="sol",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    completions = FakeCompletions(
        [{"id": "duplicated-response-id"} for _ in range(103)]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_log=completions,
    )
    output_root = tmp_path / "semantic-judge"

    with pytest.raises(mod.ScreeningJudgeError, match="duplicated"):
        mod.run_screening_judge(
            report_path=report_path,
            output_root=output_root,
            tier="sol",
            config=config,
            client=client,
            resume=False,
        )

    tier_dir = output_root / "sol"
    state = json.loads((tier_dir / "run_state.json").read_text())
    assert state["status"] == "failed"
    assert "duplicated" in state["failure"]
    assert not (tier_dir / "results.json").exists()
    assert not (tier_dir / "metrics.json").exists()


def test_invalid_usage_has_no_score_or_metrics(tmp_path: Path) -> None:
    mod = _module()
    report_path = _write_source_report(tmp_path)
    evaluation_input = mod.load_screening_report(report_path)
    config = mod.build_screening_config(
        evaluation_input,
        tier="terra",
        base_url=mod.FULL_EVALUATOR.TEST_PRIMARY_BASE_URL,
        max_tokens=400,
    )
    responses: list[dict[str, object]] = [
        {
            "id": f"invalid-usage-response-{attempt}",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 99,
            },
        }
        for attempt in range(3)
    ]
    responses.extend({"id": f"screening-response-{index}"} for index in range(1, 103))
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    output_root = tmp_path / "semantic-judge"

    results = mod.run_screening_judge(
        report_path=report_path,
        output_root=output_root,
        tier="terra",
        config=config,
        client=client,
        resume=False,
    )

    tier_dir = output_root / "terra"
    failed = json.loads(
        (tier_dir / "judgments" / "100K_0_q0_abstention__n000.json").read_text()
    )
    assert failed["status"] == "failed"
    assert "score" not in failed
    assert results["status"] == "incomplete"
    assert results["metrics"] is None
    assert not (tier_dir / "metrics.json").exists()
    assert not (tier_dir / "evaluation_audit.json").exists()
