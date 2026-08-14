import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_v88_gpt55_beam.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v88_gpt55_beam", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FakeCompletions:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.contents)
        if isinstance(value, BaseException):
            raise value
        usage_value = value.get(
            "usage",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        return SimpleNamespace(
            id=value.get("id", "judge-response-1"),
            model=value.get("model", "openai/gpt-4o-mini"),
            choices=[
                SimpleNamespace(
                    finish_reason=value.get("finish_reason", "stop"),
                    message=SimpleNamespace(
                        content=value["content"],
                        refusal=value.get("refusal"),
                    ),
                )
            ],
            usage=(SimpleNamespace(**usage_value) if usage_value is not None else None),
        )


def fake_client(contents):
    completions = FakeCompletions(contents)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions), completions_log=completions
    )


def record(
    question_id="100K_0_q0_abstention",
    question_type="abstention",
    *,
    question_index=0,
    chat_size="100K",
    conversation_index=0,
):
    return {
        "question_id": question_id,
        "question_index": question_index,
        "chat_size": chat_size,
        "conversation_index": conversation_index,
        "question_type": question_type,
        "difficulty": "easy",
        "question": "What is known?",
        "answer": "Nothing is known.",
        "rubric": ["The response should say that no information is available."],
    }


def judge_config(profile="primary", proxy_log=None):
    return MOD.evaluation_config(
        profile=profile,
        base_url=(
            MOD.TEST_PRIMARY_BASE_URL
            if profile == "primary"
            else MOD.TEST_SECONDARY_BASE_URL
        ),
        max_tokens=200,
        max_retries=1,
        proxy_log=proxy_log,
    )


def raw_response(
    *,
    content='{"score": 1, "reason": "checked"}',
    response_id="judge-response-1",
    model="openai/gpt-4o-mini",
    usage=None,
):
    usage = usage or {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    return {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content, "refusal": None},
            }
        ],
        "usage": usage,
    }


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"score": 1, "reason": "ok"}\n```',
        '{"score": 0.75, "reason": "not an allowed score"}',
        '{"score": 1, "reason": "ok", "extra": true}',
        '{"score": false, "reason": "wrong type"}',
        '{"score": 1, "reason": ""}',
    ],
)
def test_strict_judgment_rejects_nonconforming_output(content):
    with pytest.raises(MOD.EvaluationError):
        MOD.parse_strict_judgment(content)


def test_strict_judgment_accepts_only_three_scores():
    for score in (0, 0.5, 1.0):
        result = MOD.parse_strict_judgment(
            json.dumps({"score": score, "reason": "criterion comparison"})
        )
        assert result["score"] == float(score)


def test_judge_job_is_atomic_resumable_and_records_models(tmp_path):
    config = judge_config()
    client = fake_client(
        [
            {
                "content": '{"score": 0.5, "reason": "Partially present."}',
                "id": "resp-judge-1",
                "model": "openai/gpt-4o-mini",
            }
        ]
    )
    value = MOD.judge_one_job(
        client,
        output_dir=tmp_path,
        record=record(),
        nugget_index=0,
        nugget=record()["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    assert value["status"] == "complete"
    assert value["score"] == 0.5
    assert value["requested_model"] == "openai/gpt-4o-mini"
    assert value["response_model"] == "openai/gpt-4o-mini"
    assert value["response_id"] == "resp-judge-1"

    resumed = MOD.judge_one_job(
        fake_client([]),
        output_dir=tmp_path,
        record=record(),
        nugget_index=0,
        nugget=record()["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    assert resumed == value


def test_localhost_client_configuration_disables_environment_proxy():
    assert MOD.should_trust_environment_proxy("http://127.0.0.1:8199/v1") is False
    assert MOD.should_trust_environment_proxy("http://localhost:8199/v1") is False
    assert MOD.should_trust_environment_proxy("https://openrouter.ai/api/v1") is True


def test_started_attempt_without_terminal_fails_closed_on_resume(tmp_path):
    config = judge_config()
    current = record()
    first = fake_client([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        MOD.judge_one_job(
            first,
            output_dir=tmp_path,
            record=current,
            nugget_index=0,
            nugget=current["rubric"][0],
            input_sha256="input-sha",
            config_hash=MOD.stable_hash(config),
            config=config,
            sleep_fn=lambda _: None,
        )
    assert len(first.completions_log.calls) == 1

    second = fake_client(
        [{"content": '{"score": 1, "reason": "must not be called"}'}]
    )
    with pytest.raises(MOD.EvaluationError, match="unresolved provider call"):
        MOD.judge_one_job(
            second,
            output_dir=tmp_path,
            record=current,
            nugget_index=0,
            nugget=current["rubric"][0],
            input_sha256="input-sha",
            config_hash=MOD.stable_hash(config),
            config=config,
            sleep_fn=lambda _: None,
        )
    assert second.completions_log.calls == []


def test_failed_response_identity_is_retained_after_retry(tmp_path):
    config = judge_config()
    config["max_retries"] = 2
    current = record()
    value = MOD.judge_one_job(
        fake_client(
            [
                {
                    "content": '{"score": 1, "reason": "wrong model"}',
                    "id": "rejected-id",
                    "model": "wrong-model",
                },
                {
                    "content": '{"score": 1, "reason": "accepted"}',
                    "id": "accepted-id",
                    "model": "openai/gpt-4o-mini",
                },
            ]
        ),
        output_dir=tmp_path,
        record=current,
        nugget_index=0,
        nugget=current["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    assert value["logical_judge_calls"] == 2
    assert value["physical_http_attempts"] == 2
    assert value["attempts"][0]["response_evidence"]["response_id"] == "rejected-id"
    assert value["attempts"][1]["response_evidence"]["response_id"] == "accepted-id"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(logical_judge_calls=999),
        lambda value: value.update(physical_http_attempts=-7),
        lambda value: value.update(attempts=["not-an-attempt"]),
    ],
)
def test_stored_judgment_rejects_malformed_attempt_accounting(
    tmp_path, mutation
):
    config = judge_config()
    current = record()
    value = MOD.judge_one_job(
        fake_client([{"content": '{"score": 1, "reason": "checked"}'}]),
        output_dir=tmp_path,
        record=current,
        nugget_index=0,
        nugget=current["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    mutation(value)
    with pytest.raises(MOD.EvaluationError, match="attempt"):
        MOD.validate_stored_judgment(
            value,
            identifier=value["job_id"],
            record=current,
            nugget_index=0,
            nugget=current["rubric"][0],
            input_sha256="input-sha",
            config_hash=MOD.stable_hash(config),
            config=config,
        )


def test_failed_judge_response_never_gets_zero_score(tmp_path):
    config = judge_config()
    client = fake_client([{"content": "not json"}])
    value = MOD.judge_one_job(
        client,
        output_dir=tmp_path,
        record=record(),
        nugget_index=0,
        nugget=record()["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    assert value["status"] == "failed"
    assert "score" not in value
    persisted = json.loads(
        MOD.judgment_path(tmp_path, MOD.job_id(record()["question_id"], 0)).read_text()
    )
    assert "score" not in persisted


def test_aggregate_reports_nugget_only_metrics_for_all_ten_types():
    config = judge_config()
    records = []
    judgments = {}
    for index, question_type in enumerate(MOD.EXPECTED_QUESTION_TYPES):
        current = record(f"100K_0_q{index}_{question_type}", question_type)
        records.append(current)
        identifier = MOD.job_id(current["question_id"], 0)
        judgments[identifier] = {
            "status": "complete",
            "score": 1.0 if index % 2 == 0 else 0.0,
            "reason": "checked",
            "requested_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
            "response_id": f"judge-{index}",
        }
    result = MOD.aggregate_results({"records": records}, judgments, config)
    assert result["status"] == "complete"
    assert result["metrics"]["overall"]["avg_score"] == 0.5
    assert result["metrics"]["overall"]["passed"] == 5
    assert set(result["metrics"]["by_question_type"]) == set(
        MOD.EXPECTED_QUESTION_TYPES
    )
    event_metric = result["metrics"]["event_ordering"]
    assert event_metric["official_tau_b_times_f1"] is None
    assert event_metric["official_metric_status"] == "not_computed"


def test_incomplete_aggregate_has_no_metrics_and_no_imputed_score():
    current = record()
    result = MOD.aggregate_results({"records": [current]}, {}, judge_config())
    assert result["status"] == "incomplete"
    assert result["metrics"] is None
    assert result["evaluations"][0]["score"] is None


def test_call_judge_rejects_wrong_actual_model():
    client = fake_client(
        [
            {
                "content": '{"score": 1, "reason": "checked"}',
                "id": "wrong-model-response",
                "model": "not-the-requested-model",
            }
        ]
    )
    with pytest.raises(MOD.JudgeResponseError, match="differs from required"):
        MOD.call_judge(
            client,
            model="openai/gpt-4o-mini",
            record=record(),
            nugget=record()["rubric"][0],
            max_tokens=200,
            expected_response_model="openai/gpt-4o-mini",
            judge_profile="primary",
        )


def test_primary_model_matcher_accepts_only_canonical_name_or_valid_date_suffix():
    accepted = (
        "openai/gpt-4o-mini",
        "gpt-4o-mini",
        "openai/gpt-4o-mini-2024-07-18",
        "gpt-4o-mini-2025-01-01",
    )
    rejected = (
        "OPENAI/GPT-4O-MINI",
        "openai/gpt-4o",
        "openai/gpt-4o-mini-preview",
        "openai/gpt-4o-mini-2024-13-40",
        "openai/gpt-4o-mini-2024-07-18-extra",
        "anthropic/gpt-4o-mini",
    )
    assert all(MOD.response_model_matches("primary", model) for model in accepted)
    assert not any(MOD.response_model_matches("primary", model) for model in rejected)


def test_call_judge_accepts_primary_date_suffixed_response_model():
    result = MOD.call_judge(
        fake_client(
            [
                {
                    "content": '{"score": 1, "reason": "checked"}',
                    "id": "dated-model-response",
                    "model": "openai/gpt-4o-mini-2024-07-18",
                }
            ]
        ),
        model="openai/gpt-4o-mini",
        record=record(),
        nugget=record()["rubric"][0],
        max_tokens=200,
        expected_response_model="openai/gpt-4o-mini",
        judge_profile="primary",
    )
    assert result["response_model"] == "openai/gpt-4o-mini-2024-07-18"


@pytest.mark.parametrize(
    ("field", "value"),
    (("id", None), ("model", None)),
)
def test_call_judge_rejects_missing_response_identity_before_content(field, value):
    response = {
        "content": '{"score": 1, "reason": "checked"}',
        "id": "response-id",
        "model": "openai/gpt-4o-mini",
    }
    response[field] = value
    with pytest.raises(MOD.JudgeResponseError, match="missing model or response id"):
        MOD.call_judge(
            fake_client([response]),
            model="openai/gpt-4o-mini",
            record=record(),
            nugget=record()["rubric"][0],
            max_tokens=200,
            expected_response_model="openai/gpt-4o-mini",
            judge_profile="primary",
        )


def test_stored_judgment_is_bound_to_profile_and_model(tmp_path):
    config = judge_config()
    value = MOD.judge_one_job(
        fake_client(
            [
                {
                    "content": '{"score": 1, "reason": "checked"}',
                    "id": "bound-response",
                    "model": "openai/gpt-4o-mini",
                }
            ]
        ),
        output_dir=tmp_path,
        record=record(),
        nugget_index=0,
        nugget=record()["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    value["judge_profile"] = "secondary"
    with pytest.raises(MOD.EvaluationError, match="judge_profile"):
        MOD.validate_stored_judgment(
            value,
            identifier=value["job_id"],
            record=record(),
            nugget_index=0,
            nugget=record()["rubric"][0],
            input_sha256="input-sha",
            config_hash=MOD.stable_hash(config),
            config=config,
        )


@pytest.mark.parametrize(
    "mutation, message",
    (
        (lambda value: value.update(score=0.0), "differs from its raw response"),
        (
            lambda value: value.update(
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                }
            ),
            "metadata differs from its raw response",
        ),
    ),
)
def test_stored_judgment_reparses_raw_score_reason_and_usage(
    tmp_path, mutation, message
):
    config = judge_config()
    current = record()
    value = MOD.judge_one_job(
        fake_client(
            [
                {
                    "content": '{"score": 1, "reason": "checked"}',
                    "id": "raw-bound-response",
                    "model": "openai/gpt-4o-mini",
                }
            ]
        ),
        output_dir=tmp_path,
        record=current,
        nugget_index=0,
        nugget=current["rubric"][0],
        input_sha256="input-sha",
        config_hash=MOD.stable_hash(config),
        config=config,
        sleep_fn=lambda _: None,
    )
    mutation(value)
    value["response_evidence_sha256"] = MOD.response_evidence_sha256(value)
    with pytest.raises(MOD.EvaluationError, match=message):
        MOD.validate_stored_judgment(
            value,
            identifier=value["job_id"],
            record=current,
            nugget_index=0,
            nugget=current["rubric"][0],
            input_sha256="input-sha",
            config_hash=MOD.stable_hash(config),
            config=config,
        )


def test_output_lock_is_exclusive(tmp_path):
    first = MOD.acquire_output_lock(tmp_path)
    try:
        with pytest.raises(MOD.EvaluationError, match="already using"):
            MOD.acquire_output_lock(tmp_path)
    finally:
        MOD.release_output_lock(first)


def test_incomplete_results_remove_stale_metrics(tmp_path):
    MOD.atomic_json(tmp_path / "metrics.json", {"stale": True})
    MOD.write_result_artifacts(tmp_path, {"status": "incomplete", "metrics": None})
    assert not (tmp_path / "metrics.json").exists()


def test_formal_config_binds_profile_labels_and_evaluator_source():
    primary = judge_config()
    secondary_log = Path(__file__)
    secondary = judge_config("secondary", secondary_log)
    MOD.validate_formal_config(primary)
    MOD.validate_formal_config(secondary)
    assert primary["comparison_scope"] == "project_unified_protocol_only"
    assert primary["comparable_to_project_unified_protocol"] is True
    assert primary["comparable_to_published_beam_official"] is False
    assert primary["comparable_to_mem0"] is False
    assert "comparable" not in primary
    assert secondary["comparison_label"].startswith("secondary_non_comparable")
    assert secondary["comparable_to_project_unified_protocol"] is False
    assert secondary["comparable_to_published_beam_official"] is False
    assert secondary["comparable_to_mem0"] is False
    assert primary["evaluator_source_sha256"] == MOD.sha256_file(MOD.SCRIPT_PATH)


@pytest.mark.parametrize(
    ("profile", "base_url"),
    (
        ("primary", "https://example.test/v1"),
        ("secondary", "http://127.0.0.1:8200/v1"),
    ),
)
def test_formal_config_rejects_noncanonical_profile_base_url(
    profile, base_url, tmp_path
):
    proxy_log = tmp_path / "proxy.jsonl" if profile == "secondary" else None
    if proxy_log:
        proxy_log.write_text("", encoding="utf-8")
    config = judge_config(profile, proxy_log)
    config["base_url"] = base_url
    with pytest.raises(MOD.EvaluationError, match="non-canonical base URL"):
        MOD.validate_formal_config(config)


def test_resume_invalidation_removes_all_derived_artifacts(tmp_path):
    state = {
        "status": "complete",
        "finished_at": "old",
        "results_sha256": "old-results",
        "metrics_sha256": "old-metrics",
        "post_evaluation_audit": "passed",
        "post_evaluation_audit_error": "old-error",
        "evaluation_audit_sha256": "old-audit",
    }
    for filename in ("results.json", "metrics.json", "evaluation_audit.json"):
        MOD.atomic_json(tmp_path / filename, {"stale": True})
    updated = MOD.invalidate_derived_artifacts(tmp_path, state)
    assert updated["status"] == "running"
    assert not any(
        (tmp_path / filename).exists()
        for filename in ("results.json", "metrics.json", "evaluation_audit.json")
    )
    for key in (
        "finished_at",
        "results_sha256",
        "metrics_sha256",
        "post_evaluation_audit",
        "post_evaluation_audit_error",
        "evaluation_audit_sha256",
    ):
        assert key not in updated
    assert json.loads((tmp_path / "run_state.json").read_text()) == updated


def test_post_audit_failure_removes_passed_artifacts_and_numeric_metrics(tmp_path):
    state = {
        "status": "complete",
        "evaluation_audit_sha256": "old-audit",
        "post_evaluation_audit": "passed",
    }
    results = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "status": "complete",
        "metrics": {"overall": {"avg_score": 1.0}},
    }
    MOD.atomic_json(tmp_path / "results.json", results)
    MOD.atomic_json(tmp_path / "metrics.json", results["metrics"])
    MOD.atomic_json(tmp_path / "evaluation_audit.json", {"status": "passed"})
    MOD.persist_post_audit_failure(
        tmp_path, state, results, MOD.EvaluationError("integrity failure")
    )
    failed_results = json.loads((tmp_path / "results.json").read_text())
    failed_state = json.loads((tmp_path / "run_state.json").read_text())
    assert failed_results["status"] == "failed_post_evaluation_audit"
    assert failed_results["metrics"] is None
    assert failed_state["status"] == "failed"
    assert failed_state["metrics_sha256"] is None
    assert "evaluation_audit_sha256" not in failed_state
    assert failed_state["results_sha256"] == MOD.sha256_file(tmp_path / "results.json")
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "evaluation_audit.json").exists()


def make_formal_input(tmp_path):
    manifest_path = tmp_path / "run_manifest.json"
    MOD.atomic_json(
        manifest_path,
        {
            "schema_version": 2,
            "benchmark": "BEAM",
            "status": "complete",
            "failed_conversations": [],
        },
    )
    records = []
    for chat_size, indices in MOD.FORMAL_SELECTION.items():
        for conversation_index in indices:
            question_index = 0
            for question_type in MOD.EXPECTED_QUESTION_TYPES:
                for _ in range(MOD.FORMAL_QUESTIONS_PER_TYPE_PER_CONVERSATION):
                    records.append(
                        record(
                            (
                                f"{chat_size}_{conversation_index}_q"
                                f"{question_index}_{question_type}"
                            ),
                            question_type,
                            question_index=question_index,
                            chat_size=chat_size,
                            conversation_index=conversation_index,
                        )
                    )
                    question_index += 1
    evaluation_input = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "metric_scope": "rubric_nugget_only",
        "method": "NativeMem-v8.8+calendar",
        "answer_model": "gpt-5.5",
        "formal_scope_verified": True,
        "selected_conversations": MOD.FORMAL_SELECTION,
        "run_dir": str(tmp_path.resolve()),
        "run_manifest_sha256": MOD.sha256_file(manifest_path),
        "question_count": len(records),
        "rubric_nugget_count": len(records),
        "question_type_counts": {
            question_type: MOD.FORMAL_QUESTION_TYPE_COUNT
            for question_type in MOD.EXPECTED_QUESTION_TYPES
        },
        "records": records,
    }
    input_path = tmp_path / "evaluation_input.json"
    MOD.atomic_json(input_path, evaluation_input)
    MOD.atomic_json(
        tmp_path / "audit.json",
        {
            "schema_version": 1,
            "status": "passed",
            "benchmark": "BEAM",
            "formal_scope_verified": True,
            "selected_conversations": MOD.FORMAL_SELECTION,
            "conversation_count": 45,
            "questions": MOD.FORMAL_QUESTION_COUNT,
            "run_dir": str(tmp_path.resolve()),
            "run_manifest_path": str(manifest_path.resolve()),
            "run_manifest_sha256": MOD.sha256_file(manifest_path),
            "evaluation_input_path": str(input_path.resolve()),
            "evaluation_input_sha256": MOD.sha256_file(input_path),
        },
    )
    return input_path


def rewrite_formal_input(input_path, value, *, refresh_manifest=False):
    manifest_path = input_path.parent / "run_manifest.json"
    if refresh_manifest:
        value["run_manifest_sha256"] = MOD.sha256_file(manifest_path)
    MOD.atomic_json(input_path, value)
    audit_path = input_path.parent / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["evaluation_input_sha256"] = MOD.sha256_file(input_path)
    if refresh_manifest:
        audit["run_manifest_sha256"] = MOD.sha256_file(manifest_path)
    MOD.atomic_json(audit_path, audit)


def test_formal_input_requires_exact_45_by_20_conversation_inventory(tmp_path):
    input_path = make_formal_input(tmp_path)
    assert len(MOD.load_evaluation_input(input_path)["records"]) == 900

    forged = json.loads(input_path.read_text())
    for question_index, current in enumerate(forged["records"]):
        current["chat_size"] = "100K"
        current["conversation_index"] = 0
        current["question_index"] = question_index
        current["question_id"] = f"100K_0_q{question_index}_{current['question_type']}"
    rewrite_formal_input(input_path, forged)
    with pytest.raises(MOD.EvaluationError, match="selected conversation"):
        MOD.load_evaluation_input(input_path)


def test_formal_input_requires_sibling_audit_and_matching_input_hash(tmp_path):
    input_path = make_formal_input(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.unlink()
    with pytest.raises(MOD.EvaluationError, match="no sibling audit.json"):
        MOD.load_evaluation_input(input_path)

    input_path = make_formal_input(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["evaluation_input_sha256"] = "0" * 64
    MOD.atomic_json(audit_path, audit)
    with pytest.raises(MOD.EvaluationError, match="does not match"):
        MOD.load_evaluation_input(input_path)


def test_formal_input_requires_manifest_path_hash_and_complete_status(tmp_path):
    input_path = make_formal_input(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["run_manifest_path"] = str(tmp_path / "other-manifest.json")
    MOD.atomic_json(audit_path, audit)
    with pytest.raises(MOD.EvaluationError, match="run_manifest_path"):
        MOD.load_evaluation_input(input_path)

    input_path = make_formal_input(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    MOD.atomic_json(manifest_path, manifest)
    evaluation_input = json.loads(input_path.read_text())
    rewrite_formal_input(input_path, evaluation_input, refresh_manifest=True)
    with pytest.raises(MOD.EvaluationError, match="not complete and failure-free"):
        MOD.load_evaluation_input(input_path)


def test_formal_input_rejects_invalid_chat_size_as_evaluation_error(tmp_path):
    input_path = make_formal_input(tmp_path)
    evaluation_input = json.loads(input_path.read_text())
    evaluation_input["records"][0]["chat_size"] = []
    rewrite_formal_input(input_path, evaluation_input)
    with pytest.raises(MOD.EvaluationError, match="invalid chat size"):
        MOD.load_evaluation_input(input_path)


def make_complete_evaluation(tmp_path, monkeypatch, profile="primary"):
    monkeypatch.setattr(MOD, "FORMAL_QUESTION_COUNT", 10)
    monkeypatch.setattr(MOD, "FORMAL_QUESTION_TYPE_COUNT", 1)
    monkeypatch.setattr(MOD, "FORMAL_QUESTIONS_PER_CONVERSATION", 10)
    monkeypatch.setattr(MOD, "FORMAL_QUESTIONS_PER_TYPE_PER_CONVERSATION", 1)
    monkeypatch.setattr(MOD, "FORMAL_SELECTION", {"100K": [0]})
    records = [
        record(
            f"100K_0_q{index}_{question_type}",
            question_type,
            question_index=index,
        )
        for index, question_type in enumerate(MOD.EXPECTED_QUESTION_TYPES)
    ]
    manifest_path = tmp_path / "run_manifest.json"
    MOD.atomic_json(
        manifest_path,
        {
            "schema_version": 2,
            "benchmark": "BEAM",
            "status": "complete",
            "failed_conversations": [],
        },
    )
    evaluation_input = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "metric_scope": "rubric_nugget_only",
        "method": "NativeMem-v8.8+calendar",
        "answer_model": "gpt-5.5",
        "formal_scope_verified": True,
        "selected_conversations": MOD.FORMAL_SELECTION,
        "run_dir": str(tmp_path.resolve()),
        "run_manifest_sha256": MOD.sha256_file(manifest_path),
        "question_count": len(records),
        "rubric_nugget_count": len(records),
        "question_type_counts": {
            question_type: 1 for question_type in MOD.EXPECTED_QUESTION_TYPES
        },
        "records": records,
    }
    input_path = tmp_path / "evaluation_input.json"
    MOD.atomic_json(input_path, evaluation_input)
    MOD.atomic_json(
        tmp_path / "audit.json",
        {
            "schema_version": 1,
            "status": "passed",
            "benchmark": "BEAM",
            "formal_scope_verified": True,
            "selected_conversations": MOD.FORMAL_SELECTION,
            "conversation_count": 1,
            "questions": len(records),
            "run_dir": str(tmp_path.resolve()),
            "run_manifest_path": str(manifest_path.resolve()),
            "run_manifest_sha256": MOD.sha256_file(manifest_path),
            "evaluation_input_path": str(input_path.resolve()),
            "evaluation_input_sha256": MOD.sha256_file(input_path),
        },
    )
    output_dir = tmp_path / f"evaluation-{profile}"
    proxy_log = tmp_path / "judge-proxy.jsonl" if profile == "secondary" else None
    if proxy_log:
        proxy_log.write_text("", encoding="utf-8")
    config = judge_config(profile, proxy_log)
    input_sha256 = MOD.sha256_file(input_path)
    state = MOD.load_or_create_state(
        output_dir,
        input_path=input_path,
        input_sha256=input_sha256,
        config=config,
        total_jobs=len(records),
        resume=False,
    )
    proxy_entries = []
    for index, current in enumerate(records):
        response_id = f"judge-response-{index}"
        judgment = MOD.judge_one_job(
            fake_client(
                [{
                    "content": '{"score": 1, "reason": "checked"}',
                    "id": response_id,
                    "model": config["expected_response_model"],
                }]
            ),
            output_dir=output_dir,
            record=current,
            nugget_index=0,
            nugget=current["rubric"][0],
            input_sha256=input_sha256,
            config_hash=state["config_hash"],
            config=config,
            sleep_fn=lambda _: None,
        )
        assert judgment["status"] == "complete"
        proxy_entries.append(
            {
                "status": "success",
                "requested_model": config["model"],
                "actual_model": config["expected_response_model"],
                "response_id": response_id,
                "attempts": 2,
                "unsupported_parameters": ["max_output_tokens"],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
    if proxy_log:
        proxy_log.write_text(
            "".join(json.dumps(entry) + "\n" for entry in proxy_entries),
            encoding="utf-8",
        )
    judgments, missing = MOD.load_all_judgments(
        output_dir,
        evaluation_input,
        input_sha256=input_sha256,
        config_hash=state["config_hash"],
        config=config,
    )
    assert not missing
    results = MOD._results_payload(
        evaluation_input,
        judgments,
        input_sha256=input_sha256,
        config=config,
        config_hash=state["config_hash"],
    )
    MOD.write_result_artifacts(output_dir, results)
    state.update(
        {
            "status": "complete",
            "completed_nuggets": len(judgments),
            "failed_nuggets": 0,
            "missing_nuggets": 0,
            "results_sha256": MOD.sha256_file(output_dir / "results.json"),
            "metrics_sha256": MOD.sha256_file(output_dir / "metrics.json"),
        }
    )
    MOD.atomic_json(output_dir / "run_state.json", state)
    return input_path, output_dir, proxy_log


def test_post_evaluation_audit_recomputes_primary_results(tmp_path, monkeypatch):
    input_path, output_dir, _ = make_complete_evaluation(tmp_path, monkeypatch)
    report = MOD.post_evaluation_audit(input_path, output_dir)
    assert report["status"] == "passed"
    assert report["judge_profile"] == "primary"
    assert report["judge_response_ids"] == 10
    stored = json.loads((output_dir / "results.json").read_text())
    stored["evaluations"][0]["score"] = 0.0
    MOD.atomic_json(output_dir / "results.json", stored)
    with pytest.raises(MOD.EvaluationError, match="recomputed judgments"):
        MOD.post_evaluation_audit(input_path, output_dir)


def test_audit_only_profile_mismatch_invalidates_old_formal_artifacts(
    tmp_path, monkeypatch
):
    input_path, output_dir, _ = make_complete_evaluation(tmp_path, monkeypatch)
    MOD.atomic_json(output_dir / "evaluation_audit.json", {"status": "passed"})
    with pytest.raises(SystemExit) as exc_info:
        MOD.main(
            [
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--profile",
                "secondary",
                "--audit-only",
            ]
        )
    assert exc_info.value.code == 2
    results = json.loads((output_dir / "results.json").read_text())
    state = json.loads((output_dir / "run_state.json").read_text())
    assert results["status"] == "failed_post_evaluation_audit"
    assert results["metrics"] is None
    assert state["status"] == "failed"
    assert "evaluation_audit_sha256" not in state
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "evaluation_audit.json").exists()


def test_primary_formal_cli_requires_explicit_model_request_gate(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        MOD.main(
            [
                str(tmp_path / "missing-input.json"),
                "--profile",
                "primary",
            ]
        )
    assert exc_info.value.code == 2
    assert "--allow-model-requests" in capsys.readouterr().err


def test_secondary_post_audit_exactly_matches_proxy_ids(tmp_path, monkeypatch):
    input_path, output_dir, proxy_log = make_complete_evaluation(
        tmp_path, monkeypatch, profile="secondary"
    )
    report = MOD.post_evaluation_audit(input_path, output_dir, proxy_log=proxy_log)
    assert report["proxy_evidence"]["matched_response_ids"] == 10
    assert report["proxy_evidence"]["matched_proxy_upstream_attempts"] == 20
    assert report["proxy_evidence"]["requested_output_limit_enforced"] is False
    lines = proxy_log.read_text().splitlines()
    proxy_log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(MOD.EvaluationError, match="absent from the proxy log"):
        MOD.post_evaluation_audit(input_path, output_dir, proxy_log=proxy_log)


def test_secondary_post_audit_rejects_proxy_fallback_model(tmp_path, monkeypatch):
    input_path, output_dir, proxy_log = make_complete_evaluation(
        tmp_path, monkeypatch, profile="secondary"
    )
    entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    entries[0]["actual_model"] = "fallback-model"
    proxy_log.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    with pytest.raises(MOD.EvaluationError, match="wrong model"):
        MOD.post_evaluation_audit(input_path, output_dir, proxy_log=proxy_log)
