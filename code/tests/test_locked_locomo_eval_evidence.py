from __future__ import annotations

import hashlib
import json
import os
import runpy
import socket
import sys
import time
from pathlib import Path
from typing import Any

import requests
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluation" / "eval_full.py"
EXPECTED_EVALUATOR_SHA256 = (
    "17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b"
)


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: Any) -> None:
    def reject_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("test attempted a real network connection")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)


def _write_input(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "sample0_questions.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question": "When did the event happen?",
                    "gold": "7 May 2023",
                    "answer": "2023-05-07",
                    "category": 2,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _response(payload: dict[str, Any]) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(  # noqa: SLF001
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    response.url = "https://openrouter.ai/api/v1/chat/completions"
    return response


def _run_native(run_dir: Path, fake_post: Any) -> None:
    original_post = requests.post
    original_argv = sys.argv[:]
    requests.post = fake_post
    sys.argv = [
        str(EVALUATOR),
        str(run_dir),
        "answer-model-test",
        "http://answerer.invalid/v1",
        "answer-key-test",
        "openrouter-key-test",
    ]
    try:
        runpy.run_path(str(EVALUATOR), run_name="__main__")
    finally:
        sys.argv = original_argv
        requests.post = original_post


def test_instrumented_execution_is_byte_identical_and_records_response_evidence(
    tmp_path: Path,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    assert hashlib.sha256(EVALUATOR.read_bytes()).hexdigest() == (
        EXPECTED_EVALUATOR_SHA256
    )
    native_dir = tmp_path / "native"
    instrumented_dir = tmp_path / "instrumented"
    _write_input(native_dir)
    _write_input(instrumented_dir)

    response_payload = {
        "id": "chatcmpl-test-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {"message": {"content": '{"label":"CORRECT"}'}}
        ],
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 7,
            "total_tokens": 108,
        },
    }
    observed_requests: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> requests.Response:
        observed_requests.append({"args": args, "kwargs": kwargs})
        return _response(response_payload)

    _run_native(native_dir, fake_post)
    evidence_path = tmp_path / "judge_requests.jsonl"
    original_post = requests.post
    requests.post = fake_post
    try:
        instrumented.run_locked_evaluator(
            [
                str(instrumented_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )
    finally:
        requests.post = original_post

    assert (native_dir / "eval_full.json").read_bytes() == (
        instrumented_dir / "eval_full.json"
    ).read_bytes()
    assert len(observed_requests) == 2
    assert observed_requests[0] == observed_requests[1]
    for observed in observed_requests:
        assert observed["args"] == (
            "https://openrouter.ai/api/v1/chat/completions",
        )
        assert observed["kwargs"]["json"]["model"] == "openai/gpt-4o-mini"
        assert observed["kwargs"]["json"]["temperature"] == 0
        assert observed["kwargs"]["json"]["response_format"] == {
            "type": "json_object"
        }

    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "judge_request_started",
        "judge_request_succeeded",
    ]
    success = rows[1]
    assert success["attempt"] == 1
    assert success["request"]["requested_model"] == "openai/gpt-4o-mini"
    assert success["request"]["prompt_sha256"] == hashlib.sha256(
        observed_requests[1]["kwargs"]["json"]["messages"][0]["content"].encode(
            "utf-8"
        )
    ).hexdigest()
    assert len(success["request"]["json_canonical_sha256"]) == 64
    request_identity = {
        "json": observed_requests[1]["kwargs"]["json"],
        "method": "POST",
        "url": "https://openrouter.ai/api/v1/chat/completions",
    }
    assert success["request"]["request_canonical_sha256"] == hashlib.sha256(
        json.dumps(
            request_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    response_body = _response(response_payload).content
    response_body_sha256 = hashlib.sha256(response_body).hexdigest()
    body_artifact = (
        tmp_path
        / "judge_requests.response_bodies"
        / f"{response_body_sha256}.body"
    )
    assert success["response"] == {
        "body_artifact": str(body_artifact.resolve()),
        "body_bytes": len(response_body),
        "body_sha256": response_body_sha256,
        "id": "chatcmpl-test-1",
        "model": "openai/gpt-4o-mini",
        "usage": response_payload["usage"],
    }
    assert body_artifact.read_bytes() == response_body

    manifest = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["evaluator"]["sha256_before"] == EXPECTED_EVALUATOR_SHA256
    assert manifest["evaluator"]["sha256_after"] == EXPECTED_EVALUATOR_SHA256
    assert manifest["requests"] == {"failed": 0, "started": 1, "succeeded": 1}
    assert manifest["run_binding"] == {
        "answerer_base": "http://answerer.invalid/v1",
        "answerer_model": "answer-model-test",
        "eval_output": {
            "bytes": (instrumented_dir / "eval_full.json").stat().st_size,
            "path": str((instrumented_dir / "eval_full.json").resolve()),
            "sha256": hashlib.sha256(
                (instrumented_dir / "eval_full.json").read_bytes()
            ).hexdigest(),
        },
        "inputs": [
            {
                "bytes": (instrumented_dir / "sample0_questions.json").stat().st_size,
                "path": str(
                    (instrumented_dir / "sample0_questions.json").resolve()
                ),
                "sha256": hashlib.sha256(
                    (instrumented_dir / "sample0_questions.json").read_bytes()
                ).hexdigest(),
            }
        ],
        "run_dir": str(instrumented_dir.resolve()),
    }


def test_retry_attempts_and_exceptions_are_recorded_and_audited(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "retry"
    _write_input(run_dir)
    response_payload = {
        "id": "chatcmpl-test-retry",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {"message": {"content": '{"label":"CORRECT"}'}}
        ],
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 7,
            "total_tokens": 108,
        },
    }
    physical_attempts = 0

    def flaky_post(*_args: Any, **_kwargs: Any) -> requests.Response:
        nonlocal physical_attempts
        physical_attempts += 1
        if physical_attempts < 3:
            raise requests.Timeout(f"timeout-{physical_attempts}")
        return _response(response_payload)

    monkeypatch.setattr(requests, "post", flaky_post)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    evidence_path = tmp_path / "retry.jsonl"
    instrumented.run_locked_evaluator(
        [
            str(run_dir),
            "answer-model-test",
            "http://answerer.invalid/v1",
            "answer-key-test",
            "openrouter-key-test",
        ],
        evidence_path=evidence_path,
    )

    assert physical_attempts == 3
    assert json.loads((run_dir / "eval_full.json").read_text(encoding="utf-8"))[
        "overall"
    ] == 1.0
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "judge_request_started",
        "judge_request_failed",
        "judge_request_started",
        "judge_request_failed",
        "judge_request_started",
        "judge_request_succeeded",
    ]
    assert [row["attempt"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert [rows[index]["error"]["message"] for index in (1, 3)] == [
        "timeout-1",
        "timeout-2",
    ]
    report = instrumented.audit_evidence(evidence_path)
    assert report == {
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "failed_attempts": 2,
        "failed_only_request_hashes": [],
        "requests_started": 3,
        "response_ids": ["chatcmpl-test-retry"],
        "status": "passed",
        "successful_attempts": 1,
    }


def test_instrumentation_preserves_category_filter_parser_and_output_schema(
    tmp_path: Path,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    records = [
        {
            "question_id": f"q-{category}",
            "question": f"Question {category}?",
            "gold": f"Gold {category}",
            "answer": f"Answer {category}",
            "category": category,
        }
        for category in range(1, 6)
    ]
    native_dir = tmp_path / "native-categories"
    instrumented_dir = tmp_path / "instrumented-categories"
    for run_dir in (native_dir, instrumented_dir):
        run_dir.mkdir()
        (run_dir / "sample0_questions.json").write_text(
            json.dumps(records), encoding="utf-8"
        )

    labels = {
        "Question 1?": '{"label":"CORRECT"}',
        "Question 2?": '{"label":"WRONG"}',
        "Question 3?": '{"unexpected":"value"}',
        "Question 4?": '{"label":"correct"}',
    }

    def fake_post(*_args: Any, **kwargs: Any) -> requests.Response:
        prompt = kwargs["json"]["messages"][0]["content"]
        question = next(question for question in labels if question in prompt)
        return _response(
            {
                "id": f"chatcmpl-{question.split()[1].rstrip('?')}",
                "model": "openai/gpt-4o-mini",
                "choices": [{"message": {"content": labels[question]}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )

    _run_native(native_dir, fake_post)
    original_post = requests.post
    requests.post = fake_post
    try:
        instrumented.run_locked_evaluator(
            [
                str(instrumented_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=tmp_path / "categories.jsonl",
        )
    finally:
        requests.post = original_post

    native_bytes = (native_dir / "eval_full.json").read_bytes()
    instrumented_bytes = (instrumented_dir / "eval_full.json").read_bytes()
    assert instrumented_bytes == native_bytes
    output = json.loads(native_bytes)
    assert set(output) == {"by_category", "n", "overall", "records"}
    assert output["n"] == 4
    assert [record["category"] for record in output["records"]] == [1, 2, 3, 4]
    assert [record["judge_score"] for record in output["records"]] == [1, 0, 0, 1]


def test_evidence_extraction_failure_cannot_change_evaluator_output(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    native_dir = tmp_path / "native-evidence-failure"
    instrumented_dir = tmp_path / "instrumented-evidence-failure"
    _write_input(native_dir)
    _write_input(instrumented_dir)
    payload = {
        "id": "chatcmpl-json-only",
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"content": '{"label":"CORRECT"}'}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }

    class JsonOnlyResponse:
        @property
        def content(self) -> bytes:
            raise RuntimeError("raw body unavailable")

        def json(self) -> dict[str, Any]:
            return payload

    def fake_post(*_args: Any, **_kwargs: Any) -> JsonOnlyResponse:
        return JsonOnlyResponse()

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _run_native(native_dir, fake_post)
    monkeypatch.setattr(requests, "post", fake_post)
    evidence_path = tmp_path / "evidence-failure.jsonl"
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="evidence recording failed",
    ):
        instrumented.run_locked_evaluator(
            [
                str(instrumented_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )

    assert (instrumented_dir / "eval_full.json").read_bytes() == (
        native_dir / "eval_full.json"
    ).read_bytes()
    manifest = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "evidence_failed"
    assert manifest["recording_errors"] == [
        {
            "message": "raw body unavailable",
            "request_sequence": 1,
            "stage": "response_evidence",
            "type": "builtins.RuntimeError",
        }
    ]


def test_evidence_redacts_authorization_values_from_exceptions(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "redaction"
    _write_input(run_dir)

    def failing_post(*_args: Any, **kwargs: Any) -> requests.Response:
        raise requests.ConnectionError(
            f"provider rejected {kwargs['headers']['Authorization']}"
        )

    monkeypatch.setattr(requests, "post", failing_post)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    evidence_path = tmp_path / "redaction.jsonl"
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="judge request exhausted all evaluator attempts",
    ):
        instrumented.run_locked_evaluator(
            [
                str(run_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )

    artifacts = evidence_path.read_text(encoding="utf-8") + (
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    for artifact in sorted(instrumented.request_body_dir_for(evidence_path).glob("*")):
        artifacts += artifact.read_text(encoding="utf-8")
    for artifact in sorted(instrumented.response_body_dir_for(evidence_path).glob("*")):
        artifacts += artifact.read_text(encoding="utf-8")
    assert "openrouter-key-test" not in artifacts
    assert "answer-key-test" not in artifacts
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    errors = [row["error"]["message"] for row in rows if "error" in row]
    assert errors == ["provider rejected [REDACTED]"] * 3
    report = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )["audit_report"]
    assert len(report["failed_only_request_hashes"]) == 1
    assert report["failed_only_request_hashes"][0] == rows[0]["request"][
        "request_canonical_sha256"
    ]


@pytest.mark.parametrize("tamper_kind", ["request_hashes", "response_metadata"])
def test_audit_recomputes_metadata_from_raw_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
    tamper_kind: str,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / tamper_kind
    _write_input(run_dir)
    payload = {
        "id": "chatcmpl-audit-source",
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"content": '{"label":"CORRECT"}'}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: _response(payload))
    evidence_path = tmp_path / f"{tamper_kind}.jsonl"
    instrumented.run_locked_evaluator(
        [
            str(run_dir),
            "answer-model-test",
            "http://answerer.invalid/v1",
            "answer-key-test",
            "openrouter-key-test",
        ],
        evidence_path=evidence_path,
    )

    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert Path(rows[0]["request"]["json_artifact"]).is_file()
    if tamper_kind == "request_hashes":
        for row in rows:
            row["request"]["prompt_sha256"] = "0" * 64
            row["request"]["json_canonical_sha256"] = "1" * 64
            row["request"]["request_canonical_sha256"] = "2" * 64
        error = "request metadata differs from raw artifact"
    else:
        rows[-1]["response"]["id"] = "chatcmpl-forged"
        rows[-1]["response"]["model"] = "forged/model"
        rows[-1]["response"]["usage"] = {"total_tokens": 999999}
        error = "response metadata differs from raw artifact"
    evidence_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = instrumented.manifest_path_for(evidence_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(instrumented.LockedEvaluatorEvidenceError, match=error):
        instrumented.audit_evidence(evidence_path)


def test_cli_reads_api_keys_from_named_environment_variables(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    monkeypatch.setenv("TEST_ANSWERER_SECRET", "answerer-secret-value")
    monkeypatch.setenv("TEST_JUDGE_SECRET", "judge-secret-value")
    captured: dict[str, Any] = {}

    def fake_run(evaluator_args: Any, *, evidence_path: Path) -> None:
        captured["evaluator_args"] = list(evaluator_args)
        captured["evidence_path"] = evidence_path

    monkeypatch.setattr(instrumented, "run_locked_evaluator", fake_run)
    argv = [
        "--evidence",
        str(tmp_path / "evidence.jsonl"),
        "--answerer-key-env",
        "TEST_ANSWERER_SECRET",
        "--openrouter-key-env",
        "TEST_JUDGE_SECRET",
        str(tmp_path / "run"),
        "answer-model-test",
        "http://answerer.invalid/v1",
    ]
    assert "answerer-secret-value" not in argv
    assert "judge-secret-value" not in argv
    assert instrumented.main(argv) == 0
    assert captured == {
        "evaluator_args": [
            str(tmp_path / "run"),
            "answer-model-test",
            "http://answerer.invalid/v1",
            "answerer-secret-value",
            "judge-secret-value",
        ],
        "evidence_path": tmp_path / "evidence.jsonl",
    }


def test_exact_evaluator_copy_is_rejected_even_when_sha_matches(tmp_path: Path) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    copied_evaluator = tmp_path / "eval_full.py"
    copied_evaluator.write_bytes(EVALUATOR.read_bytes())
    assert hashlib.sha256(copied_evaluator.read_bytes()).hexdigest() == (
        EXPECTED_EVALUATOR_SHA256
    )
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="only permitted evaluator path",
    ):
        instrumented.verify_locked_evaluator(copied_evaluator)


def test_launcher_fails_closed_when_automatic_evidence_audit_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "missing-usage"
    _write_input(run_dir)
    payload_without_usage = {
        "id": "chatcmpl-missing-usage",
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"content": '{"label":"CORRECT"}'}}],
    }
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _response(payload_without_usage),
    )
    evidence_path = tmp_path / "missing-usage.jsonl"
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="judge response evidence is incomplete",
    ):
        instrumented.run_locked_evaluator(
            [
                str(run_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )

    assert json.loads((run_dir / "eval_full.json").read_text(encoding="utf-8"))[
        "overall"
    ] == 1.0
    manifest = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "audit_failed"
    assert manifest["audit_error"] == {
        "message": "judge response evidence is incomplete",
        "type": (
            "scripts.evaluation.run_locked_locomo_eval_with_evidence."
            "LockedEvaluatorEvidenceError"
        ),
    }


@pytest.mark.parametrize(
    ("tamper_kind", "error"),
    [
        ("missing_outcome", "start/outcome pairing differs"),
        ("changed_output", "run binding differs"),
    ],
)
def test_audit_rejects_incomplete_ledger_and_changed_eval_output(
    tmp_path: Path,
    monkeypatch: Any,
    tamper_kind: str,
    error: str,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / tamper_kind
    _write_input(run_dir)
    payload = {
        "id": f"chatcmpl-{tamper_kind}",
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"content": '{"label":"CORRECT"}'}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: _response(payload))
    evidence_path = tmp_path / f"{tamper_kind}.jsonl"
    instrumented.run_locked_evaluator(
        [
            str(run_dir),
            "answer-model-test",
            "http://answerer.invalid/v1",
            "answer-key-test",
            "openrouter-key-test",
        ],
        evidence_path=evidence_path,
    )

    if tamper_kind == "missing_outcome":
        rows = evidence_path.read_text(encoding="utf-8").splitlines()
        evidence_path.write_text(rows[0] + "\n", encoding="utf-8")
        manifest_path = instrumented.manifest_path_for(evidence_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence"]["sha256"] = hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        with (run_dir / "eval_full.json").open("ab") as handle:
            handle.write(b"\n")

    with pytest.raises(instrumented.LockedEvaluatorEvidenceError, match=error):
        instrumented.audit_evidence(evidence_path)


def test_programmatic_launcher_restores_answerer_environment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "restore-environment"
    _write_input(run_dir)
    payload = {
        "id": "chatcmpl-restore-environment",
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"content": '{"label":"CORRECT"}'}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: _response(payload))
    original = {
        "ANSWERER_MODEL": "prior-model",
        "ANSWERER_BASE": "prior-base",
        "ANSWERER_KEY": "prior-key",
    }
    for key, value in original.items():
        monkeypatch.setenv(key, value)

    instrumented.run_locked_evaluator(
        [
            str(run_dir),
            "answer-model-test",
            "http://answerer.invalid/v1",
            "answer-key-test",
            "openrouter-key-test",
        ],
        evidence_path=tmp_path / "restore-environment.jsonl",
    )

    assert {key: os.environ.get(key) for key in original} == original


def test_launcher_rejects_request_exhausted_after_three_provider_failures(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "exhausted"
    _write_input(run_dir)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.Timeout("provider unavailable")
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    evidence_path = tmp_path / "exhausted.jsonl"
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="judge request exhausted all evaluator attempts",
    ):
        instrumented.run_locked_evaluator(
            [
                str(run_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )

    assert json.loads((run_dir / "eval_full.json").read_text(encoding="utf-8"))[
        "overall"
    ] == 0.0
    manifest = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "audit_failed"
    assert len(manifest["audit_report"]["failed_only_request_hashes"]) == 1


def test_launcher_rejects_response_missing_evaluator_content_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from scripts.evaluation import run_locked_locomo_eval_with_evidence as instrumented

    run_dir = tmp_path / "malformed-response"
    _write_input(run_dir)
    calls = 0

    def malformed_post(*_args: Any, **_kwargs: Any) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(
            {
                "id": f"chatcmpl-malformed-{calls}",
                "model": "openai/gpt-4o-mini",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                },
            }
        )

    monkeypatch.setattr(requests, "post", malformed_post)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    evidence_path = tmp_path / "malformed-response.jsonl"
    with pytest.raises(
        instrumented.LockedEvaluatorEvidenceError,
        match="response body lacks evaluator content path",
    ):
        instrumented.run_locked_evaluator(
            [
                str(run_dir),
                "answer-model-test",
                "http://answerer.invalid/v1",
                "answer-key-test",
                "openrouter-key-test",
            ],
            evidence_path=evidence_path,
        )

    assert calls == 3
    assert json.loads((run_dir / "eval_full.json").read_text(encoding="utf-8"))[
        "overall"
    ] == 0.0
    manifest = json.loads(
        instrumented.manifest_path_for(evidence_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "audit_failed"
