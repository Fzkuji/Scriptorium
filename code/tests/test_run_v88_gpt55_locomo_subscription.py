import importlib.util
import argparse
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_v88_gpt55_locomo_subscription.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v88_gpt55_locomo_subscription", SCRIPT
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_health_url_accepts_only_loopback():
    assert MOD.health_url("http://127.0.0.1:8199/v1") == (
        "http://127.0.0.1:8199/healthz"
    )
    with pytest.raises(ValueError, match="loopback"):
        MOD.health_url("https://example.com/v1")


def test_validate_proxy_health_requires_off_fixed_concurrency_and_log(tmp_path):
    log = tmp_path / "proxy_requests.jsonl"
    health = {
        "status": "ok",
        "auth_readable": True,
        "max_concurrency": 2,
        "max_attempts": 2,
        "read_timeout_s": 150.0,
        "requested_reasoning_effort": "none",
        "request_log": str(log),
    }
    MOD.validate_proxy_health(health, log)

    health["requested_reasoning_effort"] = "low"
    with pytest.raises(RuntimeError, match="reasoning.effort=none"):
        MOD.validate_proxy_health(health, log)


def test_proxy_summary_verifies_zero_reasoning(tmp_path):
    log = tmp_path / "proxy_requests.jsonl"
    records = [
        {
            "status": "success",
            "requested_reasoning_effort": "none",
            "actual_reasoning_effort": "none",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
    ]
    log.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = MOD.summarize_proxy_log(log)

    assert summary["successes"] == 1
    assert summary["reasoning_tokens"] == 0
    assert summary["thinking_off_verified"] is True

    records[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
    log.write_text(json.dumps(records[0]) + "\n")
    assert MOD.summarize_proxy_log(log)["thinking_off_verified"] is False


def test_cli_requires_explicit_model_request_authorization(tmp_path, capsys):
    with pytest.raises(SystemExit):
        MOD.main(["--output-dir", str(tmp_path / "out")])
    assert "allow-model-requests" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_subscription_disables_sdk_retries_and_uses_longer_timeout():
    args = argparse.Namespace(proxy_base_url="http://127.0.0.1:8199/v1")
    MOD.prepare_shared_args(args)
    env = MOD.shared.experiment_env(args)

    assert env["NATIVEMEM_OPENAI_MAX_RETRIES"] == "0"
    assert env["NATIVEMEM_HTTP_TIMEOUT"] == "600"


def test_import_sample0_memory_copies_and_records_provenance(tmp_path):
    source_root = tmp_path / "old"
    memory = source_root / "memory_sample0"
    memory.mkdir(parents=True)
    (memory / "topic.md").write_text("memory", encoding="utf-8")
    (source_root / "run_manifest.json").write_text(
        json.dumps({"fingerprint": "old-fingerprint"}), encoding="utf-8"
    )
    output = tmp_path / "new"
    output.mkdir()

    provenance = MOD.import_sample0_memory(source_root, output)

    assert (output / "memory_sample0" / "topic.md").read_text() == "memory"
    assert provenance["source_fingerprint"] == "old-fingerprint"
    seed = json.loads((output / "sample0_questions.json").read_text())
    assert seed[0]["question_id"] == "_build_stats"
    assert provenance["memory_sha256"] in seed[0]["notes"]
