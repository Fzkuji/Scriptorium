from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_r116_formal as auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r116_formal_contract as contract  # noqa: E402
import run_r116_formal as runner  # noqa: E402


def test_preregistration_freezes_two_complete_scopes_and_live_sources():
    prereg = contract.validate_preregistration()

    assert prereg["status"] == "frozen_before_formal_model_requests"
    assert prereg["benchmarks"]["locomo"]["scope"]["questions"] == 1986
    assert prereg["benchmarks"]["locomo"]["scope"]["primary_cat1_4"] == 1540
    assert prereg["benchmarks"]["locomo"]["scope"]["category_5"] == 446
    assert prereg["benchmarks"]["longmemeval-s"]["scope"]["questions"] == 500
    assert (
        prereg["benchmarks"]["longmemeval-s"]["scope"]["abstention_questions"]
        == 30
    )
    assert prereg["retrieval_and_answer"]["visible_token_budget_tokens"] == 20_000
    assert prereg["retrieval_and_answer"]["single_visible_token_gate"] is True
    assert prereg["retrieval_and_answer"]["retrieval_model"] == "gpt-5.5"
    assert prereg["retrieval_and_answer"]["answer_model"] == "gpt-5.5"
    assert prereg["actual_model_evidence"][
        "gateway_origin_derived_from_validated_root"
    ] is True
    assert prereg["actual_model_evidence"]["exact_provider_window_required"] is True
    for relative, expected in prereg["source_hashes"].items():
        assert answer_contract.sha256_file(ROOT / relative) == expected


def test_formal_question_inventory_and_session_mapping_are_exact():
    locomo = contract.question_specs(contract.LOCOMO)
    lme = contract.question_specs(contract.LONGMEMEVAL)

    assert len(locomo) == 1986
    assert sum(row["source_recall_eligible"] for row in locomo) == 1533
    assert len(lme) == 500
    assert sum(row["abstention"] for row in lme) == 30
    assert all(row["gold_source_granularity"] == "session" for row in lme)
    turn_index, session_by_turn = contract.turn_index_and_session_map(lme[0])
    assert set(turn_index) == set(session_by_turn)
    assert set(lme[0]["gold_source_ids"]) <= set(session_by_turn.values())


def test_two_benchmark_synthetic_run_is_idempotent_and_tamper_evident(tmp_path):
    output = tmp_path / "r116-formal-synthetic"
    first = runner.run_synthetic(output)
    second = runner.run_synthetic(output)
    independent = auditor.audit_run(output)

    assert first == second == independent
    assert independent["status"] == "passed"
    assert independent["mode"] == "synthetic_no_network"
    assert independent["questions"] == 2
    assert independent["source_recall_questions"] == 2
    assert independent["mean_mapped_source_recall"] == 1.0
    assert independent["budget_tokens_per_question"] == 20_000
    assert independent["network_requests"] == 0
    assert independent["response_ids"] == 8
    assert independent["actual_model_evidence"]["events"] == 8
    assert independent["aggregate_outputs"]["results_sha256"]
    assert independent["aggregate_outputs"]["hypotheses_sha256"]
    lme_recall = next(
        path
        for path in output.glob("questions/*/source_recall.json")
        if json.loads(path.read_text(encoding="utf-8"))["benchmark"]
        == contract.LONGMEMEVAL
    )
    recall = json.loads(lme_recall.read_text(encoding="utf-8"))
    assert recall["gold_source_granularity"] == "session"
    assert recall["derivation"] == "delivered_Dn_m_to_haystack_session_ids"
    assert recall["mapped_source_recall"] == 1.0

    recall["retrieved_source_ids"] = []
    lme_recall.write_text(json.dumps(recall), encoding="utf-8")
    with pytest.raises(Exception, match="hash|recall"):
        auditor.audit_run(output)


def test_incomplete_synthetic_root_and_unauthorized_formal_are_rejected(
    tmp_path, monkeypatch
):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "partial").write_text("x", encoding="utf-8")
    with pytest.raises(runner.R116RunError, match="incomplete"):
        runner.run_synthetic(incomplete)

    with pytest.raises(SystemExit) as missing_authority:
        runner.parse_args(
            [
                "--formal",
                "--benchmark",
                "locomo",
                "--output-dir",
                str(tmp_path / "formal"),
            ]
        )
    with pytest.raises(SystemExit) as synthetic_authority:
        runner.parse_args(
            [
                "--synthetic-sanity",
                "--output-dir",
                str(tmp_path / "synthetic"),
                "--allow-model-requests",
            ]
        )
    assert missing_authority.value.code == 2
    assert synthetic_authority.value.code == 2

    monkeypatch.setattr(
        contract,
        "run_preflight",
        lambda *args, **kwargs: {
            "benchmarks": {
                "locomo": {"status": "blocked", "reason": "source incomplete"}
            }
        },
    )
    proxy_started = False

    def forbidden_proxy(*args, **kwargs):
        nonlocal proxy_started
        proxy_started = True
        raise AssertionError("proxy must not start while preflight is blocked")

    monkeypatch.setattr(runner.controlled_answer, "_start_proxy", forbidden_proxy)
    with pytest.raises(contract.R116FormalError, match="blocked"):
        runner.run_formal(
            benchmark="locomo",
            output_dir=tmp_path / "blocked-formal",
            gateway_root=tmp_path / "unused-gateway",
            allow_model_requests=True,
        )
    assert proxy_started is False


def test_persisted_preflight_is_explicitly_blocked_without_model_requests():
    report = answer_contract.read_json(contract.DEFAULT_PREFLIGHT)

    assert report["schema_version"] == contract.PREFLIGHT_SCHEMA
    assert report["status"] == "blocked"
    assert report["evidence_mapping"]["status"] == "passed"
    assert report["benchmarks"]["locomo"]["status"] == "blocked"
    assert report["benchmarks"]["longmemeval-s"]["status"] == "blocked"
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0
