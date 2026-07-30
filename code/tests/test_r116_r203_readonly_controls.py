from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_r116_nativemem_shared_answer as r116_auditor  # noqa: E402
import audit_r203_readonly_views as r203_auditor  # noqa: E402
import audit_readonly_nativemem_control as generic_auditor  # noqa: E402
import readonly_nativemem_control as control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
import run_r116_nativemem_shared_answer as r116  # noqa: E402
import run_r203_readonly_views as r203  # noqa: E402
from src.evaluation.visible_token_budget import TokenCounter  # noqa: E402


def _memory_fixture(root: Path) -> Path:
    memory = root / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline/2025/01").mkdir(parents=True)
    (memory / "topics/facts.md").write_text(
        "[2025-01-01] Uma records a public fact [D1:1].\n",
        encoding="utf-8",
    )
    (memory / "timeline/2025/01/01.md").write_text(
        "[2025-01-01] Uma records a public fact [D1:1].\n",
        encoding="utf-8",
    )
    return memory


def _turn_index() -> dict[str, dict[str, str]]:
    return {
        "D1:1": {
            "date": "1 January 2025",
            "speaker": "Uma",
            "text": "I recorded a public fact.",
        }
    }


def test_r116_synthetic_is_audited_and_complete_rerun_is_idempotent(tmp_path):
    output = tmp_path / "r116"
    first = r116.run_synthetic(output)
    second = r116.run_synthetic(output)
    independent = r116_auditor.audit_run(output)

    assert first == second == independent
    assert independent["status"] == "passed"
    assert independent["network_requests"] == 0
    assert independent["budget_tokens"] == 20_000
    assert independent["model"] == "gpt-5.5"
    assert independent["memory_unchanged"] is True
    assert independent["question_audit"]["mapped_source_recall"] == 1.0
    assert independent["formal_status"] == "blocked_waiting_frozen_nativemem_memory"


def test_r116_rejects_incomplete_existing_root_and_formal_cli(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(r116.R116Error, match="incomplete"):
        r116.run_synthetic(incomplete)
    with pytest.raises(SystemExit) as formal:
        r116.parse_args(["--output-dir", str(tmp_path / "formal")])
    with pytest.raises(SystemExit) as network:
        r116.parse_args(
            [
                "--output-dir",
                str(tmp_path / "network"),
                "--synthetic-sanity",
                "--allow-model-requests",
            ]
        )
    assert formal.value.code == 2
    assert network.value.code == 2


def test_r203_four_conditions_are_read_only_and_byte_identical(tmp_path):
    output = tmp_path / "r203"
    first = r203.run_synthetic(output)
    second = r203.run_synthetic(output)
    independent = r203_auditor.audit_run(output)

    assert first == second == independent
    assert independent["conditions"] == list(control.CONDITIONS)
    assert independent["condition_count"] == 4
    assert independent["network_requests"] == 0
    assert independent["byte_identical_start"] is True
    assert independent["memory_unchanged"] is True
    hashes = {
        report["memory_sha256"]
        for report in independent["condition_audits"].values()
    }
    assert len(hashes) == 1
    for condition, report in independent["condition_audits"].items():
        assert report["mapped_source_recall"] == 1.0
        stage = report["stage_evidence"]
        assert stage["condition_id"] == condition
        assert stage["gold_source_mapping_complete"] is True
        assert stage["gold_source_in_canonical_entries"] is True
        assert stage["gold_source_survived_maintenance"] is True
        assert stage["gold_source_path_valid"] is True
        assert stage["retrieval_reached_gold_source"] is True
        assert stage["all_gold_required"] is True
        for count in stage["stage_counts"].values():
            assert count["expected_count"] == 1
            assert count["ratio"] in {0.0, 1.0}
        assert stage["trace_ids"]["mapping"]
        assert stage["trace_ids"]["canonical_entries"]
        assert stage["trace_ids"]["maintenance"]
        assert stage["trace_ids"]["paths"]
        assert stage["trace_ids"]["retrieval"]
        if condition == "dual_no_source":
            assert report["source_resolution_tokens"] == 0
            assert stage["resolver_enabled"] is False
            assert stage["source_resolution_returned_gold_content"] is False
            assert stage["trace_ids"]["source_resolution"] == []
        else:
            assert report["source_resolution_tokens"] > 0
            assert stage["resolver_enabled"] is True
            assert stage["source_resolution_returned_gold_content"] is True
            assert stage["trace_ids"]["source_resolution"]
    assert (
        independent["formal_status"]
        == "blocked_waiting_new_all_ten_flex_source"
    )


def test_r203_requires_explicit_formal_mode_and_authority(tmp_path):
    with pytest.raises(SystemExit) as formal:
        r203.parse_args(["--output-dir", str(tmp_path / "formal")])
    with pytest.raises(SystemExit) as unauthorized:
        r203.parse_args(
            [
                "--formal",
                "--output-dir",
                str(tmp_path / "formal"),
                "--gateway-root",
                str(tmp_path / "gateway"),
            ]
        )
    with pytest.raises(SystemExit) as network:
        r203.parse_args(
            [
                "--output-dir",
                str(tmp_path / "network"),
                "--synthetic-sanity",
                "--allow-model-requests",
            ]
        )
    assert formal.value.code == 2
    assert unauthorized.value.code == 2
    assert network.value.code == 2


def test_r203_stage_evidence_provenance_is_fail_closed_before_calls(tmp_path):
    memory = _memory_fixture(tmp_path)
    common = {
        "run_id": "stage-fail-closed",
        "method": r203.METHOD,
        "condition": "dual_source",
        "memory_root": memory,
        "turn_index": _turn_index(),
        "question_id": "s0_q0",
        "question": "What did Uma record?",
        "gold_source_ids": ["D1:1"],
        "source_recall_eligible": True,
        "completion_resource": None,
        "answer_client": None,
        "tokenizer": TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        "formal": False,
        "proxy_log": None,
    }
    missing = tmp_path / "missing-stage"
    with pytest.raises(control.ReadOnlyControlError, match="requires complete"):
        control.execute_question(artifact_dir=missing, **common)
    assert not missing.exists()

    malformed = tmp_path / "malformed-stage"
    with pytest.raises(control.ReadOnlyControlError, match="provenance"):
        control.execute_question(
            artifact_dir=malformed,
            stage_provenance={},
            **common,
        )
    assert not malformed.exists()


def test_readonly_store_rejects_path_escape_and_disabled_source_resolution(tmp_path):
    memory = _memory_fixture(tmp_path)
    topic_store = control.ReadOnlyMemoryStore(
        root=memory,
        condition="topic_source",
        turn_index=_turn_index(),
    )
    with pytest.raises(control.ReadOnlyControlError, match="escapes"):
        topic_store.execute("read_memory_file", {"path": "../outside.md"})
    with pytest.raises(control.ReadOnlyControlError, match="condition views"):
        topic_store.execute(
            "read_memory_file", {"path": "timeline/2025/01/01.md"}
        )

    no_source_store = control.ReadOnlyMemoryStore(
        root=memory,
        condition="dual_no_source",
        turn_index=_turn_index(),
    )
    with pytest.raises(control.ReadOnlyControlError, match="disabled"):
        no_source_store.execute("resolve_sources", {"source_ids": ["D1:1"]})


def test_retrieval_returns_inaccessible_memory_path_as_structured_tool_error(
    tmp_path,
):
    memory = _memory_fixture(tmp_path)
    artifact = tmp_path / "attempt"
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = controlled_answer.FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="missing-path-run",
        response_text="<answer>insufficient evidence</answer>",
    )

    class RecordingCompletionResource(control.ScriptedCompletionResource):
        def __init__(self):
            super().__init__(
                [
                    [
                        {
                            "name": "read_memory_file",
                            "arguments": {"path": "topics/does-not-exist.md"},
                        }
                    ],
                    [],
                ]
            )
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            return super().create(**kwargs)

    retrieval = RecordingCompletionResource()
    result = control.execute_question(
        artifact_dir=artifact,
        run_id="missing-path-run",
        method="missing-path-regression",
        condition="topic_source",
        memory_root=memory,
        turn_index=_turn_index(),
        question_id="s0_q0",
        question="What did Uma record?",
        gold_source_ids=[],
        source_recall_eligible=False,
        completion_resource=retrieval,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        formal=False,
        proxy_log=proxy_log,
    )

    tool_message = retrieval.requests[1]["messages"][-1]
    error_payload = json.loads(tool_message["content"].split("\n", 2)[-1])
    assert tool_message["role"] == "tool"
    assert error_payload == {
        "error": {
            "code": "inaccessible_memory_path",
            "message": "read path is not an accessible markdown file",
        },
        "ok": False,
    }
    assert result["status"] == "complete"
    assert result["answer"]["text"] == "insufficient evidence"
    assert result["memory"]["unchanged"] is True


def test_gold_labels_never_enter_retrieval_or_answer_prompts(tmp_path):
    memory = _memory_fixture(tmp_path)
    artifact = tmp_path / "attempt"
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = controlled_answer.FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="canary-run",
        response_text="<answer>public fact</answer>",
    )
    retrieval = control.ScriptedCompletionResource(
        [
            [
                {
                    "name": "read_memory_file",
                    "arguments": {"path": "topics/facts.md"},
                }
            ],
            [],
        ]
    )
    canary = "D99:999"
    result = control.execute_question(
        artifact_dir=artifact,
        run_id="canary-run",
        method="canary",
        condition="topic_source",
        memory_root=memory,
        turn_index=_turn_index(),
        question_id="s0_q0",
        question="What did Uma record?",
        gold_source_ids=[canary],
        source_recall_eligible=True,
        completion_resource=retrieval,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        formal=False,
        proxy_log=proxy_log,
    )

    request_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((artifact / "calls").glob("*.request.json"))
    ]
    model_visible = json.dumps(request_payloads, ensure_ascii=False)
    assert canary not in model_visible
    assert canary not in answer_client.prompts[0]
    assert result["diagnostics"]["gold_source_ids"] == [canary]
    assert result["memory"]["unchanged"] is True


def test_independent_auditors_reject_artifact_and_memory_tampering(tmp_path):
    r116_output = tmp_path / "r116"
    r116.run_synthetic(r116_output)
    ledger = (
        r116_output
        / "questions/s0_q0/attempt-0001/retrieval_model_ledger.jsonl"
    )
    ledger.write_text(ledger.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="hash|ledger|schema"):
        r116_auditor.audit_run(r116_output)

    r203_output = tmp_path / "r203"
    r203.run_synthetic(r203_output)
    memory_file = r203_output / "conditions/dual_source/memory/topics/travel.md"
    memory_file.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(r203_auditor.R203AuditError, match="memory changed"):
        r203_auditor.audit_run(r203_output)

    stage_output = tmp_path / "r203-stage"
    r203.run_synthetic(stage_output)
    stage_path = (
        stage_output
        / "conditions/dual_source/questions/s0_q0/attempt-0001/stage_evidence.json"
    )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["retrieval_reached_gold_source"] = False
    stage_path.write_text(json.dumps(stage) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="stage-evidence hash"):
        r203_auditor.audit_run(stage_output)


def test_source_spec_expansion_handles_ranges_and_compact_lists():
    expected = [
        "D1:2",
        "D1:3",
        "D1:4",
        "D1:7",
        "D2:1",
    ]
    source_text = "memory [D1:2-4,7,D2:1]"
    assert control.expand_source_specs(["[D1:2-4,7,D2:1]"]) == expected
    assert generic_auditor._expand_sources(source_text) == control.source_ids_in_text(
        source_text
    )
    assert set(generic_auditor._expand_sources(source_text)) == set(expected)
