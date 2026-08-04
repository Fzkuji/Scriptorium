from __future__ import annotations

import json
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import pytest


try:
    importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    pytest.skip(
        "formal LongMemEval shared-answer tests require the pinned M1 environment",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.longmemeval_m1 import audit_longmemeval_shared_answers as auditor  # noqa: E402
from scripts.controlled_locomo import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from scripts.longmemeval_m1 import longmemeval_m1_contract as input_contract  # noqa: E402
from scripts.longmemeval_m1 import longmemeval_shared_answer_contract as contract  # noqa: E402
from scripts.longmemeval_m1 import run_longmemeval_shared_answers as runner  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_matrix(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("longmemeval-shared") / "matrix"
    report = runner.run_synthetic_matrix(output)
    assert report["status"] == "passed"
    return output


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): contract.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_four_method_synthetic_matrix_is_no_network(
    synthetic_matrix: Path,
) -> None:
    report = auditor.audit_synthetic_matrix(synthetic_matrix)
    assert report["formal_methods"] == list(contract.FORMAL_METHODS)
    assert report["questions"] == 4
    assert report["logical_answer_calls_simulated"] == 4
    assert report["model_calls"] == 0
    assert report["network_calls"] == 0
    assert report["shared_protocol_status"] == "passed"
    assert report["frozen_context_consumed_verbatim"] is True
    assert report["retrieval_ranking_or_truncation_recomputed"] is False
    assert report["formal_executable_methods"] == list(contract.FORMAL_METHODS)
    assert set(report["backend_plan_isolation"]) == {"mem0", "graphiti"}
    assert all(
        value["status"] == "isolated"
        for value in report["backend_plan_isolation"].values()
    )
    for method in contract.FORMAL_METHODS:
        method_report = report["method_reports"][method]
        assert method_report["status"] == "passed"
        assert method_report["network_requests"] == 0
        assert method_report["simulated_client_attempts"] == 1
        binding = contract.read_json(
            next(
                (synthetic_matrix / "answers" / method / "items").glob(
                    "*/input_binding.json"
                )
            )
        )
        assert binding["frozen_input_consumption"] == {
            "retrieval_recomputed": False,
            "ranking_recomputed": False,
            "truncation_recomputed": False,
            "second_visible_token_gate_applied": False,
            "context_consumed_verbatim": True,
        }
        source = contract.read_json(
            synthetic_matrix / "answers" / method / "run_manifest.json"
        )["identity"]["source"]
        if method in contract.BACKEND_INPUT_METHODS:
            assert source["source_auditor"]["kind"] == (
                "synthetic_backend_input_auditor"
            )
            assert source["source_plan"]["scope"] == "smoke"
            assert source["source_plan"]["workspace_count"] == 1
            assert binding["source"]["backend_workspace_path"].endswith(
                f"/{method}/items/0000_e47becba/backend"
            )
            assert (
                len(binding["source"]["backend_workspace_after_retrieval_sha256"]) == 64
            )
        else:
            assert source["source_auditor"]["kind"] == "synthetic_source_auditor"
            assert source["source_plan"] is None


def test_synthetic_resume_is_no_clobber(synthetic_matrix: Path) -> None:
    before = _file_hashes(synthetic_matrix)
    report = runner.run_synthetic_matrix(synthetic_matrix)
    after = _file_hashes(synthetic_matrix)
    assert report["status"] == "passed"
    assert after == before


def test_audit_republication_validates_without_clobber(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    first = {"status": "passed", "audited_at": "first", "items": 1}
    second = {"status": "passed", "audited_at": "second", "items": 1}
    runner._publish_audit_no_clobber(path, first)
    original_hash = contract.sha256_file(path)
    runner._publish_audit_no_clobber(path, second)
    assert contract.sha256_file(path) == original_hash
    with pytest.raises(contract.SharedAnswerError, match="differs"):
        runner._publish_audit_no_clobber(
            path, {"status": "failed", "audited_at": "third", "items": 1}
        )


def test_answer_auditor_rejects_frozen_input_tamper(
    synthetic_matrix: Path,
) -> None:
    answer_path = next(
        (synthetic_matrix / "sources" / "bm25" / "items").glob("*/answer_input.json")
    )
    original = answer_path.read_bytes()
    payload = json.loads(answer_path.read_text(encoding="utf-8"))
    payload["context"]["text"] += " tampered"
    try:
        answer_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(contract.SharedAnswerError):
            auditor.audit_synthetic_matrix(synthetic_matrix)
    finally:
        answer_path.write_bytes(original)


def test_backend_artifact_tamper_is_rejected(
    synthetic_matrix: Path,
) -> None:
    retrieval_path = next(
        (synthetic_matrix / "sources" / "mem0" / "items").glob("*/retrieval_raw.json")
    )
    original = retrieval_path.read_bytes()
    payload = json.loads(original)
    payload["records"][0]["text"] = "tampered backend retrieval"
    try:
        retrieval_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(contract.SharedAnswerError):
            auditor.audit_synthetic_matrix(synthetic_matrix)
    finally:
        retrieval_path.write_bytes(original)


def test_orphan_question_start_refuses_repeat(tmp_path: Path) -> None:
    tokenizer = input_contract.FormalTokenizer.resolve()
    source_dir = tmp_path / "source"
    source = runner.create_synthetic_source(
        source_run_dir=source_dir,
        method="bm25",
        tokenizer=tokenizer,
    )
    preregistration = contract.read_json(runner.DEFAULT_PREREGISTRATION)
    protocol = contract.shared_protocol(preregistration)
    output_dir = tmp_path / "answer"
    output_dir.mkdir()
    proxy_log = output_dir / "proxy.jsonl"
    proxy_log.write_text("", encoding="utf-8")
    identity = runner._manifest_identity(
        mode="synthetic_no_network",
        method="bm25",
        output_dir=output_dir,
        source_descriptor=source["descriptor"],
        protocol=protocol,
        proxy_log=proxy_log,
    )
    manifest = runner._create_or_validate_manifest(
        output_dir=output_dir,
        identity=identity,
        expected_items=1,
    )
    item = source["items"][0]
    item_dir = (
        output_dir
        / "items"
        / contract.item_directory_name(item["dataset_index"], item["question_id"])
    )
    item_dir.mkdir(parents=True)
    logical_call_id = f"{manifest['run_id']}:bm25:0000:{item['question_id']}"
    with answer_contract.DurableLedger(
        item_dir / "answer_ledger.jsonl", run_id=logical_call_id
    ) as ledger:
        ledger.append("question_started", {"synthetic_crash": True})

    class NeverClient:
        called = False

        def complete(self, **_: Any) -> Any:
            self.called = True
            raise AssertionError("orphan resume must not call the answer client")

    client = NeverClient()
    with pytest.raises(contract.SharedAnswerError, match="refusing repeat"):
        runner.answer_one_frozen_item(
            run_id=manifest["run_id"],
            mode="synthetic_no_network",
            method="bm25",
            output_dir=output_dir,
            source_run_dir=source_dir,
            source_descriptor=source["descriptor"],
            source_item_record=item,
            tokenizer=tokenizer,
            client=client,
            proxy_log=proxy_log,
        )
    assert client.called is False


@pytest.mark.parametrize("method", ["bm25", "mem0", "graphiti"])
def test_hard_cap_input_above_20k_is_rejected(method: str) -> None:
    tokenizer = input_contract.FormalTokenizer.resolve()
    payload = runner._synthetic_source_input(method=method, tokenizer=tokenizer)
    payload["context"]["visible_tokens"] = 20_001
    payload["context"]["budget"]["cumulative_visible_tokens"] = 20_001
    with pytest.raises(contract.SharedAnswerError, match="20K budget"):
        contract.validate_frozen_answer_input(
            payload,
            method=method,
            tokenizer=tokenizer,
        )


def test_full_context_truncation_is_rejected() -> None:
    tokenizer = input_contract.FormalTokenizer.resolve()
    payload = runner._synthetic_source_input(method="full_context", tokenizer=tokenizer)
    payload["context"]["budget"]["truncation_allowed"] = True
    with pytest.raises(contract.SharedAnswerError, match="capped or truncated"):
        contract.validate_frozen_answer_input(
            payload,
            method="full_context",
            tokenizer=tokenizer,
        )


def test_formal_execution_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(contract.SharedAnswerError, match="explicit"):
        runner.run_formal_answers(
            method="full_context",
            source_run_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
            dataset_path=runner.DEFAULT_DATASET,
            preregistration_path=runner.DEFAULT_PREREGISTRATION,
            gateway_root=tmp_path / "gateway",
            retries=1,
            allow_model_requests=False,
        )


def test_preflight_keeps_missing_backend_inputs_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "full_context").mkdir(parents=True)
    (root / "bm25").mkdir()

    def fake_source(**kwargs: Any) -> dict[str, Any]:
        method = kwargs["method"]
        return {
            "descriptor": {"method": method, "synthetic_test": True},
            "manifest": {},
            "items": [],
            "live_audit": {"status": "passed"},
            "recorded_audit": {"status": "passed"},
        }

    monkeypatch.setattr(runner, "audit_formal_source", fake_source)
    report = runner.preflight_matrix(
        root=root,
        dataset_path=runner.DEFAULT_DATASET,
        preregistration_path=runner.DEFAULT_PREREGISTRATION,
    )
    assert report["status"] == "blocked"
    assert report["ready_methods"] == ["full_context", "bm25"]
    assert report["blocked_methods"] == ["mem0", "graphiti"]
    assert report["methods"]["mem0"]["status"] == "blocked_missing_frozen_input"
    assert report["methods"]["graphiti"]["status"] == "blocked_missing_frozen_input"
    assert report["model_calls"] == 0
    assert report["network_calls"] == 0


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (runner, "audit_formal_source"),
        (auditor, "_audit_formal_source"),
    ],
)
@pytest.mark.parametrize("method", contract.FORMAL_METHODS)
def test_four_method_formal_source_uses_exact_auditor(
    module: Any,
    entrypoint: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def base(**kwargs: Any) -> dict[str, Any]:
        calls.append(f"base:{kwargs['method']}")
        return {"route": "base"}

    def backend(**kwargs: Any) -> dict[str, Any]:
        calls.append(f"backend:{kwargs['method']}:{kwargs['formal']}")
        return {"route": "backend"}

    monkeypatch.setattr(module, "_audit_base_formal_source", base)
    monkeypatch.setattr(module, "_audit_backend_source", backend)
    result = getattr(module, entrypoint)(
        method=method,
        source_run_dir=Path("unused-source"),
        dataset_path=Path("unused-dataset"),
        preregistration_path=Path("unused-preregistration"),
    )
    if method in contract.BASE_INPUT_METHODS:
        assert result == {"route": "base"}
        assert calls == [f"base:{method}"]
    else:
        assert result == {"route": "backend"}
        assert calls == [f"backend:{method}:True"]


def test_preflight_routes_all_four_formal_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    for method in contract.FORMAL_METHODS:
        (root / method).mkdir(parents=True)

    def fake_source(**kwargs: Any) -> dict[str, Any]:
        method = kwargs["method"]
        return {"descriptor": {"method": method}, "items": [], "manifest": {}}

    monkeypatch.setattr(runner, "audit_formal_source", fake_source)
    report = runner.preflight_matrix(
        root=root,
        dataset_path=runner.DEFAULT_DATASET,
        preregistration_path=runner.DEFAULT_PREREGISTRATION,
    )
    assert report["status"] == "ready"
    assert report["formal_executable_methods"] == list(contract.FORMAL_METHODS)
    assert report["ready_methods"] == list(contract.FORMAL_METHODS)
    assert report["blocked_methods"] == []
    assert report["methods"]["mem0"]["expected_source_auditor"]["kind"] == (
        "backend_input_auditor"
    )
    assert report["methods"]["graphiti"]["expected_source_auditor"]["kind"] == (
        "backend_input_auditor"
    )


def test_legacy_smoke_plans_fail_closed_on_workspace_collision() -> None:
    for method in contract.BACKEND_INPUT_METHODS:
        report = runner.inspect_backend_plan_inventory(
            root=runner.DEFAULT_ROOT,
            method=method,
            dataset_path=runner.DEFAULT_DATASET,
            preregistration_path=runner.DEFAULT_PREREGISTRATION,
        )
        assert report["formal_plan_status"] == "valid"
        assert report["legacy_smoke_plan_status"] == ("rejected_workspace_overlap")
        assert report["workspace_isolation"]["status"] == ("rejected_workspace_overlap")
