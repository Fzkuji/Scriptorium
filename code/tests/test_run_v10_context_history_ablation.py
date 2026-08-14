import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts import run_v10_context_history_ablation as runner


ROOT = Path(__file__).resolve().parents[1]


def test_context_matrix_has_requested_representations_and_budgets():
    assert list(runner.VARIANTS) == [
        "none",
        "raw20",
        "raw50",
        "events20",
        "events50",
        "summary100",
        "summary250",
    ]
    assert runner.VARIANTS["raw20"].budget_unit == "raw_turns"
    assert runner.VARIANTS["raw20"].display_name == "前序原始消息（最近 20 条）"
    assert runner.VARIANTS["none"].display_name == "无前序历史（仅看当前 6 条消息）"
    assert runner.VARIANTS["events50"].budget == 50
    assert runner.VARIANTS["summary250"].summary_max_words == 250


def test_frozen_env_changes_only_context_fields():
    fields = {
        "NATIVEMEM_V10_CONTEXT_MODE",
        "NATIVEMEM_V10_CONTEXT_ITEMS",
        "NATIVEMEM_V10_SUMMARY_MAX_WORDS",
    }
    baseline = runner.frozen_env(runner.VARIANTS["none"])
    for variant in runner.VARIANTS.values():
        candidate = runner.frozen_env(variant)
        assert {key: value for key, value in candidate.items() if key not in fields} == {
            key: value for key, value in baseline.items() if key not in fields
        }
    assert baseline["NATIVEMEM_V10_WRITE_TURNS"] == "6"
    assert baseline["NATIVEMEM_V8_VERIFY"] == "on"
    assert baseline["NATIVEMEM_V8_CONCURRENCY"] == "1"


def test_default_dry_run_writes_14_run_manifest(tmp_path):
    results_dir = tmp_path / "dry"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v10_context_history_ablation.py"),
            "--results-dir",
            str(results_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((results_dir / "experiment_manifest.json").read_text())
    status = json.loads((results_dir / "status.json").read_text())
    assert len(manifest["expected_runs"]) == 14
    assert manifest["samples"] == [1]
    assert manifest["max_sessions"] == 1
    assert manifest["source_hashes"]["evaluator"] == runner.EVALUATOR_SHA256
    assert status["phase"] == "dry_run"


def test_deepseek_flash_is_exact_openrouter_model():
    args = argparse.Namespace(
        gpt_model="gpt-5.5",
        gpt_base="http://127.0.0.1:8199/v1",
        minimax_model="minimax/minimax-m2.7",
        minimax_base="https://openrouter.ai/api/v1",
        minimax_key_env="OPENROUTER_API_KEY",
        minimax_trust_proxy=True,
        deepseek_model="deepseek/deepseek-v4-flash",
        deepseek_base="https://openrouter.ai/api/v1",
        deepseek_key_env="OPENROUTER_API_KEY",
        deepseek_trust_proxy=True,
    )
    spec = runner.model_specs(args)["deepseek_flash"]
    assert spec.model == "deepseek/deepseek-v4-flash"
    assert spec.reasoning == "none"
