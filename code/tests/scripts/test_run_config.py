"""Config-file defaults for benchmark runners."""

import json
from pathlib import Path

import pytest

from scripts.nativemem.common import run_config
from scripts.nativemem.locomo.config import parse_args


def write_config(tmp_path: Path, payload: dict, name: str = "run.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


BASE = {
    "output_dir": "out",
    "base_url": "https://provider.example/v1",
    "api_key": "builder-key",
    "judge_api_key": "judge-key",
    "input_usd_per_million": 0.27,
    "output_usd_per_million": 1.1,
}


def test_config_satisfies_required_flags(tmp_path: Path):
    config = write_config(tmp_path, BASE)

    args = parse_args(["--config", str(config)])

    assert args.api_key == "builder-key"
    assert args.judge_api_key == "judge-key"
    assert args.input_usd_per_million == 0.27
    assert args.base_url == "https://provider.example/v1"


def test_explicit_flag_overrides_config(tmp_path: Path):
    config = write_config(tmp_path, {**BASE, "model": "from-config"})

    args = parse_args(["--config", str(config), "--model", "from-flag"])

    assert args.model == "from-flag"


def test_api_key_can_be_read_from_a_file(tmp_path: Path):
    secret = tmp_path / "key.txt"
    secret.write_text("secret-from-file\n", encoding="utf-8")
    payload = {key: value for key, value in BASE.items() if key != "api_key"}
    config = write_config(tmp_path, {**payload, "api_key_file": "key.txt"})

    args = parse_args(["--config", str(config)])

    # Keeps the credential out of shell history and out of the config file.
    assert args.api_key == "secret-from-file"


def test_relative_paths_resolve_against_the_config_file(tmp_path: Path):
    nested = tmp_path / "configs"
    nested.mkdir()
    config = write_config(nested, {**BASE, "output_dir": "results/run-1"}, "a.json")

    args = parse_args(["--config", str(config)])

    assert args.output_dir == nested / "results/run-1"


def test_absolute_paths_are_left_alone(tmp_path: Path):
    target = tmp_path / "elsewhere"
    config = write_config(tmp_path, {**BASE, "output_dir": str(target)})

    args = parse_args(["--config", str(config)])

    assert args.output_dir == target


def test_unknown_option_is_rejected(tmp_path: Path):
    config = write_config(tmp_path, {**BASE, "typo_option": 1})

    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_missing_config_file_is_reported(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_args(["--config", str(tmp_path / "absent.json")])


def test_malformed_config_is_reported(tmp_path: Path):
    config = tmp_path / "bad.json"
    config.write_text("{not json", encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_setting_both_key_and_key_file_is_refused(tmp_path: Path):
    secret = tmp_path / "key.txt"
    secret.write_text("x", encoding="utf-8")
    config = write_config(tmp_path, {**BASE, "api_key_file": "key.txt"})

    with pytest.raises(run_config.ConfigError, match="not both"):
        run_config.load(config)


def test_empty_key_file_is_refused(tmp_path: Path):
    secret = tmp_path / "key.txt"
    secret.write_text("   \n", encoding="utf-8")
    payload = {key: value for key, value in BASE.items() if key != "api_key"}
    config = write_config(tmp_path, {**payload, "api_key_file": "key.txt"})

    with pytest.raises(run_config.ConfigError, match="empty"):
        run_config.load(config)


def test_runner_still_works_without_a_config(tmp_path: Path):
    args = parse_args([
        "--output-dir", str(tmp_path),
        "--base-url", "https://provider.example/v1",
        "--api-key", "k",
        "--judge-api-key", "j",
        "--input-usd-per-million", "0.27",
        "--output-usd-per-million", "1.1",
    ])

    assert args.api_key == "k"
    assert args.config is None


def test_cache_reads_are_priced_at_their_own_rate():
    """Cache reads cost a fraction of fresh input; pricing them as input
    overstated cost by orders of magnitude on cache-heavy runs."""
    from scripts.nativemem.run_locomo import summarize_usage

    records = [{
        "phase": "build",
        "calls": 1,
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "cache_read_tokens": 10_000_000,
    }]

    priced = summarize_usage(
        records,
        input_usd_per_million=0.25,
        output_usd_per_million=0.50,
        cache_read_usd_per_million=0.005,
    )
    naive = summarize_usage(
        records, input_usd_per_million=0.25, output_usd_per_million=0.50
    )

    assert priced["totals"]["estimated_cost_usd"] == 0.30
    assert naive["totals"]["estimated_cost_usd"] == 2.75
