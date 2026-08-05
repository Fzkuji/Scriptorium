"""Load experiment defaults from a JSON config file.

A benchmark run takes six required flags and around thirty in total, and the
API key has to be retyped on every invocation, which puts credentials in shell
history. A config file supplies the defaults; explicit flags still win, so
every documented command keeps working unchanged.

Credentials are named by file, never written inline, matching the convention
`scripts/configs/model_capacity.example.json` already uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONFIG_FLAG = "--config"
# Read from a file so a key never lands in shell history or a results manifest.
SECRET_FILE_KEYS = {
    "api_key": "api_key_file",
    "judge_api_key": "judge_api_key_file",
}


class ConfigError(ValueError):
    """The config file is unusable. The message names the offending key."""


def add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        CONFIG_FLAG,
        type=Path,
        metavar="PATH",
        help=(
            "JSON file supplying defaults for any option below. "
            "Explicit command-line flags override it."
        ),
    )


def load(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"config file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config file must contain a JSON object: {path}")
    return _resolve_secret_files(payload, base=path.parent)


def _resolve_secret_files(
    payload: dict[str, Any], *, base: Path
) -> dict[str, Any]:
    resolved = dict(payload)
    for target, file_key in SECRET_FILE_KEYS.items():
        location = resolved.pop(file_key, None)
        if location is None:
            continue
        if resolved.get(target):
            raise ConfigError(
                f"set either {target} or {file_key}, not both"
            )
        secret_path = Path(str(location)).expanduser()
        if not secret_path.is_absolute():
            secret_path = base / secret_path
        if not secret_path.is_file():
            raise ConfigError(f"{file_key} does not point at a file: {secret_path}")
        secret = secret_path.read_text(encoding="utf-8").strip()
        if not secret:
            raise ConfigError(f"{file_key} is empty: {secret_path}")
        resolved[target] = secret
    return resolved


def apply(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    *,
    path_keys: tuple[str, ...] = (),
) -> None:
    """Seed parser defaults from the config named in ``argv``.

    Called before ``parse_args`` so a config value satisfies ``required=True``
    and an explicit flag still overrides it.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument(CONFIG_FLAG, type=Path)
    known, _ = probe.parse_known_args(argv)
    if known.config is None:
        return

    values = load(known.config)
    known_dest = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - known_dest)
    if unknown:
        raise ConfigError(
            f"config file has options this command does not accept: {unknown}"
        )
    base = Path(known.config).expanduser().resolve().parent
    for key in path_keys:
        raw = values.get(key)
        if isinstance(raw, str) and raw:
            candidate = Path(raw).expanduser()
            # Relative paths read against the config file, so a config can be
            # moved with its inputs.
            values[key] = candidate if candidate.is_absolute() else base / candidate
    parser.set_defaults(**values)
    for action in parser._actions:
        if action.dest in values:
            action.required = False
