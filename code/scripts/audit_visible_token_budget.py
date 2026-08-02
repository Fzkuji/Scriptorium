#!/usr/bin/env python3
"""Audit an R004/G0.2 visible-token trace and return a nonzero status on failure."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402


def _path_identity(path: Path) -> str:
    return unicodedata.normalize(
        "NFC",
        os.path.normcase(os.path.realpath(path)),
    ).casefold()


def _ensure_distinct(paths: dict[str, Path]) -> None:
    names = list(paths)
    for index, left_name in enumerate(names):
        left = paths[left_name]
        if left.is_symlink():
            raise ValueError(f"{left_name} must not be a symlink: {left}")
        for right_name in names[index + 1 :]:
            right = paths[right_name]
            if _path_identity(left) == _path_identity(right):
                raise ValueError(f"path collision: {left_name} and {right_name}")
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ValueError(f"inode collision: {left_name} and {right_name}")


def _atomic_write(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"audit report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python3 scripts/audit_visible_token_budget.py --trace "
            "/private/tmp/nativemem-r004-sanity-001/visible-token-trace.jsonl"
        ),
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--memory-before", type=Path)
    parser.add_argument("--memory-after", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.manifest or args.trace.with_suffix(args.trace.suffix + ".manifest.json")
    paths = {"trace": args.trace, "manifest": manifest}
    if args.report is not None:
        paths["report"] = args.report
    _ensure_distinct(paths)

    report = audit_visible_token_trace(
        args.trace,
        manifest_path=manifest,
        memory_before_path=args.memory_before,
        memory_after_path=args.memory_after,
        require_complete=True,
    )
    if args.report is not None:
        _atomic_write(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["audit_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
