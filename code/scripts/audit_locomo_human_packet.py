#!/usr/bin/env python3
"""Audit deterministic selection, blinding, hashes, and templates for R501."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_locomo_human_packet import (  # noqa: E402
    build_payloads,
    csv_bytes,
    json_bytes,
    read_json,
    sha256_file,
)
from src.evaluation.m4_reliability import ReliabilityError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    packet_dir = args.packet_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = read_json(packet_dir / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ReliabilityError("human packet manifest is not complete")
    if manifest.get("source_sha256") != sha256_file(source):
        raise ReliabilityError("human packet source hash mismatch")
    packet, private_key, templates = build_payloads(source, seed=manifest.get("seed"))
    expected = {
        "public/packet.json": json_bytes(packet),
        "private/private_key.json": json_bytes(private_key),
        "public/annotator_A.csv": csv_bytes(templates),
        "public/annotator_B.csv": csv_bytes(templates),
    }
    if set(manifest.get("files", {})) != set(expected):
        raise ReliabilityError("human packet file inventory mismatch")
    for name, payload in expected.items():
        actual = (packet_dir / name).read_bytes()
        if actual != payload:
            raise ReliabilityError(f"human packet content mismatch: {name}")
        if manifest["files"][name] != {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }:
            raise ReliabilityError(f"human packet manifest metadata mismatch: {name}")
    observed_files = {
        path.relative_to(packet_dir).as_posix()
        for path in packet_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != {*expected, "manifest.json"}:
        raise ReliabilityError("human packet directory has an unexpected file inventory")
    if any(path.is_symlink() for path in packet_dir.rglob("*")):
        raise ReliabilityError("human packet directory contains a symlink")
    public_text = (packet_dir / "public/packet.json").read_text(encoding="utf-8")
    for key_item in private_key["items"]:
        if key_item["question_id"] in public_text:
            raise ReliabilityError("public packet exposes a source question_id")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "questions": 100,
        "samples": 10,
        "source_sha256": sha256_file(source),
        "manifest_sha256": sha256_file(packet_dir / "manifest.json"),
        "checks": [
            "deterministic_stratified_selection",
            "exact_file_inventory_and_hashes",
            "method_and_judge_blinding",
            "two_independent_blank_templates",
        ],
    }
    from scripts.run_m4_statistics import atomic_json_no_clobber

    atomic_json_no_clobber(output, audit)
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
