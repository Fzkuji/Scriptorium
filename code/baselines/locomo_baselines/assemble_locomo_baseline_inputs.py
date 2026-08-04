#!/usr/bin/env python3
"""Validate and assemble ten LoCoMo adapter outputs for unified evaluation.

The adapter contract stores one ``_build_stats`` record followed by all
question records for one LoCoMo conversation.  This script checks those files
against the pinned local dataset, combines them without changing record
content, and writes an auditable manifest.  It performs no model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.locomo_baselines import locomo_baseline_contract as contract


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
EXPECTED_QUESTIONS = 1986


class AssemblyError(RuntimeError):
    """Raised when a baseline input does not satisfy the adapter contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read valid JSON from {path}: {exc}") from exc


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def expected_gold(qa: dict[str, Any]) -> str:
    if "answer" in qa:
        return str(qa["answer"])
    if "adversarial_answer" in qa:
        return str(qa["adversarial_answer"])
    raise AssemblyError("dataset question has neither answer nor adversarial_answer")


def validate_sample(
    sample_index: int,
    rows: Any,
    dataset_item: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        contract.validate_sample_records(rows, dataset_item, sample_index)
    except contract.ContractError as exc:
        raise AssemblyError(str(exc)) from exc
    return rows[0], rows[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and assemble full LoCoMo baseline adapter outputs"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="adapter source file whose hash is recorded in the manifest",
    )
    parser.add_argument("--builder-model", required=True)
    parser.add_argument("--builder-base-url", required=True)
    parser.add_argument("--usage-tracking", required=True)
    return parser.parse_args()


def path_identity(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False)).casefold()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    adapter_path = args.adapter.expanduser().resolve()

    try:
        dataset = contract.load_dataset(dataset_path)
    except contract.ContractError as exc:
        raise AssemblyError(str(exc)) from exc
    if not adapter_path.is_file():
        raise AssemblyError(f"adapter source does not exist: {adapter_path}")

    input_paths = [input_dir / f"sample{index}_questions.json" for index in range(10)]
    protected = [dataset_path, adapter_path, *input_paths, output_path, manifest_path]
    identities = [path_identity(path) for path in protected]
    if len(set(identities)) != len(identities):
        raise AssemblyError("dataset, adapter, inputs, output, and manifest must be distinct paths")
    try:
        manifest_path.unlink()
    except FileNotFoundError:
        pass

    builds: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for sample_index, dataset_item in enumerate(dataset):
        path = input_paths[sample_index]
        raw_rows = read_json(path)
        try:
            rows = contract.normalize_sample_records(
                raw_rows,
                dataset_item,
                sample_index,
                builder_model=args.builder_model,
                builder_base_url=args.builder_base_url,
                usage_tracking=args.usage_tracking,
            )
        except contract.ContractError as exc:
            raise AssemblyError(str(exc)) from exc
        build, sample_questions = validate_sample(sample_index, rows, dataset_item)
        build_copy = dict(build)
        build_copy["sample_index"] = sample_index
        builds.append(build_copy)
        questions.extend(sample_questions)
        inputs.append({
            "sample_index": sample_index,
            "path": str(path),
            "sha256": sha256_file(path),
            "questions": len(sample_questions),
        })

    ids = [str(record["question_id"]) for record in questions]
    categories = Counter(int(record["category"]) for record in questions)
    if len(questions) != EXPECTED_QUESTIONS or len(set(ids)) != EXPECTED_QUESTIONS:
        raise AssemblyError("assembled LoCoMo inventory is not 1,986 unique questions")
    if dict(sorted(categories.items())) != EXPECTED_CATEGORIES:
        raise AssemblyError(f"unexpected category inventory: {dict(categories)}")

    payload = [*builds, *questions]
    atomic_json(output_path, payload)
    manifest = {
        "schema_version": 1,
        "benchmark": "LoCoMo",
        "method": args.method,
        "status": "inputs_complete",
        "created_at": utc_now(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "conversations": 10,
            "questions": EXPECTED_QUESTIONS,
            "primary_questions_cat1_4": 1540,
            "adversarial_questions_cat5": 446,
            "categories": {str(key): value for key, value in EXPECTED_CATEGORIES.items()},
        },
        "adapter": {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
        },
        "assembler": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract": {
            "path": str(Path(contract.__file__).resolve()),
            "sha256": sha256_file(Path(contract.__file__).resolve()),
        },
        "builder": {
            "requested_model": args.builder_model,
            "base_url": args.builder_base_url,
            "usage_tracking": args.usage_tracking,
        },
        "inputs": inputs,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "records": len(payload),
            "build_records": len(builds),
            "question_records": len(questions),
        },
        "evaluation": {
            "answerer": "pending",
            "primary_judge": "pending",
            "secondary_judge": "pending",
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        f"assembled {len(questions)} questions for {args.method}; "
        f"output={output_path}; manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
