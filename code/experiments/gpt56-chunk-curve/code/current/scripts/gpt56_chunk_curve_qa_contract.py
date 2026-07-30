"""Read-only bindings for QA over completed GPT-5.6 chunk-curve builds."""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from scripts import run_gpt56_chunk_curve as build_runner
from scripts import controlled_locomo_answer_contract as answer_contract
from scripts import readonly_nativemem_control as readonly_control
from scripts import r115_beam_control_contract as beam_answer_contract


LOCKED_LOCOMO_EVALUATOR_SHA256 = build_runner.LOCKED_LOCOMO_EVALUATOR_SHA256
PROTOCOL_ID = "gpt56-w32-screening-qa-v1"
LEGACY_ANSWER_MAX_TOKENS = 4_096
ANSWER_MAX_TOKENS = 10_000
ACCEPTED_ANSWER_MAX_TOKENS = (
    LEGACY_ANSWER_MAX_TOKENS,
    ANSWER_MAX_TOKENS,
)
ANSWER_RETRIES = 3
QUESTION_MAX_ATTEMPTS = 8
EMPTY_OUTPUT_RETRY_REASON = "upstream_completed_empty_output"
TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON = (
    "upstream_transient_provider_server_error"
)
RETRYABLE_QUESTION_ERRORS = [
    EMPTY_OUTPUT_RETRY_REASON,
    TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON,
]
SHORT_ANSWER_PROMPT_KIND = readonly_control.DEFAULT_ANSWER_PROMPT_KIND
SHORT_ANSWER_PROMPT_SHA256 = readonly_control.DEFAULT_ANSWER_PROMPT_SHA256
BEAM_ANSWER_PROMPT_KIND = "beam-r115-answer-v1"
BEAM_ANSWER_PROMPT_SHA256 = answer_contract.sha256_bytes(
    beam_answer_contract.ANSWER_PROMPT.encode("utf-8")
)
ANSWER_PROMPTS = {
    "locomo": {
        "kind": SHORT_ANSWER_PROMPT_KIND,
        "template_sha256": SHORT_ANSWER_PROMPT_SHA256,
    },
    "longmemeval-s": {
        "kind": SHORT_ANSWER_PROMPT_KIND,
        "template_sha256": SHORT_ANSWER_PROMPT_SHA256,
    },
    "beam-100k": {
        "kind": BEAM_ANSWER_PROMPT_KIND,
        "template_sha256": BEAM_ANSWER_PROMPT_SHA256,
    },
}
TOKENIZER_IDENTITY = {
    "implementation": "tiktoken",
    "implementation_version": answer_contract.EXPECTED_TIKTOKEN_VERSION,
    "encoding_name": answer_contract.EXPECTED_ENCODING,
    "requested_model": answer_contract.EXPECTED_MODEL,
    "resolution": "tiktoken_named_fallback",
    "fallback_encoding": answer_contract.EXPECTED_ENCODING,
    "fallback_reason": "requested_model_not_in_tiktoken_mapping",
    "provider_exact": False,
    "counting_note": (
        "Local tiktoken count used for an enforceable experiment budget; "
        "it is not a provider-reported exact count."
    ),
}


class QAContractError(RuntimeError):
    """A requested QA input does not satisfy the frozen build contract."""


def assemble_beam_answer_prompt(*, question: str, deliveries: Sequence[Any]) -> str:
    return beam_answer_contract.ANSWER_PROMPT.format(
        memories=answer_contract.delivered_memory_block(deliveries),
        question=question,
    )


@dataclass(frozen=True)
class RunBinding:
    run_id: str
    row: dict[str, Any]
    run_dir: Path


def verify_locomo_evaluator(path: Path = build_runner.LOCOMO_EVALUATOR) -> str:
    """Enforce the repository-level LoCoMo evaluator lock."""

    actual = build_runner.sha256_file(path)
    if actual != LOCKED_LOCOMO_EVALUATOR_SHA256:
        raise QAContractError(
            "LoCoMo evaluator does not match the locked SHA-256"
        )
    return actual


def _read_matrix(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [row.get("run_id") for row in rows]
    if None in run_ids or len(run_ids) != len(set(run_ids)):
        raise QAContractError("matrix contains missing or duplicate run IDs")
    return rows


def select_completed_runs(
    *,
    matrix_path: Path,
    build_root: Path,
    run_ids: Sequence[str],
) -> list[RunBinding]:
    """Bind an explicit ordered run list to strictly completed build paths."""

    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise QAContractError("run IDs must be an explicit non-empty unique list")
    by_id = {str(row["run_id"]): row for row in _read_matrix(matrix_path)}
    bindings: list[RunBinding] = []
    for run_id in run_ids:
        row = by_id.get(run_id)
        if row is None:
            raise QAContractError(f"unknown run ID: {run_id}")
        if str(row.get("write_turns")) != "32":
            raise QAContractError(f"writer window must be 32: {run_id}")
        if row.get("reasoning_effort") != "none":
            raise QAContractError(f"reasoning_effort must be none: {run_id}")
        path = build_runner.run_dir(build_root, row)
        if not build_runner.valid_complete(path, row):
            raise QAContractError(f"run is not complete: {run_id}")
        bindings.append(RunBinding(run_id=run_id, row=row, run_dir=path))
    return bindings


def source_token_count_for_method(
    conversation: Mapping[str, Any],
    method: str,
) -> int:
    """Recompute source tokens with the method recorded by the build."""

    texts: list[str] = []
    session_number = 1
    while f"session_{session_number}" in conversation:
        turns = conversation[f"session_{session_number}"]
        if not isinstance(turns, list):
            raise QAContractError("conversation session is not a list")
        texts.extend(
            str(turn.get("text", ""))
            for turn in turns
            if isinstance(turn, Mapping)
        )
        session_number += 1
    joined = "\n".join(texts)
    if method == "chars_div_4_fallback":
        return max(1, len(joined) // 4)
    if method == "o200k_base":
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(joined))
    raise QAContractError(f"unsupported source tokenizer: {method}")


def load_arrow_row(path: Path, index: int) -> dict[str, Any]:
    """Read one Arrow IPC row without importing the datasets feature schema."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise QAContractError("Arrow row index must be a non-negative integer")
    import pyarrow as pa

    with path.open("rb") as handle:
        table = pa.ipc.open_stream(handle).read_all()
    if index >= table.num_rows:
        raise QAContractError(
            f"row {index} is outside Arrow table with {table.num_rows} rows"
        )
    rows = table.slice(index, 1).to_pylist()
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise QAContractError("Arrow row did not decode to one object")
    return rows[0]


def validate_unit_binding(
    binding: RunBinding,
    *,
    locomo_data: Path = build_runner.LOCOMO_DATA,
    longmemeval_data: Path = build_runner.LONGMEMEVAL_DATA,
    beam_arrow: Path = build_runner.BEAM_ARROW,
) -> dict[str, Any]:
    """Rebuild and validate the immutable source payload for one run."""

    unit = build_runner.read_json(binding.run_dir / "unit.json")
    row = binding.row
    for key in ("run_id", "benchmark", "unit_id", "selection"):
        if unit.get(key) != row.get(key):
            raise QAContractError(f"unit binding differs at {key}")
    selection = row["selection"]
    if row["benchmark"] == "locomo":
        raw_dataset_path = locomo_data.expanduser().resolve()
        dataset = build_runner.read_json(locomo_data)
        index = selection["sample_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(dataset)
        ):
            raise QAContractError("LoCoMo sample index is invalid")
        item = dataset[index]
        if item.get("sample_id") != selection.get("sample_id"):
            raise QAContractError("LoCoMo unit identity changed")
        conversation = item["conversation"]
        questions = [
            {
                "question_id": f"q{question_index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
            }
            for question_index, qa in enumerate(item["qa"])
            if qa.get("category") in (1, 2, 3, 4)
        ]
        metadata = {"sample_id": item["sample_id"]}
    elif row["benchmark"] == "longmemeval-s":
        from scripts.run_v88_gpt55_longmemeval import longmemeval_to_locomo

        raw_dataset_path = longmemeval_data.expanduser().resolve()
        dataset = build_runner.read_json(longmemeval_data)
        index = selection["dataset_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(dataset)
        ):
            raise QAContractError("LongMemEval dataset index is invalid")
        item = dataset[index]
        if item.get("question_id") != selection.get("question_id"):
            raise QAContractError("LongMemEval unit identity changed")
        conversation = longmemeval_to_locomo(item, index)
        questions = [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "question_date": item["question_date"],
                "gold": item["answer"],
                "question_type": item["question_type"],
                "answer_session_ids": item["answer_session_ids"],
            }
        ]
        metadata = {"dataset_index": index}
    elif row["benchmark"] == "beam-100k":
        from scripts.run_v88_gpt55_beam import conversation_to_native, extract_questions

        raw_dataset_path = beam_arrow.expanduser().resolve()
        index = selection["conversation_index"]
        item = load_arrow_row(beam_arrow, index)
        if str(item.get("conversation_id")) != str(selection.get("conversation_id")):
            raise QAContractError("BEAM unit identity changed")
        conversation, source_metadata = conversation_to_native(item)
        questions = extract_questions(item)
        for question_index, question in enumerate(questions):
            question["question_id"] = (
                f"{item['conversation_id']}-q{question_index}"
            )
        metadata = {
            "conversation_id": str(item["conversation_id"]),
            "source_id_count": len(source_metadata["source_id_map"]),
        }
    else:
        raise QAContractError(f"unsupported benchmark: {row['benchmark']}")
    if unit.get("metadata") != metadata:
        raise QAContractError("unit metadata differs")
    if unit.get("questions") != questions:
        raise QAContractError("unit questions differ")
    if unit.get("messages") != build_runner.source_message_count(conversation):
        raise QAContractError("source message count differs")
    tokenizer = unit.get("source_tokenizer")
    if not isinstance(tokenizer, str):
        raise QAContractError("unit source tokenizer is invalid")
    source_tokens = source_token_count_for_method(conversation, tokenizer)
    if unit.get("source_tokens") != source_tokens:
        raise QAContractError("source token count differs")
    return {
        "conversation": conversation,
        "questions": questions,
        "metadata": metadata,
        "unit": unit,
        "raw_dataset": {
            "path": str(raw_dataset_path),
            "sha256": build_runner.sha256_file(raw_dataset_path),
        },
    }


def build_preregistration(
    *,
    bindings: Sequence[RunBinding],
    validated_units: Sequence[Mapping[str, Any]],
    matrix_path: Path,
    upstream: str | None = None,
    upstream_code_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, request-free QA preregistration payload."""

    if not bindings or len(bindings) != len(validated_units):
        raise QAContractError("bindings and validated units must align")
    normalized_upstream: str | None = None
    if upstream is not None:
        parsed = urlsplit(upstream)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise QAContractError("subscription upstream origin is invalid")
        host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
        normalized_upstream = f"http://{host}:{parsed.port}"
    if upstream_code_sha256 is not None and (
        normalized_upstream is None
        or re.fullmatch(r"[0-9a-f]{64}", upstream_code_sha256) is None
    ):
        raise QAContractError("subscription upstream code SHA-256 is invalid")
    source_paths = {
        "matrix": matrix_path,
        "build_runner": build_runner.ROOT / "scripts/run_gpt56_chunk_curve.py",
        "qa_contract": Path(__file__).resolve(),
        "qa_runner": build_runner.ROOT / "scripts/run_gpt56_chunk_curve_qa.py",
        "qa_auditor": (
            build_runner.ROOT / "scripts/audit_gpt56_chunk_curve_qa.py"
        ),
        "controlled_qa_proxy": (
            build_runner.ROOT / "scripts/controlled_subscription_qa_proxy.py"
        ),
        "controlled_answer_contract": Path(answer_contract.__file__).resolve(),
        "evaluation_prompts": build_runner.ROOT / "src/evaluation/prompts.py",
        "controlled_answer_runner": (
            build_runner.ROOT / "scripts/run_controlled_locomo_answers.py"
        ),
        "gpt55_run_proxy": build_runner.ROOT / "scripts/gpt55_run_proxy.py",
        "readonly_control": Path(readonly_control.__file__).resolve(),
        "readonly_auditor": (
            build_runner.ROOT / "scripts/audit_readonly_nativemem_control.py"
        ),
        "visible_token_budget": (
            build_runner.ROOT / "src/evaluation/visible_token_budget.py"
        ),
        "visible_token_audit": (
            build_runner.ROOT / "src/evaluation/visible_token_audit.py"
        ),
        "durable_model_ledger": (
            build_runner.ROOT / "src/evaluation/durable_model_ledger.py"
        ),
        "locomo_evaluator": build_runner.LOCOMO_EVALUATOR,
        "locomo_evidence_launcher": (
            build_runner.ROOT
            / "scripts/run_locked_locomo_eval_with_evidence.py"
        ),
        "longmemeval_evaluator": (
            build_runner.ROOT
            / "benchmarks/longmemeval/src/evaluation/evaluate_qa.py"
        ),
        "longmemeval_answer_runner": (
            build_runner.ROOT / "scripts/run_v88_gpt55_longmemeval.py"
        ),
        "beam_converter": (
            build_runner.ROOT / "scripts/run_v88_gpt55_beam.py"
        ),
        "beam_screening_evaluator": (
            build_runner.ROOT / "scripts/eval_beam_screening_subset.py"
        ),
        "beam_screening_semantic_judge": (
            build_runner.ROOT / "scripts/judge_beam_screening_subset.py"
        ),
        "beam_answer_contract": (
            build_runner.ROOT / "scripts/r115_beam_control_contract.py"
        ),
    }
    verify_locomo_evaluator(source_paths["locomo_evaluator"])
    source_hashes = {
        name: build_runner.sha256_file(path)
        for name, path in source_paths.items()
    }
    inputs: list[dict[str, Any]] = []
    question_count = 0
    for binding, validated in zip(bindings, validated_units, strict=True):
        unit = validated.get("unit")
        conversation = validated.get("conversation")
        questions = validated.get("questions")
        raw_dataset = validated.get("raw_dataset")
        if not isinstance(unit, Mapping) or unit.get("run_id") != binding.run_id:
            raise QAContractError("validated unit is bound to a different run")
        if not isinstance(conversation, Mapping):
            raise QAContractError("validated conversation is invalid")
        if not isinstance(questions, list):
            raise QAContractError("validated questions are invalid")
        if not isinstance(raw_dataset, Mapping):
            raise QAContractError("validated raw dataset binding is invalid")
        raw_dataset_path = Path(str(raw_dataset.get("path", ""))).resolve()
        if (
            not raw_dataset_path.is_file()
            or raw_dataset.get("sha256")
            != build_runner.sha256_file(raw_dataset_path)
        ):
            raise QAContractError("validated raw dataset binding differs")
        turn_index = readonly_control.build_turn_index(conversation)
        question_count += len(questions)
        paths = {
            "build": binding.run_dir / "build.json",
            "unit": binding.run_dir / "unit.json",
            "success": binding.run_dir / "memory" / "_SUCCESS.json",
        }
        inputs.append(
            {
                "run_id": binding.run_id,
                "benchmark": binding.row["benchmark"],
                "tier": binding.row["tier"],
                "writer_model": binding.row["model"],
                "write_turns": str(binding.row["write_turns"]),
                "run_dir": str(binding.run_dir.resolve()),
                "question_count": len(questions),
                "build_sha256": build_runner.sha256_file(paths["build"]),
                "unit_sha256": build_runner.sha256_file(paths["unit"]),
                "success_sha256": build_runner.sha256_file(paths["success"]),
                "canonical_conversation_sha256": (
                    answer_contract.canonical_hash(conversation)
                ),
                "turn_index_sha256": answer_contract.canonical_hash(turn_index),
                "normalized_questions_sha256": (
                    answer_contract.canonical_hash(questions)
                ),
                "raw_dataset": {
                    "path": str(raw_dataset_path),
                    "sha256": raw_dataset["sha256"],
                },
                "memory_before": readonly_control.memory_descriptor(
                    binding.run_dir / "memory"
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "evidence_class": "screening_build",
        "qa": {
            "answer_model": readonly_control.EXPECTED_MODEL,
            "visible_token_limit": 20_000,
            "max_retrieval_rounds": 12,
            "condition": "dual_source",
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "accepted_answer_max_tokens": list(ACCEPTED_ANSWER_MAX_TOKENS),
            "completion_cap_enforcement": "posthoc_usage_validation",
            "upstream_ignored_client_parameters": ["max_output_tokens"],
            "answer_retries": ANSWER_RETRIES,
            "question_max_attempts": QUESTION_MAX_ATTEMPTS,
            "retryable_question_errors": RETRYABLE_QUESTION_ERRORS,
            "answer_prompts": ANSWER_PROMPTS,
            "tokenizer": TOKENIZER_IDENTITY,
        },
        "run_ids": [binding.run_id for binding in bindings],
        "question_count": question_count,
        "inputs": inputs,
        "subscription_upstream": {
            "origin": normalized_upstream,
            "code_sha256": upstream_code_sha256,
        },
        "execution_runtime": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "packages": {
                name: importlib_metadata.version(name)
                for name in ("tiktoken", "openai", "httpx")
            },
        },
        "source_hashes": source_hashes,
    }
