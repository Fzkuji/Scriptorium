#!/usr/bin/env python3
"""Generate formal LoCoMo answers with the shared GPT-5.5 answerer and R004 gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.locomo_baselines import audit_gpt55_locomo_baselines as baseline_auditor  # noqa: E402
from scripts.controlled_locomo import controlled_locomo_answer_contract as contract  # noqa: E402
from scripts.gateways import openai_gpt55_flex_gateway as flex_gateway  # noqa: E402
from scripts.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402
from scripts.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402
from scripts.evaluation.visible_token_budget import (  # noqa: E402
    TokenCounter,
    VisibleTokenBudgetGate,
    snapshot_memory_bytes,
)


RUN_PROXY = ROOT / "scripts/controlled_locomo/controlled_gpt55_run_proxy.py"
BASE_RUN_PROXY = ROOT / "scripts/controlled_locomo/gpt55_run_proxy.py"
UPSTREAM_PROXY = ROOT / "scripts/gateways/openai_gpt55_flex_gateway.py"
FLEX_EVIDENCE = ROOT / "scripts/gateways/openai_gpt55_flex_gateway_evidence.py"
AUDITOR = ROOT / "scripts/controlled_locomo/audit_controlled_locomo_answers.py"
DEFAULT_PYTHON = Path("/opt/miniconda3/bin/python3")


@dataclass(frozen=True)
class AnswerCallResult:
    raw_output: str
    response_id: str
    response_model: str
    usage: dict[str, Any]
    logical_calls: int
    client_http_attempts: int
    upstream_http_attempts: int
    proxy_event_ids: tuple[str, ...]
    request_sha256s: tuple[str, ...]
    response_sha256s: tuple[str, ...]
    unsupported_parameters: tuple[str, ...]


class RetryableEmptyOutputError(contract.ControlledAnswerError):
    """A 500 response that must restart the whole question attempt."""

    status_code = 500


class AnswerClient(Protocol):
    def complete(
        self,
        *,
        prompt: str,
        question_id: str,
        logical_call_id: str,
        ledger: contract.DurableLedger,
    ) -> AnswerCallResult: ...


def _nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise contract.ControlledAnswerError(f"invalid {name}: {value!r}")
    return value


def _parse_response(
    *,
    body: bytes,
    request_sha256: str,
    question_id: str,
    logical_call_id: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contract.ControlledAnswerError(f"answer response is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise contract.ControlledAnswerError("answer response is not an object")
    if payload.get("model") != contract.EXPECTED_MODEL:
        raise contract.ControlledAnswerError(
            f"answer response model differs: {payload.get('model')!r}"
        )
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise contract.ControlledAnswerError("answer response ID is missing")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise contract.ControlledAnswerError("answer response must have one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise contract.ControlledAnswerError("answer response did not finish with stop")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise contract.ControlledAnswerError("answer response content is empty")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise contract.ControlledAnswerError("answer usage is missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _nonnegative_int(usage.get(key), name=f"usage.{key}")
    proxy_meta = payload.get("proxy_meta")
    if not isinstance(proxy_meta, dict):
        raise contract.ControlledAnswerError("inner proxy metadata is missing")
    upstream_attempts = _nonnegative_int(
        proxy_meta.get("attempts"), name="proxy_meta.attempts"
    )
    if upstream_attempts < 1:
        raise contract.ControlledAnswerError("inner proxy reports no upstream request")
    unsupported = proxy_meta.get("unsupported_parameters", [])
    if not isinstance(unsupported, list) or not all(
        isinstance(value, str) for value in unsupported
    ):
        raise contract.ControlledAnswerError("unsupported-parameter metadata is invalid")
    exclusive = payload.get("exclusive_proxy_meta")
    if not isinstance(exclusive, dict):
        raise contract.ControlledAnswerError("exclusive proxy linkage is missing")
    expected_linkage = {
        "question_id": question_id,
        "logical_call_id": logical_call_id,
        "request_sha256": request_sha256,
        "client_http_attempts": 1,
        "upstream_http_attempts": upstream_attempts,
        "unsupported_parameters": unsupported,
    }
    mismatches = {
        key: {"expected": value, "actual": exclusive.get(key)}
        for key, value in expected_linkage.items()
        if exclusive.get(key) != value
    }
    if mismatches:
        raise contract.ControlledAnswerError(
            f"exclusive proxy linkage differs: {mismatches}"
        )
    event_id = exclusive.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise contract.ControlledAnswerError("exclusive proxy event ID is missing")
    return {
        "raw_output": content,
        "response_id": response_id,
        "response_model": payload["model"],
        "usage": usage,
        "upstream_http_attempts": upstream_attempts,
        "unsupported_parameters": unsupported,
        "proxy_event_id": event_id,
    }


def _parse_error_proxy_metadata(
    *,
    body: bytes,
    request_sha256: str,
    question_id: str,
    logical_call_id: str,
) -> dict[str, Any]:
    """Validate the exclusive linkage needed to account for a failed request."""

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contract.ControlledAnswerError(
            "answer error response is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise contract.ControlledAnswerError("answer error response is not an object")
    exclusive = payload.get("exclusive_proxy_meta")
    if not isinstance(exclusive, dict):
        raise contract.ControlledAnswerError(
            "answer error response lacks exclusive proxy linkage"
        )
    raw_upstream_attempts = exclusive.get("upstream_http_attempts")
    upstream_attempts = (
        None
        if raw_upstream_attempts is None
        else _nonnegative_int(
            raw_upstream_attempts,
            name="exclusive_proxy_meta.upstream_http_attempts",
        )
    )
    unsupported = exclusive.get("unsupported_parameters", [])
    if not isinstance(unsupported, list) or not all(
        isinstance(value, str) for value in unsupported
    ):
        raise contract.ControlledAnswerError(
            "answer error response has invalid unsupported parameters"
        )
    expected_linkage = {
        "question_id": question_id,
        "logical_call_id": logical_call_id,
        "request_sha256": request_sha256,
        "client_http_attempts": 1,
    }
    if any(exclusive.get(key) != value for key, value in expected_linkage.items()):
        raise contract.ControlledAnswerError(
            "answer error response exclusive proxy linkage differs"
        )
    event_id = exclusive.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise contract.ControlledAnswerError(
            "answer error response exclusive event ID is missing"
        )
    error = payload.get("error")
    if not isinstance(error, str) or not error:
        raise contract.ControlledAnswerError(
            "answer error response error text is missing"
        )
    return {
        "proxy_event_id": event_id,
        "upstream_http_attempts": upstream_attempts,
        "unsupported_parameters": unsupported,
        "error": error,
    }


class HttpAnswerClient:
    """Direct client whose retries and physical HTTP requests are explicit."""

    def __init__(
        self,
        *,
        base_url: str,
        retries: int,
        answer_max_tokens: int,
        timeout_seconds: int = 360,
    ):
        if retries < 1:
            raise ValueError("retries must be positive")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.retries = retries
        self.answer_max_tokens = answer_max_tokens
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(
        self,
        *,
        prompt: str,
        question_id: str,
        logical_call_id: str,
        ledger: contract.DurableLedger,
    ) -> AnswerCallResult:
        request_payload = {
            "model": contract.EXPECTED_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.answer_max_tokens,
            "temperature": 0,
        }
        body = json.dumps(
            request_payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        request_sha256 = contract.sha256_bytes(body)
        request_hashes: list[str] = []
        response_hashes: list[str] = []
        event_ids: list[str] = []
        upstream_attempts = 0
        unsupported: set[str] = set()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request_hashes.append(request_sha256)
            ledger.append(
                "physical_http_attempt_started",
                {
                    "attempt": attempt,
                    "question_id": question_id,
                    "logical_call_id": logical_call_id,
                    "request_sha256": request_sha256,
                },
            )
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer x",
                    "X-Controlled-Question-ID": question_id,
                    "X-Controlled-Logical-Call-ID": logical_call_id,
                },
                method="POST",
            )
            try:
                response = self.opener.open(request, timeout=self.timeout_seconds)
                response_body = response.read()
                http_status = response.status
                response_hash = contract.sha256_bytes(response_body)
                response_hashes.append(response_hash)
                parsed = _parse_response(
                    body=response_body,
                    request_sha256=request_sha256,
                    question_id=question_id,
                    logical_call_id=logical_call_id,
                )
                event_ids.append(parsed["proxy_event_id"])
                upstream_attempts += parsed["upstream_http_attempts"]
                unsupported.update(parsed["unsupported_parameters"])
                ledger.append(
                    "physical_http_attempt_finished",
                    {
                        "attempt": attempt,
                        "status": "accepted",
                        "http_status": http_status,
                        "response_sha256": response_hash,
                        "response_id": parsed["response_id"],
                        "proxy_event_id": parsed["proxy_event_id"],
                        "upstream_http_attempts": parsed["upstream_http_attempts"],
                        "unsupported_parameters": parsed["unsupported_parameters"],
                    },
                )
                return AnswerCallResult(
                    raw_output=parsed["raw_output"],
                    response_id=parsed["response_id"],
                    response_model=parsed["response_model"],
                    usage=parsed["usage"],
                    logical_calls=1,
                    client_http_attempts=attempt,
                    upstream_http_attempts=upstream_attempts,
                    proxy_event_ids=tuple(event_ids),
                    request_sha256s=tuple(request_hashes),
                    response_sha256s=tuple(response_hashes),
                    unsupported_parameters=tuple(sorted(unsupported)),
                )
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                response_hash = contract.sha256_bytes(response_body)
                response_hashes.append(response_hash)
                try:
                    error_metadata = _parse_error_proxy_metadata(
                        body=response_body,
                        request_sha256=request_sha256,
                        question_id=question_id,
                        logical_call_id=logical_call_id,
                    )
                except Exception as metadata_exc:
                    ledger.append(
                        "physical_http_attempt_finished",
                        {
                            "attempt": attempt,
                            "status": "unaccountable_error",
                            "http_status": exc.code,
                            "response_sha256": response_hash,
                            "error": f"{type(metadata_exc).__name__}: {metadata_exc}"[:1000],
                        },
                    )
                    raise contract.ControlledAnswerError(
                        "failed answer request lacks auditable exclusive-proxy metadata"
                    ) from metadata_exc
                event_ids.append(error_metadata["proxy_event_id"])
                unsupported.update(error_metadata["unsupported_parameters"])
                current_upstream_attempts = error_metadata[
                    "upstream_http_attempts"
                ]
                if current_upstream_attempts is None:
                    error_text = str(error_metadata["error"])
                    if (
                        exc.code != 500
                        or "upstream completed with empty output"
                        not in error_text.lower()
                    ):
                        ledger.append(
                            "physical_http_attempt_finished",
                            {
                                "attempt": attempt,
                                "status": "unaccountable_error",
                                "http_status": exc.code,
                                "response_sha256": response_hash,
                                "proxy_event_id": error_metadata[
                                    "proxy_event_id"
                                ],
                                "upstream_http_attempts": None,
                                "unsupported_parameters": error_metadata[
                                    "unsupported_parameters"
                                ],
                                "error": error_text,
                            },
                        )
                        raise contract.ControlledAnswerError(
                            "failed answer request has unknown provider attempts"
                        )
                    last_error = RetryableEmptyOutputError(error_text)
                    ledger.append(
                        "physical_http_attempt_finished",
                        {
                            "attempt": attempt,
                            "status": "retryable_empty_output",
                            "http_status": exc.code,
                            "response_sha256": response_hash,
                            "proxy_event_id": error_metadata[
                                "proxy_event_id"
                            ],
                            "upstream_http_attempts": None,
                            "unsupported_parameters": error_metadata[
                                "unsupported_parameters"
                            ],
                            "error": str(last_error),
                        },
                    )
                    raise last_error
                upstream_attempts += current_upstream_attempts
                last_error = contract.ControlledAnswerError(
                    f"answer HTTP {exc.code}: {response_body[:500]!r}"
                )
                ledger.append(
                    "physical_http_attempt_finished",
                    {
                        "attempt": attempt,
                        "status": "error",
                        "http_status": exc.code,
                        "response_sha256": response_hash,
                        "proxy_event_id": error_metadata["proxy_event_id"],
                        "upstream_http_attempts": error_metadata[
                            "upstream_http_attempts"
                        ],
                        "unsupported_parameters": error_metadata[
                            "unsupported_parameters"
                        ],
                        "error": str(last_error),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                ledger.append(
                    "physical_http_attempt_finished",
                    {
                        "attempt": attempt,
                        "status": "error",
                        "http_status": None,
                        "response_sha256": None,
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                    },
                )
            if attempt < self.retries:
                time.sleep(min(2 * attempt, 10))
        raise contract.ControlledAnswerError(
            f"answer call failed after {self.retries} attempts: {last_error}"
        )


class FakeAnswerClient:
    """No-network client used only by the synthetic sanity test."""

    def __init__(
        self,
        *,
        proxy_log: Path,
        run_id: str,
        response_text: str = "<answer>synthetic answer</answer>",
    ):
        self.proxy_log = proxy_log
        self.run_id = run_id
        self.response_text = response_text
        self.prompts: list[str] = []
        proxy_log.parent.mkdir(parents=True, exist_ok=True)
        if proxy_log.exists() or proxy_log.is_symlink():
            raise FileExistsError(proxy_log)
        proxy_log.touch(mode=0o600)

    def complete(
        self,
        *,
        prompt: str,
        question_id: str,
        logical_call_id: str,
        ledger: contract.DurableLedger,
    ) -> AnswerCallResult:
        self.prompts.append(prompt)
        body = json.dumps(
            {
                "model": contract.EXPECTED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request_hash = contract.sha256_bytes(body)
        response_id = f"fake-{uuid.uuid4().hex}"
        event_id = f"exclusive-{uuid.uuid4().hex}"
        usage = {
            "prompt_tokens": len(prompt.encode("utf-8")),
            "completion_tokens": len(self.response_text.encode("utf-8")),
            "total_tokens": len(prompt.encode("utf-8"))
            + len(self.response_text.encode("utf-8")),
        }
        response_payload = {
            "id": response_id,
            "model": contract.EXPECTED_MODEL,
            "choices": [
                {
                    "message": {"content": self.response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        response_bytes = json.dumps(
            response_payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        response_hash = contract.sha256_bytes(response_bytes)
        ledger.append(
            "physical_http_attempt_started",
            {
                "attempt": 1,
                "question_id": question_id,
                "logical_call_id": logical_call_id,
                "request_sha256": request_hash,
            },
        )
        record = {
            "run_id": self.run_id,
            "started_at": contract.utc_now(),
            "finished_at": contract.utc_now(),
            "status": "success",
            "http_status": 200,
            "requested_model": contract.EXPECTED_MODEL,
            "actual_model": contract.EXPECTED_MODEL,
            "response_id": response_id,
            "event_id": event_id,
            "question_id": question_id,
            "logical_call_id": logical_call_id,
            "request_sha256": request_hash,
            "response_sha256": response_hash,
            "client_http_attempts": 1,
            "upstream_http_attempts": 1,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": usage,
            "error": None,
        }
        with self.proxy_log.open("a", encoding="utf-8") as handle:
            handle.write(contract.canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        ledger.append(
            "physical_http_attempt_finished",
            {
                "attempt": 1,
                "status": "accepted",
                "http_status": 200,
                "response_sha256": response_hash,
                "response_id": response_id,
                "proxy_event_id": event_id,
                "upstream_http_attempts": 1,
                "unsupported_parameters": ["max_output_tokens"],
            },
        )
        return AnswerCallResult(
            raw_output=self.response_text,
            response_id=response_id,
            response_model=contract.EXPECTED_MODEL,
            usage=usage,
            logical_calls=1,
            client_http_attempts=1,
            upstream_http_attempts=1,
            proxy_event_ids=(event_id,),
            request_sha256s=(request_hash,),
            response_sha256s=(response_hash,),
            unsupported_parameters=("max_output_tokens",),
        )


def answer_one(
    *,
    run_id: str,
    method: str,
    question_id: str,
    question: str,
    memories: Sequence[Any],
    input_record_sha256: str,
    attempt_dir: Path,
    tokenizer: TokenCounter,
    budget_policy: str,
    budget_tokens: int | None,
    model_context_limit_tokens: int,
    answer_max_tokens: int,
    preregistration_sha256: str,
    client: AnswerClient,
) -> dict[str, Any]:
    """Answer one question without accepting gold/evidence/category arguments."""

    rendered = contract.render_memories_individually(memories)
    if budget_policy == contract.HARD_CAP_POLICY:
        if budget_tokens is None or budget_tokens <= 0:
            raise contract.ControlledAnswerError("hard-cap question lacks a budget")
        effective_gate_budget = budget_tokens
        declared_budget: int | str = budget_tokens
    elif budget_policy == contract.FULL_CONTEXT_POLICY:
        if method != "full_context" or budget_tokens is not None:
            raise contract.ControlledAnswerError("full-context budget policy is inconsistent")
        effective_gate_budget = sum(
            tokenizer.count(item.rendered_text) for item in rendered
        )
        if effective_gate_budget <= 0:
            raise contract.ControlledAnswerError("full-context rendered no visible tokens")
        declared_budget = "unbounded"
    else:
        raise contract.ControlledAnswerError(f"unsupported budget policy: {budget_policy}")

    trace_path = attempt_dir / "visible_tokens.jsonl"
    manifest_path = attempt_dir / "visible_tokens.manifest.json"
    ledger_path = attempt_dir / "question_ledger.jsonl"
    memory_bytes = contract.canonical_json(
        {
            "question_id": question_id,
            "rendered_memories": [
                {
                    "input_index": item.input_index,
                    "rendered_rank": item.rendered_rank,
                    "rendered_text": item.rendered_text,
                }
                for item in rendered
            ],
        }
    ).encode("utf-8")
    snapshot = snapshot_memory_bytes(memory_bytes, label=f"{method}:{question_id}")
    logical_call_id = f"{run_id}:{question_id}:{attempt_dir.name}"
    deliveries = []
    with contract.DurableLedger(ledger_path, run_id=logical_call_id) as ledger:
        ledger.append(
            "question_started",
            {
                "method": method,
                "question_id": question_id,
                "input_record_sha256": input_record_sha256,
                "budget_policy": budget_policy,
                "declared_budget_tokens": declared_budget,
                "effective_gate_budget_tokens": effective_gate_budget,
                "preregistration_sha256": preregistration_sha256,
            },
        )
        gate = VisibleTokenBudgetGate(
            trace_path=trace_path,
            manifest_path=manifest_path,
            run_id=logical_call_id,
            configured_budget_tokens=effective_gate_budget,
            tokenizer=tokenizer,
            memory_before=snapshot,
            overflow_policy="truncate",
            metadata={
                "method": method,
                "question_id": question_id,
                "budget_policy": budget_policy,
                "declared_budget_tokens": declared_budget,
                "preregistration_sha256": preregistration_sha256,
                "source_resolution_expected_tokens": 0,
            },
        )
        try:
            for item in rendered:
                deliveries.append(
                    gate.deliver_tool_result(
                        event_id=f"memory-{item.rendered_rank:04d}",
                        raw_text=item.rendered_text,
                        tool_name="retrieved_memory",
                        tool_call_id=f"{question_id}:memory:{item.rendered_rank}",
                        metadata={
                            "input_index": item.input_index,
                            "rendered_rank": item.rendered_rank,
                        },
                    )
                )
            prompt = contract.assemble_answer_prompt(
                question=question, deliveries=deliveries
            )
            prompt_sha256 = contract.sha256_bytes(prompt.encode("utf-8"))
            prompt_local_tokens = tokenizer.count(prompt)
            local_context_total = prompt_local_tokens + answer_max_tokens
            if local_context_total > model_context_limit_tokens:
                raise contract.ControlledAnswerError(
                    "local prompt tokens plus completion reservation exceed the "
                    f"declared model context limit: {local_context_total} > "
                    f"{model_context_limit_tokens}"
                )
            if budget_policy == contract.FULL_CONTEXT_POLICY and any(
                delivery.decision != "delivered" for delivery in deliveries
            ):
                raise contract.ControlledAnswerError(
                    "full-context delivery was truncated or rejected; it cannot be "
                    "reported as full-context"
                )
            ledger.append(
                "prompt_ready",
                {
                    "prompt_sha256": prompt_sha256,
                    "prompt_local_tokens": prompt_local_tokens,
                    "model_context_limit_tokens": model_context_limit_tokens,
                    "answer_completion_reservation_tokens": answer_max_tokens,
                    "delivered_event_ids": [item.event_id for item in deliveries],
                    "delivered_text_sha256": contract.sha256_bytes(
                        contract.delivered_memory_block(deliveries).encode("utf-8")
                    ),
                },
            )
            call = client.complete(
                prompt=prompt,
                question_id=question_id,
                logical_call_id=logical_call_id,
                ledger=ledger,
            )
            if call.response_model != contract.EXPECTED_MODEL:
                raise contract.ControlledAnswerError("accepted answer model differs")
            provider_total_tokens = call.usage.get("total_tokens")
            if (
                not isinstance(provider_total_tokens, int)
                or isinstance(provider_total_tokens, bool)
                or provider_total_tokens < 0
                or provider_total_tokens > model_context_limit_tokens
            ):
                raise contract.ControlledAnswerError(
                    "provider-reported total usage is invalid or exceeds the "
                    "preregistered model context limit"
                )
            actual_usage = {
                "requested_model": contract.EXPECTED_MODEL,
                "response_model": call.response_model,
                "response_id": call.response_id,
                "logical_answer_calls": call.logical_calls,
                "client_http_attempts": call.client_http_attempts,
                "upstream_http_attempts": call.upstream_http_attempts,
                "provider_usage": call.usage,
                "exclusive_proxy_event_ids": list(call.proxy_event_ids),
                "unsupported_parameters": list(call.unsupported_parameters),
            }
            gate_manifest = gate.finalize(
                memory_after=snapshot,
                actual_model_usage=actual_usage,
                reason=(
                    "full_context_accounted_complete"
                    if budget_policy == contract.FULL_CONTEXT_POLICY
                    else None
                ),
                metadata={
                    "prompt_sha256": prompt_sha256,
                    "question_id": question_id,
                    "response_id": call.response_id,
                },
            )
            ledger.append(
                "question_completed",
                {
                    "prompt_sha256": prompt_sha256,
                    "response_id": call.response_id,
                    "response_model": call.response_model,
                    "gate_trace_sha256": gate_manifest["trace_sha256"],
                    "logical_answer_calls": call.logical_calls,
                    "client_http_attempts": call.client_http_attempts,
                    "upstream_http_attempts": call.upstream_http_attempts,
                },
            )
        except Exception:
            gate.close_incomplete()
            raise

    gate_audit = audit_visible_token_trace(
        trace_path,
        manifest_path=manifest_path,
        require_complete=True,
    )
    if gate_audit.get("audit_status") != "pass":
        raise contract.ControlledAnswerError(
            f"post-answer visible-token audit failed: {gate_audit.get('errors')}"
        )
    summary = {
        "cumulative_visible_tokens": gate_audit.get("cumulative_visible_tokens"),
        "cumulative_source_resolution_tokens": gate_audit.get(
            "cumulative_source_resolution_tokens"
        ),
    }
    if summary["cumulative_source_resolution_tokens"] != 0:
        raise contract.ControlledAnswerError("baseline source-resolution subtotal differs")
    result = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "logical_call_id": logical_call_id,
        "method": method,
        "question_id": question_id,
        "input_record_sha256": input_record_sha256,
        "question_sha256": contract.sha256_bytes(question.encode("utf-8")),
        "budget": {
            "policy": budget_policy,
            "declared_tokens": declared_budget,
            "effective_gate_tokens": effective_gate_budget,
            "tokenizer": tokenizer.identity,
            "visible_tokens": summary["cumulative_visible_tokens"],
            "source_resolution_tokens": 0,
            "full_context_matched_cap_claimed": False,
        },
        "prompt": {
            "sha256": prompt_sha256,
            "local_tokens": prompt_local_tokens,
            "model_context_limit_tokens": model_context_limit_tokens,
            "answer_completion_reservation_tokens": answer_max_tokens,
            "local_context_check_passed": True,
            "provider_reported_total_tokens": call.usage["total_tokens"],
            "provider_context_check_passed": True,
            "memory_payload_sha256": contract.sha256_bytes(
                contract.delivered_memory_block(deliveries).encode("utf-8")
            ),
        },
        "answer": {
            "text": contract.extract_answer(call.raw_output),
            "raw_output": call.raw_output,
            "raw_output_sha256": contract.sha256_bytes(
                call.raw_output.encode("utf-8")
            ),
            "requested_model": contract.EXPECTED_MODEL,
            "response_model": call.response_model,
            "response_id": call.response_id,
            "usage": call.usage,
            "logical_answer_calls": call.logical_calls,
            "client_http_attempts": call.client_http_attempts,
            "upstream_http_attempts": call.upstream_http_attempts,
            "exclusive_proxy_event_ids": list(call.proxy_event_ids),
            "request_sha256s": list(call.request_sha256s),
            "response_sha256s": list(call.response_sha256s),
            "unsupported_parameters": list(call.unsupported_parameters),
        },
        "artifacts": {
            "visible_token_trace": trace_path.name,
            "visible_token_trace_sha256": contract.sha256_file(trace_path),
            "visible_token_manifest": manifest_path.name,
            "visible_token_manifest_sha256": contract.sha256_file(manifest_path),
            "question_ledger": ledger_path.name,
            "question_ledger_sha256": contract.sha256_file(ledger_path),
        },
        "preregistration_sha256": preregistration_sha256,
    }
    return result


def _parse_budget(value: str) -> int | None:
    if value == "unbounded":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "budget must be a positive integer or literal 'unbounded'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("budget must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=contract.FORMAL_METHODS, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument(
        "--budget-policy",
        choices=(contract.HARD_CAP_POLICY, contract.FULL_CONTEXT_POLICY),
        required=True,
    )
    parser.add_argument(
        "--budget-tokens",
        type=_parse_budget,
        required=True,
        help="explicit integer for hard_cap, literal 'unbounded' for full-context",
    )
    parser.add_argument("--model-context-limit-tokens", type=int, required=True)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument(
        "--gateway-root",
        type=Path,
        required=True,
        help="active OpenAI GPT-5.5 Flex gateway result root",
    )
    args = parser.parse_args(argv)
    if not args.allow_model_requests:
        parser.error("formal controlled answers require --allow-model-requests")
    return args


def _validate_args(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if args.model_context_limit_tokens <= 0 or args.answer_max_tokens <= 0:
        raise contract.ControlledAnswerError("context and completion limits must be positive")
    if args.retries < 1:
        raise contract.ControlledAnswerError("retries must be positive")
    if not isinstance(args.gateway_contract, dict):
        raise contract.ControlledAnswerError("validated Flex gateway contract is absent")
    if args.method == "full_context":
        if args.budget_policy != contract.FULL_CONTEXT_POLICY or args.budget_tokens is not None:
            raise contract.ControlledAnswerError(
                "full_context requires --budget-policy full_context_unbounded_accounted "
                "and --budget-tokens unbounded"
            )
    elif (
        args.budget_policy != contract.HARD_CAP_POLICY
        or args.budget_tokens != contract.EXPECTED_HARD_BUDGET
    ):
        raise contract.ControlledAnswerError(
            "controlled retrieval baselines require an explicit 20000-token hard cap"
        )
    rows = protocol["rows"]
    row = next((item for item in rows if item.get("method") == args.method), None)
    if not isinstance(row, dict):
        raise contract.ControlledAnswerError("method is absent from preregistration")
    expected_budget = "unbounded" if args.budget_tokens is None else args.budget_tokens
    for key, actual in (
        ("budget_policy", args.budget_policy),
        ("budget_tokens", expected_budget),
        ("model_context_limit_tokens", args.model_context_limit_tokens),
        ("answer_completion_reservation_tokens", args.answer_max_tokens),
    ):
        if row.get(key) != actual:
            raise contract.ControlledAnswerError(
                f"CLI {key} differs from preregistration: {actual!r} != {row.get(key)!r}"
            )
    return row


def _load_formal_input(
    input_dir: Path, method: str, row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report = baseline_auditor.audit(input_dir)
    if report.get("status") != "passed" or report.get("method") != method:
        raise contract.ControlledAnswerError("baseline input audit or method failed")
    binding = row.get("input")
    if not isinstance(binding, dict):
        raise contract.ControlledAnswerError("preregistered input binding is missing")
    manifest_path = input_dir / "run_manifest.json"
    inputs_manifest_path = input_dir / "inputs_manifest.json"
    questions_path = input_dir / "questions.json"
    observed = {
        "run_dir": str(input_dir),
        "method": method,
        "run_fingerprint": contract.read_json(manifest_path).get("fingerprint"),
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "inputs_manifest_sha256": contract.sha256_file(inputs_manifest_path),
        "questions_sha256": contract.sha256_file(questions_path),
        "dataset_sha256": report.get("dataset_sha256"),
    }
    for key, value in observed.items():
        if binding.get(key) != value:
            raise contract.ControlledAnswerError(
                f"baseline input binding differs in {key}"
            )
    rows = contract.read_json(questions_path)
    if not isinstance(rows, list):
        raise contract.ControlledAnswerError("questions.json is not a list")
    build_rows = [row for row in rows if row.get("question_id") == "_build_stats"]
    questions = [row for row in rows if row.get("question_id") != "_build_stats"]
    if len(build_rows) != 10 or len(questions) != contract.EXPECTED_QUESTIONS:
        raise contract.ControlledAnswerError("formal input inventory differs")
    for question in questions:
        contract.validate_question_record(question)
    categories = Counter(question["category"] for question in questions)
    if dict(sorted(categories.items())) != contract.EXPECTED_CATEGORIES:
        raise contract.ControlledAnswerError("formal input categories differ")
    run_manifest = contract.read_json(manifest_path)
    dataset_path = Path(run_manifest["config"]["dataset"]).expanduser().resolve()
    if dataset_path.is_symlink() or not dataset_path.is_file():
        raise contract.ControlledAnswerError("raw dataset path is unavailable")
    if contract.sha256_file(dataset_path) != report.get("dataset_sha256"):
        raise contract.ControlledAnswerError("raw dataset hash differs")
    dataset = contract.read_json(dataset_path)
    if not isinstance(dataset, list) or len(dataset) != 10:
        raise contract.ControlledAnswerError("raw dataset inventory differs")
    for record in questions:
        reference = contract.canonical_scoring_reference(dataset, record["question_id"])
        if (
            record["question"] != reference["question"]
            or record["category"] != reference["category"]
            or record["evidence"] != reference["evidence"]
        ):
            raise contract.ControlledAnswerError(
                f"input row differs from raw dataset: {record['question_id']}"
            )
        # Baseline input ``gold`` is legacy retrieval-only metadata.  For cat5
        # rows without an explicit raw answer it equals the distractor and is
        # never propagated as canonical gold.
        if record["category"] == 5:
            expected_legacy_gold = (
                reference["canonical_gold"]
                if reference["canonical_gold_source"] == "raw_dataset.answer"
                else reference["adversarial_distractor"]
            )
            if record["gold"] != expected_legacy_gold:
                raise contract.ControlledAnswerError(
                    f"cat5 legacy gold field differs: {record['question_id']}"
                )
        elif record["gold"] != reference["canonical_gold"]:
            raise contract.ControlledAnswerError(
                f"canonical input gold differs: {record['question_id']}"
            )
    return questions, dataset, report


def _source_hashes() -> dict[str, str]:
    sources = {
        "runner": Path(__file__).resolve(),
        "contract": Path(contract.__file__).resolve(),
        "run_proxy": RUN_PROXY,
        "base_run_proxy": BASE_RUN_PROXY,
        "upstream_proxy": UPSTREAM_PROXY,
        "flex_gateway_evidence": FLEX_EVIDENCE,
        "auditor": AUDITOR,
        "visible_token_budget": ROOT / "src/evaluation/visible_token_budget.py",
        "visible_token_audit": ROOT / "src/evaluation/visible_token_audit.py",
        "prompts": ROOT / "src/evaluation/prompts.py",
    }
    return {key: contract.sha256_file(path) for key, path in sources.items()}


def _create_or_validate_manifest(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    tokenizer: TokenCounter,
    run_id: str,
) -> dict[str, Any]:
    output_dir = args.output_dir
    path = output_dir / "run_manifest.json"
    expected_config = {
        "method": args.method,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "preregistration": str(args.preregistration),
        "preregistration_sha256": args.preregistration_sha256,
        "protocol_content_sha256": protocol["protocol_content_sha256"],
        "budget_policy": args.budget_policy,
        "budget_tokens": "unbounded" if args.budget_tokens is None else args.budget_tokens,
        "model_context_limit_tokens": args.model_context_limit_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "retries": args.retries,
        "model": contract.EXPECTED_MODEL,
        "gateway_contract": args.gateway_contract,
        "model_requests_authorized": args.allow_model_requests,
        "python": str(args.python),
        "python_resolved": str(args.python.resolve()),
        "python_sha256": contract.sha256_file(args.python.resolve()),
        "runner_python_resolved": str(Path(sys.executable).resolve()),
        "runner_python_sha256": contract.sha256_file(Path(sys.executable).resolve()),
        "tokenizer": tokenizer.identity,
        "formal_row": row,
    }
    if path.exists():
        manifest = contract.read_json(path)
        if not isinstance(manifest, dict):
            raise contract.ControlledAnswerError("existing run manifest is invalid")
        if manifest.get("config") != expected_config:
            raise contract.ControlledAnswerError("resume configuration differs")
        if manifest.get("source_hashes") != _source_hashes():
            raise contract.ControlledAnswerError("resume source hashes differ")
        return manifest
    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "answering",
        "run_id": run_id,
        "created_at": contract.utc_now(),
        "config": expected_config,
        "source_hashes": _source_hashes(),
        "inventory": {
            "questions": contract.EXPECTED_QUESTIONS,
            "primary_cat1_4": contract.EXPECTED_PRIMARY,
            "adversarial_cat5": contract.EXPECTED_ADVERSARIAL,
        },
        "scoring_label_policy": protocol["scoring_labels"],
    }
    contract.atomic_json_no_clobber(path, manifest)
    return manifest


def load_flex_gateway_contract(result_root: Path) -> dict[str, Any]:
    try:
        return flex_evidence.active_contract(result_root)
    except flex_evidence.EvidenceError as exc:
        raise contract.ControlledAnswerError(str(exc)) from exc


def _start_proxy(
    *,
    output_dir: Path,
    python: Path,
    gateway_contract: Mapping[str, Any],
    run_id: str,
) -> tuple[subprocess.Popen, dict[str, Any], Any]:
    try:
        provider_contract = flex_evidence.validate_recorded_contract(
            gateway_contract
        )
        provider_window = flex_evidence.capture_start(
            Path(str(provider_contract["result_root"]))
        )
    except flex_evidence.EvidenceError as exc:
        raise contract.ControlledAnswerError(
            f"cannot capture Flex provider start: {exc}"
        ) from exc
    upstream = str(provider_contract["origin"])
    invocation_id = f"invocation-{uuid.uuid4().hex}"
    invocation_dir = output_dir / "proxy" / invocation_id
    invocation_dir.mkdir(parents=True, exist_ok=False)
    ready = invocation_dir / "ready.json"
    log = invocation_dir / "requests.jsonl"
    process_log_path = invocation_dir / "process.log"
    process_log = process_log_path.open("x", encoding="utf-8")
    started_at = contract.utc_now()
    process = subprocess.Popen(
        [
            str(python),
            str(RUN_PROXY),
            "--port",
            "0",
            "--upstream",
            upstream,
            "--log",
            str(log),
            "--ready",
            str(ready),
            "--run-id",
            run_id,
        ],
        cwd=ROOT,
        stdout=process_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            process_log.close()
            raise contract.ControlledAnswerError("exclusive proxy exited before readiness")
        if ready.is_file():
            metadata = contract.read_json(ready)
            if isinstance(metadata, dict) and metadata.get("run_id") == run_id:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                try:
                    health = json.loads(
                        opener.open(
                            f"http://127.0.0.1:{metadata['port']}/healthz", timeout=5
                        ).read()
                    )
                    upstream_health = health.get("upstream_health")
                    if (
                        health.get("status") == "ok"
                        and isinstance(upstream_health, dict)
                        and upstream_health.get("schema")
                        == "openai-gpt55-flex-health/v1"
                        and upstream_health.get("requested_model")
                        == flex_gateway.REQUESTED_MODEL
                        and upstream_health.get("provider_model")
                        == flex_gateway.PROVIDER_MODEL
                        and upstream_health.get("service_tier")
                        == flex_gateway.SERVICE_TIER
                        and isinstance(upstream_health.get("budget"), dict)
                        and upstream_health["budget"].get("max_cost_usd")
                        == provider_contract["max_cost_usd"]
                    ):
                        record = {
                            "run_id": run_id,
                            "invocation_id": invocation_id,
                            "invocation_dir": str(invocation_dir.relative_to(output_dir)),
                            "started_at": started_at,
                            "pid": process.pid,
                            "base_url": metadata["base_url"],
                            "ready": str(ready.relative_to(output_dir)),
                            "log": str(log.relative_to(output_dir)),
                            "process_log": str(process_log_path.relative_to(output_dir)),
                            "health": health,
                            "upstream": upstream,
                            "gateway_result_root": provider_contract["result_root"],
                            "gateway_root_marker_sha256": provider_contract[
                                "root_marker_sha256"
                            ],
                            "gateway_ready_sha256": provider_contract["ready_sha256"],
                            "gateway_max_cost_usd": provider_contract["max_cost_usd"],
                            "provider_model": flex_gateway.PROVIDER_MODEL,
                            "service_tier": flex_gateway.SERVICE_TIER,
                            "provider_window": provider_window,
                        }
                        start_path = invocation_dir / "start.json"
                        contract.atomic_json_no_clobber(start_path, record)
                        record["start"] = str(start_path.relative_to(output_dir))
                        record["start_sha256"] = contract.sha256_file(start_path)
                        record["start_bytes"] = start_path.stat().st_size
                        return process, record, process_log
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(0.1)
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=10)
    process_log.close()
    raise contract.ControlledAnswerError("exclusive proxy did not become healthy")


def _stop_proxy(
    process: subprocess.Popen,
    record: dict[str, Any],
    process_log: Any,
    output_dir: Path,
) -> dict[str, Any]:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    process_log.close()
    record = dict(record)
    record.update(
        {
            "finished_at": contract.utc_now(),
            "returncode": process.returncode,
            "wrapper_sha256": contract.sha256_file(RUN_PROXY),
            "base_wrapper_sha256": contract.sha256_file(BASE_RUN_PROXY),
            "upstream_proxy_sha256": contract.sha256_file(UPSTREAM_PROXY),
            "flex_evidence_sha256": contract.sha256_file(FLEX_EVIDENCE),
        }
    )
    provider_window_error = None
    try:
        record["provider_window"] = flex_evidence.capture_end(
            record["provider_window"]
        )
    except flex_evidence.EvidenceError as exc:
        provider_window_error = str(exc)
        record["provider_window_error"] = provider_window_error
    for key in ("start", "ready", "log", "process_log"):
        path = output_dir / record[key]
        record[f"{key}_sha256"] = contract.sha256_file(path)
        record[f"{key}_bytes"] = path.stat().st_size
    manifest_path = output_dir / record["invocation_dir"] / "manifest.json"
    contract.atomic_json_no_clobber(manifest_path, record)
    if provider_window_error is not None:
        raise contract.ControlledAnswerError(
            f"cannot close Flex provider window: {provider_window_error}"
        )
    return record


def _recover_interrupted_proxies(output_dir: Path, run_id: str) -> None:
    """Capture an interrupted exclusive wrapper without using shared windows."""

    proxy_root = output_dir / "proxy"
    if not proxy_root.exists():
        return
    contract.reject_symlink_components(proxy_root)
    if proxy_root.is_symlink() or not proxy_root.is_dir():
        raise contract.ControlledAnswerError("proxy artifact root is not a regular directory")
    for invocation_dir in sorted(proxy_root.glob("invocation-*")):
        contract.reject_symlink_components(invocation_dir)
        if invocation_dir.is_symlink() or not invocation_dir.is_dir():
            raise contract.ControlledAnswerError(
                f"proxy invocation is not a regular directory: {invocation_dir}"
            )
        manifest_path = invocation_dir / "manifest.json"
        if manifest_path.exists():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise contract.ControlledAnswerError(
                    f"proxy manifest is not a regular file: {manifest_path}"
                )
            continue
        start_path = invocation_dir / "start.json"
        if not start_path.is_file() or start_path.is_symlink():
            raise contract.ControlledAnswerError(
                f"interrupted proxy lacks a start record: {invocation_dir}"
            )
        record = contract.read_json(start_path)
        if not isinstance(record, dict):
            raise contract.ControlledAnswerError("interrupted proxy start record is invalid")
        if (
            record.get("run_id") != run_id
            or record.get("invocation_id") != invocation_dir.name
            or record.get("invocation_dir")
            != str(invocation_dir.relative_to(output_dir))
        ):
            raise contract.ControlledAnswerError("interrupted proxy directory linkage differs")
        artifact_paths: dict[str, Path] = {"start": start_path}
        for key, expected_name in (
            ("ready", "ready.json"),
            ("log", "requests.jsonl"),
            ("process_log", "process.log"),
        ):
            relative = record.get(key)
            expected_relative = str(
                (invocation_dir / expected_name).relative_to(output_dir)
            )
            if relative != expected_relative:
                raise contract.ControlledAnswerError(
                    f"interrupted proxy {key} path linkage differs"
                )
            path = output_dir / relative
            contract.reject_symlink_components(path)
            try:
                path.resolve().relative_to(output_dir.resolve())
            except ValueError as exc:
                raise contract.ControlledAnswerError(
                    f"interrupted proxy {key} path escapes output"
                ) from exc
            if path.is_symlink() or not path.is_file():
                raise contract.ControlledAnswerError(
                    f"interrupted proxy {key} artifact is unavailable"
                )
            artifact_paths[key] = path
        contract.ensure_distinct_paths(artifact_paths)
        pid = record.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise contract.ControlledAnswerError("interrupted proxy PID is invalid")
        running = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            running = False
        recovery_action = "original_process_not_running"
        returncode: int | None = None
        if running:
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            expected = (
                str(RUN_PROXY) in command
                and run_id in command
                and str(artifact_paths["log"]) in command
                and str(artifact_paths["ready"]) in command
            )
            if expected:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                recovery_action = "terminated_original_process"
                returncode = -signal.SIGTERM
            else:
                recovery_action = "original_pid_reused"
        finished = dict(record)
        finished.update(
            {
                "status": "interrupted_recovered",
                "recovery_action": recovery_action,
                "finished_at": contract.utc_now(),
                "returncode": returncode,
                "start": str(start_path.relative_to(output_dir)),
                "start_sha256": contract.sha256_file(start_path),
                "start_bytes": start_path.stat().st_size,
                "wrapper_sha256": contract.sha256_file(RUN_PROXY),
                "base_wrapper_sha256": contract.sha256_file(BASE_RUN_PROXY),
                "upstream_proxy_sha256": contract.sha256_file(UPSTREAM_PROXY),
                "flex_evidence_sha256": contract.sha256_file(FLEX_EVIDENCE),
            }
        )
        try:
            finished["provider_window"] = flex_evidence.capture_end(
                record.get("provider_window", {})
            )
        except flex_evidence.EvidenceError as exc:
            finished["provider_window_error"] = str(exc)
        for key in ("ready", "log", "process_log"):
            path = artifact_paths[key]
            finished[f"{key}_sha256"] = contract.sha256_file(path)
            finished[f"{key}_bytes"] = path.stat().st_size
        contract.atomic_json_no_clobber(manifest_path, finished)


def _next_attempt_dir(question_dir: Path) -> Path:
    question_dir.mkdir(parents=True, exist_ok=True)
    attempts = question_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    numbers = []
    for path in attempts.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    attempt = attempts / f"attempt-{(max(numbers, default=0) + 1):04d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def _write_completion(
    *,
    output_dir: Path,
    question_id: str,
    attempt_dir: Path,
    result: dict[str, Any],
) -> Path:
    result_path = attempt_dir / "result.json"
    contract.atomic_json_no_clobber(result_path, result)
    checkpoint = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "complete",
        "question_id": question_id,
        "attempt_dir": str(attempt_dir.relative_to(output_dir)),
        "result": str(result_path.relative_to(output_dir)),
        "result_sha256": contract.sha256_file(result_path),
        "completed_at": contract.utc_now(),
    }
    checkpoint_path = output_dir / "completed" / f"{question_id}.json"
    contract.atomic_json_no_clobber(checkpoint_path, checkpoint)
    return checkpoint_path


def _run(args: argparse.Namespace) -> int:
    args.input_dir = args.input_dir.expanduser().absolute()
    args.output_dir = args.output_dir.expanduser().absolute()
    args.preregistration = args.preregistration.expanduser().absolute()
    args.python = args.python.expanduser().absolute()
    args.gateway_root = args.gateway_root.expanduser().absolute()
    contract.ensure_distinct_paths(
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "preregistration": args.preregistration,
            "gateway_root": args.gateway_root,
        }
    )
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.preregistration = args.preregistration.resolve()
    try:
        args.output_dir.relative_to(args.input_dir)
    except ValueError:
        pass
    else:
        raise contract.ControlledAnswerError("output directory must not be inside input")
    try:
        args.input_dir.relative_to(args.output_dir)
    except ValueError:
        pass
    else:
        raise contract.ControlledAnswerError("input directory must not be inside output")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise contract.ControlledAnswerError("configured Python is not executable")
    if Path(sys.executable).resolve() != args.python.resolve():
        raise contract.ControlledAnswerError(
            "formal runner and configured proxy Python executables differ"
        )
    protocol = contract.validate_preregistration(
        args.preregistration,
        expected_file_sha256=args.preregistration_sha256,
    )
    row = _validate_args(args, protocol)
    expected_prompt_sources = protocol["answerer"]["prompt_source_hashes"]
    if expected_prompt_sources != contract.prompt_source_hashes(ROOT):
        raise contract.ControlledAnswerError("prompt/gate source hashes differ from freeze")
    tokenizer = contract.formal_token_counter()
    questions, dataset, _input_report = _load_formal_input(
        args.input_dir, args.method, row
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"controlled-{args.method}-{uuid.uuid4().hex}"
    with contract.FileLock(args.output_dir / ".answer-run.lock"):
        manifest = _create_or_validate_manifest(
            args=args,
            protocol=protocol,
            row=row,
            tokenizer=tokenizer,
            run_id=run_id,
        )
        run_id = manifest["run_id"]
        _recover_interrupted_proxies(args.output_dir, run_id)
        if (args.output_dir / "complete.json").exists():
            from scripts.controlled_locomo.audit_controlled_locomo_answers import audit_run  # noqa: PLC0415

            final_report = audit_run(args.output_dir, require_complete_manifest=True)
            contract.atomic_json_replace(args.output_dir / "audit.json", final_report)
            print(json.dumps(final_report, ensure_ascii=False, indent=2))
            return 0
        run_ledger_path = args.output_dir / "run_ledger.jsonl"
        with contract.DurableLedger(run_ledger_path, run_id=run_id) as run_ledger:
            from scripts.controlled_locomo.audit_controlled_locomo_answers import (  # noqa: PLC0415
                _load_proxy_entries,
                audit_question,
            )

            resume_proxy_entries, _resume_proxy_report = _load_proxy_entries(
                args.output_dir, run_id
            )
            pending: list[dict[str, Any]] = []
            for record in questions:
                checkpoint = args.output_dir / "completed" / f"{record['question_id']}.json"
                if checkpoint.exists():
                    # Full independent re-audit is required before resume may skip.
                    audit_question(
                        output_dir=args.output_dir,
                        input_record=record,
                        raw_dataset=dataset,
                        run_manifest=manifest,
                        protocol=protocol,
                        checkpoint_path=checkpoint,
                        proxy_entries=resume_proxy_entries,
                        require_proxy_log=True,
                    )
                    run_ledger.append(
                        "resume_question_reaudited",
                        {"question_id": record["question_id"]},
                    )
                else:
                    pending.append(record)
            if not pending:
                run_ledger.append("resume_no_pending_questions", {})
            else:
                process, proxy_record, process_log = _start_proxy(
                    output_dir=args.output_dir,
                    python=args.python,
                    gateway_contract=args.gateway_contract,
                    run_id=run_id,
                )
                run_ledger.append("proxy_started", proxy_record)
                proxy_finished: dict[str, Any] | None = None
                try:
                    client = HttpAnswerClient(
                        base_url=proxy_record["base_url"],
                        retries=args.retries,
                        answer_max_tokens=args.answer_max_tokens,
                    )
                    for index, record in enumerate(pending, start=1):
                        question_id = record["question_id"]
                        question_dir = args.output_dir / "questions" / question_id
                        attempt_dir = _next_attempt_dir(question_dir)
                        input_record_sha256 = contract.canonical_hash(record)
                        run_ledger.append(
                            "question_attempt_started",
                            {
                                "question_id": question_id,
                                "pending_index": index,
                                "pending_total": len(pending),
                                "attempt_dir": str(attempt_dir.relative_to(args.output_dir)),
                                "input_record_sha256": input_record_sha256,
                            },
                        )
                        result = answer_one(
                            run_id=run_id,
                            method=args.method,
                            question_id=question_id,
                            question=record["question"],
                            memories=record["memories"],
                            input_record_sha256=input_record_sha256,
                            attempt_dir=attempt_dir,
                            tokenizer=tokenizer,
                            budget_policy=args.budget_policy,
                            budget_tokens=args.budget_tokens,
                            model_context_limit_tokens=args.model_context_limit_tokens,
                            answer_max_tokens=args.answer_max_tokens,
                            preregistration_sha256=args.preregistration_sha256,
                            client=client,
                        )
                        reference = contract.canonical_scoring_reference(dataset, question_id)
                        # This attachment occurs after model completion.  ``answer_one``
                        # cannot receive any of these label/evidence fields.
                        result["scoring_reference"] = {
                            "category": reference["category"],
                            "canonical_gold": reference["canonical_gold"],
                            "canonical_gold_source": reference["canonical_gold_source"],
                            "adversarial_distractor": reference["adversarial_distractor"],
                            "evidence": reference["evidence"],
                            "attached_after_answer": True,
                            "baseline_input_gold_authoritative": False,
                        }
                        checkpoint = _write_completion(
                            output_dir=args.output_dir,
                            question_id=question_id,
                            attempt_dir=attempt_dir,
                            result=result,
                        )
                        run_ledger.append(
                            "question_attempt_completed",
                            {
                                "question_id": question_id,
                                "checkpoint": str(checkpoint.relative_to(args.output_dir)),
                                "checkpoint_sha256": contract.sha256_file(checkpoint),
                            },
                        )
                finally:
                    proxy_finished = _stop_proxy(
                        process, proxy_record, process_log, args.output_dir
                    )
                    run_ledger.append("proxy_finished", proxy_finished)

        completed = list((args.output_dir / "completed").glob("s*_q*.json"))
        if len(completed) != contract.EXPECTED_QUESTIONS:
            raise contract.ControlledAnswerError(
                f"answer inventory incomplete: {len(completed)}/{contract.EXPECTED_QUESTIONS}"
            )
        from scripts.controlled_locomo.audit_controlled_locomo_answers import audit_run  # noqa: PLC0415

        report = audit_run(args.output_dir, require_complete_manifest=False)
        if report.get("status") != "passed":
            raise contract.ControlledAnswerError("pre-final answer audit failed")
        complete_path = args.output_dir / "complete.json"
        if not complete_path.exists():
            complete = {
                "schema_version": contract.SCHEMA_VERSION,
                "status": "complete",
                "run_id": run_id,
                "completed_at": contract.utc_now(),
                "inventory": report["inventory"],
                "calls": report["calls"],
                "proxy": report["proxy"],
                "run_ledger_sha256": contract.sha256_file(
                    args.output_dir / "run_ledger.jsonl"
                ),
                "preregistration_sha256": args.preregistration_sha256,
            }
            contract.atomic_json_no_clobber(complete_path, complete)
        final_report = audit_run(args.output_dir, require_complete_manifest=True)
        contract.atomic_json_replace(args.output_dir / "audit.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.gateway_root = args.gateway_root.expanduser().resolve()
    try:
        provider_lock = flex_evidence.acquire_consumer_lock(args.gateway_root)
    except flex_evidence.EvidenceError as exc:
        raise contract.ControlledAnswerError(str(exc)) from exc
    try:
        args.gateway_contract = load_flex_gateway_contract(args.gateway_root)
        return _run(args)
    finally:
        provider_lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except contract.ControlledAnswerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
