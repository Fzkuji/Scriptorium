#!/usr/bin/env python3
"""Execute the locked LoCoMo evaluator while recording HTTP evidence.

The evaluator itself remains the sole scoring implementation.  This launcher
only intercepts its ``requests.post`` transport boundary, delegates every call
unchanged, and records hashes and provider response identity beside the normal
``eval_full.json`` output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import requests


ROOT = Path(__file__).resolve().parents[2]
LOCKED_EVALUATOR = ROOT / "scripts" / "evaluation" / "eval_full.py"
LOCKED_EVALUATOR_SHA256 = (
    "17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b"
)


class LockedEvaluatorEvidenceError(RuntimeError):
    """The locked evaluator or its external evidence ledger is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sensitive_request_values(kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    headers = kwargs.get("headers")
    if not isinstance(headers, Mapping):
        return ()
    values: set[str] = set()
    for key, value in headers.items():
        if str(key).lower() not in {
            "authorization",
            "api-key",
            "x-api-key",
            "x-auth-token",
        }:
            continue
        rendered = str(value)
        if rendered:
            values.add(rendered)
        prefix, separator, token = rendered.partition(" ")
        if separator and prefix.lower() == "bearer" and token:
            values.add(token)
    return tuple(sorted(values, key=len, reverse=True))


def _redacted_error_message(error: BaseException, sensitive: Sequence[str]) -> str:
    message = str(error)
    for value in sensitive:
        message = message.replace(value, "[REDACTED]")
    return message


def manifest_path_for(evidence_path: Path) -> Path:
    return evidence_path.with_name(f"{evidence_path.name}.manifest.json")


def response_body_dir_for(evidence_path: Path) -> Path:
    return evidence_path.with_name(f"{evidence_path.stem}.response_bodies")


def request_body_dir_for(evidence_path: Path) -> Path:
    return evidence_path.with_name(f"{evidence_path.stem}.request_bodies")


def verify_locked_evaluator(path: Path = LOCKED_EVALUATOR) -> str:
    if path.is_symlink() or path.expanduser().resolve() != LOCKED_EVALUATOR.resolve():
        raise LockedEvaluatorEvidenceError(
            f"only permitted evaluator path is {LOCKED_EVALUATOR}"
        )
    if not path.is_file() or path.is_symlink():
        raise LockedEvaluatorEvidenceError(
            f"locked LoCoMo evaluator is missing or unsafe: {path}"
        )
    actual = sha256_file(path)
    if actual != LOCKED_EVALUATOR_SHA256:
        raise LockedEvaluatorEvidenceError(
            "scripts/evaluation/eval_full.py changed; LoCoMo evaluation is blocked "
            f"(expected {LOCKED_EVALUATOR_SHA256}, got {actual})"
        )
    return actual


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LockedEvaluatorEvidenceError(f"artifact is missing or unsafe: {path}")
    resolved = path.resolve()
    return {
        "bytes": resolved.stat().st_size,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def _run_binding(
    run_dir: Path,
    *,
    answerer_model: str,
    answerer_base: str,
    require_output: bool,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    if not resolved_run_dir.is_dir() or run_dir.is_symlink():
        raise LockedEvaluatorEvidenceError(
            f"LoCoMo evaluator run directory is missing or unsafe: {run_dir}"
        )
    inputs = [
        _artifact_descriptor(path)
        for path in sorted(resolved_run_dir.glob("sample*_questions.json"))
    ]
    if not inputs:
        raise LockedEvaluatorEvidenceError(
            f"LoCoMo evaluator has no sample question inputs: {resolved_run_dir}"
        )
    output_path = resolved_run_dir / "eval_full.json"
    output = _artifact_descriptor(output_path) if output_path.is_file() else None
    if require_output and output is None:
        raise LockedEvaluatorEvidenceError(
            f"LoCoMo evaluator output is missing: {output_path}"
        )
    return {
        "answerer_base": answerer_base,
        "answerer_model": answerer_model,
        "eval_output": output,
        "inputs": inputs,
        "run_dir": str(resolved_run_dir),
    }


class JudgeRequestEvidenceRecorder:
    """Thread-safe append-only evidence for the evaluator's HTTP attempts."""

    def __init__(self, evidence_path: Path) -> None:
        self.path = evidence_path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            raise LockedEvaluatorEvidenceError(
                f"judge evidence path already exists: {self.path}"
            )
        self.request_body_dir = request_body_dir_for(self.path)
        self.response_body_dir = response_body_dir_for(self.path)
        for kind, directory in (
            ("request", self.request_body_dir),
            ("response", self.response_body_dir),
        ):
            if directory.exists() or directory.is_symlink():
                raise LockedEvaluatorEvidenceError(
                    f"judge {kind} body directory already exists: {directory}"
                )
        self.path.open("xb").close()
        self.request_body_dir.mkdir()
        self.response_body_dir.mkdir()
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._attempts_by_request: dict[str, int] = {}
        self.started = 0
        self.succeeded = 0
        self.failed = 0
        self.recording_errors: list[dict[str, Any]] = []

    def _recording_error_locked(
        self,
        *,
        stage: str,
        request_sequence: int,
        error: BaseException,
        sensitive: Sequence[str] = (),
    ) -> None:
        self.recording_errors.append(
            {
                "message": _redacted_error_message(error, sensitive),
                "request_sequence": request_sequence,
                "stage": stage,
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
            }
        )

    def _append_locked(self, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self.path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())

    def _request_metadata(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        url = str(args[0] if args else kwargs.get("url"))
        request_json = kwargs.get("json")
        request_json_bytes = canonical_json_bytes(request_json)
        request_json_sha256 = hashlib.sha256(request_json_bytes).hexdigest()
        request_json_artifact = (
            self.request_body_dir / f"{request_json_sha256}.json"
        )
        _atomic_bytes(request_json_artifact, request_json_bytes)
        requested_model = (
            request_json.get("model") if isinstance(request_json, Mapping) else None
        )
        prompt: Any = None
        if isinstance(request_json, Mapping):
            messages = request_json.get("messages")
            if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
                prompt = messages[0].get("content")
        return {
            "json_artifact": str(request_json_artifact),
            "json_bytes": len(request_json_bytes),
            "json_canonical_sha256": request_json_sha256,
            "method": "POST",
            "prompt_sha256": (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if isinstance(prompt, str)
                else None
            ),
            "request_canonical_sha256": canonical_sha256(
                {"json": request_json, "method": "POST", "url": url}
            ),
            "requested_model": requested_model,
            "url": url,
        }

    def _response_metadata(self, response: Any) -> dict[str, Any]:
        body = bytes(response.content)
        body_sha256 = hashlib.sha256(body).hexdigest()
        body_artifact = self.response_body_dir / f"{body_sha256}.body"
        _atomic_bytes(body_artifact, body)
        result: dict[str, Any] = {
            "body_artifact": str(body_artifact),
            "body_bytes": len(body),
            "body_sha256": body_sha256,
            "id": None,
            "model": None,
            "usage": None,
        }
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            result["parse_error"] = type(exc).__name__
            return result
        if isinstance(payload, Mapping):
            result["id"] = payload.get("id")
            result["model"] = payload.get("model")
            result["usage"] = payload.get("usage")
        return result

    @contextmanager
    def intercept_requests_post(self) -> Iterator[None]:
        original_post: Callable[..., Any] = requests.post

        def recorded_post(*args: Any, **kwargs: Any) -> Any:
            sensitive_values = _sensitive_request_values(kwargs)
            request_error: BaseException | None = None
            try:
                request = self._request_metadata(args, kwargs)
            except BaseException as exc:
                request_error = exc
                request = {
                    "json_artifact": None,
                    "json_bytes": None,
                    "json_canonical_sha256": None,
                    "method": "POST",
                    "prompt_sha256": None,
                    "request_canonical_sha256": None,
                    "requested_model": None,
                    "url": str(args[0] if args else kwargs.get("url")),
                }
            with self._lock:
                self._next_sequence += 1
                sequence = self._next_sequence
                request_hash = str(
                    request.get("request_canonical_sha256") or f"unavailable-{sequence}"
                )
                attempt = self._attempts_by_request.get(request_hash, 0) + 1
                self._attempts_by_request[request_hash] = attempt
                self.started += 1
                if request_error is not None:
                    self._recording_error_locked(
                        stage="request_evidence",
                        request_sequence=sequence,
                        error=request_error,
                        sensitive=sensitive_values,
                    )
                try:
                    self._append_locked(
                        {
                            "attempt": attempt,
                            "event": "judge_request_started",
                            "request": request,
                            "request_sequence": sequence,
                            "schema_version": 1,
                            "timestamp": utc_now(),
                        }
                    )
                except BaseException as exc:
                    self._recording_error_locked(
                        stage="request_ledger",
                        request_sequence=sequence,
                        error=exc,
                        sensitive=sensitive_values,
                    )
            try:
                response = original_post(*args, **kwargs)
            except BaseException as exc:
                with self._lock:
                    self.failed += 1
                    try:
                        self._append_locked(
                            {
                                "attempt": attempt,
                                "error": {
                                    "message": _redacted_error_message(
                                        exc, sensitive_values
                                    ),
                                    "type": (
                                        f"{type(exc).__module__}."
                                        f"{type(exc).__qualname__}"
                                    ),
                                },
                                "event": "judge_request_failed",
                                "request": request,
                                "request_sequence": sequence,
                                "schema_version": 1,
                                "timestamp": utc_now(),
                            }
                        )
                    except BaseException as recording_exc:
                        self._recording_error_locked(
                            stage="exception_ledger",
                            request_sequence=sequence,
                            error=recording_exc,
                            sensitive=sensitive_values,
                        )
                raise
            with self._lock:
                self.succeeded += 1
                try:
                    response_evidence = self._response_metadata(response)
                    self._append_locked(
                        {
                            "attempt": attempt,
                            "event": "judge_request_succeeded",
                            "request": request,
                            "request_sequence": sequence,
                            "response": response_evidence,
                            "schema_version": 1,
                            "timestamp": utc_now(),
                        }
                    )
                except BaseException as exc:
                    self._recording_error_locked(
                        stage="response_evidence",
                        request_sequence=sequence,
                        error=exc,
                        sensitive=sensitive_values,
                    )
                    try:
                        self._append_locked(
                            {
                                "attempt": attempt,
                                "error": {
                                    "message": _redacted_error_message(
                                        exc, sensitive_values
                                    ),
                                    "type": (
                                        f"{type(exc).__module__}."
                                        f"{type(exc).__qualname__}"
                                    ),
                                },
                                "event": "judge_evidence_failed",
                                "request": request,
                                "request_sequence": sequence,
                                "schema_version": 1,
                                "timestamp": utc_now(),
                            }
                        )
                    except BaseException as recording_exc:
                        self._recording_error_locked(
                            stage="response_evidence_error_ledger",
                            request_sequence=sequence,
                            error=recording_exc,
                            sensitive=sensitive_values,
                        )
            return response

        requests.post = recorded_post
        try:
            yield
        finally:
            requests.post = original_post

    def write_manifest(
        self,
        *,
        evaluator_path: Path,
        sha256_before: str,
        sha256_after: str,
        run_binding: Mapping[str, Any],
        status: str,
    ) -> Path:
        manifest = {
            "evaluator": {
                "path": str(evaluator_path.resolve()),
                "sha256_after": sha256_after,
                "sha256_before": sha256_before,
            },
            "evidence": {
                "path": str(self.path),
                "sha256": sha256_file(self.path),
            },
            "launcher": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "requests": {
                "failed": self.failed,
                "started": self.started,
                "succeeded": self.succeeded,
            },
            "recording_errors": self.recording_errors,
            "request_bodies": {
                "files": len(list(self.request_body_dir.glob("*.json"))),
                "path": str(self.request_body_dir),
            },
            "response_bodies": {
                "files": len(list(self.response_body_dir.glob("*.body"))),
                "path": str(self.response_body_dir),
            },
            "run_binding": dict(run_binding),
            "schema_version": 1,
            "status": status,
        }
        path = manifest_path_for(self.path)
        _atomic_json(path, manifest)
        return path


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedEvaluatorEvidenceError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LockedEvaluatorEvidenceError(f"JSON artifact is not an object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def audit_evidence(evidence_path: Path) -> dict[str, Any]:
    """Independently validate pairing and provenance of every HTTP attempt."""

    evidence_path = evidence_path.expanduser().resolve()
    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise LockedEvaluatorEvidenceError(
            f"judge evidence is missing or unsafe: {evidence_path}"
        )
    manifest_path = manifest_path_for(evidence_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise LockedEvaluatorEvidenceError(
            f"judge evidence manifest is missing or unsafe: {manifest_path}"
        )
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise LockedEvaluatorEvidenceError("judge evidence manifest is not complete")
    if manifest.get("recording_errors") != []:
        raise LockedEvaluatorEvidenceError("judge evidence contains recording errors")
    run_binding = manifest.get("run_binding")
    if not isinstance(run_binding, Mapping):
        raise LockedEvaluatorEvidenceError("LoCoMo evaluator run binding is invalid")
    run_dir_value = run_binding.get("run_dir")
    answerer_model = run_binding.get("answerer_model")
    answerer_base = run_binding.get("answerer_base")
    if not all(
        isinstance(value, str) and value
        for value in (run_dir_value, answerer_model, answerer_base)
    ):
        raise LockedEvaluatorEvidenceError("LoCoMo evaluator run binding is invalid")
    expected_run_binding = _run_binding(
        Path(str(run_dir_value)),
        answerer_model=str(answerer_model),
        answerer_base=str(answerer_base),
        require_output=True,
    )
    if dict(run_binding) != expected_run_binding:
        raise LockedEvaluatorEvidenceError("LoCoMo evaluator run binding differs")
    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator != {
        "path": str(LOCKED_EVALUATOR.resolve()),
        "sha256_after": LOCKED_EVALUATOR_SHA256,
        "sha256_before": LOCKED_EVALUATOR_SHA256,
    }:
        raise LockedEvaluatorEvidenceError("locked evaluator provenance differs")
    verify_locked_evaluator()
    evidence = manifest.get("evidence")
    if not isinstance(evidence, Mapping) or evidence != {
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
    }:
        raise LockedEvaluatorEvidenceError("judge evidence hash or path differs")
    launcher = manifest.get("launcher")
    launcher_path = Path(__file__).resolve()
    if not isinstance(launcher, Mapping) or launcher != {
        "path": str(launcher_path),
        "sha256": sha256_file(launcher_path),
    }:
        raise LockedEvaluatorEvidenceError("judge evidence launcher provenance differs")
    request_body_dir = request_body_dir_for(evidence_path)
    request_body_manifest = manifest.get("request_bodies")
    if (
        not request_body_dir.is_dir()
        or request_body_dir.is_symlink()
        or not isinstance(request_body_manifest, Mapping)
        or request_body_manifest.get("path") != str(request_body_dir)
    ):
        raise LockedEvaluatorEvidenceError("judge request body provenance differs")
    response_body_dir = response_body_dir_for(evidence_path)
    response_body_manifest = manifest.get("response_bodies")
    if (
        not response_body_dir.is_dir()
        or response_body_dir.is_symlink()
        or not isinstance(response_body_manifest, Mapping)
        or response_body_manifest.get("path") != str(response_body_dir)
    ):
        raise LockedEvaluatorEvidenceError("judge response body provenance differs")

    rows: list[dict[str, Any]] = []
    try:
        raw_lines = evidence_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LockedEvaluatorEvidenceError("judge evidence cannot be read") from exc
    for line_number, raw in enumerate(raw_lines, start=1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LockedEvaluatorEvidenceError(
                f"judge evidence line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict) or row.get("schema_version") != 1:
            raise LockedEvaluatorEvidenceError(
                f"judge evidence line {line_number} has invalid schema"
            )
        rows.append(row)

    starts: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    attempts_by_hash: dict[str, int] = {}
    request_body_paths: set[Path] = set()
    for row in rows:
        sequence = row.get("request_sequence")
        attempt = row.get("attempt")
        request = row.get("request")
        if not isinstance(sequence, int) or sequence < 1 or not isinstance(attempt, int):
            raise LockedEvaluatorEvidenceError("judge evidence sequence or attempt is invalid")
        if not isinstance(request, Mapping):
            raise LockedEvaluatorEvidenceError("judge evidence request is invalid")
        request_hash = request.get("request_canonical_sha256")
        request_json_sha256 = request.get("json_canonical_sha256")
        request_artifact_value = request.get("json_artifact")
        expected_request_artifact = (
            request_body_dir / f"{request_json_sha256}.json"
            if _is_sha256(request_json_sha256)
            else None
        )
        if (
            not _is_sha256(request_hash)
            or not _is_sha256(request_json_sha256)
            or not _is_sha256(request.get("prompt_sha256"))
            or not isinstance(request_artifact_value, str)
            or Path(request_artifact_value) != expected_request_artifact
            or expected_request_artifact is None
            or not expected_request_artifact.is_file()
            or expected_request_artifact.is_symlink()
            or request.get("method") != "POST"
            or request.get("url")
            != "https://openrouter.ai/api/v1/chat/completions"
            or request.get("requested_model") != "openai/gpt-4o-mini"
        ):
            raise LockedEvaluatorEvidenceError(
                "request metadata differs from raw artifact"
            )
        request_json_bytes = expected_request_artifact.read_bytes()
        try:
            request_json = json.loads(request_json_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LockedEvaluatorEvidenceError(
                "request metadata differs from raw artifact"
            ) from exc
        messages = request_json.get("messages") if isinstance(request_json, Mapping) else None
        prompt = (
            messages[0].get("content")
            if isinstance(messages, list)
            and messages
            and isinstance(messages[0], Mapping)
            else None
        )
        expected_request = {
            "json_artifact": str(expected_request_artifact),
            "json_bytes": len(request_json_bytes),
            "json_canonical_sha256": hashlib.sha256(request_json_bytes).hexdigest(),
            "method": "POST",
            "prompt_sha256": (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if isinstance(prompt, str)
                else None
            ),
            "request_canonical_sha256": canonical_sha256(
                {
                    "json": request_json,
                    "method": "POST",
                    "url": "https://openrouter.ai/api/v1/chat/completions",
                }
            ),
            "requested_model": (
                request_json.get("model")
                if isinstance(request_json, Mapping)
                else None
            ),
            "url": "https://openrouter.ai/api/v1/chat/completions",
        }
        if (
            request_json_bytes != canonical_json_bytes(request_json)
            or dict(request) != expected_request
        ):
            raise LockedEvaluatorEvidenceError(
                "request metadata differs from raw artifact"
            )
        request_body_paths.add(expected_request_artifact)
        event = row.get("event")
        if event == "judge_request_started":
            if sequence in starts:
                raise LockedEvaluatorEvidenceError("judge request sequence is duplicated")
            expected_attempt = attempts_by_hash.get(str(request_hash), 0) + 1
            if attempt != expected_attempt:
                raise LockedEvaluatorEvidenceError("judge request attempt sequence differs")
            attempts_by_hash[str(request_hash)] = attempt
            starts[sequence] = row
        elif event in {"judge_request_succeeded", "judge_request_failed"}:
            if sequence in outcomes:
                raise LockedEvaluatorEvidenceError("judge request has duplicate outcomes")
            outcomes[sequence] = row
        else:
            raise LockedEvaluatorEvidenceError("judge evidence event type is invalid")

    if set(starts) != set(outcomes):
        raise LockedEvaluatorEvidenceError("judge request start/outcome pairing differs")
    actual_request_body_paths = set(request_body_dir.glob("*.json"))
    if (
        actual_request_body_paths != request_body_paths
        or request_body_manifest.get("files") != len(actual_request_body_paths)
    ):
        raise LockedEvaluatorEvidenceError("judge request body inventory differs")
    response_ids: list[str] = []
    response_body_paths: set[Path] = set()
    failed_request_hashes: set[str] = set()
    successful_request_hashes: set[str] = set()
    failed = 0
    succeeded = 0
    for sequence, start in starts.items():
        outcome = outcomes[sequence]
        if (
            outcome.get("attempt") != start.get("attempt")
            or outcome.get("request") != start.get("request")
        ):
            raise LockedEvaluatorEvidenceError("judge request outcome identity differs")
        if outcome["event"] == "judge_request_failed":
            error = outcome.get("error")
            if (
                not isinstance(error, Mapping)
                or not isinstance(error.get("type"), str)
                or not error.get("type")
                or not isinstance(error.get("message"), str)
            ):
                raise LockedEvaluatorEvidenceError("judge request exception is invalid")
            failed_request_hashes.add(
                str(start["request"]["request_canonical_sha256"])
            )
            failed += 1
            continue
        response = outcome.get("response")
        body_sha256 = response.get("body_sha256") if isinstance(response, Mapping) else None
        body_artifact_value = (
            response.get("body_artifact") if isinstance(response, Mapping) else None
        )
        expected_body_artifact = (
            response_body_dir / f"{body_sha256}.body"
            if _is_sha256(body_sha256)
            else None
        )
        if (
            not isinstance(response, Mapping)
            or not isinstance(body_artifact_value, str)
            or Path(body_artifact_value) != expected_body_artifact
            or expected_body_artifact is None
            or not expected_body_artifact.is_file()
            or expected_body_artifact.is_symlink()
            or not isinstance(response.get("body_bytes"), int)
            or response.get("body_bytes", -1) < 0
            or sha256_file(expected_body_artifact) != body_sha256
            or expected_body_artifact.stat().st_size != response.get("body_bytes")
            or not isinstance(response.get("id"), str)
            or not response.get("id")
            or not isinstance(response.get("model"), str)
            or not response.get("model")
            or not isinstance(response.get("usage"), Mapping)
        ):
            raise LockedEvaluatorEvidenceError("judge response evidence is incomplete")
        raw_response_body = expected_body_artifact.read_bytes()
        try:
            raw_response = json.loads(raw_response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LockedEvaluatorEvidenceError(
                "response metadata differs from raw artifact"
            ) from exc
        expected_response = {
            "body_artifact": str(expected_body_artifact),
            "body_bytes": len(raw_response_body),
            "body_sha256": hashlib.sha256(raw_response_body).hexdigest(),
            "id": raw_response.get("id") if isinstance(raw_response, Mapping) else None,
            "model": (
                raw_response.get("model") if isinstance(raw_response, Mapping) else None
            ),
            "usage": (
                raw_response.get("usage") if isinstance(raw_response, Mapping) else None
            ),
        }
        if dict(response) != expected_response:
            raise LockedEvaluatorEvidenceError(
                "response metadata differs from raw artifact"
            )
        choices = (
            raw_response.get("choices") if isinstance(raw_response, Mapping) else None
        )
        message = (
            choices[0].get("message")
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], Mapping)
            else None
        )
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise LockedEvaluatorEvidenceError(
                "response body lacks evaluator content path"
            )
        response_body_paths.add(expected_body_artifact)
        successful_request_hashes.add(
            str(start["request"]["request_canonical_sha256"])
        )
        response_ids.append(str(response["id"]))
        succeeded += 1
    if len(response_ids) != len(set(response_ids)):
        raise LockedEvaluatorEvidenceError("judge response ID is duplicated")
    actual_body_paths = set(response_body_dir.glob("*.body"))
    if (
        actual_body_paths != response_body_paths
        or response_body_manifest.get("files") != len(actual_body_paths)
    ):
        raise LockedEvaluatorEvidenceError("judge response body inventory differs")
    expected_counts = {
        "failed": failed,
        "started": len(starts),
        "succeeded": succeeded,
    }
    if manifest.get("requests") != expected_counts:
        raise LockedEvaluatorEvidenceError("judge evidence counts differ from manifest")
    return {
        "evaluator_sha256": LOCKED_EVALUATOR_SHA256,
        "failed_attempts": failed,
        "failed_only_request_hashes": sorted(
            failed_request_hashes - successful_request_hashes
        ),
        "requests_started": len(starts),
        "response_ids": response_ids,
        "status": "passed",
        "successful_attempts": succeeded,
    }


def _mark_audit_failed(
    evidence_path: Path,
    error: BaseException,
    *,
    report: Mapping[str, Any] | None = None,
) -> None:
    path = manifest_path_for(evidence_path.expanduser().resolve())
    manifest = _read_json_object(path)
    manifest["audit_error"] = {
        "message": str(error),
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
    }
    if report is not None:
        manifest["audit_report"] = dict(report)
    manifest["status"] = "audit_failed"
    _atomic_json(path, manifest)


def run_locked_evaluator(
    evaluator_args: Sequence[str],
    *,
    evidence_path: Path,
    evaluator_path: Path = LOCKED_EVALUATOR,
) -> dict[str, Any]:
    if len(evaluator_args) != 5:
        raise LockedEvaluatorEvidenceError(
            "locked evaluator requires exactly five positional arguments"
        )
    run_dir = Path(str(evaluator_args[0]))
    answerer_model = str(evaluator_args[1])
    answerer_base = str(evaluator_args[2])
    initial_run_binding = _run_binding(
        run_dir,
        answerer_model=answerer_model,
        answerer_base=answerer_base,
        require_output=False,
    )
    sha256_before = verify_locked_evaluator(evaluator_path)
    recorder = JudgeRequestEvidenceRecorder(evidence_path)
    original_argv = sys.argv[:]
    answerer_environment = {
        key: os.environ.get(key)
        for key in ("ANSWERER_MODEL", "ANSWERER_BASE", "ANSWERER_KEY")
    }
    completed = False
    sha256_after = sha256_before
    try:
        sys.argv = [str(evaluator_path), *map(str, evaluator_args)]
        with recorder.intercept_requests_post():
            runpy.run_path(str(evaluator_path), run_name="__main__")
        completed = True
    finally:
        sys.argv = original_argv
        for key, value in answerer_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sha256_after = sha256_file(evaluator_path)
        recorder.write_manifest(
            evaluator_path=evaluator_path,
            sha256_before=sha256_before,
            sha256_after=sha256_after,
            run_binding={
                **initial_run_binding,
                "eval_output": (
                    _artifact_descriptor(run_dir.expanduser().resolve() / "eval_full.json")
                    if (run_dir.expanduser().resolve() / "eval_full.json").is_file()
                    else None
                ),
            },
            status=(
                "complete"
                if (
                    completed
                    and sha256_after == LOCKED_EVALUATOR_SHA256
                    and not recorder.recording_errors
                )
                else (
                    "evidence_failed"
                    if completed and recorder.recording_errors
                    else "failed"
                )
            ),
        )
    if sha256_after != LOCKED_EVALUATOR_SHA256:
        raise LockedEvaluatorEvidenceError(
            "scripts/evaluation/eval_full.py changed while the evaluator was executing"
        )
    if recorder.recording_errors:
        raise LockedEvaluatorEvidenceError(
            "judge evidence recording failed; evaluator output was left unchanged"
        )
    try:
        report = audit_evidence(recorder.path)
    except LockedEvaluatorEvidenceError as exc:
        _mark_audit_failed(recorder.path, exc)
        raise
    failed_only = report["failed_only_request_hashes"]
    if failed_only:
        error = LockedEvaluatorEvidenceError(
            "judge request exhausted all evaluator attempts: "
            f"{len(failed_only)} request(s)"
        )
        _mark_audit_failed(recorder.path, error, report=report)
        raise error
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked LoCoMo evaluator with external HTTP evidence"
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--answerer-key-env", default="ANSWERER_KEY")
    parser.add_argument("--openrouter-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("run_dir")
    parser.add_argument("answerer_model")
    parser.add_argument("answerer_base")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    answerer_key = os.environ.get(args.answerer_key_env, "")
    openrouter_key = os.environ.get(args.openrouter_key_env, "")
    if not answerer_key:
        raise LockedEvaluatorEvidenceError(
            f"answerer key environment variable is unset: {args.answerer_key_env}"
        )
    if not openrouter_key:
        raise LockedEvaluatorEvidenceError(
            "OpenRouter key environment variable is unset: "
            f"{args.openrouter_key_env}"
        )
    run_locked_evaluator(
        [
            args.run_dir,
            args.answerer_model,
            args.answerer_base,
            answerer_key,
            openrouter_key,
        ],
        evidence_path=args.evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
