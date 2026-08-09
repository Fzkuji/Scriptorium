"""Enforceable visible-token accounting for controlled retrieval experiments.

The gate sits at the final text boundary before a tool or source-resolution
result is delivered to the answer model.  It does not count provider prompt or
completion tokens.  Provider-reported usage is stored separately at
finalization time.

Local token counts are always approximations with an explicit tokenizer
identity.  They must never be represented as provider-exact counts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "visible-token-budget-v1"
TRUNCATION_ALGORITHM = "encoded-prefix-decode-recount-v1"
ZERO_HASH = "0" * 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_fingerprint(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"sha256": sha256_bytes(encoded), "utf8_bytes": len(encoded)}


def _record_hash(record_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(record_without_hash).encode("utf-8"))


def _path_identity(path: Path) -> str:
    return unicodedata.normalize(
        "NFC",
        os.path.normcase(os.path.realpath(os.path.abspath(path))),
    ).casefold()


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"output path contains symlink component: {current}")


def _safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(metadata or {})
    # Round-trip now so an unserializable or non-finite value cannot corrupt a
    # trace after earlier records have already been committed.
    return json.loads(canonical_json(value))


@dataclass(frozen=True)
class MemorySnapshot:
    """Content fingerprint for a file, directory, or explicit byte value."""

    descriptor: dict[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.descriptor["sha256"])


def snapshot_memory_bytes(data: bytes, *, label: str = "memory") -> MemorySnapshot:
    return MemorySnapshot(
        {
            "kind": "bytes",
            "label": label,
            "sha256": sha256_bytes(data),
            "byte_count": len(data),
        }
    )


def _directory_digest(entries: list[dict[str, Any]]) -> str:
    digest_input = [
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "byte_count": entry["byte_count"],
        }
        for entry in entries
    ]
    return sha256_bytes(canonical_json(digest_input).encode("utf-8"))


def snapshot_memory_path(path: str | os.PathLike[str]) -> MemorySnapshot:
    """Hash a memory artifact without following symlinks.

    Directory hashes include relative paths, per-file hashes, and byte counts.
    This makes renames observable and lets the offline auditor recompute the
    tree root from the recorded entry inventory.  The auditor can additionally
    compare the descriptor with a live path when a preserved snapshot exists.
    """

    root = Path(path)
    if root.is_symlink():
        raise ValueError(f"memory snapshot path must not be a symlink: {root}")
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        data = root.read_bytes()
        return MemorySnapshot(
            {
                "kind": "file",
                "path": str(root.resolve()),
                "sha256": sha256_bytes(data),
                "byte_count": len(data),
            }
        )
    if not root.is_dir():
        raise ValueError(f"unsupported memory snapshot type: {root}")

    entries: list[dict[str, Any]] = []
    for item in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        if item.is_symlink():
            raise ValueError(f"memory snapshot tree contains symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"memory snapshot tree contains non-file: {item}")
        data = item.read_bytes()
        entries.append(
            {
                "path": item.relative_to(root).as_posix(),
                "sha256": sha256_bytes(data),
                "byte_count": len(data),
            }
        )
    return MemorySnapshot(
        {
            "kind": "directory",
            "path": str(root.resolve()),
            "sha256": _directory_digest(entries),
            "byte_count": sum(entry["byte_count"] for entry in entries),
            "file_count": len(entries),
            "entries": entries,
        }
    )


class TokenCounter:
    """A reversible local tokenizer with a fully recorded identity."""

    def __init__(self, *, identity: Mapping[str, Any], encoder: Any, decoder: Any):
        self.identity = json.loads(canonical_json(dict(identity)))
        if self.identity.get("provider_exact") is not False:
            raise ValueError("local token counter must declare provider_exact=false")
        self._encode = encoder
        self._decode = decoder

    @classmethod
    def utf8_bytes(
        cls,
        *,
        requested_model: str,
        fallback_reason: str = "explicit_utf8_byte_fallback",
    ) -> TokenCounter:
        identity = {
            "implementation": "builtin_utf8_bytes",
            "implementation_version": "1",
            "encoding_name": "utf8_bytes_v1",
            "requested_model": requested_model,
            "resolution": "explicit_byte_fallback",
            "fallback_reason": fallback_reason,
            "provider_exact": False,
            "counting_note": (
                "One token equals one UTF-8 byte. This is deterministic budget "
                "accounting, not a provider tokenizer estimate."
            ),
        }
        return cls(
            identity=identity,
            encoder=lambda text: list(text.encode("utf-8")),
            decoder=lambda token_ids: bytes(token_ids).decode("utf-8", errors="ignore"),
        )

    @classmethod
    def resolve(
        cls,
        *,
        requested_model: str,
        fallback_encoding: str = "o200k_base",
        allow_byte_fallback: bool = False,
    ) -> TokenCounter:
        """Resolve tiktoken model mapping, then an explicit named fallback.

        If tiktoken is unavailable, byte fallback is used only when explicitly
        enabled.  This avoids silently changing the unit of a formal budget.
        """

        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as exc:
            if not allow_byte_fallback:
                raise RuntimeError(
                    "tiktoken is unavailable; install it or explicitly enable "
                    "the UTF-8 byte fallback"
                ) from exc
            return cls.utf8_bytes(
                requested_model=requested_model,
                fallback_reason="tiktoken_not_installed",
            )

        version = importlib.metadata.version("tiktoken")
        try:
            encoding = tiktoken.encoding_for_model(requested_model)
            resolution = "tiktoken_model_mapping"
            fallback_reason = None
        except KeyError:
            encoding = tiktoken.get_encoding(fallback_encoding)
            resolution = "tiktoken_named_fallback"
            fallback_reason = "requested_model_not_in_tiktoken_mapping"

        identity = {
            "implementation": "tiktoken",
            "implementation_version": version,
            "encoding_name": encoding.name,
            "requested_model": requested_model,
            "resolution": resolution,
            "fallback_encoding": fallback_encoding,
            "fallback_reason": fallback_reason,
            "provider_exact": False,
            "counting_note": (
                "Local tiktoken count used for an enforceable experiment budget; "
                "it is not a provider-reported exact count."
            ),
        }
        return cls(identity=identity, encoder=encoding.encode, decoder=encoding.decode)

    @classmethod
    def from_identity(cls, identity: Mapping[str, Any]) -> TokenCounter:
        implementation = identity.get("implementation")
        if implementation == "builtin_utf8_bytes":
            expected = cls.utf8_bytes(
                requested_model=str(identity.get("requested_model", "")),
                fallback_reason=str(identity.get("fallback_reason", "")),
            )
            if expected.identity != dict(identity):
                raise ValueError("recorded UTF-8 byte tokenizer identity is invalid")
            return expected
        if implementation != "tiktoken":
            raise ValueError(
                f"unsupported tokenizer implementation: {implementation!r}"
            )

        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("tiktoken is required to audit this trace") from exc
        installed = importlib.metadata.version("tiktoken")
        recorded = str(identity.get("implementation_version"))
        if installed != recorded:
            raise RuntimeError(
                f"tiktoken version mismatch: trace={recorded}, installed={installed}"
            )
        encoding = tiktoken.get_encoding(str(identity.get("encoding_name")))
        # Some frozen benchmark contracts deliberately treat strings that look
        # like tokenizer control tokens as ordinary dataset text.  Preserve
        # that recorded policy during independent reconstruction instead of
        # reverting to tiktoken's default special-token rejection.
        if identity.get("disallowed_special") == []:

            def encoder(text: str) -> list[int]:
                return list(encoding.encode(text, disallowed_special=()))
        else:
            encoder = encoding.encode
        return cls(identity=identity, encoder=encoder, decoder=encoding.decode)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("visible content must be a rendered string")
        return list(self._encode(text))

    def decode(self, token_ids: Iterable[int]) -> str:
        return str(self._decode(list(token_ids)))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, limit: int) -> str:
        """Return a deterministic decoded token prefix within ``limit``."""

        if limit < 0:
            raise ValueError("token limit must be non-negative")
        token_ids = self.encode(text)
        if len(token_ids) <= limit:
            return text
        candidate_ids = token_ids[:limit]
        while True:
            try:
                candidate = self.decode(candidate_ids)
                is_valid = (
                    (bool(candidate) or not candidate_ids)
                    and text.startswith(candidate)
                    and self.count(candidate) <= limit
                )
            except Exception:  # noqa: BLE001 - invalid tokenizer boundary
                is_valid = False
                candidate = ""
            if is_valid:
                return candidate
            if not candidate_ids:
                return ""
            candidate_ids.pop()


@dataclass(frozen=True)
class DeliveryResult:
    event_id: str
    kind: str
    decision: str
    delivered_text: str | None
    delivered_tokens: int
    cumulative_visible_tokens: int
    exhausted: bool
    finalizer_required: bool

    @property
    def may_deliver(self) -> bool:
        return self.delivered_text is not None


class VisibleTokenBudgetGate:
    """Durable hard gate for visible retrieval content.

    A new trace is required for each independently budgeted retrieval.  The
    trace is append-only, hash chained, fsynced, and sealed by a separate
    manifest on finalization.
    """

    def __init__(
        self,
        *,
        trace_path: str | os.PathLike[str],
        run_id: str,
        configured_budget_tokens: int,
        tokenizer: TokenCounter,
        memory_before: MemorySnapshot,
        overflow_policy: str = "truncate",
        manifest_path: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if configured_budget_tokens < 0:
            raise ValueError("configured budget must be non-negative")
        if overflow_policy not in {"truncate", "reject"}:
            raise ValueError("overflow_policy must be 'truncate' or 'reject'")
        if not run_id:
            raise ValueError("run_id must be non-empty")

        self.trace_path = Path(trace_path)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.trace_path.with_suffix(self.trace_path.suffix + ".manifest.json")
        )
        self._lock_path = self.trace_path.with_suffix(self.trace_path.suffix + ".lock")
        self._lock_fd: int | None = None
        self._owns_lock = False
        self._file: Any = None
        self._mutex = threading.RLock()
        self._finalized = False
        self._closed = False
        self._validate_output_paths()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Detect a parent changed to a symlink during directory creation.
        self._validate_output_paths()
        if self.trace_path.exists() or self.manifest_path.exists():
            raise FileExistsError("token-budget trace or manifest already exists")

        try:
            self._lock_fd = os.open(
                self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"token-budget trace is already locked: {self._lock_path}"
            ) from exc
        self._owns_lock = True
        self._tokenizer = tokenizer
        self._run_id = run_id
        self._budget = configured_budget_tokens
        self._overflow_policy = overflow_policy
        self._memory_before = memory_before
        self._last_hash = ZERO_HASH
        self._sequence = 0
        self._event_ids: set[str] = set()
        self._cumulative = 0
        self._source_cumulative = 0
        self._exhausted = configured_budget_tokens == 0
        self._exhaustion_sequence: int | None = 0 if self._exhausted else None
        header = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "header",
            "sequence": 0,
            "run_id": run_id,
            "created_at": _utc_now(),
            "config": {
                "configured_budget_tokens": configured_budget_tokens,
                "overflow_policy": overflow_policy,
                "tokenizer": tokenizer.identity,
                "accounting_scope": (
                    "rendered tool_result and source_resolution text delivered "
                    "across the model boundary"
                ),
                "provider_exact": False,
                "truncation_algorithm": TRUNCATION_ALGORITHM,
                "overflow_ends_retrieval": True,
                "finalizer_required": True,
            },
            "memory_before": memory_before.descriptor,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "metadata": _safe_metadata(metadata),
        }
        try:
            os.write(self._lock_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(self._lock_fd)
            self._file = self.trace_path.open("x", encoding="utf-8")
            self._append(header)
        except Exception:
            if self._file is not None and not self._file.closed:
                self._file.close()
            self._closed = True
            self._release_lock()
            raise

    def _validate_output_paths(self) -> None:
        paths = {
            "trace": self.trace_path,
            "manifest": self.manifest_path,
            "lock": self._lock_path,
        }
        for name, path in paths.items():
            _reject_symlink_components(path)
            if path.is_symlink():
                raise ValueError(f"{name} path must not be a symlink: {path}")
        names = list(paths)
        for index, left_name in enumerate(names):
            left = paths[left_name]
            for right_name in names[index + 1 :]:
                right = paths[right_name]
                if _path_identity(left) == _path_identity(right):
                    raise ValueError(
                        f"output path collision: {left_name} and {right_name}"
                    )
                if left.exists() and right.exists() and os.path.samefile(left, right):
                    raise ValueError(
                        f"output inode collision: {left_name} and {right_name}"
                    )

    @property
    def cumulative_visible_tokens(self) -> int:
        return self._cumulative

    @property
    def remaining_tokens(self) -> int:
        return self._budget - self._cumulative

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def finalizer_required(self) -> bool:
        return not self._finalized

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        if self._file is None or self._file.closed:
            raise RuntimeError("token-budget trace is not writable")
        record["previous_record_hash"] = self._last_hash
        record["record_hash"] = _record_hash(record)
        self._file.write(canonical_json(record) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._last_hash = str(record["record_hash"])
        return record

    def deliver_tool_result(
        self,
        *,
        event_id: str,
        raw_text: str,
        tool_name: str,
        tool_call_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DeliveryResult:
        combined = dict(metadata or {})
        combined.update({"tool_name": tool_name, "tool_call_id": tool_call_id})
        return self.deliver(
            event_id=event_id,
            kind="tool_result",
            raw_text=raw_text,
            metadata=combined,
        )

    def deliver_source_resolution(
        self,
        *,
        event_id: str,
        raw_text: str,
        source_ids: list[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> DeliveryResult:
        combined = dict(metadata or {})
        combined["source_ids"] = list(source_ids)
        return self.deliver(
            event_id=event_id,
            kind="source_resolution",
            raw_text=raw_text,
            metadata=combined,
        )

    def deliver(
        self,
        *,
        event_id: str,
        kind: str,
        raw_text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> DeliveryResult:
        with self._mutex:
            if self._finalized or self._closed:
                raise RuntimeError("token-budget gate is finalized or closed")
            if kind not in {"tool_result", "source_resolution"}:
                raise ValueError("kind must be tool_result or source_resolution")
            if not event_id or event_id in self._event_ids:
                raise ValueError(f"event_id must be unique and non-empty: {event_id!r}")
            if not isinstance(raw_text, str):
                raise TypeError("raw_text must be the exact rendered string")
            self._event_ids.add(event_id)
            self._sequence += 1

            raw_tokens = self._tokenizer.count(raw_text)
            budget_before = self.remaining_tokens
            delivered_text: str | None
            decision: str

            if self._exhausted:
                delivered_text = None
                decision = "rejected_budget_exhausted"
            elif raw_tokens <= budget_before:
                delivered_text = raw_text
                decision = "delivered"
            elif self._overflow_policy == "reject":
                delivered_text = None
                decision = "rejected_overflow"
                self._exhausted = True
            else:
                candidate = self._tokenizer.truncate(raw_text, budget_before)
                if candidate:
                    delivered_text = candidate
                    decision = "truncated"
                else:
                    delivered_text = None
                    decision = "rejected_truncation_empty"
                # An overflow terminates retrieval even if decoding a valid
                # Unicode prefix leaves a small remainder under byte fallback.
                self._exhausted = True

            delivered_tokens = (
                self._tokenizer.count(delivered_text)
                if delivered_text is not None
                else 0
            )
            if delivered_tokens > budget_before:
                raise RuntimeError(
                    "gate attempted to deliver beyond the remaining budget"
                )
            self._cumulative += delivered_tokens
            if kind == "source_resolution":
                self._source_cumulative += delivered_tokens
            if self._cumulative == self._budget:
                self._exhausted = True
            if self._exhausted and self._exhaustion_sequence is None:
                self._exhaustion_sequence = self._sequence

            raw_fp = text_fingerprint(raw_text)
            delivered_fp = (
                text_fingerprint(delivered_text) if delivered_text is not None else None
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "delivery",
                "sequence": self._sequence,
                "run_id": self._run_id,
                "created_at": _utc_now(),
                "event_id": event_id,
                "kind": kind,
                "raw": {
                    "text": raw_text,
                    **raw_fp,
                    "tokens": raw_tokens,
                },
                "delivered": (
                    {
                        "text": delivered_text,
                        **(delivered_fp or {}),
                        "tokens": delivered_tokens,
                    }
                    if delivered_text is not None
                    else None
                ),
                "decision": decision,
                "budget_before_tokens": budget_before,
                "budget_after_tokens": self.remaining_tokens,
                "cumulative_visible_tokens": self._cumulative,
                "source_resolution_tokens": (
                    delivered_tokens if kind == "source_resolution" else 0
                ),
                "cumulative_source_resolution_tokens": self._source_cumulative,
                "exhausted": self._exhausted,
                "finalizer_required": True,
                "metadata": _safe_metadata(metadata),
            }
            self._append(record)
            return DeliveryResult(
                event_id=event_id,
                kind=kind,
                decision=decision,
                delivered_text=delivered_text,
                delivered_tokens=delivered_tokens,
                cumulative_visible_tokens=self._cumulative,
                exhausted=self._exhausted,
                finalizer_required=True,
            )

    def finalize(
        self,
        *,
        memory_after: MemorySnapshot,
        actual_model_usage: Mapping[str, Any],
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write the required finalizer and seal the trace manifest."""

        with self._mutex:
            if self._finalized or self._closed:
                raise RuntimeError("token-budget gate is already finalized or closed")
            usage = _safe_metadata(actual_model_usage)
            self._sequence += 1
            final_reason = reason or (
                "budget_exhausted" if self._exhausted else "normal"
            )
            summary = {
                "configured_budget_tokens": self._budget,
                "cumulative_visible_tokens": self._cumulative,
                "cumulative_source_resolution_tokens": self._source_cumulative,
                "remaining_tokens": self.remaining_tokens,
                "exhausted": self._exhausted,
                "exhaustion_sequence": self._exhaustion_sequence,
                "delivery_record_count": len(self._event_ids),
            }
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "finalizer",
                "sequence": self._sequence,
                "run_id": self._run_id,
                "created_at": _utc_now(),
                "reason": final_reason,
                "memory_before_sha256": self._memory_before.sha256,
                "memory_after": memory_after.descriptor,
                "memory_changed": memory_after.sha256 != self._memory_before.sha256,
                "actual_model_usage": usage,
                "summary": summary,
                "metadata": _safe_metadata(metadata),
            }
            self._append(record)
            self._file.close()

            trace_data = self.trace_path.read_bytes()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "run_id": self._run_id,
                "trace_file": self.trace_path.name,
                "trace_sha256": sha256_bytes(trace_data),
                "record_count": self._sequence + 1,
                "final_record_hash": self._last_hash,
                "tokenizer": self._tokenizer.identity,
                "configured_budget_tokens": self._budget,
                "memory_before_sha256": self._memory_before.sha256,
                "memory_after_sha256": memory_after.sha256,
                "summary": summary,
                "sealed_at": _utc_now(),
            }
            try:
                _atomic_write_json(self.manifest_path, manifest)
            except Exception:
                # The trace contains a finalizer but cannot be accepted without
                # its independently hashed manifest.
                self._closed = True
                self._release_lock()
                raise
            self._finalized = True
            self._closed = True
            self._release_lock()
            return manifest

    def close_incomplete(self) -> None:
        """Release resources without creating a complete manifest.

        This is intended for exception cleanup.  The independent auditor will
        reject the incomplete trace, so callers cannot mistake it for a usable
        formal result.
        """

        with self._mutex:
            if self._closed or self._finalized:
                return
            if self._file is not None and not self._file.closed:
                self._file.close()
            self._closed = True
            self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        if self._owns_lock:
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
            self._owns_lock = False

    def __enter__(self) -> VisibleTokenBudgetGate:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._finalized:
            self.close_incomplete()
        return False


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is an atomic no-clobber create on the same
        # filesystem. It cannot overwrite a file or symlink created by another
        # process after the gate's initial path validation.
        os.link(temp_name, path)
        published = True
        os.unlink(temp_name)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def copy_snapshot(source: Path, destination: Path) -> MemorySnapshot:
    """Create a preserved snapshot directory for sanity and integration tests."""

    source_snapshot = snapshot_memory_path(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
    copied_snapshot = snapshot_memory_path(destination)
    if (
        copied_snapshot.descriptor.get("kind") != source_snapshot.descriptor.get("kind")
        or copied_snapshot.sha256 != source_snapshot.sha256
        or copied_snapshot.descriptor.get("byte_count")
        != source_snapshot.descriptor.get("byte_count")
    ):
        raise RuntimeError("copied memory snapshot differs from its source")
    return copied_snapshot


__all__ = [
    "DeliveryResult",
    "MemorySnapshot",
    "SCHEMA_VERSION",
    "TRUNCATION_ALGORITHM",
    "TokenCounter",
    "VisibleTokenBudgetGate",
    "canonical_json",
    "copy_snapshot",
    "sha256_bytes",
    "snapshot_memory_bytes",
    "snapshot_memory_path",
    "text_fingerprint",
]
