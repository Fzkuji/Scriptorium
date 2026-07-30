#!/usr/bin/env python3
"""Strictly audit a NativeMem v8.8+calendar GPT-5.5 BEAM run.

The collector follows the on-disk schema emitted by
``scripts/run_v88_gpt55_beam.py``.  It does not load the Hugging Face dataset
or call a model.  A successful audit atomically writes ``evaluation_input.json``
and ``audit.json`` inside the run directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402

EXPECTED_SPLIT_ROWS = {"100K": 20, "1M": 35}
FORMAL_SELECTION = {
    "100K": list(range(10)),
    "1M": list(range(EXPECTED_SPLIT_ROWS["1M"])),
}
FORMAL_CONVERSATION_COUNT = 45
FORMAL_QUESTION_COUNT = 900
EXPECTED_QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
EXPECTED_DATASET = "Mohammadta/BEAM"
EXPECTED_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
BUILD_MARKER = "_beam_build.json"
AUDIT_LOCK_FILENAME = ".beam-audit.lock"
RUNNER_LOCK_FILENAME = ".launcher.lock"
REQUIRED_SOURCE_FILES = {
    "scripts/run_v88_gpt55_beam.py",
    "src/adapters/run_nativemem.py",
    "src/nativemem.py",
    "src/v8_memory.py",
    "src/chatgpt_proxy.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "scripts/gpt55_run_proxy.py",
}
CONTEXT_POLICY_VERSION = "beam-local-context-v1"
CONTEXT_TOKENIZER_PACKAGE = "tiktoken"
CONTEXT_TOKENIZER_VERSION = "0.12.0"
CONTEXT_TOKENIZER_ENCODING = "o200k_base"
LOCAL_REQUEST_TOKEN_LIMIT = 96_000
TOTAL_TOOL_CONTENT_TOKEN_LIMIT = 48_000
PER_TOOL_CONTENT_TOKEN_LIMIT = 24_000
CONTEXT_OUTPUT_RESERVATION_TOKENS = 1_200
CONTEXT_SAFETY_MARGIN_TOKENS = 8_192
CONTEXT_TRACE_SCHEMA_VERSION = "beam-context-safety-trace-v1"
INTERACTION_TRACE_SCHEMA_VERSION = "beam-interaction-trace-v1"
TRUNCATION_ALGORITHM = "decoded-prefix-plus-marker-recount-v1"
TOOL_TRUNCATION_MARKER = (
    "\n[Tool output truncated by beam-local-context-v1. Request a "
    "narrower range or a more specific search.]"
)
COMPACT_EVIDENCE_TRUNCATION_MARKER = (
    "\n[Additional retrieved evidence omitted by beam-local-context-v1.]"
)
FINALIZE_INSTRUCTION = (
    "Tool use is now finished. Using only the evidence already retrieved "
    "above, answer the original BEAM question completely. Follow its "
    "requested format. If the evidence is genuinely absent, use the "
    "specified insufficient-information response. Return the final answer "
    "inside <answer></answer>."
)
BEAM_SINGLE_PROMPT = """You answer one BEAM benchmark question by navigating a NativeMem library.
You may use list_memory, search_memory, read_memory, and read_original only.
The first three tools expose a read-only view of the memory-library root.

The library has two views:
- topics/: topic files or nested topic directories. Read the complete relevant
  files instead of relying on filename or keyword overlap alone.
- timeline/YYYY/MM/DD.md: atomic events ordered by date. An event ends with one
  or more [Dn:m] references and may link back to its topic file.

[Dn:m] references identify original conversation turns. Pass all references
that may matter to read_original in one call. The returned turns include their
observation date. Use the original turns to verify names, numbers, dates,
constraints, preferences, updates, and contradictions.

Current library structure:
{structure}

Question:
{question}

Required procedure:
1. Use the structure above to read every topic likely to contain evidence. Use
   search_memory with several alternative terms only when topic navigation is
   insufficient.
2. For temporal or ordering questions, inspect the relevant timeline files and
   nearby dates. For contradictions or updates, prefer the latest supported
   statement while preserving any distinction the question asks about.
3. Call read_original for all candidate [Dn:m] references before answering when
   references are available. Check every requested part of the question.
4. Give a complete BEAM-style answer. There is no short-answer word limit.
   Follow any format or preference requested by the question. If the evidence
   is genuinely absent after a relevant search, answer exactly: "I don't have
   enough information to answer this question."
5. Put the final answer inside <answer></answer>. Do not put analysis inside the
   tags.

Tool results may end with a deterministic truncation notice. When that occurs,
request a narrower file range or a more specific search instead of assuming
that omitted lines are absent.
"""
EXPECTED_CONTEXT_CONFIG = {
    "policy_version": CONTEXT_POLICY_VERSION,
    "trace_schema_version": CONTEXT_TRACE_SCHEMA_VERSION,
    "interaction_trace_schema_version": INTERACTION_TRACE_SCHEMA_VERSION,
    "tokenizer_package": CONTEXT_TOKENIZER_PACKAGE,
    "tokenizer_version": CONTEXT_TOKENIZER_VERSION,
    "tokenizer_encoding": CONTEXT_TOKENIZER_ENCODING,
    "provider_exact": False,
    "local_request_token_limit": LOCAL_REQUEST_TOKEN_LIMIT,
    "total_tool_content_token_limit": TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
    "per_tool_content_token_limit": PER_TOOL_CONTENT_TOKEN_LIMIT,
    "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
    "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
    "truncation_algorithm": TRUNCATION_ALGORITHM,
    "overflow_action": "truncate_tool_prefix_then_compact_finalization",
}
EXPECTED_TOOL_SCHEMA_SHA256 = (
    "b4ecee1f0a13c60c7c0b6a5787a8b8ce8ad2e8898ae3eb6fc56028c8d4b5fc1f"
)


class AuditError(RuntimeError):
    """Raised when a formal BEAM artifact fails an integrity check."""


def reject_symlink_components(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise AuditError(f"{label} contains a symbolic-link component: {current}")
    return absolute


def validate_no_symlinks(root: Path) -> None:
    reject_symlink_components(root, "artifact tree")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"artifact tree contains a symbolic link: {path}")
        if not path.is_dir() and not path.is_file():
            raise AuditError(f"artifact tree contains a non-regular entry: {path}")


def full_tree_sha256(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise AuditError(f"artifact tree contains a symbolic link: {path}")
        if path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append({
                "path": relative,
                "type": "file",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        else:
            raise AuditError(f"artifact tree contains a non-regular entry: {path}")
    return stable_hash(entries)


def acquire_audit_locks(run_dir: Path) -> tuple[Any, Any]:
    """Exclude another audit and hold the runner lock for one stable snapshot."""
    run_dir = reject_symlink_components(run_dir, "run directory")
    if not run_dir.is_dir():
        raise AuditError(f"run directory is missing: {run_dir}")
    audit_handle = (run_dir / AUDIT_LOCK_FILENAME).open("w")
    try:
        fcntl.flock(audit_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        audit_handle.close()
        raise AuditError(f"another BEAM audit is already using {run_dir}") from exc

    runner_path = run_dir / RUNNER_LOCK_FILENAME
    if not runner_path.is_file():
        fcntl.flock(audit_handle.fileno(), fcntl.LOCK_UN)
        audit_handle.close()
        raise AuditError(f"runner lock file is missing: {runner_path}")
    runner_handle = runner_path.open("r+")
    try:
        fcntl.flock(runner_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        runner_handle.close()
        fcntl.flock(audit_handle.fileno(), fcntl.LOCK_UN)
        audit_handle.close()
        raise AuditError(f"BEAM runner is active in {run_dir}") from exc
    return audit_handle, runner_handle


def release_audit_locks(handles: tuple[Any, Any] | None) -> None:
    if handles is None:
        return
    audit_handle, runner_handle = handles
    try:
        fcntl.flock(runner_handle.fileno(), fcntl.LOCK_UN)
        runner_handle.close()
    finally:
        fcntl.flock(audit_handle.fileno(), fcntl.LOCK_UN)
        audit_handle.close()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_context_tokenizer() -> tuple[Any, dict[str, Any]]:
    """Load the exact tokenizer frozen by the runner; never approximate."""
    try:
        version = importlib_metadata.version(CONTEXT_TOKENIZER_PACKAGE)
        module = importlib.import_module(CONTEXT_TOKENIZER_PACKAGE)
    except (importlib_metadata.PackageNotFoundError, ModuleNotFoundError) as exc:
        raise AuditError(
            "BEAM audit requires tiktoken==0.12.0; no tokenizer fallback is allowed"
        ) from exc
    if version != CONTEXT_TOKENIZER_VERSION:
        raise AuditError(
            "BEAM audit tokenizer version mismatch: expected "
            f"{CONTEXT_TOKENIZER_VERSION}, received {version}"
        )
    return module.get_encoding(CONTEXT_TOKENIZER_ENCODING), {
        "package": CONTEXT_TOKENIZER_PACKAGE,
        "version": CONTEXT_TOKENIZER_VERSION,
        "encoding": CONTEXT_TOKENIZER_ENCODING,
        "model_alias": "gpt-5.5",
        "provider_exact": False,
    }


def _local_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))


def _request_descriptor(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": messages}
    if tools is not None:
        payload["tools"] = tools
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    local_tokens = _local_tokens(tokenizer, rendered)
    bounded_total = (
        local_tokens
        + CONTEXT_OUTPUT_RESERVATION_TOKENS
        + CONTEXT_SAFETY_MARGIN_TOKENS
    )
    return {
        "local_tokens": local_tokens,
        "payload_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "payload_utf8_bytes": len(rendered.encode("utf-8")),
        "bounded_total_tokens": bounded_total,
        "within_local_limit": bounded_total <= LOCAL_REQUEST_TOKEN_LIMIT,
    }


def _safe_decoded_prefix(
    text: str,
    token_ids: list[int],
    count: int,
    tokenizer: Any,
) -> tuple[str, int]:
    count = max(0, min(count, len(token_ids)))
    while count:
        candidate = tokenizer.decode(token_ids[:count])
        if text.startswith(candidate):
            return candidate, count
        count -= 1
    return "", 0


def _truncate_tool_content(
    text: str,
    *,
    tokenizer: Any,
    limit: int,
    marker: str = TOOL_TRUNCATION_MARKER,
) -> tuple[str, int, int, bool]:
    """Independent implementation of the frozen decoded-prefix policy."""
    raw_tokens = tokenizer.encode(text, disallowed_special=())
    if len(raw_tokens) <= limit:
        return text, len(raw_tokens), len(raw_tokens), False
    marker_tokens = tokenizer.encode(marker, disallowed_special=())
    if limit <= 0:
        return "", len(raw_tokens), 0, True
    if len(marker_tokens) > limit:
        delivered, _ = _safe_decoded_prefix(
            marker, marker_tokens, limit, tokenizer
        )
        return delivered, len(raw_tokens), _local_tokens(tokenizer, delivered), True
    prefix_slots = max(0, limit - len(marker_tokens))
    while True:
        prefix, used_slots = _safe_decoded_prefix(
            text, raw_tokens, prefix_slots, tokenizer
        )
        delivered = prefix + marker
        delivered_count = _local_tokens(tokenizer, delivered)
        if delivered_count <= limit:
            return delivered, len(raw_tokens), delivered_count, True
        if used_slots <= 0:
            raise AuditError("truncation marker exceeds its verified token limit")
        prefix_slots = max(0, used_slots - max(1, delivered_count - limit))


def _resolve_memory_path(memory_dir: Path, raw_path: Any) -> Path:
    root = memory_dir.resolve()
    candidate = Path(str(raw_path or ".").strip())
    if candidate.is_absolute():
        raise ValueError("memory tool paths must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("memory tool path leaves the memory directory")
    if not resolved.exists():
        raise FileNotFoundError(
            f"memory path does not exist: {str(raw_path or '.').strip()}"
        )
    return resolved


def _memory_markdown_files(path: Path, root: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".md" else []
    return sorted(
        candidate
        for candidate in path.rglob("*.md")
        if candidate.resolve().is_relative_to(root)
        if not any(
            part.startswith(".") or part == "raw"
            for part in candidate.relative_to(path).parts
        )
    )


def _execute_memory_tool(
    tool_name: str, arguments: dict[str, Any], memory_dir: Path
) -> str:
    """Independent read-only reexecution for memory-backed tool traces."""
    root = memory_dir.resolve()
    target = _resolve_memory_path(root, arguments.get("path", "."))
    if tool_name == "list_memory":
        recursive = bool(arguments.get("recursive", False))
        if target.is_file():
            candidates = [target]
        elif recursive:
            candidates = sorted(target.rglob("*"))
        else:
            candidates = sorted(target.iterdir())
        lines: list[str] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                continue
            relative = resolved.relative_to(root)
            if any(
                part.startswith(".") or part == "raw" for part in relative.parts
            ):
                continue
            if resolved.is_dir():
                lines.append(f"{relative.as_posix()}/")
            elif resolved.suffix == ".md":
                lines.append(relative.as_posix())
            if len(lines) >= 2000:
                lines.append("(listing truncated at 2000 entries)")
                break
        return "\n".join(lines) or "(no Markdown files)"
    if tool_name == "search_memory":
        query = str(arguments.get("query", "") or "").strip()
        if not query:
            raise ValueError("search_memory requires a non-empty query")
        try:
            max_results = max(
                1, min(200, int(arguments.get("max_results", 80)))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("max_results must be an integer") from exc
        hits: list[str] = []
        needle = query.casefold()
        for path in _memory_markdown_files(target, root):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if needle not in line.casefold():
                        continue
                    relative = path.resolve().relative_to(root).as_posix()
                    hits.append(f"{relative}:{line_number}:{line.rstrip()}")
                    if len(hits) >= max_results:
                        hits.append(
                            f"(search truncated at {max_results} results)"
                        )
                        return "\n".join(hits)
        return "\n".join(hits) or "(no matches)"
    if tool_name == "read_memory":
        if not target.is_file() or target.suffix != ".md":
            raise ValueError("read_memory path must name a Markdown file")
        try:
            start = max(1, int(arguments.get("start_line", 1)))
            end = int(arguments.get("end_line", 2000))
        except (TypeError, ValueError) as exc:
            raise ValueError("line bounds must be integers") from exc
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        end = min(end, start + 1999)
        lines: list[str] = []
        with target.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number < start:
                    continue
                if line_number > end:
                    break
                lines.append(f"{line_number}: {line.rstrip()}")
        return "\n".join(lines) or "(requested range is empty)"
    raise ValueError(f"unsupported memory tool: {tool_name}")


def _execute_memory_tool_for_audit(
    tool_name: str, arguments: dict[str, Any], memory_dir: Path
) -> tuple[str, str | None]:
    try:
        return _execute_memory_tool(tool_name, arguments, memory_dir), None
    except (
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        ValueError,
    ) as exc:
        message = f"Error: {type(exc).__name__}: {exc}"
        return message, message


def _count_memory_entries(path: Path) -> int:
    count = 0
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if re.match(r"^\[\d{4}-\d{2}-\d{2}\]", line):
                    count += 1
    except OSError:
        pass
    return count


def _memory_structure_map(memory_dir: Path) -> str:
    """Recompute the frozen ``dir`` structure map from persisted memory."""
    if not memory_dir.is_dir():
        return "(empty memory)"

    def walk(directory: Path, depth: int) -> list[str]:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return []
        subdirectories = [
            item
            for item in entries
            if item.is_dir()
            and not item.name.startswith(".")
            and item.name != "raw"
        ]
        markdown = [
            item
            for item in entries
            if item.is_file()
            and item.name.endswith(".md")
            and not item.name.startswith(".")
        ]
        prefix = "  " * depth
        lines: list[str] = []
        if markdown:
            total = sum(_count_memory_entries(path) for path in markdown)
            unit = "file" if len(markdown) == 1 else "files"
            lines.append(f"{prefix}({len(markdown)} {unit}, {total} entries)")
        for subdirectory in subdirectories:
            sub_total = 0
            for root, _, filenames in os.walk(subdirectory):
                if "raw" in Path(root).parts:
                    continue
                sub_total += sum(
                    _count_memory_entries(Path(root) / filename)
                    for filename in filenames
                    if filename.endswith(".md")
                )
            lines.append(f"{prefix}{subdirectory.name}/ [{sub_total}]")
            if subdirectory.name == "timeline" and depth == 0:
                lines.append(
                    f"{prefix}  (路径模板 timeline/YYYY/MM/DD.md：三层依次是年/月/日，"
                    "月日两位数字)"
                )
            lines.extend(walk(subdirectory, depth + 1))
        return lines

    lines = walk(memory_dir, 0)
    return "\n".join(lines) if lines else "(empty memory)"


def _compact_finalize_messages(
    *,
    tokenizer: Any,
    question: str,
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Independently reconstruct the compact finalization payload."""
    header = (
        "Answer the BEAM question using only the retrieved tool evidence below. "
        "Follow the requested format and return the final answer inside "
        "<answer></answer>. If the evidence is genuinely absent, answer exactly: "
        "\"I don't have enough information to answer this question.\"\n\n"
        f"Question:\n{question}\n\nRetrieved tool evidence:\n"
    )
    raw_evidence = "".join(
        f"\n[{item['tool']} step={item['step']}]\n{item['delivered_text']}"
        for item in evidence
        if item.get("delivered_text")
    )
    raw = header + raw_evidence
    request_limit = (
        LOCAL_REQUEST_TOKEN_LIMIT
        - CONTEXT_OUTPUT_RESERVATION_TOKENS
        - CONTEXT_SAFETY_MARGIN_TOKENS
    )
    raw_messages = [{"role": "user", "content": raw}]
    raw_request_tokens = _request_descriptor(
        tokenizer, raw_messages, None
    )["local_tokens"]
    delivered_evidence = raw_evidence
    truncated = False
    if raw_request_tokens > request_limit:
        truncated = True
        header_messages = [{"role": "user", "content": header}]
        header_request_tokens = _request_descriptor(
            tokenizer, header_messages, None
        )["local_tokens"]
        if header_request_tokens > request_limit:
            raise AuditError(
                "BEAM question alone exceeds the compact-finalization limit"
            )
        evidence_limit = max(0, request_limit - header_request_tokens)
        while True:
            delivered_evidence, _, _, _ = _truncate_tool_content(
                raw_evidence,
                tokenizer=tokenizer,
                limit=evidence_limit,
                marker=COMPACT_EVIDENCE_TRUNCATION_MARKER,
            )
            delivered = header + delivered_evidence
            delivered_request_tokens = _request_descriptor(
                tokenizer, [{"role": "user", "content": delivered}], None
            )["local_tokens"]
            if delivered_request_tokens <= request_limit:
                break
            if evidence_limit <= 0:
                raise AuditError(
                    "compact-finalization header cannot satisfy the request limit"
                )
            evidence_limit = max(
                0,
                evidence_limit
                - max(1, delivered_request_tokens - request_limit),
            )
    delivered = header + delivered_evidence
    delivered_messages = [{"role": "user", "content": delivered}]
    delivered_request_tokens = _request_descriptor(
        tokenizer, delivered_messages, None
    )["local_tokens"]
    record = {
        "schema_version": "beam-compact-finalization-v1",
        "raw_content_tokens": _local_tokens(tokenizer, raw),
        "delivered_content_tokens": _local_tokens(tokenizer, delivered),
        "raw_request_tokens": raw_request_tokens,
        "delivered_request_tokens": delivered_request_tokens,
        "truncated": truncated,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "delivered_sha256": hashlib.sha256(
            delivered.encode("utf-8")
        ).hexdigest(),
        "source_tool_trace_sha256": stable_hash(evidence),
    }
    return delivered_messages, record


def parse_time(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AuditError(f"timestamp has no timezone: {value!r}")
    return parsed


def parse_indices(spec: object, size: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(size))
    selected: set[int] = set()
    try:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(value) for value in part.split("-", 1))
                low, high = sorted((first, last))
                selected.update(range(low, high + 1))
            else:
                selected.add(int(part))
    except ValueError as exc:
        raise AuditError(f"invalid conversation selection: {spec!r}") from exc
    values = sorted(selected)
    if not values or any(index < 0 or index >= size for index in values):
        raise AuditError(f"conversation selection {spec!r} is invalid for {size} rows")
    return values


def selected_conversations(
    manifest: dict[str, Any], *, allow_fixture: bool = False
) -> dict[str, list[int]]:
    """Reconstruct the union of selections across all resume invocations."""
    history = manifest.get("selection_history")
    if not isinstance(history, list) or not history:
        selection = manifest.get("selection")
        history = [selection] if isinstance(selection, dict) else []
    if not history:
        raise AuditError("manifest has no selection history")

    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise AuditError("manifest has no dataset sources")
    chosen: dict[str, set[int]] = defaultdict(set)
    for invocation, selection in enumerate(history):
        if not isinstance(selection, dict):
            raise AuditError(f"selection history entry {invocation} is not an object")
        sizes = selection.get("chat_sizes")
        if not isinstance(sizes, list) or not sizes:
            raise AuditError(f"selection history entry {invocation} has no splits")
        limit = selection.get("limit_per_split")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise AuditError(f"selection history entry {invocation} has invalid limit")
        for chat_size in sizes:
            if chat_size not in EXPECTED_SPLIT_ROWS:
                raise AuditError(f"unsupported selected BEAM split: {chat_size!r}")
            source = sources.get(chat_size)
            if not isinstance(source, dict):
                raise AuditError(f"manifest lacks source metadata for {chat_size}")
            rows = source.get("rows")
            expected_rows = EXPECTED_SPLIT_ROWS[chat_size]
            if source.get("kind") == "huggingface":
                if rows != expected_rows:
                    raise AuditError(
                        f"{chat_size} source has {rows!r} rows, expected {expected_rows}"
                    )
            elif not allow_fixture:
                raise AuditError(
                    f"{chat_size} source is {source.get('kind')!r}; "
                    "formal audit requires Hugging Face data"
                )
            if not isinstance(rows, int) or rows < 1:
                raise AuditError(f"{chat_size} source has invalid row count")
            indices = parse_indices(selection.get("conversations", "all"), rows)
            if limit is not None:
                indices = indices[:limit]
            chosen[chat_size].update(indices)

    expected_source_splits = set(chosen)
    if set(sources) != expected_source_splits:
        raise AuditError(
            "dataset source splits differ from the union of selected splits: "
            f"sources={sorted(sources)}, selected={sorted(expected_source_splits)}"
        )
    return {split: sorted(indices) for split, indices in sorted(chosen.items())}


def memory_tree_sha256(memory_dir: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in memory_dir.rglob("*.md")
        if "raw" not in path.relative_to(memory_dir).parts
    )
    for path in paths:
        digest.update(path.relative_to(memory_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return len(paths), digest.hexdigest()


def memory_payload_descriptor(memory_dir: Path) -> dict[str, Any]:
    """Independently hash all memory files except the replaceable marker."""
    memory_dir = reject_symlink_components(memory_dir, "memory directory")
    files: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(memory_dir.rglob("*")):
        relative = path.relative_to(memory_dir).as_posix()
        if path.is_symlink():
            raise AuditError(f"memory payload contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AuditError(f"memory payload contains special node: {relative}")
        stat = path.stat(follow_symlinks=False)
        if stat.st_nlink != 1:
            raise AuditError(f"memory payload contains hardlink: {relative}")
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            raise AuditError(f"memory payload reuses inode: {relative}")
        seen_inodes.add(inode)
        if relative == BUILD_MARKER:
            continue
        files.append({
            "path": relative,
            "size": stat.st_size,
            "sha256": sha256_file(path),
        })
    return {
        "file_count": len(files),
        "tree_sha256": stable_hash(files),
        "files": files,
    }


def expected_result_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    questions = sorted(
        checkpoint.get("questions", {}).values(),
        key=lambda question: question.get("question_index", 10**9),
    )
    return {
        "metadata": {
            key: checkpoint.get(key)
            for key in (
                "schema_version",
                "benchmark",
                "method",
                "chat_size",
                "conversation_index",
                "conversation_id",
                "status",
                "created_at",
                "updated_at",
                "completed_at",
                "input_hash",
                "config_hash",
                "config",
                "source",
                "build",
            )
        },
        "questions": questions,
        "errors": checkpoint.get("errors", []),
    }


def validate_source(manifest: dict[str, Any], chat_size: str) -> dict[str, Any]:
    source = manifest["sources"][chat_size]
    if source.get("kind") == "huggingface":
        expected = {
            "dataset": EXPECTED_DATASET,
            "config": "default",
            "revision": EXPECTED_REVISION,
            "split": chat_size,
            "rows": EXPECTED_SPLIT_ROWS[chat_size],
        }
        mismatches = {
            key: {"expected": value, "actual": source.get(key)}
            for key, value in expected.items()
            if source.get(key) != value
        }
        if mismatches:
            raise AuditError(f"{chat_size} dataset metadata mismatch: {mismatches}")
        if not str(source.get("fingerprint", "")).strip():
            raise AuditError(f"{chat_size} dataset fingerprint is missing")
    return source


def validate_memory(conv_dir: Path, checkpoint: dict[str, Any]) -> tuple[int, str]:
    memory_dir = conv_dir / "memory"
    if not memory_dir.is_dir():
        raise AuditError(f"missing memory directory: {memory_dir}")
    marker_path = memory_dir / BUILD_MARKER
    if not marker_path.is_file():
        raise AuditError(f"missing build marker: {marker_path}")
    marker = read_json(marker_path)
    marker_schema = marker.get("schema_version")
    if marker_schema not in (1, 2):
        raise AuditError(f"invalid build marker schema in {conv_dir}")
    for key in ("input_hash", "config_hash"):
        if marker.get(key) != checkpoint.get(key):
            raise AuditError(f"{conv_dir.name} marker {key} mismatch")
    stats = marker.get("stats")
    build = checkpoint.get("build")
    if not isinstance(stats, dict) or not isinstance(build, dict):
        raise AuditError(f"{conv_dir.name} has invalid build metadata")
    if stats.get("status") != "complete" or build.get("status") != "complete":
        raise AuditError(f"{conv_dir.name} build is not complete")

    markdown_files, memory_hash = memory_tree_sha256(memory_dir)
    if markdown_files <= 0:
        raise AuditError(f"{conv_dir.name} has no persisted Markdown memory")
    for record_name, record in (("marker", stats), ("checkpoint", build)):
        if record.get("memory_sha256") != memory_hash:
            raise AuditError(f"{conv_dir.name} {record_name} memory hash mismatch")
        if record.get("markdown_files") != markdown_files:
            raise AuditError(f"{conv_dir.name} {record_name} Markdown count mismatch")
    # The marker is the atomically published build record.  Checkpoint model
    # evidence must agree with it; neither record is a proxy-log substitute.
    for key in (
        "events",
        "expected_dia_ids",
        "stored_dia_ids",
        "chunks_total",
        "chunks_with_events",
        "calls",
        "tokens_in",
        "tokens_out",
        "requested_models",
        "response_models",
        "finish_reasons",
    ):
        if stats.get(key) != build.get(key):
            raise AuditError(f"{conv_dir.name} marker/checkpoint {key} mismatch")
    required_positive = (
        "events",
        "expected_dia_ids",
        "stored_dia_ids",
        "chunks_total",
        "chunks_with_events",
        "calls",
        "tokens_in",
    )
    for key in required_positive:
        if int(build.get(key, 0) or 0) <= 0:
            raise AuditError(f"{conv_dir.name} build has invalid {key}")
    if build.get("chunks_with_events") != build.get("chunks_total"):
        raise AuditError(f"{conv_dir.name} has a build chunk without events")
    if build.get("stored_dia_ids", 0) > build.get("expected_dia_ids", 0):
        raise AuditError(f"{conv_dir.name} stores more dia ids than expected")
    if build.get("requested_models") != ["gpt-5.5"]:
        raise AuditError(f"{conv_dir.name} build did not request only GPT-5.5")
    if build.get("response_models") != ["gpt-5.5"]:
        raise AuditError(f"{conv_dir.name} build did not receive only GPT-5.5")
    if build.get("finish_reasons") != ["stop"]:
        raise AuditError(f"{conv_dir.name} build has non-stop responses")
    if marker_schema == 2:
        provenance = marker.get("import_provenance")
        if (not isinstance(provenance, dict)
                or provenance.get("schema")
                != "nativemem.beam-build-import.v1"):
            raise AuditError(f"{conv_dir.name} import provenance is invalid")
        if build.get("import_provenance") != provenance:
            raise AuditError(
                f"{conv_dir.name} marker/checkpoint import provenance mismatch")
        if not build.get("imported_existing_build"):
            raise AuditError(f"{conv_dir.name} imported-build flag is absent")
        current_payload = memory_payload_descriptor(memory_dir)
        if (current_payload["tree_sha256"]
                != provenance.get("destination_payload_sha256")):
            raise AuditError(
                f"{conv_dir.name} imported destination payload mismatch")
        source_memory = Path(str(provenance.get("source_memory", ""))).resolve()
        source_checkpoint = Path(
            str(provenance.get("source_checkpoint", ""))).resolve()
        source_marker = source_memory / BUILD_MARKER
        if (ROOT not in source_memory.parents
                or not source_memory.is_dir() or source_memory.is_symlink()
                or not source_checkpoint.is_file()
                or source_checkpoint.is_symlink()
                or not source_marker.is_file() or source_marker.is_symlink()):
            raise AuditError(f"{conv_dir.name} import source is unsafe or missing")
        if sha256_file(source_checkpoint) != provenance.get(
                "source_checkpoint_sha256"):
            raise AuditError(
                f"{conv_dir.name} imported source checkpoint changed")
        if sha256_file(source_marker) != provenance.get("source_marker_sha256"):
            raise AuditError(f"{conv_dir.name} imported source marker changed")
        source_payload = memory_payload_descriptor(source_memory)
        if (source_payload["tree_sha256"]
                != provenance.get("source_payload_sha256")
                or source_payload["tree_sha256"]
                != current_payload["tree_sha256"]):
            raise AuditError(f"{conv_dir.name} imported source payload changed")
        source_marker_value = read_json(source_marker)
        if (source_marker_value.get("schema_version") != 1
                or source_marker_value.get("input_hash")
                != checkpoint.get("input_hash")
                or source_marker_value.get("config_hash")
                != provenance.get("source_config_hash")):
            raise AuditError(f"{conv_dir.name} imported source identity mismatch")
        source_checkpoint_value = read_json(source_checkpoint)
        source_hashes = source_checkpoint_value.get("config", {}).get(
            "source_sha256", {})
        if source_hashes.get("scripts/run_v88_gpt55_beam.py") != provenance.get(
                "source_runner_sha256"):
            raise AuditError(f"{conv_dir.name} imported source runner mismatch")
        source_stats = source_marker_value.get("stats", {})
        for key in (
            "events", "expected_dia_ids", "stored_dia_ids", "chunks_total",
            "chunks_with_events", "calls", "tokens_in", "tokens_out",
            "requested_models", "response_models", "finish_reasons",
            "memory_sha256", "markdown_files",
        ):
            if source_stats.get(key) != build.get(key):
                raise AuditError(
                    f"{conv_dir.name} imported build differs for {key}")
    elif build.get("imported_existing_build") or build.get("import_provenance"):
        raise AuditError(
            f"{conv_dir.name} schema-v1 marker claims imported provenance")
    return markdown_files, memory_hash


def validate_source_id_map(conv_dir: Path, checkpoint: dict[str, Any]) -> None:
    source_map = read_json(conv_dir / "source_id_map.json")
    if source_map.get("schema_version") != 1:
        raise AuditError(f"{conv_dir.name} source-id map schema mismatch")
    mapping = source_map.get("mapping")
    if not isinstance(mapping, dict):
        raise AuditError(f"{conv_dir.name} source-id mapping is not an object")
    mapping_hash = stable_hash(mapping)
    input_meta = checkpoint.get("config", {}).get("input", {})
    if source_map.get("sha256") != mapping_hash:
        raise AuditError(f"{conv_dir.name} source-id map self-hash mismatch")
    if input_meta.get("source_id_map_sha256") != mapping_hash:
        raise AuditError(f"{conv_dir.name} source-id map checkpoint hash mismatch")
    if input_meta.get("source_id_count") != len(mapping):
        raise AuditError(f"{conv_dir.name} source-id count mismatch")
    occurrences = sum(len(values) for values in mapping.values())
    if input_meta.get("source_id_occurrences") != occurrences:
        raise AuditError(f"{conv_dir.name} source-id occurrence count mismatch")


def _parse_tool_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(raw or "{}")
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value, None
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"Error: {type(exc).__name__}: {exc}"


def _validate_normalized_tool_call(
    value: Any, *, question_id: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"id", "type", "function"}:
        raise AuditError(f"{question_id} has an invalid normalized tool call")
    function = value.get("function")
    if (
        value.get("type") != "function"
        or not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(function, dict)
        or set(function) != {"name", "arguments"}
        or not isinstance(function.get("name"), str)
        or not function["name"]
        or not isinstance(function.get("arguments"), str)
    ):
        raise AuditError(f"{question_id} has an invalid normalized tool call")
    return value


def _answer_from_content(content: str) -> str | None:
    stripped = re.sub(
        r"<think>.*?</think>", "", content or "", flags=re.DOTALL
    ).strip()
    match = re.search(
        r"<answer>(.*?)</answer>",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match or not match.group(1).strip():
        return None
    return match.group(1).strip()


def _validate_context_trace(
    retrieval: dict[str, Any],
    *,
    question_id: str,
    question_text: str,
    expected_answer: str,
    stored_memories: Any,
    memory_dir: Path,
    tokenizer: Any,
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    """Reexecute tools and replay every locally observed request payload."""
    context = retrieval.get("context_safety")
    interaction = retrieval.get("interaction_trace")
    tool_trace = retrieval.get("tool_trace")
    expected_context_keys = {
        "policy_version",
        "trace_schema_version",
        "tokenizer",
        "local_request_token_limit",
        "total_tool_content_token_limit",
        "per_tool_content_token_limit",
        "output_reservation_tokens",
        "safety_margin_tokens",
        "raw_tool_tokens",
        "delivered_tool_tokens",
        "truncated_tool_results",
        "tool_budget_exhausted",
        "request_observations",
        "request_local_tokens",
        "max_request_local_tokens",
        "max_sent_request_local_tokens",
        "rejected_request_candidates",
        "context_compaction_used",
        "compact_finalization",
    }
    if not isinstance(context, dict) or set(context) != expected_context_keys:
        raise AuditError(f"{question_id} has an invalid context-safety schema")
    exact_context = {
        "policy_version": CONTEXT_POLICY_VERSION,
        "trace_schema_version": CONTEXT_TRACE_SCHEMA_VERSION,
        "tokenizer": tokenizer_identity,
        "local_request_token_limit": LOCAL_REQUEST_TOKEN_LIMIT,
        "total_tool_content_token_limit": TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
        "per_tool_content_token_limit": PER_TOOL_CONTENT_TOKEN_LIMIT,
        "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
        "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
    }
    for key, expected in exact_context.items():
        if context.get(key) != expected:
            raise AuditError(f"{question_id} context-safety {key} mismatch")

    interaction_keys = {
        "schema_version",
        "initial_prompt",
        "structure",
        "tools",
        "finalize_instruction",
        "responses",
        "termination_reason",
    }
    if not isinstance(interaction, dict) or set(interaction) != interaction_keys:
        raise AuditError(f"{question_id} has an invalid interaction trace")
    if interaction.get("schema_version") != INTERACTION_TRACE_SCHEMA_VERSION:
        raise AuditError(f"{question_id} interaction schema mismatch")
    initial_prompt = interaction.get("initial_prompt")
    structure = interaction.get("structure")
    tools = interaction.get("tools")
    if (
        not isinstance(initial_prompt, str)
        or not isinstance(structure, str)
        or structure != _memory_structure_map(memory_dir)
        or initial_prompt != BEAM_SINGLE_PROMPT.format(
            structure=structure, question=question_text
        )
        or interaction.get("finalize_instruction") != FINALIZE_INSTRUCTION
        or not isinstance(tools, list)
        or len(tools) != 4
    ):
        raise AuditError(f"{question_id} interaction prompt/tool schema mismatch")
    tool_names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
        ):
            raise AuditError(f"{question_id} has an invalid request tool schema")
        tool_names.append(function["name"])
    if tool_names != [
        "list_memory",
        "search_memory",
        "read_memory",
        "read_original",
    ]:
        raise AuditError(f"{question_id} request tool inventory mismatch")
    if stable_hash(tools) != EXPECTED_TOOL_SCHEMA_SHA256:
        raise AuditError(f"{question_id} request tool schema hash mismatch")

    responses = interaction.get("responses")
    response_keys = {
        "step",
        "phase",
        "request_sequence",
        "response_id",
        "model",
        "finish_reason",
        "refusal",
        "content",
        "tool_calls",
    }
    if not isinstance(responses, list) or not responses:
        raise AuditError(f"{question_id} has no response trace")
    seen_response_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    flattened_calls: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(responses, 1):
        if not isinstance(item, dict) or set(item) != response_keys:
            raise AuditError(f"{question_id} response trace schema mismatch")
        if (
            item.get("step") != index
            or item.get("phase")
            not in {"beam_v8_single", "beam_v8_single_finalize"}
            or not isinstance(item.get("request_sequence"), int)
            or item["request_sequence"] < 1
            or item.get("model") != "gpt-5.5"
            or not isinstance(item.get("response_id"), str)
            or not item["response_id"]
            or item["response_id"] in seen_response_ids
            or not isinstance(item.get("refusal"), str)
            or not isinstance(item.get("content"), str)
            or not isinstance(item.get("tool_calls"), list)
        ):
            raise AuditError(f"{question_id} has invalid response evidence")
        seen_response_ids.add(item["response_id"])
        calls = [
            _validate_normalized_tool_call(call, question_id=question_id)
            for call in item["tool_calls"]
        ]
        if calls:
            if item.get("finish_reason") != "tool_calls" or item["refusal"]:
                raise AuditError(f"{question_id} has an invalid tool response")
            for call in calls:
                if call["id"] in seen_tool_call_ids:
                    raise AuditError(f"{question_id} reuses a tool-call id")
                seen_tool_call_ids.add(call["id"])
                flattened_calls.append((index, call))
        elif (
            item.get("finish_reason") != "stop"
            or item["refusal"]
            or _answer_from_content(item["content"]) is None
        ):
            raise AuditError(f"{question_id} has an invalid final response trace")

    if retrieval.get("response_ids") != [item["response_id"] for item in responses]:
        raise AuditError(f"{question_id} response-id trace mismatch")
    if retrieval.get("response_models") != ["gpt-5.5"]:
        raise AuditError(f"{question_id} response-model trace mismatch")
    if retrieval.get("steps") != len(responses):
        raise AuditError(f"{question_id} response-step trace mismatch")

    if not isinstance(tool_trace, list) or len(tool_trace) != len(flattened_calls):
        raise AuditError(f"{question_id} tool trace count mismatch")
    common_tool_keys = {
        "step",
        "tool_call_id",
        "tool",
        "arguments",
        "raw_sha256",
        "delivered_sha256",
        "raw_tokens",
        "delivered_tokens",
        "truncated",
        "remaining_tokens_before",
        "applied_token_limit",
        "truncation_algorithm",
        "cumulative_delivered_tokens",
        "delivered_text",
    }
    recomputed_errors: list[dict[str, Any]] = []
    expected_memories: list[dict[str, str]] = []
    cumulative = 0
    raw_total = 0
    traces_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, (trace, expected_call) in enumerate(
        zip(tool_trace, flattened_calls, strict=True)
    ):
        expected_step, call = expected_call
        expected_keys = set(common_tool_keys)
        if call["function"]["name"] == "read_original":
            expected_keys.add("raw_text")
        if not isinstance(trace, dict) or set(trace) != expected_keys:
            raise AuditError(f"{question_id} tool trace {index} schema mismatch")
        function = call["function"]
        arguments, argument_error = _parse_tool_arguments(function["arguments"])
        if (
            trace.get("step") != expected_step
            or trace.get("tool_call_id") != call["id"]
            or trace.get("tool") != function["name"]
            or trace.get("arguments") != arguments
            or trace.get("truncation_algorithm") != TRUNCATION_ALGORITHM
        ):
            raise AuditError(f"{question_id} tool trace {index} identity mismatch")
        tool_error = argument_error
        if argument_error is not None:
            raw_output = argument_error
        elif function["name"] == "read_original":
            raw_output = trace.get("raw_text")
            if not isinstance(raw_output, str):
                raise AuditError(f"{question_id} read_original raw text is absent")
        else:
            raw_output, tool_error = _execute_memory_tool_for_audit(
                function["name"], arguments, memory_dir
            )
        if tool_error is not None:
            recomputed_errors.append({
                "step": expected_step,
                "tool": function["name"],
                "arguments": arguments,
                "error": tool_error,
            })
        raw_tokens = _local_tokens(tokenizer, raw_output)
        remaining = max(0, TOTAL_TOOL_CONTENT_TOKEN_LIMIT - cumulative)
        applied_limit = min(PER_TOOL_CONTENT_TOKEN_LIMIT, remaining)
        delivered, expected_raw_tokens, delivered_tokens, truncated = (
            _truncate_tool_content(
                raw_output, tokenizer=tokenizer, limit=applied_limit
            )
        )
        raw_sha256 = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        delivered_sha256 = hashlib.sha256(delivered.encode("utf-8")).hexdigest()
        cumulative += delivered_tokens
        raw_total += raw_tokens
        exact = {
            "raw_sha256": raw_sha256,
            "delivered_sha256": delivered_sha256,
            "raw_tokens": expected_raw_tokens,
            "delivered_tokens": delivered_tokens,
            "truncated": truncated,
            "remaining_tokens_before": remaining,
            "applied_token_limit": applied_limit,
            "cumulative_delivered_tokens": cumulative,
            "delivered_text": delivered,
        }
        for key, expected in exact.items():
            if trace.get(key) != expected:
                raise AuditError(
                    f"{question_id} tool trace {index} {key} mismatch"
                )
        if function["name"] == "read_original" and delivered.strip():
            expected_memories.append({"text": delivered, "date": ""})
        traces_by_step[expected_step].append(trace)

    if retrieval.get("tool_input_errors") != recomputed_errors:
        raise AuditError(f"{question_id} tool-input-error ledger mismatch")
    if stored_memories != expected_memories:
        raise AuditError(f"{question_id} read_original memory ledger mismatch")
    context_exact = {
        "raw_tool_tokens": raw_total,
        "delivered_tool_tokens": cumulative,
        "truncated_tool_results": sum(
            bool(trace["truncated"]) for trace in tool_trace
        ),
        "tool_budget_exhausted": cumulative >= TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
    }
    for key, expected in context_exact.items():
        if context.get(key) != expected:
            raise AuditError(f"{question_id} context-safety {key} mismatch")

    observations = context.get("request_observations")
    observation_keys = {
        "sequence",
        "phase",
        "attempt",
        "sent",
        "outcome",
        "reason",
        "tools_enabled",
        "message_count",
        "output_reservation_tokens",
        "safety_margin_tokens",
        "local_tokens",
        "payload_sha256",
        "payload_utf8_bytes",
        "bounded_total_tokens",
        "within_local_limit",
    }
    if not isinstance(observations, list) or not observations:
        raise AuditError(f"{question_id} has no request observations")
    response_by_request = {item["request_sequence"]: item for item in responses}
    if len(response_by_request) != len(responses):
        raise AuditError(f"{question_id} reuses a request sequence")
    replay_messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_prompt}
    ]
    compact_messages: list[dict[str, str]] | None = None
    compact_record: dict[str, Any] | None = None
    final_answer_seen = False
    next_final_attempt = {"finalize_full": 1, "finalize_compact": 1}
    for sequence, observation in enumerate(observations, 1):
        if not isinstance(observation, dict) or set(observation) != observation_keys:
            raise AuditError(f"{question_id} request observation schema mismatch")
        if observation.get("sequence") != sequence:
            raise AuditError(f"{question_id} request sequence mismatch")
        phase = observation.get("phase")
        attempt = observation.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise AuditError(f"{question_id} request attempt is not an integer")
        if phase == "retrieve":
            if attempt != 1:
                raise AuditError(f"{question_id} retrieve attempt mismatch")
            expected_messages = replay_messages
            expected_tools = tools
        elif phase == "finalize_full":
            expected_messages = [
                *replay_messages,
                {"role": "user", "content": FINALIZE_INSTRUCTION},
            ]
            expected_tools = None
        elif phase == "finalize_compact":
            if compact_messages is None:
                compact_messages, compact_record = _compact_finalize_messages(
                    tokenizer=tokenizer,
                    question=question_text,
                    evidence=tool_trace,
                )
            expected_messages = compact_messages
            expected_tools = None
        else:
            raise AuditError(f"{question_id} has an unknown request phase")
        descriptor = _request_descriptor(
            tokenizer, expected_messages, expected_tools
        )
        exact_observation = {
            "tools_enabled": expected_tools is not None,
            "message_count": len(expected_messages),
            "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            **descriptor,
        }
        for key, expected in exact_observation.items():
            if observation.get(key) != expected:
                raise AuditError(
                    f"{question_id} request observation {sequence} {key} mismatch"
                )
        sent = observation.get("sent")
        outcome = observation.get("outcome")
        reason = observation.get("reason")
        response_item = response_by_request.pop(sequence, None)
        if sent is True:
            if phase in next_final_attempt:
                if attempt != next_final_attempt[phase]:
                    raise AuditError(f"{question_id} final retry sequence mismatch")
                next_final_attempt[phase] += 1
            if not descriptor["within_local_limit"]:
                raise AuditError(f"{question_id} sent a request over the hard cap")
            if outcome == "response_received":
                if response_item is None or reason is not None:
                    raise AuditError(f"{question_id} response observation mismatch")
            elif outcome == "client_error":
                if (
                    response_item is not None
                    or phase == "retrieve"
                    or not isinstance(reason, str)
                    or not reason
                ):
                    raise AuditError(f"{question_id} client-error evidence mismatch")
            else:
                raise AuditError(f"{question_id} sent observation has bad outcome")
        elif sent is False:
            if phase == "finalize_full" and attempt != 0:
                raise AuditError(f"{question_id} final candidate attempt mismatch")
            allowed_reason = {
                "retrieve": {"local_request_limit"},
                "finalize_full": {
                    "local_request_limit",
                    "prior_retrieval_overflow",
                },
                "finalize_compact": set(),
            }[phase]
            if (
                outcome != "not_sent"
                or reason not in allowed_reason
                or response_item is not None
            ):
                raise AuditError(f"{question_id} rejected-request evidence mismatch")
            if reason == "local_request_limit" and descriptor["within_local_limit"]:
                raise AuditError(f"{question_id} rejected an in-limit request")
        else:
            raise AuditError(f"{question_id} request sent flag is not Boolean")

        if response_item is None:
            continue
        expected_response_phase = (
            "beam_v8_single" if phase == "retrieve"
            else "beam_v8_single_finalize"
        )
        if response_item["phase"] != expected_response_phase:
            raise AuditError(f"{question_id} response/request phase mismatch")
        calls = response_item["tool_calls"]
        if phase == "retrieve" and calls:
            replay_messages.append({
                "role": "assistant",
                "content": response_item["content"],
                "tool_calls": calls,
            })
            step_traces = traces_by_step.get(response_item["step"], [])
            if len(step_traces) != len(calls):
                raise AuditError(f"{question_id} response/tool replay mismatch")
            for call, trace in zip(calls, step_traces, strict=True):
                replay_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": trace["delivered_text"],
                })
        else:
            answer = _answer_from_content(response_item["content"])
            if answer != expected_answer:
                raise AuditError(f"{question_id} final response/answer mismatch")
            final_answer_seen = True
            if sequence != len(observations):
                raise AuditError(f"{question_id} has requests after the final answer")

    if response_by_request or not final_answer_seen:
        raise AuditError(f"{question_id} interaction replay is incomplete")
    local_token_list = [item["local_tokens"] for item in observations]
    sent_tokens = [
        item["local_tokens"] for item in observations if item["sent"]
    ]
    if (
        context.get("request_local_tokens") != local_token_list
        or context.get("max_request_local_tokens") != max(local_token_list)
        or context.get("max_sent_request_local_tokens") != max(sent_tokens)
        or context.get("rejected_request_candidates")
        != sum(not item["sent"] for item in observations)
    ):
        raise AuditError(f"{question_id} request aggregate mismatch")
    compact_used = any(
        item["phase"] == "finalize_compact" for item in observations
    )
    if context.get("context_compaction_used") is not compact_used:
        raise AuditError(f"{question_id} compaction flag mismatch")
    if compact_used:
        if compact_record is None:
            compact_messages, compact_record = _compact_finalize_messages(
                tokenizer=tokenizer,
                question=question_text,
                evidence=tool_trace,
            )
        if context.get("compact_finalization") != compact_record:
            raise AuditError(f"{question_id} compact-finalization record mismatch")
    elif context.get("compact_finalization") is not None:
        raise AuditError(f"{question_id} has an unexpected compact-finalization record")
    expected_termination = (
        "model_answer"
        if responses[-1]["phase"] == "beam_v8_single"
        else observations[-1]["phase"]
    )
    if interaction.get("termination_reason") != expected_termination:
        raise AuditError(f"{question_id} termination reason mismatch")
    return {
        "raw_tool_tokens": raw_total,
        "delivered_tool_tokens": cumulative,
        "truncated_tool_results": context["truncated_tool_results"],
        "max_sent_request_local_tokens": max(sent_tokens),
        "context_compaction_used": compact_used,
    }


def validate_question(
    question: dict[str, Any],
    *,
    chat_size: str,
    conv_idx: int,
    question_index: int,
    conversation_id: str,
    memory_hash: str,
    memory_dir: Path,
    tokenizer: Any,
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    question_type = question.get("question_type")
    expected_id = f"{chat_size}_{conv_idx}_q{question_index}_{question_type}"
    if question.get("question_id") != expected_id:
        raise AuditError(f"question id mismatch: expected {expected_id!r}")
    exact = {
        "question_index": question_index,
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": conversation_id,
        "memory_sha256": memory_hash,
        "status": "complete",
        "answer_format": "answer_tag",
    }
    for key, value in exact.items():
        if question.get(key) != value:
            raise AuditError(f"{expected_id} has mismatched {key}")
    if question_type not in EXPECTED_QUESTION_TYPES:
        raise AuditError(f"{expected_id} has unknown question type")
    if not str(question.get("question", "")).strip():
        raise AuditError(f"{expected_id} has an empty question")
    if not str(question.get("answer", "")).strip():
        raise AuditError(f"{expected_id} has an empty answer")
    rubric = question.get("rubric")
    if (
        not isinstance(rubric, list)
        or not rubric
        or any(not isinstance(item, str) or not item.strip() for item in rubric)
    ):
        raise AuditError(f"{expected_id} has an invalid or empty rubric")
    if question.get("gold_field") is None or question.get("gold") is None:
        raise AuditError(f"{expected_id} has missing gold metadata")

    retrieval = question.get("retrieval")
    if not isinstance(retrieval, dict):
        raise AuditError(f"{expected_id} has no retrieval metadata")
    steps = int(retrieval.get("steps", 0) or 0)
    calls = int(retrieval.get("calls", 0) or 0)
    tokens_in = int(retrieval.get("tokens_in", 0) or 0)
    response_ids = retrieval.get("response_ids")
    tool_input_errors = retrieval.get("tool_input_errors")
    if steps <= 0 or calls <= 0 or tokens_in <= 0:
        raise AuditError(f"{expected_id} has incomplete retrieval accounting")
    if steps != calls:
        raise AuditError(f"{expected_id} retrieval steps/calls differ")
    if retrieval.get("response_models") != ["gpt-5.5"]:
        raise AuditError(f"{expected_id} did not receive only GPT-5.5 responses")
    if not isinstance(tool_input_errors, list):
        raise AuditError(f"{expected_id} has no tool-input-error ledger")
    for index, item in enumerate(tool_input_errors):
        if (not isinstance(item, dict)
                or set(item) != {"step", "tool", "arguments", "error"}
                or not isinstance(item.get("step"), int)
                or item["step"] < 1 or item["step"] > steps
                or not isinstance(item.get("tool"), str)
                or not isinstance(item.get("arguments"), dict)
                or not str(item.get("error", "")).startswith("Error: ")):
            raise AuditError(
                f"{expected_id} has invalid tool-input error {index}")
    if (
        not isinstance(response_ids, list)
        or len(response_ids) != calls
        or any(
            not isinstance(value, str) or not value.strip() for value in response_ids
        )
        or len(set(response_ids)) != len(response_ids)
    ):
        raise AuditError(f"{expected_id} has invalid response-id evidence")
    context_summary = _validate_context_trace(
        retrieval,
        question_id=expected_id,
        question_text=question["question"],
        expected_answer=question["answer"],
        stored_memories=question.get("memories"),
        memory_dir=memory_dir,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
    )

    return {
        "question_id": expected_id,
        "question_index": question_index,
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": conversation_id,
        "question_type": question_type,
        "difficulty": question.get("difficulty"),
        "question": question["question"],
        "gold_field": question["gold_field"],
        "gold": question["gold"],
        "rubric": rubric,
        "answer": question["answer"],
        "memory_sha256": memory_hash,
        "answer_model": "gpt-5.5",
        "answer_response_ids": response_ids,
        "retrieval": {
            key: retrieval.get(key)
            for key in (
                "latency_s",
                "steps",
                "calls",
                "tokens_in",
                "tokens_out",
                "llm_time_s",
                "tool_input_errors",
            )
        } | {"context_safety": context_summary},
        "source_chat_ids": question.get("source_chat_ids", []),
        "plan_reference": question.get("plan_reference"),
        "abstention_type": question.get("abstention_type"),
        "why_unanswerable": question.get("why_unanswerable"),
    }


def load_proxy_window(
    path: Path, start_value: object, end_value: object
) -> list[dict[str, Any]]:
    start, end = parse_time(start_value), parse_time(end_value)
    if end < start:
        raise AuditError("manifest finished before it was created")
    selected = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                timestamp = parse_time(entry["timestamp"])
            except Exception as exc:  # noqa: BLE001
                raise AuditError(f"invalid proxy log line {line_number}") from exc
            if start <= timestamp <= end:
                selected.append(entry)
    return selected


def _resolve_proxy_log(run_dir: Path, value: Path | None) -> Path:
    if value is None:
        env_value = os.environ.get("CHATGPT_PROXY_LOG")
        value = Path(env_value) if env_value else None
    if value is None:
        raise AuditError(
            "proxy request log is required; pass --proxy-log or set CHATGPT_PROXY_LOG"
        )
    value = value.expanduser()
    if not value.is_absolute():
        root_candidate = (ROOT / value).resolve()
        run_candidate = (run_dir / value).resolve()
        value = root_candidate if root_candidate.is_file() else run_candidate
    value = value.resolve()
    if not value.is_file():
        raise AuditError(f"proxy request log is missing: {value}")
    return value


def _validate_method(
    manifest: dict[str, Any], *, allow_fixture: bool = False
) -> dict[str, Any]:
    method = manifest.get("method")
    if not isinstance(method, dict):
        raise AuditError("manifest method is not an object")
    required = {
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "single_model_retrieve_answer": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "tidy": True,
        "sections": True,
        "article": False,
        "verify": True,
        "merge_lines": True,
        "max_topics": 30,
        "max_rounds": 12,
        "top_k": 20,
        "memory_tool_input_error_policy": "return-validated-error-to-model-v2",
        "context_safety": EXPECTED_CONTEXT_CONFIG,
        "dataset": EXPECTED_DATASET,
        "dataset_revision": EXPECTED_REVISION,
        "calendar": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": method.get(key)}
        for key, expected in required.items()
        if method.get(key) != expected
    }
    if mismatches:
        raise AuditError(f"method configuration mismatch: {mismatches}")
    if allow_fixture:
        if method.get("provider") not in {
            "offline_fixture_no_model_requests",
            None,
        }:
            raise AuditError("fixture method has a formal provider")
    else:
        formal_provider = {
            "provider": "openai_api_flex_via_exclusive_child_proxy",
            "explicit_model_request_authorization": True,
            "provider_evidence_mode": (
                "bounded_gateway_segments_with_exclusive_child_proxy"
            ),
        }
        for key, value in formal_provider.items():
            if method.get(key) != value:
                raise AuditError(f"formal method {key} differs")
        gateway_root = method.get("gateway_root")
        if not isinstance(gateway_root, str) or not Path(gateway_root).is_absolute():
            raise AuditError("formal method gateway_root is invalid")
        if method.get("proxy_request_log") is not None:
            raise AuditError("formal method cannot freeze an arbitrary proxy log")
    build_import = method.get("build_import")
    if manifest.get("build_import") != build_import:
        raise AuditError("manifest/method build-import provenance mismatch")
    if build_import is not None:
        if not isinstance(build_import, dict) or set(build_import) != {
            "root", "run_manifest_sha256"
        }:
            raise AuditError("build-import descriptor is invalid")
        import_root = Path(str(build_import.get("root", ""))).resolve()
        import_manifest = import_root / "run_manifest.json"
        if (ROOT not in import_root.parents or not import_root.is_dir()
                or import_root.is_symlink() or not import_manifest.is_file()
                or import_manifest.is_symlink()
                or sha256_file(import_manifest)
                != build_import.get("run_manifest_sha256")):
            raise AuditError("build-import root identity changed")
    hashes = method.get("source_sha256")
    if not isinstance(hashes, dict):
        raise AuditError("manifest has no source hashes")
    if not REQUIRED_SOURCE_FILES.issubset(hashes):
        missing = sorted(REQUIRED_SOURCE_FILES - set(hashes))
        raise AuditError(f"manifest source hashes are missing {missing}")
    for relative, recorded_hash in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != recorded_hash:
            raise AuditError(f"source hash mismatch: {relative}")
    return method


def audit_provider_evidence(
    run_dir: Path, manifest: dict[str, Any], method: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = manifest.get("provider_evidence")
    if (
        not isinstance(provider, dict)
        or provider.get("schema") != "openai-gpt55-flex-invocations/v1"
        or provider.get("active_run_id") is not None
        or provider.get("gateway_root") != method.get("gateway_root")
    ):
        raise AuditError("manifest Flex provider evidence is missing or active")
    invocations = provider.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise AuditError("manifest has no closed Flex provider invocation")
    evidence_root = (run_dir / "provider_evidence").resolve()
    reports: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, record in enumerate(invocations):
        if not isinstance(record, dict) or not isinstance(record.get("record_path"), str):
            raise AuditError(f"provider invocation {position} is invalid")
        record_path = Path(record["record_path"]).resolve()
        if not record_path.is_relative_to(evidence_root):
            raise AuditError("provider invocation record escapes result evidence root")
        stored = read_json(record_path)
        if record.get("record_sha256") != sha256_file(record_path):
            raise AuditError("provider invocation record hash differs")
        if stored != {
            key: value for key, value in record.items()
            if key not in {"record_path", "record_sha256"}
        }:
            raise AuditError("provider invocation manifest copy differs")
        try:
            reports.append(flex_evidence.audit_invocation(stored))
            records = flex_evidence.load_child_proxy_records(
                Path(stored["consumer_log"]["path"]),
                expected_run_id=str(stored["run_id"]),
            )
        except flex_evidence.EvidenceError as exc:
            raise AuditError(f"provider invocation {position} failed: {exc}") from exc
        for item in records:
            request_id = str(item["gateway_request_id"])
            if request_id in seen:
                raise AuditError("gateway request ID appears in multiple invocations")
            seen.add(request_id)
        consumers.extend(records)
    return reports, consumers


def _audit_unlocked(
    run_dir: Path,
    *,
    proxy_log: Path | None,
    allow_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = reject_symlink_components(run_dir, "run directory")
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest.get("schema_version") != 2:
        raise AuditError("manifest schema version is not 2")
    if manifest.get("status") != "complete":
        raise AuditError(f"run status is {manifest.get('status')!r}, not 'complete'")
    if manifest.get("benchmark") != "BEAM":
        raise AuditError("manifest benchmark is not BEAM")
    if manifest.get("failed_conversations") not in (None, []):
        raise AuditError("manifest reports failed conversations")
    selections = selected_conversations(manifest, allow_fixture=allow_fixture)
    if not allow_fixture and selections != FORMAL_SELECTION:
        raise AuditError(
            "formal BEAM scope must be exactly 100K conversations 0-9 and "
            "all 1M conversations 0-34; "
            f"selected={selections}"
        )
    method = _validate_method(manifest, allow_fixture=allow_fixture)
    tokenizer, tokenizer_identity = _load_context_tokenizer()
    if all(
        manifest["sources"][chat_size].get("kind") == "huggingface"
        for chat_size in selections
    ):
        expected_fingerprint = stable_hash(
            {"benchmark": "BEAM", "method": method, "fixture": None}
        )
        if manifest.get("run_fingerprint") != expected_fingerprint:
            raise AuditError("formal run fingerprint does not match method metadata")
    expected_keys = {
        f"{chat_size}:{conv_idx}"
        for chat_size, indices in selections.items()
        for conv_idx in indices
    }
    manifest_conversations = manifest.get("conversations")
    if not isinstance(manifest_conversations, dict):
        raise AuditError("manifest conversations is not an object")
    if set(manifest_conversations) != expected_keys:
        raise AuditError(
            "manifest conversation keys differ from selected conversations: "
            f"missing={sorted(expected_keys - set(manifest_conversations))}, "
            f"extra={sorted(set(manifest_conversations) - expected_keys)}"
        )

    records: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    build_calls = 0
    qa_calls = 0
    conversation_reports: dict[str, Any] = {}
    split_reports: dict[str, Any] = {}
    for chat_size, indices in selections.items():
        source = validate_source(manifest, chat_size)
        split_start = len(records)
        for conv_idx in indices:
            key = f"{chat_size}:{conv_idx}"
            if manifest_conversations[key].get("status") != "complete":
                raise AuditError(f"{key} is not complete in the manifest")
            conv_dir = run_dir / chat_size / f"conversation_{conv_idx:03d}"
            validate_no_symlinks(conv_dir)
            checkpoint_path = conv_dir / "checkpoint.json"
            results_path = conv_dir / "results.json"
            recorded_checkpoint = manifest_conversations[key].get("checkpoint")
            if (
                not isinstance(recorded_checkpoint, str)
                or not recorded_checkpoint.strip()
            ):
                raise AuditError(f"{key} manifest has no checkpoint path")
            recorded_path = Path(recorded_checkpoint).expanduser()
            if not recorded_path.is_absolute():
                recorded_path = ROOT / recorded_path
            if recorded_path.resolve() != checkpoint_path.resolve():
                raise AuditError(f"{key} manifest checkpoint path mismatch")
            checkpoint = read_json(checkpoint_path)
            if checkpoint.get("schema_version") != 1:
                raise AuditError(f"{key} checkpoint schema is not 1")
            exact = {
                "benchmark": "BEAM",
                "method": "NativeMem-v8.8+calendar",
                "chat_size": chat_size,
                "conversation_index": conv_idx,
                "status": "complete",
                "source": source,
            }
            for field, expected in exact.items():
                if checkpoint.get(field) != expected:
                    raise AuditError(f"{key} checkpoint has mismatched {field}")
            if not str(checkpoint.get("conversation_id", "")).strip():
                raise AuditError(f"{key} has no conversation id")
            if not str(checkpoint.get("input_hash", "")).strip():
                raise AuditError(f"{key} has no input hash")
            config = checkpoint.get("config")
            if not isinstance(config, dict) or stable_hash(config) != checkpoint.get(
                "config_hash"
            ):
                raise AuditError(f"{key} configuration hash mismatch")
            for field, expected in method.items():
                if config.get(field) != expected:
                    raise AuditError(f"{key} config differs from method at {field}")
            if (
                config.get("chat_size") != chat_size
                or config.get("conversation_index") != conv_idx
            ):
                raise AuditError(f"{key} has mismatched per-conversation config")

            _, memory_hash = validate_memory(conv_dir, checkpoint)
            memory_full_hash = full_tree_sha256(conv_dir / "memory")
            validate_source_id_map(conv_dir, checkpoint)
            questions = checkpoint.get("questions")
            if not isinstance(questions, dict) or len(questions) != 20:
                raise AuditError(
                    f"{key} has {len(questions or {})} questions, expected 20"
                )
            if checkpoint.get("question_count") != 20:
                raise AuditError(f"{key} checkpoint question count is not 20")
            type_counts: Counter[str] = Counter()
            for question_index in range(20):
                candidates = [
                    question
                    for question in questions.values()
                    if question.get("question_index") == question_index
                ]
                if len(candidates) != 1:
                    raise AuditError(
                        f"{key} has {len(candidates)} records for question {question_index}"
                    )
                record = validate_question(
                    candidates[0],
                    chat_size=chat_size,
                    conv_idx=conv_idx,
                    question_index=question_index,
                    conversation_id=checkpoint["conversation_id"],
                    memory_hash=memory_hash,
                    memory_dir=conv_dir / "memory",
                    tokenizer=tokenizer,
                    tokenizer_identity=tokenizer_identity,
                )
                duplicate = response_ids.intersection(record["answer_response_ids"])
                if duplicate:
                    raise AuditError(
                        f"response ids are reused across questions: {sorted(duplicate)}"
                    )
                response_ids.update(record["answer_response_ids"])
                type_counts[record["question_type"]] += 1
                qa_calls += int(record["retrieval"]["calls"] or 0)
                records.append(record)
            expected_types = {
                question_type: 2 for question_type in EXPECTED_QUESTION_TYPES
            }
            if dict(type_counts) != expected_types:
                raise AuditError(
                    f"{key} question-type inventory differs: {dict(type_counts)}"
                )

            if read_json(results_path) != expected_result_payload(checkpoint):
                raise AuditError(f"{key} results.json differs from checkpoint.json")
            build_calls += int(checkpoint["build"].get("calls", 0) or 0)
            conversation_reports[key] = {
                "questions": 20,
                "question_type_counts": dict(type_counts),
                "rubric_nuggets": sum(
                    len(record["rubric"]) for record in records[-20:]
                ),
                "memory_sha256": memory_hash,
                "memory_tree_full_sha256": memory_full_hash,
                "build_calls": checkpoint["build"].get("calls"),
                "qa_calls": sum(
                    int(record["retrieval"]["calls"] or 0) for record in records[-20:]
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "results_sha256": sha256_file(results_path),
            }
        split_records = records[split_start:]
        split_reports[chat_size] = {
            "conversations": len(indices),
            "questions": len(split_records),
            "rubric_nuggets": sum(len(record["rubric"]) for record in split_records),
            "question_type_counts": dict(
                Counter(record["question_type"] for record in split_records)
            ),
        }

    expected_questions = len(expected_keys) * 20
    if not allow_fixture and (
        len(expected_keys) != FORMAL_CONVERSATION_COUNT
        or expected_questions != FORMAL_QUESTION_COUNT
    ):
        raise AuditError(
            "formal BEAM inventory is not 45 conversations / 900 questions"
        )
    if len(records) != expected_questions:
        raise AuditError(
            f"collected {len(records)} questions, expected {expected_questions}"
        )
    question_ids = [record["question_id"] for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise AuditError("collected question ids are duplicated")
    total_types = Counter(record["question_type"] for record in records)
    expected_per_type = len(expected_keys) * 2
    if dict(total_types) != {
        question_type: expected_per_type for question_type in EXPECTED_QUESTION_TYPES
    }:
        raise AuditError(f"global question-type inventory differs: {dict(total_types)}")

    provider_reports: list[dict[str, Any]] = []
    resolved_proxy: Path | None = None
    if allow_fixture:
        recorded_value = method.get("proxy_request_log")
        if proxy_log is not None or recorded_value:
            resolved_proxy = _resolve_proxy_log(run_dir, proxy_log)
            recorded_proxy = _resolve_proxy_log(
                run_dir, Path(str(recorded_value))
            )
            if resolved_proxy != recorded_proxy:
                raise AuditError(
                    "proxy log differs from the path frozen in the method "
                    f"configuration: recorded={recorded_proxy}, "
                    f"supplied={resolved_proxy}"
                )
            proxy_entries = load_proxy_window(
                resolved_proxy,
                manifest.get("created_at"),
                manifest.get("finished_at"),
            )
        else:
            proxy_entries = [
                {
                    "status": "success",
                    "requested_model": "gpt-5.5",
                    "actual_model": "gpt-5.5",
                    "response_id": response_id,
                    "attempts": 1,
                    "unsupported_parameters": [],
                }
                for response_id in sorted(response_ids)
            ]
    else:
        if proxy_log is not None:
            raise AuditError("formal audit rejects --proxy-log; use provider evidence")
        provider_reports, provider_records = audit_provider_evidence(
            run_dir, manifest, method
        )
        proxy_entries = []
        for record in provider_records:
            normalized = dict(record)
            normalized.setdefault("attempts", 1)
            normalized.setdefault("unsupported_parameters", [])
            proxy_entries.append(normalized)
    successes = [entry for entry in proxy_entries if entry.get("status") == "success"]
    proxy_upstream_attempts = 0
    unsupported_parameters: set[str] = set()
    max_completion_tokens = 0
    responses_over_1200 = 0
    for position, entry in enumerate(proxy_entries):
        attempts = entry.get("attempts")
        unsupported = entry.get("unsupported_parameters")
        if (
            entry.get("status") not in {"success", "error"}
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts <= 0
            or not isinstance(unsupported, list)
            or any(not isinstance(value, str) for value in unsupported)
        ):
            raise AuditError(f"proxy window entry {position} has invalid attempts")
        proxy_upstream_attempts += attempts
        unsupported_parameters.update(unsupported)
        if entry.get("status") == "success":
            completion = (entry.get("usage") or {}).get("completion_tokens")
            if isinstance(completion, int) and not isinstance(completion, bool):
                max_completion_tokens = max(max_completion_tokens, completion)
                responses_over_1200 += int(completion > 1200)
    success_by_id: dict[str, dict[str, Any]] = {}
    for entry in successes:
        response_id = str(entry.get("response_id", "")).strip()
        if response_id not in response_ids:
            continue
        if response_id in success_by_id:
            raise AuditError(f"duplicate matched QA proxy response id: {response_id}")
        if (
            entry.get("requested_model") != "gpt-5.5"
            or entry.get("actual_model") != "gpt-5.5"
        ):
            raise AuditError(f"matched QA proxy response {response_id} is not GPT-5.5")
        success_by_id[response_id] = entry
    missing_ids = sorted(response_ids - set(success_by_id))
    if missing_ids:
        raise AuditError(
            f"{len(missing_ids)} QA response ids are absent from the proxy log; "
            f"first={missing_ids[0]}"
        )
    recorded_calls = build_calls + qa_calls
    matched_upstream_attempts = sum(
        entry["attempts"] for entry in success_by_id.values()
    )
    matched_unsupported = sorted({
        value
        for entry in success_by_id.values()
        for value in entry["unsupported_parameters"]
    })

    records.sort(
        key=lambda record: (
            ("100K", "1M").index(record["chat_size"]),
            record["conversation_index"],
            record["question_index"],
        )
    )
    evaluation_input = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "metric_scope": "rubric_nugget_only",
        "method": "NativeMem-v8.8+calendar",
        "answer_model": "gpt-5.5",
        "run_dir": str(run_dir),
        "run_manifest_sha256": manifest_sha256,
        "formal_scope_verified": not allow_fixture,
        "selected_conversations": selections,
        "question_count": len(records),
        "rubric_nugget_count": sum(len(record["rubric"]) for record in records),
        "question_type_counts": dict(total_types),
        "records": records,
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "BEAM",
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "run_dir": str(run_dir),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": manifest_sha256,
        "evaluation_input_path": str((run_dir / "evaluation_input.json").resolve()),
        "selected_conversations": selections,
        "formal_scope_verified": not allow_fixture,
        "conversation_count": len(expected_keys),
        "questions": len(records),
        "question_type_counts": dict(total_types),
        "rubric_nuggets": evaluation_input["rubric_nugget_count"],
        "empty_answers": 0,
        "source_hashes_match": True,
        "splits": split_reports,
        "conversations": conversation_reports,
        "model_evidence": {
            "recorded_build_calls": build_calls,
            "recorded_qa_calls": qa_calls,
            "recorded_total_calls": recorded_calls,
            "build": {
                "source": "checkpoint_and_atomic_build_marker",
                "requested_models": ["gpt-5.5"],
                "response_models": ["gpt-5.5"],
                "response_id_linkage": (
                    "gateway_child_consumer_log" if not allow_fixture
                    else "unavailable_offline_fixture"
                ),
                "proxy_calls_used_as_substitute": 0,
            },
            "qa": {
                "source": (
                    "checkpoint_response_ids_matched_to_gateway_child_log"
                    if not allow_fixture else "offline_fixture"
                ),
                "requested_models": ["gpt-5.5"],
                "response_models": ["gpt-5.5"],
                "content_hash_linkage": not allow_fixture,
                "logical_proxy_requests": len(success_by_id),
                "upstream_http_attempts": matched_upstream_attempts,
                "upstream_internal_retries": (
                    matched_upstream_attempts - len(success_by_id)
                ),
                "unsupported_parameters": matched_unsupported,
                "requested_output_limit_enforced": (
                    "max_output_tokens" not in matched_unsupported
                ),
            },
            "qa_response_ids": len(response_ids),
            "qa_response_ids_matched_in_proxy_log": len(response_ids),
            "provider_evidence": {
                "mode": (
                    "bounded_gateway_segments" if not allow_fixture
                    else "offline_fixture"
                ),
                "explicit_model_request_authorization": (
                    method.get("explicit_model_request_authorization")
                ),
                "consumer_provider_correspondence": not allow_fixture,
                "invocations": provider_reports,
                "gateway_request_ids": [
                    entry.get("gateway_request_id") for entry in proxy_entries
                    if entry.get("gateway_request_id")
                ],
                "response_ids": [
                    entry.get("response_id") for entry in proxy_entries
                ],
            },
            "proxy_window": {
                "path": str(resolved_proxy) if resolved_proxy else None,
                "entries": len(proxy_entries),
                "successes": len(successes),
                "errors": len(proxy_entries) - len(successes),
                "matched_qa_successes": len(success_by_id),
                "matched_requested_models": ["gpt-5.5"],
                "matched_actual_models": ["gpt-5.5"],
                "unrelated_entries_ignored": len(proxy_entries) - len(success_by_id),
                "logical_proxy_requests": len(proxy_entries),
                "upstream_http_attempts": proxy_upstream_attempts,
                "upstream_internal_retries": proxy_upstream_attempts - len(proxy_entries),
                "unsupported_parameters": sorted(unsupported_parameters),
                "requested_output_limit_enforced": (
                    "max_output_tokens" not in unsupported_parameters
                ),
                "max_observed_completion_tokens": max_completion_tokens,
                "responses_over_requested_1200": responses_over_1200,
            },
        },
        "scoring_scope": {
            "primary": "rubric_nugget_mean",
            "event_ordering": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
            "reason": (
                "No verified official tau-b times F1 implementation is used. "
                "The vendored runner aligns generated events with an LLM and "
                "averages normalized tau-b with nugget score, which is not the "
                "requested official tau-b times F1 metric."
            ),
        },
        "evaluation_input_sha256": None,
    }
    return evaluation_input, report


def audit(
    run_dir: Path,
    *,
    proxy_log: Path | None,
    allow_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = reject_symlink_components(run_dir, "run directory")
    locks = acquire_audit_locks(run_dir)
    try:
        return _audit_unlocked(
            run_dir, proxy_log=proxy_log, allow_fixture=allow_fixture
        )
    finally:
        release_audit_locks(locks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly audit a completed GPT-5.5 BEAM run"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--proxy-log",
        type=Path,
        help="chatgpt_proxy JSONL request log (or set CHATGPT_PROXY_LOG)",
    )
    parser.add_argument("--allow-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir = reject_symlink_components(args.run_dir, "run directory")
    audit_path = run_dir / "audit.json"
    audit_path.unlink(missing_ok=True)
    locks: tuple[Any, Any] | None = None
    try:
        locks = acquire_audit_locks(run_dir)
        evaluation_input, report = _audit_unlocked(
            run_dir, proxy_log=args.proxy_log, allow_fixture=args.allow_fixture
        )
        input_path = run_dir / "evaluation_input.json"
        atomic_json(input_path, evaluation_input)
        report["evaluation_input_sha256"] = sha256_file(input_path)
        atomic_json(audit_path, report)
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    finally:
        release_audit_locks(locks)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
