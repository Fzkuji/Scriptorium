#!/usr/bin/env python3
"""Run NativeMem v8.8+calendar on LongMemEval-S with GPT-5.5.

Each LongMemEval question owns a separate memory directory and checkpoint.
Completed checkpoints are immutable: a repeated invocation skips them, while
``--resume`` permits retrying an interrupted build or answer stage.  All JSON
state is written through ``os.replace`` so a terminated process cannot leave a
half-written checkpoint that looks complete.

The script deliberately imports the NativeMem backend only after the model and
method environment has been frozen.  Importing this module from an offline unit
test therefore never creates a model request.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


DEFAULT_DATA = (
    ROOT / "benchmarks" / "longmemeval" / "data" /
    "longmemeval_s_cleaned.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "v88-calendar-gpt55-longmemeval-s-20260714"
SCHEMA_VERSION = 1
EXPECTED_LONGMEMEVAL_SIZE = 500

LME_SINGLE_PROMPT = """You are answering one LongMemEval question by reading a NativeMem library.
The bash tool is read-only and its working directory is the library root.

The library has two complementary views:
- topics/: topic-oriented memory files with inline [Dn:m] source anchors.
- timeline/YYYY/MM/DD.md: chronological atomic events with [Dn:m] anchors.
Use read_original with the complete anchor string when exact wording or nearby
turns are needed. Never infer an answer only from a file name.

Library structure:
{structure}

Current Date: {question_date}
Question: {question}

Procedure:
1. Read all semantically relevant topic files. Use several grep terms only as
   a fallback; wording in the question and memory may differ.
2. For temporal, update, counting, or multi-session questions, inspect every
   relevant event and verify source turns with read_original.
3. Prefer the most recent applicable fact when memories conflict. Compute
   dates or quantities only from recorded facts and the Current Date.
4. If the history genuinely lacks the requested information, say so.
   Do not invent a fact. This rule is required for abstention questions.
5. Return a direct and complete answer. Output exactly one
   <answer>...</answer> block after tool use. Do not expose reasoning.
"""

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_READ_ONLY_COMMANDS = {
    "cat", "cut", "find", "grep", "head", "ls", "pwd", "rg", "sed",
    "sort", "tail", "uniq", "wc",
}

METHOD_ENV = {
    "NATIVEMEM_PROMPT": "v8",
    "NATIVEMEM_STORE_MODE": "oneshot",
    "NATIVEMEM_V8_SINGLE": "1",
    "NATIVEMEM_CHUNK_TURNS": "6",
    "NATIVEMEM_V8_SEGMENT": "fixed",
    "NATIVEMEM_V8_TIDY": "on",
    "NATIVEMEM_V8_SECTIONS": "on",
    "NATIVEMEM_V8_ARTICLE": "off",
    "NATIVEMEM_V8_VERIFY": "on",
    "NATIVEMEM_V8_MERGE_LINES": "on",
    "NATIVEMEM_V8_TIDY_COMBINED": "off",
    "NATIVEMEM_V8_MAX_TOPICS": "30",
    "NATIVEMEM_TOPK": "20",
    "NATIVEMEM_V8_MAX_ROUNDS": "12",
    "NATIVEMEM_V8_MAX_TOKENS": "1200",
    "NATIVEMEM_V8_READ_CONTEXT": "1",
    "NATIVEMEM_V8_MAP": "dir",
    "NATIVEMEM_V8_MAP_INLINE": "8",
    "NATIVEMEM_V8_MAX_DEPTH": "4",
    "NATIVEMEM_V8_REWRITE_MIN": "8",
    "NATIVEMEM_V8_SEGMENT_MAX": "10",
}


class DataValidationError(ValueError):
    """The benchmark item cannot be represented without losing information."""


class ExistingStateError(RuntimeError):
    """An incomplete item exists and the caller did not authorize resume."""


class EmptyStageError(RuntimeError):
    """A swallowed backend/API failure produced an empty stage result."""


def _retry_delay(exc: Exception, attempt: int) -> int:
    return 30 if type(exc).__name__ == "RateLimitError" else attempt


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    """Atomically replace ``path`` with UTF-8 JSON and fsync the new file."""
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


_LME_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")


def normalize_lme_date(raw: object) -> str:
    """Convert a LongMemEval timestamp to the ISO day used by v8."""
    match = _LME_DATE_RE.search(str(raw or "").strip())
    if not match:
        raise DataValidationError(f"unparseable LongMemEval date: {raw!r}")
    year, month, day = map(int, match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise DataValidationError(f"invalid LongMemEval date: {raw!r}") from exc


def validate_item(item: object, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DataValidationError(f"item {index} is not an object")
    required = ("question_id", "question_type", "question", "answer",
                "question_date", "answer_session_ids",
                "haystack_sessions", "haystack_dates",
                "haystack_session_ids")
    missing = [key for key in required if key not in item]
    if missing:
        raise DataValidationError(f"item {index} missing fields: {missing}")
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    session_ids = item["haystack_session_ids"]
    if not isinstance(sessions, list) or not sessions:
        raise DataValidationError(f"item {index} has no haystack sessions")
    if not isinstance(dates, list) or not isinstance(session_ids, list):
        raise DataValidationError(f"item {index} session metadata is not a list")
    if len(sessions) != len(dates) or len(sessions) != len(session_ids):
        raise DataValidationError(
            f"item {index} session/date/id lengths differ: "
            f"{len(sessions)}/{len(dates)}/{len(session_ids)}"
        )
    if not str(item["question_id"]).strip() or not str(item["question"]).strip():
        raise DataValidationError(f"item {index} has an empty question id/text")
    for raw_date in dates:
        normalize_lme_date(raw_date)
    if item.get("question_date"):
        normalize_lme_date(item["question_date"])
    return item


def longmemeval_to_locomo(item: object, index: int = 0) -> dict[str, Any]:
    """Convert one LongMemEval history into ``build_memory``'s conversation.

    Session order is preserved because LongMemEval-S already supplies histories
    in timestamp order.  Every source turn receives a deterministic ``Dn:m``
    anchor, including assistant turns required by the assistant-memory subset.
    """
    sample = validate_item(item, index)
    conv: dict[str, Any] = {"speaker_a": "user", "speaker_b": "assistant"}
    for session_number, (session, raw_date) in enumerate(
        zip(sample["haystack_sessions"], sample["haystack_dates"]), start=1
    ):
        if not isinstance(session, list) or not session:
            raise DataValidationError(
                f"item {index} session {session_number} is empty or not a list"
            )
        turns = []
        for turn_number, turn in enumerate(session, start=1):
            if not isinstance(turn, dict):
                raise DataValidationError(
                    f"item {index} session {session_number} turn "
                    f"{turn_number} is not an object"
                )
            role = str(turn.get("role", "")).strip()
            content = turn.get("content")
            # The cleaned S file contains a small number of explicit empty
            # messages.  Keep them so source turn positions and Dn:m anchors
            # remain stable; split_into_chunks_structured ignores empty text
            # during build, while build_turn_index still preserves the mapping.
            if not role or not isinstance(content, str):
                raise DataValidationError(
                    f"item {index} session {session_number} turn "
                    f"{turn_number} has an empty role or non-string content"
                )
            turns.append({
                "speaker": role,
                "text": content,
                "dia_id": f"D{session_number}:{turn_number}",
            })
        conv[f"session_{session_number}"] = turns
        conv[f"session_{session_number}_date_time"] = normalize_lme_date(raw_date)
    return conv


def load_dataset(path: Path, expected_count: int | None = None) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise DataValidationError("LongMemEval data root must be a list")
    if expected_count is not None and len(data) != expected_count:
        raise DataValidationError(
            f"expected {expected_count} LongMemEval items, found {len(data)}"
        )
    for index, item in enumerate(data):
        validate_item(item, index)
    return data


def validate_longmemeval_s(data: list[dict[str, Any]]) -> None:
    """Reject the 500-item oracle file, which has the same top-level schema."""
    mean_sessions = sum(len(item["haystack_sessions"]) for item in data) / len(data)
    if mean_sessions < 20:
        raise DataValidationError(
            f"dataset averages only {mean_sessions:.1f} sessions/item; this is "
            "likely LongMemEval-oracle, not LongMemEval-S"
        )


def select_indices(total: int, start: int, limit: int | None) -> list[int]:
    if start < 0 or start >= total:
        raise DataValidationError(f"--start must be in 0..{total - 1}")
    if limit is not None and limit < 1:
        raise DataValidationError("--limit must be positive")
    stop = total if limit is None else min(total, start + limit)
    return list(range(start, stop))


def safe_question_id(question_id: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")
    return cleaned[:100] or "unknown"


def model_question(item: dict[str, Any]) -> str:
    """Match LongMemEval's official answer prompt by exposing current date."""
    return (
        f"Current Date: {item['question_date']}\n"
        f"Question: {item['question']}"
    )


def validate_read_only_command(command: object) -> tuple[bool, str]:
    """Allow a small read-only shell grammar rooted in the memory directory."""
    if not isinstance(command, str) or not command.strip():
        return False, "empty command"
    if any(marker in command for marker in (
        "\n", "\r", "`", "$", ";", "&&", "||", ">", "<", "(", ")"
    )):
        return False, "shell control or redirection is not allowed"
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False, "command cannot be parsed"
    if not tokens:
        return False, "empty command"
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "|":
            if not segments[-1]:
                return False, "empty pipeline segment"
            segments.append([])
        else:
            segments[-1].append(token)
    if not segments[-1]:
        return False, "empty pipeline segment"
    for segment in segments:
        program = segment[0]
        if program not in _READ_ONLY_COMMANDS:
            return False, f"command {program!r} is not read-only"
        for token in segment[1:]:
            if token.startswith(("/", "~")):
                return False, "absolute and home paths are not allowed"
            if re.search(r"(^|/)\.\.($|/)", token):
                return False, "parent-directory traversal is not allowed"
            if re.search(r"(^|/)(raw|sources)($|/)", token):
                return False, "source transcript paths are not available"
        if program == "find" and any(
            token in {"-delete", "-exec", "-execdir", "-ok", "-okdir",
                      "-fprint", "-fprintf", "-fls"}
            for token in segment[1:]
        ):
            return False, "mutating find actions are not allowed"
        if program == "sed":
            options = [token for token in segment[1:] if token.startswith("-")]
            if any(token != "-n" for token in options):
                return False, "only sed -n is allowed"
            scripts = [
                token for token in segment[1:]
                if not token.startswith("-") and not Path(token).suffix == ".md"
                and "/" not in token
            ]
            if scripts and not all(re.fullmatch(r"\d+(,\d+)?p", token)
                                   for token in scripts):
                return False, "only numeric sed print ranges are allowed"
        if program == "sort" and any(
            token == "-o" or token.startswith("--output") for token in segment[1:]
        ):
            return False, "sort output files are not allowed"
    return True, "ok"


def normalize_workspace_command(command: object, memory_dir: Path) -> object:
    if not isinstance(command, str):
        return command
    root = str(memory_dir.resolve())
    return command.replace(f"{root}/", "./").replace(root, ".")


def execute_read_only_bash(backend: Any, args: object, memory_dir: Path) -> str:
    args = args if isinstance(args, dict) else {}
    command = normalize_workspace_command(args.get("command", ""), memory_dir)
    allowed, reason = validate_read_only_command(command)
    if not allowed:
        return f"Command rejected: {reason}. Use read-only ls/cat/grep/find/head/tail/wc."
    return backend.execute_tool(
        "bash", {"command": command}, str(memory_dir), hide_raw=True
    )


def collect_and_answer_longmemeval(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    *,
    prompt_template: str | None = None,
    review_prompt: str | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    """Answer from a disposable full-memory workspace."""
    with tempfile.TemporaryDirectory(prefix="nativemem-qa-") as temporary:
        workspace = Path(temporary) / "memory"
        shutil.copytree(memory_dir, workspace)
        return _collect_and_answer_longmemeval(
            backend,
            item,
            workspace,
            turn_index,
            prompt_template=prompt_template,
            review_prompt=review_prompt,
        )


def execute_workspace_bash(command: object, memory_dir: Path) -> str:
    if not isinstance(command, str) or not command.strip():
        return "Error: empty command"
    profile = (
        "(version 1) (allow default) (deny network*) "
        "(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\")) "
        "(deny file-write* (subpath \"/Users\") (subpath \"/Applications\") "
        "(subpath \"/Library\") (subpath \"/System\") (subpath \"/opt\") "
        "(subpath \"/usr\") (subpath \"/bin\") (subpath \"/sbin\") "
        "(subpath \"/private/etc\") (subpath \"/Volumes\"))"
    )
    try:
        result = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, "/bin/bash", "-c", command],
            cwd=memory_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    output = f"{result.stdout}{result.stderr}".strip()
    return output or "(no output)"


def _collect_and_answer_longmemeval(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    *,
    prompt_template: str | None = None,
    review_prompt: str | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    """Use the v8 single-agent retrieval loop with LongMemEval answer rules."""
    injected = getattr(backend, "collect_and_answer_longmemeval", None)
    if callable(injected) and prompt_template is None and review_prompt is None:
        return injected(item, memory_dir, turn_index)

    max_rounds = int(os.environ.get("NATIVEMEM_V8_MAX_ROUNDS", "12"))
    max_tokens = int(os.environ.get("NATIVEMEM_V8_MAX_TOKENS", "1200"))
    read_context = int(os.environ.get("NATIVEMEM_V8_READ_CONTEXT", "1"))
    prompt = (prompt_template or LME_SINGLE_PROMPT).format(
        structure=backend.memory_structure_map(str(memory_dir)),
        memory_root=str(memory_dir.resolve()),
        question_date=item["question_date"],
        question=item["question"],
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools = copy.deepcopy(backend.TOOLS) + [copy.deepcopy(backend._V8_READ_TOOL)]
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") == "bash":
            function["description"] = (
                "Run a shell command inside the disposable memory workspace."
            )
    collected: list[str] = []
    trace: list[dict[str, Any]] = []
    final_text = ""
    steps = 0
    consecutive_errors = 0
    review_requested = False
    invalid_outputs: list[dict[str, Any]] = []
    thinking = os.environ.get("NATIVEMEM_THINKING")
    provider_options = (
        {"extra_body": {"thinking": {"type": thinking}}} if thinking else {}
    )
    for _round in range(max_rounds + (2 if review_prompt else 0)):
        try:
            response = backend.client.chat.completions.create(
                model=backend.ALIYUN_MODEL,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=0.0,
                **provider_options,
            )
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            trace.append({"type": "model_error", "error": type(exc).__name__})
            if consecutive_errors >= 3:
                break
            time.sleep(_retry_delay(exc, consecutive_errors))
            continue
        consecutive_errors = 0
        backend.log_usage(response, phase="longmemeval_single")
        steps += 1
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        content = message.content or ""
        if not tool_calls:
            candidate = (
                "" if getattr(choice, "finish_reason", None) == "length" else content
            )
            if review_prompt and not review_requested and candidate.strip():
                review_requested = True
                trace.append({"type": "same_model_review"})
                messages.extend([
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": review_prompt},
                ])
                continue
            final_text = candidate
            if not _ANSWER_RE.search(candidate):
                invalid_outputs.append({
                    "phase": "retrieval",
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_length": len(content),
                    "content_preview": content[:160],
                })
            break
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        })
        for tool_call in tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                args = {}
            name = tool_call.function.name
            if name == "read_original":
                dia_ids = args.get("dia_ids", [])
                if isinstance(dia_ids, str):
                    dia_ids = [dia_ids]
                if not isinstance(dia_ids, list):
                    dia_ids = []
                original = backend.read_turns(
                    turn_index, dia_ids, context=read_context
                )
                event_lines = backend.memory_event_lines(str(memory_dir), dia_ids)
                block = (
                    f"{event_lines}\n{original}" if event_lines else original
                )
                if block.strip():
                    collected.append(block)
                output = original or "(no matching source turns)"
                trace.append({"type": "read_original", "dia_ids": dia_ids})
            elif name == "bash":
                output = execute_workspace_bash(args.get("command", ""), memory_dir)
                trace.append({
                    "type": "bash", "command": args.get("command", ""),
                    "accepted": True, "sandboxed": True,
                })
            else:
                output = "Unknown tool"
                trace.append({"type": "unknown_tool", "name": name})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })

    match = _ANSWER_RE.search(final_text)
    if not match:
        finalize = {
            "role": "user",
            "content": (
                "Tool use has ended. Answer the user's question directly "
                "using the information above. "
                "Output exactly one <answer>...</answer> block."
            ),
        }
        for attempt in range(3):
            try:
                response = backend.client.chat.completions.create(
                    model=backend.ALIYUN_MODEL,
                    messages=messages + [finalize],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    **provider_options,
                )
            except Exception as exc:  # noqa: BLE001
                trace.append({
                    "type": "finalize_error", "attempt": attempt + 1,
                    "error": type(exc).__name__,
                })
                time.sleep(_retry_delay(exc, attempt + 1))
                continue
            backend.log_usage(response, phase="longmemeval_finalize")
            steps += 1
            choice = response.choices[0]
            candidate = choice.message.content or ""
            match = (
                None if getattr(choice, "finish_reason", None) == "length"
                else _ANSWER_RE.search(candidate)
            )
            if match:
                final_text = candidate
                break
            invalid_outputs.append({
                "phase": "finalize",
                "finish_reason": getattr(choice, "finish_reason", None),
                "content_length": len(candidate),
                "content_preview": candidate[:160],
            })
    if not match or not match.group(1).strip():
        model_errors = [
            event for event in trace
            if event.get("type") in {"model_error", "finalize_error"}
        ]
        raise EmptyStageError(
            "LongMemEval answer did not contain a complete non-empty <answer> "
            f"block; invalid_outputs={json.dumps(invalid_outputs[-4:])}; "
            f"model_errors={json.dumps(model_errors[-8:])}"
        )
    memories = [{"text": text, "date": ""} for text in collected]
    return memories, steps, match.group(1).strip(), trace


def item_paths(output_dir: Path, index: int, question_id: object) -> dict[str, Path]:
    item_dir = output_dir / "items" / f"{index:04d}_{safe_question_id(question_id)}"
    return {
        "item_dir": item_dir,
        "memory_dir": item_dir / "memory",
        "checkpoint": item_dir / "checkpoint.json",
    }


def method_config(request_concurrency: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        key: value for key, value in METHOD_ENV.items()
    }
    config["NATIVEMEM_V8_CONCURRENCY"] = str(request_concurrency)
    return config


def configure_environment(args: argparse.Namespace) -> dict[str, str]:
    """Freeze the current v8.8 path before importing NativeMem modules."""
    config = method_config(args.request_concurrency)
    os.environ.update(config)
    os.environ.update({
        # nativemem.py gives MODEL precedence over BUILDER_MODEL.  Set both so
        # an inherited shell value cannot silently change the backbone.
        "MODEL": args.model,
        "BUILDER_MODEL": args.model,
        "BUILDER_BASE": args.base_url,
        "BUILDER_KEY": args.api_key,
        "ALIYUN_KEY": args.api_key,
        "NATIVEMEM_TRUST_PROXY": "1" if args.trust_proxy else "0",
    })
    # The calendar-enhanced v8 path is the selected method.  An inherited v9
    # switch would silently replace its builder with the two-tier pipeline.
    os.environ.pop("NATIVEMEM_V9_PIPELINE", None)
    os.environ.pop("NATIVEMEM_V9_SCRIBE_MODE", None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    return config


def load_backend() -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import src.adapters.run_nativemem as backend  # noqa: PLC0415
    return backend


def verify_backend_contract(backend: Any) -> None:
    required = ("build_memory", "build_turn_index")
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise RuntimeError(f"NativeMem backend missing required interfaces: {missing}")
    if not callable(getattr(backend, "collect_and_answer_longmemeval", None)):
        answer_requirements = (
            "memory_structure_map", "execute_tool", "read_turns",
            "memory_event_lines", "log_usage",
        )
        missing_answer = [
            name for name in answer_requirements
            if not callable(getattr(backend, name, None))
        ]
        missing_answer += [
            name for name in ("client", "ALIYUN_MODEL", "TOOLS", "_V8_READ_TOOL")
            if getattr(backend, name, None) is None
        ]
        if missing_answer:
            raise RuntimeError(
                "NativeMem backend missing LongMemEval answer interfaces: "
                f"{missing_answer}"
            )
    prompt = getattr(getattr(backend, "v8_memory", None),
                     "_V8_DISTILL_PROMPT", "")
    if "{calendar}" not in prompt or "日历" not in prompt:
        raise RuntimeError(
            "src/v8_memory.py does not contain the v8 calendar-enhanced prompt"
        )


def tracker_reset(backend: Any, phase: str) -> None:
    tracker = getattr(backend, "tracker", None)
    if tracker is not None:
        tracker.reset(phase)


def tracker_snapshot(backend: Any, phase: str) -> dict[str, Any]:
    tracker = getattr(backend, "tracker", None)
    if tracker is None:
        return {"calls": None, "tokens_in": None, "tokens_out": None,
                "llm_time_s": None}
    return tracker.snapshot(phase)


def memory_stats(memory_dir: Path) -> dict[str, int]:
    files = [path for path in memory_dir.rglob("*.md") if path.is_file()]
    return {
        "markdown_files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def memory_is_valid(memory_dir: Path) -> bool:
    stats = memory_stats(memory_dir) if memory_dir.is_dir() else {}
    return bool(stats.get("markdown_files", 0) and stats.get("bytes", 0))


def checkpoint_is_complete(
    checkpoint: object, item: dict[str, Any], index: int, memory_dir: Path
) -> tuple[bool, str]:
    if not isinstance(checkpoint, dict):
        return False, "not_an_object"
    if checkpoint.get("status") != "complete":
        return False, f"status={checkpoint.get('status', 'missing')}"
    if checkpoint.get("dataset_index") != index:
        return False, "dataset_index_mismatch"
    if str(checkpoint.get("question_id")) != str(item["question_id"]):
        return False, "question_id_mismatch"
    answer = checkpoint.get("answer", {})
    if not isinstance(answer, dict) or not str(answer.get("hypothesis", "")).strip():
        return False, "empty_answer"
    build = checkpoint.get("build", {})
    retrieval = checkpoint.get("retrieval", {})
    if not isinstance(build, dict) or build.get("status") != "complete":
        return False, "build_incomplete"
    if not isinstance(retrieval, dict) or retrieval.get("status") != "complete":
        return False, "retrieval_incomplete"
    if not memory_is_valid(memory_dir):
        return False, "memory_missing_or_empty"
    return True, "complete"


def base_checkpoint(
    item: dict[str, Any], index: int, paths: dict[str, Path], run_meta: dict[str, Any]
) -> dict[str, Any]:
    sessions = item["haystack_sessions"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "dataset_index": index,
        "question_id": str(item["question_id"]),
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "question_date": (
            normalize_lme_date(item["question_date"])
            if item.get("question_date") else None
        ),
        "question_date_raw": item.get("question_date"),
        "answer_session_ids": item.get("answer_session_ids", []),
        "input": {
            "sessions": len(sessions),
            "turns": sum(len(session) for session in sessions),
            "empty_source_turns": sum(
                not str(turn.get("content", "")).strip()
                for session in sessions for turn in session
            ),
            "source_session_ids": item["haystack_session_ids"],
            "source_dates": item["haystack_dates"],
        },
        "method": run_meta["method"],
        "models": run_meta["models"],
        "config": run_meta["config"],
        "code": run_meta["code"],
        "request_audit": run_meta.get("request_audit", {}),
        "paths": {
            "item_dir": str(paths["item_dir"]),
            "memory_dir": str(paths["memory_dir"]),
            "checkpoint": str(paths["checkpoint"]),
        },
        "created_at": utc_now(),
    }


def _record_failure(
    checkpoint: dict[str, Any], checkpoint_path: Path, stage: str, exc: Exception
) -> None:
    checkpoint["status"] = "failed"
    checkpoint["updated_at"] = utc_now()
    checkpoint["error"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    atomic_json(checkpoint_path, checkpoint)


def run_item(
    index: int,
    item: dict[str, Any],
    output_dir: Path,
    backend: Any,
    run_meta: dict[str, Any],
    resume: bool,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Build and answer one independent LongMemEval item."""
    paths = item_paths(output_dir, index, item["question_id"])
    old: dict[str, Any] | None = None
    if paths["checkpoint"].exists():
        try:
            loaded = read_json(paths["checkpoint"])
            old = loaded if isinstance(loaded, dict) else None
        except Exception:  # noqa: BLE001
            old = None
    complete, reason = checkpoint_is_complete(old, item, index, paths["memory_dir"])
    if complete:
        return True, "already complete", old
    if old and old.get("status") == "complete":
        raise ExistingStateError(
            f"item {index} has a completed but invalid checkpoint ({reason}); "
            "it will not be overwritten"
        )
    if paths["item_dir"].exists() and not resume:
        raise ExistingStateError(
            f"item {index} has existing incomplete state ({reason}); rerun with --resume"
        )

    paths["item_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint = base_checkpoint(item, index, paths, run_meta)
    authoritative = dict(checkpoint)
    if old:
        if (old.get("dataset_index") not in (None, index)
                or str(old.get("question_id", item["question_id"]))
                != str(item["question_id"])):
            raise ExistingStateError(
                f"item {index} checkpoint identity does not match the dataset"
            )
        checkpoint.update(old)
    for key in ("schema_version", "dataset_index", "question_id",
                "question_type", "question", "gold", "question_date",
                "question_date_raw", "answer_session_ids", "input", "paths"):
        checkpoint[key] = authoritative[key]
    # Refresh immutable experiment metadata when resuming an incomplete item,
    # but preserve the original creation timestamp and completed build record.
    checkpoint.update({
        "method": run_meta["method"],
        "models": run_meta["models"],
        "config": run_meta["config"],
        "code": run_meta["code"],
        "request_audit": run_meta.get("request_audit", {}),
        "updated_at": utc_now(),
    })
    checkpoint.pop("error", None)

    try:
        conv = longmemeval_to_locomo(item, index)
    except Exception as exc:  # noqa: BLE001
        _record_failure(checkpoint, paths["checkpoint"], "normalize", exc)
        return False, str(exc), checkpoint

    prior_build = checkpoint.get("build", {})
    build_reusable = (
        isinstance(prior_build, dict)
        and prior_build.get("status") == "complete"
        and memory_is_valid(paths["memory_dir"])
    )
    if not build_reusable:
        if paths["memory_dir"].exists():
            shutil.rmtree(paths["memory_dir"])
        checkpoint["status"] = "building"
        checkpoint["build"] = {"status": "running", "started_at": utc_now()}
        atomic_json(paths["checkpoint"], checkpoint)
        phase = f"lme_{index}_build"
        tracker_reset(backend, phase)
        wall_start = time.monotonic()
        usage = None
        try:
            reported_seconds, events = backend.build_memory(
                conv, str(paths["memory_dir"])
            )
            usage = tracker_snapshot(backend, phase)
            stats = memory_stats(paths["memory_dir"])
            if int(events) <= 0 or stats["markdown_files"] <= 0 or stats["bytes"] <= 0:
                raise EmptyStageError(
                    "v8 build returned no events or produced an empty memory directory"
                )
            checkpoint["build"] = {
                "status": "complete",
                "started_at": checkpoint["build"]["started_at"],
                "finished_at": utc_now(),
                "reported_time_s": round(float(reported_seconds), 3),
                "wall_time_s": round(time.monotonic() - wall_start, 3),
                "events": int(events),
                "usage": usage,
                **stats,
            }
            checkpoint["status"] = "built"
            checkpoint["updated_at"] = utc_now()
            atomic_json(paths["checkpoint"], checkpoint)
        except Exception as exc:  # noqa: BLE001
            # Snapshot even on failure; the tracker is then removed and cannot
            # contaminate a later item.
            checkpoint["build"]["usage"] = (
                usage if usage is not None else tracker_snapshot(backend, phase)
            )
            checkpoint["build"]["wall_time_s"] = round(
                time.monotonic() - wall_start, 3
            )
            checkpoint["build"]["status"] = "failed"
            _record_failure(checkpoint, paths["checkpoint"], "build", exc)
            return False, str(exc), checkpoint

    checkpoint["status"] = "retrieving"
    checkpoint["retrieval"] = {"status": "running", "started_at": utc_now()}
    checkpoint.pop("answer", None)
    atomic_json(paths["checkpoint"], checkpoint)
    phase = f"lme_{index}_retrieve_answer"
    tracker_reset(backend, phase)
    wall_start = time.monotonic()
    usage = None
    try:
        turn_index = backend.build_turn_index(conv)
        expected_turns = checkpoint["input"]["turns"]
        if len(turn_index) != expected_turns:
            raise EmptyStageError(
                f"turn index has {len(turn_index)} entries; expected {expected_turns}"
            )
        query = model_question(item)
        memories, steps, answer, tool_trace = collect_and_answer_longmemeval(
            backend, item, paths["memory_dir"], turn_index
        )
        usage = tracker_snapshot(backend, phase)
        hypothesis = str(answer or "").strip()
        if not hypothesis:
            raise EmptyStageError(
                "single-model retrieval/answer returned an empty hypothesis"
            )
        if int(steps) <= 0:
            raise EmptyStageError(
                "single-model retrieval/answer completed without a model step"
            )
        checkpoint["retrieval"] = {
            "status": "complete",
            "started_at": checkpoint["retrieval"]["started_at"],
            "finished_at": utc_now(),
            "wall_time_s": round(time.monotonic() - wall_start, 3),
            "steps": int(steps),
            "memories_count": len(memories),
            "memories": memories,
            "tool_trace": tool_trace,
            "usage": usage,
            "turn_index_entries": len(turn_index),
            "model_question": query,
        }
        checkpoint["answer"] = {
            "status": "complete",
            "hypothesis": hypothesis,
            "model": run_meta["models"]["answerer"],
            "single_model_context": True,
            "finished_at": utc_now(),
        }
        checkpoint["status"] = "complete"
        checkpoint["completed_at"] = utc_now()
        checkpoint["updated_at"] = utc_now()
        checkpoint.pop("error", None)
        atomic_json(paths["checkpoint"], checkpoint)
        return True, "complete", checkpoint
    except Exception as exc:  # noqa: BLE001
        checkpoint["retrieval"]["usage"] = (
            usage if usage is not None else tracker_snapshot(backend, phase)
        )
        checkpoint["retrieval"]["wall_time_s"] = round(
            time.monotonic() - wall_start, 3
        )
        checkpoint["retrieval"]["status"] = "failed"
        _record_failure(checkpoint, paths["checkpoint"], "retrieve_answer", exc)
        return False, str(exc), checkpoint


def collect_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for path in sorted((output_dir / "items").glob("*/checkpoint.json")):
        try:
            checkpoint = read_json(path)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(checkpoint, dict):
            checkpoints.append(checkpoint)
    def _index(item: dict[str, Any]) -> int:
        try:
            return int(item.get("dataset_index", 10**9))
        except (TypeError, ValueError):
            return 10**9
    checkpoints.sort(key=_index)
    return checkpoints


def has_durable_complete_payload(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict) or checkpoint.get("status") != "complete":
        return False
    build = checkpoint.get("build")
    retrieval = checkpoint.get("retrieval")
    answer = checkpoint.get("answer")
    paths = checkpoint.get("paths")
    if not all(isinstance(value, dict)
               for value in (build, retrieval, answer, paths)):
        return False
    if build.get("status") != "complete" or retrieval.get("status") != "complete":
        return False
    if answer.get("status") != "complete" or not str(answer.get("hypothesis", "")).strip():
        return False
    memory_dir = paths.get("memory_dir")
    return bool(memory_dir and memory_is_valid(Path(memory_dir)))


def refresh_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Regenerate aggregate outputs solely from durable item checkpoints."""
    checkpoints = collect_checkpoints(output_dir)
    completed = [item for item in checkpoints if has_durable_complete_payload(item)]
    records = [{
        "dataset_index": item["dataset_index"],
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["gold"],
        "hypothesis": item["answer"]["hypothesis"],
        "build": item["build"],
        "retrieval": item["retrieval"],
        "models": item["models"],
        "config": item["config"],
    } for item in completed]
    atomic_json(output_dir / "results.json", records)
    jsonl = "".join(json.dumps({
        "question_id": item["question_id"],
        "hypothesis": item["answer"]["hypothesis"],
    }, ensure_ascii=False) + "\n" for item in completed)
    atomic_text(output_dir / "hypotheses.jsonl", jsonl)

    statuses = Counter(str(item.get("status", "unknown")) for item in checkpoints)
    manifest["checkpoint_counts"] = dict(sorted(statuses.items()))
    manifest["completed"] = len(completed)
    manifest["updated_at"] = utc_now()
    atomic_json(output_dir / "run_manifest.json", manifest)


def build_run_meta(args: argparse.Namespace, config: dict[str, str]) -> dict[str, Any]:
    openrouter = getattr(args, "provider", "openai-flex") == "openrouter"
    return {
        "method": {
            "name": "NativeMem",
            "version": "v8.8+calendar",
            "calendar": True,
            "single_model_retrieve_answer": True,
        },
        "models": {
            "builder": args.model,
            "retriever": args.model,
            "answerer": args.model,
            "provider": (
                "openrouter_via_cost_gateway" if openrouter
                else "openai_api_flex_via_exclusive_child_proxy"
            ),
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
        },
        "config": config,
        "code": {
            "git_commit": git_head(),
            "runner_sha256": sha256_file(Path(__file__)),
            "nativemem_sha256": sha256_file(ROOT / "src" / "nativemem.py"),
            "v8_memory_sha256": sha256_file(ROOT / "src" / "v8_memory.py"),
            "adapter_sha256": sha256_file(
                ROOT / "src" / "adapters" / "run_nativemem.py"
            ),
            "proxy_sha256": sha256_file(ROOT / "src" / "chatgpt_proxy.py"),
            "flex_gateway_sha256": sha256_file(
                ROOT / "src" / "openai_gpt55_flex_gateway.py"
            ),
            "flex_evidence_sha256": sha256_file(
                ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py"
            ),
            "child_proxy_sha256": sha256_file(
                ROOT / "scripts" / "gpt55_run_proxy.py"
            ),
        },
        "request_audit": {
            "mode": (
                "exclusive_openrouter_gateway" if openrouter
                else "bounded_gateway_segments_with_exclusive_child_proxy"
            ),
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
            "explicit_model_request_authorization": True,
        },
    }


def create_or_resume_manifest(
    output_dir: Path,
    data_path: Path,
    dataset_sha256: str,
    data_count: int,
    run_meta: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / "run_manifest.json"
    if path.exists():
        old = read_json(path)
        if not isinstance(old, dict):
            raise ExistingStateError("existing run_manifest.json is invalid")
        expected = {
            "dataset_sha256": dataset_sha256,
            "models": run_meta["models"],
            "config": run_meta["config"],
            "code": run_meta["code"],
            "request_audit": run_meta.get("request_audit", {}),
        }
        mismatches = [key for key, value in expected.items() if old.get(key) != value]
        if mismatches:
            raise ExistingStateError(
                f"existing output uses different {', '.join(mismatches)}; "
                "choose another --output-dir"
            )
        if not resume and any((output_dir / "items").glob("*/checkpoint.json")):
            # Completed items will still be protected by run_item, but the
            # explicit error prevents accidental reuse of an incomplete run.
            incomplete = [
                item for item in collect_checkpoints(output_dir)
                if item.get("status") != "complete"
            ]
            if incomplete:
                raise ExistingStateError(
                    "existing run contains incomplete checkpoints; use --resume"
                )
        return old
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "LongMemEval-S",
        "output_dir": str(output_dir),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "dataset_items": data_count,
        "method": run_meta["method"],
        "models": run_meta["models"],
        "config": run_meta["config"],
        "code": run_meta["code"],
        "request_audit": run_meta.get("request_audit", {}),
        "created_at": utc_now(),
        "completed": 0,
        "checkpoint_counts": {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NativeMem v8.8+calendar LongMemEval-S runner"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--provider", choices=("openai-flex", "openrouter"),
        default="openai-flex",
    )
    parser.add_argument("--gateway-root", type=Path, required=True)
    parser.add_argument("--gateway-base-url")
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--request-concurrency", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    openrouter = args.provider == "openrouter"
    args.model = "openai/gpt-4o-mini" if openrouter else "gpt-5.5"
    args.api_key = "local-openrouter-gateway" if openrouter else "x"
    args.trust_proxy = False
    if not args.allow_model_requests:
        parser.error("formal LongMemEval execution requires --allow-model-requests")
    if args.request_concurrency < 1:
        parser.error("--request-concurrency must be positive")
    if openrouter:
        if not args.gateway_base_url:
            parser.error("OpenRouter mode requires --gateway-base-url")
        gateway_root = args.gateway_root.expanduser().resolve()
        try:
            marker = read_json(gateway_root / "openrouter_gpt4o_mini_root.json")
            ready = read_json(gateway_root / "gateway_ready.json")
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read marked OpenRouter gateway: {exc}")
        if (
            marker.get("schema") != "openrouter-gpt4o-mini-result-root/v1"
            or marker.get("returned_alias") != "openai/gpt-4o-mini"
            or ready.get("schema") != "openrouter-gpt4o-mini-ready/v1"
            or ready.get("requested_model") != "openai/gpt-4o-mini"
            or Path(str(ready.get("result_root", ""))).resolve() != gateway_root
            or ready.get("base_url") != args.gateway_base_url
        ):
            parser.error("OpenRouter gateway binding does not match its markers")
        args.base_url = args.gateway_base_url
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"dataset does not exist: {data_path}")

    # The official runner is intentionally strict: a wrong local file must not
    # be reported as a LongMemEval-S full run merely because its schema matches.
    try:
        data = load_dataset(data_path, expected_count=EXPECTED_LONGMEMEVAL_SIZE)
        validate_longmemeval_s(data)
        indices = select_indices(len(data), args.start, args.limit)
    except DataValidationError as exc:
        parser.error(str(exc))

    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error(
            f"manifest exists at {manifest_path}; pass --resume or use a new "
            "--output-dir"
        )
    if not openrouter and args.resume and manifest_path.exists():
        try:
            previous = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read existing manifest: {exc}")
        provider = previous.get("provider_evidence")
        if (
            not isinstance(provider, dict)
            or provider.get("schema") != "openai-gpt55-flex-invocations/v1"
            or provider.get("active_run_id") is not None
        ):
            parser.error(
                "existing output is not a closed Flex-evidence run; use a new "
                "--output-dir"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    # A second launcher could otherwise modify the same item checkpoint while
    # the first process is between its build and answer stages.
    lock_file = (output_dir / ".launcher.lock").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another launcher is already using {output_dir}")
    invocation = None
    if not openrouter:
        invocation = flex_evidence.begin_child_invocation(
            args.gateway_root,
            output_dir / "provider_evidence" /
            f"invocation-{utc_now().replace(':', '')}",
        )
        args.base_url = invocation.base_url
        os.environ["CHATGPT_PROXY_LOG"] = str(invocation.log_path)
    else:
        os.environ.pop("CHATGPT_PROXY_LOG", None)
    try:
        config = configure_environment(args)
        backend = load_backend()
        verify_backend_contract(backend)
        run_meta = build_run_meta(args, config)
        manifest = create_or_resume_manifest(
            output_dir, data_path, sha256_file(data_path), len(data),
            run_meta, args.resume,
        )
    except ExistingStateError as exc:
        if invocation is not None:
            invocation.abort()
        parser.error(str(exc))
    except Exception:
        if invocation is not None:
            invocation.abort()
        raise

    if openrouter:
        provider_evidence = manifest.setdefault("provider_evidence", {
            "schema": "openrouter-gpt4o-mini-consumer/v1",
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
            "root_marker_sha256": sha256_file(
                args.gateway_root.expanduser().resolve() /
                "openrouter_gpt4o_mini_root.json"
            ),
        })
        if provider_evidence.get("gateway_root") != str(
            args.gateway_root.expanduser().resolve()
        ):
            parser.error("existing run uses a different OpenRouter gateway root")
    else:
        provider_evidence = manifest.setdefault("provider_evidence", {
            "schema": "openai-gpt55-flex-invocations/v1",
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
            "active_run_id": None,
            "invocations": [],
        })
        if provider_evidence.get("gateway_root") != str(
            args.gateway_root.expanduser().resolve()
        ):
            invocation.abort()
            parser.error("existing run uses a different Flex gateway root")
        provider_evidence["active_run_id"] = invocation.run_id

    manifest.setdefault("invocations", []).append({
        "started_at": utc_now(),
        "start": args.start,
        "limit": args.limit,
        "indices": [indices[0], indices[-1]],
        "resume": args.resume,
    })
    refresh_outputs(output_dir, manifest)

    failures: list[int] = []
    try:
        for position, index in enumerate(indices, start=1):
            item = data[index]
            print(
                f"[{position}/{len(indices)}] item {index} "
                f"{item['question_id']} ({item['question_type']})",
                flush=True,
            )
            try:
                ok, detail, _checkpoint = run_item(
                    index, item, output_dir, backend, run_meta, args.resume
                )
            except ExistingStateError as exc:
                ok, detail = False, str(exc)
            print(f"  {'complete' if ok else 'failed'}: {detail}", flush=True)
            if not ok:
                failures.append(index)
            refresh_outputs(output_dir, manifest)
            if failures and args.fail_fast:
                break
        evidence_record = invocation.finish() if invocation is not None else None
    except Exception:
        if invocation is not None:
            invocation.abort()
        manifest["status"] = "failed"
        manifest["last_invocation_finished_at"] = utc_now()
        refresh_outputs(output_dir, manifest)
        raise

    if invocation is not None:
        provider_evidence["active_run_id"] = None
        provider_evidence["invocations"].append(evidence_record)

    manifest["last_invocation_finished_at"] = utc_now()
    manifest["last_invocation_failures"] = failures
    completed_count = sum(
        has_durable_complete_payload(checkpoint)
        for checkpoint in collect_checkpoints(output_dir)
    )
    manifest["status"] = (
        "failed" if failures else
        "complete" if completed_count == len(data) else
        "partial"
    )
    refresh_outputs(output_dir, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
