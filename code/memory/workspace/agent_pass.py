"""Run one agent against the staged workspace, committing as it goes.

Every phase that changes memory does it this way: the model edits a staged
copy through tools it was handed, each turn is committed against the topic
contract, and a turn that violates it is rolled back and put to the model
again with the reason it was refused. Which tools, and what the model is
asked to do with them, is the phase's business and not this module's.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # the SDK is only needed to actually run a writer
    from ..agent_runtime import ClaudeCodeAgent
from ..config import MemoryConfig
from ..prompts import FEW_SHOT_INSTRUCTIONS, SYSTEM_PROMPT
from .layout import runtime_dir
from .staging import MemoryWorkspace


def _baseline(workspace: MemoryWorkspace) -> tuple[Any, ...] | None:
    """Snapshot the stage, or None if it cannot be parsed.

    A turn whose edits were rejected leaves the stage rolled back, but a
    malformed file that reached disk another way would make this raise. That
    is a reason to skip committing, never to abort the whole build.
    """
    try:
        return workspace.baseline()
    except Exception:
        workspace._refresh_stage()
        try:
            return workspace.baseline()
        except Exception:
            return None


def _commit_turn(
    workspace: MemoryWorkspace,
    baseline: tuple[Any, ...] | None,
    audit: list[dict[str, Any]],
) -> str | None:
    """Install what the turn staged. Returns the error text if it failed."""
    if baseline is None:
        workspace._refresh_stage()
        return None
    if not workspace.stage_is_dirty():
        return None
    try:
        workspace.commit_edits(*baseline)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        audit.append({
            "tool": "commit", "status": "error", "output": message,
        })
        return message
    audit.append({
        "tool": "commit",
        "status": "ok",
        "count": workspace.last_created_blocks,
        "topic_paths": workspace.last_changed_topics,
    })
    return None


def _record_trajectory(
    memory_dir: str | Path,
    stage: str,
    system_prompt: str,
    prompt: str,
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    """Append one agent run to the runtime directory's agent-history.jsonl.

    What the model was sent, every tool call it made, and what it replied.
    Usage counters say a batch took sixty turns; only this says what those
    turns were doing, and a failed run is the one most worth reading.
    """
    record = {
        "stage": stage,
        "system_prompt": system_prompt,
        "prompt": prompt,
    }
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
        record["turns"] = getattr(error, "turns", [])
    else:
        record.update({
            "turns": result.turns,
            "reply": result.reply or result.text,
            "rounds": result.num_turns,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "stop_reason": result.stop_reason,
        })
    # The runtime directory, not the workspace root: this is a log of how
    # the memory was produced, not memory, and the transaction compares the
    # workspace against the stage to decide what an edit changed.
    path = runtime_dir(memory_dir) / "agent-history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pass(
    memory_dir: str | Path,
    *,
    agent: ClaudeCodeAgent,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    config: MemoryConfig | None = None,
    history_dir: str | Path | None = None,
    stage: str | None = None,
    tools: Any,
    guidance: Any = None,
    protocol: Any = None,
) -> list[dict[str, Any]]:
    """`tools` builds the tool list for this pass, given (workspace, audit).

    `guidance` turns a rejection message into the correction to send with it,
    and `protocol` appends whatever the phase wants said to a model that
    brings no file tools of its own. Both are the phase's business: what a
    refused edit means depends on which verbs the phase offered.
    """
    config = config or MemoryConfig()
    # Verification runs against a throwaway copy of the memory. Its history
    # belongs with the real workspace, or it is deleted with the copy.
    history_dir = memory_dir if history_dir is None else history_dir
    workspace = MemoryWorkspace(memory_dir, config=config)
    try:
        if source_sessions:
            workspace.archive_sessions(source_sessions)
            workspace._refresh_stage()
        task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
        audit: list[dict[str, Any]] = []
        system_prompt = (
            f"{SYSTEM_PROMPT}\n{FEW_SHOT_INSTRUCTIONS}"
            if config.few_shot_instructions
            else SYSTEM_PROMPT
        )
        # Edits made through the built-in file tools never pass through the
        # shell tool, so the transaction runs once the turn is over.
        baseline = _baseline(workspace)
        stage = stage or ("write" if source_sessions else "organize")
        if protocol is not None and not getattr(agent, "has_file_tools", True):
            task = f"{task}\n{protocol(memory_dir)}"
        try:
            result = agent.run(
                prompt=task,
                system_prompt=system_prompt,
                cwd=workspace.stage_dir,
                tools=tools(workspace, audit),
                max_turns=config.max_turns,
                max_budget_usd=config.max_budget_usd,
            )
        except BaseException as exc:
            _record_trajectory(
                history_dir, stage, system_prompt, task, error=exc
            )
            raise
        _record_trajectory(history_dir, stage, system_prompt, task, result)
        if usage_logger is not None:
            usage_logger(result)
        error = _commit_turn(workspace, baseline, audit)
        if error is not None:
            # A rejected turn was rolled back, so the pre-repair stage is the
            # committed state. Snapshot it now: reading it after the repair
            # edits would parse the model's malformed text and raise.
            repair_baseline = _baseline(workspace)
            repair_prompt = (
                "Your edits were rejected and discarded:\n\n"
                f"{error}\n\n"
                + (f"{guidance(error)}\n\n" if guidance and guidance(error) else "")
                + "The workspace is back to its state before this turn. "
                "Redo the work, fixing what the message reports.\n\n"
                f"{task}"
            )
            try:
                repair = agent.run(
                    prompt=repair_prompt,
                    system_prompt=system_prompt,
                    cwd=workspace.stage_dir,
                    tools=tools(workspace, audit),
                    max_turns=config.max_turns,
                    max_budget_usd=config.max_budget_usd,
                )
            except BaseException as exc:
                _record_trajectory(
                    history_dir, f"{stage}-repair", system_prompt,
                    repair_prompt, error=exc,
                )
                raise
            _record_trajectory(
                history_dir, f"{stage}-repair", system_prompt,
                repair_prompt, repair,
            )
            if usage_logger is not None:
                usage_logger(repair)
            _commit_turn(workspace, repair_baseline, audit)
        if final_output is not None:
            final_output.append(result.text)
        audit.append({
            "tool": "agent",
            "status": "ok",
            "reason": result.stop_reason or "complete",
            "rounds": result.num_turns,
            "turns": result.turns,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "anthropic_equivalent_cost_usd": (
                result.anthropic_equivalent_cost_usd
            ),
        })
        return audit
    finally:
        shutil.rmtree(workspace.stage_dir, ignore_errors=True)
