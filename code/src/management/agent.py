"""Memory Writer and Manager execution through Claude Code."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent_runtime import ClaudeCodeAgent
from .config import MemoryConfig
from .errors import CoreCapacityError
from .prompts import (
    CORE_CAPACITY_REPAIR_TASK,
    SYSTEM_PROMPT,
    WRITER_SHELL_EXAMPLES,
)
from .tools import CoreRepairState, management_tools
from .workspace import MemoryWorkspace


@dataclass(frozen=True)
class CommitFailure:
    """A rejected staged commit with its structured underlying exception."""

    exception: Exception

    @property
    def message(self) -> str:
        return f"{type(self.exception).__name__}: {self.exception}"


def _failure_message(failure: CommitFailure | Exception | str) -> str:
    if isinstance(failure, CommitFailure):
        return failure.message
    if isinstance(failure, Exception):
        return f"{type(failure).__name__}: {failure}"
    return str(failure)


def _core_capacity_error(
    failure: CommitFailure | Exception | str,
) -> CoreCapacityError | None:
    exception = (
        failure.exception if isinstance(failure, CommitFailure) else failure
    )
    return exception if isinstance(exception, CoreCapacityError) else None


def render_conversation(
    turns: list[tuple[str, str]], refs: list[str]
) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


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


def _committed_baseline(
    workspace: MemoryWorkspace,
) -> tuple[Any, ...] | None:
    """Snapshot committed memory without disturbing staged built-in edits."""
    committed = MemoryWorkspace(workspace.memory_dir, config=workspace.config)
    try:
        return _baseline(committed)
    finally:
        shutil.rmtree(committed.stage_dir, ignore_errors=True)


def _commit_turn(
    workspace: MemoryWorkspace,
    baseline: tuple[Any, ...] | None,
    audit: list[dict[str, Any]],
) -> CommitFailure | None:
    """Install what the turn staged. Return a structured rejection on failure."""
    if baseline is None:
        workspace._refresh_stage()
        return None
    if not workspace.stage_is_dirty():
        return None
    try:
        workspace.commit_edits(*baseline)
    except Exception as exc:
        failure = CommitFailure(exc)
        audit.append({
            "tool": "commit", "status": "error", "output": failure.message,
        })
        return failure
    audit.append({
        "tool": "commit",
        "status": "ok",
        "count": workspace.last_created_blocks,
        "topic_paths": workspace.last_changed_topics,
    })
    return None


def _run_agent(
    memory_dir: str | Path,
    *,
    agent: ClaudeCodeAgent,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    live_audit_path: str | Path | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    workspace = MemoryWorkspace(memory_dir, config=config)
    progress_path = (
        Path(str(live_audit_path) + ".progress.json")
        if live_audit_path is not None else None
    )

    def record_progress(event: dict[str, Any]) -> None:
        if progress_path is None:
            return
        payload = {
            **event,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, progress_path)
    try:
        if source_sessions:
            workspace.archive_sessions(source_sessions)
            workspace._refresh_stage()
        task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
        audit: list[dict[str, Any]] = []
        system_prompt = (
            f"{SYSTEM_PROMPT}\n{WRITER_SHELL_EXAMPLES}"
            if config.writer_shell_examples
            else SYSTEM_PROMPT
        )
        try:
            result = agent.run(
                prompt=task,
                system_prompt=system_prompt,
                cwd=workspace.stage_dir,
                tools=management_tools(
                    workspace, audit, live_audit_path=live_audit_path
                ),
                max_turns=config.max_turns,
                max_budget_usd=config.max_budget_usd,
                progress_fn=record_progress,
            )
        except Exception as exc:
            # The staged workspace is intentionally discarded, but callers
            # still need the tool trace to diagnose loops and platform errors.
            setattr(exc, "scriptorium_audit", list(audit))
            raise
        if usage_logger is not None:
            usage_logger(result)
        trajectory_results = [result]
        core_repair_records: list[dict[str, Any]] = []
        # Shell calls commit immediately, while built-in file tools leave edits
        # staged until the trajectory ends. Snapshot the current committed tree
        # now, so shell work from this same turn is not mistaken for a new edit.
        baseline = _committed_baseline(workspace)
        error = _commit_turn(workspace, baseline, audit)
        if error is not None:
            capacity_error = _core_capacity_error(error)
            if capacity_error is not None:
                repair_error: CommitFailure | Exception | str | None = error
                for repair_number in range(
                    1, config.core_repair_max_trajectories + 1
                ):
                    current_capacity = _core_capacity_error(repair_error)
                    if current_capacity is None:
                        break
                    # Core-only capacity failures preserve the otherwise valid
                    # staged turn. Repair the excess in place and validate the
                    # result against the original committed baseline.
                    repair_baseline = baseline
                    state = CoreRepairState(
                        limit=current_capacity.limit,
                        target=config.effective_core_repair_target_tokens,
                        max_checks=config.core_repair_max_checks,
                        stagnation_limit=config.core_repair_stagnation_limit,
                    )
                    repair_prompt = CORE_CAPACITY_REPAIR_TASK.format(
                        limit=current_capacity.limit,
                        token_count=current_capacity.token_count,
                        target=state.target,
                        max_checks=state.max_checks,
                    )
                    try:
                        repair = agent.run(
                            prompt=repair_prompt,
                            system_prompt=system_prompt,
                            cwd=workspace.stage_dir,
                            tools=management_tools(
                                workspace,
                                audit,
                                live_audit_path=live_audit_path,
                                core_repair_state=state,
                            ),
                            max_turns=config.max_turns,
                            max_budget_usd=config.max_budget_usd,
                            progress_fn=record_progress,
                        )
                    except Exception as exc:
                        setattr(exc, "scriptorium_audit", list(audit))
                        raise
                    if usage_logger is not None:
                        usage_logger(repair)
                    trajectory_results.append(repair)
                    repair_error = _commit_turn(
                        workspace, repair_baseline, audit
                    )
                    core_repair_records.append({
                        "trajectory": repair_number,
                        "mode": "incremental_staged",
                        "trigger_token_count": current_capacity.token_count,
                        "hard_limit": state.limit,
                        "target": state.target,
                        "checks": list(state.checks),
                        "stagnated": state.stagnated,
                        "status": (
                            "ok" if repair_error is None else "rejected"
                        ),
                        "error": (
                            _failure_message(repair_error)
                            if repair_error is not None else None
                        ),
                    })
                    audit.append({
                        "tool": "core_capacity_repair",
                        "status": core_repair_records[-1]["status"],
                        **core_repair_records[-1],
                    })
                    if repair_error is None:
                        break
                    if _core_capacity_error(repair_error) is None:
                        break
                if repair_error is not None:
                    exc = RuntimeError(
                        "memory writer core-capacity repair was rejected: "
                        + _failure_message(repair_error)
                    )
                    setattr(exc, "scriptorium_audit", list(audit))
                    raise exc
            else:
                # Generic validation failures retain the legacy single repair.
                repair_baseline = _baseline(workspace)
                try:
                    repair = agent.run(
                        prompt=(
                            "Your edits were rejected and discarded:\n\n"
                            f"{_failure_message(error)}\n\n"
                            "The workspace is back to its state before this "
                            "turn. Redo the work, fixing what the message "
                            "reports.\n\n"
                            f"{task}"
                        ),
                        system_prompt=system_prompt,
                        cwd=workspace.stage_dir,
                        tools=management_tools(
                            workspace, audit,
                            live_audit_path=live_audit_path,
                        ),
                        max_turns=config.max_turns,
                        max_budget_usd=config.max_budget_usd,
                        progress_fn=record_progress,
                    )
                except Exception as exc:
                    setattr(exc, "scriptorium_audit", list(audit))
                    raise
                if usage_logger is not None:
                    usage_logger(repair)
                trajectory_results.append(repair)
                repair_error = _commit_turn(
                    workspace, repair_baseline, audit
                )
                if repair_error is not None:
                    exc = RuntimeError(
                        "memory writer repair was rejected: "
                        + _failure_message(repair_error)
                    )
                    setattr(exc, "scriptorium_audit", list(audit))
                    raise exc
        if final_output is not None:
            final_output.append(trajectory_results[-1].text)
        audit.append({
            "tool": "agent",
            "status": "ok",
            "reason": trajectory_results[-1].stop_reason or "complete",
            "rounds": sum(item.num_turns for item in trajectory_results),
            "input_tokens": sum(item.input_tokens for item in trajectory_results),
            "output_tokens": sum(item.output_tokens for item in trajectory_results),
            "anthropic_equivalent_cost_usd": sum(
                item.anthropic_equivalent_cost_usd or 0.0
                for item in trajectory_results
            ),
            **({"core_repair": core_repair_records}
               if core_repair_records else {}),
        })
        return audit
    finally:
        shutil.rmtree(workspace.stage_dir, ignore_errors=True)
