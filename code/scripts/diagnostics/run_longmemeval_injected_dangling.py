"""Run one diagnostic item with one injected dangling-link commit failure."""

from __future__ import annotations

import os
import sys

from src.management import agent as management_agent


def main() -> int:
    original_commit = management_agent._commit_turn
    injected = False

    def inject_once(workspace, baseline, audit):
        nonlocal injected
        if not injected:
            injected = True
            workspace._refresh_stage()
            failure = management_agent.CommitFailure(
                ValueError("dangling block link: diagnostic-missing-target")
            )
            audit.append({
                "tool": "diagnostic_fault_injection",
                "status": "error",
                "output": failure.message,
            })
            return failure
        return original_commit(workspace, baseline, audit)

    management_agent._commit_turn = inject_once
    sys.argv = [
        "run_longmemeval.py",
        "--config", os.environ["CONFIG_PATH"],
        "--base-url", "https://opencode.ai/zen/go",
        "--provider-name", "opencode-go",
        "--api-key", os.environ["OPENCODE_KEY"],
        "--resume",
    ]
    from scripts.runners.run_longmemeval import main as runner_main

    return int(runner_main())


if __name__ == "__main__":
    raise SystemExit(main())
