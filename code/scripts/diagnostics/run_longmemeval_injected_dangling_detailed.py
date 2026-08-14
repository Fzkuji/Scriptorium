"""Run one diagnostic item with one real staged dangling-link fault."""

from __future__ import annotations

import os
import sys

from src.management import agent as management_agent
from src.markdown import parse_topic_tree


def main() -> int:
    original_commit = management_agent._commit_turn
    injected = False

    def inject_once(workspace, baseline, audit):
        nonlocal injected
        result = original_commit(workspace, baseline, audit)
        if result is None and not injected:
            # Inject only after the writer's real edit has committed and all
            # source references have been normalized. This makes dangling-link
            # validation, rather than an earlier formatting rule, the fault
            # under test.
            injected = True
            workspace._refresh_stage()
            injection_baseline = workspace.baseline()
            units = parse_topic_tree(workspace.stage_dir / "topics")
            source = units[0]
            path = workspace.stage_dir / "topics" / source.topic_path
            text = path.read_text(encoding="utf-8")
            replacement = (
                source.content
                + " [diagnostic missing target]"
                + "(#^diagnostic-missing-target)"
            )
            if source.content not in text:
                raise RuntimeError(
                    "diagnostic source paragraph could not be located"
                )
            path.write_text(
                text.replace(source.content, replacement, 1),
                encoding="utf-8",
            )
            audit.append({
                "tool": "diagnostic_fault_injection",
                "status": "ok",
                "source_file": f"topics/{source.topic_path}",
                "source_block": source.memory_id,
                "target": "diagnostic-missing-target",
            })
            return original_commit(workspace, injection_baseline, audit)
        return result

    management_agent._commit_turn = inject_once
    sys.argv = [
        "run_longmemeval.py",
        "--config", os.environ["CONFIG_PATH"],
        "--base-url", "https://opencode.ai/zen/go",
        "--provider-name", "opencode-go",
        "--api-key", os.environ["OPENCODE_KEY"],
        "--resume",
        "--allow-resume-drift",
    ]
    from scripts.runners.run_longmemeval import main as runner_main

    return int(runner_main())


if __name__ == "__main__":
    raise SystemExit(main())
