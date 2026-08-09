#!/usr/bin/env python3
"""Check one Claude Agent SDK MCP tool round-trip without exposing credentials."""

from __future__ import annotations

import argparse
from pathlib import Path

from claude_agent_sdk import tool

from src.agent_runtime import ClaudeCodeAgent, ClaudeCodeConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-file", type=Path, required=True)
    args = parser.parse_args()

    @tool("ping", "Return the supplied value.", {"value": str})
    async def ping(arguments):
        return {"content": [{"type": "text", "text": arguments["value"]}]}

    key = args.api_key_file.resolve().read_text(encoding="utf-8").strip()
    agent = ClaudeCodeAgent(ClaudeCodeConfig(
        base_url=args.base_url,
        api_key=key,
        model=args.model,
    ))
    result = agent.run(
        prompt="Call ping exactly once with value OK, then reply exactly OK.",
        system_prompt="You must use the provided ping tool exactly once.",
        cwd=Path.cwd(),
        tools=[ping],
        max_turns=5,
    )
    print(f"success turns={result.num_turns} reply={result.text.strip()}")


if __name__ == "__main__":
    main()
