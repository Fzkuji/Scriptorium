from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

from src.agent_runtime import (
    AgentExecutionError,
    ClaudeCodeAgent,
    ClaudeCodeConfig,
)


def test_installed_sdk_can_create_an_in_process_mcp_server() -> None:
    @tool("ping", "Return the supplied value.", {"value": str})
    async def ping(arguments):
        return {
            "content": [{"type": "text", "text": arguments["value"]}],
        }

    server = create_sdk_mcp_server("test", tools=[ping])

    assert server["type"] == "sdk"


def test_agent_uses_isolated_bare_nonpersistent_claude_code(
    tmp_path: Path,
) -> None:
    captured = {}

    async def fake_query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        yield AssistantMessage(
            content=[TextBlock("done")],
            model="test-model",
            usage={"input_tokens": 11, "output_tokens": 3},
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=120,
            duration_api_ms=100,
            is_error=False,
            num_turns=2,
            session_id="test-session",
            total_cost_usd=0.002,
            usage={
                "input_tokens": 11,
                "output_tokens": 3,
                "cache_creation_input_tokens": 40,
                "cache_read_input_tokens": 900,
            },
            result="done",
        )

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig(
            base_url="https://gateway.example",
            api_key="secret-token",
            model="test-model",
        ),
        query_fn=fake_query,
    )

    result = agent.run(
        prompt="answer",
        system_prompt="system",
        cwd=tmp_path,
        max_turns=20,
    )

    options = captured["options"]
    assert captured["prompt"] == "answer"
    assert result.text == "done"
    assert result.num_turns == 2
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    # Cached prefixes are billable input and must survive the SDK boundary.
    assert result.cache_creation_input_tokens == 40
    assert result.cache_read_input_tokens == 900
    assert result.anthropic_equivalent_cost_usd == 0.002
    # The built-in file tools must be sent as schemas, not merely permitted:
    # `allowed_tools` filters, `tools` is what the model actually sees.
    assert options.tools == ["Read", "Edit", "Write", "Grep", "Glob"]
    assert options.allowed_tools == ["Read", "Edit", "Write", "Grep", "Glob"]
    assert options.setting_sources == []
    assert options.thinking == {"type": "disabled"}
    assert options.extra_args == {
        "bare": None,
        "no-session-persistence": None,
        "strict-mcp-config": None,
    }
    assert options.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert options.env["ANTHROPIC_API_KEY"] == "secret-token"
    config_root = Path(options.env["CLAUDE_CONFIG_DIR"])
    assert config_root != Path.home() / ".claude"
    assert not config_root.exists()


def test_agent_raises_framework_error_without_leaking_api_key(
    tmp_path: Path,
) -> None:
    async def fake_query(*, prompt, options):
        del prompt, options
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=10,
            duration_api_ms=5,
            is_error=True,
            num_turns=1,
            session_id="test-session",
            result=None,
            errors=["provider rejected request"],
        )

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig(
            base_url="https://gateway.example",
            api_key="secret-token",
            model="test-model",
        ),
        query_fn=fake_query,
    )

    with pytest.raises(AgentExecutionError, match="provider rejected request") as error:
        agent.run(prompt="answer", system_prompt="system", cwd=tmp_path)
    assert "secret-token" not in str(error.value)


def test_agent_preserves_api_status_when_cli_exits_after_error_result(
    tmp_path: Path,
) -> None:
    async def fake_query(*, prompt, options):
        del prompt, options
        message = ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=True,
            num_turns=1,
            session_id="test-session",
            result=None,
        )
        message.api_error_status = 529
        yield message
        raise Exception("Claude Code returned an error result: success")

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig(
            base_url="https://gateway.example",
            api_key="secret-token",
            model="test-model",
        ),
        query_fn=fake_query,
    )

    with pytest.raises(AgentExecutionError, match="API status 529"):
        agent.run(prompt="answer", system_prompt="system", cwd=tmp_path)


def test_result_records_every_tool_call_and_its_outcome(tmp_path: Path) -> None:
    """Built-in file tools never reach the MCP layer.

    Without this record a trajectory that burned its turn budget retrying one
    Edit leaves nothing behind to say what it was retrying.
    """
    from claude_agent_sdk import ToolResultBlock, ToolUseBlock

    async def fake_query(*, prompt, options):
        yield AssistantMessage(
            content=[ToolUseBlock(
                id="call-1",
                name="Edit",
                input={"file_path": "topics/a.md", "old_string": ""},
            )],
            model="test-model",
            usage={},
        )
        yield AssistantMessage(
            content=[ToolResultBlock(
                tool_use_id="call-1",
                content="String to replace not found",
                is_error=True,
            )],
            model="test-model",
            usage={},
        )
        yield ResultMessage(
            subtype="success", duration_ms=10, duration_api_ms=5,
            is_error=False, num_turns=2, session_id="s",
            total_cost_usd=0.0, usage={}, result="",
        )

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig(base_url="https://x", api_key="k", model="m"),
        query_fn=fake_query,
    )

    result = agent.run(prompt="p", system_prompt="s", cwd=tmp_path)

    assert result.turns[0]["tool"] == "Edit"
    assert result.turns[0]["arguments"]["file_path"] == "topics/a.md"
    assert result.turns[1]["is_error"] is True
    assert "String to replace not found" in result.turns[1]["result"]


def test_a_failed_trajectory_still_carries_its_turns(tmp_path: Path) -> None:
    from claude_agent_sdk import ToolUseBlock

    async def fake_query(*, prompt, options):
        yield AssistantMessage(
            content=[ToolUseBlock(
                id="call-1", name="Edit", input={"file_path": "topics/a.md"},
            )],
            model="test-model",
            usage={},
        )
        yield ResultMessage(
            subtype="error_max_turns", duration_ms=10, duration_api_ms=5,
            is_error=True, num_turns=60, session_id="s",
            total_cost_usd=0.0, usage={},
            result="Reached maximum number of turns (60)",
        )

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig(base_url="https://x", api_key="k", model="m"),
        query_fn=fake_query,
    )

    with pytest.raises(AgentExecutionError) as failure:
        agent.run(prompt="p", system_prompt="s", cwd=tmp_path)

    assert failure.value.turns[0]["tool"] == "Edit"
