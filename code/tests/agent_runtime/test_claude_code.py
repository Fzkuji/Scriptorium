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
            usage={"input_tokens": 11, "output_tokens": 3},
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
    assert result.total_cost_usd == 0.002
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.setting_sources == []
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
