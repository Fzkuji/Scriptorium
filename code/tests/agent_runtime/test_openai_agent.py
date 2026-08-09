"""The writing loop against a stub endpoint: what stops it, and when."""

import time
from types import SimpleNamespace

from memory.agent_runtime.openai_agent import OpenAIAgentConfig, OpenAIWriterAgent


class _NeverFinishes:
    """An endpoint that answers every turn with another tool call.

    Left alone the loop would run to its turn ceiling, which is what makes it
    the case worth testing: whatever stops it is the limit under test, not the
    model deciding it is done.
    """

    def __init__(self, seconds_per_turn: float = 0.0):
        self.seconds_per_turn = seconds_per_turn
        self.calls = 0
        self.timeouts: list[float] = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, *, timeout: float, **_: object) -> "_NeverFinishes":
        self.timeouts.append(timeout)
        return self

    def create(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        time.sleep(self.seconds_per_turn)
        call = SimpleNamespace(
            id=f"call_{self.calls}",
            function=SimpleNamespace(name="remember", arguments="{}"),
        )
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[call]),
                finish_reason="tool_calls",
            )],
        )


def _agent(client: _NeverFinishes) -> OpenAIWriterAgent:
    return OpenAIWriterAgent(
        OpenAIAgentConfig(base_url="http://stub", api_key="k", model="m"),
        client=client,
    )


def _run(client: _NeverFinishes, tmp_path, **limits):
    return _agent(client).run(
        prompt="p", system_prompt="s", cwd=tmp_path, tools=[], **limits
    )


def test_the_turn_ceiling_stops_a_loop_that_would_not_stop_itself(tmp_path):
    result = _run(_NeverFinishes(), tmp_path, max_turns=4)

    assert result.stop_reason == "max_turns"
    assert result.num_turns == 4


def test_a_spent_wall_clock_budget_stops_the_loop_before_the_turn_ceiling(tmp_path):
    client = _NeverFinishes(seconds_per_turn=0.05)

    result = _run(client, tmp_path, max_turns=50, max_seconds=0.12)

    assert result.stop_reason == "max_seconds"
    # Turns of 0.05s against a 0.12s budget: the count is what the budget
    # bought, not the ceiling it never reached.
    assert 2 <= result.num_turns < 50
    assert client.calls == result.num_turns
    # Each call is handed only the time that is left, so no one call can
    # outlast the budget the way the client's own three-minute default would.
    assert client.timeouts and max(client.timeouts) <= 0.12


def test_the_budget_never_costs_the_first_turn(tmp_path):
    """A caller that arrives with no time left still gets one attempt.

    The alternative is an Add that reports success having written nothing,
    which is worse than answering late.
    """
    client = _NeverFinishes(seconds_per_turn=0.05)

    result = _run(client, tmp_path, max_turns=50, max_seconds=0.0001)

    assert result.num_turns == 1
    assert client.calls == 1
