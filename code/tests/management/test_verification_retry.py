"""Verification retries a missing structured output instead of aborting."""

import pytest

from src.agent_runtime import AgentResult
from src.management import verification
from src.management.config import MemoryConfig


def result(structured):
    return AgentResult(
        text="", structured_output=structured, num_turns=1,
        input_tokens=1, output_tokens=1,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
        anthropic_equivalent_cost_usd=0.0,
        duration_ms=1, duration_api_ms=1,
        stop_reason="end_turn", session_id="s",
    )


class Flaky:
    """Misses structured output on the first call, succeeds on the second."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        return result(self.outputs.pop(0))


def test_transient_miss_is_retried(tmp_path):
    agent = Flaky([None, {"question": "q"}])

    got = verification._structured(
        agent, prompt="p", schema={}, cwd=tmp_path,
        usage_logger=None, config=MemoryConfig(),
    )

    assert got == {"question": "q"}
    assert agent.calls == 2


def test_persistent_miss_still_fails(tmp_path):
    agent = Flaky([None] * verification.STRUCTURED_OUTPUT_ATTEMPTS)

    with pytest.raises(ValueError, match="attempts"):
        verification._structured(
            agent, prompt="p", schema={}, cwd=tmp_path,
            usage_logger=None, config=MemoryConfig(),
        )

    assert agent.calls == verification.STRUCTURED_OUTPUT_ATTEMPTS
