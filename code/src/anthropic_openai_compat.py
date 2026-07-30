"""Minimal OpenAI chat-completions facade over Anthropic Messages."""

from types import SimpleNamespace

from anthropic import Anthropic


def _content(value):
    return value if isinstance(value, str) else ""


def _messages(messages):
    system = []
    converted = []
    for message in messages:
        get = message.get if isinstance(message, dict) else lambda key, default=None: getattr(message, key, default)
        role = get("role", "assistant")
        if role == "system":
            system.append(_content(get("content")))
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": get("tool_call_id"),
                "content": _content(get("content")),
            }
            if converted and converted[-1]["role"] == "user" and isinstance(
                converted[-1]["content"], list
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        blocks = []
        text = _content(get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in get("tool_calls") or []:
            function = call.function if hasattr(call, "function") else call["function"]
            arguments = (
                function.arguments
                if hasattr(function, "arguments")
                else function.get("arguments", "{}")
            )
            import json

            try:
                arguments = json.loads(arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            blocks.append({
                "type": "tool_use",
                "id": call.id if hasattr(call, "id") else call["id"],
                "name": function.name if hasattr(function, "name") else function["name"],
                "input": arguments,
            })
        converted.append({"role": role, "content": blocks or text})
    return "\n\n".join(filter(None, system)), converted


class AnthropicOpenAICompat:
    def __init__(self, api_key, base_url):
        self._client = Anthropic(api_key=api_key, base_url=base_url)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model, messages, tools=None, max_tokens=4096, **kwargs):
        system, messages = _messages(messages)
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [{
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"],
            } for tool in tools]
        response = self._client.messages.create(**request)
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            SimpleNamespace(
                id=block.id,
                function=SimpleNamespace(
                    name=block.name,
                    arguments=__import__("json").dumps(block.input),
                ),
                type="function",
            )
            for block in response.content if block.type == "tool_use"
        ]
        usage = SimpleNamespace(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        return SimpleNamespace(
            id=response.id,
            model=response.model,
            usage=usage,
            choices=[SimpleNamespace(
                finish_reason={
                    "tool_use": "tool_calls",
                    "max_tokens": "length",
                }.get(response.stop_reason, "stop"),
                message=SimpleNamespace(
                    role="assistant", content=text or None, tool_calls=calls or None
                ),
            )],
        )
