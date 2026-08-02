"""Minimal OpenAI-compatible proxy that forwards to codebuddy CLI."""
import subprocess
import json
import uuid
import re
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        tools = body.get("tools", [])

        prompt_parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                prompt_parts.append(f"[{role}]: {content}")

        if tools:
            tool_desc = "Available tools:\n"
            for t in tools:
                fn = t.get("function", {})
                tool_desc += f"- {fn.get('name')}: {fn.get('description')}\n"
                params = fn.get("parameters", {}).get("properties", {})
                if params:
                    tool_desc += f"  Parameters: {json.dumps(params)}\n"
            prompt_parts.insert(0, tool_desc + "\nTo call a tool, output JSON: {\"tool_calls\": [{\"name\": \"...\", \"arguments\": {...}}]}\nIf you don't need a tool, just respond normally.\n")

        prompt = "\n".join(prompt_parts)

        try:
            result = subprocess.run(
                ["codebuddy", "-p", prompt],
                capture_output=True, text=True, timeout=60,
                input=""
            )
            text = result.stdout.strip()
            if not text:
                text = result.stderr.strip() or "No response"
        except subprocess.TimeoutExpired:
            text = "Timeout"
        except Exception as e:
            text = f"Error: {e}"

        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Check if response contains tool calls
        tool_calls = None
        if tools and '{"tool_calls"' in text:
            try:
                tc_match = re.search(r'\{.*"tool_calls".*\}', text, re.DOTALL)
                if tc_match:
                    tc_data = json.loads(tc_match.group())
                    tool_calls = []
                    for tc in tc_data.get("tool_calls", []):
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {}))
                            }
                        })
            except:
                pass

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text if not tool_calls else None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4, "total_tokens": (len(prompt) + len(text)) // 4},
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # suppress logs

if __name__ == "__main__":
    port = 8199
    print(f"Codebuddy proxy on http://localhost:{port}")
    HTTPServer(("localhost", port), Handler).serve_forever()
