"""In-process shim launcher for the MIRIX REST server (used by run_mirix.py).

Why this exists (instead of third_party/mirix/scripts/start_server.py):
MIRIX's OpenAIClient.build_request_data forces tool_choice="required" (or a
forced-function object) whenever tools are present
(third_party/mirix/mirix/llm_api/openai_client.py:152). Aliyun's
OpenAI-compatible endpoint rejects that for hybrid-thinking models
("The tool_choice parameter does not support being set to required or object
in thinking mode", HTTP 400). Verified fix: send extra_body
{"enable_thinking": false}, after which both "required" and forced-function
tool_choice succeed.

This launcher monkeypatches OpenAIClient.build_request_data IN PROCESS to
append that extra_body -- no file in third_party/mirix is modified. Everything
else (SQLite backend, BM25 retrieval, agents) is stock MIRIX.

Usage: <mirix-venv-python> src/adapters/_mirix_server_shim.py <port>
Env: BUILD_EMBEDDINGS_FOR_MEMORY=false is set by run_mirix.py.
"""

import os
import sys

MIRIX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party", "mirix")
sys.path.insert(0, MIRIX_ROOT)

from mirix.llm_api.openai_client import OpenAIClient  # noqa: E402

_orig_build_request_data = OpenAIClient.build_request_data


async def _patched_build_request_data(self, *args, **kwargs):
    data = await _orig_build_request_data(self, *args, **kwargs)
    extra_body = data.setdefault("extra_body", {})
    extra_body["enable_thinking"] = False
    return data


OpenAIClient.build_request_data = _patched_build_request_data

# Second in-process patch: MIRIX master has contradictory content-format
# expectations in /memory/retrieve/conversation (rest_api.py). has_content
# only accepts list-of-dict content ([{"type","text"}]), while
# extract_topics_and_temporal_info does `prefix + " " + part` and therefore
# only accepts list-of-str content. Net effect upstream: topic extraction
# ALWAYS fails ("can only concatenate str (not dict) to str"), silently
# degrading retrieval to recent-items-only. We normalize dict parts to their
# text before delegating so the LLM topic extraction + BM25 path actually runs.
from mirix.server import rest_api  # noqa: E402

_orig_extract = rest_api.extract_topics_and_temporal_info


async def _patched_extract(messages, llm_config):
    norm = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            content = [p.get("text", "") if isinstance(p, dict) else p for p in content]
            msg = {**msg, "content": content}
        norm.append(msg)
    return await _orig_extract(norm, llm_config)


rest_api.extract_topics_and_temporal_info = _patched_extract


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8531
    import uvicorn
    from mirix.server.rest_api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
