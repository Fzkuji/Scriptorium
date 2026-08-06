# Scriptorium — Add / Search service

The Agent Memory Leaderboard evaluates a memory system through two synchronous
HTTP endpoints. This directory serves Scriptorium behind that contract. The
platform owns Answer and Eval; everything here is Add, Search, and health.

## Run with Docker

Build from the repository root:

```bash
docker build -f code/scriptorium_serve/Dockerfile -t scriptorium-memory .
```

Run. Credentials are injected at start and are never baked into the image:

```bash
docker run -p 8000:8000 \
  -e SCRIPTORIUM_WRITER_BASE_URL=https://openrouter.ai/api/v1 \
  -e SCRIPTORIUM_WRITER_API_KEY=<key> \
  -e SCRIPTORIUM_WRITER_MODEL=openai/gpt-4o-mini \
  -e SCRIPTORIUM_SERVICE_TOKEN=<bearer token> \
  -v scriptorium-workspaces:/data/workspaces \
  scriptorium-memory
```

| Variable | Required | Meaning |
|---|---|---|
| `SCRIPTORIUM_WRITER_BASE_URL` | yes | OpenAI-compatible chat-completions endpoint |
| `SCRIPTORIUM_WRITER_API_KEY` | yes | Key for that endpoint |
| `SCRIPTORIUM_WRITER_MODEL` | no | Defaults to `gpt-4o-mini`, which the academic track mandates |
| `SCRIPTORIUM_SERVICE_TOKEN` | no | Bearer token for Add/Search. Unset means no auth |
| `SCRIPTORIUM_WORKSPACES` | no | Where per-user memory lives. Defaults to `/data/workspaces` in the image |

Without Docker:

```bash
pip install -r requirements.txt fastapi 'uvicorn[standard]'
PYTHONPATH=code uvicorn scriptorium_serve.server:app --host 0.0.0.0 --port 8000
```

## Endpoints

`POST /add` stores one chunk of messages for one `user_id`. It returns 200 only
after the write is committed and retrievable, as the contract requires.

`POST /search` returns memory evidence for a `user_id`, ordered by descending
relevance, capped at `top_k`. It never synthesises an answer.

`GET /health` is unauthenticated and returns 200 when the service is up.

## Checking a deployment

```bash
python -m scriptorium_serve.smoke --base-url http://127.0.0.1:8000 --token <token>
```

This runs the platform's own sequence: health, Add, immediate Search, and a
cross-user isolation probe. It reports the Add latency against the 1200s limit.

Offline tests, no endpoint or key needed:

```bash
cd code && python -m pytest scriptorium_serve -q
```

## How it works

Add hands the chunk to Scriptorium's writer, which edits a per-user Markdown
workspace inside a staged transaction. The model writes topic notes in which
every sentence is footnoted to the message it came from; a patch that cites a
source it was not given, or that breaks the topic format, is refused whole and
the workspace is left byte-identical. Source messages are append-only.

Search runs BM25 over the topics the writer authored and the archived source
messages, and returns the matching blocks as evidence.

Isolation is by directory: each `user_id` gets its own workspace, so no query
can reach another user's memory. Writes to one workspace are serialised.

## Design notes

Add runs one writer pass over the chunk and does not trigger cross-session
reorganisation. A chunk is at most 20 messages, and a request must finish
inside the platform's 1200s timeout, so whole-workspace maintenance is left out
of the request path.

BM25 returns up to 100 results here because the platform fixes `top_k` at 100.

Evaluation data is treated as transient: it is not used for training or
analysis, and message contents are not written to logs.

## Provenance

Scriptorium is our own system; this service is a thin contract adapter over it.
The writing protocol, transaction rules, Markdown format and BM25 retrieval all
live in `code/src` and are unchanged by this directory.

The writer normally runs inside Claude Code, which speaks Anthropic's protocol.
Because the academic track mandates gpt-4o-mini, `src/agent_runtime/openai_agent.py`
drives the same protocol against an OpenAI-compatible endpoint instead. It is a
drop-in for `ClaudeCodeAgent`: same call signature, same result type, same
workspace, tools and prompts. Only the model transport differs.
