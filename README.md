# Scriptorium, as embedded in OpenProgram

This branch is a deployment artifact, not a place to develop. It holds
exactly the files that live at `openprogram/memory/scriptorium/` in the
OpenProgram repository — same paths, same contents — so syncing is a
directory copy and drift is a `diff`.

    rsync -a --delete --exclude __pycache__ ./ \
        <openprogram>/openprogram/memory/scriptorium/

## What is here, and what is not

The implementation behind OpenProgram's `MemoryProvider`: the write
transaction, retrieval, the topic format, the prompts, and the process
that does the writing.

Not here: the benchmark harness, the question-answering agent and its
tool server, writer-capacity calibration, spend accounting,
multi-workspace layering, write-time verification. Those belong to the
research work and live on `main`.

Three files have no counterpart on `main` — they are the adapter, and
only make sense inside OpenProgram:

| File | Role |
|---|---|
| `provider.py` | Satisfies `MemoryProvider` |
| `writing.py` | Accumulate turns, write, reorganise |
| `__init__.py` | Exports the provider |

## Branches

| Branch | For |
|---|---|
| `main` | The research line — paper experiments and the full harness |
| `public` | What ships to Claude Code and Codex: the CLI, the MCP server, the plugin |
| `openprogram` | This one |

A change to the shared implementation is made on `main` and copied here;
a change to the adapter is made here, or in OpenProgram and copied back.
