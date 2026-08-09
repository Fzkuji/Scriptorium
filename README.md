# Scriptorium

**Agent memory you can read.** The model keeps its own notes as Markdown files
— one topic per file, every sentence footnoted to the message it came from — so
memory opens in an editor, diffs in Git, and can always be traced back to what
was actually said.

```markdown
# Calvin's music career

## Craft and drive

Calvin writes new tunes, does studio sessions, and loves collaborating with
other artists.[^e-310b5c4c8e] ^36d94ab9

Calvin hit a creative block in May 2023. Dave advised taking a break, and
Calvin planned to follow that advice.[^e-078d330831][^e-fca34f2ddb] ^2b157171

[^e-310b5c4c8e]: Time: `2023-05-08`; Sources: [thread_4f2a…/msg_9c11…](../../sources/thread_4f2a….md#source-9c11…)
```

That file was written by the model, not by a template. Each `^id` is a block
another note can link to; each `[^e-…]` footnote carries the date and the
archived message it rests on. Nothing is stored that you cannot open.

## Install

```bash
pip install git+https://github.com/Fzkuji/Scriptorium.git
claude plugin marketplace add Fzkuji/Scriptorium
claude plugin install scriptorium@scriptorium
```

Start a new session and memory is live. The plugin brings its own MCP server
pointed at `~/.scriptorium/memory`, created on first use, and injects a
protocol telling the model when to search memory before answering and when to
save what you just told it. Registering the server without that protocol
leaves six tools the model is free to ignore.

For Codex, which has no plugin marketplace, clone the repository and run
`./claude-plugin/install-codex.sh`. It writes the hook and the MCP entry into
`~/.codex/`, leaves the rest of that configuration alone, and reverses with
`--uninstall`.

Full setup, the tool reference and the error codes:
[`docs/integrations/claude-code.md`](docs/integrations/claude-code.md).

## How it works

**One write is one transaction.** New evidence and the note citing it commit
together, checked against the revision you read. A patch that cites a source it
did not supply, links a block that does not exist, or breaks the topic format
is refused whole, and the workspace is left byte-identical.

**Evidence is append-only.** `sources/**` holds what was said, written only by
the runtime. A note can be rewritten freely; what it rests on cannot.

**Views are derived, not authored.** `timeline/`, `recent_events.jsonl` and
`relations.json` are rebuilt after every successful write, so the model never
maintains an index by hand.

**Retrieval reads files.** BM25 over blocks and sources by default, with an
embedding backend if one is installed — and if it is not, embedding search says
so rather than silently returning something else.

**Layers, not copies.** A per-repository memory and one global memory are read
as a single memory with qualified paths (`global:topics/person.md`); each layer
stays a complete workspace that moves with its repository.

## Checking a workspace

```bash
scriptorium validate --workspace ~/.scriptorium/memory
```

Parses every topic, checks source and block links, and rebuilds the derived
views in a scratch copy without touching the original.

## Layout

```text
code/
  scriptorium/     installable facade: CLI and MCP server
  src/
    management/    the write transaction, staging and validation
    retrieval/     search over blocks and sources
    markdown/      the topic format
    prompts/       what the model is told
    runtime/       workspace state and derived views
    agent_runtime/ Claude Code process adapter
  tests/           grouped by implementation responsibility
claude-plugin/     plugin, session hook and the memory protocol
docs/              setup and tool reference
```

## Development

```bash
pip install -r requirements.txt
cd code && python -m pytest
```

Python 3.12 or newer.

## License

MIT. See [LICENSE](LICENSE).
