# Scriptorium plugin

Usage guidance for the `scriptorium` MCP server. All capability comes from the
server; the plugin makes the model use it. Three pieces:

- A **SessionStart hook** injecting the memory protocol — when to search
  before answering, and the excuses that lead to answering from nothing.
  Tools the model merely *has* go unused.
- A **Stop hook** handing each finished turn to `scriptorium ingest`, which
  writes the conversation into memory once about 16k tokens of it have
  accumulated. Saving is not the model's job and does not interrupt it.
- A **skill** with the mechanics of each call: which tool finds what, how one
  `memory_update` is shaped, what each rejection code means.

The server works without the plugin. Memory rarely happens without it.

## Install

```bash
pip install git+https://github.com/Fzkuji/Scriptorium.git
claude plugin marketplace add Fzkuji/Scriptorium
claude plugin install scriptorium@scriptorium
```

The plugin carries its own MCP server pointed at `~/.scriptorium/memory`,
created on first use. A default path is what makes the install one command;
it is a starting point, not a recommendation to keep everything in one
workspace.

To put memory elsewhere, or to run one workspace per project, register a
server yourself and the plugin's own becomes redundant:

```bash
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace ~/memory
```

Codex reads the same directory through `.codex-plugin/plugin.json`, so the
skill, the hooks and the MCP server come across unchanged.

Background writing runs as you: it uses the login and default model your own
CLI already has, and needs no second API key. To spend less on it, point the
writer at a cheaper model with `scriptorium ingest --model claude-haiku-4-5`.

Full setup, tool reference and error codes:
[`docs/integrations/claude-code.md`](../docs/integrations/claude-code.md).
