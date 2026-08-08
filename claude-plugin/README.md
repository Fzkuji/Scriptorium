# Scriptorium plugin

Usage guidance for the `scriptorium` MCP server. All capability comes from the
server; the plugin makes the model use it. Two pieces:

- A **SessionStart hook** injecting the memory protocol — when to search before
  answering, when to save, and the excuses that lose facts. Tools the model
  merely *has* go unused; a workspace nobody reads or writes is an empty
  directory with a schema.
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

For Codex, which has no marketplace, run `./install-codex.sh` from a
checkout. It writes the hook and the MCP entry into `~/.codex/`, leaves the
rest of that configuration alone, and reverses with `--uninstall`.

Full setup, tool reference and error codes:
[`docs/integrations/claude-code.md`](../docs/integrations/claude-code.md).
