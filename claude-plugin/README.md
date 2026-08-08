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

The plugin deliberately does not register an MCP server of its own. You
register the server yourself, pointing at your workspace:

```bash
scriptorium init ~/memory
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace ~/memory
```

Bundling a second server here would collide with that registration, and the
workspace path cannot be guessed.

Install the plugin from a local checkout:

```bash
claude plugin validate ./claude-plugin --strict
```

Full setup, tool reference and error codes:
[`docs/integrations/claude-code.md`](../docs/integrations/claude-code.md).
