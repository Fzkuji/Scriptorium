# Agent Memory Harness plugin

Usage guidance for the `agent-memory` MCP server. The plugin is optional: it
adds a skill describing when to read and write memory, and nothing else. All
capability comes from the server, which works without this plugin installed.

The plugin deliberately does not register an MCP server of its own. You
register the server yourself, pointing at your workspace:

```bash
agent-memory init ~/memory
claude mcp add --scope user agent-memory -- \
  agent-memory mcp --workspace ~/memory
```

Bundling a second server here would collide with that registration, and the
workspace path cannot be guessed.

Install the plugin from a local checkout:

```bash
claude plugin validate ./claude-plugin --strict
```

Full setup, tool reference and error codes:
[`docs/integrations/claude-code.md`](../docs/integrations/claude-code.md).
