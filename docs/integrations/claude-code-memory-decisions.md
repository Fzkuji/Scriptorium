# Claude Code memory integration decisions

Status: discussion record. This document separates decisions stated by the
user from matters that remain open. An open item is not an implementation
requirement until it is moved into the confirmed section.

Last updated: 2026-08-12

## Confirmed decisions

### Memory scope and placement

- The system has both global memory and project memory.
- Placement is determined from the content of each memory, not by a fixed
  category table.
- A model decides whether a memory belongs in the global or project layer.

### Memory selection

- A model decides what is worth saving from the available conversation data.
- There is no predefined allowlist of memory categories.

### Background processing

- Normal memory construction runs in the background.
- Each Claude Code session tracks and saves its own new conversation content.
- Multiple sessions write into the same configured memory address.
- The background Writer uses a cheap model by default.
- The Writer model is configurable manually and may also be configured by an
  agent.
- Failed background work must be retried.
- Memory organization runs in the background.
- Retrieval-quality checking runs in the background; the conversational agent
  is not responsible for it.

### Active saving

- Background writing is the main path, but an active saving mechanism is also
  required.
- The exact active-saving behavior is not yet decided.

### Version history

- Every memory workspace is a Git repository.
- Memory changes are committed automatically.
- Git is a preservation and recovery mechanism for accidental or incorrect
  changes.
- Git operations are owned by the Runtime, not by a model.

### Quality objective

- The primary quality objective is whether the system can quickly find the
  memory needed for a later request.
- Retrieval-quality checks may run as part of background organization.

## Discussed direction, not yet fixed

- Trigger background saving after roughly 100,000 tokens of accumulated
  context.
- Use a Claude Code plugin as the installation surface.

These are current directions only. Their exact semantics have not been
approved.

## Open questions

- The token or lifecycle condition that triggers processing within each
  session; 100,000 tokens is only a candidate value.
- When pending content below the threshold is processed.
- What an active save requests and whether it causes immediate background
  processing.
- Which cheap model is the default Writer.
- How manual and agent-driven model configuration are represented.
- Retry limits, delay policy, persistence, and treatment of deterministic
  failures.
- The final plugin, hook, MCP, and worker installation boundary.
- The concrete retrieval-quality queries, metrics, thresholds, and response to
  a failed check.
- Whether any safety or privacy rule limits the model's freedom to select
  memories.
