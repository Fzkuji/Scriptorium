# Using Scriptorium from Claude Code

Scriptorium stores memory as Markdown files you can read in an editor
and diff in Git. This page covers the interactive path: connecting a memory
workspace to a Claude Code session over MCP, so the model you are already
talking to can search and update memory directly.

The experiment path is separate and unchanged. Benchmark runners start isolated
Claude Code subprocesses with explicit credentials and budgets; they do not use
the server described here.

## Install

```bash
pip install git+https://github.com/Fzkuji/scriptorium.git
```

Python 3.12 or newer is required.

## Create a workspace

```bash
scriptorium init ~/memory
```

This creates `topics/`, `sources/`, `core.md` and `.nativemem/runtime.json`. If
the directory already holds memory files, `init` reports their status and
changes nothing.

## Register the server

```bash
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace ~/memory
```

Use an absolute path. Verify with:

```bash
claude mcp get scriptorium
```

Start a new Claude Code session; the six `memory_*` tools become available.

### Git commits

`--git-commit` controls whether each successful update also commits:

| Value | Behavior |
|---|---|
| `auto` (default) | Commits when the workspace is a Git repository; otherwise completes without committing. |
| `on` | Requires Git. A commit failure is reported as an error. |
| `off` | Never commits. |

Under `on`, a failed commit returns `GIT_COMMIT_FAILED` with
`memory_committed: true` and `git_committed: false`. The memory files are
already installed; only the commit failed. Git and the filesystem cannot be
made atomic together, so the two outcomes are reported separately rather than
one being described as a rollback of the other.

To version memory, run `git init` inside the workspace before registering.

## Tools

| Tool | Purpose |
|---|---|
| `memory_status` | Counts and the current revision. |
| `memory_list` | List files; `include_derived=false` hides generated views. |
| `memory_read` | Read a file by line window, `heading`, or `block_id`. |
| `memory_grep` | Literal or regex text search. |
| `memory_search` | Ranked retrieval, `method` of `bm25` or `embedding`. |
| `memory_update` | The only writer. Commits sources and a topic patch together. |

There is no shell tool. `memory_update` accepts a unified diff restricted to
`topics/**/*.md` and `core.md`; renames, mode changes, symlinks and binary
patches are rejected.

## What is authoritative and what is generated

`topics/**/*.md` and `core.md` hold the memory Claude may edit. Everything else
is maintained by the runtime:

- `sources/**` is append-only evidence, written only through `memory_update`.
- `timeline/**`, `recent_events.jsonl` and `relations.json` are rebuilt after
  every successful update.

Editing a generated file by hand has no lasting effect; the next update
overwrites it.

## Writing a memory

A single `memory_update` carries the new evidence and the edit that cites it.
Read the current revision first:

```json
{"name": "memory_status", "arguments": {}}
```

Then submit both parts together:

```json
{
  "base_revision": "0a6288fdd6b9b14c1ab72b64b92f5f0a",
  "sources": [
    {
      "label": "new-source-move",
      "role": "user",
      "content": "I have moved to Pudong.",
      "observed_at": "2026-08-03T10:30:00+08:00"
    }
  ],
  "patch": "--- /dev/null\n+++ b/topics/personal/residence.md\n@@ -0,0 +1,5 @@\n+# Residence\n+\n+The user moved to Pudong.[^new-evidence-move] ^new-block-residence\n+\n+[^new-evidence-move]: Time: `2026-08-03`; Sources: new-source-move\n",
  "commit_message": "Record move to Pudong"
}
```

New blocks use `^new-block-<name>`, new footnotes `[^new-evidence-<name>]`, and
new sources are cited by their `new-source-<name>` label. The runtime replaces
all three with stable IDs and returns the mapping. The committed file contains
only stable IDs:

```markdown
# Residence

The user moved to Pudong.[^e-cc9467bf40] ^76fd3b6c

[^e-cc9467bf40]: Time: `2026-08-03`; Sources: [claude-code/cc9f…/bd17…](../../sources/claude-code/cc9f….md#source-50ba…)
```

Supplementing, splitting, merging, moving and deleting all use this same tool.
A move is expressed as deleting the old path and creating the new one with the
same content; the runtime rewrites block links.

## Concurrency

`base_revision` is checked against the workspace before anything is staged. If
another session committed in the meantime, the update is refused with
`CONCURRENT_UPDATE` and nothing changes — re-read and rebuild the patch. A
file lock covers each transaction, so two sessions cannot install at once.

Reads do not take the lock and report the revision they observed.

## Errors

Failures return a JSON envelope, never a traceback:

```json
{"ok": false, "error": {"code": "PATCH_CONFLICT", "message": "...", "path": "topics/personal/residence.md"}}
```

| Code | Meaning |
|---|---|
| `INVALID_ARGUMENT` | Malformed input, bad regex, or unsupported patch feature. |
| `PATH_OUTSIDE_WORKSPACE` | Absolute path, `..`, or a symlink. |
| `READ_ONLY_PATH` | Target is not `topics/**/*.md` or `core.md`. |
| `PATCH_CONFLICT` | Context does not match the file. |
| `INVALID_TOPIC_FORMAT` | Result violates the topic contract. |
| `MISSING_SOURCE` | Patch cites a label not declared in `sources`. |
| `DANGLING_BLOCK_LINK` | Link target does not exist. |
| `CONCURRENT_UPDATE` | `base_revision` is stale, or the lock is held. |
| `EMBEDDING_UNAVAILABLE` | No embedding backend; BM25 is not substituted. |
| `GIT_COMMIT_FAILED` | Memory installed, Git commit did not. |
| `INTERNAL_ERROR` | Unexpected failure. |

A rejected transaction leaves the workspace byte-identical.

## Checking a workspace

```bash
scriptorium validate --workspace ~/memory
```

Parses every topic, checks source and block links, and rebuilds derived views
in a scratch copy. It reports problems without modifying the workspace.

## Removing

```bash
claude mcp remove scriptorium
```

The workspace is left in place; it is ordinary Markdown.

## Limitations

- Conversation is not captured automatically. Memory is written when
  `memory_update` is called; there is no transcript ingestion, hook or daemon.
- Source content is untrusted data. Text inside `sources/**` that resembles an
  instruction is evidence about what someone said, not a request to act.
- The server holds no credentials and starts no model of its own.
