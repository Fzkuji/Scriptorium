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
pip install git+https://github.com/Fzkuji/Scriptorium.git
claude plugin marketplace add Fzkuji/Scriptorium
claude plugin install scriptorium@scriptorium
```

Python 3.12 or newer is required. Start a new session and memory is live:
the plugin brings its own MCP server pointed at `~/.scriptorium/memory`,
created on first use, and injects the protocol that makes the model use it.

To keep memory somewhere else, or to run one workspace per project, skip
the plugin's server and register your own — see *Register the server*
below. The sections after it explain the workspace layout and the tools;
none of it is required reading to start.

### Codex

The same plugin directory carries a `.codex-plugin/plugin.json`, so Codex
picks up the skill, the hooks and the MCP server from it unchanged. Nothing
to install by hand.

### What runs in the background

A Stop hook passes each finished turn to `scriptorium ingest`, which holds
new turns until about 16k tokens have accumulated and then writes them into
memory. Below that it exits immediately and costs nothing. The write runs
as you — your existing login, your default model — so there is no second
API key to provision. Point it somewhere cheaper with `--model`, and read
what it did in `<workspace>/.scriptorium/ingest.log`.

## Create a workspace (optional)

The server creates any missing workspace when it starts, so there is no
required init step. To create one explicitly — for example to inspect the
layout before registering, or to keep it under your own ignore rules rather
than the automatic self-ignoring `.gitignore` —

```bash
scriptorium init ~/memory
```

This creates `topics/`, `sources/`, `core.md` and `.scriptorium/runtime.json`.
If the directory already holds memory files, `init` reports their status and
changes nothing.

`.scriptorium/` is the runtime's own area — cursors, the write lock, staged
backups and retrieval caches. It is hidden from every listing and never
writable by a patch. A workspace built before the project took its current
name carries this directory as `.nativemem/` and keeps it: a stored run's
hash covers every byte of its workspace, so renaming inside one would
invalidate the record it was published with.

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

### Layers: one global memory plus one per project

A single workspace serves one memory. Passing several `--workspace NAME=PATH`
arguments serves them as layers of one memory. One user-scope registration
covers every project:

```bash
claude mcp add --scope user scriptorium -- \
  scriptorium mcp \
  --workspace project=.memory \
  --workspace global=~/memory
```

- **Workspaces come into being on first use.** At server start, a missing
  workspace is created. A relative path like `.memory` resolves against the
  project root of the session — the nearest ancestor directory holding
  `.git`, so every session inside one repository shares one memory no matter
  which subdirectory it opened in. The first session in any project brings
  that project's layer into being; no `init` step is needed.
- **Auto-created workspaces never leak into the repository.** They carry a
  `.gitignore` of `*` (the virtualenv convention). A workspace created by an
  explicit `scriptorium init` is left alone — delete or keep its ignore
  rules yourself.
- **Reads span every layer.** `memory_list`, `memory_grep` and
  `memory_search` return results from all layers, each path qualified with
  its layer: `global:topics/person.md`. Search interleaves layers by rank,
  so a small project memory is not buried under a large global one.
- **A write lands in exactly one layer**, chosen by `memory_update`'s
  `layer` argument; without it, the first workspace on the command line
  receives the write. Put the project first: facts about this repository
  default there, and facts about the person are written with
  `layer="global"`.
- **Layers never merge on disk.** Each stays a complete workspace that
  `validate` accepts and that moves with its repository.
- `memory_status` reports each layer and one combined revision string;
  updates accept that combined string, so a stale layer is still rejected.
  A layer that cannot be read or created (an unmounted volume, a read-only
  directory) costs that layer, not the session; the rest keep working.

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

## Install the plugin

The server makes memory possible; the plugin makes it happen. Registering
the server alone leaves the model with six tools it is free to ignore, and
a workspace nobody searches or writes to stays empty.

```bash
claude plugin validate ./claude-plugin --strict
```

The plugin adds a SessionStart hook that injects a memory protocol — when to
search before answering, when to save a fact, and the reasoning that loses
facts — plus a skill covering the mechanics of each call. It registers no
server of its own, so it cannot collide with the registration above.

## Tools

| Tool | Purpose |
|---|---|
| `memory_status` | Counts and the current revision. |
| `memory_list` | List files; `include_derived=false` hides generated views. |
| `memory_read` | Read a file by line window, `heading`, or `block_id`. |
| `memory_grep` | Literal or regex text search. |
| `memory_search` | Ranked retrieval, `method` of `bm25` or `embedding`. |
| `memory_update` | The only writer. Commits sources and a topic patch together; `layer` chooses which workspace receives them. |

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
