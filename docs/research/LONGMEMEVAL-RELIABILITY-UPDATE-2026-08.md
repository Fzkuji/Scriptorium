# LongMemEval reliability update (August 2026)

This note summarizes reusable implementation changes developed during the
formal LongMemEval-S runs on macOS and WSL. It is intentionally separate from
generated memories, answers, judge outputs, secrets, and machine-local launch
state, which are not stored in Git.

## Scope

The update improves failure containment, checkpoint safety, auditability, and
cross-machine handoff. It does not change the LongMemEval questions, gold
answers, retrieval semantics, answer prompt, or judge protocol.

The detailed WSL incident chronology remains in
`docs/research/LONGMEMEVAL-WSL-EXPERIMENT-FINDINGS.md`. Operational setup and
handoff instructions remain in `docs/research/MAC-LONGMEMEVAL-RUNBOOK.md` and
`docs/research/LONGMEMEVAL-WSL-RUNBOOK.md`.

## Runtime reliability changes

### Bounded generic validation repair

Memory-writer validation failures may enter a bounded generic repair loop. The
limit is configured with `generic_repair_max_trajectories` and is exposed by
both the LongMemEval runner and the conversation runner. Every repair sees the
preceding validation error, and each attempt is retained in the writer
trajectory audit. Exhaustion remains a hard failure; invalid staged memory is
never committed.

Dangling relation failures receive targeted guidance containing the missing
block identifier and explicit safety constraints. The implementation also
records attempt-numbered writer failure snapshots so retries do not overwrite
earlier evidence.

### SDK inactivity and buffering

The Claude Agent SDK adapter supports:

- a configurable inactivity timeout, defaulting to 1,800 seconds without an
  SDK message; and
- a 20 MiB SDK message buffer.

An inactivity timeout follows the existing failed-item path and preserves the
last durable build checkpoint. It does not reinterpret a silent trajectory as
success.

### Atomic stop-after-current-batch

LongMemEval construction recognizes an item-local
`.stop-after-current-batch` sentinel. The active writer batch is allowed to
commit or roll back atomically, after which the item records a recoverable
paused state.

On macOS, operators should remove the exact `launchctl` label after the
terminal checkpoint is visible; a registered service can otherwise be
rescheduled. WSL launch and resume use the same checkpoint and pause semantics
without depending on `launchctl`.

### Audited resume drift

Strict resume continues to reject config or code fingerprint changes. An
explicit `--allow-resume-drift` option permits a deliberate recovery and
appends a manifest migration containing the changed fields, previous values,
replacement values, and timestamp. This option is not an automatic fallback.

### Deterministic token handling and derived views

Token counting treats tokenizer sentinel spellings that occur as visible
benchmark text as ordinary text rather than control tokens. Derived memory and
block views report richer dangling-link context and include consistency fixes
covered by regression tests. These changes preserve the source text sent to
the writer.

## Reproducibility tooling

`code/scripts/analysis/build_longmemeval_frozen_inventory.py` creates a
deterministic inventory from atomically built checkpoints. The inventory is
the boundary for downstream answer generation: incomplete, failed, paused, or
duplicate items must not enter production QA.

Cohort manifests record index allocation and the expected source snapshot.
Machine-local worker configurations, queue assignments, launch wrappers, and
provider credentials are not part of this update.

## Verification coverage

Regression tests cover:

- SDK inactivity timeout and buffer propagation;
- successful generic repair, bounded exhaustion, and repair audit records;
- targeted dangling-link feedback;
- attempt-numbered failure snapshots;
- tokenizer sentinel text;
- derived-view consistency; and
- strict versus explicitly audited resume drift.

The focused macOS verification command is:

```bash
cd code
PATH=/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH \
  ../.venv/bin/python -m pytest \
  tests/agent_runtime/test_claude_code.py \
  tests/management/test_memory.py \
  tests/runtime/test_derived_views.py \
  tests/runtime/test_writer_capacity.py \
  tests/scripts/test_longmemeval_support.py
```

The GNU sed path is required on macOS because several shell-transaction tests
exercise GNU `sed -i` semantics. The focused suite passed 106/106 tests with
that path enabled on 2026-08-14.

The shared implementation, runner, retrieval, management, runtime, and test
files in the WSL working tree were subsequently compared by content hash with
the private repository integration and found identical. Machine-local
experiment state was intentionally excluded from that comparison.

## Repository boundary

The following remain local and must not be committed:

- `code/results/` and generated memory workspaces;
- `exports/`, answer files, judge outputs, and packaged handoff archives;
- API keys and provider credential files;
- stdout/stderr, service labels, sentinels, live checkpoints, and monitor state;
- machine-specific worker configurations or queue assignments; and
- editor workspace state, caches, and generated previews.

The repository-level `.gitignore` excludes result roots, exports, packaged
archives, runtime caches, and common API-key filenames. Before publishing a
change, stage an explicit file list rather than using `git add -A`.

## Evaluation lock

This work does not modify the repository's user-locked LoCoMo evaluator.
LoCoMo scoring must continue to use only `scripts/evaluation/eval_full.py` at
the SHA-256 required by `AGENTS.md`. LongMemEval evaluation remains governed by
its existing benchmark-specific protocol.
