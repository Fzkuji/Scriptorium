# LongMemEval macOS execution and merge runbook

This document describes how to run a non-overlapping LongMemEval build shard on
macOS and merge the completed artifacts with runs produced under WSL.  The Mac
and WSL workers must use the same source snapshot and experiment protocol.

## What is and is not stored in Git

The private repository contains source code, tests, configuration examples,
and this runbook.  It intentionally excludes:

- API keys and other credentials;
- benchmark conversations and question data;
- generated memory workspaces and experiment results;
- Python virtual environments and machine-local state.

Transfer the dataset to the Mac separately and keep the provider key in a local
file.  Never add either file with `git add -f`.

## Required protocol identity

For results to belong to the same experimental cohort, preserve all of the
following across WSL and macOS:

- the exact repository commit;
- the same LongMemEval dataset file and SHA-256;
- builder model, provider, base URL, and generation settings;
- `writer_input_token_cap=15000` and `max_turns=120`;
- `core_max_tokens=3000` and `core_repair_target_tokens=2700`;
- the same batch, verification, final-management, and source-verification
  settings;
- compatible Python dependencies and tokenizer behavior.

Machine-local output paths may differ.  Each sample index must be owned by
exactly one active worker.

## macOS setup

From a terminal on the Mac:

```bash
git clone https://github.com/Qi202/Scriptorium.git
cd Scriptorium
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Place the dataset at:

```text
code/benchmarks/longmemeval/data/longmemeval_s_cleaned.json
```

Place the provider key at the repository root as `provider-api-key.txt`, then
restrict its permissions:

```bash
chmod 600 provider-api-key.txt
```

The filename is ignored by Git.  Do not print the file or put its value in a
command line, shell history, configuration file, or log.

## One-sample compatibility smoke

Before starting a multi-worker Mac shard, copy an existing 3k configuration to
a new Mac-specific file.  Change only:

- `output_dir` to a local APFS path such as
  `~/scriptorium-runs/longmemeval-mac-smoke-r1`;
- `start` to one unassigned sample index;
- `limit` to `1`;
- `api_key_file` to `../../../provider-api-key.txt` if it is included in the
  JSON, or pass `--api-key-file ../provider-api-key.txt` at launch.

Run from the `code` directory:

```bash
../.venv/bin/python scripts/runners/run_longmemeval.py \
  --config scripts/configs/<mac-smoke-config>.json \
  --api-key-file ../provider-api-key.txt
```

Confirm that the sample reaches `status=complete`, all batches and sessions are
committed, `final_management=complete`, and the component/verification files
are present before scaling out.

## Splitting work between WSL and macOS

Allocate static, non-overlapping index ranges.  Do not let both machines write
the same sample and do not use a shared iCloud, SMB, NFS, or network directory
as a live output root.  Each worker should write to local storage; copy only
completed artifacts or checkpoint-safe stopped items.

The WSL cohort currently owns index 125--174.  Assign Mac workers only indices
outside that range unless the ownership plan is explicitly changed after the
WSL workers stop.

## Merging completed results

Merge by sample index, not by overwriting whole worker roots.  For every
completed sample retain:

- the item directory and `checkpoint.json`;
- `build-checkpoint.json` and `build.json`;
- Source, Topic, Timeline, Core, Recent, and Relations artifacts;
- verification and runtime metadata;
- the worker manifest and usage/provenance records.

Before aggregation, reject duplicate indices and verify source/config/data
hashes.  A failed or incomplete sample must not be counted as complete.

In-progress checkpoints contain absolute paths in the outer checkpoint and
manifest.  A cross-machine resume therefore requires path rewriting or a new
matching manifest.  Completed outputs are portable and do not need rebuilding.

## macOS-specific checks

- `/bin/bash` is sufficient for the current POSIX shell backend, but keep
  commands compatible with the Bash version installed on the Mac.
- Use local APFS storage for atomic rename and file-lock behavior.
- Verify executable permissions after cloning.
- Compare the first committed batch's seconds per agent turn and error rate
  against the WSL cohort before increasing concurrency.
- Start with five or fewer provider lanes; do not increase concurrency solely
  because a second machine is available.

