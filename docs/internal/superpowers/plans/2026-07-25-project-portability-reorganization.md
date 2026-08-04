# Project Portability and Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize NativeMem into `code/`, `paper/`, and `docs/` while preserving historical paths and making a copied checkout installable on another machine.

**Architecture:** Move implementation and experiment artifacts into `code/`, keep relative compatibility symlinks at the old root paths, and make `Model-Aligned-Wiki.html` the sole root documentation page. Add a Python 3.12 setup script, dependency files, a third-party revision manifest, and a result-storage audit before removing any reconstructable local environments or duplicate copies.

**Tech Stack:** Git, POSIX shell, Python 3.12, pip, pytest, relative symbolic links, JSON.

## Global Constraints

- Do not delete completed formal results or alter result contents.
- Do not rename individual result runs.
- Preserve third-party source and checked-out revisions.
- Do not store API credentials.
- Do not move the nested `paper/` repository.
- Keep old root paths operational through relative symlinks.
- Treat worker directories as distinct unless their files are proven identical.

---

### Task 1: Portable environment metadata

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `setup.sh`
- Create: `scripts/maintenance/verify_portable_layout.py`
- Test: `tests/test_portable_layout.py`

**Interfaces:**
- Consumes: current imports, Python 3.12, local benchmark and third-party directories
- Produces: `setup.sh`, `verify_portable_layout.py`, and a runnable portability test

- [ ] Write a test that requires Python setup files, relative compatibility links, required project directories, and credential-free dependency files.
- [ ] Run `pytest -q tests/test_portable_layout.py` and confirm it fails before the files exist.
- [ ] Add pinned direct dependencies used by core NativeMem, evaluation, plotting, and tests.
- [ ] Add `setup.sh` to create `.venv` with Python 3.12, install both requirement files, and run `python scripts/maintenance/verify_portable_layout.py`.
- [ ] Add the verification script without third-party dependencies.
- [ ] Run the focused test and the verification script.

### Task 2: Canonical directory layout

**Files:**
- Move: `src/` to `code/src/`
- Move: `scripts/` to `code/scripts/`
- Move: `tests/` to `code/tests/`
- Move: `benchmarks/` to `code/benchmarks/`
- Move: `experiments/` to `code/experiments/`
- Move: `figures/` to `code/figures/`
- Move: `gold_memory/` to `code/gold_memory/`
- Move: `results/` to `code/results/`
- Move: `third_party/` to `code/third_party/`
- Move: `experiment-plan.html` to `docs/experiment-plan.html`
- Modify: `.gitignore`
- Modify: `Model-Aligned-Wiki.html`

**Interfaces:**
- Consumes: the current root paths
- Produces: canonical `code/...` paths plus relative root symlinks

- [ ] Record pre-move result directory and file counts.
- [ ] Use Git-aware moves for tracked directories and filesystem moves for ignored directories.
- [ ] Create relative symlinks for every former root path.
- [ ] Update `.gitignore` for canonical paths and local `.venv`.
- [ ] Update the experiment-page link in `Model-Aligned-Wiki.html`.
- [ ] Run `python scripts/maintenance/verify_portable_layout.py`.
- [ ] Confirm post-move result directory and file counts match the pre-move counts.

### Task 3: Third-party reproducibility

**Files:**
- Create: `code/third_party/manifest.json`
- Create: `code/third_party/README.md`
- Create: `code/scripts/capture_third_party_manifest.py`

**Interfaces:**
- Consumes: each nested repository's remote URL, HEAD commit, and dirty status
- Produces: a credential-free revision manifest and reinstall instructions

- [ ] Add a test that validates unique names, HTTPS repository URLs, 40-character commits, and dirty-state recording.
- [ ] Implement the standard-library manifest generator.
- [ ] Generate the manifest from the current checkout.
- [ ] Document how to restore missing third-party repositories and create optional environments.
- [ ] Run the manifest test.

### Task 4: Result inventory and conservative cleanup

**Files:**
- Create: `code/results/storage-index.json`
- Create: `code/scripts/audit_result_storage.py`
- Test: `code/tests/test_audit_result_storage.py`

**Interfaces:**
- Consumes: result manifests, file sizes, statuses, and SHA-256 hashes
- Produces: a read-only inventory containing canonical runs, retries, archives, and exact duplicate groups

- [ ] Add a synthetic test proving that equal configurations with different worker outputs are not duplicates and equal files are reported.
- [ ] Implement the standard-library inventory scanner.
- [ ] Generate `storage-index.json` without modifying results.
- [ ] Replace only the confirmed `full_context 2` duplicate payload with a compressed archive after recording hashes and paths.
- [ ] Keep current formal results uncompressed.
- [ ] Run the storage audit test and verify recorded hashes.

### Task 5: Documentation and regression verification

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-plan.html`
- Modify: `docs/superpowers/specs/2026-07-25-project-file-organization-design.md`

**Interfaces:**
- Consumes: completed layout and generated manifests
- Produces: copy, install, configure, run, test, and restore instructions

- [ ] Rewrite README paths and commands for the canonical layout and compatibility links.
- [ ] Document required environment-variable names without values.
- [ ] Document copied-folder setup and Git-clone setup separately.
- [ ] Run `git diff --check`.
- [ ] Run the portability test and V11/LongMemEval core regression suite.
- [ ] Verify links in the two HTML entry pages.
- [ ] Commit the completed reorganization.
