# Project File Organization Design

## Goal

Separate implementation and experiment artifacts from the paper and project
documentation while preserving existing commands and historical result paths.

## Top-level structure

```text
model-aligned-wiki/
├── Model-Aligned-Wiki.html
├── code/
├── paper/
├── docs/
├── README.md
├── AGENTS.md
└── .gitignore
```

`Model-Aligned-Wiki.html` remains at the project root as the Research-Wiki
entry page. `README.md`, `AGENTS.md`, and `.gitignore` remain at the root
because they define repository use and configuration.

## Code workspace

```text
code/
├── src/
├── scripts/
├── tests/
├── benchmarks/
├── experiments/
├── figures/
├── gold_memory/
├── results/
└── third_party/
```

The existing NativeMem version split under
`code/src/nativemem_versions/{v8,v10,v11}/` remains unchanged.

## Documentation

`experiment-plan.html` moves to `docs/experiment-plan.html`. Method documents,
protocols, provider notes, experiment reports, and implementation plans remain
under `docs/`. Links in `Model-Aligned-Wiki.html`, documentation, scripts, and
tests are updated to the new canonical paths.

## Compatibility

The root retains compatibility symlinks for paths used by existing commands or
historical records:

```text
src -> code/src
scripts -> code/scripts
tests -> code/tests
benchmarks -> code/benchmarks
experiments -> code/experiments
figures -> code/figures
gold_memory -> code/gold_memory
results -> code/results
third_party -> code/baselines/third_party
experiment-plan.html -> docs/experiment-plan.html
```

New documentation uses canonical `code/...` and `docs/...` paths. Compatibility
links are not treated as canonical locations.

## Preservation rules

- Do not delete, rename, or merge individual result directories.
- Do not modify third-party repository contents.
- Preserve file history with Git moves.
- Do not move the nested `paper/` repository.
- Do not include local virtual environments, caches, logs, or `.DS_Store`.
- Update relative links after moving the experiment page.

## Portable setup

- Require Python 3.12 for the supported local environment.
- Provide a root setup command that creates `.venv`, installs the core and
  development requirements, and runs an import smoke check.
- Record every third-party repository URL and checked-out commit.
- Treat copied benchmark data, results, and third-party source as portable
  project data; do not rely on machine-specific absolute paths.
- Document required API environment variables without storing credentials.
- Keep provider-specific and heavyweight third-party dependencies optional.

## Storage cleanup

- Preserve completed formal results in an immediately readable form.
- Preserve manifests, final answers, scores, checkpoints, and failure reasons.
- Archive only superseded runs, failed attempts, retry archives, and confirmed
  duplicate copies.
- Record every archived or removed duplicate path in a machine-readable index.
- Local virtual environments are reconstructable and are excluded from the
  portable project.
- Never classify parallel worker directories as duplicates solely because their
  manifests share a configuration; workers may contain different claimed
  samples.

## Verification

- Confirm every compatibility path resolves to its canonical target.
- Run `git diff --check`.
- Run the NativeMem V11 and LongMemEval core tests.
- Check that links from `Model-Aligned-Wiki.html` and
  `docs/experiment-plan.html` resolve.
- Confirm result directory names and counts are unchanged.
- Create a fresh Python 3.12 environment through the documented setup command.
- Run the import smoke check from the project root.
