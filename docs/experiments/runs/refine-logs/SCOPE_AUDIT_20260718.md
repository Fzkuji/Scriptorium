# Scope Audit: Paper Changes and Writer Input Granularity Experiment

Date: 2026-07-18

## Repository boundaries and recovery points

- `paper/` is an independent Git repository. Its current `main` and `origin/main` both point to `623a493`.
- The surrounding `model-aligned-wiki/` directory belongs to the parent `/Users/fzkuji/Documents/Research-Wiki` repository. It must not be staged with a repository-wide `git add -A`.
- Local paper recovery ref: `safety/pre-scope-audit-20260718` at `623a493`.
- Local outer-repository snapshot: `safety/model-aligned-wiki-pre-scope-audit-20260718` at `5e4956c`.
- Local interrupted-run snapshot: `safety/model-aligned-wiki-post-interrupt-20260718` at `3d78bf0`.
- None of the safety refs were pushed.

## What commit 623a493 contains

Commit `623a493` was created on 2026-07-17 by staging the entire dirty `paper/` repository. It contains work produced between 2026-07-12 and 2026-07-15, not the later Writer Input Granularity experiment.

The commit contains three groups:

1. Earlier manuscript and figure edits: `0main.tex`, `1Introduction.tex`, `2RelatedWork.tex`, `3Method.tex`, `4Experiments.tex`, `5Discussion.tex`, `5Conclusion.tex`, `INTRODUCTION_REWRITE_PLAN.md`, and the framework/alignment figure sources and PDFs.
2. Experiment-governance records created on 2026-07-14: plans, trackers, readiness audits, cost gates, preregistrations, and protocol-freeze files under `paper/refine-logs/`.
3. Generated paper PDFs.

No file in `623a493` was created by the GPT-5.6 Writer Input Granularity task. The operational error was committing all earlier uncommitted paper work in one upload commit, which made unrelated changes appear to belong to the new experiment.

## What the 2026-07-17 experiments actually changed

The earlier context-history experiment added its plan, tracker, runner, analyzer, tests, and two smoke-result trees outside `paper/`. These runs concern the form and amount of previous context supplied to the Writer.

The later Writer Input Granularity experiment added or updated:

- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `experiments/gpt56-chunk-curve/`
- `scripts/prepare_gpt56_chunk_curve.py`
- `scripts/run_gpt56_chunk_curve.py`
- `tests/test_prepare_gpt56_chunk_curve.py`
- `tests/test_run_gpt56_chunk_curve.py`
- `figures/mock_gpt56_chunk_curve/`

The mock figures and tables contain synthetic hypothetical values and are not experimental evidence.

Two `gpt-5.6-luna + BEAM-100K` pilots (`W=session` and `W=32`) were started with model requests enabled by a previously active background task. Both were interrupted on 2026-07-18, completed zero configurations, and produced no admissible result. Their status and usage records are preserved only as interrupted execution records.

## Scope decision

The paper should not be globally rewritten for this experiment.

The only eventual manuscript change is one experiment subsection in `4Experiments.tex`, provisionally titled `Effect of Writer Input Granularity on Memory Quality and Construction Cost`. It should contain one real performance--cost figure and one compact table after audited results exist. It does not require changes to the Abstract, Introduction, Related Work, Method, Discussion, Conclusion, or framework figure unless the completed evidence changes a paper-level claim.

Before any real-results commit, the experiment plan must be updated from “not started” to the actual interrupted state, and the frozen manifest must cover the runner and relevant runtime hashes. Mock artifacts and interrupted runtime logs must remain separate from real results.

## Normal commit boundaries

Future normal commits should be separated into:

1. Writer implementation, tests, and method documentation.
2. GPT-5.6 plan, frozen matrix, runner, and runner tests.
3. Proxy or benchmark-specific infrastructure, each with its own tests.
4. Mock visualization sources, explicitly labelled synthetic.
5. Audited real results only after a configuration completes.
6. The single paper experiment subsection after real results are admitted.

Large benchmark data, third-party repositories, virtual environments, cache files, proxy runtime files, memory trees, checkpoints, and incomplete run logs must not enter normal commits.
