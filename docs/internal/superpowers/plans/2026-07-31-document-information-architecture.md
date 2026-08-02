# NativeMem Document Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every project document into a responsibility-based hierarchy under `docs/`, keep `Model-Aligned-Wiki.html` as the named entry page, and eliminate ambiguous `index.html` files.

**Architecture:** Five primary documents provide the public reading path. Supporting research, protocols, runs, prompts, archives, and process records live in explicit subdirectories and are linked from their owning primary document.

**Tech Stack:** Static HTML, Markdown, shared CSS, Python document-link tests, Git.

## Global Constraints

- Preserve existing filenames unless the filename is `index.html`.
- Do not retain compatibility copies at old paths.
- Preserve user-authored content while removing responsibility overlap from primary pages.
- Update every repository reference to moved files.
- Do not modify unrelated implementation or experiment code.

### Task 1: Move documents according to the manifest

**Files:**
- Move: `Model-Aligned-Wiki.html` to `docs/Model-Aligned-Wiki.html`
- Move: root-level `docs/*.md` into `related-work/evidence/`, `experiments/*`, or `internal/analysis/`
- Move: `docs/prompts_collection/` to `docs/prompts/`
- Move: `docs/refine-logs/` to `docs/experiments/runs/refine-logs/`
- Move: `docs/superpowers/` to `docs/internal/superpowers/`
- Rename: `docs/method/index.html` to `docs/method/nativemem-method.html`
- Rename: `docs/method/report/index.html` to `docs/method/reports/method-evolution-results.html`

- [ ] Create the destination directories.
- [ ] Move each file once without creating compatibility copies.
- [ ] Confirm that no `index.html` remains.

### Task 2: Repair the five primary HTML pages

**Files:**
- Modify: `docs/Model-Aligned-Wiki.html`
- Modify: `docs/related-work/related-work.html`
- Modify: `docs/related-work/survey.html`
- Modify: `docs/method/nativemem-method.html`
- Modify: `docs/experiments/experiment.html`

- [ ] Update CSS, image, breadcrumb, home, survey, method, experiment, result, and evidence links.
- [ ] Keep the total entry focused on overview and document routing.
- [ ] Remove cross-section duplication where a canonical supporting document already exists.
- [ ] Use the shared page shell on every primary HTML page.

### Task 3: Repair Markdown and repository references

**Files:**
- Modify: `README.md`
- Modify: moved Markdown files containing relative links
- Modify: `code/tests/test_document_pages.py`

- [ ] Replace references to every old path with its destination path.
- [ ] Update method and experiment document maps.
- [ ] Update page tests to assert the new canonical files and shared CSS paths.

### Task 4: Validate the complete documentation tree

**Files:**
- Test: `code/tests/test_document_pages.py`

- [ ] Run `pytest -q code/tests/test_document_pages.py`.
- [ ] Parse local HTML and Markdown links and report unresolved targets.
- [ ] Search the repository for `Model-Aligned-Wiki.html`, old moved paths, and `index.html` references.
- [ ] Run `git diff --check` on all documentation changes.
- [ ] Open `docs/Model-Aligned-Wiki.html` for final inspection.
