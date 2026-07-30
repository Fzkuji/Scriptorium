# Window Selection Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize Section 4.7 into three navigable subsections whose figures separately communicate construction efficiency, memory granularity/coverage, and verifier recovery.

**Architecture:** Each subsection reads the existing `aggregate_by_window.csv` statistics through one reproducible standard-library Python script and exports SVG/PDF. `experiment-plan.html` embeds the SVG figures and exposes each `<h3>` through the existing static table of contents.

**Tech Stack:** HTML, Python standard library, SVG, CairoSVG CLI.

## Global Constraints

- Do not change sections outside 4.7 or its table-of-contents entries.
- Do not invent or manually hardcode result values; read `experiments/gpt56-chunk-curve/analysis/aggregate_by_window.csv`.
- Keep the existing raw QA and construction tables.

### Task 1: Generate three focused figures and restructure Section 4.7

**Files:**
- Create: `figures/window_selection/gen_construction_efficiency.py`
- Create: `figures/window_selection/gen_memory_granularity.py`
- Create: `figures/window_selection/gen_verifier_recovery.py`
- Modify: `experiment-plan.html`

**Interfaces:**
- Consumes: `aggregate_by_window.csv` with rows for windows 4, 8, 16, and 32.
- Produces: `construction_efficiency.{svg,pdf}`, `memory_granularity.{svg,pdf}`, and `verifier_recovery.{svg,pdf}`.

- [ ] Generate each SVG from the CSV and assert the four expected windows.
- [ ] Convert each SVG to vector PDF with `cairosvg`.
- [ ] Add `4.7.1`, `4.7.2`, and `4.7.3` `<h3>` headings and matching TOC links.
- [ ] Remove the old combined `build_statistics` figure from the HTML.
- [ ] Run `xmllint --noout` on SVG files, parse the HTML, and run `git diff --check`.
