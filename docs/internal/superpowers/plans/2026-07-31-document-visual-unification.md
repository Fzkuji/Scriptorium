# NativeMem Document Visual Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four current public HTML documents use one deterministic page shell and replace the stale NativeMem overview image with a state-centered multi-view architecture figure.

**Architecture:** A single local stylesheet owns document layout, typography, navigation, tables, callouts, and responsive behavior. Each page keeps only its content-specific CSS and JavaScript. The architecture figure has one editable SVG source; the PNG is generated from that SVG and is not edited separately.

**Tech Stack:** Static HTML5, CSS, inline SVG, Python standard library, pytest, macOS Quick Look PNG export.

## Global Constraints

- Modify only `Model-Aligned-Wiki.html`, `docs/survey.html`, `docs/method/index.html`, `docs/experiment-plan.html`, `docs/assets/document.css`, `code/figures/nativemem_overview.svg`, `code/figures/nativemem_overview.png`, and the focused validation test.
- Preserve all document prose, section IDs, local links, result values, and the Related Work versus Method boundary.
- Preserve the Experiments tab functions, DOM relocation calls, result-state markup, and every existing section ID.
- Use a fixed light theme, `256px` desktop sidebar, approximately `1080px` content width, `48px` desktop content padding, and `1024px` responsive breakpoint.
- Use `#4263eb` as the shared accent; retain page-specific semantic colors only where they encode experimental status or figure meaning.
- Add no package, JavaScript dependency, template system, site generator, or build step.
- Use `code/figures/nativemem_overview.svg` as the only editable figure source and derive the PNG from it.
- Do not stage or commit the four HTML files during implementation because they contain pre-existing uncommitted content changes. Inspect diffs by exact path and leave unrelated work untouched.

### Task 1: Shared document shell for Overview and Related Work

**Files:**
- Create: `docs/assets/document.css`
- Create: `code/tests/test_document_pages.py`
- Modify: `Model-Aligned-Wiki.html`
- Modify: `docs/survey.html`

**Interfaces:**
- Consumes: Existing HTML section IDs and `.toc-link` anchors.
- Produces: Shared classes `.document-shell`, `.document-sidebar`, `.document-main`, `.document-content`, `.document-breadcrumb`, `.document-header`, `.document-meta`, `.table-scroll`, and `.callout`.

- [ ] **Step 1: Write the initial structural tests**

Create `code/tests/test_document_pages.py` with a standard-library `HTMLParser` collector. For Overview and Survey, assert:

```python
PUBLIC_PAGE_STYLES = {
    "Model-Aligned-Wiki.html": "docs/assets/document.css",
    "docs/survey.html": "assets/document.css",
}

assert document.lang == "zh-CN"
assert expected_stylesheet in document.stylesheets
assert document.class_count["document-shell"] == 1
assert document.class_count["document-sidebar"] == 1
assert document.class_count["document-main"] == 1
assert len(document.ids) == len(set(document.ids))
assert set(document.fragment_links) <= set(document.ids)
```

The parser must also resolve local non-fragment `href` and `src` values relative to each page and report missing files. Ignore `http:`, `https:`, `mailto:`, and `data:` targets.

- [ ] **Step 2: Run the tests and confirm the old pages fail**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: failure because the pages use `lang="en"`, contain no shared stylesheet link, and do not expose the shared shell classes.

- [ ] **Step 3: Create the shared stylesheet**

Implement `docs/assets/document.css` with:

```css
:root {
  --doc-bg: #ffffff;
  --doc-surface: #f7f8fb;
  --doc-line: #dfe3ea;
  --doc-text: #202637;
  --doc-muted: #697386;
  --doc-accent: #4263eb;
  --doc-sidebar: 256px;
  --doc-content: 1080px;
  --b1: 0 0% 100%;
  --b2: 225 24% 97%;
  --b3: 220 16% 88%;
  --bc: 222 26% 17%;
}
```

Define the semantic shell, typography, TOC active state, breadcrumb, page header, links, code, blockquote, callout, figure, figcaption, table scrolling, progress bar, focus-visible state, reduced motion, and the `1024px` responsive layout. Below the breakpoint, convert the sidebar TOC into a compact horizontal scroll region instead of removing navigation.

- [ ] **Step 4: Migrate Overview and Survey to the shared shell**

For both pages:

- set `<html lang="zh-CN">`;
- remove DaisyUI, Tailwind, theme configuration, and duplicated common CSS;
- load the existing Google Fonts and the correct relative `document.css`;
- replace utility-based shell classes with the semantic shared classes;
- replace breadcrumb inline styles with `.document-breadcrumb`;
- place title, subtitle, and update information inside `.document-header` using `.document-lead` and `.document-meta`;
- retain Overview-only stat and semantic component CSS locally;
- retain all prose, IDs, links, tables, and scripts unchanged.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: all Overview and Survey assertions pass.

### Task 2: Migrate Method and Experiments without changing behavior

**Files:**
- Modify: `code/tests/test_document_pages.py`
- Modify: `docs/method/index.html`
- Modify: `docs/experiment-plan.html`

**Interfaces:**
- Consumes: The shared classes and CSS variables produced by Task 1.
- Produces: Four current pages with identical shell geometry and responsive behavior.

- [ ] **Step 1: Extend the tests to all four pages**

Add:

```python
PUBLIC_PAGE_STYLES.update({
    "docs/method/index.html": "../assets/document.css",
    "docs/experiment-plan.html": "assets/document.css",
})
```

Assert all four pages have the same shell classes, one `h1`, a breadcrumb, a page header, valid local paths, unique IDs, and valid fragment links. Assert Experiments still contains:

```python
for token in (
    "function moveSection(",
    "function switchTab(",
    "moveSection('protocol-block','protocol-slot')",
    "moveSection('benchmark-master-results','main-results-primary')",
):
    assert token in experiment_html
```

- [ ] **Step 2: Run the expanded tests and confirm Method and Experiments fail**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: failure for Method and Experiments because they have not yet adopted the shared shell.

- [ ] **Step 3: Migrate Method**

Replace Method’s private layout tokens and `.layout` shell with the shared stylesheet and semantic classes. Keep only Method-specific card/grid styles locally. Normalize breadcrumb, header, lead, metadata, tables, figure, footer, TOC activation, and progress behavior without changing prose or IDs.

- [ ] **Step 4: Migrate Experiments**

Replace only the duplicated document-shell styles and outer utility classes. Keep metric tabs, rank-table colors, result states, figure-specific rules, DOM relocation code, `switchTab`, and all result markup. Use the shared table overflow behavior so every table remains accessible at narrow widths.

- [ ] **Step 5: Run the four-page tests**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: all structural, anchor, local-link, and behavior-preservation assertions pass.

### Task 3: State-centered multi-view architecture figure

**Files:**
- Modify: `code/tests/test_document_pages.py`
- Create: `code/figures/nativemem_overview.svg`
- Regenerate: `code/figures/nativemem_overview.png`
- Modify: `Model-Aligned-Wiki.html`
- Modify: `docs/method/index.html`

**Interfaces:**
- Consumes: Current method definition in `docs/method/file_native_multiview_design.md`.
- Produces: One scalable source figure and one matching raster export referenced by the public pages.

- [ ] **Step 1: Add figure-content tests**

Parse `nativemem_overview.svg` and assert it has a `viewBox`, no external image references, and visible labels for:

```python
REQUIRED_FIGURE_LABELS = {
    "Agent Operations",
    "Memory Writer",
    "Memory Manager",
    "Query Navigator",
    "Multi-View Text Memory",
    "Source Memory",
    "Topical View",
    "Temporal View",
    "Recent Memory",
    "Core Memory",
    "Hyperlink Relations",
    "grep",
    "BM25",
    "Embedding",
    "Selected Evidence",
    "Evidence-Grounded Answer",
}
```

Assert Overview references `code/figures/nativemem_overview.svg` and Method references `../../code/figures/nativemem_overview.svg`.

- [ ] **Step 2: Run the figure tests and confirm they fail**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: failure because the SVG does not yet exist and both pages still reference the old PNG.

- [ ] **Step 3: Draw the SVG**

Create a `2400 × 1000` state-centered diagram:

- left: Writer, Manager, and Query Navigator;
- center: authoritative Source Memory with Topical, Temporal, Recent, and Core around it;
- center connections: source references and hyperlink relations;
- right: retrieval interfaces, selected evidence, and final answer;
- Manager annotation: incremental writing, retrieval-triggered local reorganization, and daily global management;
- Runtime annotation: paths, dates, links, source references, budgets, and commits;
- color roles: blue control/retrieval, orange validation/source, green memory state, purple LLM decisions/management, gray tools.

Keep labels readable when the image is rendered at `1080px` CSS width. Use SVG markers for arrows, rounded rectangles, and native text; do not embed a raster image.

- [ ] **Step 4: Export the PNG from the SVG**

Run Quick Look in an isolated temporary directory:

```bash
preview_dir="$(mktemp -d)"
qlmanage -t -s 2400 -o "$preview_dir" code/figures/nativemem_overview.svg
cp "$preview_dir/nativemem_overview.svg.png" code/figures/nativemem_overview.png
sips -g pixelWidth -g pixelHeight code/figures/nativemem_overview.png
```

Expected: a PNG with preserved wide aspect ratio and maximum dimension near `2400px`.

- [ ] **Step 5: Update the page references and run tests**

Point Overview and Method to the SVG and keep their captions in the shared figure style.

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
```

Expected: all document and figure assertions pass.

### Task 4: Integration and visual verification

**Files:**
- Verify all files listed above.

**Interfaces:**
- Consumes: Completed pages, shared stylesheet, SVG, and PNG.
- Produces: Verified local documents ready for user inspection.

- [ ] **Step 1: Run static checks**

Run:

```bash
cd code
pytest -q tests/test_document_pages.py
cd ..
git diff --check -- \
  Model-Aligned-Wiki.html \
  docs/survey.html \
  docs/method/index.html \
  docs/experiment-plan.html \
  docs/assets/document.css \
  code/figures/nativemem_overview.svg \
  code/tests/test_document_pages.py
```

Expected: tests pass and `git diff --check` prints nothing.

- [ ] **Step 2: Inspect the exact diff scope**

Run:

```bash
git status --short
git diff --stat -- \
  Model-Aligned-Wiki.html \
  docs/survey.html \
  docs/method/index.html \
  docs/experiment-plan.html \
  docs/assets/document.css \
  code/figures/nativemem_overview.svg \
  code/figures/nativemem_overview.png \
  code/tests/test_document_pages.py
```

Confirm no historical report, archive, prompt collection, benchmark fixture, or nested `paper/` file was modified.

- [ ] **Step 3: Verify desktop and mobile rendering**

Open Overview, Related Work, Method, and Experiments at `1440 × 1000` and `390 × 844`. Confirm:

- sidebar and content begin at identical positions on desktop;
- mobile TOC remains available without page-level horizontal overflow;
- titles, metadata, tables, callouts, and figures use the same visual rules;
- figure labels remain readable;
- Experiments tabs and relocated sections still work.

- [ ] **Step 4: Open the updated documents for the user**

Run:

```bash
open Model-Aligned-Wiki.html
open docs/method/index.html
```

Leave implementation changes uncommitted so the pre-existing document edits remain reviewable in the current working tree.
