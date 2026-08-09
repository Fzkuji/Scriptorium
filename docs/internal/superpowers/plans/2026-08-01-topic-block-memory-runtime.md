# Topic Block Memory Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make freeform Topic Markdown the single editable memory state while deterministically deriving Timeline, Recent, source links, relations, and retrieval indexes from validated paragraph blocks.

**Architecture:** Each authoritative Topic memory unit is one Markdown paragraph ending in one Obsidian-compatible block ID. Claim-adjacent footnotes carry normalized event time and stable Source links; ordinary Markdown links carry cross-memory relations. The LLM edits Topic Markdown, while the Runtime validates staged changes, resolves temporary IDs and source handles, rebuilds derived views, and atomically installs the result.

**Tech Stack:** Python standard library, existing V11 `MemoryWorkspace`, existing Markdown parser helpers, pytest, Git-backed workspace transactions.

## Global Constraints

- Source Memory is append-only and never rewritten by Topic maintenance.
- Topic Markdown is the only editable semantic memory state.
- Timeline, Recent, relation indexes, structure maps, BM25, and Embedding indexes are derived and rebuildable.
- A memory block is one coherent Markdown paragraph, not one sentence.
- A block ID contains only Latin letters, digits, and hyphens and appears once at the paragraph end.
- Evidence footnotes use `YYYY-MM-DD` or `undated`; only dated evidence produces Timeline entries.
- LLMs never edit Timeline, Recent, generated backlinks, or retrieval indexes directly.
- All mutations occur in the existing staged workspace and are installed only after validation succeeds.

### Task 1: Parse paragraph blocks and evidence annotations

**Files:**
- Modify: `code/src/nativemem_versions/v11/topic_markdown.py`
- Test: `code/tests/test_v11_topic_markdown.py`

**Interfaces:**
- Consumes: Markdown paragraphs ending in `^block-id` and footnote definitions containing a date and Source links.
- Produces: `MemoryUnit` paragraph records with stable `memory_id`, full `content`, ordered evidence annotations, flattened source references, and all semantic dates.

- [x] Add a failing parser test containing one paragraph, three claim-adjacent footnotes, one block ID, and an inline `#^target-id` relation.
- [x] Run `python -m pytest code/tests/test_v11_topic_markdown.py -q` and verify the new test fails because the current parser treats footnotes as separate memory units.
- [x] Add the smallest paragraph-block parser while retaining read compatibility for legacy `[^mem_*]` workspaces.
- [x] Add failures for duplicate block IDs, missing evidence definitions, invalid block IDs, and unbound factual paragraphs.
- [x] Run the focused parser tests until they pass.

### Task 2: Normalize staged Topic edits

**Files:**
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_memory.py`

**Interfaces:**
- Consumes: staged Topic edits containing existing IDs plus optional transaction-local `new-*` block and evidence IDs.
- Produces: canonical Topic Markdown with stable IDs, resolved Source links, and no temporary markers.

- [x] Add a failing test that edits only one phrase in a paragraph, preserves existing citations, adds one temporary evidence footnote, and leaves all unrelated footnotes byte-identical.
- [x] Add a failing test that splits a paragraph, preserves the old block ID on one output paragraph, and assigns a new ID to the other.
- [x] Add a failing test that rejects malformed or dangling block references and leaves the installed workspace unchanged.
- [x] Implement deterministic temporary-ID allocation, Source-handle resolution, and staged validation in the existing `MemoryWorkspace.shell` transaction.
- [x] Run the focused V11 memory tests until they pass.

### Task 3: Derive every non-Topic view

**Files:**
- Modify: `code/src/nativemem_versions/v11/derived_views.py`
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Test: `code/tests/test_v11_derived_views.py`
- Test: `code/tests/test_v11_memory.py`

**Interfaces:**
- Consumes: validated paragraph `MemoryUnit` objects and their evidence annotations.
- Produces: dated Timeline files, a 50-block Recent FIFO, relation/backlink data, and a compact structure map.

- [x] Add a failing test proving that two dated footnotes in one block create two Timeline entries pointing to the same Topic block.
- [x] Add a failing test proving that `undated` evidence produces no Timeline file or entry.
- [x] Add a failing test proving that Recent tracks changed block IDs without duplicating a block.
- [x] Add a failing test proving that outbound `#^id` links produce reverse backlinks and dangling targets fail validation.
- [x] Implement the minimal rebuild logic and run the focused derived-view tests.

### Task 4: Align the LLM editing contract

**Files:**
- Modify: `code/src/nativemem_versions/v11/memory.py`
- Modify: `code/src/nativemem_versions/v11/adapter.py`
- Test: `code/tests/test_v11_memory.py`

**Interfaces:**
- Consumes: Writer, local organizer, global manager, and repair prompts.
- Produces: one direct Topic-editing contract using the existing staged shell; legacy `save_memory` remains callable only for old tests/workspaces and is not advertised to the LLM.

- [x] Add failing prompt tests requiring paragraph block IDs, claim-adjacent evidence footnotes, explicit `undated`, immutable Source, and read-only derived views.
- [x] Change the exposed agent tools and prompts so Writer and Manager use the same Topic editing rules.
- [x] Record changed Topic paths and block counts from staged diffs so the adapter no longer depends on `save_memory` audit entries.
- [x] Run writer, manager, adapter, and verification tests.

### Task 5: Rewrite the method contract

**Files:**
- Modify: `docs/method/nativemem-method.html`
- Test: `code/tests/test_document_pages.py`

**Interfaces:**
- Consumes: the implemented Topic grammar and derived-view rules.
- Produces: one consistent method description covering authority, editing, validation, error recovery, and each derived view.

- [x] Replace the remaining structured-JSON Writer and separate Reconciler descriptions with the unified Markdown editing contract.
- [x] Document two linked Topic files, multi-evidence paragraphs, block-ID lifecycle, split/merge/delete behavior, and exact Runtime/LLM responsibilities.
- [x] Document Timeline generation from dated evidence footnotes, omission of `undated` evidence, Recent update rules, backlink generation, Source immutability, and index rebuilds.
- [x] Update implementation-status claims to distinguish shipped behavior from remaining deployment validation.
- [x] Run `python -m pytest code/tests/test_document_pages.py -q` and `git diff --check`.

### Task 6: Regression verification

**Files:**
- Test: `code/tests/test_v11_topic_markdown.py`
- Test: `code/tests/test_v11_derived_views.py`
- Test: `code/tests/test_v11_reconciliation.py`
- Test: `code/tests/test_v11_memory.py`
- Test: `code/tests/test_memory_bm25.py`
- Test: `code/tests/test_document_pages.py`

**Interfaces:**
- Consumes: all changes from Tasks 1-5.
- Produces: verified V11 behavior and a list of any intentionally retained legacy compatibility paths.

- [x] Run the focused test modules together.
- [x] Run the complete V11 test subset and fix only regressions caused by this change.
- [x] Inspect `git diff --check` and the final diff for unintended files.
- [x] Record test counts and any legacy compatibility boundary in the handoff.
