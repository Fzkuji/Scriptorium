# LongMemEval memory-component ablation protocol

## Decision

Build and verify the complete memory once, freeze it, and run component
ablations as read-only retrieval views. This is the default because model-based
memory construction is the expensive stage while retrieval and answering can
reuse the same checkpointed workspace.

The complete stored workspace contains six independently selectable components:

- `topics`
- `timeline`
- `sources`
- `core`
- `recent`
- `relations`

The retrieval runner accepts a comma-separated `--memory-components` list. It
does not copy, delete, or rewrite the workspace. Every answer record stores the
effective component list, and the frozen source checkpoint remains the same.
Explicit component views disable the raw Bash tool so a masked file cannot be
reached by bypassing the file inventory. BM25 and embedding indexes are built
only from visible Topic and Source files.

## Two different ablation questions

### Retrieval-time availability ablation

Use one full build and mask components during query. This answers:

> Given the same constructed memory, how much does access to this component
> contribute at retrieval and answer time?

This is the low-cost, paired comparison. It is appropriate for the main
Source/Topic/Timeline/Core/Recent/Relations component ablation.

All paired conditions must use explicit component views. Do not compare an
explicit mask directly with legacy `--condition native`: legacy native mode
retains unrestricted read-only Bash behavior and does not expose
`relations.json` through the component inventory.

Recommended full-view control:

```text
--condition native \
--memory-components topics,timeline,sources,core,recent,relations
```

Examples of leave-one-component-out views:

```text
topics,timeline,sources,core,recent
topics,timeline,sources,core,relations
topics,timeline,sources,recent,relations
topics,timeline,core,recent,relations
topics,sources,core,recent,relations
timeline,sources,core,recent,relations
```

Each view must use a distinct output directory. The memory directory and build
checkpoint must remain byte-identical across views.

### Construction-time architecture ablation

Rebuild only when the research question removes or changes a component during
memory writing itself. This answers:

> How does omitting this component change what the writer constructs, the
> downstream derived views, build cost, and final accuracy?

This is not equivalent to retrieval masking. Topic memory is upstream of
derived Core, Timeline, Recent, and Relations views, so a build made without
Topics may change every downstream component. Construction ablations therefore
need separate output roots, checkpoints, cost accounting, and verification.

Do not run construction ablations until the retrieval-time matrix identifies
which components are worth the additional build cost.

## Resume and integrity rules

- Construction continues to checkpoint after every committed token batch.
- A stopped build resumes only from a committed checkpoint.
- Retrieval ablations never modify the frozen memory workspace.
- Every condition records its component list and source checkpoint.
- A condition must not reuse an output directory created with another mask.
- Source verification is automatically inactive when `sources` is masked; the
  prompt explicitly states that Source memory is unavailable.
- Existing LongMemEval builds remain valid and can be reused for these views.

## Recommended execution order

1. Complete the planned LongMemEval full-memory builds.
2. Run the explicit six-component control on the selected evaluation set.
3. Run paired leave-one-out retrieval masks on the same frozen builds.
4. Score every condition with the same answer and judge configuration.
5. Consider construction-time rebuilds only for components whose masked effect
   is material or whose build-time interaction is itself a research question.
