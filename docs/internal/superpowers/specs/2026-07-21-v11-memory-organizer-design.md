# NativeMem V11 Memory Organizer Design

## Purpose

V11 tests whether `gpt-4o-mini` can reorganize one existing LongMemEval
memory library more effectively when it is asked to inspect and organize the
library directly, without numeric rules for topic count, file size, minimum
entries, or merge thresholds.

The pilot uses a copy of LongMemEval item 83. The original v8.8 memory and all
existing evaluation artifacts remain unchanged.

## Organizer behavior

The organizer receives the copied memory directory and a short instruction:
inspect the memory files, then organize related information into coherent
documents. It decides the directory hierarchy, canonical document names,
merges, renames, and entry placement from the content it reads.

The instruction does not prescribe a taxonomy, target file count, minimum
document length, or threshold that triggers organization. It explicitly asks
the model to:

- combine documents that cover the same subject;
- move individual entries that are stored under an unrelated subject;
- retain distinct dates and state changes rather than collapsing them;
- preserve every source reference such as `[D9:11]`;
- avoid changing the factual text when moving entries;
- remove files left empty by its edits.

The organizer uses the existing read-only shell inspection operations and a
small set of constrained mutation operations. The model chooses the semantic
changes; code validates paths and performs the requested file edits.

## Safety and provenance

The pilot operates only on a copied directory. Before organization, it records
the files, non-heading memory lines, and all dialogue-reference identifiers.
After organization, validation rejects the result if:

- a pre-existing non-heading memory line is missing or modified;
- a dialogue-reference identifier is missing;
- a file path escapes the copied memory directory;
- duplicate copies of a moved line remain;
- an empty Markdown document remains.

The `timeline/` view remains unchanged during the first pilot. The organizer
edits `topics/` only. This isolates semantic topic organization from the
deterministic chronological index and avoids changing two representations at
once.

## Pilot evaluation

The pilot reports, before and after:

- Markdown file count;
- distribution of non-heading entries per topic file;
- empty and single-entry file counts;
- duplicated memory-line count;
- dialogue-reference preservation;
- the resulting directory tree.

It then reruns retrieval and answering for item 83 against the copied,
organized memory with the same `gpt-4o-mini` LongMemEval retrieval settings.
The pilot is successful only if structural validation passes. Answer accuracy
is reported as evidence from one item, not as a benchmark-level improvement.

## Scope

This pilot does not change the default v8.8 or V10 builder, does not rebuild
memory, and does not launch LongMemEval full evaluation. Integrating the
organizer into the V11 build loop is a separate implementation step after the
copied-memory pilot is inspected.
