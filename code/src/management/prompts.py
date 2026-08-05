"""Prompts and tool schemas for memory writing and maintenance."""

SYSTEM_PROMPT = """You manage a file-native memory workspace.

Never modify files under sources/. Topic Markdown is the editable semantic memory. Timeline, Recent, and Relations are derived by the Runtime; retrieval indexes and the runtime directory's metadata are also code-managed. Never edit these derived or operational files directly. Edit core.md only for stable information needed in every interaction.

Organize topics/ with directories, files, headings, and natural prose. One memory block is one coherent Markdown paragraph, which may contain several related facts. Every memory paragraph ends with exactly one Obsidian-compatible block ID. Preserve an existing ID when its paragraph keeps the same identity; use ^new-block-<label> for a new paragraph and let the Runtime assign the stable ID.

Every non-heading prose paragraph under topics/ must be a complete memory block. When splitting a paragraph, keep its existing block ID on the resulting paragraph that preserves its identity and give every additional paragraph a distinct ^new-block-<label>. When merging paragraphs, keep one existing block ID, retain all supported content and evidence that remains true, and update or remove links to eliminated IDs in the same edit.

Place an evidence footnote immediately after the fact it supports. Preserve existing footnotes for retained facts. For new evidence use [^new-evidence-<label>] and add a definition in this exact form:
[^new-evidence-<label>]: Time: `<time>`; Sources: provider/thread_id/message_id
Replace <time> with exactly one YYYY, YYYY-MM, YYYY-MM-DD, or undated value.
Use a distinct evidence label for each newly supported claim. Use the semantic event time stated or entailed by the evidence, not merely the write time or session observation date. Resolve explicit relative expressions with the Source observation date at the available precision: for example, "yesterday" becomes YYYY-MM-DD and "last year" becomes YYYY. The observation date must not be copied onto unrelated facts. Time belongs in the footnote metadata. Do not append the resolved time to the fact merely to mirror the Time field. Preserve a date in the prose only when it is naturally part of the fact. Use undated only when no calendar year can be determined. All resolved YYYY, YYYY-MM, and YYYY-MM-DD evidence is materialized in Timeline at its original precision; undated evidence is omitted. When the same fact has evidence at different semantic times, attach consecutive footnotes with one time value per footnote. Multiple complete source handles may follow Sources, separated by `, `. Do not invent source handles.

A valid new Topic paragraph and evidence definition have this form (replace every placeholder with current content):
<complete fact>.[^new-evidence-example] ^new-block-example

[^new-evidence-example]: Time: `<time>`; Sources: <complete-source-handle-from-input>

Use ordinary Markdown links to relate memory blocks, for example [current work](../career/employment.md#^existing-block-id). Every relative Markdown link from one Topic file to another Topic `.md` file must target `#^existing-block-id` or `#^new-block-<label>`; file-only and heading-only Topic links are invalid. The Runtime resolves source handles, temporary IDs, relative paths, and backlinks, then rebuilds Timeline, Recent, and Relations after each staged edit. Retrieval code derives BM25 and Embedding candidates from committed Topic blocks; do not edit retrieval caches.

Use the shell to inspect and edit Topic Markdown. Make the smallest relevant text change, keep unrelated prose and footnotes unchanged, and preserve complete historical state changes. The Runtime normalizes temporary IDs and source handles, validates every block, rewrites relative links after moves, rebuilds all derived views, and installs the transaction only if every check succeeds. Each shell edit is one independent transaction. A Runtime format error rejects every file and directory change made by that shell call."""

WRITER_TASK = """Integrate the following conversation session into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Use the shell to create or revise the appropriate Topic Markdown paragraphs and headings. Follow the Topic block and evidence-footnote contract in the system prompt.

The complete source conversation is already included below. Inspect whichever existing Topic, Core, or Source files are useful. Do not modify files under sources/.

Preserve complete historical state changes. Use the observation date only to resolve explicit relative dates in the source, not as the default date of every fact.

Use the shell to update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Observation date:
{observation_date}

Conversation:
{conversation}"""

WRITER_BATCH_TASK = """Integrate the following conversation sessions into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Use the shell to create or revise the appropriate Topic Markdown paragraphs and headings. Follow the Topic block and evidence-footnote contract in the system prompt.

The complete source conversations are already included below. Inspect whichever existing Topic, Core, or Source files are useful. Do not modify files under sources/. Integrate every supplied session before finishing.

Preserve complete historical state changes. Use each session's observation date only to resolve explicit relative dates in that source, not as the default date of every fact.

Use the shell to update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Sessions:
{sessions}"""

MANAGER_TASK = """Organize the topic files into a coherent structure.

Use the supplied workspace structure and shell. Split or merge existing topic files, headings, and paragraphs when appropriate. Preserve source-grounded facts, evidence footnotes, valid block links, and the complete dated history. For a split, retain the old ID on one resulting paragraph and assign distinct temporary IDs to the others. For a merge, retain one old ID and update all references to removed IDs in the same edit. Keep one block ID at the end of each resulting memory paragraph."""

LOCAL_MANAGER_TASK = """Organize only the following recently updated topic files and their local structure.

Limit this maintenance pass to these topic files. Merge redundant headings, split or combine local files when useful, and repair their local links. Do not reorganize unrelated topics. Preserve source-grounded facts, evidence footnotes, block IDs, and the complete dated history. Follow the same split and merge ID rules as the global manager.

Touched topic files:
{topic_paths}"""

VERIFICATION_PROBE_TASK = """Select one concrete factual detail from this session that should be recoverable from long-term memory.

Output only JSON with this shape:
{{"question":"a natural factual question","expected_answer":"the source-grounded answer","refs":["provider/thread_id/message_id"]}}

Observation date:
{observation_date}

Conversation:
{conversation}"""

VERIFICATION_RETRIEVAL_TASK = """Answer this question using the supplied memory workspace.

Inspect whichever memory views, files, and sections you consider appropriate. Do not assume where the answer should be stored.

Question: {question}

After inspection, output exactly one <answer>...</answer> block."""

VERIFICATION_REPAIR_TASK = """Repair the memory workspace so that the question can be answered through the memory organization.

Inspect the source records, the existing memory, and the retrieval trace. Make any changes you consider useful. You may add, revise, move, merge, reorder, or remove memory content while preserving valid historical information and source grounding.

Apply all changes only to editable Topic or Core memory. Never modify files under sources/.

Question: {question}
Expected source-grounded answer: {expected_answer}
Source references: {refs}
Previous retrieval answer: {retrieved_answer}
Previous retrieval trace:
{trace}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Read the memory workspace or edit authoritative topics/ and core.md. "
                "Source and derived views are read-only; Runtime validates and rebuilds them."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]
