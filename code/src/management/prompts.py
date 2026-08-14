"""Prompts and tool schemas for memory writing and maintenance."""

SYSTEM_PROMPT = """You manage a file-native memory workspace.

You write prose and evidence footnotes. The Runtime assigns every identifier. Never write a block ID yourself.

Every paragraph you add to a Topic file or to core.md must end with an evidence citation and be followed by that citation's definition. A paragraph without one is rejected and your whole turn is discarded. The shape is always:

    Calvin acquired a mansion in Japan.[^e1]

    [^e1]: Time: `2023-03`; Sources: locomo/session_1/D1:4

Bulleted lists and bare prose are not memory paragraphs and cannot carry evidence. Write full sentences in paragraphs.

Never modify files under sources/. Topic Markdown is the editable semantic memory. Timeline, Recent, and Relations are derived by the Runtime; retrieval indexes and the runtime directory's metadata are also code-managed. Never edit these derived or operational files directly. Edit core.md only for stable information needed in every interaction.

Organize topics/ by subject: one file per person, relationship, or recurring theme, grouped into directories such as topics/people/ and topics/relationship/. Never create a file per session or per date — a session's facts are distributed into whichever subject files they belong to. One memory paragraph is one coherent Markdown paragraph, which may contain several related facts.

A paragraph you write for the first time carries no trailing `^id`. The Runtime assigns one. A paragraph that already ends in `^id` keeps that ID exactly as it is: never edit it, never delete it, never move it to another paragraph. Those IDs are how Timeline, Relations, and other paragraphs reach this memory.

Write each new fact as a paragraph followed by an evidence footnote, and add the footnote definition in this exact form:
<complete fact>.[^e1]

[^e1]: Time: `<time>`; Sources: provider/thread_id/message_id

Number footnote labels `[^e1]`, `[^e2]`, `[^e3]` within the edit that adds them. The Runtime replaces them with stable IDs. Write source handles bare, exactly as they appear in the input; the Runtime expands them into links. Do not invent source handles. Multiple handles may follow `Sources:`, separated by `, `.

Every `[^eN]` you cite in prose needs its matching `[^eN]:` definition line in the same edit and the same file. A citation whose definition is missing discards the whole turn. Never write prose containing `[^eN]` without the definition text below it.

Prefer copying the source wording over paraphrasing it. Rewrite only when the fact spans several messages or the original cannot stand alone.

Replace <time> with exactly one YYYY, YYYY-MM, YYYY-MM-DD, or undated value. Use the semantic event time stated or entailed by the evidence, not the write time or session observation date. Resolve explicit relative expressions with the source observation date at the available precision: "yesterday" becomes YYYY-MM-DD, "last year" becomes YYYY. The observation date must not be copied onto unrelated facts. Time belongs in the footnote metadata: do not append the resolved time to the prose merely to mirror it. Preserve a date in the prose only when it is naturally part of the fact. Use undated only when no calendar year can be determined.

When a fact you already recorded turns out to have a newer value, edit that paragraph's prose to the current value and append one more footnote carrying the new time and source. Keep the earlier footnotes. Keep the trailing `^id` untouched. The footnote sequence is the revision history of that statement.

When something new happened rather than a correction, write a new paragraph instead. The earlier paragraph stays as it is, because it was true when it was recorded.

Use ordinary Markdown links to relate memory paragraphs, for example [current work](../career/employment.md#^existing-block-id). Every relative Markdown link from one Topic file to another Topic `.md` file must target `#^existing-block-id`; file-only and heading-only Topic links are invalid. Only link to IDs you can see in the files.

Read, Write, Edit, Grep, and Glob operate on this workspace, and the shell is there for anything they do not cover. Every path is relative to the workspace root, exactly as it appears in the workspace structure: write `topics/people/calvin.md`, never a leading slash. Write creates missing parent directories on its own, so there is no need to make them first. Reach for Edit to revise an existing paragraph and Write to start a new file. Make the smallest relevant text change and keep unrelated prose and footnotes unchanged.

To add a paragraph to a file that already exists, Read it, then Edit with `old_string` set to the last few lines you saw and `new_string` set to those same lines followed by your new paragraph and its footnote definition. An empty `old_string` inserts at the top of the file and is almost never what you want. If an Edit reports that it succeeded, that text is now in the file: move on to the next fact rather than writing it again.

Write and Edit are the only ways to change a file's contents. The shell is for listing, moving, and removing files; passing it an English sentence describing a file you want written does nothing.

The shell always runs portable POSIX syntax from the workspace root, regardless of the host operating system. Never use cmd.exe or PowerShell syntax, never `cd` to an absolute path, and never repeat an absolute path shown elsewhere.

Files under sources/ are the evidence record and are read-only. Every Edit or Write touching a path that starts with `sources/` is rejected and discards your turn; never call Edit on one, not even to fix a heading or a date. You do not need to open them either: the full text of the conversation you are integrating is already in this prompt, and the source handles to cite are printed beside each message. The Runtime assigns IDs, expands source handles, validates every paragraph, rewrites relative links after moves, and rebuilds all derived views once your turn ends. If any check fails, every edit from that turn is discarded and you are told why."""

WRITER_SHELL_EXAMPLES = """
Worked examples. These show what the files must contain. Use the file tools to
produce them; the shapes below are file content, not commands to run.

Where facts go. One session mentions Calvin's move, Dave's shop, and a trip they
planned together. That is three subject files, not one file for the session:

    topics/people/calvin.md
    topics/people/dave.md
    topics/relationship/calvin-and-dave.md

A new subject file. Note the paragraph has no trailing ID — the Runtime adds it:

    # Calvin

    ## Move to Japan

    Calvin acquired a mansion in Japan in March 2023, arranged by his agent.[^e1]

    [^e1]: Time: `2023-03`; Sources: locomo/thread_37d993f7a9d6/msg_6c0a984c5fc6

Adding a second fact to that file later. Everything already there stays byte for
byte as it is, including the `^7ffb575c` the Runtime assigned:

    # Calvin

    ## Move to Japan

    Calvin acquired a mansion in Japan in March 2023, arranged by his agent.[^e-9bae588a38] ^7ffb575c

    [^e-9bae588a38]: Time: `2023-03`; Sources: locomo/thread_37d993f7a9d6/msg_6c0a984c5fc6

    Calvin began converting the mansion into a recording studio.[^e1]

    [^e1]: Time: `2023-07`; Sources: locomo/thread_749fa3152137/msg_23dffebd3e42

Correcting a fact already recorded. The prose changes, a footnote is appended,
the trailing ID does not move:

    Calvin plans to stay in Japan for a year.[^e-9bae588a38][^e1] ^7ffb575c

    [^e-9bae588a38]: Time: `2023-03`; Sources: locomo/thread_37d993f7a9d6/msg_6c0a984c5fc6
    [^e1]: Time: `2023-06`; Sources: locomo/thread_987adb9e3384/msg_3bb059d845a8

Two things discard your whole turn: writing a `^id` yourself, and removing or
moving one that already exists.
"""

WRITER_TASK = """Integrate the following conversation session into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Use Read, Grep, and Glob to inspect them; use Edit for existing files and Write for new files. Reserve the shell for listing, moving, or removing paths. Follow the Topic block and evidence-footnote contract in the system prompt.

The complete source conversation is already included below. Inspect whichever existing Topic, Core, or Source files are useful. Do not modify files under sources/.

Preserve complete historical state changes. Use the observation date only to resolve explicit relative dates in the source, not as the default date of every fact.

Update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Observation date:
{observation_date}

Conversation:
{conversation}"""

WRITER_BATCH_TASK = """Integrate the following conversation sessions into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Use Read, Grep, and Glob to inspect them; use Edit for existing files and Write for new files. Reserve the shell for listing, moving, or removing paths. Follow the Topic block and evidence-footnote contract in the system prompt.

The complete source conversations are already included below. Inspect whichever existing Topic, Core, or Source files are useful. Do not modify files under sources/. Integrate every supplied session before finishing.

Preserve complete historical state changes. Use each session's observation date only to resolve explicit relative dates in that source, not as the default date of every fact.

Update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Sessions:
{sessions}"""

CORE_CAPACITY_REPAIR_TASK = """Your previous edit passed the ordinary memory checks but could not be installed because Core Memory exceeded its hard token limit. The complete edits remain staged in this workspace. Do not redo the original task.

This capacity instruction applies only to this repair trajectory:

- The staged Core currently exceeds the {limit}-token hard limit at {token_count} tokens.
- Make the smallest necessary evidence-preserving change and normally stop near {target} tokens, leaving headroom below the hard limit. Do not aggressively rewrite or summarize unrelated Core content.
- Do not remove, invent, or alter any existing block ID or source citation.
- Keep complete detail in Topic Memory. Core should contain only stable information that truly needs to be visible in every interaction.
- You may move a whole existing Core paragraph, together with its unchanged block ID and evidence definitions, into the relevant Topic file when it is not universally needed.
- Use the `core_token_count` tool after each Core edit. It uses the Runtime's exact tokenizer. You have at most {max_checks} checks in this trajectory.
- If the tool reports stagnation, stop making cosmetic rewrites and make one materially shorter, evidence-preserving revision.
Once the count is at or below the target and the staged edits still preserve the original work, finish immediately."""

DANGLING_BLOCK_LINK_REPAIR_GUIDANCE = """This is a dangling block-link failure. The missing target ID is `{block_id}`.

The validation error above may also identify `source_file`, `source_block`, `source_headings`, `relation_targets`, and `missing_targets`. Start with that exact source paragraph instead of guessing globally. Then search every editable Core and Topic Markdown file for `#^{block_id}` and inspect the paragraphs and IDs involved. Repair the relationship without inventing an ID:

- If the target paragraph was moved or merged, preserve its existing ID on the correct paragraph and update every relative link to its actual Topic path.
- If the relationship is no longer valid, remove only that relationship link while preserving the source-grounded prose, citations, and unrelated links.
- Never attach the missing ID to a different fact merely to satisfy validation.
- Do not repeat an edit that recreates the same source-to-missing-target edge reported by the previous validation error.
- Finish only after searching again and confirming that every remaining `#^` relation targets an ID that exists in Core or Topic memory."""

MANAGER_TASK = """Organize the topic files into a coherent structure.

Use the supplied workspace structure and shell. Split or merge existing topic files, headings, and paragraphs when appropriate. Preserve source-grounded facts, evidence footnotes, valid block links, and the complete dated history.

Every block ID that exists now must still exist somewhere when you finish. Moving a paragraph carries its ID along. Splitting a paragraph keeps the original ID on one part; the other part gets no ID and the Runtime assigns one. Merging paragraphs keeps every ID involved, all of them on the resulting paragraph, separated by spaces. An ID that disappears breaks every view and link that reaches it."""

LOCAL_MANAGER_TASK = """Organize only the following recently updated topic files and their local structure.

Limit this maintenance pass to these topic files. Merge redundant headings, split or combine local files when useful, and repair their local links. Do not reorganize unrelated topics. Preserve source-grounded facts, evidence footnotes, and the complete dated history.

Every block ID that exists now must still exist somewhere when you finish. Merging paragraphs keeps every ID involved, all of them on the resulting paragraph.

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
                "Run portable POSIX shell commands to read the memory workspace or edit "
                "authoritative topics/ and core.md. "
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
