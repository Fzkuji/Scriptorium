# Memory architecture

Memory is three phases over one substrate. A fourth, answering, belongs to
whoever is holding the conversation — in the leaderboard the platform owns it,
so nothing here produces an answer.

```
                    ┌──────────────┐
   conversation ──▶ │   writing    │──┐
                    └──────────────┘  │
                    ┌──────────────┐  ├──▶  workspace  (files on disk)
        schedule ─▶ │  organizing  │──┤
                    └──────────────┘  │
                    ┌──────────────┐  │
          query ──▶ │  retrieval   │◀─┘
                    └──────────────┘
```

Each phase is a package. What they share — staging, the topic contract, the
write transaction — is the substrate, and lives in one place beneath them.

## Package layout

```
memory/
    config.py                MemoryConfig, shared by every phase

    workspace/               the substrate
        layout.py            where things live in a workspace
        staging.py           MemoryWorkspace: stage, commit, roll back
        transaction.py       the cross-process write lock and revisions
        normalization.py     deterministic ID assignment and contract checks
        block_views.py       block-level view synchronisation
        patching.py          restricted unified-diff application
        source_archive.py    append-only evidence
        event_writing.py     structured event insertion
        agent_pass.py        run one agent against the stage, commit per turn
        retrying.py          retry counts for intermittent model failures

    writing/                 phase 1 — conversation becomes memory
        subjects.py          subject + kind → topic path
        render.py            a fact → contract-conformant block and footnote
        tools.py             remember · update · forget
        session.py           write_sessions, the protocol the writer is given

    organizing/              phase 2 — memory stays usable
        tidy.py              deterministic: merge duplicates, prune, collapse
        tools.py             read_file · write_file · edit_file · shell
        reorganize.py        organize_topics, the model-driven pass
        verify.py            is every recorded fact still findable

    retrieval/               phase 3 — memory answers for itself
        index.py             BM25 and embedding indexes
        search.py            nearest(query) — deterministic, no model
        read.py              the model reads memory and reports what it found
        tools.py             the tools that reading pass is given
        context.py           what the reader starts with

    markdown/                the topic format itself
    runtime/                 cursors, thresholds, derived views
    agent_runtime/           the processes that run a model
    prompts/                 what each pass is told
```

## Call logic

### Writing — `POST /add`

```
service._ingest(payload)
    workspace.archive_sessions(session)         evidence first, append-only
    writing.write_session(workspace, agent, session, config)
        agent_pass.run(stage="write", tools=writing.tools(...))
            the model calls, once per fact:
                remember(subject, kind, fact, sources)
                update(subject, kind, replaces, fact, sources)
                forget(subject, kind, fact)
            each call:
                writing.subjects.path_for(subject, kind)
                writing.render.block(fact, sources, when)
                staging commit + contract validation
    organizing.tidy(workspace)                  deterministic, no model
```

The model decides one thing: which facts are worth recording, and which of the
three verbs each one is. Everything downstream of that — the file it lands in,
its block ID, its footnote, its links — is computed.

### Organizing — background, and after a run

```
organizing.reorganize(workspace, agent)
    tidy(workspace)                             deterministic first
    agent_pass.run(stage="organize", tools=organizing.tools(...))
    verify.reachable(workspace)                 facts search cannot reach
```

`tidy` runs first because there is no point asking a model to think about
duplicates that can be merged by comparing strings.

### Retrieval — `POST /search`

```
service._retrieve(payload)
    retrieval.search.nearest(workspace, query)  deterministic seed, no model
    retrieval.read(workspace, query, agent, seed)
        the model reads memory and reports what bears on the request
    rows = report ++ passages the model read ++ seed
```

The seed is run first and handed over, because searching for the words in the
query is not a judgement — it is what every retrieval does before it can think.

## What each phase may touch

| | `sources/` | `topics/` | derived views |
|---|---|---|---|
| writing | append | through its three verbs | rebuilt on commit |
| organizing | never | rearrange, never restate | rebuilt on commit |
| retrieval | read | read | read |

Retrieval never writes. Organizing moves prose but does not author it: a
paragraph it touched says what it said before. Only writing states new facts,
and only through `remember`, `update` and `forget`.

## The three write verbs

`remember` records something new. `update` records something that replaces an
existing fact, linking the new paragraph to the old one, which stays as the
record of what was true before. `forget` withdraws a fact recorded in error or
retracted by the person: its wording is erased so nothing retrieves it again,
and its block ID stays so links through it still resolve.

Block IDs are identity and the Runtime assigns them. No verb removes one.

## Core memory is consolidated, never truncated

`core.md` is what every session starts with, and it has a token budget. A model
authored it directly and nothing ever shortened it, so the budget was a wall:
once full, every write touching it was refused. A workspace running for a month
stopped being able to keep a new standing fact, and the failure was silent —
the person kept saying things and it kept declining to keep them.

Core is now a view. `remember` and `update` take `standing=True` for a fact the
person will want applied in every future conversation: a preference, a
constraint, an ongoing goal. That fact lives in its topic file like any other,
with its full wording, its footnote and its block ID. Core renders it.

Because core is a rendering and not the record, it can be rewritten freely.
Nothing in it is the only copy of anything. So when the standing set outgrows
the budget, core is **consolidated, not trimmed**:

1. Standing facts are grouped by subject.
2. Each group is rewritten into the fewest sentences that still carry every
   commitment in it. Three preferences about how code should be formatted,
   stated across three months, become one sentence that says all three.
3. If it still does not fit, the groups are merged the same way, across
   subjects.

Each round is a model pass over a small, bounded input — the standing set is a
handful of facts, not the workspace — and the result is checked back against
the facts it came from: a consolidation that lost a commitment is rejected and
the previous rendering stands.

Dropping is not a step. A budget is a compression target, and the answer to
"this does not fit" is to say it more briefly, not to decide which of the
person's standing instructions no longer counts.

There is a floor: enough distinct commitments cannot be compressed into any
fixed budget. At that point core carries the commitments it can state in full,
most recently reinforced first, and names the rest as subjects with a pointer
into their topic files. That is still not loss — every one of them is in
memory and retrievable, and the reader is told they exist rather than being
left to believe core is the whole of what is standing.

## Forgetting reaches the evidence

`forget` retracts a topic block. On its own that is not deletion: the message
the fact came from is still archived under `sources/`, still indexed, still
returned by a search. A person who asks to have something removed has not had
it removed.

So `forget` erases the source too — the message text is replaced with a
tombstone, keeping its anchor so every other footnote still resolves. The
archive stays append-only in shape: nothing is deleted, no ordering changes,
only content is erased.

A message can support several facts. Where another block still cites it, the
source is left alone and `forget` says so: the person asked to withdraw one
fact, not to blind every other record that rests on the same sentence.

## Two facts, one meaning

`tidy` merges paragraphs that are textually the same. Two records of one fact
in different words are not textually the same, and deciding whether they mean
the same thing is a judgement.

The two halves split it: `tidy` finds candidates — pairs close in wording but
not identical, within one subject — and `reorganize` puts that short list to
the model, which merges or leaves them. The model reads a handful of pairs
instead of every file, and never sees the pairs a string comparison already
settled.

## Files that outgrew their subject

A conversation about one deep subject — a codebase, a case, a project — puts
everything in one file. Splitting on "this file now covers two subjects" never
fires there, and the file grows until its own timeline is unreadable.

The rule is size, not subject count: past a threshold a topic file becomes a
directory of its own headings, `topics/projects/tracker.md` becoming
`topics/projects/tracker/{database,deployment,testing}.md`. Paragraphs move
with their block IDs, and links reach blocks by ID, so nothing that pointed
into the file stops resolving.

## Why the model is given so little

The mandated model for the leaderboard is gpt-4o-mini. Asked to hand-write the
topic format it fails the contract, cannot see why, retries the same rejected
edit until its turn budget is gone, and commits nothing — a whole session
recorded as three empty files. Every rule that can be computed is computed
here, so that the model's remaining job is the one thing no rule can do:
reading a conversation and judging what in it is worth keeping.
