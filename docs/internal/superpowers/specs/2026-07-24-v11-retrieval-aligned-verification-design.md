# V11 Retrieval-Aligned Verification

## Scope

Add one verification pass after each session is written into a V11 memory
workspace. Verification must reuse the existing V11 workspace, source archive,
agent runner, shell access, memory-writing behavior, and automatic
topic/timeline/recent/source synchronization.

Verification is an agent stage invoked by Python. It is not a model tool.

## Execution order

For every session:

1. The writer agent archives the source turns and writes the session into
   memory.
2. A probe call selects one concrete fact from that session and produces:
   - a natural factual question;
   - the expected answer;
   - the supporting source references.
3. A retrieval agent receives the question and the memory workspace, but not
   the expected answer or intended location. It uses the same read behavior as
   normal V11 QA and records its navigation trace.
4. If the retrieved answer matches the expected answer, verification succeeds
   without changing memory.
5. Otherwise, a repair agent receives the question, expected fact, source
   references, retrieval trace, and memory workspace. It may inspect and
   reorganize the workspace and save the missing or corrected memory using the
   same capabilities as the writer.
6. The retrieval agent answers the same question again. The result and trace
   are recorded.
7. Processing continues with the next session.

The final manager pass remains unchanged.

## Agent boundaries

### Probe

The probe reads only the current source session. It chooses a concrete,
answerable fact. It does not choose a target memory path and does not inspect
benchmark questions or answers.

### Retrieval

The retrieval agent can inspect all generated memory views. It must not receive
the expected answer. Its tool trace is the evidence of where the model expected
the information to be stored.

### Repair

The repair agent has the same general workspace access as the writer. It may:

- add omitted memory;
- correct inaccurate memory;
- move content between files or headings;
- reorder sections or events;
- merge duplicates;
- reorganize files when the retrieval trace exposes a poor structure.

It must preserve valid historical states and source grounding. V11 code remains
responsible for rebuilding timeline and recent-event views and for maintaining
links.

## Data and audit records

Each session records:

- the probe question, expected answer, and source references;
- the first retrieval answer and tool trace;
- whether repair was invoked;
- the repair tool trace;
- the post-repair retrieval answer and tool trace;
- model-usage records under distinct verification phases.

The build must remain resumable. A failed provider request must fail the current
session visibly instead of silently marking verification as successful.

## Success criteria

- Verification runs once after every written session.
- The retrieval pass is independent of the expected answer.
- A failed probe can be repaired through the normal V11 agent workspace.
- Repair changes preserve V11 topic/timeline/recent/source consistency.
- A successful probe causes no memory mutation.
- Existing V11 writer and final manager behavior remains available.
- Tests cover the no-repair path, repair path, source isolation, per-session
  invocation order, synchronization after repair, and visible failure behavior.

## Non-goals

- No benchmark-specific prompts or question categories.
- No embedding retriever.
- No fixed topic taxonomy.
- No separate verification tool exposed to writer or manager agents.
- No full-library validation pass after every session.
