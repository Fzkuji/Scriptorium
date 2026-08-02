# V11 Linked Memory Design

## Goal

V11 stores each extracted event once through the model and lets code maintain four synchronized views: immutable sources, semantic topics, a chronological timeline, and a bounded recent-event view. The model chooses event content and topic structure; code owns identifiers, links, validation, and commits.

## Storage

- `sources/D<N>.md` stores source turns with stable `dN-M` anchors. Code writes it before the writer runs.
- `topics/**/*.md` is the model-managed semantic long-term memory.
- `timeline/YYYY/MM/DD.md` is the code-generated chronological long-term memory.
- `recent_events.jsonl` contains the newest 100 extracted events by insertion order.

Every event has a stable `event_id`. Topic documents carry an immutable HTML comment marker for each event. Timeline records and recent-event rows carry the same ID and source references.

## Agent tools

The agent has three tools:

- `shell`: freely inspect and modify a temporary copy of `topics/`.
- `save_memory`: submit one or more events with `when`, `content`, `refs`, `topic_path`, and a one-to-six-level `headings` list. Code inserts marked event blocks into staged topics.
- `commit`: validate staged topics and atomically publish them. Code then regenerates managed links, timeline backlinks, and recent-event locations.

The model never writes `sources/`, `timeline/`, or `recent_events.jsonl` directly.

## Commit and reconciliation

At commit, code scans topic files, tracks the active Markdown heading stack, and builds `event_id -> topic path and headings`. It rejects missing, duplicate, or unknown IDs and invalid source references. It computes links relative to each file, renders topic links to timeline and sources, updates timeline links to the current topic anchor, and updates recent-event location fields.

All output is prepared in a temporary directory. The formal memory is replaced only after every event occurs exactly once in topics and timeline, every recent row resolves to the same topic location, and every generated link target exists. Failed validation leaves the formal memory unchanged and returns a precise tool error so the agent can correct its staged topics.

## Retrieval boundary

The answer agent may read topics, timeline, and the bounded `recent_events.jsonl`. It may resolve source turns only through refs found in those memory views. The full source archive is not searchable through general shell retrieval.

## Compatibility

Existing completed result directories are unchanged. Runs already executing retain their loaded implementation. The linked layout applies only to newly started V11 builds.
