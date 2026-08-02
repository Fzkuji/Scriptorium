# Resumable Evaluation Design

## Scope

Improve the existing LongMemEval re-answer runner without introducing a new
orchestration system. The runner must tolerate transient provider and network
failures, preserve every completed answer, stop promptly on user interruption,
and continue from the same output directory on the next invocation.

## Behavior

- Retry connection errors, request timeouts, HTTP 429, and HTTP 5xx responses
  with bounded exponential backoff until the request succeeds or the process is
  interrupted.
- Do not retry permanent HTTP 4xx errors such as authentication or invalid
  request failures.
- Persist each completed item atomically before reporting it as complete.
- Reconstruct the aggregate `results.json` from valid per-item files.
- Treat valid existing per-item files as completed without requiring a resume
  flag. Reject conflicting or malformed existing records instead of silently
  skipping them.
- On `SIGINT` or `SIGTERM`, stop scheduling new items, cancel work that has not
  started, preserve completed items, and exit with an interrupted status.
- A later invocation with the same arguments and output directory skips valid
  completed items and processes the remainder.

## Implementation Boundary

Modify `scripts/reanswer_longmemeval_existing_memory.py` and its existing test
module only. Reuse its atomic JSON writer and per-item result layout. Extend the
shared V11 request retry helper only if the runner cannot express the required
retry policy without duplication.

## Verification

Automated tests cover transient failure recovery, permanent failure behavior,
existing-result validation, and interruption followed by continuation. The
existing retrieval-agent tests must remain green.
