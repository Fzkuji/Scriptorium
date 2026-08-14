# LongMemEval Experiment Findings

This log records runtime findings that may affect reliability, recovery, or
future tuning. Raw experiment artifacts remain the source of truth; this file
is an index of notable incidents and proposed follow-up work.

## Change and finding register

| Problem or requirement | Measure already adopted | Status | Follow-up recommendation |
| --- | --- | --- | --- |
| Core memory can exceed its hard token capacity after a writer edit. | Added a dedicated incremental core-capacity repair path. The initial writer trajectory may be followed by at most **two** repair trajectories (`core_repair_max_trajectories=2`), for a total of at most **three opportunities**. Each repair can perform up to eight capacity checks, targets 2,700 tokens for the 3k-core runs, and stops on repeated stagnation. Repair records are propagated into build trajectory metadata. | Implemented before/currently used by the formal runs. | Keep “two repairs / three total opportunities” explicit in reports. Review whether the target and stagnation threshold should vary by model after the run. |
| Claude Agent SDK could block indefinitely without emitting messages. | Added a configurable inactivity timeout, currently 1,800 seconds, around SDK message reads. | Implemented and tested. | Preserve the timeout in provider migration tests and distinguish inactivity from API/network failures in snapshots. |
| A single SDK JSON message exceeded the reader buffer. | Raised `max_buffer_size` from 1 MiB to 8 MiB (item 244 recovered), then to 20 MiB after item 487 exceeded 8 MiB. | Implemented; 20 MiB recovery pending. | Stop and inspect if 20 MiB is exceeded; do not increase it repeatedly without understanding message growth. |
| OpenCode Go runs needed provider-specific, resumable, one-item launches without exposing the API key. | Added `launch_longmemeval_opencode_item.sh`; it reads the key into the child environment, runs one item with `--limit 1 --resume`, and restricts accepted indices to the approved Mac ranges. | Implemented for the current run. | Add automated tests for key absence from argv/logs and rejection of disallowed index ranges. |
| Mac and WSL work could overlap after the experiment range expanded. | Split Mac configuration into 175–274 and 425–499; reserved 275–424 exclusively for WSL. Added five worker configs for the new 425–499 shard. | Implemented operationally. | Add a shared allocation manifest and make every launcher validate ownership before starting. |
| `launchctl submit` can reschedule a label after normal or failed exit. | Monitoring now removes the exact label immediately after `built`, or after a stopped item reaches an atomic paused checkpoint, before launching another item. | Operational mitigation implemented; launcher lifecycle is not yet self-contained. | Make terminal-state handling self-disabling and add lifecycle integration tests. |
| A writer edit and its generic repair both left a dangling block relation in item 440. | Integrity validation rejected and rolled back both invalid staged edits. The later whole-batch attempt committed successfully; the item was paused atomically at 2/9 and its label removed. | Data protected; reliability improvement deferred. | Give generic validation failures bounded retries and add a targeted dangling-link repair prompt plus structured terminal failure snapshot. |
| Formal experiments need recoverable partial progress. | Runs use per-item checkpoints, build checkpoints, writer live audit/progress, verification, atomic final commit, and `.stop-after-current-batch` sentinels. | Implemented and exercised during recovery. | Add a single machine-readable incident index that references these artifacts across workers. |

### Code and configuration references

- Core repair loop: `code/src/management/agent.py`.
- Core repair settings and validation: `code/src/management/config.py`.
- Core repair tool state/check limits: `code/src/management/tools.py`.
- Repair metadata propagation: `code/src/build.py`.
- SDK inactivity and buffer settings: `code/src/agent_runtime/claude_code.py`.
- SDK tests: `code/tests/agent_runtime/test_claude_code.py`.
- OpenCode one-item launcher:
  `code/scripts/runners/launch_longmemeval_opencode_item.sh`.
- Mac shard configurations:
  `code/scripts/configs/longmemeval_ms_100_index175_274_core3k_mac_worker*.json`
  and
  `code/scripts/configs/longmemeval_ms_75_index425_499_core3k_mac_worker*.json`.

## Evidence policy

- Preserve the original result directories, checkpoints, live audits, stdout,
  stderr, and failure snapshots.
- Do not edit or delete failed trajectories to make a run appear successful.
- Treat an item as complete only when its checkpoint is `status=built` and its
  verification, final management, and atomic commit have completed.
- Apply code changes only between controlled runs unless a change is required
  to recover a stopped item and is separately tested and authorized.

## Incident timeline policy

This document is also the chronological incident index for the formal run.
Monitoring must consider appending an entry whenever a lane is stopped,
paused, blocked, or recovered, or whenever a setting/code change is made in
response to an observed failure. Routine transient errors that recover without
intervention do not need an entry unless they recur or reveal a new root cause.

Each incident entry should preserve:

- the observation and decision times (ISO 8601 with timezone when available);
- worker, dataset index, provider, checkpoint position, and exact error;
- whether the event is network/account, transport, model output, validation,
  capacity, lifecycle, or inactivity related;
- the safety action, atomic checkpoint outcome, and launchctl label outcome;
- recovery authorization, configuration/code changes, and final result;
- links to the result directory, live audit, stderr/stdout, and failure
  snapshot when they exist.

Do not rewrite an earlier entry when the state changes. Append the recovery or
resolution as a later timeline event so the complete decision history remains
auditable.

## Operational incident timeline (2026-08-11 to 2026-08-12)

### 2026-08-11 16:00–17:39 HKT — item 473 dangling-link diagnosis and recovery

- Worker 3 item 473 exhausted the then-formal three generic repair
  trajectories on `dangling block link: 389239eb` and paused atomically at
  batch 2/9.
- A formal-directory-independent canary added source-file, source-block,
  heading, relation-target, and missing-target locality to the error. Real
  OpenCode Go repair succeeded on trajectory 1/3 and advanced the copied
  checkpoint from 2/9 to 3/9 with verification passing.
- The user authorized formal recovery. Item 473 subsequently completed 9/9,
  its exact label was removed, and worker 3 continued with item 474.
- Classification: validation/model-output reliability. Outcome: resolved;
  detailed repair feedback retained.

### 2026-08-11 20:15–20:37 HKT — item 460 Core capacity exhaustion

- Worker 2 item 460 reached batch 7 and reported
  `CoreCapacityError: Core Memory exceeds 3000 tokens: 3022` after the then
  configured three generic repair trajectories were exhausted.
- A stop-after-current-batch sentinel was created. During controlled recovery,
  the item ultimately completed 9/9 with final management and verification;
  the exact item 460 label was removed before worker 2 continued.
- This incident contributed to the later user-approved increase of formal
  `generic_repair_max_trajectories` from 3 to 10 for all active Mac workers.
- Classification: capacity/model-output repair. Outcome: recovered without
  increasing the 3,000-token Core hard limit.

### 2026-08-11 23:20–2026-08-12 04:47 HKT — item 492 certificate failures

- Worker 5 item 492 produced repeated
  `UNKNOWN_CERTIFICATE_VERIFICATION_ERROR` failures near batch 0 and entered a
  launchctl restart cycle. A sentinel was created and exact termination/removal
  was initially blocked by approval state.
- The item later reached a valid built 9/9 checkpoint. After explicit user
  approval to restore five lanes, the stale exact label was removed and item
  493 was started normally.
- Classification: provider/TLS transport plus launchctl lifecycle. Outcome:
  recovered; no certificate workaround or verification bypass was introduced.

### 2026-08-12 12:09–12:49 HKT — item 498 inactivity pause

- Worker 5 item 498 stopped advancing after batch 1/9, sessions 6/51, source
  turns 59/489. For more than 35 minutes the checkpoint, stdout, and stderr did
  not change, while launchctl still showed a live process and no explicit API
  or program error.
- Under the 30-minute inactivity rule, monitoring created
  `.stop-after-current-batch`. The sentinel took effect at the next atomic
  boundary: build checkpoint `paused`, batches 2/9, sessions 11/51, source
  turns 121/489. The KeepAlive label was then removed and confirmed absent.
- Item 499 was not started. Item 498 remains a recoverable pending incident
  awaiting an explicit resume decision.
- Classification: unexplained inactivity/lifecycle. Outcome: safely paused;
  no data loss and no repair-budget exhaustion.

### 2026-08-12 13:56 HKT — item 498 user-authorized resume

- The user explicitly authorized worker 5 recovery. Monitoring reconfirmed the
  atomic paused build checkpoint at batches 2/9, sessions 11/51, source turns
  121/489; the exact launchctl label was absent and the stop sentinel was still
  present.
- The sentinel was removed and item 498 was resumed with the normal strict
  one-item launcher. The new exact label
  `com.scriptorium.longmemeval.w5.i498` entered `running` with PID 12616; a new
  non-overwriting resume stdout/stderr pair was created.
- Item 499 remains gated on item 498 reaching a verified, atomically committed
  `built` checkpoint. Outcome: recovery in progress.

### 2026-08-12 15:06 HKT — item 498 recovery completed

- Resumed item 498 completed batches 9/9, sessions 51/51, source turns
  489/489 with final management complete and a `built` checkpoint. No generic
  or Core repair was required after recovery.
- The exact item 498 label was removed before worker 5 started item 499 with a
  new exact label and non-overwriting logs.
- Outcome: resolved. The earlier inactivity pause caused no data loss and did
  not recur during the resumed portion.

### 2026-08-12 14:29 HKT — item 62 inactivity stop initiated

- Worker 2 item 62 remained at batches 1/9, sessions 6/55, source turns
  52/526 for more than 30 minutes. Its exact launchctl service still reported
  `running` (PID 10313, no prior exit), while item stdout contained only the
  item header and stderr was empty.
- Monitoring created `.stop-after-current-batch` under the item directory.
  Item 63 is gated until item 62 either advances normally or reaches an atomic
  paused/failed/built checkpoint and its exact label is handled.
- Classification: unexplained inactivity/lifecycle, similar symptom class to
  item 498 but a separate worker. Outcome: safe stop in progress.

### 2026-08-12 14:31–14:36 HKT — item 62 atomically paused

- The item 62 sentinel took effect at the next atomic boundary. Its build
  checkpoint is now `paused` at batches 2/9, sessions 12/55, source turns
  114/526.
- The exact launchctl label was removed; because KeepAlive briefly recreated
  it, monitoring repeated the exact-label removal and then confirmed the
  service was absent. Item 63 remains unstarted.
- Outcome: safely paused and recoverable, awaiting an explicit resume decision.

### 2026-08-12 15:38 HKT — item 62 user-authorized recovery

- The user authorized direct bounded recovery for inactivity, timeout,
  disconnected-stream, and other transient network/lifecycle failures that do
  not require a structural code change. The item-local stop sentinel was
  removed only after confirming the old exact label was absent and the atomic
  checkpoint remained `paused` at batches 2/9, sessions 12/55, source turns
  114/526.
- Worker 2 item 62 was submitted again under the same exact launchctl label;
  the service entered `running` with PID 30237. Item 63 remains gated until
  item 62 reaches a verified `built` checkpoint.
- Recovery policy: future occurrences in this transient class may be resumed
  automatically from the atomic checkpoint with a bounded retry and an
  appended incident event. Structural code/configuration failures, account or
  quota errors, persistent rate limiting, buffer overflow, repair exhaustion,
  or integrity risk still require the existing safe-stop policy.
- Outcome: recovery started; checkpoint advancement remains under monitoring.

## 2026-08-11: dangling block link during item 440

### Observation

- Scope: Mac LongMemEval-S, dataset index 440, worker 1.
- Provider/model: OpenCode Go / `deepseek-v4-flash`.
- Failure: `memory writer repair was rejected: ValueError: dangling block
  link: 5d41b633`.
- The first generated edit left a relation pointing to block `^5d41b633` after
  the target block had been removed, moved, or merged.
- The generic repair trajectory received the validation error but still
  produced an invalid workspace, so the commit was rejected and rolled back.
- A subsequent whole-batch retry produced a valid commit. The item is safely
  paused at the atomic 2/9 checkpoint pending an explicit recovery decision.
- No invalid edit was committed to the result.

### Raw evidence

- Result item:
  `code/results/formal/longmemeval-ms-75-index425-499-core3k-mac-worker1-r1/items/0440_0ddfec37_abs/`
- Runner output:
  `code/results/formal/longmemeval-ms-75-index425-499-core3k-mac-worker1-r1/opencode-item0440-launchctl.stdout`
- Writer trajectories:
  `writer-live-batch-001-attempt-001.jsonl` and
  `writer-live-batch-001-attempt-002.jsonl` in the item directory.
- Relevant validation: `code/src/management/block_views.py`.
- Relevant generic repair path: `code/src/management/agent.py`.

### Assessment

The integrity guard and rollback worked as designed. The reliability gap is
that generic validation failures receive only one LLM repair trajectory, and
there is no deterministic repair for a dangling block relation. The failure
appears generation-dependent rather than a deterministic parser failure,
because a later attempt committed the same batch successfully.

### Candidate improvements

1. Add bounded generic validation repair attempts, initially capped at three.
2. Give dangling-link failures a targeted repair prompt containing the missing
   ID and the files/blocks that reference it.
3. Consider a deterministic pre-repair step that removes a relation only when
   its intended replacement cannot be resolved unambiguously; never guess a
   target silently.
4. Add tests covering a moved/deleted target, stale inbound link, rollback,
   successful second repair, and exhaustion of the repair limit.
5. Record structured repair attempts and terminal validation failures in a
   durable per-item failure snapshot, not only runner stdout.

## 2026-08-11: dangling block link repeated on item 486

### Observation

- Scope: Mac LongMemEval-S, dataset index 486, worker 5.
- Failure: `memory writer repair was rejected: ValueError: dangling block
  link: eb197ee0`.
- This is the second lane to produce the same class of program-level writer
  consistency failure after item 440 (`5d41b633`). It therefore triggered the
  multi-lane safety rule rather than being treated as an isolated generation.
- Stop-after-current-batch sentinels were created for active items 456, 471,
  426, and 486. No subsequent items may start until each current item reaches
  an atomic paused/built checkpoint and the exact launchctl labels are removed.

### Raw evidence

- Result item:
  `code/results/formal/longmemeval-ms-75-index425-499-core3k-mac-worker5-r1/items/0486_4388e9dd/`
- Runner output:
  `code/results/formal/longmemeval-ms-75-index425-499-core3k-mac-worker5-r1/opencode-item0486-launchctl.stdout`

### Assessment and recommendation

The recurrence makes bounded generic validation retries and a targeted
dangling-link repair path higher priority. Before resuming items 440 or 486,
reproduce the failure in a fixed workspace fixture, add repair/rollback tests,
and run a small recovery-only validation. Do not alter the active formal run's
repair semantics in place.

## 2026-08-11: Claude Agent SDK JSON message buffer

### Observation

- Item 244 exceeded the Claude Agent SDK default 1 MiB JSON message buffer at
  7/9 batches.
- The SDK `max_buffer_size` was explicitly raised to 8 MiB and the item later
  completed successfully from its atomic checkpoint.

### Follow-up

- Item 487 later exceeded 8 MiB. After preserving a 5/9 atomic checkpoint and
  confirming that the limit controls SDK transport rather than model context,
  the configured ceiling was raised to 20 MiB for one bounded recovery.
- Keep the 20 MiB limit covered by runtime tests.
- If a 20 MiB overflow occurs, stop the affected lane and investigate message
  growth; do not keep increasing the limit.

## 2026-08-11: launchctl rescheduled completed or failed jobs

### Observation

- Jobs created with `launchctl submit` remained registered after exit and
  could be scheduled again, including after a completed checkpoint or a
  nonzero item failure.
- This caused avoidable duplicate starts until monitoring removed the exact
  service label.

### Mitigation in the current run

- As soon as an item reaches `status=built`, remove its exact launchctl label
  before starting the next item.
- For a stopped or paused item, remove its exact label after the atomic
  checkpoint is confirmed.

### Candidate improvements

1. Make the launcher exit successfully without starting work when it sees a
   built checkpoint or an active stop sentinel.
2. Add a wrapper that removes or disables its own exact service label after a
   terminal state.
3. Add an integration test for built, paused, and failed exit states to ensure
   none can silently rerun.

## Deferred change plan

After the active LongMemEval run finishes:

1. Review all writer failures and group them by root cause.
2. Implement bounded generic repair retries and structured failure snapshots.
3. Add dangling-link and launchctl lifecycle tests.
4. Run a small fixed-index recovery test before resuming any failed formal
   item.
5. Record the code revision and test results in this document.

## 2026-08-11: bounded generic repair prototype and item 486 canary

### Implemented prototype

- Added `generic_repair_max_trajectories` to `MemoryConfig`; its default is 1,
  preserving the prior formal-run behavior.
- Generic validation repairs now run in a bounded loop and report their attempt
  number and configured limit.
- A dangling-link failure receives targeted guidance containing the missing
  block ID, instructions to locate every inbound relation, preserve an ID when
  a paragraph was moved/merged, and remove only an invalid relationship when
  no unambiguous target exists.
- The LongMemEval and conversation runners expose
  `--generic-repair-max-trajectories`.

### Tests

- Added tests for success on the third repair trajectory, exhaustion after
  three repair trajectories, and inclusion of the missing ID and safety rules
  in the targeted prompt.
- Targeted repair/core tests passed: `5 passed`.
- The broader management and SDK selection produced `66 passed, 4 failed`;
  the four failures are pre-existing macOS shell-test incompatibilities around
  native `sed -i`/transaction injection and are unrelated to this prototype.

### Live diagnostic canary

- Copied item 486's formal 5/9 checkpoint into
  `code/results/diagnostic/longmemeval-ms-item486-generic-repair10/`.
- The copied canary uses at most 10 generic repair trajectories. Ten is a
  diagnostic ceiling, not the proposed formal default.
- The formal item directory and checkpoint remain untouched and stopped.
- The canary resumes only the remaining item 486 batches. Because the original
  invalid staged edit was rolled back, this cannot deterministically replay
  that exact model output; it tests recovery behavior and captures any new
  validation failure with the expanded repair budget.

### Canary result and formal setting

- A second isolated canary injected one deterministic
  `dangling block link: diagnostic-missing-target` failure at commit time.
- The real OpenCode Go repair trajectory received the targeted guidance and
  succeeded on trajectory 1; the repaired batch passed validation and was
  atomically checkpointed. The attempt, mode, trigger, status, and error are
  stored in build trajectory metadata.
- Based on this result, the formal `425-499` worker configs explicitly use
  `generic_repair_max_trajectories: 3`. Ten remains diagnostic-only.
- Formal resume required both a config change and a code-fingerprint change.
  A default-deny `--allow-resume-drift` option was added: it works only with
  `--resume` and records the previous/replacement metadata plus changed fields
  in `manifest_migrations`. The option was used only for the first resumed
  invocation; subsequent items use normal strict matching.
- Targeted tests for repair behavior and audited manifest migration passed
  (`5 passed`).

### Item 473: insufficient dangling-link locality

- Formal item 473 exhausted all three generic repair trajectories on
  `dangling block link: 389239eb` and was safely paused at batch 2/9.
- The repair loop did return the preceding validation error to the model on
  every trajectory, but the old error exposed only the missing target ID. It
  did not identify the source file or source block that created the invalid
  edge, leaving the model to search the whole memory tree and making repeated
  but ineffective edits more likely.
- Dangling-link validation now preserves its original machine-detectable
  prefix while also reporting `source_file`, `source_block`,
  `source_headings`, the full `relation_targets`, and all `missing_targets`.
  The dedicated repair prompt tells the model to begin at that exact source
  paragraph and not recreate the same rejected edge.
- Generic repair lifecycle records are now durably appended to the live JSONL
  as each attempt starts and finishes, including terminal failures. Writer
  failure snapshots use a batch-and-attempt filename so retries cannot replace
  an earlier failure record.
- Targeted validation, prompt, repair-limit, repair-recording, and writer
  checkpoint tests passed (`21 passed`). A broader management run still has
  four pre-existing macOS shell/transaction-test failures unrelated to this
  change.
- A formal-directory-independent canary was cloned from item 473's 2/9 atomic
  checkpoint. It injects one real staged dangling link, invokes the actual
  validator, and lets OpenCode Go repair it with the formal three-trajectory
  ceiling. The formal item remains untouched while this canary runs.

### Operational incident timeline

- 2026-08-12 21:00 HKT — w3/item40 failed before completing batch 0
  (checkpoint `failed`, 0/9 batches) with raw error `API Error: Unable to
  connect to API (ConnectionRefused)` in
  `items/0040_15745da0/writer-failure-batch-000-attempt-001.json`. Classified
  as an isolated provider connection failure. The stale/unregistered launch
  state was checked and one bounded strict resume was submitted; the label
  exited immediately and the atomic checkpoint remained failed. No item41 or
  duplicate runner was started. A later retry requires a healthy endpoint and
  the normal bounded network-recovery rule.
- 2026-08-13 07:35 HKT — w1/item43 and w2/item44 were each building at the
  2/9 atomic checkpoint when the OpenCode account's raw provider response
  established a low-balance concurrency ceiling of three requests; the live
  workload had four memory builders plus one production-QA request source.
  Classified as persistent account concurrency throttling (`HTTP 429`, limit
  `3`). With explicit user authorization to stop w1-w2, item-local
  `.stop-after-current-batch` sentinels were created in both item directories.
  This requests a recoverable pause only after the active batch commits; no
  later queue item was started and the exact labels are retained until each
  checkpoint reaches an atomic terminal state. Evidence:
  `code/results/formal/longmemeval-ms-10-index35-44-core3k-mac-worker1-r1/items/0043_ccb36322/`
  and
  `code/results/formal/longmemeval-ms-10-index35-44-core3k-mac-worker2-r1/items/0044_001be529/`.
- 2026-08-13 07:54 HKT — after the user explicitly authorized sending
  item258 to `https://opencode.ai/zen/go`, w4/item258 was submitted as the
  third request source. Local manifest preflight exited before provider work
  with raw error `ExistingStateError: existing output uses different config,
  code`. Classified as strict resume-drift protection rather than a provider
  or data error. The exact launchctl label
  `com.scriptorium.longmemeval.formal.macw4.i258` was removed and verified
  absent; no retry or later item was started. Evidence:
  `code/results/formal/longmemeval-ms-100-index175-274-core3k-mac-worker4-r1/launch-012.stderr`.
  Recovery requires an explicit audited decision to permit
  `--allow-resume-drift`; the endpoint authorization alone was not treated as
  authorization to bypass fingerprint protection.
- 2026-08-13 08:02 HKT — the user explicitly approved audited resume drift
  for w4/item258. The exact label
  `com.scriptorium.longmemeval.formal.macw4.i258` was submitted with
  `--allow-resume-drift`; it is running from the 0/9 checkpoint and provider
  work has begun. The manifest atomically records a migration changing
  `config` and `code`, including complete previous and replacement metadata.
  Classified as an authorized audited recovery. Evidence:
  `code/results/formal/longmemeval-ms-100-index175-274-core3k-mac-worker4-r1/run_manifest.json`,
  `items/0258_4dfccbf7/build-checkpoint.json`, and `launch-013.stderr`.
- 2026-08-13 08:37 HKT — the user explicitly authorized only items 351, 369,
  395, and 424 from the otherwise prohibited 275-424 range for OpenCode. New
  item-scoped configs and an exact launcher allowlist were added; all other
  indices in the range remain rejected. The first ordered pair failed during
  local batch planning before provider work: w1/item351 raw error
  `Encountered text corresponding to disallowed special token
  '<|endoftext|>'`; w2/item369 raw error `one complete message exceeds the
  calibrated Writer input limit`. Classified respectively as tokenizer input
  incompatibility and oversized indivisible source message, not API/key
  failures. Both exact launchctl labels were removed and verified absent; no
  item395 or item424 runner was started, preserving queue order. Evidence:
  `code/results/formal/longmemeval-ms-4-index351-369-395-424-core3k-mac-worker1-r1/launch-001.stdout`
  and the corresponding `mac-worker2-r1/launch-001.stdout`.
- 2026-08-13 09:31 HKT — with explicit user authorization to run the four
  supplemental items concurrently, the two local planning failures were
  resolved with narrowly scoped audited changes. Token counting now treats
  tokenizer sentinel spellings in visible benchmark text as ordinary text
  (`disallowed_special=()`), without changing the source or transmitted
  content; a targeted regression test passed. Item369's single-message Writer
  input was measured at 18,343 tokens versus the formal 15,000 cap (3,343,
  22.3% over), so only its item-scoped config uses a 19,000-token cap. Items
  351 and 369 were resumed with audited config/code drift, and items395 and
  424 were newly submitted. All four exact labels are running; their initial
  atomic checkpoints are respectively 0/9, 0/7, 0/9, and 0/9 with no current
  error. Evidence: the four
  `code/results/formal/longmemeval-ms-4-index351-369-395-424-core3k-mac-worker*-r1/`
  directories and `code/tests/runtime/test_writer_capacity.py` (`4 passed`
  targeted run).
- 2026-08-13 11:03 HKT — user-authorized completion recovery resumed
  w1/item351 from its atomic 9/9, 49/49-session, 505/505-turn checkpoint after
  the earlier safety pause (`stop-after-current-batch`; raw provider error:
  no explicit error). The runner completed final management and atomically
  committed `status=built`, bringing the selected inventory to 215/215 built.
  The final five missing production answers (items 258, 351, 369, 395, 424)
  were frozen explicitly and generated with DeepSeek V4 Flash. The alternate
  OpenCode key returned the raw error `401 Invalid API key` for items258 and
  351; the run was safely interrupted at 0/5, then retried with the previously
  verified old key and completed 5/5. GPT-4o-mini via OpenRouter judged the
  five additions 4/5; combining them without rejudging the prior 210 produced
  215 unique, nonempty answers and 215 scores, overall accuracy 0.91162790698.
  All exact LongMemEval launchctl labels are absent. Evidence:
  `code/results/formal/longmemeval-ms-4-index351-369-395-424-core3k-mac-worker1-r1/items/0351_gpt4_78cf46a3/checkpoint.json`,
  `code/results/formal/longmemeval-production-qa-deepseek-opencode-final5-r1/answers/results.json`,
  `code/results/formal/longmemeval-judge-gpt4o-mini-openrouter-final5-r1/evaluation.json`,
  and `code/results/formal/longmemeval-judge-gpt4o-mini-openrouter-215-r1/evaluation.json`.
- 2026-08-13 12:48 HKT — the user explicitly authorized production QA and
  GPT-4o-mini judging for exactly 37 imported built memories: indices142-174,
  302, 303, 334, and 362. The downloaded handoff archive contained 46 built
  memories, but indices133-141 were excluded. Initial local validation failed
  all 37 before provider work with raw error `ValueError: invalid source
  memory` because portable checkpoints stored `paths.memory_dir` relative to
  the package while the current runner resolves it from the repository root.
  Only the extracted checkpoint copies were updated to absolute paths; memory
  contents were unchanged. The resumed single-worker DeepSeek V4 Flash run
  completed 37/37 nonempty, unique answers with source verification, and
  OpenRouter GPT-4o-mini judged 32/37 correct (overall 0.86486486486). Evidence:
  `code/results/formal/longmemeval-imported-qa-deepseek-opencode-index142-174-302-303-334-362-r1/answers/results.json`
  and
  `code/results/formal/longmemeval-imported-judge-gpt4o-mini-openrouter-index142-174-302-303-334-362-r1/evaluation.json`.
- 2026-08-13 HKT — the user explicitly authorized supplemental LongMemEval-S
  memory construction for dataset index 394, otherwise inside the prohibited
  275-424 interval. A narrowly scoped item-only configuration and launcher
  allowlist entry were added for w5, using the formal OpenCode Go endpoint,
  DeepSeek V4 Flash, 3k core, 20 MiB SDK buffering inherited from the runner,
  and at most 10 generic repair trajectories. All other previously prohibited
  indices remain rejected. Classification: authorized scope/configuration
  change. Planned evidence directory:
  `code/results/formal/longmemeval-ms-1-index394-core3k-mac-worker5-r1/`.
- 2026-08-13 13:29 HKT — the user reported that item394 was already being
  built elsewhere and revoked the local supplemental-build request. The local
  w5/item394 run was therefore stopped. A stop-after-current-batch sentinel
  was written while batch 1 was already active; the last durable atomic state
  remained 1/9 batches and 5 sessions. The next in-flight batch showed no new
  activity after `2026-08-13T05:28:47.459589+00:00`, so the exact launchctl
  label `com.scriptorium.longmemeval.formal.macw5.i394` was removed and
  verified absent. Raw provider error: no explicit error. Classification:
  user-requested cancellation / duplicate external work. The item-only
  launcher allowlist and config were removed; no automatic resume is allowed.
  The partial atomic evidence was retained without deletion at
  `code/results/formal/longmemeval-ms-1-index394-core3k-mac-worker5-r1/`.
