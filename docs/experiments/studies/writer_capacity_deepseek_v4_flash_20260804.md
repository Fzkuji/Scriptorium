# Writer input capacity, deepseek-v4-flash, 2026-08-04

Artifact: `results/model_capacity/openrouter--deepseek-deepseek-v4-flash/capacity-40sess-20260804-r1/calibration.json`

Status: **inconsistent**. The reported `safe_input_tokens` of 32732 is not
usable and `WriterCapacity.load` refuses the artifact.

## Setup

| | |
|---|---|
| Provider | OpenRouter, `deepseek/deepseek-v4-flash` |
| Workload | 40 sessions × 20 messages, 54,472 tokens, 40 durable facts |
| Levels | 4096, 8192, 16384, 32768, 49152 |
| Trials | 3 probes per level, 15 total |
| `max_turns` | 500 |
| Cost / time | $4.30, 3.1 hours |

## Results

| Candidate | Batches | Pass | source_cov min / mean | fact precision | Round limit hit |
|---|---|---|---|---|---|
| 4096 | 18 | 3/3 | 1.000 / 1.000 | 1.0 | 0 |
| 8192 | 8 | 2/3 | 0.850 / 0.950 | 1.0 | 0 |
| 16384 | 4 | 2/3 | 0.725 / 0.908 | 1.0 | 0 |
| 32768 | 2 | 3/3 | 1.000 / 1.000 | 1.0 | 0 |
| 49152 | 2 | 2/3 | 0.150 / 0.717 | 1.0 | 0 |

No trial hit the turn limit and no trial fabricated a fact.

## Input size is not the limit

The model completed the task perfectly at 32768 and at 49152, the two largest
sizes tested. 49152 is 1.2× the largest LoCoMo conversation (40,713 tokens).
Failures did not concentrate at the top; the curve is U-shaped, with both ends
clean and the middle degraded. Capacity cannot improve as input grows, so the
shape means something other than input size decided these trials. That is what
`capacity_inversions` records, and why the artifact is marked inconsistent
rather than published as a capacity number.

The earlier figures of 4096, 12288, 24576, 27314 and 32768 came from runs with
a 20-turn ceiling. This workload needs 24 to 383 model calls depending on how
many batches the input is split into, so every small candidate was cut off long
before finishing and recorded as a capacity failure. Raising the ceiling to 500
removed that artifact entirely: `round_limit_trials` is 0 at every level here.

## Two distinct failure modes

Three trials failed, and they are not the same kind of failure.

**Content loss** — 8192/facts-c (0.850) and 16384/facts-b (0.725). Source and
fact coverage fall together, so whole records never reached memory. Both trials
used an unusually high number of calls per batch (29.9 and 45.0 against a
passing-trial mean of 22.8): the agent worked harder and still lost material.
These are the levels where the input is split into 4 to 8 batches, which is
enough that memory written earlier must be reconciled with each new batch, but
not so few that a batch carries most of the conversation.

**Reference loss** — 49152/facts-b. `fact_coverage` is 1.000 while
`source_coverage` is 0.150: every one of the 40 archive labels was written to
memory, but only 15% carried the source reference back to the message it came
from. The trial finished in 74 calls and 7.5 minutes, so the model considered
the task done. This is not a recall failure; it is the source-referencing part
of the Writer protocol being dropped while the content survives.

The second mode matters more than the first for this project, because
source-referenced memory is a core claim of the method. It says the constraint
degrades before the content does, and only at the largest input size.

## What this does not establish

Three probes per level is too few to estimate a failure rate. Each failure is a
single observation, and 8192, 16384 and 49152 each failed exactly once. The
distinction between "this level is unreliable" and "this workload is unreliable
at a rate independent of level" cannot be made from this data.

## Next steps

1. Re-run 8192 and 16384 with 8 or more probes to measure whether the failure
   rate there is genuinely higher than at 4096 and 32768.
2. Re-run 49152 to see whether reference loss reproduces, and inspect the
   written memory directly rather than only its coverage numbers.
3. LoCoMo cannot bound this model: its largest conversation is 40,713 tokens and
   the model handled 49,152 cleanly. BEAM is available locally with 100K
   (~136,000 tokens), 500K and 1M splits and is the dataset for probing further.
