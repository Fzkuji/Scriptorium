# M4 reliability protocol

This protocol is frozen before formal M1--M3 scores are inspected. All M4
programs are offline and make no model or network calls.

The exact comparison inventory, correction families, non-hypothesis rows, and
M5 interaction support rule are frozen in
`paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json`. Later pairing specs may
only bind completed artifact paths, SHA-256 values, metric fields, and exact
question sets to that inventory; they may not add or remove comparisons after
formal scores are available.

## Paired statistics

`scripts/freeze_m4_paired_inputs.py` first reads a frozen comparison spec. Each
method artifact is bound by SHA-256, exact complete-status metadata, expected
question count, a cluster field or LoCoMo question-ID cluster rule, and dotted
metric fields. `scripts/audit_m4_paired_inputs.py` reconstructs the exact
question pairing before statistics are permitted.

`scripts/run_m4_statistics.py` then accepts one frozen JSON object with
`schema_version=1`, `status="frozen"`, a stable `analysis_id`, and a non-empty
`comparisons` array. Each comparison declares `comparison_id`, `family`,
`kind` (`binary` or `continuous`), method labels, a metric label, and paired
records with exactly these statistical fields:

```json
{
  "question_id": "s0_q0",
  "cluster_id": "s0",
  "left": 1,
  "right": 0
}
```

The estimand is the question-weighted mean of `left-right`. The 95% interval
uses 10,000 deterministic bootstrap repetitions that resample conversations
and retain every question in each selected conversation. Binary comparisons
also use the two-sided exact McNemar test. McNemar p-values are Holm-adjusted
within each declared family. Continuous metrics and costs receive paired
clustered bootstrap intervals; this protocol does not add an unregistered
continuous-outcome hypothesis test.

`scripts/audit_m4_statistics.py` independently regenerates every replicate,
test, and correction from the frozen input.

## Failure attribution and qualitative cases

`scripts/run_m4_failure_analysis.py` requires explicit boolean stage evidence
and trace identifiers for every question. It does not infer a missing source
from an absent field. Questions with incomplete gold-source mapping are
excluded from source-stage attribution.

The earliest primary stage is assigned in this order:

1. extraction omission;
2. wrong path;
3. maintenance error;
4. navigation miss;
5. source-resolution error;
6. judge ambiguity when an incorrect primary judgment disagrees with the
   sensitivity judgment;
7. answer error.

Later observed stages and judge disagreement are retained as secondary tags.
Qualitative cases are the first two SHA-256-ordered cases within each primary
label under seed `m4-qualitative-v1`; outcome direction is not part of the
selection key. `scripts/audit_m4_failure_analysis.py` regenerates all labels
and selections.

## Human validation packet

`scripts/build_locomo_human_packet.py` requires a complete 1,540-question
LoCoMo category-1--4 artifact scored by the frozen primary judge. It selects
exactly ten questions from each conversation. Each present category receives
at least one position; remaining positions use proportional largest-remainder
allocation. Records within each conversation/category stratum and final packet
order use frozen SHA-256 ordering.

The public packet omits method identity, source record identifiers, automated
judge labels, conversation index, and category. The output separates
`public/` from `private/private_key.json`. Annotators receive only the public
packet and their own blank CSV. The two annotator identifiers must differ and
all 100 labels must be present before
`scripts/score_locomo_human_agreement.py` produces three-class agreement,
binary agreement excluding any uncertain pair, Cohen's kappa, and agreement
with the primary automated judge.

Automated execution cannot produce independent human labels. Until two label
files are returned, R502 remains `EXTERNAL-LABELS` and the paper reports no
human accuracy or human agreement number.
