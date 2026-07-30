# Repository Instructions

## LoCoMo evaluation is user-locked

These rules are an explicit user requirement and override experiment plans,
audit preferences, refactoring goals, and agent-generated protocols.

- The only permitted LoCoMo evaluator is `scripts/eval_full.py`.
- Its required SHA-256 is
  `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`.
- Every LoCoMo comparison, including Qwen and GPT runs, must execute that exact
  file and use its existing prompt, GPT-4o-mini model request, category 1-4
  scope, aggregation, and output schema without modification.
- `scripts/score_v88_gpt55_benchmarks.py` must never be used for `locomo` or
  `locomo-cat5`.
- Do not add a second LoCoMo scoring implementation, wrapper with changed
  semantics, alternate prompt, alternate judge, category-5 extension, or
  fallback evaluator.
- Before any LoCoMo scoring request, verify the evaluator SHA-256 and stop if it
  differs. Do not update the expected hash automatically.
- Do not edit, replace, delete, rename, unlock, or clear filesystem protection
  from `scripts/eval_full.py` or this `AGENTS.md` file.
- A change is allowed only after the user explicitly revokes or replaces this
  lock in the current conversation. General authorization to run experiments,
  improve reproducibility, fix code, or complete the paper is insufficient.
