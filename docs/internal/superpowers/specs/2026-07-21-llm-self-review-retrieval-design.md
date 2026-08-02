# LLM Self-Review Retrieval Pilot Design

## Goal

Test whether generic prompt reflection and one same-model review turn improve
LongMemEval retrieval without embeddings, task-specific rules, or rebuilding
memory.

## Scope

- Reuse the frozen NativeMem libraries from the existing stratified-30 run.
- Evaluate nine questions previously attributed to missing or partial retrieval:
  `83, 109, 141, 153, 166, 299, 326, 389, 460`.
- Evaluate six previously correct controls:
  `7, 159, 193, 246, 373, 449`.
- Keep GPT-4o-mini, the file map, `inline=128`, `read_context=1`, and the
  existing read-only tools fixed.
- Score every new answer with the official LongMemEval judge prompt.

## Variants

`reflective-prompt` adds only generic instructions: form a provisional answer,
consider what missing or conflicting memory could change it, generate alternate
wordings when useful, and continue searching until the answer is sufficiently
supported.

`same-model-review` uses the same prompt. When the model first attempts a final
answer, the controller returns that draft to the same model and asks it to check
for missing or conflicting memory. The model may use the existing tools again or
return a revised answer.

Neither variant names benchmark question types or prescribes evidence counts,
search terms, files, or paths.

## Decision Rule

Compare both variants with the existing baseline on the same 15 questions.
Select a variant only if it recovers wrong questions without reducing official
judge accuracy on the six controls. Run the selected variant on all 30 existing
memories for confirmation.

## Safety and Cost

The pilot never invokes memory construction. Each variant writes to a separate
result directory and supports resume from per-item JSON files. Model requests
use the existing marked OpenRouter gateway and contain no persisted API secret.
