"""Unified evaluation package for NativeMem experiments.

Implements docs/unified_evaluation_protocol.md §6:
- answerer.py: unified short-answer generation (Mem0 original prompt +
  <answer></answer> extraction), same model for every system.
- judges.py: binary LLM judges — LoCoMo ACCURACY_PROMPT (Mem0/Nemori/
  LightMem identical copy) and LongMemEval official 5-template anscheck.
- metrics.py: set-based F1 (A-Mem, comparable-papers standard), LoCoMo
  official F1 (footnoted extra), BLEU-1..4, ROUGE-1/2/L, exact match.
- evaluate.py: CLI orchestrator with resumable per-question JSON output.

BEAM is deferred (protocol §6.1: optional extension; add eval_beam when the
dataset is pulled).
"""
