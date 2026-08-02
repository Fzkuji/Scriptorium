# LLM Memory Systems — Experiment Settings Survey

Compiled: 2026-07-01. Covers experiment configurations for all major memory systems evaluated on LoCoMo, LongMemEval, and BEAM.

---

## 1. LoCoMo (ACL 2024, Snap Research)

**Benchmark**: 10 multi-session conversations, 1,986 QA pairs.
- Category 1: Multi-hop (282)
- Category 2: Temporal (321)
- Category 3: Open-domain (96)
- Category 4: Single-hop (841)
- Category 5: Adversarial (446)

**Official eval code**: `locomo/task_eval/evaluation.py`
- F1: token-level, with stemming (PorterStemmer), after normalization (remove articles, punctuation, lowercase)
- Multi-hop (cat 1): splits by comma, partial F1 per sub-answer
- Adversarial (cat 5): keyword matching — contains "no information available" or "not mentioned" → 1, else 0
- Also implements EM, BERTScore, ROUGE-L (secondary)

**Human performance**: F1 = 87.9

---

## 2. Mem0 (ECAI 2025)

### Build
- **Model**: GPT-4o-mini (original), GPT-4o (ablation)
- **Storage**: Vector DB (Qdrant), key-value memory entries
- **Granularity**: Per-message ingestion (CHUNK_SIZE=1)
- **Extraction**: LLM extracts structured memory entries from each message

### Answer
- **Model**: Same as build model (GPT-4o-mini or GPT-4o)
- **Prompt**: 7-step reasoning chain (scan all memories → entity verification → combine → select best → temporal grounding → inclusion check → commit). Very detailed, ~100 lines.
- **Retrieval**: Vector similarity search, top-k results sorted chronologically

### Evaluation
- **Judge model**: GPT-5 (in mem0-benchmarks code, default)
- **Judge format**: 0/1 binary (CORRECT/WRONG), returns JSON `{"reasoning": "...", "label": "CORRECT"}`
- **Judge rules**: 7 detailed rules — partial credit (1/N items correct = CORRECT), paraphrase acceptance, extra detail is fine, **14-day date tolerance**, semantic overlap, same referent, focus on knowledge not wording
- **Adversarial**: **Excluded** (`CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]`)
- **Benchmarks**: LoCoMo (10 samples), LongMemEval, BEAM
- **Metrics reported**: LLM Judge (primary), F1, BLEU-1
- **Open source**: Yes (github.com/mem0ai/mem0, github.com/mem0ai/memory-benchmarks)

### Reported Results (LoCoMo)
| Model | Judge | F1 | BLEU-1 |
|---|---|---|---|
| GPT-4o-mini | 66.9% | 38.72 | 27.13 |
| GPT-4o | — | 36.09 | — |
| 2026 new algo | 92.5% | — | — |

---

## 3. A-Mem (arXiv 2025)

### Build
- **Model**: GPT-4o-mini (default), GPT-4o, Qwen2.5-1.5B/3B (ablation)
- **Storage**: Vector DB + agentic note system (each memory has note, links, evolution chain)
- **Granularity**: Per-message, with memory evolution (each new memory triggers link generation + potential memory updates)
- **Embedding**: all-MiniLM-L6-v2 (SentenceTransformer)

### Answer
- **Model**: Same as build model
- **Prompt**: Plain text answer generation from retrieved memories
- **Retrieval**: Embedding similarity + keyword matching

### Evaluation
- **Judge**: No LLM judge — uses **12 automatic metrics** only
- **Metrics**: F1, BLEU-1/2/3/4, ROUGE-1/2/L, BERTScore (P/R/F1), METEOR, SBERT similarity
- **Adversarial**: Same as LoCoMo official (keyword matching)
- **Benchmarks**: LoCoMo (10 samples)
- **Open source**: Yes (github.com/WujiangXu/A-mem)

### Reported Results (LoCoMo F1)
| Model | Overall | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial |
|---|---|---|---|---|---|---|
| GPT-4o-mini | ~35 | 27.02 | 45.85 | 12.14 | 44.65 | 50.03 |
| GPT-4o | ~35 | 32.86 | 39.41 | 17.10 | 48.43 | 36.35 |

---

## 4. Mnemis (ACL 2026, Microsoft)

### Build
- **Model**: GPT-4.1-mini (memory construction)
- **Storage**: Hierarchical graph (entity-based episodes + category nodes)
- **Embedding**: Qwen3-Embedding-0.6B (dim=128)
- **Re-ranker**: Qwen3-Reranker-8B

### Answer
- **Model**: GPT-4.1-mini
- **Retrieval**: Dual-route — embedding search + graph traversal, top-k=10 episodes, top-2k=20 entities

### Evaluation
- **Judge**: Not explicitly stated (likely GPT-4.1-mini)
- **Judge format**: LLM-as-Judge (0/1 binary)
- **Adversarial**: **Excluded** from reported results
- **Benchmarks**: LoCoMo (10 samples), LongMemEval
- **Metrics**: LLM Judge (primary), F1, BLEU-1
- **Open source**: Yes (github.com/microsoft/Mnemis)

### Reported Results
| Benchmark | Judge | F1 | BLEU-1 |
|---|---|---|---|
| LoCoMo | 93.9% | — | — |
| LongMemEval | 91.6% | — | — |

---

## 5. MemMachine (2025/2026)

### Build
- **Model**: GPT-4.1-mini
- **Storage**: Raw conversation chunks (preserves ground truth)
- **Retrieval**: Memory mode (direct retrieval) + Agent mode (multi-step reasoning)

### Evaluation
- **Judge format**: LLM-as-Judge (0/1)
- **Adversarial**: **Not reported**
- **Metrics**: LLM Judge, F1, BLEU-1

### Reported Results (LoCoMo)
| Mode | Judge | F1 | BLEU-1 |
|---|---|---|---|
| Agent mode | 91.7% | 27.85 | 17.32 |
| Memory mode | 91.2% | — | — |

Note: Judge=91% but F1=28% — answers are correct but verbose, penalizing F1 precision.

---

## 6. Memory-R1 (arXiv 2026)

### Build
- **Model**: LLaMA-3.1-8B-Instruct or Qwen2.5-7B-Instruct (RL fine-tuned)
- **Storage**: Memory bank with structured operations
- **Training**: GRPO/PPO with EM reward signal, trained on 152 QA pairs from LoCoMo

### Answer
- **Model**: Same RL-tuned model (Answer Agent with Memory Distillation policy)
- **Retrieval**: RAG-based

### Evaluation
- **Judge**: LLM-as-Judge
- **Metrics**: F1, BLEU-1, LLM Judge (all three reported)
- **Data split**: 152 train, 81 val, 1,307 test
- **Open source**: Yes (github.com/yansikuan/memory-r1)

### Reported Results (LoCoMo)
| Model | Algo | Judge | F1 |
|---|---|---|---|
| Qwen2.5-7B | GRPO | 62.7% | 45.0 |
| LLaMA-3.1-8B | PPO | 57.5% | 41.1 |

---

## 7. Synthius-Mem (arXiv 2026)

### Build
- **Storage**: Structured persona knowledge extraction (not raw conversation)
- **Key feature**: Hallucination-resistant extraction architecture

### Evaluation
- **Adversarial**: **Included and separately reported** (99.6% robustness)
- **Benchmarks**: LoCoMo (1,813 questions)
- **Judge**: LLM-as-Judge

### Reported Results (LoCoMo)
- Overall: 94.4% (including adversarial)
- Adversarial: 99.6%
- Only system reporting near-perfect adversarial performance

---

## 8. ByteRover (2025)

### Build
- **Model**: Gemini 3 Flash (context tree construction) + Gemini 3 Pro (justification)
- **Storage**: Hierarchical Context Tree (LLM-curated)

### Evaluation
- **Judge**: Gemini 3 Flash
- **Judge prompt**: Uses **Hindsight's evaluation prompt** (adopted without modification for apples-to-apples comparison)
- **Adversarial**: Excluded from main score
- **Benchmarks**: LoCoMo, LongMemEval

### Reported Results
| Benchmark | Config | Judge |
|---|---|---|
| LoCoMo | Flash only | 90.9% |
| LoCoMo | Flash+Pro | 92.2%, later 96.1% |
| LongMemEval | — | 92.8% |

---

## 9. TiMem (arXiv 2026)

### Build
- **Model**: GPT-4o-mini
- **Storage**: Temporal hierarchical structure (time-indexed memory)
- **Key feature**: 52% memory reduction through temporal compression

### Evaluation
- **Judge**: LLM-as-Judge
- **Benchmarks**: LoCoMo

### Reported Results
- LoCoMo Judge: 75.3%

---

## 10. HORMA (arXiv 2026)

### Build
- **Model**: Claude Sonnet 4.5 (primary agent + memory manager)
- **Storage**: File-system-like hierarchical structure (summarized entities linked to raw trajectories)
- **Navigation agent**: Lightweight RL-trained agent for hierarchy traversal

### Evaluation
- **Metrics**: F1 (primary)
- **Benchmarks**: LoCoMo, LongMemEval, ALFWorld
- **Key claim**: At most 22.17% of baseline token usage

---

## 11. MemOS (arXiv 2025)

### Build
- **Model**: Qwen2.5-7B-Instruct (7B scale), Qwen2.5-72B-Instruct (72B scale)
- **Storage**: Tree-based multi-layered, multi-granularity memory

### Evaluation
- **Metrics**: F1, BLEU-1
- **Benchmarks**: LoCoMo

### Reported Results (LoCoMo F1)
| Scale | F1 |
|---|---|
| 7B | 37.05 |
| 72B | 42.79 |

---

## 12. SimpleMem (arXiv 2026)

### Build
- **Model**: Qwen2.5-3B-Instruct (generative backbone)
- **Embedding**: nomic-embed-text-v1.5
- **Storage**: Efficient lifelong memory with simple architecture

### Evaluation
- **Uses official LoCoMo-10 evaluation script** with LLM-as-Judge
- **Metrics**: F1, LLM Judge

### Reported Results (LoCoMo F1)
| Model | Overall | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|---|
| GPT-4.1-mini | 43.24 | 51.12 | 43.46 | 58.62 | 19.76 |
| GPT-4o | 39.06 | 45.41 | 35.89 | 56.71 | 18.23 |

---

## 13. Hindsight (arXiv 2025)

### Build
- **Storage**: Agent memory that retains, recalls, and reflects

### Evaluation
- **Judge prompt**: Defined specific LoCoMo preamble + LongMemEval category-specific preambles. ByteRover adopted these prompts verbatim.
- **Benchmarks**: LoCoMo, LongMemEval

### Reported Results
- LoCoMo: 89.6% (with larger backbone)
- LongMemEval: 91.4%

---

## 14. ES-Mem (arXiv 2026)

### Build
- **Storage**: Event segmentation-based coherent memory units
- **Evaluated on**: LoCoMo (600 turns avg per conversation)

### Reported Results (LoCoMo F1)
- Overall: 45.56

---

## 15. DeltaMem (arXiv 2026)

### Build
- **Model**: 8B model with RL training
- **Storage**: Online memory with delta updates

### Evaluation
- **Judge**: LLM-as-Judge
- **Benchmarks**: LoCoMo

### Reported Results
- LoCoMo Judge: 75.1%

---

## 16. MemPalace (arXiv 2026)

### Build
- **Storage**: Spatial metaphor-based memory architecture (method of loci)

### Evaluation
- **Benchmarks**: LongMemEval

### Reported Results
- LongMemEval: 96.6% (highest reported)

Note: Independent audit (GitHub Issue #29, April 2026) raised concerns about benchmark claims. Maintainers acknowledged.

---

## Summary: Evaluation Protocol Comparison

| Method | Build Model | Answerer | Judge Model | Judge Type | Metrics | Adversarial | Benchmarks |
|---|---|---|---|---|---|---|---|
| **LoCoMo official** | — | Various | — | F1 auto | F1, EM, BERTScore | Keyword match | LoCoMo |
| **Mem0** | GPT-4o-mini | Same | GPT-5 | 0/1 binary | Judge, F1, BLEU-1 | **Excluded** | LoCoMo, LME, BEAM |
| **A-Mem** | GPT-4o-mini | Same | — | Auto only | 12 metrics (F1, BLEU, ROUGE, BERT, METEOR, SBERT) | Keyword match | LoCoMo |
| **Mnemis** | GPT-4.1-mini | Same | Not stated | 0/1 binary | Judge, F1, BLEU-1 | **Excluded** | LoCoMo, LME |
| **MemMachine** | GPT-4.1-mini | Same | Not stated | 0/1 binary | Judge, F1, BLEU-1 | **Not reported** | LoCoMo |
| **Memory-R1** | 7B/8B RL | Same | LLM | 0/1 binary | Judge, F1, BLEU-1 | Not stated | LoCoMo |
| **Synthius-Mem** | — | — | LLM | 0/1 binary | Judge | **Included (99.6%)** | LoCoMo |
| **ByteRover** | Gemini 3 | Same | Gemini 3 Flash | 0/1 binary | Judge | Excluded | LoCoMo, LME |
| **TiMem** | GPT-4o-mini | Same | LLM | 0/1 binary | Judge | Not stated | LoCoMo |
| **HORMA** | Claude 4.5 | Same | — | F1 auto | F1 | Not stated | LoCoMo, LME |
| **MemOS** | Qwen 7B/72B | Same | — | Auto | F1, BLEU-1 | Not stated | LoCoMo |
| **SimpleMem** | Qwen 3B | GPT-4.1-mini | LLM | 0/1 binary | Judge, F1 | Not stated | LoCoMo |
| **Hindsight** | — | — | LLM | 0/1 binary | Judge | Not stated | LoCoMo, LME |
| **ES-Mem** | — | — | — | Auto | F1 | Not stated | LoCoMo |
| **DeltaMem** | 8B RL | Same | LLM | 0/1 binary | Judge | Not stated | LoCoMo |
| **MemPalace** | — | — | LLM | 0/1 binary | Judge | Not stated | LME |

### Key Observations

1. **Judge model is not standardized** — ranges from GPT-4o-mini to GPT-5 to Gemini 3. Results across papers are **not directly comparable**.
2. **Most papers report 3 metrics**: LLM Judge (primary), F1, BLEU-1. Judge is the comparison metric.
3. **Adversarial handling varies**: Mem0/Mnemis exclude it; Synthius-Mem includes it; most don't report.
4. **F1 and Judge can diverge hugely**: MemMachine has Judge=91% but F1=28%. Answer length is the main factor.
5. **Build and answer model are usually the same** — the same LLM that builds memory also answers questions.
6. **Hindsight's judge prompt** has become a de facto standard — ByteRover adopts it verbatim for comparability.
7. **Only fair comparison is same-judge**: papers that compare against baselines re-run them with the same judge.

### LongMemEval Official Evaluation

- **Judge**: LLM (default GPT-4o-mini), answers "yes" or "no"
- **Per-category prompts** differ:
  - single-session / multi-session: standard correctness check
  - temporal-reasoning: allows off-by-one day errors
  - knowledge-update: correct if updated answer is present (even with old info)
  - single-session-preference: rubric-based, need not satisfy all points
- **Mem0's LongMemEval prompt** is more elaborate (unified, with abstention matching, off-by-one, superset rules)

### Our NativeMem Status

| Metric | NativeMem v3 | Notes |
|---|---|---|
| **F1** | 0.376 (sample 0) | With concise-answer prompt |
| **BLEU-1** | 0.080 | Low due to verbose answers |
| **LLM Judge** | Not yet computed with standard 0/1 | Previous 0-100 score: 55.3 |
| **Adversarial F1** | 0.723 | With "No information available" prompt |
| Build model | deepseek-v4-flash | — |
| Samples evaluated | 1 (sample 0, 199 questions) | Need all 10 |

**To align with standard evaluation**: use Mem0's judge prompt (0/1 binary, 14-day tolerance), report F1 + BLEU-1 + Judge, evaluate all 10 samples.

---

Sources:
- [Mem0 Paper](https://arxiv.org/html/2504.19413v1)
- [Mem0 Benchmarks](https://github.com/mem0ai/memory-benchmarks)
- [Mnemis Paper](https://arxiv.org/pdf/2602.15313)
- [Mnemis GitHub](https://github.com/microsoft/Mnemis)
- [MemMachine Blog](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- [Memory-R1](https://arxiv.org/html/2508.19828v5)
- [Synthius-Mem](https://arxiv.org/abs/2604.11563)
- [ByteRover Blog](https://www.byterover.dev/blog/benchmark-ai-agent-memory)
- [ByteRover Paper](https://arxiv.org/html/2604.01599v1)
- [HORMA Paper](https://arxiv.org/pdf/2606.11680)
- [TiMem Paper](https://arxiv.org/pdf/2601.02845)
- [SimpleMem Paper](https://arxiv.org/pdf/2601.02553)
- [ES-Mem Paper](https://arxiv.org/pdf/2601.07582)
- [Hindsight Paper](https://arxiv.org/html/2512.12818v1)
- [MemOS GitHub](https://github.com/MemTensor/MemOS)
- [A-Mem GitHub](https://github.com/WujiangXu/A-mem)
- [LongMemEval Eval Code](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)
- [Mem0 2026 Blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [LoCoMo Benchmark Overview](https://www.emergentmind.com/topics/locomo-benchmark-scores)
