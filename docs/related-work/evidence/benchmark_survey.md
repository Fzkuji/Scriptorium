# LLM Memory Systems Benchmark Survey

Compiled: 2026-07-01. Covers all major systems evaluated on LoCoMo, LongMemEval, and BEAM.

---

## 1. LoCoMo Benchmark Overview

- **Source**: Snap Research, ACL 2024
- **Scale**: 10 multi-session conversations, 1,986 QA pairs
- **Categories**: Single-hop (841), Multi-hop (282), Temporal (321), Open-domain (96), Adversarial (446)
- **Standard metrics**: F1 (token-level), BLEU-1, LLM-as-Judge (0/1 binary)
- **Human performance**: F1 = 87.9

---

## 2. LoCoMo Results: LLM-as-Judge (Accuracy %)

The primary metric used by most recent papers. 0/1 binary classification (CORRECT/WRONG).

| Method | Venue | LLM | Judge Model | Overall | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial | Notes |
|--------|-------|-----|-------------|---------|------------|-----------|----------|-------------|-------------|-------|
| **ByteRover 2.0** | Blog 2025 | Gemini 3 Flash/Pro | Gemini 3 Flash | **96.1** | 95.4 | 85.1 | 94.4 | 77.2 | — | Uses Hindsight eval prompt |
| **Synthius-Mem** | arXiv 2026 | — | — | **94.4** | — | — | — | — | **99.6** | Only system reporting adversarial |
| **Mnemis** | ACL 2026 | GPT-4.1-mini | — | **93.9** | 86.9 | 77.2 | 74.2 | 56.6 | — | Dual-route hierarchical graph |
| **Mem0 (2026 algo)** | Blog 2026 | — | — | **92.5** | — | — | — | — | — | Updated algorithm |
| **MemMachine v0.2** | Blog 2025 | GPT-4.1-mini | — | **91.7** | 94.4 | 89.7 | 89.1 | 75.0 | — | Agent mode |
| **MemMachine v0.2** | Blog 2025 | GPT-4.1-mini | — | **91.2** | 94.4 | 89.7 | 89.1 | 75.0 | — | Memory mode |
| **Hindsight** | arXiv 2025 | — | — | **89.6** | 86.2 | 70.8 | 83.8 | 95.1 | — | |
| **T-Mem** | arXiv 2026 | — | — | **80.3** | — | — | — | — | — | |
| **MemOS (0630)** | arXiv 2025 | — | — | **75.8** | — | — | — | — | — | |
| **TiMem** | arXiv 2026 | GPT-4o-mini | — | **75.3** | — | — | — | — | — | 52% memory reduction |
| **DeltaMem** | arXiv 2026 | 8B RL | — | **75.1** | — | — | — | — | — | RL-based |
| **Mem0 (original)** | ECAI 2025 | GPT-4o-mini | — | **66.9** | 67.1 | 51.2 | 55.5 | 72.9 | — | Excludes adversarial |
| **Memory-R1 (GRPO)** | arXiv 2026 | Qwen2.5-7B | — | **62.7** | — | — | — | — | — | RL-trained, 7B model |
| **Memory-R1 (PPO)** | arXiv 2026 | LLaMA-3.1-8B | — | **57.5** | — | — | — | — | — | |

### Notes on LLM-as-Judge
- Most papers use **0/1 binary** (CORRECT/WRONG), not 0-100 continuous
- Mem0's judge has **14-day date tolerance**, partial credit, paraphrase acceptance
- ByteRover uses **Hindsight's evaluation prompt** (different judge template)
- **Adversarial category mostly excluded** by most papers (Mem0 explicitly `CATEGORIES_TO_EVALUATE = [1,2,3,4]`)
- Judge model varies: GPT-4o-mini, GPT-4o, Gemini 3 Flash — **not standardized**

---

## 3. LoCoMo Results: F1 Score

Token-level F1 using LoCoMo's official `evaluation.py`. Adversarial uses keyword matching ("no information available" / "not mentioned").

| Method | LLM | Overall F1 | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial |
|--------|-----|------------|------------|-----------|----------|-------------|-------------|
| **Human** | — | **87.9** | — | — | — | — | — |
| **ES-Mem** | — | **45.56** | — | — | — | — | — |
| **SimpleMem** | GPT-4.1-mini | **43.24** | 51.12 | 43.46 | 58.62 | 19.76 | — |
| **Memory-R1 (GRPO)** | Qwen2.5-7B | **45.0** | — | — | — | — | — |
| **Memory-R1 (PPO)** | LLaMA-3.1-8B | **41.1** | — | — | — | — | — |
| **SimpleMem** | GPT-4o | **39.06** | 45.41 | 35.89 | 56.71 | 18.23 | — |
| **Mem0** | GPT-4o-mini | **38.72** | 38.72 | 28.64 | 48.93 | 47.65 | — |
| **Mem0** | GPT-4o | **36.09** | 39.12 | 35.13 | 52.38 | 17.73 | — |
| **Full Context** | GPT-4o-mini | **35.57** | 45.89 | 25.01 | 23.83 | 15.49 | — |
| **Mem0** | GPT-4.1-mini | **34.20** | 41.30 | 30.14 | 48.91 | 16.43 | — |
| **A-Mem** | GPT-4o-mini | **~35** | 27.02 | 45.85 | 12.14 | 44.65 | 50.03 |
| **A-Mem** | GPT-4o | **~35** | 32.86 | 39.41 | 17.10 | 48.43 | 36.35 |
| **GPT-4-turbo** | 4K context | **~32** | — | — | — | — | — |
| **MemMachine v0.2** | GPT-4.1-mini | **27.85** | 31.27 | 24.97 | 25.49 | 14.29 | — |
| **LightMem** | GPT-4.1-mini | **24.63** | 33.79 | 24.96 | 20.55 | 19.21 | — |
| **Full Context** | GPT-4.1-mini | **18.70** | 18.68 | 25.02 | 12.04 | 19.05 | — |
| **A-Mem** | Qwen2.5-1.5B | **~26** | 18.23 | 24.32 | 16.48 | 23.63 | 46.00 |

### Notes on F1
- F1 is heavily penalized by long answers (low precision)
- MemMachine has high Judge (91.2%) but low F1 (27.85%) — answers are correct but verbose
- A-Mem is the only method that reports adversarial F1 (keyword matching)
- F1 and Judge can diverge significantly — **Judge is the primary comparison metric**

---

## 4. LoCoMo Results: BLEU-1

| Method | LLM | Overall B1 | Single-hop | Multi-hop | Temporal | Open-domain |
|--------|-----|------------|------------|-----------|----------|-------------|
| **Mem0** | GPT-4o-mini | **27.13** | 27.13 | 21.58 | 40.51 | 38.72 |
| **MemMachine v0.2** | GPT-4.1-mini | **17.32** | 18.68 | 17.95 | 15.21 | 10.59 |

### Notes on BLEU-1
- Few papers report BLEU-1 in detail
- Same issue as F1: penalizes verbose answers

---

## 5. LongMemEval Results

| Method | LLM | Overall | single-session-user | single-session-asst | knowledge-update | multi-session | temporal | abstention |
|--------|-----|---------|--------------------|--------------------|-----------------|--------------|----------|-----------|
| **MemPalace** | — | **96.6** | — | — | — | — | — | — |
| **OMEGA** | — | **95.4** | — | — | — | — | — | — |
| **Mem0 (2026)** | — | **94.4** | 94.3 | 97.1 | 100.0 | 70.7 | — | — |
| **Mnemis** | GPT-4.1-mini | **91.6** | — | — | — | — | — | — |
| **ByteRover** | — | **92.8** | — | — | — | — | — | — |
| **EverMemOS** | — | **83.0** | — | — | — | — | — | — |
| **TiMem** | GPT-4o-mini | **76.9** | — | — | — | — | — | — |
| **NativeMem v3** | deepseek-v4-flash | **73.3** | 100.0 | 80.0 | 60.0 | 40.0 | 80.0 | — |
| **ProMem** | — | **69.6** | — | — | — | — | — | — |

### Notes
- LongMemEval uses **LLM-as-Judge binary accuracy** (yes/no)
- Each category has a different judge prompt (temporal allows off-by-one)
- Default judge: gpt-4o-mini
- Our NativeMem result is on 30 samples (5 per category), not the full 500

---

## 6. BEAM Results

| Method | BEAM-1M | BEAM-10M |
|--------|---------|----------|
| **Mem0** | 64.1 | 48.6 |

Few papers report BEAM results.

---

## 7. Our NativeMem v3 Results (for reference)

**LoCoMo Sample 0, 199 questions, deepseek-v4-flash:**

| Metric | Value |
|--------|-------|
| LJ Score (0-100, our custom) | 55.3 |
| LJ w/o Adversarial | 69.5 |
| F1 (official, verbose answers) | 0.12 |
| Found rate | 86% |
| Validation pass rate | 100% |
| Build calls | 480 |
| Build time | 1,231s |
| Avg retrieval steps | 7.1 |

**Per category (LJ 0-100):**

| Category | LJ |
|----------|-----|
| Multi-hop | 53.2 |
| Temporal | 79.7 |
| Open-domain | 73.4 |
| Session | 59.5 |
| Adversarial | 9.4 |

### Key gaps vs. published results:
1. Our LJ is 0-100 continuous, theirs is 0/1 binary — **not comparable**
2. Our F1 is very low (0.12) because answers are too verbose
3. We use deepseek-v4-flash; most top results use GPT-4o-mini or GPT-4.1-mini
4. We only evaluated Sample 0 (199 questions); standard is all 10 samples (1,986 questions)
5. We need to switch to 0/1 binary judge with proper judge prompt (Mem0's or Hindsight's)

---

## 8. Evaluation Configuration Summary

| Aspect | LoCoMo Standard | LongMemEval Standard | Our Current |
|--------|----------------|---------------------|-------------|
| Primary metric | LLM Judge (0/1) | LLM Judge (0/1) | LJ (0-100) — wrong |
| Secondary metrics | F1, BLEU-1 | — | F1 only |
| Judge model | Varies (GPT-4o-mini typical) | gpt-4o-mini | deepseek-v4-flash |
| Date tolerance | 14 days (Mem0) | off-by-one | none |
| Partial credit | yes (Mem0 judge) | yes | no |
| Adversarial | mostly excluded | N/A | included |
| Samples | all 10 conversations | full 500 | 1 sample / 30 samples |
| Answer format | concise preferred | concise preferred | verbose |

---

## Sources

- [Mem0 Paper (ECAI 2025)](https://arxiv.org/abs/2504.19413)
- [A-Mem Paper](https://arxiv.org/abs/2502.12110)
- [Mnemis (ACL 2026)](https://arxiv.org/abs/2602.15313)
- [TiMem](https://arxiv.org/abs/2601.02845)
- [SimpleMem](https://arxiv.org/abs/2601.02553)
- [Synthius-Mem](https://arxiv.org/abs/2604.11563)
- [MemMachine Blog](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- [ByteRover Blog](https://www.byterover.dev/blog/benchmark-ai-agent-memory)
- [Memory-R1 (DeepWiki)](https://deepwiki.com/yansikuan/memory-r1/5.1-locomo-benchmark)
- [DeltaMem](https://arxiv.org/abs/2604.01560)
- [MemOS](https://github.com/MemTensor/MemOS)
- [Mem0 Benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- [Mem0 memory-benchmarks repo](https://github.com/mem0ai/memory-benchmarks)
- [LoCoMo Benchmark Scores](https://www.emergentmind.com/topics/locomo-benchmark-scores)
