# 已发表结果大矩阵（Published Results Matrix）

> 生成日期 2026-07-02。本文档把 19 个来源（论文 / GitHub README / 官方博客）在 LoCoMo 与 LongMemEval 上的**自报数字**合并成矩阵。
>
> **核心警示**：所有数字均为各来源自报，**同一方法在不同论文里的数字普遍不同**（builder/answerer/judge/题目子集/指标实现全都不统一）。**只有同一配置组（同一张子表）内的数字可横向比较**；跨表比较必须逐项核对脚注。数字忠实转录，不做修饰；`—` = 该来源未报；`?` = 原始来源存疑或内部不一致。±std 一律省略（Mem0/TiMem/Nemori v1/MemOS v2 原文带 std，见各自抽取报告）。

## 0. 来源标记与配置速查

| 标记 | 来源 | 构建/回答 LLM | Judge | LoCoMo 范围 | LongMemEval | 备注 |
|---|---|---|---|---|---|---|
| [Mem0] | arXiv 2504.19413 (ECAI'25) | gpt-4o-mini / gpt-4o-mini | gpt-4o-mini（代码），10 次判分取均值 | cat1-4，1540 题 | 未评 | Mem0 宽松 0/1 judge 的源头 |
| [Mem0-26] | mem0 README 2026-04 | "production stack"（未细披露） | 未披露 | 1540 题 | 500 题 | 新算法，非论文 |
| [Mem0-Plat] | mem0-benchmarks repo | Platform v3；answerer=**GPT-5** | **GPT-5**（新版宽 judge，14 天容忍） | 1540 题 | 500 题 | 与 [Mem0] 论文协议不同代 |
| [Nemori] | arXiv 2508.03341 v4（v1 注明） | gpt-4o-mini 或 gpt-4.1-mini（同款构建+回答） | gpt-4o-mini（论文文字；代码默认 4.1-mini ?） | cat1-4，1540 题 | LME-S 500 题 | baseline 全部自跑 |
| [Nemori-V5] | Nemori GitHub README | gpt-4.1-mini（OpenRouter） | 4.1-mini ?（代码默认） | 1540 题 | — | 与两版论文数字都不同 |
| [LightMem] | arXiv 2510.18866 v4 (ICLR'26) + GitHub | gpt-4o-mini / Qwen3-30B-A3B / GLM-4.6（构建=回答） | gpt-4o-mini（论文）；README 另有 qwen2.5-32b 列 | cat1-4，1540 题 | LME-S 500 题（弃 5 坏样本记错） | 论文 LoCoMo 只有 overall ACC |
| [SimpleMem] | arXiv 2601.02553 v3 | 7 个 backbone 分表 | LoCoMo **无 judge**；LME judge=gpt-4.1-mini | 名义 4 类（自称 1986 题 ?） | LME-S，**Average=6 类宏平均** | LoCoMo Average=4 类宏平均（非题数加权） |
| [MemMachine] | memmachine.ai 博客 2025-12 | 构建未披露；answerer gpt-4o-mini / gpt-4.1-mini | gpt-4o-mini（Mem0 eval 代码） | cat1-4，1540 题 | repo README（answerer gpt-5-mini，judge gpt-4o）? | LME 数字非正式自报，存疑 |
| [MemOS] | arXiv 2507.03724 v4（v2 注明） | GPT-4o-mini（全部方法） | gpt-4o-mini（repo 默认），LoCoMo 3 次判分 | cat1-4，1540 题 | LME-S ?（自制通用 judge 模板） | v4=MemOS-1031，v2=MemOS-0630 |
| [Mnemis] | arXiv 2602.15313 (ACL'26) | GPT-4o-mini / GPT-4.1-mini 分块 | **GPT-4.1-mini 统一**（各数据集官方 prompt） | cat1-4，1540 题 | LME-S 500 题 | baseline 行=转录各家自报数字 |
| [MIRIX] | arXiv 2507.07957 v1 | gpt-4.1-mini | 正文称 GPT-4.1 / 代码钉死 gpt-4o-mini ? | cat1-4，名义 1540（run1/3 实为 1542） | 未评 | 3 run 平均；**类别标签错位**（见 §1 警示） |
| [Hindsight] | arXiv 2512.12818 v1 | GPT-OSS-20B/120B/Gemini-3（Gemini 仅回答） | **GPT-OSS-120B** | 4 类（题数未写 ?） | LME-S 500 题 | LoCoMo baseline 全转引 Backboard 页 |
| [E-mem] | arXiv 2601.21714 (ICML'26) | master gpt-4o-mini 或 Qwen2.5-14B + assistant Qwen3-4B | **无 judge**（F1/B1） | 4 类主表 + adversarial 单列 | 未评 | baseline 自跑 |
| [ES-Mem] | arXiv 2601.07582 | gpt-4o-mini / Qwen2.5-3B / Llama3.2-3B | LoCoMo 无 judge；LME 官方 prompt（judge 模型未写 ?） | 4 类（题数未写 ?） | LME-S 500 题 | "F1 45.56" 的真正出处 |
| [LoCoMo] | arXiv 2402.17753 (ACL'24) | GPT-3.5/GPT-4-turbo/Llama-2 等 | **无 judge**（规则 F1；cat5=子串规则） | **5 类含 adversarial，50 对话 7512 题** | 未评（早于 LME） | 与 locomo10（1986 题）不同集 |
| [A-Mem] | arXiv 2502.12110 v11 (NeurIPS'25) | 多 backbone（构建=回答） | **无 judge**（set-F1/B1 等） | **5 类含 adversarial**（repo=1986 题；正文写 7512 ?） | 未评 | cat5 评分 reference=诱导答案 |
| [Zep] | arXiv 2501.13956 v1 | 图构建 gpt-4o-mini；回答 gpt-4o-mini / gpt-4o | **GPT-4o**（LME 官方 per-type prompt） | 论文无 LoCoMo（repo 实验 0.8032，gpt-4o-mini） | LME-S 500 题 | 官网 90.2%/94.7% 无方法学 ? |
| [TiMem] | arXiv 2601.02845 | 内部+回答 gpt-4o-mini（LME 另报 gpt-4o） | LoCoMo gpt-4o-mini（Mem0 prompt）；LME 官方 prompt | cat1-4，1540 题 | LME-S 500 题 | baseline 全部自跑 |
| [Memory-R1] | arXiv 2508.19828 v5 | RL 微调 LLaMA-3.1-8B / Qwen2.5-7B | 未披露 ?（MemGPT+Mem0 式宽松 prompt） | **cat1-4 的 1,307 题测试切分**（非 1540） | LME 零样本（overall=六类宏平均） | answer prompt 硬限 5-6 词 |
| [Synthius] | arXiv 2604.11563 | Synthius 用 GPT-4.1-mini；controlled baseline 用 Gemini 3 Flash | GPT-4.1-mini | **1,813 题含 adversarial 442 题** ? | 未评 | 无同行评审痕迹，正文与表格互相矛盾 |
| [ByteRover] | arXiv 2604.01599 + 官方博客 | curate=Gemini 3 Flash；justifier=Gemini 3.1 Pro（博客 Run1=Flash） | **Gemini 3 Flash**（Hindsight prompt） | 4 类 1,536 题（OD 只算 92 题 ?） | LME-S 500 题 | "同 harness 重跑" baseline 与各家自报逐位相同，存疑 ? |
| [Memobase] | memodb-io/memobase GitHub | Memobase 构建；answerer 默认 **gpt-4o** | gpt-4o-mini（代码硬编码） | cat1-4，1540 题 | 未评 | mem0 eval fork；**类别标签错位** |
| [EverMemOS] | arXiv 2601.02163 | GPT-4o-mini / GPT-4.1-mini 分块 | **GPT-4o-mini + 两辅助 judge 取均值**（MemOS 协议） | cat1-4，1540 题 | LME-S（baseline 转录 MemOS leaderboard） | Mem0/MemU/MemOS/Zep 用官方 API 建忆 |
| [DeltaMem] | arXiv 2604.01560 v1 | 记忆 agent Qwen3-4B/8B(+RL) 等；**回答统一 GPT-4o-mini** | 未披露 ? | 4 类（题数未明说 ?） | 未评 | Memory-R1 行搬运错位、Zep/LangMem 行列轮转（见警示） |
| [OMEGA] | omegamax.co 博客 | OMEGA v1.0.0；**GPT-4.1 回答+判分** | GPT-4.1 | — | 500 题；headline=5 类不加权均值 ? | 95.4 无法从分项复算（应为 94.56/93.2） |
| [MemPalace] | arXiv 2604.21284（批判性分析） | — | — | Recall@10 60.3/88.9 | **Recall@5 96.6（检索指标，非 QA）**；端到端 QA ~67.2 ? | 96.6 不可入 QA 对比列 |
| [LME-off] | arXiv 2410.10813 (ICLR'25) | 各长上下文 LLM | GPT-4o（官方 per-type prompt） | — | S/M/Oracle | 无任何 baseline 的 per-type 表 |

**全局警示：mem0 系类别标签错位。** LoCoMo 官方类别 ID：cat1=multi-hop(282)、cat2=temporal(321)、cat3=open-domain(96)、cat4=single-hop(841)、cat5=adversarial(446)。但 mem0 评测代码谱系（[Mem0] 论文表、[MIRIX]、[Memobase]、[DeltaMem] 等）把 cat1 标成 "Single-hop"、cat4 标成 "Temporal" 等，**per-category 标签与真实类别错位**；[MemMachine]、[Mnemis]、[Nemori]、[TiMem]、[LightMem] 等用真实映射。本文档一律**按各来源打印的列名转录**，错位来源在脚注标注 ⚠。跨来源引用 per-category 数字前必须先按题数（282/321/96/841）对齐；overall 不受影响。

---

## 1. LoCoMo — LLM-as-a-Judge（0/1 accuracy，%）

行序统一为：single-hop / multi-hop / temporal / open-domain /（adversarial）/ overall，**按各来源自己的列标签填入**。

### 1-A 组：answerer = gpt-4o-mini 系

#### A1. [Mem0] 论文 Table 1/2（builder/answerer/judge 全 gpt-4o-mini；⚠ 标签疑错位）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| A-Mem*[Mem0重跑] | 39.79 | 18.85 | 49.91 | 54.05 | 48.38 |
| LangMem[Mem0] | 62.23 | 47.92 | 23.43 | 71.12 | 58.10 |
| Zep[Mem0] | 61.70 | 41.35 | 49.31 | 76.60 | 65.99 |
| OpenAI[Mem0] | 63.79 | 42.92 | 21.71 | 62.29 | 52.90 |
| Mem0[Mem0] | 67.13 | 51.15 | 55.51 | 72.93 | 66.88 |
| Mem0ᵍ[Mem0] | 65.71 | 47.19 | 58.13 | 75.71 | 68.44 |
| Full-ctx[Mem0] | — | — | — | — | 72.90 |
| RAG(k2,256)[Mem0] | — | — | — | — | 60.97 |

脚注：judge=gpt-4o-mini（MemGPT 式宽松 prompt，无 14 天规则）；1540 题；LoCoMo/ReadAgent/MemoryBank/MemGPT/A-Mem 转抄行无 J 值故未列。⚠ 按 mem0 代码谱系，"single-hop" 列实为 cat1（真实 multi-hop 282 题），"temporal" 列实为 cat4（真实 single-hop 841 题）。

#### A2. [Nemori] v4 Table 2（构建=回答 gpt-4o-mini；judge gpt-4o-mini；全部自跑；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Full-ctx[Nemori] | 83.0 | 66.8 | 56.2 | 48.6 | 72.3 |
| RAG-4096[Nemori] | 32.0 | 31.3 | 23.7 | 32.6 | 30.2 |
| LangMem[Nemori] | 61.4 | 52.4 | 24.9 | 47.6 | 51.3 |
| Zep[Nemori] | 63.2 | 50.5 | 58.9 | 39.6 | 58.5 |
| Mem0[Nemori] | 68.1 | 60.3 | 50.4 | 40.6 | 61.3 |
| A-MEM[Nemori] | 58.2 | 43.6 | 54.2 | 22.9 | 52.5 |
| MemoryOS[Nemori] | 62.5 | 52.5 | 38.0 | 45.8 | 54.5 |
| Nemori[Nemori] | 81.9 | 61.7 | 67.6 | 45.8 | 73.0 |

脚注：Mem0/Zep 用其商业 API 构建+检索、Nemori 接回答模型。v1 的 Nemori 自报为 74.4（0.744），v4 重跑降至 73.0。

#### A3. [LightMem] GitHub README（backbone/judge 均 gpt-4o-mini；真实标签；论文 Table 3 只报 overall）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| FullText[LightMem] | 86.56 | 68.79 | 50.16 | 56.25 | 73.83 ? |
| NaiveRAG[LightMem] | 70.99 | 55.32 | 56.39 | 47.92 | 63.64 |
| A-MEM[LightMem] | 72.06 | 56.03 | 60.44 | 31.25 | 64.16 |
| MemoryOS(eval)[LightMem] | 67.06 | 56.74 | 40.19 | 45.83 | 58.25 |
| MemoryOS(pypi)[LightMem] | 63.97 | 52.13 | 36.76 | 43.75 | 54.87 |
| Mem0-OSS[LightMem] | 38.41 | 30.85 | 37.07 | 34.38 | 36.49 |
| Mem0(api)[LightMem] | 66.47 | 56.38 | 59.19 | 43.75 | 61.69 |
| Mem0-g(api)[LightMem] | 65.99 | 54.26 | 57.01 | 39.58 | 60.32 |
| LangMem[LightMem] | — | — | — | — | 57.20 |
| LightMem(512,0.7)[LightMem] | 77.41 | 62.41 | 74.14 | 44.79 | 71.95 |
| LightMem(768,0.8)[LightMem] | 76.81 | 67.02 | 76.32 | 45.83 | 72.99 |

脚注：FullText 论文 Table 3 写 71.83、README 写 73.83（自报冲突 ?）。README 另有 qwen2.5-32b judge 全套（LightMem 最高 74.35）。Qwen3-30B backbone 组：LightMem(1024,0.8) overall 72.60、FullText 74.87、Mem0-OSS 43.31。

#### A4. [MemOS] v4 Table 3 + v2 Table 3（GPT-4o-mini 全部；judge gpt-4o-mini，3 runs；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MIRIX[MemOS] | 68.22 | 54.26 | 68.54 | 46.88 | 64.33 |
| Mem0[MemOS] | 73.33 | 58.75 | 52.34 | 45.83 | 64.57 |
| Zep[MemOS-v4] | 66.23 | 52.12 | 54.82 | 33.33 | 59.22 |
| Memobase[MemOS] | 73.12 | 64.65 | 81.20 | 53.12 | 72.01 |
| MemU[MemOS] | 66.34 | 63.12 | 27.10 | 50.01 | 56.55 |
| Supermemory[MemOS] | 67.30 | 51.12 | 31.77 | 42.67 | 55.34 |
| MemOS-1031[MemOS] | 81.09 | 67.49 | 75.18 | 55.90 | 75.80 |
| MemOS-0630[MemOS-v2] | 78.44 | 64.30 | 73.21 | 55.21 | 73.31 |
| LangMem[MemOS-v2] | 68.21 | 56.74 | 24.09 | 49.65 | 55.76 |
| Zep[MemOS-v2] | 50.42 | 42.20 | 19.11 | 38.19 | 41.62 |
| OpenAI[MemOS-v2] | 61.83 | 60.28 | 28.25 | 32.99 | 52.75 |
| Full-ctx[MemOS-v2] | — | — | — | — | 71.58 |

脚注：全部为 MemOS 团队非官方复现实现。v2 RAG 最好配置（256,k2）= 56.54。v4 的 Zep 是重跑新数字（v2 的 41.62 → v4 的 59.22，自家两版差 17.6）。

#### A5. [TiMem] Table 1（内部+回答 gpt-4o-mini；judge gpt-4o-mini + Mem0 prompt；自跑；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemoryBank[TiMem] | 46.18 | 33.36 | 29.34 | 36.67 | 39.77 |
| A-MEM[TiMem] | 52.82 | 38.37 | 60.87 | 43.75 | 51.29 |
| Mem0[TiMem] | 62.09 | 50.14 | 59.25 | 37.70 | 57.79 |
| MemoryOS[TiMem] | 68.37 | 52.76 | 52.46 | 46.67 | 60.79 |
| MemOS[TiMem] | 76.07 | 56.85 | 69.47 | 45.14 | 69.24 |
| TiMem[TiMem] | 81.43 | 62.20 | 77.63 | 52.08 | 75.30 |

Table 7 换回答模型（内部仍 gpt-4o-mini，LLJ-G judge）：TiMem+GPT-4o 74.03、+Qwen3-8B 66.04、+Qwen3-32B 74.61、+Qwen3-235B-A22B 80.45。

#### A6. [EverMemOS] Table 1 GPT-4o-mini 块（judge=gpt-4o-mini+两辅助 judge 均值；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemoryOS[EverMemOS] | 62.43 | 56.50 | 37.18 | 40.28 | 54.70 |
| Mem0[EverMemOS] | 66.71 | 58.16 | 55.45 | 40.62 | 61.00 |
| MemU[EverMemOS] | 72.77 | 62.41 | 33.96 | 46.88 | 61.15 |
| MemOS[EverMemOS] | 81.45 | 69.15 | 72.27 | 60.42 | 75.87 |
| Zep[EverMemOS] | 88.11 | 71.99 | 74.45 | 66.67 | 81.06 |
| EverMemOS[EverMemOS] | 91.08 | 86.17 | 81.93 | 66.67 | 86.76 |

脚注：三 judge 平均（非单 judge），与 A1–A5 不可直接同列比。Mem0/MemU/MemOS/Zep 用官方 API 建忆，仅回答阶段统一骨干——其 Zep 81.06 显著高于其他论文的 Zep 复现值。

#### A7. [DeltaMem] Table 1（回答统一 GPT-4o-mini；judge 未披露 ?；⚠ 标签有错位/轮转）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| A-Mem[DeltaMem] | 54.05 | 39.79 | 49.91 | 18.85 | 48.38 |
| Mem0[DeltaMem] | 60.88 | 52.84 | 59.50 | 40.62 | 57.86 |
| LangMem[DeltaMem]⚠ | 71.12 | 62.23 | 23.43 | 47.92 | 58.11 |
| Zep[DeltaMem]⚠ | 76.60 | 61.70 | 49.31 | 41.35 | 65.99 |
| LightMem[DeltaMem] | 72.06 | 53.90 | 69.16 | 44.79 | 66.43 |
| Memory-R1-7B[DeltaMem]⚠⚠ | 67.81 | 49.61 | 69.16 | 20.71 | 62.34 |
| DeltaMem-4B | 75.62 | 61.35 | 70.09 | 47.92 | 70.13 |
| DeltaMem-8B | 77.41 | 61.70 | 69.78 | 46.88 | 71.04 |
| DeltaMem-4o-mini | 80.38 | 58.87 | 73.21 | 47.92 | 72.92 |
| DeltaMem-4B-RL | 80.02 | 65.60 | 72.59 | 51.04 | 74.03 |
| DeltaMem-8B-RL | 82.05 | 64.18 | 73.83 | 51.04 | 75.13 |

脚注：⚠ Zep/LangMem 行相对 Mem0 论文原标签发生列轮转；⚠⚠ Memory-R1-7B 行是从 Memory-R1 论文搬运后**错位打乱**的（其 overall 62.34 实为原文 single-hop J），引用须极度小心。DeltaMem 各变体名指记忆管理 agent（Qwen3-4B/8B/GPT-4o-mini，RL=GRPO 训练）。

#### A8. [MemMachine] 博客（answerer gpt-4o-mini；judge gpt-4o-mini，Mem0 eval 代码；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemMachine-memory[MemMachine] | 94.65 | 87.59 | 73.52 | 70.83 | 87.47 |
| MemMachine-agent[MemMachine] | 93.94 | 84.04 | 80.69 | 73.96 | 88.12 |

#### A9. [Mnemis] Table 1 GPT-4o-mini 块（**judge=GPT-4.1-mini**，非 4o-mini；baseline 为转录各家自报；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Full-ctx[Mnemis自跑] | 83.0 | 66.8 | 56.2 | 48.6 | 72.3 |
| RAG[Mnemis自跑] | 73.5 | 59.9 | 62.9 | 63.5 | 68.2 |
| LangMem[Mnemis转] | 61.4 | 52.4 | 24.9 | 47.6 | 51.3 |
| MemOS[Mnemis转] | 78.4 | 64.3 | 73.2 | 55.2 | 73.3 |
| Mem0[Mnemis转] | 68.1 | 60.3 | 50.4 | 40.6 | 61.3 |
| Zep[Mnemis转] | 63.2 | 50.5 | 58.9 | 39.6 | 58.5 |
| Nemori[Mnemis转] | 82.1 | 65.3 | 71.0 | 44.8 | 74.4 |
| EMem-G[Mnemis转] | 82.3 | 74.7 | 76.0 | 57.3 | 78.0 |
| Mnemis[Mnemis] | 95.7 | 89.7 | 77.6 | 79.2 | 89.8 |

脚注：转录列 = Nemori v1 等各家自报数字原样搬运（如 Nemori 74.4 是其 v1 数字）。Mnemis repo 泄露 cat5 adversarial ≈34.3（论文不报）。

#### A10. 其他 gpt-4o-mini answerer 的 overall-only 数字

| 方法 | overall J | 来源与配置 |
|---|---|---|
| Zep（getzep repo 实验 2025-12） | 80.32 | gpt-4o-mini 回答+判分，10 runs 均值（min 79.8/max 81.2），跳过 cat5 [Zep-repo] |
| Nemori v1（论文初版自报） | 74.4 | 同 A2 配置，v4 重跑为 73.0 [Nemori-v1] |

### 1-B 组：answerer = gpt-4.1-mini 系

#### B1. [Nemori] v4 Table 2 GPT-4.1-mini 块（judge gpt-4o-mini；自跑；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Full-ctx[Nemori] | 86.9 | 77.2 | 74.2 | 56.6 | 80.6 |
| RAG-4096[Nemori] | 35.9 | 31.7 | 27.4 | 28.8 | 32.9 |
| LangMem[Nemori] | 84.5 | 71.0 | 50.8 | 59.0 | 73.4 |
| Zep[Nemori] | 66.9 | 53.7 | 60.2 | 43.8 | 61.6 |
| Mem0[Nemori] | 71.4 | 68.2 | 56.9 | 47.9 | 66.3 |
| A-MEM[Nemori] | 64.0 | 55.7 | 66.7 | 37.5 | 61.4 |
| MemoryOS[Nemori] | 68.9 | 62.4 | 37.7 | 60.4 | 60.6 |
| Nemori[Nemori] | 87.0 | 74.8 | 77.3 | 56.3 | 80.8 |

[Nemori-V5]（GitHub，judge 默认 4.1-mini ?）：Nemori overall 83.05；single-hop 88.59 / multi-hop 79.43 / temporal 78.82 / open-domain 59.38。

#### B2. [MIRIX] Table 2 gpt-4.1-mini 块（judge 正文 GPT-4.1 / 代码 gpt-4o-mini ?；⚠ 标签错位，按打印列名转录）

| 方法 | "Single Hop"（实为 cat1 multi-hop 282 题） | "Multi-Hop"（实为 cat2 temporal 321 题） | "Temporal"（实为 cat4 single-hop 841 题） | "Open Domain"（cat3 96 题，无错位） | overall |
|---|---|---|---|---|---|
| LangMem[MIRIX重跑] | 74.47 | 61.06 | 86.92 | 67.71 | 78.05 |
| RAG-500[MIRIX重跑] | 37.94 | 37.69 | 61.83 | 48.96 | 51.62 |
| Zep[MIRIX重跑,官方实现] | 79.43 | 69.16 | 83.33 | 73.96 | 79.09 |
| Mem0[MIRIX重跑] | 62.41 | 57.32 | 66.47 | 44.79 | 62.47 |
| MIRIX[MIRIX] | 85.11 | 83.70 | 88.39 | 65.62 | 85.38 |
| Full-ctx[MIRIX] | 88.53 | 77.70 | 92.70 | 71.88 | 87.52 |

脚注：MIRIX 按**真实类别**应读作 multi-hop 85.11 / temporal 83.70 / single-hop 88.39 / open-domain 65.62。3 run 平均（逐 run 83.98/87.34/84.82）。另：MIRIX 用 mem0 实现重跑 Zep 只有 49.09（疑 bug，未入表）；Full-Context (gpt-4o-mini) 77.51。

#### B3. [Mnemis] Table 1/8 GPT-4.1-mini 块（judge GPT-4.1-mini 统一；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | adversarial | overall |
|---|---|---|---|---|---|---|
| Full-ctx[Mnemis自跑] | 86.9 | 77.2 | 74.2 | 56.6 | — | 80.6 |
| RAG[Mnemis自跑] | 76.5 | 64.9 | 76.6 | 67.7 | — | 73.8 |
| LangMem[Mnemis转] | 84.5 | 71.0 | 50.8 | 59.0 | — | 73.4 |
| Mem0[Mnemis转] | 71.4 | 68.2 | 56.9 | 47.9 | — | 66.3 |
| Zep[Mnemis转] | 66.9 | 53.7 | 60.2 | 43.8 | — | 61.6 |
| Nemori[Mnemis转] | 84.9 | 75.1 | 77.6 | 51.0 | — | 79.5 |
| PREMem[Mnemis转] | 66.2 | 61.0 | 74.8 | 46.9 | — | 65.8 |
| EverMemOS[Mnemis转] | 96.1 | 91.1 | 89.7 | 70.8 | — | 92.3 |
| EMem-G[Mnemis转] | 90.5 | 79.6 | 80.8 | 71.7 | — | 85.3 |
| MIRIX[Mnemis转]⚠ | 85.1 | 83.7 | 88.4 | 65.6 | — | 85.4 |
| MemU[Mnemis转] | 94.9 | 88.3 | 92.5 | 77.1 | — | 92.1 |
| Mnemis(k=10)[Mnemis] | 96.2 | 91.8 | 90.3 | 82.3 | 34.30（repo 泄露） | 93.3 |
| Mnemis(k=30)[Mnemis] | 97.1 | 92.9 | 90.7 | 79.2 | 34.75（repo 泄露） | 93.9 |

脚注：⚠ MIRIX 列是 Mnemis 按 MIRIX 打印列名照抄导致的错位转录（真实类别见 B2 脚注）。Mnemis headline 93.9=k=30，对齐预算（k=10，"following Nemori"）为 93.3。MIRIX/MemU/EmergenceMem 行为"配置不可比"的公开最好成绩。

#### B4. [MemMachine] 博客 gpt-4.1-mini 块（judge gpt-4o-mini；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemMachine-memory[MemMachine] | 94.41 | 89.72 | 89.10 | 75.00 | 91.23 |
| MemMachine-agent[MemMachine] | 95.12 | 88.30 | 91.59 | 71.88 | 91.69 |
| Mem0 main/HEAD[MemMachine自跑] | — | — | — | — | 80.00 |

#### B5. [EverMemOS] Table 1 GPT-4.1-mini 块（judge=三 judge 均值；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemoryOS[EverMemOS] | 67.30 | 59.34 | 42.26 | 59.03 | 60.11 |
| Mem0[EverMemOS] | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU[EverMemOS] | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS[EverMemOS] | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep[EverMemOS] | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverMemOS[EverMemOS] | 96.67 | 91.84 | 89.72 | 76.04 | 93.05 |

#### B6. [Synthius] Table 3/4（Synthius 用 GPT-4.1-mini；judge GPT-4.1-mini；**1,813 题含 adversarial，overall 与其他组不可比**）

| 方法 | single-hop | multi-hop | temporal | open-domain | adversarial | overall（含 adv） |
|---|---|---|---|---|---|---|
| Synthius-Mem[Synthius] | 96.73 | 94.34 | 89.32 | 77.33 | 99.55 | 94.37 |
| Full-ctx[Synthius,Gemini3Flash] | 71.8 | 86.0 | 35.7 | 93.6 | 87.8 | 85.46 |
| Embedding-RAG[Synthius,Gemini3Flash] | 40.4 | 28.0 | 26.8 | 56.4 | 95.9 | 57.74 |
| Sliding-Window[Synthius] | 10.6 | 4.4 | 12.5 | 11.4 | 96.5 | 31.26 |
| Summarization[Synthius] | 6.9 | 3.2 | 14.3 | 3.7 | 97.7 | 27.86 |
| ENGRAM[Synthius转] | 79.90 | 79.79 | 72.68 | 72.92 | — | 77.55 |
| MemMachine[Synthius转] | 94.65 | 87.59 | 73.52 | 70.83 | — | 91.69 |

脚注：正文叙述与表格数字互相矛盾（"temporal 94.2 / multi-hop 85.7 / adversarial 99.9" ?），以表为准。其 Table 2 排行余下各行均为转引他家自报（MemOS 69.24、Mem0 66.9、MemPalace 60.3 等）。

### 1-C 组：其他 / 未披露 answerer

#### C1. [Hindsight] Table 4（judge GPT-OSS-120B；4 类；baseline 全转引 Backboard 官方页自报）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Backboard[转] | 89.36 | 75.00 | 91.90 | 91.20 | 90.00 |
| Memobase[转] | 70.92 | 46.88 | 85.05 | 77.17 | 75.78 |
| Zep[转] | 74.11 | 66.04 | 79.79 | 67.71 | 75.14 |
| Mem0-Graph[转] | 65.71 | 47.19 | 58.13 | 75.71 | 68.44 |
| Mem0[转] | 67.13 | 51.15 | 55.51 | 72.93 | 66.88 |
| LangMem[转] | 62.23 | 47.92 | 23.43 | 71.12 | 58.10 |
| OpenAI[转] | 63.79 | 42.92 | 21.71 | 62.29 | 52.90 |
| Hindsight-OSS-20B | 74.11 | 64.58 | 76.32 | 90.96 | 83.18 |
| Hindsight-OSS-120B | 76.79 | 62.50 | 79.44 | 93.68 | 85.67 |
| Hindsight-Gemini-3 | 86.17 | 70.83 | 83.80 | 95.12 | 89.61 |

脚注：转引列的 judge/answerer 各不相同（多为 Mem0 论文原数字或 Backboard 页数字），Hindsight 三列为本文实测（GPT-OSS-120B judge）。博客 v0.4.19 自家 harness 重跑：LoCoMo 92.0。

#### C2. [ByteRover] 论文 Table 3 + 博客（justifier Gemini 3.1 Pro；judge Gemini 3 Flash + Hindsight prompt；4 类 1,536 题）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| HonCho[ByteRover] | 93.2 | 84.0 | 88.2 | 77.1 | 89.9 |
| Hindsight[ByteRover]? | 86.2 | 70.8 | 83.8 | 95.1 | 89.6 |
| Memobase[ByteRover]? | 70.9 | 46.9 | 85.1 | 77.2 | 75.8 |
| Zep[ByteRover]? | 74.1 | 66.0 | 79.8 | 67.7 | 75.1 |
| Mem0[ByteRover]? | 67.1 | 51.2 | 55.5 | 72.9 | 66.9 |
| OpenAI[ByteRover]? | 63.8 | 42.9 | 21.7 | 62.3 | 52.9 |
| ByteRover(论文) | 97.5 | 93.3 | 97.8 | 85.9 | 96.1 |
| ByteRover(博客 Run2) | 95.4 | 85.1 | 94.4 | 77.2 | 92.2 |

脚注：? 标记列——caption 声称同 harness 重跑，但数字与各家自报逐位相同，"重跑"声明存疑。论文 96.1 与博客 92.2 是不同 justifier 的 run，per-category 不可混拼。

#### C3. [Memory-R1]（微调 LLaMA-3.1-8B / Qwen2.5-7B；judge 未披露；**1,307 题测试切分**；answer 限 5-6 词）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| RAG[MR1,LLaMA] | 13.81 | 20.48 | 4.65 | 15.96 | 13.62 |
| A-Mem[MR1,LLaMA] | 44.76 | 34.93 | 36.43 | 49.38 | 44.76 |
| Mem0[MR1,LLaMA] | 43.93 | 37.35 | 31.40 | 52.27 | 45.68 |
| MemoryOS[MR1,LLaMA] | 52.72 | 31.33 | 23.64 | 57.36 | 48.20 |
| Memory-SFT[MR1,LLaMA] | 56.90 | 37.35 | 54.65 | 63.27 | 58.76 |
| Memory-R1-GRPO[MR1,LLaMA] | 59.83 | 53.01 | 51.55 | 68.78 | 62.74 |
| Memory-R1-GRPO[MR1,Qwen] | 62.34 | 40.96 | 49.61 | 67.81 | 61.51 |

脚注：**62.74 属 LLaMA-8B GRPO**（非 Qwen）。Qwen 组 baseline overall：RAG 12.17 / A-Mem 40.78 / Mem0 53.30 / MemoryOS 51.26 / SFT 61.13 / PPO 59.53。Qwen 家族 scaling GRPO：3B 57.92 / 7B 57.46 / 14B 63.50（PPO-14B 65.26）。

#### C4. [Memobase] GitHub（answerer 默认 gpt-4o；judge gpt-4o-mini；⚠ mem0 fork 标签错位，按打印列名转录）

| 方法 | "Single-Hop"（实为 cat1，282 题） | "Multi-Hop"（实为 cat2？96 题档 ?） | "Temporal"（实为 cat2/321 题档） | "Open Domain"（841 题档） | overall |
|---|---|---|---|---|---|
| Memobase v0.0.32[Memobase] | 63.83 | 52.08 | 80.37 | 71.82 | 70.91 |
| Memobase v0.0.37[Memobase] | 70.92 | 46.88 | 85.05 | 77.17 | 75.78 |
| Zep*[Zep 团队补交 issue#101] | 74.11 | 66.04 | 79.79 | 67.71 | 75.14 |

脚注：其余行为转抄 Mem0 论文（与 A1 相同）。Memobase fork 的 category 映射 1=single_hop(282Q)/2=temporal(321Q)/3=multi_hop(96Q)/4=open_domain(841Q)，与官方命名相反，per-category 引用前必须按题数对齐。

#### C5. [Mem0-Plat] / [Mem0-26]（answerer=judge=GPT-5，新版宽 judge；与 [Mem0] 论文不可比）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Mem0 Platform Top-200[Mem0-Plat README] | 91.2（top-k 均值） | 91.3 | 92.0 | 72.7 | 92.5（Top-50 91.8） |
| Mem0 Platform（JSON run 20260406） | 92.27 | 93.26 | 92.83 | 76.04 | 91.56 |
| Mem0 新算法[Mem0-26 README] | — | — | — | — | 91.6（Old 同栈 71.4） |

#### C6. [LoCoMo] 官方论文（无 LLM judge，此处仅列 Human 供参照）与 [MemPalace]

- Human [LoCoMo]（F1 口径，5 类 7512 题）：single-hop 95.1 / multi-hop 85.8 / temporal 92.6 / open-domain 75.4 / adversarial 89.4 / overall 87.9 —— 常被当"人类上限"转引，注意它是 **F1 不是 judge accuracy**。
- MemPalace：LoCoMo Recall@10 60.3（raw）/ 88.9（hybrid）——**检索召回指标**，Synthius Table 2 里的 "MemPalace 60.3" 即此，不是 QA accuracy。

---

## 2. LoCoMo — F1（词面指标；各家实现不同：LoCoMo 官方=词干化 partial F1，A-Mem 系=set-based 去重 token F1，跨实现系统性不可比）

### 2-A 组：answerer = gpt-4o-mini 系

#### A1. [Mem0] 论文 Table 1（F1；⚠ 标签疑错位；无 overall）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| LoCoMo[转抄] | 25.02 | 12.04 | 18.41 | 40.36 | — |
| ReadAgent[转抄] | 9.15 | 5.31 | 12.60 | 9.67 | — |
| MemoryBank[转抄] | 5.00 | 5.56 | 9.68 | 6.61 | — |
| MemGPT[转抄] | 26.65 | 9.15 | 25.52 | 41.04 | — |
| A-Mem[转抄] | 27.02 | 12.14 | 45.85 | 44.65 | — |
| A-Mem*[重跑] | 20.76 | 9.22 | 35.40 | 33.34 | — |
| LangMem | 35.51 | 26.04 | 30.75 | 40.91 | — |
| Zep | 35.74 | 19.37 | 42.00 | 49.56 | — |
| OpenAI | 34.30 | 20.09 | 14.04 | 39.31 | — |
| Mem0 | 38.72 | 28.64 | 48.93 | 47.65 | — |
| Mem0ᵍ | 38.09 | 24.32 | 51.55 | 49.27 | — |

⚠ 注意：A-Mem 论文自己（下 C1）对同一组数字的标签是 MH 25.02 / Temp 18.41 / OD 12.04 / SH 40.36——与 Mem0 转抄的标签完全错开，直接证明 mem0 系标签错位。

#### A2. [Nemori] v4（F1；judge 无关；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Full-ctx | 53.1 | 35.4 | 44.1 | 24.5 | 46.2 |
| RAG-4096 | 22.2 | 18.6 | 19.5 | 19.0 | 20.8 |
| LangMem | 38.8 | 33.5 | 31.9 | 29.4 | 35.8 |
| Zep | 39.7 | 27.5 | 44.8 | 22.9 | 37.5 |
| Mem0 | 44.4 | 34.3 | 44.4 | 27.1 | 41.5 |
| A-MEM | 35.6 | 24.0 | 38.1 | 9.0 | 32.4 |
| MemoryOS | 43.7 | 35.2 | 38.5 | 26.0 | 39.9 |
| Nemori | 54.8 | 38.1 | 57.3 | 23.9 | 50.3 |

（F1=A-Mem 同款 set-based 去重 token F1。）

#### A3. [E-mem] Table 1 GPT-4o-mini 组（master gpt-4o-mini + assistant Qwen3-4B；自跑；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Long-Context | 46.68 | 29.23 | 25.97 | 16.87 | 37.31 |
| RAG | 52.45 | 27.50 | 46.07 | 23.23 | 44.73 |
| A-mem | 44.65 | 27.02 | 45.85 | 12.14 | 39.65 |
| Mem0 | 47.65 | 38.72 | 48.93 | 28.64 | 45.10 |
| MemoryOS | 48.62 | 35.27 | 41.15 | 20.02 | 42.84 |
| LightMem | 41.79 | 29.78 | 43.71 | 16.89 | 38.44 |
| GAM | 47.74 | 34.84 | 53.91 | 26.03 | 45.31 |
| E-mem | 59.23 | 42.64 | 59.82 | 24.89 | 54.17 |

v2 附录补充（overall F1）：GPT-4o Long-Context 48.75、GPT-5.1 Long-Context 53.31、AriGraph 45.76、HippoRAG 47.81。Qwen2.5-14B master 组 overall：E-mem 57.04 / GAM 50.41 / MemoryOS 40.28 / Long-Context 38.31 / RAG 38.27 / Mem0 36.04 / LightMem 31.39 / A-mem 28.98。E-mem adversarial 只在 Table 3 单列（F1 85.11–95.74 随 assistant 变化，非主表口径）。

#### A4. [ES-Mem] Table 1 GPT-4o-mini 组（自跑；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemGPT | 41.01 | 26.65 | 25.52 | 9.15 | 33.17 |
| MemoryBank | 6.61 | 5.00 | 9.68 | 5.56 | 6.89 |
| A-Mem | 44.65 | 27.02 | 45.85 | 12.14 | 39.65 |
| MemoryOS | 48.62 | 35.27 | 41.15 | 20.02 | 42.84 |
| Zep | 49.56 | 35.74 | 42.00 | 19.37 | 43.56 |
| Mem0 | 47.65 | 38.72 | 48.93 | 28.64 | 45.09 |
| LangMem | 40.91 | 35.51 | 30.75 | 26.04 | 36.87 |
| Nemori | 46.33 | 32.36 | 55.99 | 29.19 | 44.72 |
| LightMem | 47.64 | 32.11 | 53.79 | 26.14 | 44.73 |
| ES-Mem | 50.07 | 36.52 | 47.90 | 24.77 | 45.56 |

小模型组 overall F1：Qwen2.5-3B — ES-Mem 30.86 / H-Mem 24.13 / MemoryOS 23.69 / A-Mem 17.91；Llama3.2-3B — ES-Mem 30.46 / H-Mem 25.89 / A-Mem 24.83 / MemoryOS 23.56。

#### A5. gpt-4o-mini 系 overall-only F1 汇总

| 方法 | overall F1 | 来源与备注 |
|---|---|---|
| TiMem | 54.40 | [TiMem]（另 ROUGE-L 54.68；baseline：MemOS 45.02 / MemoryOS 45.36 / Mem0 42.52 / A-MEM 30.37 / MemoryBank 25.78） |
| LightMem(768,0.8) | 47.77 | [LightMem GitHub]（(512,0.7) 47.75 / (768,0.7) 47.18；Qwen 骨干组 38.22/43.18/44.78；无 per-category F1） |
| MemOS-1031 | 45.27 | [MemOS v4]（同表：Memobase 50.18 / Mem0 43.46 / Zep 41.23 / MemU 35.15 / Supermemory 34.87 / MIRIX 28.10） |
| MemOS-0630 | 44.42 | [MemOS v2]（per-cat：SH 45.55 / MH 35.57 / Temp 53.67 / OD 29.64；Mem0 47.26/35.24/45.40/27.80/43.46；LangMem 39.18；Zep 26.88；OpenAI 32.30） |
| DeltaMem-8B-RL | 50.72 | [DeltaMem]（per-cat 54.57/37.44/58.79/29.04；Mem0 42.55 / Zep 43.57⚠ / LangMem 36.88⚠ / LightMem 44.81 / A-Mem 29.96；标签警示同 §1-A7） |
| MemMachine-memory (4o-mini) | 22.00 | [MemMachine]（mem0 eval 代码 F1，0.2200；agent 22.10；量纲同为 %，但 answer 限 5-6 词 → 系统性偏低） |
| MIRIX（论文未印） | 24.16 | [MIRIX run 文件]（真实 cat ID：cat1 21.14 / cat2 29.66 / cat3 14.12 / cat4 24.25） |

### 2-B 组：answerer = gpt-4.1-mini 系

#### B1. [SimpleMem] Table 1 GPT-4.1-mini 组（F1；**Average=4 类宏平均**，非题数加权；无 judge）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall（宏平均） |
|---|---|---|---|---|---|
| LoCoMo(full-ctx) | 18.68 | 25.02 | 12.04 | 19.05 | 18.70 |
| ReadAgent | 9.18 | 6.48 | 5.31 | 7.66 | 7.16 |
| MemoryBank | 5.72 | 5.00 | 5.94 | 5.16 | 5.46 |
| MemGPT | 25.59 | 17.72 | 19.44 | 11.29 | 18.51 |
| A-Mem | 41.02 | 25.06 | 51.01 | 13.22 | 32.58 |
| LightMem | 33.79 | 24.96 | 20.55 | 19.21 | 24.63 |
| Mem0 | 41.30 | 30.14 | 48.91 | 16.43 | 34.20 |
| SimpleMem | 51.12 | 43.46 | 58.62 | 19.76 | 43.24 |

其他 backbone 组的 SimpleMem/Mem0 宏平均 F1：GPT-4o 39.06/36.09；Qwen3-Plus 37.49/35.85；Qwen2.5-1.5B 25.23/23.77；Qwen2.5-3B 17.98/13.03；Qwen3-1.7B 24.59/21.19；Qwen3-8B 33.45/25.80（per-cat 见原始抽取报告）。

#### B2. 其他 gpt-4.1-mini 系 F1

| 方法 | overall F1 | 来源 |
|---|---|---|
| MemMachine-memory (4.1-mini) | 27.85 | [MemMachine]（per-cat：MH 24.97 / Temp 25.49 / OD 14.29 / SH 31.27；agent 25.03） |
| Nemori (4.1-mini) | 52.1 | [Nemori v4]（per-cat：SH 55.7 / MH 40.8 / Temp 58.7 / OD 31.7；Full-ctx 53.3 / LangMem 47.6 / Mem0 43.5 / Zep 36.9 / A-MEM 39.4 / MemoryOS 39.9 / RAG 23.5） |
| Nemori V5 | 56.64 | [Nemori-V5]（0.5664 SH / 0.4338 MH / 0.5913 Temp / 0.2736 OD） |

### 2-C 组：其他 backbone / 特殊口径

#### C1. [A-Mem] 论文 Table 1（**5 类含 adversarial**，无 overall；repo=locomo10 1986 题；set-F1）

| 方法 | single-hop | multi-hop | temporal | open-domain | adversarial | overall |
|---|---|---|---|---|---|---|
| LoCoMo[A-Mem,4o-mini] | 40.36 | 25.02 | 18.41 | 12.04 | 69.23 | — |
| ReadAgent[A-Mem] | 9.67 | 9.15 | 12.60 | 5.31 | 9.81 | — |
| MemoryBank[A-Mem] | 6.61 | 5.00 | 9.68 | 5.56 | 7.36 | — |
| MemGPT[A-Mem] | 41.04 | 26.65 | 25.52 | 9.15 | 43.29 | — |
| A-Mem[A-Mem,4o-mini] | 44.65 | 27.02 | 45.85 | 12.14 | 50.03 | — |
| A-Mem[A-Mem,GPT-4o] | 48.43 | 32.86 | 39.41 | 17.10 | 36.35 | — |

脚注：cat5 评分 reference=诱导答案（非官方关键词规则）；本地 survey 曾把 27.02 错标为 single-hop——A-Mem 原表列序是 MH/Temp/OD/SH/Adv。

#### C2. [LoCoMo] 官方论文（规则 F1+词干化；5 类；50 对话 7,512 题，非 locomo10）

| 方法 | single-hop | multi-hop | temporal | open-domain | adversarial | overall |
|---|---|---|---|---|---|---|
| Human | 95.1 | 85.8 | 92.6 | 75.4 | 89.4 | 87.9 |
| GPT-3.5-turbo(4k) | 29.9 | 23.3 | 17.5 | 29.5 | 12.8 | 22.4 |
| GPT-4-turbo(4k) | 23.4 | 23.4 | 10.4 | 24.6 | 70.2 | 32.1 |
| GPT-3.5-16K@16K | 56.4 | 42.0 | 20.3 | 37.2 | 2.1 | 37.8 |
| RAG-Observation top5 | 44.3 | 30.6 | 41.9 | 40.2 | 44.7 | 41.4 |

#### C3. [Memory-R1]（1,307 题；微调骨干；answer 限 5-6 词）

overall F1：LLaMA-GRPO **45.02**（per-cat 35.73/35.65/49.86/47.42）、Qwen-GRPO 43.14、LLaMA-SFT 42.81、LLaMA-PPO 41.05、MemoryOS 35.04/34.64（LLaMA/Qwen）、Mem0 30.41/30.61、A-Mem 29.20/26.08、RAG 11.41/8.97。

#### C4. [Memobase]（answerer gpt-4o；⚠ 打印标签）

v0.0.37 F1：\"single_hop\"(cat1,282Q) 46.29 / \"temporal\"(cat2,321Q) 64.23 / \"multi_hop\"(cat3,96Q) 22.93 / \"open_domain\"(cat4,841Q) 51.55；无 overall。

---

## 3. LoCoMo — BLEU-1（%；nltk BLEU-1+smoothing 为主流实现；TiMem 报 ROUGE-L 不报 BLEU）

### 3-A 组：answerer = gpt-4o-mini 系

#### A1. [Mem0] 论文（⚠ 标签疑错位；无 overall）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| LoCoMo[转抄] | 19.75 | 11.16 | 14.77 | 29.05 | — |
| ReadAgent[转抄] | 6.48 | 5.12 | 8.87 | 7.66 | — |
| MemoryBank[转抄] | 4.77 | 5.94 | 6.99 | 5.16 | — |
| MemGPT[转抄] | 17.72 | 7.44 | 19.44 | 34.34 | — |
| A-Mem[转抄] | 20.09 | 12.00 | 36.67 | 37.06 | — |
| A-Mem*[重跑] | 14.90 | 8.81 | 31.08 | 27.58 | — |
| LangMem | 26.86 | 22.32 | 25.84 | 33.63 | — |
| Zep | 23.30 | 14.82 | 34.53 | 38.92 | — |
| OpenAI | 23.72 | 15.42 | 11.25 | 31.16 | — |
| Mem0 | 27.13 | 21.58 | 40.51 | 38.72 | — |
| Mem0ᵍ | 26.03 | 18.82 | 40.28 | 40.30 | — |

#### A2. [Nemori] v4（BLEU；真实标签）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Full-ctx | 44.7 | 26.1 | 36.1 | 17.2 | 37.8 |
| RAG-4096 | 18.6 | 11.7 | 15.7 | 13.5 | 16.4 |
| LangMem | 33.1 | 23.9 | 26.2 | 23.5 | 29.4 |
| Zep | 33.7 | 19.3 | 38.1 | 15.7 | 30.9 |
| Mem0 | 37.7 | 25.2 | 37.6 | 19.4 | 34.2 |
| A-MEM | 29.2 | 18.8 | 33.8 | 8.6 | 27.0 |
| MemoryOS | 37.7 | 24.1 | 27.5 | 19.2 | 31.9 |
| Nemori | 43.8 | 26.0 | 47.6 | 18.5 | 39.7 |

#### A3. [E-mem] GPT-4o-mini 组（BLEU-1）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| Long-Context | 37.54 | 22.76 | 19.42 | 13.70 | 29.57 |
| RAG | 47.94 | 20.13 | 40.35 | 17.94 | 39.40 |
| A-mem | 37.06 | 20.09 | 36.67 | 12.01 | 32.31 |
| Mem0 | 37.06 | 27.13 | 40.15 | 21.58 | 34.92 |
| MemoryOS | 42.99 | 25.22 | 30.76 | 16.52 | 35.54 |
| LightMem | 37.83 | 24.90 | 39.72 | 13.92 | 34.37 |
| GAM | 40.90 | 27.72 | 43.93 | 19.48 | 37.78 |
| E-mem | 50.58 | 34.38 | 44.57 | 18.15 | 44.34 |

Qwen2.5-14B 组 overall B1：E-mem 46.75 / GAM 43.48 / MemoryOS 34.72 / RAG 33.07 / Long-Context 31.89 / Mem0 29.91 / LightMem 27.15 / A-mem 24.47。

#### A4. [ES-Mem] GPT-4o-mini 组（BLEU-1）

| 方法 | single-hop | multi-hop | temporal | open-domain | overall |
|---|---|---|---|---|---|
| MemGPT | 34.34 | 17.72 | 19.44 | 7.44 | 26.51 |
| MemoryBank | 5.16 | 4.77 | 6.99 | 5.94 | 5.52 |
| A-Mem | 37.06 | 20.09 | 36.67 | 12.00 | 32.30 |
| MemoryOS | 42.99 | 25.22 | 30.76 | 16.52 | 35.53 |
| Zep | 38.92 | 23.30 | 34.53 | 14.82 | 33.64 |
| Mem0 | 38.72 | 27.13 | 40.51 | 21.58 | 35.91 |
| LangMem | 33.63 | 26.86 | 25.84 | 22.32 | 30.06 |
| Nemori | 38.52 | 23.36 | 48.25 | 22.31 | 36.75 |
| LightMem | 40.91 | 23.79 | 46.96 | 20.19 | 37.74 |
| ES-Mem | 45.21 | 27.75 | 42.12 | 18.51 | 39.70 |

#### A5. gpt-4o-mini 系 overall-only BLEU-1 汇总

| 方法 | overall B1 | 来源 |
|---|---|---|
| DeltaMem-8B-RL | 39.78 | [DeltaMem]（"BS"列=B1；per-cat 43.40/25.90/47.61/22.67；-4o-mini 39.81；Mem0 33.79 / Zep 33.64⚠ / LightMem 34.17 / LangMem 30.06⚠ / A-Mem 24.82） |
| MemOS-0630 | 36.88 | [MemOS v2]（per-cat：SH 38.32 / MH 26.71 / Temp 46.37 / OD 22.40；Mem0 35.97 / LangMem 32.59 / OpenAI 25.64 / Zep 21.55；v4 未报 B1） |
| LightMem(768,0.8) | 36.57 | [LightMem GitHub]（(512,0.7) 36.16 / (768,0.7) 36.12；Qwen 组 31.15/35.37/37.06；overall only） |
| MemMachine-memory (4o-mini) | 13.00 | [MemMachine]（0.1300；agent 13.16；4.1-mini memory 17.32 / agent 14.79） |
| MIRIX（论文未印） | 15.82 | [MIRIX run 文件]（真实 cat ID：15.26/20.18/10.42/14.98） |

### 3-B/C 组：其他 answerer

| 方法 | single-hop | multi-hop | temporal | open-domain | adversarial | overall |
|---|---|---|---|---|---|---|
| SimpleMem[SimpleMem,4.1-mini] | 43.53 | 38.82 | 50.10 | 18.04 | — | 37.62（宏平均） |
| Mem0[SimpleMem,4.1-mini] | 36.17 | 27.62 | 44.82 | 14.94 | — | 30.89 |
| A-Mem[SimpleMem,4.1-mini] | 36.99 | 17.32 | 44.75 | 14.75 | — | 28.45 |
| LightMem[SimpleMem,4.1-mini] | 29.66 | 21.66 | 18.39 | 17.68 | — | 21.85 |
| A-Mem[A-Mem论文,4o-mini] | 37.06 | 20.09 | 36.67 | 12.00 | 49.47 | — |
| Memobase v0.0.37[Memobase,gpt-4o]⚠ | 35.16(打印"single_hop"=cat1) | 17.58(打印"multi_hop"=cat3) | 47.58(打印"temporal"=cat2) | 40.89(打印"open_domain"=cat4) | — | — |
| Memory-R1-GRPO[MR1,LLaMA] | 27.70 | 30.77 | 38.27 | 41.24 | — | 37.51 |

SimpleMem 其他 backbone 宏平均 B1：GPT-4o 27.25（Temporal 格 20.57 原文异常低 ?）、Qwen3-Plus 31.69、Qwen3-8B 29.00。Memory-R1 Qwen-GRPO overall B1 36.44。

---

## 4. LongMemEval（LME-S，500 题）— per-type accuracy（%）

行序：SSU=single-session-user(70) / SSA=single-session-assistant(56) / SSP=single-session-preference(30) / MS=multi-session(133) / KU=knowledge-update(78) / TR=temporal-reasoning(133) / overall(500 题加权) / task-avg（仅宏平均来源）。各家 judge 不同（官方 GPT-4o / Mem0 GPT-5 / SimpleMem gpt-4.1-mini / Hindsight GPT-OSS-120B / ByteRover Gemini 3 Flash…），**跨表数字不可直接比**。

### 4-A 组：answerer = gpt-4o-mini

#### A1. [Zep] 论文 Table 2/3（judge GPT-4o，官方 per-type prompt）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[Zep] | 81.4 | 81.8 | 30.0 | 40.6 | 76.9 | 36.5 | 55.4 |
| Zep[Zep] | 92.9 | 75.0 | 53.3 | 47.4 | 74.4 | 54.1 | 63.8 |

#### A2. [Nemori]（judge：LME 官方模板，judge 模型未写 ?；v1=v4 同数）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[Nemori] | 78.6 | 89.3 | 6.7 | 38.3 | 78.2 | 42.1 | 55.0 |
| Nemori[Nemori] | 88.6 | 83.9 | 46.7 | 51.1 | 61.5 | 61.7 | 64.2 |

#### A3. [LightMem] Table 2/7 GPT-4o-mini 块（judge GPT-4o-mini + LME 官方 5 模板；弃 5 坏样本记错）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| FullText | 87.14 | 89.29 | 36.67 | 45.45 | 76.92 | 31.58 | 56.80 |
| NaiveRAG | 90.00 | 98.21 | 53.33 | 48.48 | 67.95 | 39.85 | 61.00 |
| LangMem | 60.00 | 46.43 | 60.00 | 20.30 | 66.67 | 15.79 | 37.20 |
| A-MEM | 92.86 | 96.43 | 46.67 | 48.87 | 64.11 | 47.36 | 62.60 |
| MemoryOS | 80.00 | 64.29 | 30.00 | 31.06 | 48.72 | 32.33 | 44.80 |
| Mem0 | 81.43 | 41.07 | 60.00 | 46.21 | 70.12 | 40.15 | 53.61 |
| LightMem(0.7,512) | 87.14 | 32.14 | 68.18 | 71.74 | 83.12 | 67.18 | 68.64 |

Qwen3-30B 骨干块 overall：LightMem(0.6,768) 70.20 ?（Table 8/9 同配置写 73.20）、A-MEM 65.20、NaiveRAG 60.80、FullText 54.80、LangMem 50.80、MemoryOS 49.60、Mem0 39.51。GLM-4.6 块 overall：LightMem 最高 73.20、NaiveRAG 73.20、A-MEM 70.60、LangMem 49.20、FullText 36.71。

#### A4. [MemOS] v4 Table 4（GPT-4o-mini；自制通用 judge 模板，1 run）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| MIRIX | 72.9 | 63.6 | 53.3 | 30.1 | 52.6 | 25.6 | 43.49 |
| Zep | 92.9 | 75.0 | 53.3 | 47.4 | 74.4 | 54.1 | 63.8 |
| Mem0 | 82.9 | 26.8 | 90.0 | 63.2 | 66.7 | 72.2 | 66.4 |
| Memobase | 92.9 | 23.2 | 80.1 | 66.9 | 89.7 | 75.9 | 72.4 |
| Supermemory | 85.7 | 58.9 | 89.9 | 52.6 | 55.1 | 44.4 | 58.4 |
| MemU | 67.1 | 19.6 | 76.7 | 42.1 | 41.0 | 17.3 | 38.4 |
| MemOS-1031 | 95.7 | 67.9 | 96.7 | 70.7 | 74.3 | 77.4 | 77.8 |

#### A5. [TiMem] Table 2（answer GPT-4o-mini；LME 官方 prompt）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| MemoryBank | 29.71 | 50.00 | 12.00 | 9.77 | 21.79 | 17.14 | 21.04 |
| A-MEM | 82.86 | 87.50 | 39.33 | 40.30 | 72.82 | 36.09 | 55.44 |
| Mem0 | 94.29 | 51.79 | 50.00 | 66.17 | 78.72 | 49.17 | 64.96 |
| MemoryOS | 81.14 | 78.18 | 51.33 | 44.81 | 56.15 | 53.38 | 58.04 |
| MemOS | 93.71 | 67.86 | 50.67 | 58.80 | 76.67 | 65.11 | 68.68 |
| TiMem | 95.71 | 82.14 | 63.33 | 70.83 | 86.16 | 68.42 | 76.88 |

#### A6. [ES-Mem] Table 2（回答 GPT-4o-mini；judge 官方 prompt、模型未写 ?）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full Text | 87.14 | 89.29 | 36.67 | 45.45 | 76.92 | 31.58 | 56.89 |
| Naive RAG | 90.00 | 98.21 | 53.33 | 48.48 | 67.95 | 39.85 | 60.90 |
| LangMem | 60.00 | 46.43 | 60.00 | 20.30 | 66.67 | 15.79 | 37.20 |
| A-Mem | 92.86 | 96.43 | 46.67 | 48.87 | 64.11 | 47.36 | 62.20 |
| MemoryOS | 80.00 | 64.29 | 30.00 | 31.06 | 48.72 | 32.33 | 44.66 |
| Mem0 | 81.43 | 41.07 | 60.00 | 46.21 | 70.12 | 40.15 | 53.51 |
| LightMem | 87.14 | 32.14 | 68.18 ? | 71.74 | 83.12 | 67.18 | 69.81 |
| ES-Mem | 84.29 | 82.14 | 73.33 | 66.17 | 78.21 | 64.66 | 72.40 |

（注意 ES-Mem 的 baseline per-type 与 LightMem A3 表逐字相同、overall 却略不同——两家口径沿袭同一来源。）

#### A7. [Mnemis] Table 2 GPT-4o-mini 块（**judge GPT-4.1-mini**；Mem0/Zep/Nemori 行为转录）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[自跑] | 78.6 | 89.3 | 6.7 | 38.3 | 78.2 | 42.1 | 55.0 |
| RAG[自跑] | 88.6 | 91.1 | 70.0 | 47.4 | 70.5 | 63.2 | 67.2 |
| Mem0[转] | 91.4 | 96.4 | 34.0 | 66.2 | 74.4 | 63.9 | 71.1 |
| Zep[转] | 92.9 | 75.0 | 53.3 | 47.4 | 74.4 | 54.1 | 63.2 |
| Nemori[转] | 88.6 | 83.9 | 46.7 | 51.1 | 61.5 | 61.7 | 64.2 |
| EMem-G[转] | 87.0 | 87.5 | 32.2 | 73.6 | 94.4 | 74.8 | 77.9 |
| Mnemis | 97.1 | 100.0 | 90.0 | 76.7 | 92.3 | 83.5 | 87.2 |

### 4-B 组：answerer = gpt-4.1-mini

#### B1. [SimpleMem] Table 2（judge gpt-4.1-mini；**Average=6 类宏平均**）

| 方法 | SSU | SSA | SSP | MS | KU | TR | task-avg（宏平均） |
|---|---|---|---|---|---|---|---|
| Full-ctx[SimpleMem] | 47.14 | 32.14 | 60.00 | 30.08 | 41.03 | 27.06 | 39.57 |
| Mem0[SimpleMem] | 87.14 | 48.21 | 63.33 | 50.37 | 69.23 | 40.60 | 59.81 |
| LightMem[SimpleMem] | 88.57 | 21.43 | 76.67 | 47.37 | 92.30 | 85.71 | 68.67 |
| SimpleMem[SimpleMem] | 85.71 | 75.00 | 76.67 | 60.92 | 79.48 | 83.46 | 76.87 |

GPT-4.1 回答组 task-avg：SimpleMem 83.97 / LightMem 76.86 / Mem0 58.51 / Full-ctx 56.72（per-type 见原始抽取）。

#### B2. [Mnemis] Table 2/9 GPT-4.1-mini 块（judge GPT-4.1-mini；多列为转录）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[自跑] | 85.7 | 98.2 | 16.7 | 51.1 | 76.9 | 60.2 | 65.6 |
| RAG[自跑] | 82.9 | 94.6 | 86.7 | 54.9 | 80.8 | 67.7 | 72.6 |
| PREMem[转] | 92.9 | 12.5 | 36.7 | 57.1 | 84.6 | 59.4 | 60.8 |
| Mem0[转] | 94.3 | 96.4 | 86.7 | 66.9 | 87.2 | 75.9 | 80.8 |
| Nemori[转] | 90.0 | 92.9 | 86.7 | 55.6 | 79.5 | 72.2 | 74.6 |
| EverMemOS[转] | 100.0 | 78.6 | 96.7 | 78.5 | 87.2 | 71.2 | 82.0 |
| EMem-G[转] | 94.8 | 87.5 | 50.0 | 82.6 | 94.4 | 83.7 | 84.9 |
| EmergenceMem[转] | 98.6 | 100.0 | 60.0 | 81.2 | 83.3 | 85.7 | 86.0 |
| Mnemis | 98.6 | 100.0 | 100.0 | 86.5 | 93.6 | 86.5 | 91.6 |

#### B3. [Nemori] 4.1-mini 块 与 [EverMemOS]（EverMemOS 自跑 GPT-4.1-mini，baseline 转录 MemOS leaderboard）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[Nemori] | 85.7 | 98.2 | 16.7 | 51.1 | 76.9 | 60.2 | 65.6 |
| Nemori[Nemori] | 90.0 | 92.9 | 86.7 | 55.6 | 79.5 | 72.2 | 74.6 |
| MemU[EverMemOS转] | 67.14 | 19.64 | 76.67 | 42.10 | 41.02 | 17.29 | 38.40 |
| Zep[EverMemOS转] | 92.90 | 75.00 | 53.30 | 47.40 | 74.40 | 54.10 | 63.80 |
| Mem0[EverMemOS转] | 82.86 | 26.78 | 90.00 | 63.15 | 66.67 | 72.18 | 66.40 |
| MemOS[EverMemOS转] | 95.71 | 67.86 | 96.67 | 70.67 | 74.26 | 77.44 | 77.80 |
| EverMemOS[EverMemOS] | 97.14 | 85.71 | 93.33 | 73.68 | 89.74 | 77.44 | 83.00 |

（EverMemOS 的 baseline 列与 MemOS Table 4（4-A4）基本同源，answerer 实为 GPT-4o-mini。）

### 4-C 组：其他 answerer / judge

#### C1. [Zep] GPT-4o 块 与 [TiMem] GPT-4o 块

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Full-ctx[Zep,GPT-4o] | 81.4 | 94.6 | 20.0 | 44.3 | 78.2 | 45.1 | 60.2 |
| Zep[Zep,GPT-4o] | 92.9 | 80.4 | 56.7 | 57.9 | 83.3 | 62.4 | 71.2 |
| MemOS[TiMem,GPT-4o] | 92.86 | 63.69 | 64.44 | 68.42 | 76.07 | 71.43 | 73.07 |
| Mem0[TiMem,GPT-4o] | 95.71 | 55.00 | 60.67 | 65.11 | 84.87 | 51.88 | 67.56 |
| A-MEM[TiMem,GPT-4o] | 90.00 | 83.21 | 56.67 | 45.26 | 87.18 | 46.77 | 63.40 |
| MemoryOS[TiMem,GPT-4o] | 82.86 | 80.00 | 53.33 | 51.13 | 60.00 | 54.59 | 61.20 |
| TiMem[TiMem,GPT-4o] | 96.28 | 85.71 | 55.33 | 72.78 | 87.69 | 73.38 | 78.96 |

#### C2. [Hindsight] Table 3（Hindsight 列=GPT-OSS-120B judge；ᵃ 列=转引 Supermemory 技报，GPT-4o judge）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| FC(GPT-4o)ᵃ | 81.4 | 94.6 | 20.0 | 44.3 | 78.2 | 45.1 | 60.2 |
| FC(OSS-20B) | 38.6 | 80.4 | 20.0 | 21.1 | 60.3 | 31.6 | 39.0 |
| Zep(GPT-4o)ᵃ | 92.9 | 80.4 | 56.7 | 57.9 | 83.3 | 62.4 | 71.2 |
| Supermemory(GPT-4o)ᵃ | 97.1 | 96.4 | 70.0 | 71.4 | 88.5 | 76.7 | 81.6 |
| Supermemory(GPT-5)ᵃ | 97.1 | 100.0 | 76.7 | 75.2 | 87.2 | 81.2 | 84.6 |
| Supermemory(Gemini-3)ᵃ | 98.6 | 98.2 | 70.0 | 76.7 | 89.7 | 82.0 | 85.2 |
| Hindsight(OSS-20B) | 95.7 | 94.6 | 66.7 | 79.7 | 84.6 | 79.7 | 83.6 |
| Hindsight(OSS-120B) | 100.0 | 98.2 | 86.7 | 81.2 | 92.3 | 85.7 | 89.0 |
| Hindsight(Gemini-3) | 97.1 | 96.4 | 80.0 | 87.2 | 94.9 | 91.0 | 91.4 |

博客 v0.4.19 重跑：LME 94.6（仅图，无 per-type 文本值）。

#### C3. [ByteRover] Table 4（judge Gemini 3 Flash；† = 转引原论文、骨干与 judge 不同）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Chronos† | 94.3 | 100.0 | 80.0 | 91.7 | 96.2 | 90.2 | 92.6 |
| Hindsight | 97.1 | 96.4 | 80.0 | 87.2 | 94.9 | 91.0 | 91.4 |
| HonCho | 94.3 | 96.4 | 90.0 | 85.0 | 94.9 | 88.7 | 90.4 |
| SmartSearch† | 100.0 | 85.7 | 96.7 | 84.2 | 93.6 | 82.7 | 88.4 |
| Memora† | 98.6 | 78.6 | 83.3 | 78.2 | 97.4 | 89.5 | 87.4 |
| TiMem†? | 96.3 | 85.7 | 55.3 | 72.8 | 87.7 | 73.4 | 79.0 |
| Zep† | 92.9 | 80.4 | 56.7 | 57.9 | 83.3 | 62.4 | 71.2 |
| Full-ctx† | 81.4 | 94.6 | 20.0 | 44.3 | 78.2 | 45.1 | 60.2 |
| ByteRover | 98.6 | 98.2 | 96.7 | 84.2 | 98.7 | 91.7 | 92.8 |

? ByteRover 转引的 TiMem† 行与 TiMem 论文自报（4-A5）不一致，来源不明。博客 Run 2（Gemini 3.1 Pro）92.2。

#### C4. [Mem0-Plat]（answerer=judge=GPT-5，宽 judge）与 [MemMachine repo]（answerer gpt-5-mini，judge gpt-4o，非正式 ?）

| 方法 | SSU | SSA | SSP | MS | KU | TR | overall |
|---|---|---|---|---|---|---|---|
| Mem0-Plat Top-200 | 98.6 | 98.2 | 96.7 | 88.0 | 93.6 | 97.0 | 94.4 |
| Mem0-Plat Top-50 | 98.6 | 98.2 | 93.3 | 93.2 | 93.6 | 94.0 | 94.8 |
| Mem0-Plat JSON run | 97.14 | 100.0 | 96.67 | 86.47 | 96.15 | 93.23 | 93.4 |
| Mem0-OSS 抽取=GPT-5 | 95.7 | 92.9 | 93.3 | 83.5 | 91.0 | 94.7 | 91.0 |
| MemMachine[repo README ?] | 98.57 | 100.00 | 96.67 | 93.23 | 93.59 | 96.24 | 95.80 |

Mem0 OSS 其他抽取模型 overall：GPT-OSS-120B 89.8 / Llama-4-Maverick 88.6 / Gemma-4-31B 88.6。Mem0 博客另有一版 4 类分解（94.3/97.1/100.0/70.7）与 repo README 矛盾且无配置，勿用 ?。

#### C5. [Memory-R1] Table 4（零样本；微调骨干；judge ?；overall=六类宏平均）

| 方法 | SSU | "OD"(?=SSA) | SSP | MS | KU | TR | task-avg（宏平均） |
|---|---|---|---|---|---|---|---|
| LLaMA-GRPO | 87.1 | 33.9 | 63.3 | 57.9 | 52.6 | 45.1 | 55.4 |
| Qwen-GRPO | 91.4 | 26.8 | 66.7 | 63.2 | 65.4 | 41.4 | 57.8 |
| A-Mem[MR1,LLaMA] | — | — | — | — | — | — | 54.20 |
| Mem0[MR1,LLaMA] | — | — | — | — | — | — | 41.20 |

#### C6. [OMEGA]（GPT-4.1 回答+判分；自定义 5 类归并）与 [LME-off] / [MemPalace]

| 方法 | SSU+SSA 合并（"Single-Session Recall"） | SSP | MS | KU | TR | overall（raw 计数） | task-avg（自报 headline） |
|---|---|---|---|---|---|---|---|
| OMEGA[omegamax] | 99（125/126） | 100（30/30） | 83（111/133） | 96（75/78） | 94（125/133） | 93.2（466/500） | 95.4 ?（复算仅 94.56） |

- [LME-off] 官方论文只有 overall：GPT-4o S=60.6（CoN 64.0，Oracle 87.0/92.4）、Llama-3.1-70B 33.4、Llama-3.1-8B 45.4、Phi-3-14B 38.0、Phi-3.5-4B 34.2；商业系统 97 题子集 pilot：ChatGPT(GPT-4o) 57.7、Coze 33.0。**无任何 baseline 的 6 类 per-type 表**。
- [MemPalace]：96.6 = **Recall@5（检索，无 LLM）**，端到端 QA 仅 ~67.2 ?（回答/judge 未披露）。不可入本节任何 QA 列。
- Zep 官网营销页宣称 LME 90.2%（无 per-type、无方法学）?，与论文 71.2 不是同一套结果。

---

## 5. 同方法跨论文数字离散度（"自报数字不可信"的直接证据）

LoCoMo overall LLM-Judge（除注明外均为 cat1-4 口径）：

| 方法 | 各来源报的数字（值 [来源, answerer]） | 最大差值 |
|---|---|---|
| **Mem0** | 66.88 [Mem0自报, 4o-mini]；61.3 [Nemori, 4o-mini]；66.3 [Nemori, 4.1-mini]；61.69(api) / **36.49(OSS)** [LightMem, 4o-mini]；64.57 [MemOS, 4o-mini]；57.79 [TiMem, 4o-mini]；61.00 [EverMemOS, 4o-mini]；64.20 [EverMemOS, 4.1-mini]；57.86 [DeltaMem, 4o-mini]；62.47 [MIRIX, 4.1-mini]；**80.00** [MemMachine, 4.1-mini, main/HEAD]；45.68/53.30 [Memory-R1, 1307题]；**91.6** [Mem0-26]；**92.5** [Mem0-Plat, GPT-5] | 同 4o-mini 口径 36.49–66.88 = **Δ30.4**（剔 OSS 后 Δ9.1）；全口径 36.49–92.5 = **Δ56.0** |
| **Zep** | 65.99 [Mem0]；58.5/61.6 [Nemori 4o/4.1]；41.62 [MemOS-v2]；59.22 [MemOS-v4]；75.14 [Zep*补交, Memobase/MIRIX 转]；79.09 [MIRIX, 官方实现]；49.09 [MIRIX, mem0 实现, 疑bug]；81.06/85.22 [EverMemOS 4o/4.1]；80.32 [Zep-repo 自跑]；94.7 [Zep官网 ?] | 41.62–85.22 = **Δ43.6**（同一 MemOS 两版自己就差 17.6） |
| **A-Mem** | 48.38 [Mem0 重跑]；52.5/61.4 [Nemori 4o/4.1]；64.16 [LightMem]；51.29 [TiMem]；48.38 [DeltaMem 转]；44.76/40.78 [Memory-R1, 1307题] | 44.76–64.16 = **Δ19.4** |
| **LangMem** | 58.10 [Mem0]；51.3/73.4 [Nemori 4o/4.1]；57.20 [LightMem]；55.76 [MemOS-v2]；58.11 [DeltaMem]；78.05 [MIRIX, 4.1-mini] | 51.3–78.05 = **Δ26.8**（answerer 换挡是主因） |
| **MemoryOS** | 54.5/60.6 [Nemori 4o/4.1]；58.25(eval)/54.87(pypi) [LightMem]；60.79 [TiMem]；54.70/60.11 [EverMemOS 4o/4.1]；48.20/51.26 [Memory-R1] | 48.20–60.79 = **Δ12.6** |
| **MemOS** | 73.31 [自报 v2-0630]；75.80 [自报 v4-1031]；69.24 [TiMem]；73.3 [Mnemis 转]；75.87/80.76 [EverMemOS 4o/4.1] | 69.24–80.76 = **Δ11.5** |
| **MIRIX** | 85.38 [自报, 4.1-mini]；64.33 [MemOS, 4o-mini]；85.4 [Mnemis 转, per-cat 错位] | 64.33–85.38 = **Δ21.1** |
| **Full-context** | 72.90 [Mem0]；72.3/80.6 [Nemori 4o/4.1]；71.83(论文)/73.83(README) [LightMem]；71.58 [MemOS-v2]；77.51/87.52 [MIRIX 4o/4.1]；85.46 [Synthius, 含adv, Gemini] | 71.58–87.52 = **Δ16.0**（LightMem 自己两处就差 2.0） |
| **Nemori** | 74.4 [自报 v1]；73.0 [自报 v4]；80.8 [自报 v4, 4.1-mini]；83.05 [自报 GitHub V5]；74.4/79.5 [Mnemis 转] | 自家四个版本 73.0–83.05 = **Δ10.1** |
| **LightMem** | 71.95/72.99 [自报, 4o-mini]；66.43 [DeltaMem] | Δ6.6 |

LoCoMo overall F1（词面）：

| 方法 | 各来源 | 最大差值 |
|---|---|---|
| **Mem0** | ~无自报 overall [Mem0]；41.5/43.5 [Nemori]；45.10 [E-mem]；45.09 [ES-Mem]；34.20 [SimpleMem, 宏平均]；42.52 [TiMem]；42.55 [DeltaMem]；43.46 [MemOS]；30.41/30.61 [Memory-R1]；27.85 [MemMachine, 5-6词限制] | 27.85–45.10 = **Δ17.3** |
| **LightMem** | 47.75/47.77 [自报 GitHub]；44.81 [DeltaMem]；38.44 [E-mem]；44.73 [ES-Mem]；24.63 [SimpleMem, 宏平均] | 24.63–47.77 = **Δ23.1** |
| **A-Mem** | 无 overall [A-Mem 自报]；32.4/39.4 [Nemori]；39.65 [E-mem]；39.65 [ES-Mem]；32.58 [SimpleMem]；30.37 [TiMem]；29.96 [DeltaMem]；29.20/26.08 [Memory-R1] | 26.08–39.65 = **Δ13.6** |

LongMemEval overall：

| 方法 | 各来源 | 最大差值 |
|---|---|---|
| **Mem0** | 53.61 [LightMem]；64.96 [TiMem]；66.4 [MemOS]；71.1/80.8 [Mnemis 4o/4.1 转]；59.81 [SimpleMem, 宏平均]；53.51 [ES-Mem]；66.40 [EverMemOS 转]；41.20/46.80 [Memory-R1 宏平均]；**94.4/94.8** [Mem0-Plat, GPT-5]；93.4 [Mem0-26 博客] | 41.20–94.8 = **Δ53.6**（同 4o-mini 系 53.51–66.4 = Δ12.9） |
| **Zep** | 63.8/71.2 [自报 4o-mini/4o]；63.2 [Mnemis 转]；63.80 [EverMemOS 转]；71.2 [ByteRover†/Hindsightᵃ 转]；90.2 [官网 ?] | 论文口径 Δ7.4；含官网 Δ26.4 |
| **Full-context** | 55.4/60.2 [Zep]；55.0/65.6 [Nemori]；56.80/54.80/36.71 [LightMem GPT/Qwen/GLM]；56.89 [ES-Mem]；39.57/56.72 [SimpleMem 宏]；39.0 [Hindsight OSS-20B]；60.2 [ByteRover†] | 36.71–65.6 = **Δ28.9** |
| **TiMem** | 76.88/78.96 [自报 4o-mini/4o]；79.0 [ByteRover† 转, per-type 与自报对不上 ?] | Δ2.1（但转引 per-type 不一致） |
| **LightMem** | 68.64 [自报]；68.67 [SimpleMem]；69.81 [ES-Mem] | Δ1.2（罕见的一致案例） |

**结论**：同一方法、同一 benchmark 的自报/转报数字最大可差 30–56 点（Mem0 LoCoMo Δ56.0、Mem0 LME Δ53.6、Zep LoCoMo Δ43.6）；即便固定 answerer 档位（gpt-4o-mini 系），Mem0 仍有 Δ30.4（OSS vs API 实现）、A-Mem Δ15.8、Zep Δ39.4（41.62 vs 81.06）。加上 mem0 系类别标签错位（§1 警示、A-Mem vs Mem0 对同一组 F1 的标签互斥），**跨论文拼表不具备可比性，统一协议重跑是唯一可靠对比方式**——这是我们论文 motivation 的直接证据。

---

## 6. 与我们统一协议的映射（gpt-5.4-mini answerer + 宽松 0/1 judge + cat1-4，1540 题）

### 6.1 口径最接近的组

我们的协议 =「mini 档 answerer + Mem0 式宽松二元 judge + cat1-4 全 1540 题 + overall 题数加权」。最接近的是：

1. **1-A 组（gpt-4o-mini 系）中 judge=gpt-4o-mini 的子表**：A1 [Mem0]、A2 [Nemori]、A3 [LightMem]、A4 [MemOS]、A5 [TiMem]、A8 [MemMachine]——answerer 档位、judge 家族、题目子集、指标定义四项全对齐，仅 answerer 具体型号不同（gpt-4o-mini vs 我们的 gpt-5.4-mini）。
2. **1-B 组（gpt-4.1-mini 系）**：B1 [Nemori]、B4 [MemMachine]、B5 [EverMemOS]（judge 为三模型均值，稍弱）、B2 [MIRIX]（judge 自相矛盾但代码=gpt-4o-mini）、B3 [Mnemis]（judge=4.1-mini，宽严接近）。

### 6.2 可进我们对比表的列（须加脚注"原论文自报，answerer/judge 见注"）

| 引用列 | 数字（LoCoMo overall J） | 必须的脚注 |
|---|---|---|
| Mem0 [Mem0 自报] | 66.88 | answerer/judge=gpt-4o-mini；per-category 勿引（标签错位） |
| Nemori [Nemori v4 自报] | 73.0（4o-mini）/ 80.8（4.1-mini） | judge=gpt-4o-mini（论文文字） |
| LightMem [自报] | 72.99（最好配置 768,0.8） | 论文报 offline-update 后 ACC |
| MemOS-1031 [自报] | 75.80 | judge 3 次判分均值；勿引成 MemOS-0630（那是 73.31） |
| TiMem [自报] | 75.30 | judge=gpt-4o-mini + Mem0 prompt |
| MemMachine [自报] | 87.47（4o-mini）/ 91.23–91.69（4.1-mini） | Mem0 eval 代码原样；构建 LLM 未披露 |
| MIRIX [自报] | 85.38 | answerer=4.1-mini；judge 正文/代码矛盾；per-category 必须按真实类别重映射后再引 |
| EverMemOS [自报] | 86.76（4o-mini）/ 93.05（4.1-mini） | judge=三模型均值，比单 judge 口径略不同 |
| Mnemis [自报] | 93.3（k=10）/ 93.9（k=30） | judge=GPT-4.1-mini；对齐检索预算引 93.3 |
| DeltaMem [自报] | 75.13（8B-RL） | judge 未披露；baseline 行有搬运错位，只引其自身 |
| Zep [getzep repo] | 80.32 | gpt-4o-mini 回答+判分 10 runs；论文无 LoCoMo 自报，勿引 66.0 或 94.7 |
| LME 侧同理 | Nemori 64.2/74.6；LightMem 68.64；MemOS-1031 77.8；TiMem 76.88；Mnemis 87.2/91.6；EverMemOS 83.0；SimpleMem 76.87（注明宏平均）；Zep 63.8/71.2（注明 GPT-4o judge） | 每行必须带 judge 注脚 |

**per-category 引用规则**：只引"真实标签"来源（Nemori/TiMem/LightMem/MemMachine/Mnemis/EverMemOS/Hindsight）；mem0 谱系来源（[Mem0]、[MIRIX]、[Memobase]、[DeltaMem]）的 per-category 必须先按题数（282/321/96/841）核对再重映射，或干脆只引 overall。

### 6.3 完全不能放进对比表的数字

| 数字 | 原因 |
|---|---|
| Hindsight 89.61 / ByteRover 96.1 / HonCho 89.9 / Backboard 90.00 | judge 换成 GPT-OSS-120B / Gemini 3 Flash，非 GPT-mini 系宽松 judge；Hindsight/ByteRover 的 baseline 列还是转引自报，"同条件"声明存疑 |
| Synthius-Mem 94.37 | 1,813 题**含 adversarial 442 题**，分母与 cat1-4 口径不同；overall 被 adversarial 99.55 拉高 |
| Memory-R1 62.74/61.51 | 只测 1,307 题子集（1:1:8 切分），且 answer 硬限 5-6 词 |
| Mem0 91.6 [Mem0-26] / 92.5 [Mem0-Plat] | answerer+judge=GPT-5，judge prompt 更宽（14 天容忍等），配置代际完全不同 |
| MemPalace 96.6 | Recall@5 检索指标，不是 QA accuracy |
| OMEGA 95.4 | GPT-4.1 既回答又判分；headline 是 5 类不加权均值且复算不出（应 94.56/93.2） |
| LoCoMo 官方表全部 / A-Mem 论文表全部 | F1 规则口径 + 含 adversarial + 题目集不同（7512 / 1986 题）；且 A-Mem cat5 用诱导答案当 reference |
| SimpleMem LoCoMo 43.24 | 无 LLM judge（纯 F1），Average=宏平均，题数口径自述含混（"1,986 questions...four types" ?）——只能进 F1 段落，不能进 judge 表 |
| E-mem 54.17 / ES-Mem 45.56 | 同上，纯 F1 口径，无 judge |
| Zep 官网 90.2/94.7、Mem0 博客 4 类 LME 分解、MemMachine repo LME 95.8 | 无方法学披露 / 与有计数版本矛盾 / 非正式自报 |
| LightMem 论文两处内部冲突值（67.78 vs 67.68、70.20 vs 73.20）、LightMem FullText 71.83 vs 73.83 | 引用时选一并注明另一处 |

### 6.4 给论文写作的操作建议

1. 主对比表用**我们统一重跑**的数字；自报数字只进"Reported (not comparable)"斜体列或附录表，每格带来源+judge 脚注。
2. Motivation 段直接引 §5：同方法自报差值最大 Δ56（Mem0）、Δ43.6（Zep），以及 A-Mem↔Mem0 对同一组 F1 数字的类别标签互斥，论证"published numbers are not mutually comparable"。
3. 若需引"最强已发表系统"：LoCoMo mini 档 judge 口径下是 Mnemis 93.3/93.9 与 EverMemOS 93.05、MemMachine 91.7；LME 下是 Mnemis 91.6、Mem0-Plat 94.4（后者 GPT-5 口径，只可注脚提及）。

---

*来源明细与逐表出处见同目录原始抽取报告（mem0 / nemori / lightmem / simplemem / memmachine-memos / mnemis-mirix / hindsight-emem / locomo-official-amem / longmemeval-side / rest 各节），以及 `docs/benchmark_survey.md`、`docs/experiment_settings_survey.md`（注意其中已知错误已在各抽取报告勘误）。*
