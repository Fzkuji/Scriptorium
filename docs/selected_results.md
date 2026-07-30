# 结果选用策略（Selected Results Policy）

更新：2026-07-02。依据 [published_results_matrix.md](published_results_matrix.md) 的 129 组交叉数据。

## 核心规则

1. **优先引用第三方重跑数字**（不引自报）——交叉矩阵证明自报系统性偏高（Mem0 +5，MIRIX +21，Zep +9~17）
2. **被多家重跑且聚合的 → 直接引用重跑中位数/区间**，标注来源
3. **只有自报、无人独立复现的 → 我们自己复现**（用统一协议；我们的数字就是它的首个第三方复现）
4. **重跑数字发散的（A-Mem、Zep、MIRIX）→ 我们复现当仲裁**
5. Full-context 是跨源校准尺（gpt-4o-mini 系稳定在 71.6~73.8）——我们协议跑出的 Full-context 落进该区间即证明口径可比

## 分桶（LoCoMo Judge）

### 桶 A：引用第三方重跑（不自己跑，除非审稿人质疑）

| 方法 | 重跑来源数 | 重跑区间 | 引用值建议 |
|---|---|---|---|
| LangMem | 4 家 | 51.3~58.1（MIRIX 78.1 为强模型离群）| ~55 区间 |
| MemoryOS | 4 家 | 54.5~60.8 | ~57 区间 |
| MemOS | 2 家 | 69.2~80.8 | ~73 区间 |
| MemU | 2 家 | 56.6~66.7 | ~61 区间 |
| RAG 基线 | 多家 | 30.2~68.2（chunk 配置差异大）| 按配置注明 |

### 桶 B：我们复现（校准 + 仲裁）

| 方法 | 原因 | 适配器 |
|---|---|---|
| Full-context | 校准尺（预期 72~74）| run_naive.py |
| BM25 | 下界 | run_naive.py |
| Mem0 | 主对标；6 家重跑 57.8~64.6，我们出仲裁数（预期 ~61）| run_mem0.py |
| A-Mem | 重跑最飘（44.8~64.2），仲裁 | run_amem.py |
| Zep | 重跑飘（41.6~85.2），仲裁 | run_zep.py |
| MIRIX | 自报 85.4 vs 唯一重跑 64.3（Δ21），仲裁 | run_mirix.py |

### 桶 C：我们复现（无人复现过——首个第三方复现，贡献点）

| 方法 | 自报 | 适配器 |
|---|---|---|
| Nemori | 73.0 / 80.8 | run_nemori.py |
| LightMem | 72.0~73.0 | run_lightmem.py — ✓ 已复现验证（s0 4o-mini 重判 73.0 ≈ 自报），主表直接采用其自报 (512,0.7)=72.0，不再补跑 |
| EverMemOS | 86.8 / 93.1 | run_evermemos.py |
| MemMachine | 87.5~91.7 | run_memmachine.py |
| Hindsight | 89.6 | run_hindsight.py |
| SimpleMem | F1 43.2（无 J 自报）| run_simplemem.py |
| E-Mem | 78.0~85.3（Mnemis 转录）| run_emem.py |

### 桶 D：只能引用 + 标注不可比（无代码）

Mnemis（93.3）、TiMem（75.3，PG 依赖判不可行）、Memory-R1（62.7）、Synthius-Mem（94.4）、DeltaMem、ByteRover（96.1）、MemPalace/OMEGA（LongMemEval 榜）、HORMA（51.6，方法最近竞品，related work 重点定性对比）。

## 执行阶段

- **Phase 1（先跑，出核心表）**：NativeMem + 桶 B 六个 —— LoCoMo sample 0 → 全 10 sample
- **Phase 2（补齐贡献点）**：桶 C 七个 —— 至少 LoCoMo 全量 + LongMemEval-S
- **Phase 3（按需）**：审稿人质疑桶 A 哪个就补哪个（适配器都在）

## 指标呈现（每方法一行全指标）

主表列：LLM-Judge（GPT-5.5, 0/1）| F1（set-based）| BLEU-1 | 来源标记（实测 / 引用†）
附录：f1_official、ROUGE、EM、per-category 全展开、自报 vs 重跑对照表（交叉矩阵精简版）

## 模型配置（正式实验）

- Builder/Answerer：gpt-5.4-mini（订阅，档位对齐文献的 4o-mini/4.1-mini）
- Judge：gpt-5.5（订阅）
- 消融：builder 换 deepseek-v4-flash 验证模型不敏感
