# Baseline 适配器

每个 `run_*.py` 只做**建库 + 检索**，产出统一 `questions.json`；答题/判分统一由
`src/evaluation/evaluate.py` 完成。详细文档见 `docs/baseline_adapters.md`。

## 适配器清单

| 适配器 | 系统 | 备注 |
| --- | --- | --- |
| `run_naive.py` | full-context / BM25 | `--method` 必填，两个朴素基线 |
| `run_mem0.py` | Mem0（ECAI 2025） | |
| `run_amem.py` | A-Mem（NeurIPS 2025） | 构建最慢（每 turn 2 次 LLM） |
| `run_lightmem.py` | LightMem（ICLR 2026） | 可选 `--k` / `--qdrant-dir` |
| `run_nemori.py` | Nemori（arXiv 2508.03341） | |
| `run_zep.py` | Zep / Graphiti（arXiv 2501.13956） | 嵌入式 Kuzu，无 Neo4j |
| `run_memoryos.py` | MemoryOS（arXiv 2506.06326） | |
| `run_memos.py` | MemOS（arXiv 2507.03724） | general_text 开源子集，口径须注明 |
| `run_simplemem.py` | SimpleMem（aiming-lab） | 检索期每题 4–8 次 LLM |
| `run_hindsight.py` | Hindsight（v0.8.4） | venv `.venv-hindsight`，自动 re-exec |
| `run_memmachine.py` | MemMachine | venv `third_party/memmachine_venv`；未启用 LLM 检索 agent |
| `run_mirix.py` | MIRIX（arXiv 2507.07957） | venv `third_party/mirix-venv` + 本地 server 子进程（`_mirix_server_shim.py`） |
| `run_emem.py` | E-Mem（ICML 2026，arXiv 2601.21714） | 块路由 + 每块 LLM 证据抽取，k≤6，非 top-k 语义 |
| `run_evermemos.py` | EverMemOS / EverOS（arXiv 2601.02163） | venv `third_party/evermemos_venv` + 本地 uvicorn 子进程（`_evermemos_server_shim.py`） |
| `run_nativemem.py` | NativeMem（我们的系统） | |

不可行（无适配器，见 `docs/baseline_adapters.md` 小节）：Mnemis（构建端未开源）、
Memobase（client-server + PG/pgvector/Redis 硬依赖）、TiMem（PostgreSQL 硬依赖 + 裸 SQL 检索）。

## 契约

CLI（所有适配器一致）：

```
--sample N            # LoCoMo 对话索引（默认 0）
--output PATH         # 输出 JSON（必填）
--max-sessions N      # 开发用：只 ingest 前 N 个 session
--questions-limit N   # 开发用：只处理前 N 个问题
```

额外：`run_naive.py` 必填 `--method {full_context,bm25}`；`run_lightmem.py` 可选 `--k` / `--qdrant-dir`。

输出：JSON 数组，首条 `_build_stats`（build_time_s / num_memories / notes + 效率四键），其后每题：

```json
{"question_id": "s{sample}_q{idx}", "question": "...", "gold": "...", "category": 2,
 "memories": [{"text": "...", "date": "..."}],
 "retrieval": {"latency_s": 0.01, "k": 20, "calls": 0, "tokens_in": 0, "tokens_out": 0}}
```

效率字段（LightMem 口径，由 `_usage_tracker.py` 在进程内 patch OpenAI SDK 统计）：

- `_build_stats` 新增四键：`build_calls`（构建期 LLM 调用次数）、`build_tokens_in`（prompt
  tokens）、`build_tokens_out`（completion tokens）、`build_llm_time_s`（LLM 调用累计墙钟秒）。
  跳过构建（如 `run_nativemem.py --memory-dir` 复用已有库）时四键为 `null`。
- 每题 `retrieval` 新增三键：`calls` / `tokens_in` / `tokens_out` —— 该题检索阶段的 LLM
  用量。纯 embedding 检索的系统恒为 0（正常）；SimpleMem、E-Mem、NativeMem 等检索期调
  LLM 的系统 >0。
- 只统计 memory 系统自身成本；answerer/judge（`src/evaluation/evaluate.py`）成本单列，不算入。
- 已知盲区：MIRIX / EverMemOS 的 LLM 调用发生在独立 venv 的 server 子进程内，主进程
  tracker 拦截不到，计数为 0（需在 shim 子进程内安装 tracker 才能统计真实用量）。

约定：top-20 检索；builder LLM = `deepseek-v4-flash`@阿里云（`BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY`
环境变量可覆盖）；embedder = 本地 all-MiniLM-L6-v2（384d）；**不修改 third_party 任何文件**。

## 命令速查

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"

# 冒烟（约定：2 sessions × 3 题）
python3 src/adapters/run_naive.py    --sample 0 --method bm25 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_bm25.json
python3 src/adapters/run_mem0.py     --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_mem0.json
python3 src/adapters/run_amem.py     --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_amem.json
python3 src/adapters/run_lightmem.py --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_lightmem.json
python3 src/adapters/run_nemori.py   --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_nemori.json
python3 src/adapters/run_emem.py     --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_emem.json
python3 src/adapters/run_evermemos.py --sample 0 --max-sessions 2 --questions-limit 3 --output /tmp/smoke_evermemos.json

# 全量单 sample
python3 src/adapters/run_mem0.py --sample 0 --output results/mem0/s0/questions.json

# 评测（统一 answerer + judge）
python3 -m src.evaluation.evaluate --benchmark locomo \
    --input results/mem0/s0/questions.json --output results/mem0/s0/eval.json --label mem0_s0
```
