# Baseline 适配器（统一 LoCoMo 评测协议）

日期：2026-07-02（第二、三批同日更新）。本文档记录 baseline 记忆系统的适配结果：
第一批 5 个（naive/mem0/A-Mem/LightMem/Nemori），第二批 8 个候选（Zep、MemoryOS、MemOS、
SimpleMem、Hindsight、MemMachine、MIRIX 共 7 个可行，Mnemis 1 个不可行），第三批 4 个候选
（E-Mem、EverMemOS 可行；Memobase、TiMem 不可行）。适配器只负责
**建库 + 检索**，产出统一格式的 `questions.json`；答题（answerer）与判分（judge）统一由
`src/evaluation/evaluate.py` 完成，保证跨系统可比。

## 统一契约

所有适配器（`src/adapters/run_*.py`）接受同一组 CLI 参数：

| 参数 | 含义 |
| --- | --- |
| `--sample N` | LoCoMo 会话索引（locomo10.json 第 N 个对话，默认 0） |
| `--output PATH` | 输出 JSON 路径（必填） |
| `--max-sessions N` | 开发用：只 ingest 前 N 个 session |
| `--questions-limit N` | 开发用：只处理前 N 个问题 |

个别额外参数：`run_naive.py` 必填 `--method {full_context,bm25}`；
`run_lightmem.py` 可选 `--k`（默认 20）、`--qdrant-dir`。

输出格式（JSON 数组）：首条为 `_build_stats`（`build_time_s` / `num_memories` / `notes`），
其后每题一条：

```json
{"question_id": "s0_q0", "question": "...", "gold": "...", "category": 2,
 "memories": [{"text": "...", "date": "..."}],
 "retrieval": {"latency_s": 0.01, "k": 20}}
```

统一约定：检索 top-20；builder LLM 默认 `deepseek-v4-flash`（阿里云 compatible-mode，
`src/evaluation/llm_clients.py` 的 `BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY`，可用环境变量覆盖）；
embedder 统一本地 sentence-transformers `all-MiniLM-L6-v2`（384 维），不依赖 OpenAI embedding API。

## 效率统计（LightMem 口径）

对齐 LightMem Table 2 的效率账目（LLM 调用次数、prompt/completion tokens、LLM 墙钟时间），
所有适配器统一接线 `src/adapters/_usage_tracker.py`：

**机制**。tracker 在适配器进程内 monkey-patch OpenAI SDK 最底层的
`chat.completions.create`（sync + async，另兼容 Responses API），框架内部的每一次 LLM
调用都被计入——无需改 third_party 任何文件。计数器按 phase 隔离：build 阶段一次
`reset("build")` / `snapshot("build")`，每题检索前后一次 `reset("q")` / `snapshot("q")`。
15 个框架均直接用 OpenAI SDK（无 litellm 绑定），故拦截完备；embedding 全部走本地
sentence-transformers，无 API 成本，不计。

**字段**。`_build_stats` 增加四键，每题 `retrieval` 增加三键：

| 位置 | 键 | 含义 |
| --- | --- | --- |
| `_build_stats` | `build_calls` | 构建期 LLM 调用次数 |
| `_build_stats` | `build_tokens_in` | 构建期 prompt tokens 总量 |
| `_build_stats` | `build_tokens_out` | 构建期 completion tokens 总量 |
| `_build_stats` | `build_llm_time_s` | 构建期 LLM 调用累计墙钟秒 |
| 每题 `retrieval` | `calls` | 该题检索阶段 LLM 调用次数 |
| 每题 `retrieval` | `tokens_in` | 该题检索阶段 prompt tokens |
| 每题 `retrieval` | `tokens_out` | 该题检索阶段 completion tokens |

跳过构建时（如 `run_nativemem.py --memory-dir` 复用已有库）build 四键为 `null`。
纯 embedding 检索的系统每题三键恒为 0（预期行为）；SimpleMem / E-Mem / NativeMem
等检索期调 LLM 的系统为正值。

**口径边界**。效率数字只覆盖 memory 系统自身的成本（建库 + 检索）。answerer 与
judge 由 `src/evaluation/evaluate.py` 统一执行，其 token/时间成本**单列，不算入系统成本**
——各系统的答题模型与判分模型完全相同，该部分对比较无信息量。

**已知盲区**。MIRIX 与 EverMemOS 的 LLM 调用发生在独立 venv 的 uvicorn/server 子进程
（`_mirix_server_shim.py` / `_evermemos_server_shim.py`）内，主进程 tracker 拦截不到，
当前计数为 0；如需真实用量，须在 shim 子进程内安装 tracker（或 sitecustomize 注入）。

---

## 总表

| Baseline | 适配器 | LLM 可换（deepseek-v4-flash）？ | Embedder | 构建速度（冒烟外推） | 冒烟状态 |
| --- | --- | --- | --- | --- | --- |
| naive (full_context / bm25) | `run_naive.py` | 建库不调 LLM，非问题 | 无（bm25 纯词法） | ~0s / sample | 通过（两个 method + 全量 sample 0） |
| mem0 | `run_mem0.py` | 可（openai provider + `openai_base_url`） | 本地 ST 384d（huggingface provider） | 2 sessions 39s，中速 | 通过（2 sessions、3 题，memories 非空） |
| A-Mem | `run_amem.py` | 可（OpenAIController + `OPENAI_API_BASE`，含 json_schema 实测） | 本地 ST 384d（原生默认） | 2 sessions 1510s（每 turn 2 次 LLM 调用），最慢 | 通过（35 notes，每题 top-20） |
| LightMem | `run_lightmem.py` | 可（openai manager + `openai_base_url`；README 原生宣称支持） | 本地 ST 384d（官方一等公民） | 2 sessions 111s、3 次 LLM 调用，较快 | 通过 |
| Nemori | `run_nemori.py` | 可（裸 AsyncOpenAI，任意 base_url） | 官方仅 OpenAI API，经 Protocol 注入本地 ST 384d | 2 sessions 54s、6 次 LLM 调用，较快 | 通过 |
| Zep (Graphiti) | `run_zep.py` | 可（OpenAIGenericClient + json_object 模式） | 本地 ST 384d（进程内 shim 继承 EmbedderClient ABC） | 2 sessions 215s；全量约 30–40 分钟/sample | 通过（11 facts，memories 6/8/6） |
| MemoryOS | `run_memoryos.py` | 可（裸 OpenAI SDK + `openai_base_url`） | all-MiniLM-L6-v2 即其默认值（本地 384d） | 2 sessions 142s，中速 | 通过（41 条记忆，每题 18–20 条） |
| MemOS (general_text) | `run_memos.py` | 可（openai backend + `api_base`） | 内置 sentence_transformer 后端（本地 384d） | 2 sessions 64s，较快 | 通过（11 条记忆，top-1 命中 gold） |
| SimpleMem | `run_simplemem.py` | 可（裸 OpenAI SDK + base_url） | 本地 ST（`EMBEDDING_MODEL` 环境变量，384d） | 2 sessions 69s；但检索期每题 4–8 次 LLM 调用（16–46s/题） | 通过（13 entries，含 text+ISO date） |
| Hindsight | `run_hindsight.py` | 可（OpenAICompatibleLLM + base_url，json_object 软路径） | 本地 ST 384d（维度自动检测）+ 本地 reranker | 2 sessions 57s；全量约 10 分钟/sample | 通过（12 memory units，带日期） |
| MemMachine | `run_memmachine.py` | ingest+search 路径**零 LLM**（向量 + RRF rerank）；LLM provider 本身支持 base_url | 原生 sentence-transformer provider（本地 384d） | 2 sessions 0.8s，最快 | 通过（35 episodes，每题 20 条） |
| MIRIX | `run_mirix.py` | 可（openai-compatible `model_endpoint`） | 不需要（`BUILD_EMBEDDINGS_FOR_MEMORY=false`，检索为 topic-LLM+BM25） | 2 sessions 88s；全量约 45s/session ≈ 15 分钟/sample | 通过（14 条记忆，gold 在 top 中） |
| E-Mem | `run_emem.py` | 可（4 个 LLM 角色全 OpenAI-compatible dict） | repo 自带 huggingface provider = 本地 ST 384d + BM25(jieba) | 2 sessions build 3.1s；但检索期每题每激活块 1 次 LLM 证据抽取（4–10s/题） | 通过（每题 1 块证据，命中 gold） |
| EverMemOS | `run_evermemos.py` | 可（EVEROS_LLM__* env，纯 chat-completions） | 本地 ST 384d（shim 注入，零填充到 1024d） | 2 sessions 206s，中速 | 通过（4 episodes/50 facts，每题 4 条带日期） |
| Mnemis | —（不可行） | — | — | — | 构建端未开源，无法适配（见小节） |
| Memobase | —（不可行） | — | — | — | 严格 client-server，硬依赖 PG+pgvector+Redis（见小节） |
| TiMem | —（不可行） | — | — | — | PostgreSQL 硬依赖 + 裸 SQL 检索路径（见小节） |

所有适配器均**未修改 third_party 任何文件**（A-Mem 有一处仓库预先存在的 patch，见其小节）。
Hindsight/MemMachine/MIRIX/EverMemOS 各用专属 venv（`.venv-hindsight`、`third_party/memmachine_venv`、
`third_party/mirix-venv`、`third_party/evermemos_venv`，均已 gitignore），适配器启动时自动
re-exec / 拉起子进程，`python3 src/adapters/run_*.py` 契约命令不变。

---

## naive（下界 baseline）

- 适配器：`src/adapters/run_naive.py`（自建，无第三方框架）
- 运行示例：

```bash
python3 src/adapters/run_naive.py --sample 0 --method full_context --output results/naive_full_s0.json
python3 src/adapters/run_naive.py --sample 0 --method bm25 --output results/naive_bm25_s0.json
```

- 两个 method：
  - `full_context`：不检索，整个对话拼成 1 条超长 memory；超 160K 字符（约 50K token 预算）时保留最近内容并记入 `_build_stats.notes`
  - `bm25`：session 内每 3 轮成 chunk（带 session 日期头），`rank_bm25.BM25Okapi` 索引，每题 top-20
- 模型支持：构建阶段**零 LLM 调用**、零 embedder 需求；deepseek-v4-flash 兼容性在构建期为非问题，所有 LLM 调用由下游 answerer/judge 承担
- Patch：无
- 冒烟：两个 method 均通过。bm25 全量 sample 0 = 144 chunks，build 0.28s；full_context 全文 71,472 字符（未触发截断），build 0s。带图片轮次附加 `[shared image: blip_caption]`
- 已知阻塞点：无。依赖 `pip install rank-bm25`

## mem0

- 适配器：`src/adapters/run_mem0.py`（mem0ai 2.0.10，与 `third_party/mem0` 一致）
- 运行示例：

```bash
python3 src/adapters/run_mem0.py --sample 0 --output results/mem0_s0.json
```

- 模型支持：
  - LLM：mem0 自带 provider 工厂（非 litellm 主导）；"openai" provider 是裸 OpenAI SDK，接受 `openai_base_url` → 任意 OpenAI-compatible 端点可用，**deepseek-v4-flash 经阿里云端点可行**
  - Embedder："huggingface" provider 跑本地 SentenceTransformer，`embedding_dims` 自由（用 all-MiniLM-L6-v2, 384d）；vector store 为本地嵌入式 qdrant，无需服务
- Patch：无（third_party 未动）。关键适配：OSS 版 `Memory.add(timestamp=...)` 会直接抛
  ValueError（platform-only 参数），改为 metadata `created_at`（ISO-8601）回填日期，
  并在每 session 首条消息前缀 `[Conversation on <date>]` 给抽取 LLM 提供时间锚定
  （否则相对日期会被锚到今天，写错绝对日期）
- 冒烟：通过（2 sessions, 3 题）。build 39.3s、5 条记忆，3 题 memories 各 4/3/5 条非空
- 已知阻塞点：无

## A-Mem

- 适配器：`src/adapters/run_amem.py`（`third_party/amem`，WujiangXu/AgenticMemory 论文实验仓库）
- 运行示例：

```bash
python3 src/adapters/run_amem.py --sample 0 --output results/amem/questions.json
```

- 模型支持：
  - LLM：自带 `LLMController`（openai / ollama / sglang 后端）；"openai" 后端是裸 SDK，经
    `OPENAI_API_BASE` 环境变量指向阿里云端点，**deepseek-v4-flash 实测可行，包括其每次调用都发的
    严格 `json_schema` response_format**
  - Embedder：原生本地 sentence-transformers（默认 all-MiniLM-L6-v2, 384d，内存 numpy cosine），
    与统一协议天然一致
- Patch / shim：本适配器未改 third_party 文件；仓库有一处**预先存在**的 patch
  （memory_layer.py:37-43，OpenAIController 读 `OPENAI_API_BASE`）。两个进程内 shim：
  ① `memory_layer.re = re` — 修复该模块用 `re.sub` 却没 import re 的上游 bug（否则所有 note
  的 keywords/tags 恒空）；② `get_completion` 包 3 次重试 + schema 完整的 fallback JSON
  （上游 process_memory 无 try/except，一次瞬时 API 错误会崩掉整个 build）
- 冒烟：通过（2 sessions, 3 题）。build 1510s、35 notes，每题 top-20 非空
- 已知阻塞点：无阻塞，但构建极慢——每 turn 2 次 LLM 调用（note 构建 + evolution），
  2 sessions 已 25 分钟，全量单 sample 数百 turn，是所有 baseline 中最慢的构建

## LightMem

- 适配器：`src/adapters/run_lightmem.py`（`third_party/lightmem`，zjunlp/LightMem, ICLR 2026）
- 运行示例：

```bash
python3 src/adapters/run_lightmem.py --sample 0 --output results/lightmem_s0.json
```

- 模型支持：
  - LLM：无 litellm，自研 provider 工厂（openai/deepseek/ollama/vllm/transformers）；
    "openai" manager 裸 SDK，支持 `openai_base_url`+`api_key`；**deepseek-v4-flash 经阿里云实测可行**
    （需 `response_format=json_object`，该端点支持；README 官方宣称原生支持 deepseek-v4-flash）
  - Embedder：官方一等公民即本地 sentence-transformers（官方 LoCoMo 脚本就用 huggingface/384d），维度自由
- Patch：third_party 未改。两个进程内 workaround：① `LightMemory.compressor = None` 类属性 —
  绕过 lightmem.py L169 在 `pre_compress=False`+`topic_segment=True` 时的 AttributeError；
  ② pop `OPENROUTER_API_KEY` 防止 OpenaiManager 静默改道 OpenRouter。方法级替换：llmlingua-2
  预压缩关闭（未装且仅缩短文本）、topic segmenter 的 BERT 用本地缓存 bert-base-uncased 替代
  llmlingua-2 BERT（同机制，仅用 attention 提议切分，eager/CPU）
- 冒烟：通过（sample 0, 2 sessions, 3 题）。build 111s、72 条记忆、3 次 LLM 调用（16,472 tokens）、
  每题 top-20、latency 0.03–0.08s
- 已知阻塞点：无

## Nemori

- 适配器：`src/adapters/run_nemori.py`（`third_party/nemori`，nemori-ai/nemori v0.2.0, arXiv:2508.03341）
- 运行示例：

```bash
python3 src/adapters/run_nemori.py --sample 0 --output results/nemori-locomo-s0/questions.json
```

- 模型支持：
  - LLM：裸 `AsyncOpenAI`，无 litellm，任意 base_url；**deepseek-v4-flash 经阿里云实测可行，
    支持各生成器依赖的 `response_format=json_object`**。官方声明模型 openai/gpt-4.1-mini
    （OpenRouter）、默认 gpt-4o-mini
  - Embedder：官方仅 OpenAI embeddings API，但经 `EmbeddingProvider` Protocol 注入、维度自适应；
    适配器注入本地 all-MiniLM-L6-v2（384d）
- Patch：third_party 零修改。v0.2.0 官方要求 PostgreSQL 16 + Qdrant 服务（本机无 Docker/PG），
  绕过：适配器内自实现 in-memory Episode/Semantic/Buffer store（实现其 Protocol）+
  `QdrantClient(":memory:")` 子类，直接组装 `MemorySystem`；LLM 链路 100% 上游代码。
  为确定性改为逐 session flush（上游 add.py 的 buffer_size_min=1 有调度竞态）。检索用其 eval
  默认 vector 模式，top-10 episodes + top-10 semantic = 20。依赖补装 `asyncpg`（import 需要）
- 冒烟：通过（2 sessions, 3 题）。2 episodes + 14 semantic，build 54s，6 次 LLM 调用；
  memories 非空、格式合法
- 已知阻塞点：无

---

# 第二批适配（2026-07-02）

## Zep (Graphiti)

- 适配器：`src/adapters/run_zep.py`（graphiti-core 0.29.2，pip 装 `[kuzu]` extra，与
  `third_party/graphiti` 克隆同版本）
- 运行示例：

```bash
python3 src/adapters/run_zep.py --sample 0 --output results/zep/s0/questions.json
```

- 模型支持：
  - LLM：`OpenAIGenericClient` 支持任意 OpenAI-compatible base_url，已接 BUILDER_MODEL/BASE/KEY；
    用 `structured_output_mode="json_object"`（DeepSeek 不支持 json_schema）；**`small_model`
    必须同设**，否则回落 gpt-4.1-nano
  - Embedder：本地 all-MiniLM-L6-v2 384d，进程内 shim 继承公开 `EmbedderClient` ABC；
    须在 import 前设 `EMBEDDING_DIM=384`
- 关键点：**无需 Neo4j 服务**——用内置 KuzuDriver（嵌入式、文件级，kuzu wheel 静态捆绑 FTS
  扩展，离线可用）。检索用基础 `graphiti.search()` = EDGE_HYBRID_SEARCH_RRF（语义+FTS 融合），
  不走 cross-encoder，无需 OpenAI key
- Patch / shim（third_party 未动，docstring 有记录）：① 本地 embedder；② 手动建 Kuzu FTS 索引
  （上游 bug：`KuzuDriver.build_indices_and_constraints` 是 no-op，导致 search 报
  "doesn't have an index"）；③ 预设 `driver._database=group_id`（上游 bug：add_episode 传
  group_id 时读未初始化的 `_database`，必 AttributeError）
- 冒烟：通过（2 sessions, 3 题）。build 215s、11 facts，3 题 memories 6/8/6 条非空，日期带
  valid_at 时间锚定（"yesterday"→2023-05-07），检索延迟 0.015–0.06s
- 已知阻塞点：无阻塞，但慢——全量 19 sessions 预计约 30–40 分钟/sample（每 episode 多次
  LLM 调用，graphiti 固有开销）

## MemoryOS

- 适配器：`src/adapters/run_memoryos.py`（`third_party/memoryos`，memoryos-pypi 变体）
- 运行示例：

```bash
python3 src/adapters/run_memoryos.py --sample 0 --output results/memoryos/s0/questions.json
```

- 模型支持：
  - LLM：`Memoryos(openai_api_key, openai_base_url, llm_model)` 走裸 OpenAI SDK，
    BUILDER_MODEL/BASE/KEY 直接可用
  - Embedder：`embedding_model_name="all-MiniLM-L6-v2"` 即其默认值，本地 384d，无 embedding API
- 关键点：memoryos-pypi 变体纯进程内（JSON 文件 + faiss，已装 faiss-cpu 1.14.3），
  无 Docker/DB/云服务依赖
- Patch / shim（third_party 未动）：① importlib 把连字符目录注册为包 `memoryos`（模块内有相对
  导入，必须按包加载）；② turn 配对成 QA page，空侧补占位符（否则 updater 静默丢页）；
  ③ `short_term_capacity=1`（同官方 LoCoMo eval），结束后手动 flush 尾页
- 冒烟：通过（2 sessions, 3 题）。build 142s、41 条记忆（32 页 + 5 用户知识 + 3 助手知识 +
  1 profile，页数因多主题会话会重复入库，系其自身行为），每题 18–20 条非空，延迟约 0.05s，
  q0 命中含正确日期的原始对话页
- 已知阻塞点：无

## MemOS（general_text 无服务子集）

- 适配器：`src/adapters/run_memos.py`（`third_party/memos`，经 sys.path 指向 `src`，未 pip install）
- 运行示例：

```bash
python3 src/adapters/run_memos.py --sample 0 --output results/memos/s0/questions.json
```

- 模型支持：
  - LLM：openai backend 支持 `api_base`，BUILDER_* 直接接入，deepseek-v4-flash 已验证
  - Embedder：内置 `sentence_transformer` 本地后端，all-MiniLM-L6-v2 384d
- 关键点 / **口径注意**：用 `general_text` 后端（GeneralTextMemory）= LLM 抽取事实 → 本地嵌入 →
  qdrant 向量检索，纯进程内。**官方 LoCoMo 75.8 走的是 memos-api 服务端（TreeTextMemory，
  强依赖 Neo4j/PolarDB 图数据库），进程内无法复现**；本适配器是同一抽取 prompt
  （SIMPLE_STRUCT_MEM_READER_PROMPT）的开源无服务子集，**写结果时需注明**
- Patch / shim：未 pip install -e（MemOS 锁 transformers<5，与本机 5.12.1 冲突），改 sys.path
  加载，零修改；补装 concurrent-log-handler、ollama、prometheus-client 三个纯 import 依赖。
  时间戳：extract 会盖今天日期，适配器把 session 日期前缀进消息并回写 `metadata.updated_at`
- 冒烟：通过（2 sessions, 3 题）。build 64s、11 条记忆，3 题均检索 11 条（top-20 上限内），
  top-1 均命中 gold 相关事实且日期正确，延迟 0.05–0.09s
- 已知阻塞点：无（除口径注意）

## SimpleMem

- 适配器：`src/adapters/run_simplemem.py`（`third_party/simplemem`，aiming-lab/SimpleMem @ 60a48e8）
- 运行示例：

```bash
python3 src/adapters/run_simplemem.py --sample 0 --output results/simplemem/s0/questions.json
```

- 模型支持：
  - LLM：`LLMClient` 是裸 OpenAI SDK + base_url 透传，BUILDER_* 直接可用；streaming 已关
  - Embedder：本地 SentenceTransformer，维度运行时读取，`EMBEDDING_MODEL` 环境变量直接支持
    all-MiniLM-L6-v2（384d），无 embedding API
- 关键点：存储为嵌入式 LanceDB（本地路径）+ tantivy，纯 pip，无外部服务。**检索期每题 4–8 次
  LLM 调用**（SimpleMem 自带 planning+reflection，上游默认配置），每题 16–46s，全量比 mem0 慢。
  `retrieve()` 返回多路合并去重列表（无统一打分），截断至 top-20，`retrieval.n_retrieved`
  记录截断前数量
- Patch / shim（third_party 未动，docstring 有记录）：唯一 patch——上游 `use_tantivy=True` 建
  FTS 索引，lancedb 0.33（上游 pin 0.25.3）建得出但查不了（"Cannot perform full text search
  unless an INVERTED index has been created"），BM25 词法层静默失效；shim 替换实例的
  `_init_fts_index` 改用原生 inverted index，冒烟确认 Keyword Search 返回结果
- 冒烟：通过（2 sessions, 3 题）。build 69s、13 条 memory entries（35 dialogues），3 题各 13 条
  非空（含 text+ISO date），延迟 16–46s/题
- 已知阻塞点：无。新装依赖：lancedb、dateparser、tantivy

## Hindsight

- 适配器：`src/adapters/run_hindsight.py`（`third_party/hindsight`，upstream main @6a479dd,
  v0.8.4；原克隆为空已补拉）
- 运行示例：

```bash
python3 src/adapters/run_hindsight.py --sample 0 --output results/hindsight/s0/questions.json
```

- 模型支持：
  - LLM：provider "openai" 走 OpenAICompatibleLLM，支持自定义 base_url → BUILDER_* 直接可用；
    结构化输出默认 json_object 软路径（schema 注入 prompt），DashScope 兼容
  - Embedder：默认本地 sentence-transformers，传入 all-MiniLM-L6-v2（384d 自动检测）；
    reranker 本地 ms-marco-MiniLM-L-6-v2
- 关键点：早前 RUN.md「必须部署整个服务栈」的判断不成立——官方 benchmark runner 本身就是
  进程内直用 `hindsight_api.engine.MemoryEngine`；Postgres+pgvector 用 `pg0-embedded`（pip 包，
  自动管理嵌入式 Postgres）绕过，无 Docker/外部服务。忠实复刻官方 LoCoMo 协议：整 session JSON
  一次 `retain_batch`、event_date=session 日期、recall budget=HIGH/max_tokens=4096、
  `wait_consolidation=False`
- Patch / shim：third_party 未动；deps 装进项目根 `.venv-hindsight`，脚本头部 re-exec shim
  保持 `python3` 直跑契约（docstring 有记录）
- 冒烟：通过（2 sessions, 3 题）。build 57s、12 memory units，3 题均非空（带日期，命中 gold
  相关事实），recall 延迟 0.2–0.6s
- 已知阻塞点：无。全量单 sample 构建约 10 分钟

## MemMachine

- 适配器：`src/adapters/run_memmachine.py`（`third_party/memmachine`）
- 运行示例：

```bash
python3 src/adapters/run_memmachine.py --sample 0 --output results/memmachine/s0/questions.json
```

- 模型支持：
  - LLM：本适配器的 ingest+search 路径**完全无 LLM**（向量检索 + RRF(identity,bm25) rerank）；
    LLM provider `openai-chat-completions` 本身支持 base_url，BUILDER_* 可接
  - Embedder：原生 `sentence-transformer` provider 跑本地 all-MiniLM-L6-v2（384d）
- 关键点 / **口径注意**：MemMachine 表面是 client/server 产品（FastAPI + Postgres/Neo4j/Qdrant
  docker-compose），但 `memmachine_server` 包可直接 import，官方 LoCoMo 评测脚本本身就是进程内跑。
  外部服务全部绕过：episode/segment store 用 sqlite（aiosqlite），向量库用 `sqlite_vector_store`
  （本地文件 + usearch），LTM 用官方支持的 "event" backend（默认 "declarative" 只支持
  Neo4j/Nebula）。**官方 91.7 依赖查询期 LLM retrieval agent（ToolSelectAgent），与统一 top-20
  检索契约不符，未启用**（docstring 有记录），引用官方数字时口径不可比
- Patch / shim：third_party 未动。依赖装在专用 venv `third_party/memmachine_venv`
  （--system-site-packages，已 gitignore），适配器启动自动 re-exec；需一次性
  `memmachine-nltk-setup`（stopwords 已装）
- 冒烟：通过（2 sessions, 3 题）。build 0.8s / 35 episodes，3 题各 20 条非空，top-1 命中相关
  消息且日期正确，延迟 0.06–0.08s/题
- 已知阻塞点：无

## MIRIX

- 适配器：`src/adapters/run_mirix.py` + 进程内 shim `src/adapters/_mirix_server_shim.py`
  （`third_party/mirix`）
- 运行示例：

```bash
python3 src/adapters/run_mirix.py --sample 0 --output results/mirix/s0/questions.json
```

- 模型支持：
  - LLM：走 openai-compatible `model_endpoint`，BUILDER_MODEL/BASE/KEY 直接可用
  - Embedder：**完全不需要**——设 `BUILD_EMBEDDINGS_FOR_MEMORY=false`，检索是 topic-LLM + BM25
- 关键点：MIRIX 核心只经其 FastAPI REST 服务暴露，适配器用专属 venv（`third_party/mirix-venv`，
  已 gitignore）拉起本地 server 子进程（SQLite 后端，无 Postgres/Docker/Redis），主环境纯 HTTP 调用
- Patch / shim（三个坑，docstring 均有记录，third_party 未动）：① Aliyun thinking 模式拒绝
  `tool_choice=required` → extra_body `enable_thinking:false`；② 上游 bug：
  `/memory/retrieve/conversation` 的 has_content 要 dict 内容而 topic 提取只吃 str，导致 topic
  提取必挂、检索退化为 recent-only → shim 归一化；③ 内层 memory agent 链由环境变量
  `CHAINING_FOR_MEMORY_UPDATE`（默认 false）控制，deepseek 单轮只 search 不 insert → 设为 true
- 冒烟：通过（2 sessions, 3 题）。build 88s、14 条记忆（episodic 9 + semantic 5），每题 14 条、
  检索延迟约 1s，3 题 gold 答案均出现在 top 记忆中
- 已知阻塞点：无。全量约 45s/session，s0 约 19 sessions ≈ 15 分钟

## Mnemis（不可行）

- 适配器：**未创建**。判定不可行，阻塞点（按严重程度）：
  1. **构建端完全缺失**：`third_party/mnemis` 只有 `global_selection/`（452 行 System-2 检索
     代码）+ `results/`（预计算 per-question 预测）+ 论文 PDF/博客。base graph 构建、hierarchical
     graph 构建（MCA 聚类）、System-1 相似度检索、ingestion pipeline 全部未开源。README §5 自认
     只提供 prompts（论文附录）和 results——没有可复用的「构建记忆 + 检索」核心，只有半个检索
  2. **仅有的模块强依赖 Neo4j 服务端 + graphiti_core**：`global_selector.py` 直接跑 Cypher
     （含 `CATEGORIZES*1..` 变长路径查询），且只读图不建图——没有构建代码往 Neo4j 灌数据，
     该模块本身也跑不起来。进程内替代等于按论文附录从零重实现整个系统（估计 3–5 天），超出适配范畴
  3. Embedder：论文用 Qwen3 float32 embedding，且无构建代码可接 sentence-transformers，无从谈起
- 替代利用方式（非适配器）：直接拿 `results/locomo/results_*_ragtopk10_gtopk20_*.json` 的
  per-question predictions 用我们统一的 judge 重打分（其自报 accuracy 80.06%；LJ 93.9 是论文
  judge 口径）。可作「引用 + 统一 judge 复评」的对比数据，但不能算可跑 baseline

---

# 第三批适配（2026-07-02）

## E-Mem

- 适配器：`src/adapters/run_emem.py`（`third_party/emem`，dog-last/E-mem，ICML 2026，
  arXiv 2601.21714）
- 运行示例：

```bash
python3 src/adapters/run_emem.py --sample 0 --output results/emem/s0/questions.json
```

- 模型支持：
  - LLM：4 个 LLM 角色（块摘要 / 块内证据抽取 / manager / aggregator）全部 OpenAI-compatible
    dict 配置，已接 BUILDER_MODEL/BASE/KEY；**deepseek-v4-flash 实测可行，包括其发出的
    `tools=[]` 和 `repetition_penalty` extra_body**
  - Embedder：repo 自带 huggingface provider = 本地 sentence-transformers all-MiniLM-L6-v2
    （384d）+ BM25(jieba) 混合路由；tokenizer Qwen/Qwen3-4B 仅做 token 计数（本地缓存）
- 关键点 / **检索语义注意**：text 模式纯 API + 进程内运行，无外部服务。参数对齐官方
  config.text.yaml（block 4096 tok，max_blocks=5，bm25_boost=0.8）。E-mem **无 top-k item
  search**：每题 HybridRouter 路由激活 ≤5 块 + 活动块，各做一次 LLM 证据抽取，每块证据 =
  一条 memory（k = 实际条数 ≤6，日期内嵌文本）；**跳过 master 聚合步**（会直接合成答案，
  违反「适配器只建库+检索」契约），docstring 已记录
- Patch / shim：third_party 零修改。进程内 shim 两个（docstring 记录）：① sys.modules 中
  src 包 purge——解决本项目与 emem 的 `src` 包名冲突；② `TEXT_DATA_DIR` 环境变量指向临时目录
- 冒烟：通过（2 sessions, 3 题）。build 3.1s / 1 block（35 turns），3 题 memories 全非空且
  证据命中 gold（如 q0 检索到 "8 May, 2023 ... support group yesterday"）；追加 6-session
  冒烟验证块滚动 + HybridRouter 路径（2 blocks，k=2）
- 已知阻塞点：无阻塞，但检索期每题每激活块 1 次 LLM 调用（冒烟 4–10s/题），全量检索成本
  高于向量检索类系统

## EverMemOS（EverOS）

- 适配器：`src/adapters/run_evermemos.py` + 进程内 shim
  `src/adapters/_evermemos_server_shim.py`（`third_party/evermemos`，EverMind-AI/EverMemOS
  已改名 EverOS，main@0341f12，~10k star，arXiv:2601.02163）
- 运行示例：

```bash
python3 src/adapters/run_evermemos.py --sample 0 --output results/evermemos/s0/questions.json
```

- 模型支持：
  - LLM：经 `EVEROS_LLM__*` 环境变量接 BUILDER_MODEL/BASE/KEY（deepseek-v4-flash@阿里云，
    纯 chat-completions，无 function calling 要求）
  - Embedder：上游只有 OpenAI-HTTP 客户端，shim 进程内 monkeypatch
    `build_embedding_provider` 注入本地 all-MiniLM-L6-v2（384d 零填充到 LanceDB 硬编码的
    Vector(1024)，cosine 序不变）；rerank 不需要——user+HYBRID（episode hierarchy：hybrid
    召回 + fact MaxSim + RRF + fact eviction）只依赖 embedding
- 关键点：原企业版（MongoDB/ES/Milvus）已下线，现版本本地栈 = Markdown + 内嵌 SQLite +
  内嵌 LanceDB，无 Docker/外部服务。官方仅支持 server 模式，按 run_mirix.py 先例以本地
  uvicorn 子进程运行（venv：`third_party/evermemos_venv`，Python 3.12）。协议：mode=chat；
  每 session 一次 `/memory/add` + `/flush`（强制边界抽取）；单一 owner sender_id="user"
  （避免双 speaker episode 双份 fan-out），发言人以 "Name: text" 前缀保留；session 日期→
  epoch ms 时间戳；检索 hybrid top-20，text=episode 叙事（未含 facts），date=episode 时间戳
- Patch / shim：third_party 零修改。shim 内两处进程内改动：① 本地 MiniLM embedding provider
  替换 factory；② 新增 `GET /debug/drained`（OME idle + cascade 队列 + LanceDB 行数，供建库
  后等待索引落盘）。适配器另修一坑：requests 需 `trust_env=False`，否则 macOS 系统代理
  (:7890) 拦截 localhost 请求返回 502
- 冒烟：通过（2 sessions, 3 题）。build 206s / 4 episodes + 50 atomic facts，3 题均 4 条
  非空 memories（带正确日期），检索延迟 0.05–0.16s
- 已知阻塞点：无

## Memobase（不可行）

- 适配器：**未创建**（`third_party/memobase` 已 clone）。阻塞点（按严重程度）：
  1. **严格 client-server 架构**：pip 包 `memobase` 是纯 HTTP 客户端（httpx → project_url），
     无嵌入式/本地模式。官方 LoCoMo 评测（`docs/experiments/locomo-benchmark/src/memobase_client/`）
     连的是云端 api.memobase.dev 或自托管服务端
  2. **服务端硬依赖 PostgreSQL+pgvector 与 Redis**：`connectors.py` 在 import 时就建引擎并
     `create_tables()`（pgvector 向量列 + `CREATE EXTENSION vector`）；Redis buffer 队列含
     Lua 脚本、分布式锁，贯穿 controllers（`controllers/buffer_background.py`）。官方部署仅
     docker-compose（pg17-pgvector + redis:7.4 + api 容器）
  3. 本机无 Docker/Postgres/Redis；SQLite 无法替代 pgvector（向量列+距离算子），fakeredis
     对 Lua 脚本支持脆弱——进程内 shim 等于重写整个服务端
  4. 附加不符：embedding 仅支持 `openai|jina|ollama` API 提供方，无本地 sentence-transformers
     路径，即使 DB 解决也不满足 all-MiniLM-L6-v2 本地 384d 契约

## TiMem（不可行）

- 适配器：**未创建**（`third_party/timem` 已 clone，TiMEM-AI/timem v1.1.0，ACL 2026）。
  阻塞点（按严重程度）：
  1. **PostgreSQL 硬依赖，无进程内替代**：`storage/memory_storage_manager.py:187-261` 明确
     "PostgreSQL only, no fallback mechanism"，非 postgres provider 直接抛错；
     `storage/postgres_store.py` 用 JSONB、TSVECTOR、`to_tsvector` 触发器等 PG 专属特性，
     SQLite 换不上。`MockStorageAdapter` 存在但全库无一处接线
  2. **检索路径绕过适配器接口直连 PG**：`experiments/datasets/locomo/02_memory_retrieval.py`
     直接 `get_postgres_store()` 开 SQLAlchemy session、检查 engine pool、做泄漏 session
     清理；retrieval_nodes 里约 30 处裸 `text()` SQL，shim 无法覆盖
  3. **Qdrant 以服务端 URL 连接**（`storage/vector_store.py:111` `QdrantClient(url=...)`），
     settings.yaml 还配 Redis cache + Neo4j graph；官方部署即
     `migration/docker-compose.yml`（postgres:15 + qdrant）
  4. 复现脚本是 pytest 形态（`01_memory_generation.py` 无 CLI，纯环境变量 + 配置文件驱动），
     且模拟午夜 backfill 调度——架构是常驻服务（FastAPI + 全局连接池），不是可嵌入的库
- 此前「可 pip 装」结论只对 cloud SDK（`timem-ai`，连 api.timem.cloud）成立；自托管必须
  Docker 起库。embedder 默认 Qwen 2048d，也与 384d MiniLM 约定冲突（次要）。如坚持要它，
  只能走 cloud API（引入外部服务，不符合契约），建议放引用组

---

## 完整对比流程

1. **建库 + 检索**（每个 baseline × 每个 sample 各跑一次，产出 questions.json）：

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"

python3 src/adapters/run_naive.py      --sample 0 --method bm25 --output results/naive_bm25/s0/questions.json
python3 src/adapters/run_naive.py      --sample 0 --method full_context --output results/naive_full/s0/questions.json
python3 src/adapters/run_mem0.py       --sample 0 --output results/mem0/s0/questions.json
python3 src/adapters/run_amem.py       --sample 0 --output results/amem/s0/questions.json
python3 src/adapters/run_lightmem.py   --sample 0 --output results/lightmem/s0/questions.json
python3 src/adapters/run_nemori.py     --sample 0 --output results/nemori/s0/questions.json
python3 src/adapters/run_zep.py        --sample 0 --output results/zep/s0/questions.json
python3 src/adapters/run_memoryos.py   --sample 0 --output results/memoryos/s0/questions.json
python3 src/adapters/run_memos.py      --sample 0 --output results/memos/s0/questions.json
python3 src/adapters/run_simplemem.py  --sample 0 --output results/simplemem/s0/questions.json
python3 src/adapters/run_hindsight.py  --sample 0 --output results/hindsight/s0/questions.json
python3 src/adapters/run_memmachine.py --sample 0 --output results/memmachine/s0/questions.json
python3 src/adapters/run_mirix.py      --sample 0 --output results/mirix/s0/questions.json
python3 src/adapters/run_emem.py       --sample 0 --output results/emem/s0/questions.json
python3 src/adapters/run_evermemos.py  --sample 0 --output results/evermemos/s0/questions.json
```

2. **答题 + 判分**（统一 answerer / judge，跨系统可比）：

```bash
python3 -m src.evaluation.evaluate --benchmark locomo \
    --input results/mem0/s0/questions.json \
    --output results/mem0/s0/eval.json \
    --label mem0_s0
```

`evaluate.py` 其他参数：`--metrics`（可选指标集）、`--limit N`（只评前 N 题）、
`--skip-answerer` / `--skip-judge`（断点续评）。

3. LoCoMo 全量 = 10 个 sample（`--sample 0..9`），逐 sample 跑步骤 1–2 后汇总。

注意：builder 换模型用环境变量 `BUILDER_MODEL` / `BUILDER_BASE` / `BUILDER_KEY`
（`run_amem.py`、`run_lightmem.py` 目前把 `deepseek-v4-flash` 写死在常量里，换模型需同步改常量或改走
`BUILDER_MODEL` 环境变量——`run_mem0.py`、`run_nemori.py` 已经读环境变量）。

---

## 论文 baseline 阵容建议（2026-07-02）

### 实测组（统一协议下重跑，跨系统可比）

15 个适配器配置全部冒烟通过，全部走同一契约（builder=deepseek-v4-flash、embedder=本地
all-MiniLM-L6-v2 384d、检索 top-20、统一 answerer/judge）：

1. naive full-context（上界参照）
2. naive BM25（朴素下界）
3. Mem0（ECAI 2025）
4. A-Mem（NeurIPS 2025）
5. LightMem（ICLR 2026）
6. Nemori（arXiv 2508.03341）
7. Zep / Graphiti（arXiv 2501.13956）
8. MemoryOS（arXiv 2506.06326）
9. MemOS（arXiv 2507.03724，general_text 开源子集，须注明非官方 TreeTextMemory 服务端口径）
10. SimpleMem（aiming-lab）
11. Hindsight（v0.8.4）
12. MemMachine（未启用其查询期 LLM retrieval agent，须注明与官方 91.7 口径不同）
13. MIRIX（arXiv 2507.07957）
14. E-Mem（ICML 2026，arXiv 2601.21714，须注明检索为块路由+LLM 证据抽取、k≤6，
    跳过其 master 聚合步）
15. EverMemOS / EverOS（arXiv 2601.02163）

加上我们自己的 NativeMem（`run_nativemem.py`），实测表共 16 行。其中 MemOS、MemMachine、
E-Mem 三行建议在表注中写明「开源无服务子集 / 未启用 LLM 检索 agent / 非 top-k 检索语义」的
口径差异，避免审稿人拿官方博客数字对质。

### 引用数字组（不可跑，只引官方数字，注明口径不可比）

这些系统的官方数字来自各自的 judge/answerer/检索配置，与我们统一协议**不可直接比较**，
只能放在相关工作或表注中引用：

| 系统 | 不可跑原因 | 备注 |
| --- | --- | --- |
| Mnemis（LoCoMo 93.9） | 构建端未开源，仅半个检索模块 + 预计算结果 | 可用其 per-question predictions 做统一 judge 复评（自报 accuracy 80.06%） |
| Memobase | 严格 client-server；服务端硬依赖 PG+pgvector+Redis，本机无 Docker | embedding 也无本地路径（见第三批小节） |
| TiMem（ACL 2026） | PostgreSQL 硬依赖无 fallback + 检索路径裸 SQL 直连 PG + Qdrant/Redis/Neo4j 服务栈 | cloud SDK 可用但引入外部服务，不符契约（见第三批小节） |
| DeltaMem-RL（arXiv 2604.01560） | 无代码 | 阿里系 RL 记忆管理 |
| Memory-R1（arXiv 2508.19828） | 仓库至今仍是 "Code coming soon" 占位 | |
| Synthius-Mem（arXiv 2604.11563） | 无仓库，独立作者预印本 | 自报 94.4 超人类，数字可疑，谨慎引用 |
| MemArchitect（arXiv 2603.18330） | 无代码，"ongoing work" | |
| DeltaMem-trees（arXiv 2606.03083） | 有代码但面向 ALFWorld/ScienceWorld/WebShop 智能体任务 | 场景不匹配，基本只引数字 |

此前列为「开源可适配但未适配」的三个候选（TiMem、E-Mem、EverMemOS）第三批已全部处理完：
E-Mem、EverMemOS 已适配进实测组；TiMem 判定不可行，移入引用组。开源候选积压清零，
审稿人追问 SOTA 时以实测组 15 项 + 引用组官方数字应对。

### 与对标论文的阵容对比

| 论文 | 实测 baseline 阵容 | 规模 |
| --- | --- | --- |
| Mem0（ECAI 2025） | A-Mem、LangMem、Zep、OpenAI Memory + RAG 变体、full-context | 约 6 组 |
| Nemori（arXiv 2508.03341） | Mem0、Zep 等少数系统 + 朴素基线 | 约 4–6 组 |
| LightMem（ICLR 2026） | FullText、NaiveRAG、A-MEM、MemoryOS(eval/pypi)、Mem0(oss/api)、Mem0-g | 约 7–8 组 |
| Mnemis | Zep/Mem0/RAG 一类（开源部分只给自家结果） | 约 4–6 组 |
| **我们** | **naive×2 + 13 个真实系统，统一 builder/embedder/top-k/judge** | **15 组** |

结论：**够投稿，且是同类工作中最大的统一协议实测阵容**。对标论文普遍 4–8 组 baseline，
且多数直接抄各家自报数字（judge 口径不一）；我们 13 个真实系统全部在同一 builder LLM、
同一本地 embedder、同一检索预算、同一 answerer/judge 下重跑，公平性论证显著强于对标论文。
覆盖面上：向量抽取类（Mem0/MemOS/MemMachine）、图记忆类（Zep）、分层/OS 类
（MemoryOS/MIRIX/Hindsight）、篇章/情景类（Nemori/LightMem/A-Mem/SimpleMem/EverMemOS）、
块路由长上下文类（E-Mem）、朴素基线（full-context/BM25）各流派齐全。开源候选积压已清零；
Mnemis/Memory-R1/Memobase/TiMem 等闭源或重服务栈系统的数字放引用组即可。

---

## GPT-5.5（订阅 proxy）builder 兼容性（2026-07-02）

冒烟设置：`BUILDER_MODEL=gpt-5.5 BUILDER_BASE=http://localhost:8199/v1`（`src/chatgpt_proxy.py`
订阅 proxy），LoCoMo sample 0 前 2 sessions、3 题，判定标准 = `/tmp/smoke55_<name>.json`
存在 + 3 条题目记录 + memories 非空。embedder/检索预算与 deepseek 冒烟完全一致，
answerer/judge 不在本冒烟范围内（answerer 仍为 deepseek-v4-flash，judge 本就是 GPT-5.5）。

过程记录：首轮编排的 14 个后台任务里 12 个因脚本使用 macOS 上不存在的 GNU `timeout`
命令在启动瞬间失败（只有 memmachine、evermemos 真正跑完），本表数字为核验 agent 以同一
命令契约补跑后的结果，以实际文件为准。

| 适配器 | PASS/FAIL | 构建耗时 gpt-5.5 vs deepseek（2 sessions） | 失败根因 |
| --- | --- | --- | --- |
| mem0 | PASS | 115.5s vs 39.3s（2.9×） | — |
| A-Mem | PASS | 2113.3s vs 1510.0s（1.4×） | — |
| LightMem | PASS | 105.3s vs 111.0s（1.0×） | — |
| Nemori | PASS | 108.0s vs 54.1s（2.0×） | — |
| Zep (Graphiti) | PASS | 195.5s vs 215.5s（0.9×） | — |
| MemoryOS | PASS | 289.9s vs 142.1s（2.0×） | — |
| MemOS | PASS | 105.1s vs 63.8s（1.6×） | — |
| SimpleMem | PASS | 211.5s vs 68.7s（3.1×，含检索期 LLM 调用） | — |
| Hindsight | PASS | 84.2s vs 56.9s（1.5×） | — |
| MemMachine | PASS | 0.7s vs 0.8s（build 零 LLM，兼容性非问题） | — |
| MIRIX | PASS（proxy 修复后） | 232.9s vs 87.8s（2.7×） | 首跑 FAIL：proxy 响应缺 `created`（Pydantic `ChatCompletionResponse` 要求 datetime）；修复后又暴露 proxy 把 content-part list（`{"type":"text"}`）原样透传给 Responses API 被 400 拒。两处均为 proxy 缺陷，已修复后重跑通过 |
| E-Mem | PASS | 2.5s vs 3.1s（build 几乎零 LLM） | — |
| EverMemOS | PASS（proxy 修复后） | 181.9s vs 206.0s（0.9×） | 首跑 FAIL：proxy 响应缺 `model` 字段（Pydantic `ChatResponse` 要求 str），boundary_detect 重试 3 次后 memory/add 500、0 memories。proxy 回填后重跑通过 |
| NativeMem | PASS | 169.9s vs —（deepseek 冒烟复用已建库 0s，不可比） | — |

三处 proxy 修复均已进 `src/chatgpt_proxy.py` 并重启 8199 实例：响应回填 `model` 与
`created`（int epoch）、请求侧把 Chat Completions 的 content-part list 归一化为纯文本。
未改任何 third_party 框架代码。

另一个实测坑：订阅 proxy 上游（chatgpt.com）在高并发下会返回 403 HTML 拦截页
（MIRIX 6-agent 突发 + A-Mem 同跑时复现，单独重跑即恢复）。全量跑 GPT-5.5 builder 时
建议适配器串行、避免多系统同时压 proxy。

**结论**：14 个适配器全部可以用 GPT-5.5 做 builder，没有「只能 deepseek」的系统——
EverMemOS/MIRIX 的失败根因都在我们 proxy 的 OpenAI 规范缺口，不在框架本身，修复后全绿。
代价是 LLM 密集型构建普遍慢 1.5–3×；A-Mem 冒烟 2 sessions 已要 2113s，外推全量
（~19 sessions/sample）约 5–6 小时/sample，全量跑 GPT-5.5 不现实，建议 A-Mem 的
GPT-5.5 版本只做小样本对照或维持 deepseek。MemMachine/E-Mem/naive 构建端零 LLM，
builder 换型对其无影响。
