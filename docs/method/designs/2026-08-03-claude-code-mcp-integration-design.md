# Agent Memory Harness 与 Claude Code 集成设计

状态：待实现评审，2026-08-03。

## 1. 目标

Agent Memory Harness 保留两种互不替代的运行方式：

1. 实验运行：benchmark runner 启动隔离的 Claude Code 子进程，固定模型、预算、prompt 和结果目录，保证实验配置可复现。
2. 交互运行：用户正在使用的 Claude Code 会话通过本地 stdio MCP server 直接访问同一套 memory runtime。当前 Claude Code 负责语义判断，Runtime 负责文件约束、事务提交和派生视图更新。

交互运行不得要求第二个 LLM、额外 API key 或独立 Claude 登录态。安装和运行参数使用命令行参数或配置文件，不使用环境变量。

本阶段不实现 Claude Code 全量会话的自动监听、后台定时整理、云端 memory service。当前会话需要保存的信息由 Claude 主动提交给 MCP 工具；自动监听需要单独设计 transcript hook、增量游标和成本策略。

## 2. 安装与启动

项目提供标准 Python package 和 `agent-memory` 命令。发布前可直接从 Git 仓库安装，发布后使用相同的 PyPI package name：

```bash
pip install git+https://github.com/<owner>/Agent-Memory-Harness.git
# 发布后：pip install agent-memory-harness
```

初始化一个 memory workspace：

```bash
agent-memory init /absolute/path/to/memory
```

将该 workspace 注册到 Claude Code：

```bash
claude mcp add --scope user agent-memory -- \
  agent-memory mcp --workspace /absolute/path/to/memory
```

注册信息由 Claude Code 管理。`agent-memory mcp` 只使用命令行中的 workspace 路径，不读取或修改用户的 Claude Code 登录配置。

项目同时提供 Claude Code plugin manifest。plugin 只补充工具说明和使用规则；实际读写能力由同一个 MCP server 提供。直接执行 `claude mcp add` 仍是基础安装方式，不要求用户先配置 plugin marketplace。

## 3. 包结构与公共接口

新增 `pyproject.toml`，distribution name 为 `agent-memory-harness`，console script 为 `agent-memory`。现有实现仍保存在 `code/src/`，不进行与功能无关的全量目录改名。

Python 用户通过稳定 facade 导入：

```python
from agent_memory_harness import BuildConfig, MemoryConfig, MemoryWorkspace
```

现有 `src` import 暂时保留给仓库内部脚本和测试，避免本次集成同时修改全部 benchmark runner。公开 README 不再要求用户直接导入名为 `src` 的 package。

CLI 第一版只包含两个必要命令：

- `agent-memory init PATH`：创建并校验 workspace 基础结构。
- `agent-memory mcp --workspace PATH`：以 stdio 方式启动 MCP server。

不增加独立 daemon、Web server 或全局配置管理器。

## 4. MCP 工具

MCP server 复用现有 `MemoryWorkspace`、Markdown validator、retrieval 和派生视图代码，不实现第二套 memory lifecycle。

### 4.1 读取与检索

- `memory_status()`：返回 workspace 根目录、Topic 文件、Source 数量、派生视图状态和最近一次提交。
- `memory_read(path, heading=None, block_id=None, start_line=None, end_line=None)`：读取指定文件、章节、block 或行区间。路径必须位于 workspace 内。
- `memory_grep(query, path=None, limit=...)`：执行 exact lexical search。
- `memory_bm25(query, limit=...)`：对现有 Markdown memory index 执行 sparse lexical retrieval。
- `memory_embedding(query, limit=...)`：在已配置 embedding backend 时执行 dense retrieval；未配置时返回明确的 unavailable 状态，不影响 grep 和 BM25。

Claude Code 自己决定使用哪种检索方式以及是否继续读取来源。MCP server 不添加固定的检索路由。

### 4.2 保存当前会话证据

`memory_archive(messages, observed_at=None, session_label=None)` 接收 Claude Code 选出的当前会话片段。Runtime 为 session、message 和 source anchor 分配稳定 ID，写入 append-only Source Memory，并返回可在 Topic footnote 中使用的 source handles。

第一版不自动读取 Claude Code 的完整 transcript。Claude 只提交形成当前记忆所需的消息，避免将无关上下文复制到 memory workspace。

### 4.3 修改 Topic Memory

`memory_apply_patch(patch)` 接受标准 unified diff，只允许修改 `topics/**/*.md` 和 `core.md`。Claude Code 直接决定 Topic 文档的自然语言内容、章节结构、段落边界、内部链接和来源引用。

Runtime 在 staging transaction 中执行以下确定性操作：

1. 应用 patch，并拒绝越界路径、绝对路径、符号链接逃逸和非授权文件修改。
2. 校验 Topic block、evidence footnote、source handle 和内部链接。
3. 为 transaction-local labels 生成稳定 ID，处理 ID 冲突并改写对应引用。
4. 从 Topic 与 Source 重建 Timeline、Recent、Relations 和检索索引。
5. 全部校验通过后提交；任一步失败则不安装任何文件，并向 Claude 返回具体文件和错误原因。

交互路径不在服务器内部再次调用 LLM reconciliation。格式错误由 Claude 根据 validation error 修正后重新提交；语义拆分、合并、补充和删除仍由当前 Claude 决定。

## 5. 权限与一致性

- `sources/` 对 Claude 只读，只能通过 `memory_archive` 追加。
- Timeline、Recent、Relations 和检索索引均为 Runtime 派生状态，Claude 不直接修改。
- MCP server 不暴露任意 shell 执行接口。
- 所有 path 参数先 canonicalize，再验证其仍位于 workspace 根目录内。
- Source Memory 的既有内容不得覆盖；Topic transaction 不得产生部分提交。
- MCP server 不保存 API key，也不读取 benchmark runner 的模型配置。

## 6. Claude Code plugin

plugin 提供一份简短的项目级使用说明，使 Claude Code 知道：

- 回答历史问题前可以先查看 `memory_status`，再按需要使用 grep、BM25、embedding 和定向读取；
- 需要形成新记忆时，先通过 `memory_archive` 固化相关来源，再用 `memory_apply_patch` 更新 Topic；
- 不修改 Source 和派生文件；
- 收到 validation error 后只修正对应 Topic patch，不绕过 Runtime 校验。

plugin 不规定固定的检索步骤，也不强制每轮对话写入 memory。

## 7. README 与文档入口

根目录 README 按 GitHub 项目的使用顺序组织：

1. 方法范围与当前状态。
2. 安装与五分钟 quick start。
3. 在当前 Claude Code 会话中注册和使用 MCP。
4. Memory workspace 结构以及 Topic、Source、Timeline、Recent、Relations 的职责。
5. grep、BM25、embedding 三种检索能力。
6. Python API 与实验 runner。
7. benchmark、结果可比性和开发测试。
8. 文档导航、限制和安全边界。

`docs/Model-Aligned-Wiki.html` 继续作为研究文档入口；新增 Claude Code 集成说明并从该入口和 README 链接。实验文档仍描述隔离进程，不将其改写成交互 MCP 流程。

## 8. 验证要求

实现完成必须验证：

1. 在全新 virtual environment 中能够 build、install 并执行 `agent-memory --help`。
2. `agent-memory init` 创建可被现有 Runtime 打开的 workspace。
3. stdio MCP 能列出并调用上述工具。
4. 合法 Topic patch 能提交并更新 Timeline、Recent、Relations 和索引。
5. 非法 path、Source 修改、无效引用和格式错误均被拒绝，workspace 不产生部分更新。
6. grep、BM25 和已配置的 embedding retrieval 能返回对应 memory block。
7. Claude Code 能注册该 MCP server，并在新的交互会话中完成一次 Source 追加、Topic 修改和后续检索。
8. 现有单元测试、portable-layout check 和 hash-locked evaluator 校验保持通过。

## 9. 完成标准

一个新用户从 clone 或 pip install 开始，只需要初始化 workspace 并执行一次 `claude mcp add`，之后即可在自己正在使用的 Claude Code 会话中读写 Agent Memory Harness。实验 runner 继续使用隔离 Claude Code 进程，两条路径共享相同的 memory schema、validator、retrieval 和派生视图实现。
