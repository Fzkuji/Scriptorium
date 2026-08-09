# Scriptorium 与 Claude Code 集成技术规格

状态：已实现。实现见 `code/scriptorium/`（CLI 与 MCP server）、`code/src/retrieval/layers.py`（分层）、`code/src/management/transaction.py`、`code/src/management/patching.py`、`code/src/retrieval/inspect.py` 与 `claude-plugin/`。

与本规格的差异：`memory_update` 的 patch 只支持 create、update、delete；`scriptorium validate` 在 scratch 副本中重建派生视图；plugin 的 `author` 按 manifest schema 写成 object。

本文定义 Scriptorium 的可安装 Python package、外部 stdio MCP server、Claude Code plugin 和交付文档。实现者应以本文为接口规格，不改变已有 Topic block contract、benchmark evaluator 或实验结果。

## 1. 设计目标

系统保留两条运行路径：

1. **实验路径**：benchmark runner 通过 Claude Agent SDK 启动隔离的 Claude Code 子进程，显式传入模型、endpoint、预算和 prompt，继续用于 LoCoMo、LongMemEval 与消融实验。
2. **交互路径**：用户正在使用的 Claude Code 会话连接本地 stdio MCP server。当前 Claude 负责检索决策和 Topic 语义编辑；Memory Runtime 负责 Source 归档、格式校验、ID 生成、派生视图、原子提交和可选 Git commit。

两条路径必须共享现有 `code/src/management`、`code/src/markdown`、`code/src/retrieval` 和 `code/src/runtime`，不得复制一套交互版 memory schema 或 lifecycle。

交互路径不启动第二个 LLM，不要求额外 API key，不读取或修改 Claude Code 的登录配置。Scriptorium 自身的参数只通过函数参数、CLI 参数或配置文件传入，不新增环境变量。

## 2. 非目标

第一版不实现以下功能：

- 自动监听并保存 Claude Code 的每一条消息；
- 读取 `~/.claude/projects/**/transcript.jsonl`；
- SessionStart、PostToolUse 或 Stop hook；
- 后台定时整理和常驻 daemon；
- 远程 HTTP memory service；
- 自动选择 grep、BM25 或 embedding 的固定路由；
- 交互 MCP server 内部的 LLM reconciliation；
- 对非 Claude Code Agent 框架提供专用 adapter。

MCP 是标准协议，其他支持 MCP 的 Agent 可以复用该 server。框架专用 transcript ingestion 应在后续 adapter 中实现，不能进入第一版核心接口。

## 3. 当前实现与目标接口的对应关系

| 现有模块 | 保留能力 | 需要的最小改动 |
|---|---|---|
| `management.workspace.MemoryWorkspace` | staging、Topic 校验、派生视图、原子安装 | 增加结构化 transaction API；外部 MCP 不调用 `shell()` |
| `management.source_archive.SourceArchiveMixin` | `SourceRecord` 归档和稳定 Source link | 支持写入 stage root，使 Source 与 Topic 同时提交 |
| `management.topic_normalization.TopicNormalizationMixin` | 临时 ID 物化、Source handle 转 link、链接校验 | 支持 transaction-local Source label 的预解析 |
| `runtime.derived_views.rebuild_derived_views` | Timeline、Recent、Relations | 直接复用，不建立第二种派生格式 |
| `retrieval.views` | 文件枚举与受限读取 | 提取为不依赖 Claude SDK wrapper 的公共函数 |
| `retrieval.bm25.MemoryBM25Index` | BM25 检索和日期范围 | 直接复用，默认 `persist=False` |
| `retrieval.embedding.MemoryEmbeddingIndex` | dense retrieval | 仅在 backend 可用时启用；不作为基础安装的强制运行条件 |
| `management.tools`、`retrieval.tool_server` | Claude Agent SDK 内部 tool wrapper | 实验路径保留；外部 MCP 使用独立 FastMCP adapter |
| `runtime.state.RuntimeStateStore` | runtime state 与 Git commit | 成功事务后复用 `git_commit()` |

现有 `MemoryWorkspace.shell()` 允许在 stage 中执行任意 shell command，只能继续服务受控实验 Agent，不能暴露为用户级 MCP tool。MCP 写入必须调用新的结构化 transaction API。

## 4. 安装包与公共入口

### 4.1 Python package

仓库根目录新增 `pyproject.toml`：

- distribution name：`scriptorium`；
- Python package：`scriptorium`；
- console script：`scriptorium = scriptorium.cli:main`；
- Python 版本：与仓库当前支持版本一致，首版声明 `>=3.12`；
- 使用 setuptools 的 dynamic dependencies 从现有 `requirements.txt` 读取依赖，不重复维护两份依赖版本。

公开 Python API：

```python
from scriptorium import (
    BuildConfig,
    MemoryConfig,
    MemoryWorkspace,
    QueryConfig,
    build_memory,
    collect_answer,
)
```

现有实现仍位于 `code/src/`。实现时增加轻量 facade，使外部用户不需要 `from src import ...`；仓库内部 runner 和 hash-locked evaluator 暂不做全量 import rename。

### 4.2 CLI

只增加三个命令：

```text
scriptorium init WORKSPACE
scriptorium validate --workspace WORKSPACE
scriptorium mcp --workspace [NAME=]PATH ... [--git-commit auto|on|off]
```

- `init` 创建标准 workspace 和 `.scriptorium/runtime.json`。如果目录已含 memory 文件或运行时目录，只校验，不覆盖。
- `mcp` 在启动时创建缺失的 workspace，因此 `init` 是可选的。相对路径按会话所在仓库根解析；自动创建的 workspace 自带内容为 `*` 的 `.gitignore`。
- `validate` 解析所有 Topic/Core、检查 Source 与 block links，并重新计算派生视图到临时目录；成功时不改文件。
- `mcp` 通过 stdin/stdout 运行 FastMCP server，日志只能写 stderr。
- `--git-commit auto` 为默认值：workspace 是 Git repository 时提交，否则正常完成事务；`on` 要求 Git 可用，`off` 不提交。

不增加全局 Agent Memory 配置文件、daemon 或 Web server。

## 5. Workspace contract

```text
memory/
  core.md
  topics/**/*.md
  sources/<provider>/<thread-id>.md
  timeline/**/*.md
  recent_events.jsonl
  relations.json
  .scriptorium/runtime.json
```

- `topics/**/*.md` 与 `core.md` 是 Claude 可提出修改的权威语义状态。
- `sources/**` 是 append-only evidence，只能通过 transaction API 添加。
- `timeline/**`、`recent_events.jsonl`、`relations.json` 和检索 index 是 Runtime 派生状态。
- Topic memory unit、evidence footnote、八位十六进制 block ID、临时 `new-block-*`/`new-evidence-*`、时间精度和 `#^block-id` link 全部沿用现有方法规范。

MCP server 启动时必须验证 workspace 是目录且能够解析；不能静默创建传错的绝对路径。创建目录只由 `scriptorium init` 执行。

## 6. MCP server

使用已安装的 `mcp.server.fastmcp.FastMCP` 构建 stdio server。tool handler 只做 schema 转换、边界校验和稳定 JSON 输出，实际能力调用现有 core module。

首版暴露六个工具，不暴露 shell：

```text
memory_status
memory_list
memory_read
memory_grep
memory_search
memory_update
```

### 6.1 通用返回格式

成功：

```json
{
  "ok": true,
  "data": {},
  "revision": "<git commit or runtime revision>"
}
```

失败：

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_TOPIC_FORMAT",
    "message": "human-readable message",
    "path": "topics/personal/residence.md",
    "details": {}
  }
}
```

错误必须作为 tool error 返回，但仍保留上述 JSON，便于 Claude 定向修正。不得返回 Python traceback、stage 绝对路径或凭据。

稳定错误码至少包括：

```text
INVALID_ARGUMENT
PATH_OUTSIDE_WORKSPACE
READ_ONLY_PATH
PATCH_CONFLICT
INVALID_TOPIC_FORMAT
MISSING_SOURCE
DANGLING_BLOCK_LINK
CONCURRENT_UPDATE
EMBEDDING_UNAVAILABLE
GIT_COMMIT_FAILED
INTERNAL_ERROR
```

### 6.2 `memory_status`

输入为空。返回 workspace、revision、Topic 文件与 block 数量、Source/Recent/Timeline/Relation 数量以及 embedding 可用状态。统计只读已提交状态，不触发 rebuild 或模型调用。

### 6.3 `memory_list`

输入：

```json
{"prefix": "topics/", "include_derived": true, "limit": 200}
```

返回 workspace-relative file paths、文件字节数和可选 heading 列表。默认不列运行时目录 `.scriptorium/**`（改名前建的 workspace 沿用 `.nativemem/**`）。结果数量和输出字节都必须有硬上限。

### 6.4 `memory_read`

输入：

```json
{
  "path": "topics/personal/residence.md",
  "block_id": "8c41d20f"
}
```

定位参数规则：

- `path` 必填；
- `heading`、`block_id` 和 `offset`/`limit` 三种定位方式最多使用一种；
- `heading` 返回该 heading 到下一个同级或更高 heading 之前的内容；
- `block_id` 返回完整段落及其引用的 footnote definitions；
- 无语义定位参数时使用行窗口，不能无限制返回整个大文件；
- `sources/**` 可以读取，不能修改；
- canonical path 必须仍位于 workspace，拒绝绝对路径、`..` 和 symlink escape。

### 6.5 `memory_grep`

输入：

```json
{
  "query": "上海",
  "prefix": "topics/",
  "case_sensitive": false,
  "literal": true,
  "limit": 50
}
```

由 Python 实现 literal 或 regular-expression search，不执行系统 `grep`。返回匹配文件、行号和受长度限制的行片段。默认 `literal=true`；regex 编译失败返回 `INVALID_ARGUMENT`。

### 6.6 `memory_search`

输入：

```json
{
  "method": "bm25",
  "query": "用户什么时候搬到上海",
  "top_k": 10,
  "path_prefix": "topics/",
  "date_from": "2026-01",
  "date_to": "2026-12"
}
```

- `method` 只能是 `bm25` 或 `embedding`；grep 保持独立工具，因为其返回语义和参数不同。
- `top_k` 范围为 1–10，与现有 retrieval contract 一致。
- 日期使用已有 `YYYY`、`YYYY-MM`、`YYYY-MM-DD` inclusive overlap 语义。
- BM25 复用 `MemoryBM25Index(..., persist=False)`。
- embedding backend 未安装或未配置时返回 `EMBEDDING_UNAVAILABLE`，不得自动下载模型或改用 BM25。
- 返回每项的 score、path、block/source ID、content、time interval 和 source refs。

## 7. `memory_update`：唯一写入工具

新增、补充、改写、拆分、合并、移动和删除均通过同一个 transaction，不建立多个语义操作 API。

### 7.1 输入

```json
{
  "base_revision": "<value returned by status/read/search>",
  "sources": [
    {
      "label": "new-source-move",
      "role": "user",
      "content": "我已经搬到浦东了。",
      "observed_at": "2026-08-03T10:30:00+08:00"
    }
  ],
  "patch": "<unified diff over topics/**/*.md and/or core.md>",
  "commit_message": "Update residence memory"
}
```

约束：

- `base_revision` 必填，用于拒绝基于旧状态生成的修改。
- `sources` 可以为空；更新已有记忆时直接复用已有 Source links。
- `label` 在一次 transaction 内唯一，格式为 `new-source-[a-z0-9-]+`。
- `role` 为 `user`、`assistant`、`system` 或 `tool`。
- `content` 非空；`observed_at` 可为空，但提供时必须是 ISO 8601。
- `patch` 必填，使用 workspace-relative 路径，只允许 `topics/**/*.md` 与 `core.md`。
- `commit_message` 可选；Runtime 对长度和控制字符做限制，不允许把它解释为 shell。

### 7.2 在 Topic patch 中引用新 Source

Claude 在 footnote 的 `Sources:` 中直接写 transaction-local label：

```markdown
用户已搬到浦东。[^new-evidence-move] ^new-block-residence

[^new-evidence-move]: Time: `2026-08-03`; Sources: new-source-move
```

Runtime 必须先将 `new-source-move` 替换为本次归档生成的稳定 handle，再执行现有 Topic normalization。临时 Source label 不得出现在提交后的 Markdown 中。

### 7.3 稳定 Source identity

普通 MCP tool call 不会自动收到 Claude Code 的 `session_id` 和 `transcript_path`。第一版不伪造 Claude 原生 message ID，也不读取 Claude 私有 transcript。

Runtime 为手动提交的证据使用以下 identity：

```text
provider   = claude-code
thread_id  = first_16_hex(SHA256(canonical ordered source batch))
message_id = first_16_hex(SHA256(thread_id + ordinal + role + content + observed_at))
source_id  = claude-code/thread_id/message_id
```

canonical batch 包含按输入顺序排列的 `role`、完整 `content` 和规范化 `observed_at`。相同输入重试得到相同 Source ID，因此归档操作幂等。真正相同的证据可以共享 Source record；不同时间或不同内容不会合并。

### 7.4 事务顺序

`MemoryWorkspace` 增加一个不依赖 shell 的公共 transaction 方法，顺序固定为：

1. 获取 workspace write lock。
2. 比较 `base_revision` 与当前 revision；不一致返回 `CONCURRENT_UPDATE`。
3. 将已提交 workspace 完整复制到 stage。
4. 将 `sources` 写入 stage 中的 append-only Source tree。
5. 将 local Source labels 替换为稳定 handles。
6. 在 stage 应用并校验 unified diff；拒绝所有非 Topic/Core path。
7. 运行现有临时 block/evidence ID 物化和 Markdown normalization。
8. 解析 Topic/Core，校验 evidence、Source、Topic links、Core token limit 和全局 ID 唯一性。
9. 重建 Timeline、Recent、Relations、runtime state；使 retrieval cache 失效。
10. 使用现有 backup-and-`os.replace` 机制一次性安装 Source、Topic、Core、派生视图和 runtime state。
11. 按 `--git-commit` 策略调用 `RuntimeStateStore.git_commit()`。
12. 返回新 revision、ID 映射、变更文件和派生视图统计，释放 write lock。

任一步失败必须删除 stage 并保持已提交 workspace 字节不变。Source 也属于本次原子安装范围；不能继续使用当前 `archive_source_records()` 先写 `memory_dir`、再刷新 stage 的顺序。

### 7.5 成功返回

```json
{
  "ok": true,
  "data": {
    "source_ids": {
      "new-source-move": "claude-code/7db41a.../90b12c..."
    },
    "block_ids": {
      "new-block-residence": "8c41d20f"
    },
    "evidence_ids": {
      "new-evidence-move": "e-41b3d91a2f"
    },
    "changed_files": [
      "sources/claude-code/7db41a....md",
      "topics/personal/residence.md",
      "timeline/2026/08/03.md",
      "recent_events.jsonl",
      "relations.json"
    ],
    "git_commit": "<sha or null>"
  },
  "revision": "<new revision>"
}
```

Runtime revision 在 Git repository 中使用 `HEAD` 加 workspace fingerprint；非 Git workspace 使用纳入 Source、Topic、Core、派生视图和 runtime state 的 SHA-256 fingerprint。不能只使用文件 mtime。

## 8. 并发、Git 与恢复

- 每个 workspace 使用运行时目录下 `write.lock` 的进程级排他锁；锁只覆盖一次 update transaction。实现优先使用 Python 标准库：POSIX 使用 `fcntl.flock`，Windows 使用 `msvcrt.locking`，不为该功能增加第三方依赖。
- 读工具不获取写锁，但返回其读取时的 revision。
- `base_revision` 防止 Claude 根据旧内容覆盖另一个会话刚提交的修改。
- Git commit 仅在文件原子安装成功后执行。
- `--git-commit on` 下 commit 失败返回 `GIT_COMMIT_FAILED`。文件已经成功安装时，返回中必须明确 `memory_committed=true`、`git_committed=false`，不能声称事务整体回滚。
- `--git-commit auto` 下非 Git workspace 不报错；Git repository 中 commit 失败仍返回错误。
- MCP 不执行 `git reset`、`git checkout` 或历史删除。

Git 不能与文件安装组成跨系统原子事务，因此错误协议必须区分 memory commit 与 Git commit。实现文档和测试不得把 Git commit 失败描述成文件回滚成功。

## 9. Claude Code 集成

### 9.1 基础安装方式

发布前：

```bash
pip install git+https://github.com/Fzkuji/Scriptorium.git
```

发布后：

```bash
pip install scriptorium
```

注册（workspace 缺失时由 server 在启动时创建，`init` 可选）：

```bash
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace project=.memory --workspace global=~/memory

claude mcp get scriptorium
```

### 9.2 Plugin 内容

plugin 是可选的使用说明层，不复制 MCP server，也不自动读取 transcript：

```text
claude-plugin/
  .claude-plugin/plugin.json
  skills/agent-memory/SKILL.md
  README.md
```

`SKILL.md` 只说明：

- 何时查看 memory；
- 推荐先列出文件或检索，再定向读取，不要求固定步骤；
- 保存时调用一次 `memory_update`，把新证据和 Topic patch 放在同一 transaction；
- Source 和派生视图只读；
- validation error 由 Claude 修正 patch 后重试；
- 不要求每轮对话都检索或写入。

不在 plugin 内配置第二个 MCP server，否则会与用户通过 `claude mcp add` 注册的同名 server 重复。后续若需要 plugin 一次安装完成全部配置，应另行设计 workspace path 配置，不使用环境变量隐式传入。

## 10. 安全边界

- MCP server 不暴露 shell、任意 subprocess、任意 Python eval 或文件系统通配写入。
- 所有输入路径先解析为 canonical path，再验证位于 workspace 内且不经过 symlink。
- patch rename/copy metadata、binary patch、mode change 和 symlink creation 全部拒绝；首版只支持文本文件 create/update/delete。Topic 文件移动由同一个 patch 中“删除旧路径并以相同内容创建新路径”表达，Runtime 再重写 block links。
- 单次 Source 数量、Source 总字节、patch 字节、返回条数和返回文本必须有显式上限，并通过 CLI 常量或函数参数配置，不通过环境变量配置。
- Source 内容按不可信数据处理；plugin 不把 Source 内类似指令的文本当成执行要求。
- MCP stdout 只输出协议消息；普通日志写 stderr，且不输出完整 Source 内容。
- server 不保存 API key，不读取 benchmark model/provider 配置。

## 11. 需要新增或修改的文件

实现者可以调整具体模块名，但职责必须保持：

```text
pyproject.toml
code/scriptorium/__init__.py                # public facade, lazily resolved
code/scriptorium/cli.py                     # init / validate / mcp
code/scriptorium/mcp_server.py              # FastMCP adapter only
code/src/retrieval/layers.py                # several workspaces read as one
code/src/workspace_layout.py                # the names the runtime owns
code/src/management/workspace.py            # structured transaction
code/src/management/source_archive.py       # stage-targeted archive
code/src/retrieval/...                      # reusable read/grep/search functions
claude-plugin/.claude-plugin/plugin.json
claude-plugin/skills/scriptorium/SKILL.md
claude-plugin/README.md
README.md
docs/integrations/claude-code.md
docs/Model-Aligned-Wiki.html                 # navigation link only
```

不修改：

- `code/scripts/eval_full.py`；
- 已冻结 evaluator 及其 hash；
- 已有 formal result 文件；
- paper repository；
- 现有实验 Claude Agent SDK adapter 的行为。

## 12. 测试规格

### 12.1 Package 与 CLI

- clean virtual environment 中 build wheel、install wheel；
- `python -c "import scriptorium"`；
- `scriptorium --help`、三个 subcommand help；
- `init` 不覆盖已有 workspace；
- `validate` 不修改任何文件。

### 12.2 读取与检索

- list 隐藏 `.nativemem`；
- read 支持 line、heading、block，并附带 block footnotes；
- absolute path、`..` 和 symlink escape 被拒绝；
- grep literal/regex、大小写和结果上限；
- BM25 返回 Topic 和 Source 结果；
- 日期区间复用已有 overlap 测试；
- embedding 可用与 unavailable 两条路径都测试，测试期间不下载模型。

### 12.3 写入事务

- Source + 新 Topic block 同时提交；
- transaction-local Source/block/evidence labels 全部物化；
- 相同 Source 输入重试不重复追加；
- 补充、拆分、合并、移动、删除均沿用一个 `memory_update`；
- Timeline、Recent、Relations 和 retrieval index 失效状态正确；
- Topic 格式错误、Source 缺失、dangling link、Core 超限全部回滚；
- Source 直接修改、派生文件修改、binary/mode/symlink patch 被拒绝；
- 安装中途失败恢复 Source、Topic、Core、派生视图和 runtime state；
- 旧 `base_revision` 返回 `CONCURRENT_UPDATE`；
- 两个并发 writer 只有一个基于同一 revision 提交成功；
- Git auto/on/off 与 Git commit 失败状态分别测试。

### 12.4 MCP 与 Claude Code

- FastMCP stdio initialization、tools/list 和每个 tool call；
- stdout 无日志污染；
- `claude mcp add` 后 `claude mcp get agent-memory` 可见；
- 新 Claude Code 交互会话完成一次 status → read/search → update → read；
- `claude plugin validate claude-plugin` 通过；
- plugin 未安装时 MCP 仍完整可用。

### 12.5 回归

- 现有 management、markdown、retrieval、runtime 和 scripts tests；
- portable layout check；
- evaluator hash check；
- 不需要为此次集成重跑 LoCoMo 或 LongMemEval，因为本设计不改变 benchmark 算法。若 core module 行为发生变化，再运行最小既有 fixture 回归，而不是直接进行完整 benchmark。

## 13. README 与文档结构

根 README 按实际使用顺序编写：

1. Scriptorium 的范围和两个运行模式。
2. `pip install` 与最小依赖。
3. 当前 Claude Code 会话的三条 quick-start 命令。
4. 一个完整的检索和写入示例。
5. Workspace 结构与权威/派生状态。
6. grep、BM25、embedding 的差异和可选依赖。
7. Python API 和实验 runner。
8. benchmark 结果入口与可比性限制。
9. 安全边界、已知限制和开发测试。
10. 文档导航。

`docs/integrations/claude-code.md` 保存完整安装、移除、scope、permission、故障排查和 plugin 使用说明。`docs/Model-Aligned-Wiki.html` 只增加该页面入口，不复制全部内容。实验文档继续描述隔离 Claude Code 子进程，不能改写成交互 MCP 流程。

## 14. 实现顺序与完成标准

实现顺序必须遵守依赖关系：

1. 提取不依赖 Claude SDK wrapper 的 read/grep/search core functions。
2. 将 Source archive 改为 stage-targeted，并实现结构化原子 transaction。
3. 补并发 revision、write lock 和 Git 状态返回。
4. 增加 package facade 与 CLI。
5. 增加 FastMCP adapter。
6. 增加 plugin skill、README 和 integration docs。
7. 执行测试规格和 clean-install smoke test。

完成时必须满足：新用户安装 package、初始化 workspace、执行一次 `claude mcp add` 后，可以在当前 Claude Code 会话中检索 Source/Topic、提交一次带来源的 Topic 修改，并立即读取 Runtime 自动更新后的 Timeline、Recent 和 Relations；实验 runner 仍按原配置启动隔离 Claude Code 子进程。

## 15. 外部接口依据

- Claude Code 的 stdio MCP 注册、scope 和 plugin-provided MCP 行为：<https://code.claude.com/docs/en/mcp>
- Claude Code plugin 目录、manifest、skill 和 MCP 配置位置：<https://code.claude.com/docs/en/plugins-reference>
- Claude Code hook 才会收到 `session_id` 与 `transcript_path`：<https://code.claude.com/docs/en/hooks>
- Claude Code 本地 transcript 的存储与生命周期：<https://code.claude.com/docs/en/sessions>
