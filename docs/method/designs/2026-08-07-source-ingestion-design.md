# 来源接入设计


四种会话来源的实测结构。全部结论来自本机真实文件，非文档推断。

**item** 指基准数据集文件中的一条样本。LoCoMo 的 item 是一整段人物对话，标识为 `sample_id`（如 `conv-26`），内含 19 至 32 个会话。LongMemEval 的 item 是一道题，标识为 `question_id`（如 `e47becba`），内含约 53 个会话，其中一个为答案所在，其余为干扰项。

因此"item 内唯一"意为：该标识在单条样本内部不重复，但另一条样本中会出现同名的另一个对象。

---

## 1. 对照

| | Claude Code | Codex | LoCoMo | LongMemEval |
|---|---|---|---|---|
| 存储 | `~/.claude/projects/<编码路径>/<uuid>.jsonl` | `~/.codex/sessions/YYYY/MM/DD/rollout-<时间>-<uuid>.jsonl` | 单个 JSON，10 item | 单个 JSON，500 item |
| 会话标识 | 文件名 uuid | 文件名 uuid | `session_N` 键 | `haystack_session_ids[i]` |
| 会话标识唯一性 | 全局 | 全局 | item 内 | **跨 item 重复** |
| 消息标识 | `uuid` | `payload.id` | `dia_id` | **无** |
| 消息标识唯一性 | 全局 | 全局 | **item 内** | — |
| 归属信息 | 记录级 `cwd` | `session_meta.payload.cwd` | `sample_id` | `question_id` |
| 时间粒度 | 每条消息 | 每条消息 | 每会话 | 每会话 |
| 角色字段 | 顶层 `type` | `payload.role` | `speaker`（人名） | `role` |
| 正文字段 | `message.content`（多态） | `payload.content[].text` | `text` | `content` |
| 规模 | 单文件最大 380 MB | 2298 文件 | 2.7 MB / 5882 turn | 265 MB / 246750 turn |

---

## 2. Claude Code

### 2.1 目录

```
~/.claude/projects/<编码路径>/
  <session-uuid>.jsonl
  <session-uuid>/
    subagents/agent-<id>.jsonl
    subagents/agent-<id>.meta.json
    tool-results/
    workflows/
  memory/*.md
  sessions-index.json
```

**目录名编码不可逆。** 规则是把绝对路径中每段非字母数字字符替换为单个 `-`。三个实例：

| 目录名 | 真实路径 | 说明 |
|---|---|---|
| `-Users-fzkuji-Documents-LLM-Uncertainty` | `/Users/fzkuji/Documents/LLM Uncertainty` | 空格变 `-` |
| `-Users-fzkuji-PycharmProjects-AML-CityU-github-io` | `/Users/fzkuji/PycharmProjects/AML-CityU.github.io` | `/`、`-`、`.` 三种字符都变 `-` |
| `-private-tmp-claude-501--Users-fzkuji-...` | 含字面 `-Users-fzkuji` 的路径 | 出现 `--` |

**一个目录不对应一个项目。** 目录按会话开始时的 cwd 命名，会话中途 `cd` 不改归档位置。单个文件内 `cwd` 分布实例：28788 条 `OpenProgram`、11680 条 `OpenProgram/web`、2098 条 `/Users/fzkuji`。

**两种布局并存。** `-Users-fzkuji` 下有 24 个散放 jsonl；11 个 `-Users-fzkuji-Documents-*` 目录下 0 个 jsonl，只有 `memory/`。

`sessions-index.json`：`{version, originalPath, entries[]}`，条目含 `sessionId`、`fullPath`、`fileMtime`、`firstPrompt`、`summary`、`messageCount`、`created`、`modified`、`gitBranch`、`projectPath`、`isSidechain`。索引不做垃圾回收，存在指向已删除文件的条目。

### 2.2 记录类型

| type | 出现量（全部会话） | 含 `uuid` | 含 `sessionId` |
|---|---|---|---|
| `assistant` | 52395 | 是 | 是 |
| `user` | 33438 | 是 | 是 |
| `attachment` | 12101 | 是 | 是 |
| `system` | 11153 | 是 | 是 |
| `last-prompt` | 8749 | 否 | 是 |
| `mode` | 8649 | 否 | 是 |
| `permission-mode` | 8094 | 否 | 是 |
| `ai-title` | 7954 | 否 | 是 |
| `bridge-session` | 7537 | 否 | 是 |
| `custom-title` | 6533 | 否 | 是 |
| `agent-name` | 6408 | 否 | 是 |
| `queue-operation` | 6525 | 否 | 是 |
| `file-history-snapshot` | 4983 | 否 | 否 |
| `file-history-delta` | 503 | 否 | 否 |
| `frame-link` | 5 | 否 | 否 |

对话轮次只有 `user` 与 `assistant`。本机不存在 `summary` 类型记录，会话标题在 `ai-title` / `custom-title` 与 `sessions-index.json` 中。

`system` 的 `subtype`：`turn_duration` 1503、`stop_hook_summary` 1477、`away_summary` 294、`compact_boundary` 43、`local_command` 17、`scheduled_task_fire` 4、`model_consent_fallback` 3、`model_refusal_fallback` 2、`bridge_status` 1。

### 2.3 消息标识

`uuid` 与 `parentUuid` 在 `user` / `assistant` / `attachment` / `system` 上 100% 存在，其余类型全无。

`parentUuid` 构成链，用户与助手严格交错，根节点为 `null`。`compact_boundary` 记录带 `logicalParentUuid`，压缩处链条跳跃，需经此字段回溯。

`sessionId` 单文件唯一且等于文件名。部分记录另有蛇形 `session_id`，值相同。

### 2.4 正文

助手：`message.content[]` 中 `type == "text"` 的块取 `.text`。块类型分布（单个大文件）：`tool_use` 12688、`thinking` 6303、`text` 4857。

用户：`message.content` 多态，必须分支。

| 形态 | 出现量 | 取法 |
|---|---|---|
| `tool_result` 数组 | 12688 | 非人类文本 |
| 字符串 | 1666 | 直接取值 |
| `text` 块数组 | 363 | `content[].text` |
| 混合含 image | 319 | 取 text 块 |

约 84% 的 `user` 记录是工具结果。真实人类输入的标志：`origin: {"kind":"human"}` 与 `promptSource: "typed"`。

`tool_use` 块：`{type, id, name, input}`。`tool_result` 块：`{type, tool_use_id, is_error, content}`，经 `tool_use_id` 配对。同一记录另有顶层 `toolUseResult`，结构随工具而变，比扁平化的 `content` 信息更全。

### 2.5 时间

字段 `timestamp`，ISO-8601 UTC 毫秒带 `Z`。在 `user` / `assistant` / `attachment` / `system` / `queue-operation` / `file-history-delta` 上 100% 存在。`file-history-snapshot` 无顶层时间，嵌在 `snapshot.timestamp`。状态类记录无时间。

### 2.6 接入注意

- 单文件可达 380 MB / 80467 行，须逐行流式读取
- 子 agent 记录不在主文件内，位于 `<session-uuid>/subagents/agent-*.jsonl`，且复用父会话的 `sessionId`。主文件全部记录的 `isSidechain` 均为 `false`
- 按 `uuid` 去重，注意 `supersedesUuids`（assistant）与 `retractedMessageUuids`（system）
- 项目归属只能从记录级 `cwd` 取，不可解析目录名

---

## 3. Codex

### 3.1 目录

```
~/.codex/sessions/YYYY/MM/DD/rollout-<YYYY-MM-DDTHH-MM-SS>-<uuid>.jsonl
```

2298 个文件全部核对：目录日期与文件名时间戳一致（0 例外），文件名 uuid 与首条记录 `payload.id` 一致（0 例外）。uuid 为 UUIDv7，字典序近似时间序。

文件名时间戳是本地时间，记录内 `timestamp` 是 UTC。

`~/.codex/` 下其他相关物：

| 路径 | 内容 |
|---|---|
| `archived_sessions/` | 26 个同格式文件，无日期嵌套 |
| `history.jsonl` | 仅用户提示，`{session_id, ts, text}` |
| `session_index.jsonl` | `{id, thread_name, updated_at}`，覆盖不全 |
| `external_agent_session_imports.json` | 导入的外部 agent 会话 |
| `memories/rollout_summaries/` | 已有记忆系统产出的摘要 |

### 3.2 记录类型

每条记录恒为 `{timestamp, type, payload}` 三键。六个近期大文件约 76000 条记录的分布：

| type | 数量 |
|---|---|
| `response_item` | 47323 |
| `event_msg` | 26636 |
| `turn_context` | 1545 |
| `world_state` | 266 |
| `compacted` | 147 |
| `inter_agent_communication_metadata` | 150 |
| `session_meta` | 30 |

`response_item.payload.type` 分布：

| payload.type | 数量 | 性质 |
|---|---|---|
| `reasoning` | 14992 | 内部，`encrypted_content` 不可解 |
| `custom_tool_call` | 10778 | 内部 |
| `custom_tool_call_output` | 10775 | 内部 |
| `message` | 6098 | **对话** |
| `function_call` | 2256 | 内部 |
| `function_call_output` | 2256 | 内部 |
| `agent_message` | 150 | 子 agent |
| `tool_search_call` / `_output` | 9 | 内部 |

### 3.3 session_meta

必有字段：`id`、`timestamp`、`cwd`、`originator`、`cli_version`、`source`、`model_provider`。

版本相关字段：`git`（约 60%）、`instructions`（旧）、`base_instructions`（新）、`session_id`、`thread_source`、`history_mode`、`context_window`、`multi_agent_version`、`forked_from_id`、`parent_thread_id`、`agent_path`、`agent_nickname`。

跨 15 个 cli_version（0.53.0 至 0.146.0）schema 持续变化，除必有字段外均应按可选处理。

**并非每文件一条。** 恢复的会话在开头堆叠全部祖先会话的 meta，实例：某文件第 0 至 9 行均为 `session_meta`，顶层 `timestamp` 相同，第 0 行 `payload.id` 等于本文件 uuid，第 1 至 9 行为逆时序的祖先会话。

`git` 不可靠：25 个抽样文件中 8 个缺失；存在的 18 个中 7 个无 `repository_url`。仅在会话开始时记录一次。

### 3.4 消息标识

每个 `response_item.payload` 均有 `id`，按前缀区分类型：

| 前缀 | 类型 |
|---|---|
| `msg_` | message |
| `rs_` | reasoning |
| `ctc_` / `ctco_` | custom tool call / output |
| `fc_` / `fco_` | function call / output |

`msg_` 有两种形态：服务端生成的不透明串（助手），本地生成的 UUIDv7（用户、developer）。

`call_id` 在四种调用/输出类型上，配对调用与结果。

`internal_chat_message_metadata_passthrough.turn_id` 在每个 `response_item.payload` 上，构成轮次分组。

**`compacted.replacement_history[]` 原样内嵌早前 `response_item` 的 payload，连同原 `id`。** 顺序扫描会看到同一 id 两次，须按 id 去重或跳过 `compacted`。

行序号不可作稳定键，恢复会话时祖先 meta 会移动偏移。

### 3.5 正文

```
payload.content[i].text
```

用户与 developer、工具输出为 `type == "input_text"`；助手为 `type == "output_text"`。均在 `type == "response_item"` 且 `payload.type == "message"` 下，按 `payload.role` 区分 `user` / `assistant` / `developer`。

`content` 恒为列表。用户消息常含多段：AGENTS.md 块、`<environment_context>` 块，之后才是真实输入。干净的用户输入另见 `event_msg.user_message.message`，无前置块。

助手 `message` 可带 `phase: "commentary"`，区分叙述与最终答复。`event_msg.task_complete.last_agent_message` 存该轮最终答复。

内部记录：`role == "developer"`、`reasoning`、各类工具调用与输出、`agent_message`。

### 3.6 turn_context

扁平 payload，无 `payload.type`。每轮开始时发出。

```json
{"turn_id":"019fc6cd-11b9-7ae1-9553-2c9e8740f89c","cwd":"...",
 "workspace_roots":["..."],"current_date":"2026-08-03",
 "timezone":"Asia/Shanghai","approval_policy":"on-request",
 "sandbox_policy":{},"model":"gpt-5.6-sol","effort":"max", ...}
```

`turn_id` 是轮次分组键，同时出现在 `turn_context.turn_id`、`event_msg` 的 `task_started` / `task_complete` / `turn_aborted` / `patch_apply_end`、以及每个 `response_item` 的 `internal_chat_message_metadata_passthrough.turn_id`。一轮等于一次用户输入加全部后续记录，由 `task_started` 与 `task_complete` 括起。UUIDv7，可排序。单文件观察到 509 个不同 turn_id。

`model` 与 `effort` 逐轮记录，单会话可跨多个模型。

### 3.7 时间

顶层 `timestamp` 在每条记录上，ISO-8601 UTC 毫秒带 `Z`，为写入时刻。

`payload.timestamp` 仅在 `session_meta` 上，为会话开始时刻，比该记录顶层时间略早。

其他时间字段为 unix 数值：`task_started.started_at`（秒）、`task_complete.completed_at` / `duration_ms` / `time_to_first_token_ms`、`sub_agent_activity.occurred_at_ms`、`history.jsonl` 的 `ts`（秒）。

### 3.8 归属

`session_meta.payload.cwd` 抽样 100% 存在。相当比例的会话 `cwd` 为家目录 `/Users/fzkuji`，非项目作用域。

`cwd` 亦出现在每个 `turn_context` 与 `event_msg.thread_settings_applied.thread_settings.cwd`，设计上逐轮可变。六个抽查文件中未观察到漂移。

---

## 4. LoCoMo

### 4.1 文件

`code/benchmarks/locomo/data/locomo10.json`，2.7 MB，JSON 列表，10 个 item。

路径硬编码于 `scripts/runners/conversation/config.py:9` 与 `scripts/runners/ablation/outputs.py:13`。

### 4.2 item 结构

```json
{
  "sample_id": "conv-26",
  "qa": [],
  "conversation": {},
  "event_summary": {},
  "observation": {},
  "session_summary": {}
}
```

`sample_id` 取值：`conv-26,30,41,42,43,44,47,48,49,50`。

`qa` 共 1986 个问题，**无问题级 id**，只能用 `(sample_id, qa 下标)` 定位。

### 4.3 会话

`conversation` 是扁平字典：

```
speaker_a: "Caroline"
speaker_b: "Melanie"
session_1_date_time: "1:56 pm on 8 May, 2023"
session_1: [turn, turn, ...]
session_2_date_time: ...
session_2: [...]
```

每 item 会话数：19、19、32、29、29、28、31、30、25、30。总计 5882 turn。

**日期键与会话数组不是一一对应。** `conv-26` 有 `session_1` 至 `session_19` 数组，却有 `session_1_date_time` 至 `session_35_date_time`。枚举必须以 `session_N` 数组为准，再查对应日期键。

时间格式 `"1:56 pm on 8 May, 2023"`，12 小时制。无 turn 级时间。

### 4.4 消息

```json
{"speaker": "Caroline", "dia_id": "D1:1", "text": "<text>"}
```

字段出现率（5882 turn）：`speaker` / `dia_id` / `text` 全在；`blip_caption` 1226、`img_url` 910、`query` 888、`re-download` 206。

**`dia_id` 跨 item 重复。** 5882 个 turn 只有 1033 个不同 `dia_id`，其中 871 个在多个 item 中重复出现。`D1:1` 出现 10 次，每 item 一次。item 内唯一，全局不唯一。最小全局唯一地址为 `(sample_id, dia_id)`。`speaker` 同样跨 item 重复。

格式严格匹配 `D\d+:\d+`，100% 符合。

### 4.5 证据链接

```json
{"question": "...", "answer": "7 May 2023", "evidence": ["D1:3"], "category": 2}
```

`evidence` 为裸 `dia_id` 列表，作用域限于所在 item。1986 个问题全部带 `evidence`。

`answer` 仅在 1542 个问题上；其余 446 个（category 5，对抗类）带 `adversarial_answer`。

---

## 5. LongMemEval

### 5.1 文件

`code/benchmarks/longmemeval/data/longmemeval_s_cleaned.json`，**265 MB**，JSON 列表，500 个 item。整体 `json.load` 开销显著。

路径与 `EXPECTED_LONGMEMEVAL_SIZE = 500` 断言在 `scripts/runners/longmemeval/support.py:21-25`。

同目录 `longmemeval_oracle.json`（15 MB，同 500 个 `question_id`，仅保留答案会话）。`benchmarks/longmemeval/data.json` 为其副本。测试夹具 `tests/fixtures/longmemeval_s_tiny.json`。

### 5.2 item 结构

九个字段，500 个 item 全有：

```json
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "question_date": "2023/05/30 (Tue) 23:40",
  "answer": "Business Administration",
  "answer_session_ids": ["answer_280352e9"],
  "haystack_dates": [],
  "haystack_session_ids": [],
  "haystack_sessions": []
}
```

`question_id` 为 8 位十六进制，500 个全不重复，是真正的全局 item 标识。

`question_type` 分布：multi-session 133、temporal-reasoning 133、knowledge-update 78、single-session-user 70、single-session-assistant 56、single-session-preference 30。

### 5.3 会话

三个并行数组按下标对齐，500 个 item 全部长度相等。总计 23867 会话、246750 turn。

会话 id 两种形态：干扰项如 `sharegpt_yywfIrx_0`、`85a1be56_1`；支撑会话为 `answer_<8hex>`。

**`haystack_session_ids` 跨 item 重复。** 23867 个位置只有 19195 个不同 id，3942 个 id 被多个 item 复用（同一 ShareGPT 干扰会话被植入多个问题的干草堆）。

时间格式 `"2023/05/20 (Sat) 02:21"`，24 小时制，每会话一个，无 turn 级时间。

### 5.4 消息

```json
{"role": "user", "content": "<text>", "has_answer": false}
```

246750 个 turn 中 `role` 与 `content` 全有，`has_answer` 仅 10960 个。

**turn 无任何 id 字段。** 唯一定位方式是位置：`(question_id, 会话下标, turn 下标)`。

`has_answer` 只出现在答案会话内的 turn 上。前 100 个 item 的 4824 个会话中，170 个答案会话的 turn 全带该标志，4654 个非答案会话全无。字段存在与否标记会话是否为支撑会话，字段取值标记具体 turn。

`role` 取值 `user` / `assistant`，大致均衡。无图像字段。

### 5.5 证据链接

`answer_session_ids` 为会话 id 列表，500 个 item 全部核对为该 item `haystack_session_ids` 的子集（0 例外）。数量分布：1 个 ×176、2 个 ×250、3 个 ×41、4 个 ×19、5 个 ×11、6 个 ×3。

链接仅到会话级。更细的 turn 级信号是 `has_answer: true`，无 LoCoMo `evidence` 那样的 id 指针。

---

## 6. 标识唯一性汇总

| 来源 | 消息标识 | 唯一范围 | 达成全局唯一所需 |
|---|---|---|---|
| Claude Code | `uuid` | 全局 | 自身即可 |
| Codex | `payload.id` | 全局 | 自身即可，需按 id 去重 |
| LoCoMo | `dia_id` | item 内 | 加 `sample_id` |
| LongMemEval | 无 | — | `question_id` + 会话下标 + turn 下标 |

---

## 7. 定位

### 7.1 每条消息都可寻址

四种来源均可为每条消息给出一个确定标识，无一例外。

Claude Code 与 Codex 自带全局唯一的消息标识，直接沿用。

LoCoMo 的 `dia_id` 在 item 内唯一，加上 `sample_id` 即唯一。跨 item 重复本身不构成问题，各 item 独立评测、互不参照。

LongMemEval 无消息级标识。位置即标识：会话下标与 turn 下标在 item 内确定一条消息，加上 `question_id` 后唯一。

### 7.2 作用域

标识的唯一性只需在其所属记忆库内成立。

一个记忆库对应一个作用域：基准数据集的一个 item，或本地会话的一个项目。不同作用域各有记忆库，不共享地址空间。竞赛服务按 `user_id` 分库即此模式，`user_id` 形如 `eval:run_abc123:locomo:conv-0`，数据集与 item 已编入其中。

本地会话的作用域取自 `cwd`。Claude Code 的 `cwd` 在记录级，Codex 在 `session_meta` 与 `turn_context` 上，两者均可取得。目录名不可用于此判断，编码不可逆。

作用域外置后，地址不需要归属段：

```
<session>/<message>
```

### 7.3 地址构成

地址由会话段与消息段组成，两段均取自输入，不哈希、不重编号。

| 来源 | 会话段 | 消息段 | 示例 |
|---|---|---|---|
| Claude Code | 文件名 uuid | `uuid` | `4854e5c9/369a4104-8a9f-4d2e-b1c3-7e5a9f2b1d40` |
| Codex | 文件名 uuid | `payload.id` | `019a5363/msg_019fc6cd-65f2-7b30-9269-7da6dc9f441b` |
| LoCoMo | `session_N` | `dia_id` | `session_1/D1:1` |
| LongMemEval | 会话下标 | turn 下标 | `0031/0007` |

LongMemEval 两段均为四位定宽下标，从 `0000` 起。其余来源沿用原值。

作用域不入地址（见 7.2），由记忆库本身承载。

### 7.4 定位

地址自身编码位置，不维护地址到文件的映射表。

**会话段确定文件。** Claude Code 与 Codex 的会话段即文件名主体，按接入时记录的根目录拼接即得路径。LoCoMo 与 LongMemEval 的会话段是数据集文件内的键或下标，文件由记忆库确定。

**消息段确定文件内位置。** 分两种：

LongMemEval 的消息段是 turn 下标，直接索引数组，无需扫描。

其余三种的消息段是标识，需在文件内匹配。做法是逐行扫描，取首个标识相符的行。

**扫描代价已实测可接受。** 本机最大的 Claude 会话文件 395 MB、80532 行，逐行遍历耗时 0.19 秒。定位发生在写入校验与脚注回溯时，非高频路径，不引入偏移索引或缓存。

不缓存的直接收益是正确性：Claude Code 会话文件在会话进行中持续追加，任何预存的行号都会过期，而按标识扫描始终命中正确的行。

**匹配是字面比较。** 逐行判断目标标识字符串是否出现，命中即返回该行。无分词、无排序、无打分、无向量。

### 7.5 定位与检索

定位的输入是地址，输出是唯一确定的一条消息，用于脚注回溯与写入校验。

检索的输入是自然语言，输出是若干条按相关度排序的候选，用于在不知道地址时寻找相关内容——Writer 写入前查找相关旧记忆，回答问题时查找支撑证据。

两者输入与输出均不同，互不替代。本节只规定定位。

---

## 8. 写入

流程示意见 [memory-write-flow.html](../memory-write-flow.html)。

### 8.1 原则

记忆只追加，不改写。

已写入的块保持原样。事实发生变化时追加新块记录新版本，旧版本保留。旧版本是真实发生过的陈述，保留它才能回答"最初的说法是什么"。

由此得到四条约束：

| 约束 | 作用 |
|---|---|
| 只许局部编辑，禁止整文件重写 | 块标识天然留在原地 |
| 已有标识的行不修改、不删除、不重新编号 | 引用不断 |
| 事实变化写成新块 | 版本可追溯 |
| 行尾有标识即旧块，无标识即新块 | 新旧判断归代码 |

第四条消除了对模型的一项依赖。判断"这句话对应哪个旧块"不再需要语义理解，是字符串检查。

### 8.2 块结构

一条记忆由正文行与脚注定义两部分组成。

```markdown
Calvin is a musician who made a big life change in March 2023 by
acquiring a mansion in Japan, arranged by his agent.[^e-42b373e894] ^531d2845

[^e-42b373e894]: Time: `2023-03`; Sources: [locomo/thread_37d993f7a9d6/msg_6c0a984c5fc6](../../sources/locomo/thread_37d993f7a9d6.md#source-5563975dea57329b)
```

两个标识分工不同：

| 标识 | 位置 | 作用 | 生成方 |
|---|---|---|---|
| `^531d2845` | 正文行尾 | 块的身份，供派生视图与其他块引用 | 代码，内容哈希 |
| `[^e-42b373e894]` | 正文内与脚注定义 | 脚注编号，连接正文与其时间来源 | 代码，内容哈希 |

脚注定义中的 `Time` 与 `Sources` 由模型给出，链接形式由代码渲染。

### 8.3 职责划分

模型只做两件事：写正文、给出该条正文的时间与来源。

| 事项 | 执行方 | 原因 |
|---|---|---|
| 正文 | 模型 | 需要理解与归纳 |
| `Time` | 模型 | 事实发生时间藏在话里，与消息时间戳可以不同 |
| `Sources` | 模型 | 一句总结出自本批哪几条消息，只有写的人知道 |
| 块标识 | 代码 | 内容哈希 |
| 脚注编号 | 代码 | 内容哈希 |
| 来源链接与锚点 | 代码 | 机械转换 |
| 新旧判断 | 代码 | 行尾有无标识 |
| 派生视图 | 代码 | 按时间与引用关系重排 |

`Time` 不可由代码推导。原文出现"我去年修了辆车"时，事实发生在去年而消息发于今年，时间戳给不出答案。

`Sources` 同理。本批二十条消息，模型写出一句总结，代码无法判定它出自其中哪几条。

### 8.4 模型的输出形式

模型写出的脚注使用本地编号，不写完整路径：

```markdown
Calvin 的公寓可以俯瞰城市夜景。[^e1]

[^e1]: Time: `2023-07`; Sources: locomo/thread_bb376441501c/msg_7db4a0d171c3
```

本地编号只需在本文件内不重复。代码将其替换为内容哈希，并把来源标识展开为相对路径与锚点。

模型不生成块标识。行尾留空即表示这是新块。

### 8.5 三种情形

**新事实。** 段落此前不存在。模型追加正文行与脚注定义，行尾留空，代码补齐段落标识与脚注编号。

**事实有了新进展。** 原事实仍然成立，后续发展是另一件事。模型追加新段落，原段落一字不动。

```markdown
Calvin 计划在日本待几个月，然后前往波士顿。[^e-9bae588a38] ^7ffb575c
Calvin 后来将行程延长至一年。[^e-1f4c7a2b90] ^3e8d1c47
```

两个段落各有标识，各自可被引用。

**同一陈述被更新。** 段落说的还是同一件事，但内容需要修正为最新值。模型改写正文，追加一条脚注，保留原有脚注。

```markdown
现在是 2026 年。[^e-a1b2c3][^e-d4e5f6][^e-a7b8c9] ^531d2845

[^e-a1b2c3]: Time: `2024-01`; Sources: ...
[^e-d4e5f6]: Time: `2025-01`; Sources: ...
[^e-a7b8c9]: Time: `2026-01`; Sources: ...
```

段落标识 `^531d2845` 始终不变，引用它的派生视图不受影响。变化的是脚注数量：每次更新追加一条，携带该次的时间与来源。

正文只保留当前值。历史值不在正文中，但更新的次数与时间在脚注序列里，读到该段落即可知道这一陈述被修正过几次、各在何时。

### 8.6 段落标识的稳定性

段落标识由代码生成，一个段落一个，全程不变。

| 情形 | 段落标识 |
|---|---|
| 正文被更新 | 不变 |
| 追加脚注 | 不变 |
| 段落被重组移动位置 | 不变 |
| 一个段落拆成两个 | 原标识留给其一，另一个新建 |
| 多个段落合并成一个 | 全部原标识挂到结果段落上 |

后两种情形出现在重组阶段。合并时全部原标识必须保留，任一丢失都会使指向它的派生视图断链。

这条规则使记忆规模不必单调增长：重组可以把同一主题下的多个段落合并成一条，段落数下降，标识数不变。

### 8.7 阶段

| 序号 | 阶段 | 执行方 | 调用 |
|---|---|---|---|
| 一 | 接入 | 代码 | 无 |
| 二 | 模型写入 | 模型 | 1 次多轮 |
| 三 | 标识补齐 | 代码 | 无 |
| 四 | 派生视图与校验 | 代码 | 无 |

每批一次模型调用。编辑次数不影响调用次数。

### 8.8 校验

安装到正式目录前校验两条不变量：

| 不变量 | 违反含义 |
|---|---|
| 脚注中每个来源地址都能定位到原文 | 模型引用了不存在的消息 |
| 此前存在的每个段落标识仍能在某一段落上找到 | 有段落被误删，或标识在改写中丢失 |

任一不成立则整批回滚，记忆回到本批开始前的状态。

第二条允许段落被更新、被移动、被合并，只要标识不消失。它检查的不是内容是否改动，而是引用是否仍然可达。

### 8.9 与原流程的差别

原流程中模型写出纯文本，标识与元数据由第二次模型调用事后反推。该调用每次文件变动触发一次，输入含 topic 全文与全部来源标识。

| 环节 | 原流程 | 本流程 |
|---|---|---|
| 模型编辑方式 | 可整文件重写 | 仅允许局部追加 |
| 模型写出内容 | 纯句子 | 句子加时间加来源 |
| 旧块处理 | 被改写，标识丢失 | 原样保留 |
| 新旧判断 | 模型反推 | 代码检查行尾 |
| 对账调用 | 每次文件变动一次 | 无 |
| 对账输入下限 | 27 668 token | 0 |
| 开销随记忆增长 | 线性增长 | 不增长 |
| 历史版本 | 被覆盖 | 保留 |

原流程的对账输入下限由 topic 全文 15 179 token 与全部 568 条来源标识 12 313 token 构成，随已写入记忆总量线性增长：第 1 批约 12 000，第 25 批约 27 700。
