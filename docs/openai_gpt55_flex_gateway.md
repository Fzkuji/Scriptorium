# OpenAI GPT-5.5 Flex 正式运行网关

该网关只用于新的 OpenAI API Flex 正式运行。它与现有 ChatGPT/Codex 订阅代理采用不同的认证、计费和审计记录，二者的结果目录不得复用。

固定合同如下：

- 本地接口：`POST /v1/chat/completions`
- 本地允许的 `model`：`gpt-5.5`
- provider endpoint：`https://api.openai.com/v1/chat/completions`
- provider snapshot：`gpt-5.5-2026-04-23`
- provider `service_tier`：`flex`
- 返回给现有 runner 的 `model`：`gpt-5.5`
- 流式返回：不允许。网关需要完整 usage 才能在响应前持久化费用。

provider URL、snapshot 和 service tier 均不是 CLI 参数。请求中省略 `service_tier` 或明确写 `flex` 均可；`standard`、`default`、`auto` 和其他值会在本地拒绝。provider 返回的 `model` 或 `service_tier` 不符合固定合同时，网关返回 502，但仍依据 provider usage 记录已经发生的费用。

## 新结果根

为一次正式运行创建新的父目录，并把网关证据和实验结果放在不同子目录：

```bash
RUN_ROOT="results/openai-gpt55-flex-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

export OPENAI_API_KEY="..."
python3 -m src.openai_gpt55_flex_gateway \
  --result-root "$RUN_ROOT/gateway" \
  --max-cost-usd "<CAP>" \
  --host 127.0.0.1 \
  --port 8200
```

`gateway` 子目录必须为空或已经包含同一网关生成的 marker。非空但没有 Flex marker 的目录会被拒绝。不得把现有订阅实验目录、`CHATGPT_PROXY_LOG` 或订阅代理的输出复制到该子目录。

启动后，现有支持 `--base-url` 和 `--api-key` 的 exact GPT-5.5 runner 使用：

```bash
python3 scripts/<exact-runner>.py \
  --base-url http://127.0.0.1:8200/v1 \
  --api-key x \
  --output "$RUN_ROOT/experiment"
```

具体 runner 的其他冻结参数仍由该 runner 的 preregistration 和审计合同决定。不要使用原订阅代理地址启动同一正式运行。

## 请求兼容与转换

网关保留 Chat Completions 的 `messages`、`temperature`、`tools`、`tool_choice` 以及 assistant message 内的 `tool_calls`。它只进行以下必要转换：

- `model: gpt-5.5` 改为 provider snapshot `gpt-5.5-2026-04-23`。
- `max_tokens` 改为 `max_completion_tokens`。两个字段同时存在但数值不同会被拒绝。
- 两个 token 上限字段都省略时，加入 CLI 的 `--default-max-completion-tokens`，默认 4096。
- 强制写入 `service_tier: flex` 和 `stream: false`。

provider 完整响应通过验证后，网关把顶层 `model` 改回 `gpt-5.5`，因此现有 exact runner 仍能执行其模型一致性检查。`flex_gateway_meta` 同时给出 provider snapshot、alias、request SHA 和物理重试次数；兼容字段 `proxy_meta` 继续提供 `attempts`、`http_request_id` 和空的 `unsupported_parameters`。

## 费用和硬上限

费用使用整数 nanodollar 计算，基础 Flex 单价为：

- uncached input：$2.50 / 1M tokens
- cached input：$0.25 / 1M tokens
- output：$15.00 / 1M tokens

当 prompt usage 大于 272,000 tokens 时，完整请求采用 2 倍 input 和 1.5 倍 output 单价。reasoning tokens 是 completion tokens 的子集，不重复计费。

每个请求发送前，网关依据 GPT-5.5 的 1,050,000 context window、128,000 max output、请求的 `max_completion_tokens` 和全部 input 均未缓存的假设，计算一个保守费用上界。持久化状态满足：

```text
committed_cost + all_in_flight_reservations <= max_cost
```

并发线程和并发进程通过文件锁更新同一状态。无法预留时，本地返回 402，且不会调用 provider。请求完成后，用 provider usage 替换预留；明确的 `429 Resource Unavailable` 和确定未计费的拒绝会释放预留。transport 中断或 HTTP 200 响应缺少有效 usage 时，网关无法证明实际费用，因此保留完整预留并阻止正式审计通过。`--max-cost-usd` 写入状态后不能在同一结果根中静默修改。

进程在 provider 请求期间异常终止时，预留会保留。此状态是故意的 fail-closed 行为；不得直接删除 reservation 或重发对应逻辑请求。需要先结合 runner 的 durable ledger 和 provider 账单判断该请求是否已经计费。

## 429 与其他错误

只有 `429 Resource Unavailable` 进行指数退避重试。每次物理请求使用完全相同的 snapshot 与 `service_tier: flex` payload。达到 `--max-physical-attempts` 后返回 503 并释放预留。网关不会把 tier 改为 `auto`、`default` 或 standard。

其他 provider 4xx/5xx、transport error、usage 缺失、usage 不自洽、模型错配和 service-tier 错配均直接 fail closed。

## 本地证据

`gateway` 子目录包含：

- `openai_gpt55_flex_root.json`：provider、snapshot、tier 和 billing marker。
- `flex_cost_state.json`：累计费用、累计 usage、在途预留和固定预算。
- `flex_requests.jsonl`：每个逻辑请求的 alias、provider actual snapshot、returned alias、service tier、response ID、local/provider request SHA、规范化后的 `max_completion_tokens`、保守预留金额、cached/reasoning usage、实际费用和每次物理尝试。独立审计器会从 token 上限重算预留金额，并核对 durable state 的失败释放计数。
- `gateway_ready.json`：进程存活期间的本地 base URL；正常退出时删除。

JSONL 不保存 API key、Authorization header、messages 或完整 provider payload。`GET /healthz` 只返回固定模型合同和费用汇总，不返回 key、环境变量名、文件路径或请求内容。

停止网关后执行独立离线审计：

```bash
python3 scripts/audit_openai_gpt55_flex_gateway.py \
  --result-root "$RUN_ROOT/gateway" \
  --max-cost-usd "<CAP>" \
  --output "$RUN_ROOT/gateway_audit.json"
```

默认审计要求没有在途 reservation，并重新计算每条 billable JSONL 的费用、usage 累计、重试合法性和 durable state 总额。审计器不访问网络。

## 定价依据

- [GPT-5.5 model page](https://developers.openai.com/api/docs/models/gpt-5.5)：snapshot、1,050,000 context window、128,000 max output，以及大于 272K prompt 时完整请求采用 2 倍 input 和 1.5 倍 output。
- [Flex processing guide](https://developers.openai.com/api/docs/guides/flex-processing)：`service_tier: flex`、Batch/Flex 费率、429 Resource Unavailable 和指数退避行为。
