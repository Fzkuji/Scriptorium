# LongMemEval smoke 交接说明（2026-08-07）

## 新对话先做什么

请先阅读：

1. `README.md`
2. `docs/method/scriptorium-method.html`
3. `docs/research/research-plan.html`
4. `docs/research/longmemeval-smoke-plan.html`
5. 本文件

当前目标不是正式评分，而是先完成 LongMemEval-S 的一个完整 multi-session 样本构建 smoke。样本固定为数据集索引 75、question id `3a704032`，共 48 sessions、479 source turns。先跑 S8（每批 8 sessions，共 6 批），暂不调用 judge；成功后再决定 S4/S8 对照。

## 已确认的实验策略

- 构建模型、检索与回答：Packy API 的 `deepseek-v4-flash`。
- 正式效果 smoke 的 judge：GPT-4o-mini，但当前机械 smoke 不运行 judge。
- Memory 始终完整构建并保存 Source / Topic / Timeline / Core / Recent 等组件；后续消融优先在读取/检索阶段 mask 某组件，不为每个消融重复构建。
- 每批必须有 checkpoint 和实时审计；中断后从已提交批次恢复，不能默认整项重跑。
- 不得显示、复制或提交 API key。密钥文件为仓库根目录的 `provider-api-key.txt`，已被 gitignore。
- LoCoMo 评分受根目录 `AGENTS.md` 锁定；本任务是 LongMemEval，不要改动锁定 evaluator。

## 当前架构与 Windows 兼容改动

当前工作树有尚未提交的实现和文档改动，请在其上继续，不要还原或覆盖：

- 新增跨平台 shell backend：`code/src/shell_backend.py`。
- `MemoryConfig.shell_backend` 支持 `auto` / `posix-bash` / `native`。
- Windows 正式配置使用 Git Bash 的 `posix-bash`，提示与工具 schema 统一为 POSIX Bash；Linux `auto` 保持原行为。
- Writer、检索的输出使用 UTF-8 容错解码。
- 增加实时 Writer audit、`.progress.json`、失败快照和更丰富的 build checkpoint。
- Windows stage 改为与 memory 同盘，并修复 stage root、ACL、原子目录替换的瞬时权限重试。
- 成功工具调用会记录 `windows_retries`。

不要动用户自己的 `.obsidian/workspace.json` 和 `.obsidian/graph.json`。

## 已通过的验证

- Writer micro smoke r3：1 session、6 turns、8 rounds、7 次成功工具调用、5 blocks、3 topics、Source 引用有效。
- Writer micro smoke r4（ACL 修复后）：1 session、6 turns、10 rounds、7 次成功工具调用、4 blocks、2 topics，递归读取权限正常，Source 引用有效。
- Windows 定向测试此前通过：100 个 management/writer 测试（排除一个既有 fault-injection Windows 测试）、18 个审计相关测试、125 个 core/retrieval/SDK 测试、38 个 runner/checkpoint 测试。

## S8 完整 smoke 历史

### r2（不要恢复）

目录：`code/results/development/longmemeval-ms-item75-s8-t120-win-r2`

因 Windows 原子目录移动的 `WinError 5` 停止，已有 `manual-stop.json`。不要从此目录续跑，也不要覆盖。

### r3（失败，但前三批已可靠提交）

目录：`code/results/development/longmemeval-ms-item75-s8-t120-win-r3`

状态：

- 前 3/6 批已写入 build checkpoint：24/48 sessions、236/479 source turns、168 events。
- Writer rounds：batch 1 = 37，batch 2 = 77，batch 3 = 41。
- 第 4 批 Writer 实际完成：32 rounds、75 assistant messages、31 次工具调用全部成功、37 blocks、9 topics、0 Windows retries。
- 第 4 批随后进入批后 Topic 管理；管理 Agent 已返回成功，但 Python 清理 Claude SDK 临时目录时抛出 `WinError 145: 目录不是空的`。
- 因清理异常被当成 build 异常，第 4 批没有提交到 build checkpoint。`checkpoint.json` 已标记 failed；后台进程已退出。

这不是 API key、模型、工具调用或 `max_turns` 问题，而是 Windows `TemporaryDirectory` 清理竞态。相关位置：`code/src/agent_runtime/claude_code.py` 的 `_run()` 使用 `with tempfile.TemporaryDirectory(...)`。

## WSL 状态（最新结论）

用户开启 TUN 后，2026-08-07 已重新实测：

- WSL DNS 仍解析为 TUN Fake-IP（`198.18.x.x`），但转发已经正常。
- WSL 访问 `https://www.packyapi.ai` 返回 HTTP 200。
- WSL 访问 `https://cf.api.fan` 根路径返回 HTTP 404，表示网络与 TLS 正常，只是根路径不存在。
- 使用仓库密钥从 WSL 向 `https://www.packyapi.ai/v1/messages` 发出最小真实请求，模型 `deepseek-v4-flash`，返回 HTTP 200。密钥未输出。

因此现在优先建议切回 WSL，避开 Windows 临时目录、文件句柄、ACL 和原子移动问题。

## 新对话建议执行顺序

1. 检查 WSL 中项目依赖、Python 环境和 Claude Agent SDK/CLI 是否齐全；不要修改或输出密钥。
2. 用现有 diagnostic 做一次 WSL 单 session Writer smoke，确认：工具语法、Source 引用、Topic/Timeline/Core 保存、实时 audit、checkpoint 均正常。
3. 为 WSL 创建新的、不会覆盖旧结果的 S8 配置和结果目录；保留 `session_batch=8`、`max_turns=120`、48 sessions、build-only/no judge。
4. 后台启动，并每 5 分钟检查实时 audit；若出现程序性错误、重复工具错误或进程退出，立即停止以节省 token。
5. 若 WSL S8 完成，再检查引用可回查、组件独立保存、轮数/耗时/token，并决定是否跑 S4 对照。

不要直接把 r3 的第 4 批视为完成。若要复用 r3，必须先审计 memory transaction 与 checkpoint 的一致性；更稳妥的方案是在 WSL 新目录运行，旧目录全部保留作为失败审计。

## 可能仍要补的 Windows 修复（即使转 WSL也建议留档）

Windows 下 Claude SDK 临时目录清理应：等待子进程释放句柄、对 `WinError 145`/瞬时权限错误重试；若 Agent 已取得有效 ResultMessage，最终清理失败应记录 warning，而不应推翻成功结果。同时，Writer 与批后 Topic/Core 管理最好分别提交子阶段 checkpoint，避免 Writer 已完成却整批重做。
