# NativeMem `src/` 定版设计

## 目标

将当前 V11 设为唯一 NativeMem 实现，取消版本选择层。核心模块直接位于
`code/src/`，不增加 `nativemem/` 外层目录。删除旧实现、旧版本专用测试、旧版本
专用实验脚本和冻结代码副本。现有 benchmark、结果、第三方仓库和 memory 产物不变。

清理前状态由 Git 提交 `31cc34c` 保存。

## 目标目录

```text
code/src/
├── __init__.py
├── build.py
├── conversation.py
├── management/
│   ├── api.py
│   ├── agent.py
│   ├── block_views.py
│   ├── config.py
│   ├── event_writing.py
│   ├── model_reconciliation.py
│   ├── prompts.py
│   ├── provider.py
│   ├── source_archive.py
│   ├── reconciliation.py
│   ├── topic_normalization.py
│   ├── topic_reconciliation.py
│   ├── verification.py
│   └── workspace.py
├── markdown/
│   ├── models.py
│   ├── parser.py
│   ├── syntax.py
│   └── writer.py
├── retrieval/
│   ├── agent.py
│   ├── config.py
│   ├── context.py
│   ├── prompts.py
│   ├── runtime.py
│   ├── schemas.py
│   ├── shell.py
│   ├── tools.py
│   ├── views.py
│   ├── bm25.py
│   └── embedding.py
├── runtime/
│   ├── online.py
│   ├── state.py
│   └── derived_views.py
├── adapters/
├── evaluation/
└── providers/
```

`management/` 负责记忆写入、整理、来源归档、事务与校验；`markdown/` 负责
Topic Markdown 解析和渲染；`retrieval/` 负责 grep、BM25、Embedding 和检索
Agent；`runtime/` 负责在线处理状态和自动派生视图。通用 benchmark adapters、
evaluation 和 provider clients 保持独立。

## 入口

`code/src/__init__.py` 只导出当前正式接口，包括 `BuildConfig`、`build_memory`、
`MemoryConfig`、`MemoryWorkspace`、`QueryConfig` 和 `collect_answer`。调用方不再
使用 `load_version()`、`nativemem_versions.v11` 或任何版本环境变量。

NativeMem 实验脚本统一放入：

```text
code/scripts/nativemem/
├── run_locomo.py
├── run_longmemeval.py
├── reanswer_longmemeval.py
├── common/
├── locomo/
├── longmemeval/
└── ablation/
```

测试统一放入 `code/tests/nativemem/`，按 build、management、markdown、
retrieval 和 runtime 职责拆分。删除只验证 v7、v8、v9、v10 和版本路由的测试。

## 删除范围

- `code/src/nativemem.py`
- `code/src/v8_memory.py`
- `code/src/v10_memory.py`
- `code/src/nativemem_versions/`
- `code/src/legacy/`
- `code/src/adapters/run_nativemem.py`
- 仅服务旧 NativeMem 版本的脚本、审计脚本和测试
- `code/experiments/*/code/` 中的冻结代码副本
- `docs/method/versions/` 中的旧版本说明

已有 `code/results/`、`code/benchmarks/`、`code/third_party/`、`paper/` 和任何
生成的 memory 目录均不删除、不改名。

## 兼容边界

仓库根目录的 `src -> code/src`、`scripts -> code/scripts`、`tests -> code/tests`
等相对链接继续保留。它们只解决仓库路径兼容，不再承担 NativeMem 版本兼容。
历史结果中的代码哈希和旧路径不重写；需要复现旧版本时使用 Git 提交
`31cc34c` 或更早提交。

## 验证

重构完成必须满足：

1. 仓库活跃 Python 代码不再导入 `v8_memory`、`v10_memory`、
   `nativemem_versions` 或 `adapters.run_nativemem`。
2. 当前 build、Topic Markdown、派生视图、在线 runtime、BM25、Embedding、
   retrieval agent 和 LoCoMo/LongMemEval runner 测试通过。
3. `scripts/eval_full.py` 的 SHA-256 保持
   `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`。
4. `python scripts/verify_portable_layout.py`、导入检查和 `git diff --check` 通过。
5. `results/`、benchmark 数据和 memory 产物的文件数与清理前一致。
