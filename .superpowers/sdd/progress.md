# v8 双视图索引 + 回原文检索 — Progress Ledger

Plan: docs/superpowers/plans/2026-07-08-dual-index-retrieve-original.md
Spec: docs/superpowers/specs/2026-07-08-dual-index-retrieve-original-design.md
Branch: feature/v7-agent-memory（v8 迭代其上）
Base commit: 3d60723

## Tasks
- Task 1: complete (commits 3d60723..e411849, review clean, 3 passed)
- Task 2: complete (commits e411849..ec0fdb1, 含 fix: _sanitize_topic 防话题静默碰撞; 5 passed, review clean)
- Task 3: complete (commits ec0fdb1..5171cd2, 含 fix: 守卫标量 JSON 防 TypeError; 12 passed 全 v8 套, review clean)
- Task 4: complete (commits 5171cd2..474c0c2, gate-leak 检查 PASS v4/v6/v7 不受影响, monkeypatch 模块限定修正正确; 全套 39 passed, review clean)
- Task 5: complete (commits 474c0c2..beb6326, gate/hide_raw=False/JSONguard/端到端回原文 全过; 全套 40 passed, review clean)
- Task 6: complete (commits beb6326..d61cd82, 含 fix: dia_id 规范化)
  - 冒烟验证 v8 方案成立：双视图 timeline/+topics/ 建对；回原文端到端接通（q2 result.json 含逐字原始对话，证明 dia_id→read_turns 链路通）
  - 冒烟暴露真 bug + 已修（commit d61cd82）：distill_events 弱模型有时吐裸数字 dia_id [3,5] 丢 D<session>: 前缀，read_turns 解析不了→回原文取空。修法：代码侧用本 chunk 合法 dia_ids 列表强制规范化为 D<session>:<turn>，丢弃无效。test_v8_distill 8 passed，全套 44 passed。重跑冒烟验证检索改善进行中。

## v8-opt: dia_id 代码托管（新计划，base 5340d0c）
Plan: docs/superpowers/plans/2026-07-08-v8-code-managed-diaid.md
Spec: docs/superpowers/specs/2026-07-08-v8-code-managed-diaid-design.md
起因：冒烟发现弱模型提炼漏标 dia_id(6/9空) + 检索抄不对 id。改为代码托管：提炼带行号模型写refs、检索read_original工具必填id。
- Task opt-1: complete (commits 5340d0c..efc2feb, 含 fix: _number_chunk 按turn块编号防多行发言错位; revert-verify 证明; 49 passed, review clean)
- Task opt-2: complete (commits efc2feb..4feb0ef, read_original工具dia_id必填, 消息顺序/JSONguard/端到端 全过; 50 passed, review clean)
- Task opt-3: complete (冒烟验证优化成功)
  - dia_id 空比例：优化前 6/9 空(67%) → 优化后 0/10 空(全部规范 D<session>:<turn>)
  - 检索非空题数：优化前 1/3 → 优化后 3/3 全命中
  - 后半段 session(D2/D3)事件也标全 dia_id，不再越往后越懒
  - 结论：v8 双视图+回原文+dia_id代码托管，在弱模型 qwen3.6-flash 上跑通。弱模型两短板(提炼漏标/检索抄错)由"代码托管、模型只做确认"解决

## 最终全分支 review（opus，3d60723..4feb0ef）：NEEDS-FIXES → 已修
gate 干净、nativemem.py 0 行 diff、端到端链路正确，除一个 Critical：
- C1 [Critical] 已修（commit 96c3385）：_number_chunk 用 re.split(\n\n+) 反推 turn 边界，与 split 的 \n\n 拼接冲突，turn 内部空行(实测12/5882)仍错位→静默回错原文。根治：split_into_chunks_structured 返回结构化 per-turn 列表，按 turn 编号 block n↔dia_ids[n-1] 从构造保证。revert-verify 证明测试网住 bug。split_into_chunks 字节未变（AST SHA 同），v4/v6/v7 不受影响。51 passed。
- I1 [Important] 已修：删死码 _DIA_ID_RE
- I2 [Important] 已修：删死码 _normalize_event_dia_ids + 改正两个说谎的测试名
- M1-M5 [Minor] 安全可延后：v7 max_rounds 默认改(死默认无运行影响)、_append_sorted O(n²)(小规模OK)、build_turn_index 重复id覆盖(真实数据0重复)、同日append序、多tool_call未测。

## Minor findings (for final review)
- Task 1: v8_memory.py:2-3 未用 import os/re（brief 原样带的死代码）
- Task 1: build_turn_index dia_id 碰撞后写覆盖，重复 id 的早 turn 不可达（真实数据若有重复 id 才有风险）
- Task 1: 测试没覆盖"两 dia_id 重叠 context 不重复列"这条约束（reviewer 手验正确，缺回归测试）
- Task 2: _append_sorted 每次全文件重读重写 O(n²) + 非原子写（崩溃中途可能截断）；话题文件长了才有 scale 债
- Task 2: 同日多事件排序只按 append 顺序（无二级 dia_id 排序，spec 未要求）
- Task 3: fence-strip 正则只去行尾闭合```，闭合后有尾随文本时靠 fallback \{.*\} 兜底（自愈，非 bug）
- Task 4: run_nativemem.py:45-46 named import distill_events 被 v8_memory.distill_events 取代成死引用（无害，可删或注释）
- Task 5: _DIA_ID_RE=r"D\d+:\d+" 无词边界，理论上会误匹配 XD1:3Y 这类内嵌串（记忆格式机器生成，低风险）
- Task 5: test_v8_retrieve fake tool-call 用 key "cmd" 但 execute_tool 读 "command"，round1 tool 空转（测试仍有效，round2 直接给 dia_id；纯测试保真 nit）
- Task 6: 检索遵从度——冒烟中 q0 对应的 D1:3 格式正确、记忆里也有，但检索 agent 6 步内没在最终回答里列出它（_V8_RETRIEVE_PROMPT 遵从度问题，非 dia_id bug）。最终 review/后续 prompt 迭代定夺。

## 设计要点（防漂移）
- 事件行格式固定：[YYYY-MM-DD] 摘要 · [dia_id]（多源 [D1:3, D1:5]）
- 时间视图 timeline/{YYYY}/{MM}-{Mon}.md（Mon 英文三字母），文件内按日升序
- 话题视图 topics/{topic}.md
- with-original：hide_raw 关，NATIVEMEM_NO_ORIGINAL 不启用
- 只动 v8 分支，v4/v6/v7 不变
- 落盘由代码保证，绝不让模型手写记忆内容

## 后续优化（分支保留期间）
- 英文摘要强制（commit fc778f0）：诊断发现摘要57%中文漂移(弱模型跟中文prompt)，LoCoMo是英文数据集→英文问题grep不到中文摘要压低命中。prompt强制summary/topic英文后，冒烟中文占比57%→0%，检索命中稳定3/3，dia_id规范性保持。确定性改动，见效直接。
# v8-self-organized-upgrade 进度账
- Task 1.1: complete (commit 20c4207, 71/71 tests, 计划矛盾取舍:dir一律折叠/INLINE只管files模式)
- Task 1.2: eval 后台跑中 (bj1p0azea, results/v8map-eval-s0.log)
- Task 2.1: complete (commit 03b1866, dedup_topic_files)
- Task 2.2: complete (commit cb2ffd5, consolidate_topic_files+候选+merge prompt, 额外:_sanitize_topic对齐落盘名)
- Task 2.3/3.1/4: 等 eval 完成才能动 run_nativemem.py(避免污染评测中途加载的代码)
- Task 1.2: complete — 块一评测 方案B overall 93.4 (基线88.8, +4.6; open-domain 76.9→92.3 +15.4 正中导航病根)。dir 模式保留默认。files 对照留到最后消融批
- 决策: 2.3+3.1+4 一个 agent 顺序做,合并一次终态评测;掉分再用 env 开关消融定位
- Task 2.3: complete (commit 2ce959b, 整理接主循环)
- Task 3.1: complete (commit 920fc38, known_topics带计数; 计划外:每session从磁盘重算known_view替代显式reload,更简洁)
- Task 4: complete (commit 8c2244a, NATIVEMEM_V8_MAX_ROUNDS/MAX_TOKENS/READ_CONTEXT)
- 全量 84 passed。终态评测启动中(vs 块一93.4/基线88.8 + topic文件数收敛验收)
- 终态评测: 方案B overall 91.4 (open-domain 100, temporal 97.3, multi-hop 84.4波动); topic 72→28 收敛达成
- 3样本确认跑启动 (v8selforg-b3, 只跑方案B, 对比升级前 ab3-B 89.9)
- 【重要修正】open-domain 100% 是 judge 送分假象:弃答(Not mentioned)被判对。严格口径(弃答=错): 基线86.2 vs 升级后85.5,+2.6蒸发。topic收敛72→28仍成立。升级后弃答率4→9上升,待3样本确认是否系统性
- 教训: eval_v8_ab.sh 会 rm -rf 固定目录,块一轮 records 已被覆盖;以后跑前备份
- 3样本报告一律双口径: 标准judge分 + 严格分(弃答=错) + 弃答率
- 【重大发现】弃答送分污染所有方案B数字: ab3 B 89.9→严格76.1(53题弃答!) vs A 83.9→82.3, A/B结论反转; chunk扫描严格分 c6=73.0 c10=82.2(最优) c15=63.2 c20=77.0, "c20最高"是送分假象,大chunk丢覆盖率的原始直觉正确
- 根因: _V8_SINGLE_PROMPT 允许"Not mentioned", judge送分; A的answerer用Mem0标准prompt被逼猜测,不受污染
- 修法: B prompt 反弃答对齐标准口径(合规:行为对齐,不碰gold)。修完重跑A/B+chunk关键对比
- v8selforg-b3 继续跑(供升级前后弃答率对比), 出分双口径
- 3样本裁决(385题): 升级前 89.9/弃60/严格76.1 → 升级后 90.9/弃34/严格83.6。严格+7.5,多样本站住,升级真实有效
- prompt修复已合并(919aa22, cat-topic-first+全枚举+反弃答; 旧prompt自己教模型答Not mentioned)
- 冒烟3/3: q123 guinea pig✓ q66 marshmallows✓ q144 弃答→猜测
- 新prompt 3样本跑中(v8promptfix-b3), 出分后完成三点消融链: 76.1 → 83.6 → ?
- 【消融链第三点】prompt修复3样本: 标准91.7 弃答8 严格90.1。链: 76.1→83.6(+7.5结构)→90.1(+6.5检索策略)。标准/严格收敛,弃答清零
- 文章化实现(worktree, 同agent): 内联引用+绝对日期+session末文章化重写(dia_id集合相等硬校验)+滚动上下文+双视图互链(日期→timeline链接, timeline→topic反链), NATIVEMEM_V8_ARTICLE开关
- 文章化+互链合并(merge 35cea61, 累计11个feat commit, 136测试)。第四点评测跑中(v8article-b3)
- 注意: 文章化build会多花LLM调用(session末重写+硬校验补抽), 出分时记录build成本对比
- sample9 理想记忆精建完成(2beef24, opus手工, 未读qa): 30 session/568 turns, timeline 149行, topics 8篇(Calvin/Dave), 引用抽检20/20, 含图片caption
- 排队: v8article-b3 跑完 → 立即跑 sample9 oracle 检索(qwen方案B, --memory-dir gold_memory/sample9), 串行避免API争抢
- 【oracle对照】s9: oracle 88.0/80.4/13.9k-tok vs auto 87.3/81.6/23.6k-tok。结论: build质量非分数瓶颈(打平); 理想组织省41%检索成本; temporal差15.6(自动build日期处理可挖); 剩余~15分在qwen检索/答题
- 【紧急】文章化build成本O(n²): auto-s9 花3.24M输出tok/3.8h(s0时代302k)。session末全文重写滚雪球。修法: 按增长触发/增量重写
- O(n²)修复(51cc33c/7a66a28/517d3ed, 148测试): 生行≥8触发+增量融入+收尾扫描; 1591是events标签bug非重复处理
- 文章化3样本重跑(v8article-b3, 修好的管线) — 消融链第四点
- 【消融链四点定案】双口径(标准/严格/弃答): 1升级前 79.7/76.1/60 → 2结构升级 90.9/87.5/14 → 3检索prompt修 91.7/91.7/0 → 4文章化 88.1/88.1/0(净-3.6砍掉)。第3点严格分项 multi-hop89.2/temporal96.7/open90.5/single90.5。口径统一: 旧文档第3点严格90.1 一律重算为91.7(弃答清零后标准=严格)
- 【文章化验尸】21道翻错题(s0+s1)三桶: A写丢=0 / B写糊=7(二次换算日期算错+原词被同义替换) / C在库但检索选错文件=14(拆碎话题事实散兄弟文件命中面缩小)。结论: 信息零丢失, 败在重写失真+命中面变小; 成本还是原子行3倍(8h vs 2.5h)。文章化路线砍
- 【oracle对照定案】sample9: opus手建 88.0/80.4/13.9k-tok ≈ auto 87.3/81.6/23.6k-tok。build质量已饱和(overall打平), 剩余分在qwen检索/答题侧; oracle只赢成本(-41%)和temporal(+15.6)
- 第5点启动(v8sections-b3, 回归原子行叠三件): 分节整理(438f5fd, 模型只出分节JSON+代码搬行+逐字校验+未整理节) / 三层重复治理(28fb6b6, timeline指纹去重→topic内同义行合并→话题文件语义合并) / 检索交叉核对(64d6ef5, 答题前回timeline按人物+日期核对+兄弟话题文件ls列全都读)。判据: ≥91.7则定稿跑10样本正式(builds并行, samples3-9干净测试集), 否则继续修
