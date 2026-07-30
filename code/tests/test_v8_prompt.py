"""检索 prompt 的策略指导必须锁定，防未来被误删（轨迹诊断的 A 词面鸿沟 / B 选行失败 + 反弃答）。"""
from src.adapters.run_nativemem import _V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT


def test_placeholders_intact():
    for p in (_V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT):
        assert "{structure}" in p
        assert "{question}" in p


def test_llm_controlled_tool_strategy():
    # LLM 自适应选择结构、grep 与 BM25，不锁成固定前置流水线。
    for p in (_V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT):
        assert "cat" in p and "读全文" in p
        assert "grep" in p and "bm25_search" in p
        assert "没有固定顺序" in p
        assert "无 embedding" in p
        # A 词面鸿沟：词法工具要允许换多种措辞重试。
        assert "多种" in p and "措辞" in p


def test_line_selection_discipline():
    # B 选行失败：所有沾边的行都要枚举、相关 dia_id 全部 read_original，不许只挑一条
    for p in (_V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT):
        assert "全部" in p
        assert "只挑一条" in p


def test_distill_bans_speech_act_topic_names():
    # topic 第二级必须是内容，不许用言语行为（inquiry/feedback/sharing…）当名
    from src.v8_memory import _V8_DISTILL_PROMPT as P
    for w in ("inquiry", "feedback", "sharing", "conversation",
              "compliments", "well_wishes", "question", "response"):
        assert w in P
    # 判断标准 + 归到被评论那件事下的规则
    assert "关于什么" in P
    assert "怎么说出来" in P


def test_single_model_requires_grounded_abstention():
    # SINGLE 负责答题，证据不足或错误前提时必须允许弃答。
    assert "证据充分" in _V8_SINGLE_PROMPT
    assert "Not mentioned in the conversation" in _V8_SINGLE_PROMPT
    assert "最佳猜测" not in _V8_SINGLE_PROMPT
    # RETRIEVE 只收集证据，不加入答题口径。
    assert "Not mentioned in the conversation" not in _V8_RETRIEVE_PROMPT


def test_timeline_cross_check_discipline():
    # 检索选错文件的根因：锁定候选后必须回 timeline 用人物+日期交叉核对，散文只当导航
    for p in (_V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT):
        assert "交叉核对" in p
        assert "timeline" in p
        assert "导航" in p


def test_multi_file_per_person_discipline():
    # 事实散在兄弟文件：同一人物多个文件先 ls 列全，相关的全部 cat，不许读一个就答
    for p in (_V8_SINGLE_PROMPT, _V8_RETRIEVE_PROMPT):
        assert "ls" in p
        assert "全部 cat" in p
        assert "读一个就答" in p
