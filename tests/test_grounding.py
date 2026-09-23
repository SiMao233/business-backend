"""RAG Grounding 测试（Phase 1：证据判定与引用审计的纯函数层）。

覆盖：证据充分性判定 / 拒答策略 / 引用审计 / sources 汇总 / 出参组装 / 配置默认值。
全部断言都落在 chat 层调用之前或之后的**纯函数**上，不依赖 MySQL、Qdrant 与模型网关。
（两条对话路径的接线、以及「拒答不调用 LLM」的编排测试在 Phase 3 追加到本文件。）
"""

import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from app.core.config import Settings, get_settings
from app.modules.ai.chat import service as service_module
from app.modules.ai.chat import stream as stream_module
from app.modules.ai.chat.messages import (
    GROUNDING_RULES,
    NO_EVIDENCE_PROMPT,
    REFERENCE_PROMPT,
    build_messages,
    compose_system_prompt,
)
from app.modules.ai.chat.runtime import KNOWLEDGE_TOOL, ChatStreamContext, LlmRuntime
from app.modules.ai.chat.service import AiChatService
from app.modules.ai.grounding import (
    KNOWLEDGE_RETRIEVAL_TOOL,
    NO_EVIDENCE_TOOL_TEXT,
    EvidenceReason,
    abstain_message,
    assess_evidence,
    audit_answer,
    collect_sources,
    grounding_out,
    has_non_kb_tools,
    is_refusal,
    needs_no_evidence_note,
    should_abstain,
)

# ---- 夹具与构造助手 ----


@pytest.fixture(autouse=True)
def restore_grounding_settings():
    """用例级还原 grounding 相关配置：`get_settings()` 是 lru_cache 单例，改了会污染其他测试。"""
    settings = get_settings()
    keys = [
        key
        for key in vars(settings)
        if key.startswith(("rag_grounding_", "rag_abstain_", "rag_query_rewrite_"))
    ]
    saved = {key: getattr(settings, key) for key in keys}
    yield
    for key, value in saved.items():
        setattr(settings, key, value)


def _set(**overrides) -> None:
    """覆盖全局 grounding 配置（patch 的是 lru_cache 的同一实例）。"""
    settings = get_settings()
    for key, value in overrides.items():
        setattr(settings, key, value)


def _hit(rerank: float | None = None, **extra) -> dict:
    """构造一个命中片段：默认只带向量分（模拟未启用 Reranker 的场景）。"""
    hit: dict = {"chunk_id": uuid4().hex, "content": "片段正文", "score": 0.42}
    if rerank is not None:
        hit["rerank_score"] = rerank
    hit.update(extra)
    return hit


def _config(**overrides) -> dict:
    """构造 Agent 配置快照：默认绑定一个知识库且不挂其它工具。"""
    config: dict = {"knowledge_ids": [str(uuid4())], "tools": []}
    config.update(overrides)
    return config


def _sources(*indices: int) -> list[dict]:
    """构造 sources（形状与 `build_sources()` 一致，只保留审计用到的键）。"""
    return [{"index": i, "document_name": f"doc{i}.md", "content": "..."} for i in indices]


# 提示词测试用：一段带 [1] 编号的「参考资料」（形状与 format_context 的输出一致）
CONTEXT = "[1]（来源：员工手册.txt）迟到或早退超过 30 分钟按旷工处理。"


def _tool_hits(*contents: str) -> list[dict]:
    """构造检索命中（形状与 retrieve_hits 富化后的输出一致，只保留用到的键）。"""
    return [
        {
            "chunk_id": f"chunk-{i}",
            "document_id": None,
            "document_name": "员工手册.txt",
            "knowledge_base_id": None,
            "seq_no": i,
            "content": text,
            "score": 0.5,
        }
        for i, text in enumerate(contents, start=1)
    ]


# ---- 一、证据充分性判定：assess_evidence ----


def test_assess_no_hits_is_insufficient() -> None:
    """空命中 → 证据不足（不看阈值，也没有分数可看）。"""
    assessment = assess_evidence([])

    assert assessment.sufficient is False
    assert assessment.reason == EvidenceReason.NO_HITS
    assert assessment.hit_count == 0
    assert assessment.best_score is None


def test_assess_none_hits_is_insufficient() -> None:
    assert assess_evidence(None).reason == EvidenceReason.NO_HITS


def test_assess_low_rerank_score_is_insufficient() -> None:
    """有命中但相关度过低 → 证据不足，并上报最高分供观测与校准阈值。"""
    _set(rag_abstain_min_score=0.35)
    assessment = assess_evidence([_hit(0.10), _hit(0.20)])

    assert assessment.sufficient is False
    assert assessment.reason == EvidenceReason.LOW_RELEVANCE
    assert assessment.best_score == 0.20
    assert assessment.hit_count == 2


def test_assess_threshold_boundary_is_sufficient() -> None:
    """边界：等于阈值即视为充分（用 `>=`），避免阈值附近时答时拒。"""
    _set(rag_abstain_min_score=0.35)
    assessment = assess_evidence([_hit(0.35), _hit(0.1)])

    assert assessment.sufficient is True
    assert assessment.reason == EvidenceReason.OK
    assert assessment.best_score == 0.35


def test_assess_takes_max_score() -> None:
    _set(rag_abstain_min_score=0.35)
    assessment = assess_evidence([_hit(0.1), _hit(0.86), _hit(0.5)])

    assert assessment.sufficient is True
    assert assessment.best_score == 0.86


def test_assess_without_rerank_score_fails_open() -> None:
    """未启用 Reranker（无 rerank_score）→ fail-open：不拒答，交由提示词约束。"""
    _set(rag_abstain_min_score=0.35)

    vector_only = assess_evidence([_hit(), _hit(score=0.9)])
    assert vector_only.sufficient is True
    assert vector_only.reason == EvidenceReason.NO_SCORE
    assert vector_only.best_score is None
    assert vector_only.hit_count == 2

    # 纯关键词命中（score=None，RRF 融合后没有向量分）
    assert assess_evidence([_hit(score=None)]).reason == EvidenceReason.NO_SCORE


def test_assess_ignores_vector_and_keyword_scores() -> None:
    """向量分再高也不参与判定：只看 rerank_score。"""
    _set(rag_abstain_min_score=0.35)
    assessment = assess_evidence([_hit(0.05, score=0.99)])

    assert assessment.sufficient is False
    assert assessment.reason == EvidenceReason.LOW_RELEVANCE
    assert assessment.best_score == 0.05


def test_assess_ignores_dirty_rerank_scores() -> None:
    """脏数据（字符串 / bool / None / 非 dict）不得把判定带偏。"""
    _set(rag_abstain_min_score=0.35)
    hits = [{"rerank_score": "0.9"}, {"rerank_score": True}, {"rerank_score": None}, "not-a-dict"]

    assert assess_evidence(hits).reason == EvidenceReason.NO_SCORE


# ---- 二、拒答策略：should_abstain / needs_no_evidence_note ----


def test_has_non_kb_tools() -> None:
    assert has_non_kb_tools({"tools": ["knowledge_retrieval"]}) is False
    assert has_non_kb_tools({"tools": ["knowledge_retrieval", "web_search"]}) is True
    assert has_non_kb_tools({"tools": ["web_search"]}) is True
    assert has_non_kb_tools({"tools": []}) is False
    assert has_non_kb_tools({}) is False
    assert has_non_kb_tools(None) is False


def test_should_abstain_on_empty_evidence() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True)
    assert should_abstain([], _config()) is True


def test_should_abstain_on_low_relevance() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True, rag_abstain_min_score=0.35)
    assert should_abstain([_hit(0.1)], _config()) is True


def test_should_not_abstain_when_evidence_sufficient() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True, rag_abstain_min_score=0.35)
    assert should_abstain([_hit(0.8)], _config()) is False


def test_should_not_abstain_without_knowledge_base() -> None:
    """未绑定知识库（普通对话）→ 本功能完全不介入，避免改变既有行为。"""
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True)
    config = {"tools": [], "knowledge_ids": []}

    assert should_abstain([], config) is False
    assert needs_no_evidence_note([], config) is False


def test_should_not_abstain_when_switches_disabled() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_enabled=False)
    assert should_abstain([], _config()) is False

    _set(rag_grounding_enabled=False, rag_abstain_enabled=True)
    assert should_abstain([], _config()) is False


def test_should_not_abstain_when_non_kb_tool_configured() -> None:
    """配了 web_search 等工具：不因知识库无证据直接拒答，改由强约束提示词兜底。"""
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True)
    config = _config(tools=["knowledge_retrieval", "web_search"])

    assert should_abstain([], config) is False
    assert needs_no_evidence_note([], config) is True


def test_needs_no_evidence_note_when_abstain_disabled() -> None:
    """拒答关闭 → 不短路，改为注入无证据说明，由模型自行拒答（概率性）。"""
    _set(rag_grounding_enabled=True, rag_abstain_enabled=False)
    assert needs_no_evidence_note([], _config()) is True


def test_needs_no_evidence_note_false_when_abstaining() -> None:
    """已拒答时不会调用 LLM，也就不需要提示词。"""
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True)
    assert needs_no_evidence_note([], _config()) is False


def test_needs_no_evidence_note_false_when_sufficient_or_disabled() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_enabled=True, rag_abstain_min_score=0.35)
    assert needs_no_evidence_note([_hit(0.8)], _config()) is False

    _set(rag_grounding_enabled=False)
    assert needs_no_evidence_note([], _config()) is False


def test_knowledge_tool_name_matches_runtime() -> None:
    """守卫常量漂移：两者不一致会让「非知识库工具」判断悄悄失效。"""
    assert KNOWLEDGE_RETRIEVAL_TOOL == KNOWLEDGE_TOOL


def test_abstain_message_from_settings() -> None:
    _set(rag_abstain_message="这一点我无法从知识库确认。")
    assert abstain_message() == "这一点我无法从知识库确认。"


# ---- 三、引用审计：audit_answer / collect_sources ----


def test_audit_valid_citations() -> None:
    audit = audit_answer("退款期限是 60 天 [1]。超期后不予受理 [3]。", _sources(1, 2, 3))

    assert audit.cited == [1, 3]
    assert audit.invalid_cited == []
    assert audit.has_citation is True
    assert audit.uncited_claims == 0


def test_audit_detects_fabricated_citation() -> None:
    """模型编造引用：编号本身存在，但没有对应来源。"""
    audit = audit_answer("退款期限是 60 天 [1]，另有隐藏条款 [9]。", _sources(1))

    assert audit.cited == [1, 9]
    assert audit.invalid_cited == [9]


def test_audit_without_sources_marks_all_citations_invalid() -> None:
    audit = audit_answer("答案是 42 [1]。", None)

    assert audit.invalid_cited == [1]
    assert audit.has_citation is True


def test_audit_counts_uncited_claims() -> None:
    """整句没有任何来源标注且长度达标 → 记为一条无依据陈述。"""
    audit = audit_answer("这是一句没有任何来源标注的事实性陈述内容。", _sources(1))

    assert audit.cited == []
    assert audit.uncited_claims == 1
    assert audit.has_citation is False


def test_audit_refusal_sentence_is_not_a_claim() -> None:
    """拒答文案本身不该被记成「无依据的事实扩展」。"""
    _set(rag_abstain_message="根据当前知识库，我无法确认这个问题。")
    audit = audit_answer(abstain_message(), _sources())

    assert audit.uncited_claims == 0
    assert audit.invalid_cited == []


def test_audit_ignores_short_sentences() -> None:
    audit = audit_answer("好的。谢谢。", _sources(1))

    assert audit.uncited_claims == 0


def test_is_refusal_detects_hard_and_soft_refusal() -> None:
    """整段拒答判定（评测框架的「宽松拒答口径」依赖它）：硬拒答文案与常见拒答措辞都算。"""
    _set(rag_abstain_message="根据当前知识库，我无法确认这个问题。")

    assert is_refusal(abstain_message()) is True
    assert is_refusal("现有资料不足以回答这个问题，建议咨询人事。") is True
    assert is_refusal("迟到超过三十分钟按半天事假处理 [1]。") is False
    assert is_refusal("") is False
    assert is_refusal("   ") is False


def test_audit_handles_duplicates_zero_and_long_numbers() -> None:
    audit = audit_answer("重复标注 [1][1]，从 0 开始 [0]，位数超限 [1234]。", _sources(1))

    assert audit.cited == [0, 1]  # [1234] 超过 3 位，不视为引用角标
    assert audit.invalid_cited == [0]  # 来源编号从 1 开始


def test_audit_empty_answer() -> None:
    audit = audit_answer("", _sources(1))

    assert audit.cited == []
    assert audit.invalid_cited == []
    assert audit.uncited_claims == 0
    assert audit.has_citation is False


def test_collect_sources_merges_steps_in_order() -> None:
    steps = [
        {"step_id": "a", "kind": "retrieval", "sources": _sources(1, 2)},
        {"step_id": "b", "kind": "thinking", "sources": None},
        {"step_id": "c", "kind": "tool", "sources": _sources(3)},
    ]

    assert [s["index"] for s in collect_sources(steps)] == [1, 2, 3]


def test_collect_sources_tolerates_empty_and_dirty_input() -> None:
    assert collect_sources(None) == []
    assert collect_sources([]) == []
    assert collect_sources(
        [{"sources": "not-a-list"}, "not-a-dict", {"sources": [None, {"index": 1}]}]
    ) == [{"index": 1}]


# ---- 四、出参组装与默认配置 ----


def test_grounding_out_none_when_disabled() -> None:
    """关闭 grounding → 不产出出参，行为与改造前一致。"""
    _set(rag_grounding_enabled=False)
    assert grounding_out(assess_evidence([]), [], "答案", abstained=True) is None


def test_grounding_out_reports_audit_and_abstain() -> None:
    _set(rag_grounding_enabled=True, rag_abstain_min_score=0.35)
    assessment = assess_evidence([_hit(0.1)])

    out = grounding_out(assessment, _sources(1, 2), "内容 [1]，编造 [7]。", abstained=True)

    assert out is not None
    assert out.reason == EvidenceReason.LOW_RELEVANCE
    assert out.abstained is True
    assert out.hit_count == 1
    assert out.source_count == 2
    assert out.best_score == 0.1
    assert out.cited == [1, 7]
    assert out.invalid_cited == [7]


def test_grounding_out_logs_invalid_citation() -> None:
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        _set(rag_grounding_enabled=True)
        grounding_out(assess_evidence([_hit(0.9)]), _sources(1), "编造引用 [5]。")
    finally:
        logger.remove(sink_id)

    assert any("不存在的来源" in message for message in messages)


def test_grounding_out_no_warning_for_valid_citations() -> None:
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        _set(rag_grounding_enabled=True)
        grounding_out(assess_evidence([_hit(0.9)]), _sources(1), "基于资料 [1]。")
    finally:
        logger.remove(sink_id)

    assert messages == []


def test_grounding_out_serializes_to_camel_case() -> None:
    """出参必须转驼峰（与 steps / sources 同一套约定，前端直接消费）。"""
    _set(rag_grounding_enabled=True, rag_abstain_min_score=0.35)
    out = grounding_out(assess_evidence([_hit(0.9)]), _sources(1), "见资料 [1]。")

    assert out is not None
    assert set(out.model_dump(by_alias=True)) == {
        "reason",
        "abstained",
        "hitCount",
        "sourceCount",
        "bestScore",
        "cited",
        "invalidCited",
        "uncitedClaims",
    }


def test_config_defaults() -> None:
    """默认开启 grounding 与拒答；阈值 0.35 为初始校准值。"""
    defaults = Settings(_env_file=None)

    assert defaults.rag_grounding_enabled is True
    assert defaults.rag_abstain_enabled is True
    assert defaults.rag_abstain_min_score == 0.35
    assert defaults.rag_abstain_message


# ---- 五、提示词层：GROUNDING_RULES / NO_EVIDENCE_PROMPT ----


def test_default_prompt_output_unchanged() -> None:
    """零回归：不传新参数时，两个组装函数的输出与改造前逐字节一致。"""
    reference = REFERENCE_PROMPT.format(context=CONTEXT)

    assert compose_system_prompt("SYS", CONTEXT) == f"SYS\n\n{reference}"
    assert compose_system_prompt("", CONTEXT) == reference
    assert compose_system_prompt("SYS", "") == "SYS"

    messages = build_messages("SYS", CONTEXT, [("user", "旧问"), ("assistant", "旧答")], "新问")
    assert [m.content for m in messages] == ["SYS", reference, "旧问", "旧答", "新问"]


def test_grounding_rules_injected_before_reference_block() -> None:
    text = compose_system_prompt("SYS", CONTEXT, grounding=True)

    assert GROUNDING_RULES in text
    assert REFERENCE_PROMPT.format(context=CONTEXT) in text
    # 规则在前、资料在后：规则决定「能不能用资料外的知识」，需要更醒目
    assert text.index(GROUNDING_RULES) < text.index(CONTEXT)


def test_grounding_rules_without_system_prompt() -> None:
    messages = build_messages("", CONTEXT, [], "问", grounding=True)

    # 只有「规则 + 资料」一条 SystemMessage，不应多出空 system 消息
    assert len(messages) == 2
    assert messages[0].content.startswith(GROUNDING_RULES)
    assert messages[1].content == "问"


def test_no_evidence_prompt_injected_when_not_abstaining() -> None:
    """证据不足但未拒答（拒答关闭 / 有非知识库工具）时，必须显式告知模型「没有资料」。"""
    messages = build_messages("SYS", "", [], "问", grounding=True, no_evidence_note=True)

    assert [m.content for m in messages] == ["SYS", NO_EVIDENCE_PROMPT, "问"]
    assert "根据当前知识库，我无法确认这个问题" in NO_EVIDENCE_PROMPT
    assert "参考资料" in NO_EVIDENCE_PROMPT


def test_no_evidence_note_ignored_without_grounding() -> None:
    """总开关关闭时不得注入任何新文案（权威开关只有 `_reference_block()` 一处）。"""
    assert compose_system_prompt("SYS", "", no_evidence_note=True) == "SYS"
    assert [m.content for m in build_messages("SYS", "", [], "问", no_evidence_note=True)] == [
        "SYS",
        "问",
    ]


def test_grounding_without_context_or_note_injects_nothing() -> None:
    assert compose_system_prompt("SYS", "", grounding=True) == "SYS"
    assert compose_system_prompt("", "", grounding=True) == ""


def test_prompt_constants_are_format_safe() -> None:
    """新增常量不得含花括号，否则拼接时的 .format 会抛 KeyError。"""
    for text in (GROUNDING_RULES, NO_EVIDENCE_PROMPT, NO_EVIDENCE_TOOL_TEXT):
        assert "{" not in text
        assert "}" not in text

    # REFERENCE_PROMPT 仍可正常渲染（含 {context} 占位符）
    assert CONTEXT in REFERENCE_PROMPT.format(context=CONTEXT)


def test_grounding_rules_forbid_unsupported_fact_extension() -> None:
    """硬约束必须覆盖：禁预训练知识扩展、禁编造编号/文档名、无据必拒答。"""
    assert "预训练知识" in GROUNDING_RULES
    assert "禁止" in GROUNDING_RULES
    assert "编造编号" in GROUNDING_RULES
    assert "根据当前知识库，我无法确认这个问题。" in GROUNDING_RULES


# ---- 六、检索工具：空命中的显式处理 ----


def _knowledge_tool(monkeypatch: pytest.MonkeyPatch, hits: list[dict], cited: dict | None = None):
    """构造 knowledge_retrieval 工具（检索被替换为固定命中，不依赖 DB / Qdrant）。"""
    from app.modules.ai import tools as tools_module

    async def fake_retrieve(_db, _config, _query):
        return hits

    # 工具内部是函数级 import，patch 源模块属性即可生效
    monkeypatch.setattr("app.modules.ai.retrieval.retrieve_hits", fake_retrieve)
    return tools_module.build_tools(_config(tools=["knowledge_retrieval"]), cited)[0]


async def _invoke_tool(tool_obj, query: str, call_id: str = "call-1") -> ToolMessage:
    """按 ToolCall 形态调用工具（与 langgraph ToolNode 的实际调用方式一致）。

    刻意不用 `ainvoke({"query": ...})`：那种写法只会拿到纯文本 content、取不到 artifact
    （实测 langchain 仅在带 tool_call_id 的调用里才回填 artifact），
    而 artifact → `steps[].sources` 正是来源可追溯的关键链节。
    """
    call = {
        "name": "knowledge_retrieval",
        "args": {"query": query},
        "id": call_id,
        "type": "tool_call",
    }
    return await tool_obj.ainvoke(call)


async def test_knowledge_tool_empty_hits_returns_explicit_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空命中必须显式告知「没有资料」，而不是返回空串（空串会让模型自行发挥）。"""
    cited = {"n": 0}
    tool_obj = _knowledge_tool(monkeypatch, [], cited)

    message = await _invoke_tool(tool_obj, "知识库里没有的问题")

    assert message.content == NO_EVIDENCE_TOOL_TEXT
    assert message.artifact == []
    assert cited["n"] == 0  # 空命中不占用引用编号


async def test_knowledge_tool_returns_context_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    cited = {"n": 0}
    tool_obj = _knowledge_tool(
        monkeypatch, _tool_hits("迟到超过 30 分钟按旷工处理。"), cited
    )

    message = await _invoke_tool(tool_obj, "迟到怎么算？")

    assert message.content.startswith("[1]（来源：员工手册.txt）迟到超过 30 分钟按旷工处理。")
    assert [source["index"] for source in message.artifact] == [1]
    assert cited["n"] == 1


# ---- 七、流式路径：证据接地接线 ----


class FakeChatLlm:
    """对话模型替身：统计调用次数并记录收到的消息（用于断言「拒答时一次都没调」）。"""

    def __init__(self, reply: str = "好的") -> None:
        self.reply = reply
        self.calls = 0
        self.seen: list[str] = []

    async def astream(self, messages, **kwargs):
        self.calls += 1
        self.seen = [message.content for message in messages]
        yield SimpleNamespace(
            content=self.reply, usage_metadata=None, additional_kwargs={}, tool_call_chunks=None
        )

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.seen = [message.content for message in messages]
        return SimpleNamespace(content=self.reply, usage_metadata=None, additional_kwargs={})


class FakeRewriteLlm:
    """查询改写模型替身：只统计调用次数（用于「拒答不产生额外调用」的边界断言）。"""

    def __init__(self, content: str = "改写后的问题") -> None:
        self.content = content
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.content)


class FakeAgent:
    """`create_agent` 替身：只回放一段正文，不在单测里真跑 langgraph 图。"""

    def __init__(self, llm: FakeChatLlm, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    async def astream(self, inputs, **kwargs):
        self.llm.calls += 1
        yield (
            SimpleNamespace(
                content=self.llm.reply,
                usage_metadata=None,
                additional_kwargs={},
                tool_call_chunks=None,
            ),
            {},
        )

    async def ainvoke(self, inputs, **kwargs):
        self.llm.calls += 1
        return {"messages": [AIMessage(content=self.llm.reply)]}


def _patch_create_agent(monkeypatch: pytest.MonkeyPatch, holder: dict) -> None:
    """把 `langchain.agents.create_agent` 换成替身工厂，并捕获传入的 system_prompt。"""
    created: list[FakeAgent] = []
    holder["created"] = created

    def fake_create_agent(model, tools, system_prompt=None, **kwargs):
        agent = FakeAgent(model, system_prompt or "")
        created.append(agent)
        return agent

    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)


async def _run_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hits: list[dict],
    config: dict | None = None,
    reply: str = "好的",
    query: str = "问题",
    history: list[tuple[str, str]] | None = None,
    rewrite_llm=None,
):
    """跑一次流式对话，返回 (事件列表, 假对话模型, create_agent 捕获容器)。"""

    async def fake_retrieve(_config, _query):
        return hits

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(stream_module, "_retrieve_hits_short_session", fake_retrieve)
    monkeypatch.setattr(stream_module, "_save_reply", noop)
    monkeypatch.setattr(stream_module, "_record_usage", noop)
    holder: dict = {}
    _patch_create_agent(monkeypatch, holder)

    llm = FakeChatLlm(reply)
    ctx = ChatStreamContext(
        agent_id=uuid4(),
        conv_id=None,
        model_code="fake",
        llm=llm,
        config=config if config is not None else _config(),
        history=list(history or []),
        user_input=query,
        started_at=time.perf_counter(),
        rewrite_llm=rewrite_llm,
    )
    events = [event async for event in stream_module.stream_chat(ctx)]
    return events, llm, holder


def _steps_of_kind(done, kind: str = "retrieval") -> list:
    return [step for step in done.steps if step.kind == kind]


async def test_stream_with_evidence_calls_model_and_grounds_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, llm, _holder = await _run_stream(monkeypatch, hits=[_hit(0.9)])

    assert llm.calls == 1
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.content == "好的"
    assert done.grounding is not None
    assert done.grounding.reason == EvidenceReason.OK
    assert done.grounding.abstained is False
    assert done.grounding.best_score == 0.9
    assert done.grounding.source_count == 1
    # 提示词含严格规则与真实编号
    assert GROUNDING_RULES in llm.seen[0]
    assert "[1]" in llm.seen[0]
    # sources 的 index 与正文 [n] 的取值范围一致
    assert [source.index for source in _steps_of_kind(done)[0].sources] == [1]


async def test_stream_abstains_without_calling_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """空命中 → ABSTAIN：后端固定文案，**模型一次都没调**。"""
    events, llm, _holder = await _run_stream(monkeypatch, hits=[])

    assert llm.calls == 0
    answer = abstain_message()
    done = events[-1].data
    assert done.finish_reason == "abstain"
    assert done.content == answer
    assert done.reasoning == ""
    assert done.grounding.reason == EvidenceReason.NO_HITS
    assert done.grounding.abstained is True
    assert done.grounding.hit_count == 0
    # 仍推 delta：前端「按增量拼接正文」的既有契约不变
    assert [event.data.content for event in events if event.type == "delta"] == [answer]
    # 检索步骤保留，且带「已拒答」结论与真实耗时（可观测）
    retrieval = _steps_of_kind(done)
    assert len(retrieval) == 1
    assert "已拒答" in retrieval[0].output
    assert retrieval[0].cost_ms is not None
    assert [event.type for event in events][-1] == "done"


async def test_stream_abstains_on_low_relevance_but_keeps_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_abstain_min_score=0.35)
    events, llm, _holder = await _run_stream(monkeypatch, hits=[_hit(0.1)])

    assert llm.calls == 0
    done = events[-1].data
    assert done.finish_reason == "abstain"
    assert done.grounding.reason == EvidenceReason.LOW_RELEVANCE
    assert done.grounding.best_score == 0.1
    # 低分命中仍作为来源上报：这是「为什么拒答」的审计线索
    assert [source.index for source in _steps_of_kind(done)[0].sources] == [1]


async def test_stream_threshold_boundary_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """边界：等于阈值即作答（与 assess_evidence 的 `>=` 一致）。"""
    _set(rag_abstain_min_score=0.35)
    _events, llm, _holder = await _run_stream(monkeypatch, hits=[_hit(0.35)])

    assert llm.calls == 1


async def test_stream_without_rerank_score_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """未启用 Reranker（无 rerank_score）→ 不拒答，仅注入严格规则。"""
    events, llm, _holder = await _run_stream(monkeypatch, hits=[_hit()])

    assert llm.calls == 1
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.grounding.reason == EvidenceReason.NO_SCORE
    assert done.grounding.abstained is False


async def test_stream_abstain_disabled_injects_no_evidence_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拒答开关关闭 → 仍交给模型，但注入「本轮无资料」强约束（概率性兑底）。"""
    _set(rag_abstain_enabled=False)
    events, llm, _holder = await _run_stream(monkeypatch, hits=[])

    assert llm.calls == 1
    assert NO_EVIDENCE_PROMPT in llm.seen[0]
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.grounding.reason == EvidenceReason.NO_HITS
    assert done.grounding.abstained is False


async def test_stream_grounding_disabled_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """总开关关闭 → 不拒答、不注入新文案、不产出 grounding 出参（行为同改造前）。"""
    _set(rag_grounding_enabled=False)
    events, llm, _holder = await _run_stream(monkeypatch, hits=[])

    assert llm.calls == 1
    assert llm.seen == ["问题"]  # 无系统提示词注入
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.grounding is None


async def test_stream_grounding_disabled_has_no_rules_with_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_grounding_enabled=False)
    _events, llm, _holder = await _run_stream(monkeypatch, hits=[_hit(0.9)])

    assert GROUNDING_RULES not in llm.seen[0]
    assert "[1]（来源：未知文档）" in llm.seen[0]  # 仍走旧的 REFERENCE_PROMPT


async def test_stream_without_knowledge_ids_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """未绑定知识库 → 本功能完全不介入（普通对话行为不变）。"""
    config = {"knowledge_ids": [], "tools": [], "system_prompt": "你是助手"}
    events, llm, _holder = await _run_stream(monkeypatch, hits=[], config=config)

    assert llm.calls == 1
    assert llm.seen[0] == "你是助手"
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.grounding.reason == EvidenceReason.NO_SCORE
    assert done.grounding.abstained is False


async def test_stream_tool_mode_grounds_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tools=["knowledge_retrieval"])
    events, llm, holder = await _run_stream(monkeypatch, hits=[_hit(0.9)], config=config)

    assert llm.calls == 1
    system_prompt = holder["created"][0].system_prompt
    assert GROUNDING_RULES in system_prompt
    assert "[1]" in system_prompt
    assert events[-1].data.finish_reason == "stop"


async def test_stream_non_kb_tool_prevents_abstain(monkeypatch: pytest.MonkeyPatch) -> None:
    """配了 web_search：不因知识库无证据直接拒答，改由强约束提示词兜底。"""
    config = _config(tools=["knowledge_retrieval", "web_search"])
    events, llm, holder = await _run_stream(monkeypatch, hits=[], config=config)

    assert llm.calls == 1
    system_prompt = holder["created"][0].system_prompt
    assert NO_EVIDENCE_PROMPT in system_prompt
    assert GROUNDING_RULES not in system_prompt  # 无资料 → 只注入「无证据」说明
    done = events[-1].data
    assert done.finish_reason == "stop"
    assert done.grounding.abstained is False


async def test_stream_citation_audit_flags_invented_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型编造 [9]（无对应来源）→ 只审计上报，不改写正文。"""
    events, _llm, _holder = await _run_stream(
        monkeypatch, hits=[_hit(0.9)], reply="资料说明 [1]，另有隐藏条款 [9]。"
    )

    done = events[-1].data
    assert done.content == "资料说明 [1]，另有隐藏条款 [9]。"  # 正文原样保留
    assert done.grounding.cited == [1, 9]
    assert done.grounding.invalid_cited == [9]


async def test_stream_citation_audit_counts_uncited_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无引用的事实性长句 → 计入 uncited_claims（启发式观测指标）。"""
    events, _llm, _holder = await _run_stream(
        monkeypatch, hits=[_hit(0.9)], reply="这是一句没有任何来源标注的事实性陈述内容。"
    )

    assert events[-1].data.grounding.uncited_claims == 1


async def test_stream_abstain_answer_is_not_flagged_as_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _llm, _holder = await _run_stream(monkeypatch, hits=[])

    assert events[-1].data.grounding.uncited_claims == 0
    assert events[-1].data.grounding.cited == []


async def test_stream_abstain_does_not_add_extra_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拒答的「零调用」边界：不产生**额外**调用。

    ⚠️ 查询改写（若开启）属于**检索环节**、发生在拿到命中之前，因此仍然会发生 ——
    拒答能保证的是「判定之后不再调用任何模型」，而不是「本次请求一个模型都没调」。
    """
    _set(rag_query_rewrite_enabled=True, rag_query_rewrite_max_tokens=1024)
    rewrite_llm = FakeRewriteLlm()
    events, llm, _holder = await _run_stream(
        monkeypatch,
        hits=[],
        query="那这个怎么办？",
        history=[("user", "退款期限是多少？"), ("assistant", "60 天。")],
        rewrite_llm=rewrite_llm,
    )

    assert rewrite_llm.calls == 1  # 检索环节的改写照常发生
    assert llm.calls == 0  # 拒答：对话模型零调用
    assert events[-1].data.finish_reason == "abstain"


async def test_stream_done_grounding_is_camel_case(monkeypatch: pytest.MonkeyPatch) -> None:
    events, _llm, _holder = await _run_stream(monkeypatch, hits=[_hit(0.9)])

    payload = events[-1].data.model_dump(by_alias=True)
    assert payload["finishReason"] == "stop"
    assert payload["grounding"]["sourceCount"] == 1
    assert payload["grounding"]["abstained"] is False
    assert "invalidCited" in payload["grounding"]
    assert "uncitedClaims" in payload["grounding"]


# ---- 八、同步路径：与流式同口径 ----


async def _run_sync(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hits: list[dict],
    config: dict | None = None,
    reply: str = "好的",
):
    """跑一次同步对话，返回 (AiChatOut, 假对话模型, create_agent 捕获容器)。"""
    service = AiChatService(db=None)  # type: ignore[arg-type]
    llm = FakeChatLlm(reply)
    holder: dict = {}
    _patch_create_agent(monkeypatch, holder)
    resolved_config = config if config is not None else _config()

    async def fake_retrieve(_db, _config, _query):
        return hits

    async def fake_usage(**kwargs):
        return None

    async def fake_load(_agent_id):
        return SimpleNamespace(name="A", organization_id=None), resolved_config

    async def fake_resolve(_agent_id, _user_input, _user_id, _session_id):
        return None, []

    async def fake_build(_agent, _config, _session_id=None):
        return LlmRuntime(llm=llm, model_code="fake")

    monkeypatch.setattr(service_module, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(service_module, "persist_usage", fake_usage)
    monkeypatch.setattr(service, "_load_agent_config", fake_load)
    monkeypatch.setattr(service, "_resolve_conversation", fake_resolve)
    monkeypatch.setattr(service, "_build_llm", fake_build)

    out = await service.chat(uuid4(), "问题")
    return out, llm, holder


async def test_sync_abstains_without_calling_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    out, llm, _holder = await _run_sync(monkeypatch, hits=[])

    assert llm.calls == 0
    assert out.reply == abstain_message()
    assert out.finish_reason == "abstain"
    assert out.grounding.reason == EvidenceReason.NO_HITS
    assert out.grounding.abstained is True
    assert "已拒答" in _steps_of_kind(out)[0].output


async def test_sync_grounds_answer_with_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    out, llm, _holder = await _run_sync(
        monkeypatch, hits=[_hit(0.9)], reply="退款期限是 60 天 [1]。"
    )

    assert llm.calls == 1
    assert GROUNDING_RULES in llm.seen[0]
    assert out.finish_reason == "stop"
    assert out.grounding.reason == EvidenceReason.OK
    assert out.grounding.cited == [1]
    assert out.grounding.invalid_cited == []


async def test_sync_without_evidence_and_abstain_disabled_injects_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_abstain_enabled=False)
    out, llm, _holder = await _run_sync(monkeypatch, hits=[])

    assert llm.calls == 1
    assert NO_EVIDENCE_PROMPT in llm.seen[0]
    assert out.finish_reason == "stop"
    assert out.grounding.abstained is False


async def test_sync_tool_mode_grounds_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tools=["knowledge_retrieval"])
    _out, llm, holder = await _run_sync(monkeypatch, hits=[_hit(0.9)], config=config)

    assert llm.calls == 1
    assert GROUNDING_RULES in holder["created"][0].system_prompt


async def test_sync_and_stream_agree_on_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """两条路径必须给出同一结论（同一命中 → 同 finishReason / 同 grounding.reason / 同来源编号）。"""
    hits = [_hit(0.2)]

    out, _llm, _holder = await _run_sync(monkeypatch, hits=hits)
    events, _llm2, _holder2 = await _run_stream(monkeypatch, hits=hits)
    done = events[-1].data

    assert out.finish_reason == done.finish_reason == "abstain"
    assert out.grounding.reason == done.grounding.reason == EvidenceReason.LOW_RELEVANCE
    assert out.grounding.best_score == done.grounding.best_score == 0.2
    sync_sources = [
        source.index for step in _steps_of_kind(out) for source in (step.sources or [])
    ]
    stream_sources = [
        source.index for step in _steps_of_kind(done) for source in (step.sources or [])
    ]
    assert sync_sources == stream_sources == [1]


async def test_sync_out_grounding_is_camel_case(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _llm, _holder = await _run_sync(monkeypatch, hits=[])

    payload = out.model_dump(by_alias=True)
    assert payload["finishReason"] == "abstain"
    assert payload["grounding"]["abstained"] is True
    assert payload["grounding"]["reason"] == EvidenceReason.NO_HITS.value
    assert "invalidCited" in payload["grounding"]
