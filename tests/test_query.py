"""Query Rewrite 测试：门控 / 清洗校验 / 客户端参数 / 编排降级 / 两条对话路径的接线。

不依赖真实 MySQL / Qdrant / 网关：
- 纯函数（`needs_rewrite` / `sanitize_rewrite`）直接断言；
- 改写客户端用替身（捕获 prompt，可配置异常 / 空返回 / 超长返回 / 原样返回）；
- 对话路径用假 LLM + monkeypatch 检索，断言两条路径都做到
  「检索用**改写后** query，发给模型的仍是**原始** user_input」。

覆盖用户要求的五类：单元测试、多轮对话、指代、无上下文、Rewrite 失败。
"""

import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import Settings, get_settings
from app.modules.ai.chat import service as service_module
from app.modules.ai.chat import stream as stream_module
from app.modules.ai.chat.runtime import ChatStreamContext, LlmRuntime
from app.modules.ai.chat.service import AiChatService
from app.modules.ai.query import (
    LLM,
    NOOP,
    LlmQueryRewriter,
    NoopQueryRewriter,
    RewriteSource,
    apply_query_rewrite,
    build_rewrite_llm,
    needs_rewrite,
    sanitize_rewrite,
)

ANAPHORA = "那这个怎么办？"
REWRITTEN = "企业版退款申请流程是什么？"
# 有证据的命中（rerank 分高于默认阈值 0.35）：让对话链路能走到模型。
# 空命中自 Phase 3 起会触发 ABSTAIN（不调用模型），故给桩一条命中。
EVIDENCE_HIT = {
    "content": "企业版退款期限为 60 天。",
    "document_name": "退款政策.md",
    "rerank_score": 0.9,
}
REFUND_HISTORY = [("user", "企业版退款期限是多少？"), ("assistant", "企业版是 60 天。")]
ATTENDANCE_HISTORY = [
    ("user", "考勤制度里迟到怎么算？"),
    ("assistant", "当月累计迟到超过 30 分钟按旷工处理。"),
]


# ---- 夹具 ----


@pytest.fixture(autouse=True)
def restore_settings():
    """用例级还原配置：`get_settings()` 是 lru_cache 单例，改了会污染其他测试。"""
    settings = get_settings()
    keys = [key for key in vars(settings) if key.startswith("rag_query_rewrite_")]
    saved = {key: getattr(settings, key) for key in keys}
    yield
    for key, value in saved.items():
        setattr(settings, key, value)


def _set(**overrides) -> None:
    """覆盖全局 rag_query_rewrite_* 配置（patch 的是 lru_cache 的同一实例）。"""
    settings = get_settings()
    for key, value in overrides.items():
        setattr(settings, key, value)


def _enable(**overrides) -> None:
    """把配置切到「改写启用且参数完整」的状态（可用 overrides 覆盖单项）。"""
    values = {
        "rag_query_rewrite_enabled": True,
        "rag_query_rewrite_history_rounds": 2,
        "rag_query_rewrite_timeout": 4.0,
        "rag_query_rewrite_max_tokens": 1024,
        "rag_query_rewrite_max_chars": 200,
    }
    values.update(overrides)
    _set(**values)


class FakeRewriteLlm:
    """改写客户端替身：记录调用次数与收到的 prompt，可配置返回值 / 异常。"""

    def __init__(self, content: str = REWRITTEN, exc: Exception | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.content = content
        self.exc = exc

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(messages[0].content)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.content)


class FakeChatLlm:
    """对话模型替身：捕获收到的消息，用于断言「改写结果没有进模型上下文」。"""

    def __init__(self, reply: str = "好的") -> None:
        self.reply = reply
        self.seen: list[str] = []

    async def astream(self, messages, **kwargs):
        self.seen = [m.content for m in messages]
        yield SimpleNamespace(content=self.reply, usage_metadata=None, additional_kwargs={})

    async def ainvoke(self, messages, **kwargs):
        self.seen = [m.content for m in messages]
        return SimpleNamespace(content=self.reply, usage_metadata=None, additional_kwargs={})


# ---- 一、门控：needs_rewrite（只判断"是否值得尝试"）----


def test_needs_rewrite_without_history_is_false() -> None:
    """无上下文：没有可补全的依据，且首轮不该多花一次 LLM 调用。"""
    assert needs_rewrite(ANAPHORA, []) is False


def test_needs_rewrite_empty_query_is_false() -> None:
    assert needs_rewrite("", REFUND_HISTORY) is False
    assert needs_rewrite("   ", REFUND_HISTORY) is False


@pytest.mark.parametrize(
    "query",
    [
        "那这个怎么办？",  # 这个
        "那个流程呢？",  # 那个
        "它支持哪些支付方式？",  # 它
        "这些条件怎么算？",  # 这些
        "上述规则适用于个人版吗？",  # 上述
        "刚才说的期限是多久？",  # 刚才
        "然后呢？",  # 省略式追问
        "那企业版呢？",  # 那…呢
        "退款咋办？",  # 咋办
    ],
)
def test_needs_rewrite_detects_anaphora_and_followup(query: str) -> None:
    """指代 / 省略式追问：必须结合历史才可能检索到正确片段。"""
    assert needs_rewrite(query, REFUND_HISTORY) is True


@pytest.mark.parametrize(
    "query",
    [
        "企业版退款期限是多少？",
        "请说明企业版与个人版在退款期限上的差别",
        "ERR_4012 是什么错误？",
        "如何配置 Agent 的知识库？",
    ],
)
def test_needs_rewrite_skips_self_contained_questions(query: str) -> None:
    """简单、明确的问题不强制 Rewrite（即便有历史）。"""
    assert needs_rewrite(query, REFUND_HISTORY) is False


# ---- 二、清洗与校验：sanitize_rewrite ----


def test_sanitize_keeps_clean_output() -> None:
    assert sanitize_rewrite(REWRITTEN, ANAPHORA) == REWRITTEN


def test_sanitize_strips_code_fence() -> None:
    assert sanitize_rewrite(f"```\n{REWRITTEN}\n```", ANAPHORA) == REWRITTEN
    assert sanitize_rewrite(f"```text\n{REWRITTEN}\n```", ANAPHORA) == REWRITTEN


@pytest.mark.parametrize(
    "raw",
    [f"改写后：{REWRITTEN}", f"改写后的问题:{REWRITTEN}", f"Rewritten: {REWRITTEN}"],
)
def test_sanitize_strips_explanatory_prefix(raw: str) -> None:
    assert sanitize_rewrite(raw, ANAPHORA) == REWRITTEN


def test_sanitize_strips_surrounding_quotes() -> None:
    assert sanitize_rewrite(f'"{REWRITTEN}"', ANAPHORA) == REWRITTEN
    assert sanitize_rewrite(f"「{REWRITTEN}」", ANAPHORA) == REWRITTEN


def test_sanitize_skips_meta_line_and_takes_next() -> None:
    """模型先写一句解释再给结果 → 取有效的那一行。"""
    assert sanitize_rewrite(f"说明：结合历史补全\n{REWRITTEN}", ANAPHORA) == REWRITTEN


def test_sanitize_takes_first_line_when_explanation_follows() -> None:
    assert sanitize_rewrite(f"{REWRITTEN}\n解释：因为上文提到退款", ANAPHORA) == REWRITTEN


def test_sanitize_collapses_whitespace() -> None:
    assert sanitize_rewrite("  企业版   退款流程 是什么？ ", ANAPHORA) == "企业版 退款流程 是什么？"


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", "```\n```"])
def test_sanitize_rejects_empty_output(raw: str) -> None:
    assert sanitize_rewrite(raw, ANAPHORA) is None


def test_sanitize_rejects_non_string() -> None:
    assert sanitize_rewrite(None, ANAPHORA) is None  # type: ignore[arg-type]


def test_sanitize_rejects_over_max_chars() -> None:
    _set(rag_query_rewrite_max_chars=20)
    long_text = "企业版退款申请流程是什么" + "补充说明很多很多" * 10 + "？"
    assert sanitize_rewrite(long_text, ANAPHORA) is None


def test_sanitize_rejects_answer_like_statement() -> None:
    """把问题答成陈述句 → 既是答案又严重偏离原问题，必须回退原问题。"""
    assert sanitize_rewrite("企业版是 60 天。", ANAPHORA) is None


def test_sanitize_allows_statement_when_original_is_not_question() -> None:
    """原问题本身不是疑问句时，不强制改写结果是疑问句。"""
    assert sanitize_rewrite("企业版退款", "退款流程") == "企业版退款"


def test_sanitize_rejects_statement_rewrite_of_question() -> None:
    assert sanitize_rewrite("企业版退款", "怎么退款") is None


# ---- 三、LlmQueryRewriter.process ----


async def test_process_returns_model_source() -> None:
    _enable()
    fake = FakeRewriteLlm()
    result = await LlmQueryRewriter(fake).process(ANAPHORA, REFUND_HISTORY)

    assert result.original_query == ANAPHORA
    assert result.rewritten_query == REWRITTEN
    assert result.rewritten is True
    assert result.source == RewriteSource.MODEL
    assert result.elapsed_ms is not None
    assert fake.calls == 1


async def test_process_marks_not_needed_when_model_returns_original() -> None:
    """模型判定问题已自足而原样返回 → 不是失败，只是没有改写收益。"""
    _enable()
    fake = FakeRewriteLlm(content=" 那这个怎么办？ ")
    result = await LlmQueryRewriter(fake).process(ANAPHORA, REFUND_HISTORY)

    assert result.rewritten is False
    assert result.source == RewriteSource.NOT_NEEDED
    assert result.rewritten_query == ANAPHORA


@pytest.mark.parametrize("content", ["", "企业版是 60 天。"])
async def test_process_marks_failure_on_invalid_output(content: str) -> None:
    """空输出 / 答案式输出都属于非法结果 → FAILURE（调用方回退原问题）。"""
    _enable()
    result = await LlmQueryRewriter(FakeRewriteLlm(content=content)).process(
        ANAPHORA, REFUND_HISTORY
    )

    assert result.rewritten is False
    assert result.source == RewriteSource.FAILURE
    assert result.rewritten_query == ANAPHORA


async def test_process_prompt_contains_history_and_query() -> None:
    _enable()
    fake = FakeRewriteLlm()
    await LlmQueryRewriter(fake).process(ANAPHORA, REFUND_HISTORY)

    prompt = fake.prompts[0]
    assert "用户: 企业版退款期限是多少？" in prompt
    assert "助手: 企业版是 60 天。" in prompt
    assert f"最新问题：{ANAPHORA}" in prompt
    assert prompt.endswith("改写后的问题：")


async def test_process_uses_only_recent_rounds() -> None:
    """history_rounds 口径：用户 + 助手 = 1 个完整 round。"""
    _enable(rag_query_rewrite_history_rounds=1)
    fake = FakeRewriteLlm()
    history = [
        ("user", "旧问题甲"),
        ("assistant", "旧回答甲"),
        ("user", "企业版退款期限是多少？"),
        ("assistant", "企业版是 60 天。"),
    ]
    await LlmQueryRewriter(fake).process(ANAPHORA, history)

    prompt = fake.prompts[0]
    assert "企业版是 60 天。" in prompt  # 最近 1 轮
    assert "旧回答甲" not in prompt  # 更早的轮次被裁掉


async def test_process_includes_all_rounds_within_limit() -> None:
    _enable(rag_query_rewrite_history_rounds=2)
    fake = FakeRewriteLlm()
    history = [
        ("user", "旧问题甲"),
        ("assistant", "旧回答甲"),
        ("user", "企业版退款期限是多少？"),
        ("assistant", "企业版是 60 天。"),
    ]
    await LlmQueryRewriter(fake).process(ANAPHORA, history)
    assert "旧回答甲" in fake.prompts[0]


# ---- 四、编排：apply_query_rewrite 的短路与降级 ----


async def test_disabled_returns_identity_without_calling_llm() -> None:
    """总开关关闭 → 零外部调用（Query Rewrite 是增强能力，不是依赖）。"""
    _set(rag_query_rewrite_enabled=False)
    fake = FakeRewriteLlm()
    result = await apply_query_rewrite(fake, ANAPHORA, REFUND_HISTORY)

    assert result.source == RewriteSource.DISABLED
    assert result.rewritten is False
    assert result.rewritten_query == ANAPHORA
    assert result.elapsed_ms is None
    assert fake.calls == 0


async def test_no_context_returns_identity_without_calling_llm() -> None:
    """无上下文 → 直接不改写，零调用。"""
    _enable()
    fake = FakeRewriteLlm()

    result = await apply_query_rewrite(fake, ANAPHORA, [])
    assert result.source == RewriteSource.NO_CONTEXT

    result = await apply_query_rewrite(fake, "   ", REFUND_HISTORY)
    assert result.source == RewriteSource.NO_CONTEXT

    assert fake.calls == 0


async def test_self_contained_question_skips_llm_call() -> None:
    """有历史但问题自足 → 门控拦下，不调用模型。"""
    _enable()
    fake = FakeRewriteLlm()
    result = await apply_query_rewrite(fake, "企业版退款期限是多少？", REFUND_HISTORY)

    assert result.source == RewriteSource.NOT_NEEDED
    assert result.rewritten is False
    assert fake.calls == 0


async def test_missing_llm_falls_back_to_original() -> None:
    """配置缺失（无可用模型）→ FAILURE 且回退，不影响 RAG。"""
    _enable()
    result = await apply_query_rewrite(None, ANAPHORA, REFUND_HISTORY)

    assert result.source == RewriteSource.FAILURE
    assert result.rewritten_query == ANAPHORA
    assert result.rewritten is False


@pytest.mark.parametrize("exc", [TimeoutError("timeout"), RuntimeError("网关 503")])
async def test_llm_exception_falls_back_to_original(exc: Exception) -> None:
    """超时 / 任意异常 → 回退原问题（且不抛异常）。"""
    _enable()
    fake = FakeRewriteLlm(exc=exc)
    result = await apply_query_rewrite(fake, ANAPHORA, REFUND_HISTORY)

    assert fake.calls == 1
    assert result.source == RewriteSource.FAILURE
    assert result.rewritten_query == ANAPHORA
    assert result.elapsed_ms is not None


async def test_success_records_model_source() -> None:
    _enable()
    result = await apply_query_rewrite(FakeRewriteLlm(), ANAPHORA, REFUND_HISTORY)

    assert result.source == RewriteSource.MODEL
    assert result.rewritten is True
    assert result.rewritten_query == REWRITTEN


async def test_noop_rewriter_is_identity() -> None:
    result = await NoopQueryRewriter().process(ANAPHORA, REFUND_HISTORY)

    assert NoopQueryRewriter().name == NOOP
    assert LlmQueryRewriter(FakeRewriteLlm()).name == LLM
    assert result.rewritten_query == result.original_query == ANAPHORA
    assert result.source == RewriteSource.DISABLED


# ---- 五、改写客户端：独立预算（不得影响主 Chat 客户端）----


def test_build_rewrite_llm_uses_isolated_budget() -> None:
    _enable(rag_query_rewrite_max_tokens=1024, rag_query_rewrite_timeout=4.0)
    llm = build_rewrite_llm("deepseek-v4-flash", api_key="sk-x", base_url="https://x/v1")

    assert llm.model_name == "deepseek-v4-flash"
    assert llm.temperature == 0.0  # 同一问题改写结果稳定可比对
    assert llm.max_tokens == 1024
    assert llm.max_retries == 0  # 改写串行阻塞首 token：宁可立刻降级也不重试
    assert llm.request_timeout == 4.0


def test_build_rewrite_llm_default_api_key_placeholder() -> None:
    """供应商未配 api_key 时不得让客户端构造失败（与 vector / rerankers 同一处理方式）。"""
    assert build_rewrite_llm("m").openai_api_key.get_secret_value() == "not-set"


async def test_build_llm_returns_isolated_rewrite_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_build_llm()` 必须给出**独立**改写客户端，且不污染主 Chat 客户端参数。"""
    _enable(rag_query_rewrite_max_tokens=1024, rag_query_rewrite_timeout=4.0)
    service = AiChatService(db=None)  # type: ignore[arg-type]

    provider_id = uuid4()
    instance = SimpleNamespace(
        id=uuid4(),
        code="deepseek-v4-flash",
        status=1,
        input_price=None,
        output_price=None,
        provider_id=provider_id,
    )
    provider = SimpleNamespace(
        id=provider_id,
        code="opencode",
        base_url="https://opencode.ai/zen/go/v1",
        api_key="sk-x",
    )

    async def fake_instance(_instance_id):
        return instance

    async def fake_provider(_provider_id):
        return provider

    monkeypatch.setattr(service.instance_repo, "get_by_id", fake_instance)
    monkeypatch.setattr(service.provider_repo, "get_by_id", fake_provider)

    agent = SimpleNamespace(model_id=instance.id, name="A", organization_id=None)
    config = {"temperature": 0.7, "max_tokens": 2048}
    session_id = uuid4()
    runtime = await service._build_llm(agent, config, session_id)

    # 主客户端：沿用 Agent 配置与 ai_max_retries，未被改写参数影响
    assert runtime.llm.temperature == 0.7
    assert runtime.llm.max_tokens == 2048
    assert runtime.llm.max_retries == get_settings().ai_max_retries

    # 改写客户端：同一套供应商参数，独立预算，并复用 opencode 会话头
    rewrite_llm = runtime.rewrite_llm
    assert rewrite_llm is not None
    assert rewrite_llm.model_name == instance.code
    assert str(rewrite_llm.openai_api_base) == provider.base_url
    assert rewrite_llm.temperature == 0.0
    assert rewrite_llm.max_tokens == 1024
    assert rewrite_llm.max_retries == 0
    assert rewrite_llm.default_headers == {"x-opencode-session": session_id.hex[:8]}


# ---- 六、多轮对话 + 指代：两条对话路径的接线 ----


async def _run_stream(
    monkeypatch: pytest.MonkeyPatch,
    rewrite_llm,
    *,
    query: str = ANAPHORA,
    history: list[tuple[str, str]] | None = None,
    config: dict | None = None,
):
    """跑一次流式对话，返回 (事件列表, 检索用的 query 列表, 模型收到的消息)。"""
    retrieved: list[str] = []

    async def fake_retrieve(_config, q):
        retrieved.append(q)
        # 必须返回「有证据」的命中：空命中会触发 ABSTAIN（后端拒答、不调用模型），
        # 而本组用例断言的是「模型收到了什么」。
        return [EVIDENCE_HIT]

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(stream_module, "_retrieve_hits_short_session", fake_retrieve)
    monkeypatch.setattr(stream_module, "_save_reply", noop)
    monkeypatch.setattr(stream_module, "_record_usage", noop)

    chat_llm = FakeChatLlm()
    ctx = ChatStreamContext(
        agent_id=uuid4(),
        conv_id=None,
        model_code="fake",
        llm=chat_llm,
        config={"knowledge_ids": [str(uuid4())], "tools": []} if config is None else config,
        history=list(REFUND_HISTORY if history is None else history),
        user_input=query,
        started_at=time.perf_counter(),
        rewrite_llm=rewrite_llm,
    )
    events = [event async for event in stream_module.stream_chat(ctx)]
    return events, retrieved, chat_llm.seen


async def test_stream_chat_rewrites_query_for_retrieval_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """多轮 + 指代：检索用改写后 query，发给模型的仍是原始问题。"""
    _enable()
    fake = FakeRewriteLlm()
    events, retrieved, seen = await _run_stream(monkeypatch, fake)

    assert fake.calls == 1
    assert retrieved == [REWRITTEN]
    assert seen[-1] == ANAPHORA
    assert all(REWRITTEN not in content for content in seen)

    kinds = [event.type for event in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("tool") == 2  # 预检索的 running + done


async def test_stream_chat_retrieval_step_cost_includes_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable()
    events, _retrieved, _seen = await _run_stream(monkeypatch, FakeRewriteLlm())
    steps = [s for s in events[-1].data.steps if s.kind == "retrieval"]

    assert len(steps) == 1
    assert steps[0].cost_ms is not None  # 含改写 + 检索耗时


async def test_stream_chat_without_history_skips_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无上下文：不调用模型，检索用原问题。"""
    _enable()
    fake = FakeRewriteLlm()
    _events, retrieved, seen = await _run_stream(monkeypatch, fake, history=[])

    assert fake.calls == 0
    assert retrieved == [ANAPHORA]
    assert seen[-1] == ANAPHORA


async def test_stream_chat_rewrite_failure_retrieves_with_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewrite 失败：检索回退原问题，对话链路照常完成。"""
    _enable()
    fake = FakeRewriteLlm(exc=TimeoutError("timeout"))
    events, retrieved, seen = await _run_stream(monkeypatch, fake)

    assert fake.calls == 1
    assert retrieved == [ANAPHORA]
    assert [event.type for event in events][-1] == "done"
    assert seen[-1] == ANAPHORA


async def test_stream_chat_disabled_uses_original_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_query_rewrite_enabled=False)
    fake = FakeRewriteLlm()
    _events, retrieved, _seen = await _run_stream(monkeypatch, fake)

    assert fake.calls == 0
    assert retrieved == [ANAPHORA]


async def test_chat_sync_rewrites_query_for_retrieval_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同步路径口径与流式一致：检索用改写后 query，模型消息仍是原始问题。"""
    _enable()
    service = AiChatService(db=None)  # type: ignore[arg-type]
    rewrite = FakeRewriteLlm()
    chat_llm = FakeChatLlm()
    retrieved: list[str] = []

    async def fake_retrieve(_db, _config, q):
        retrieved.append(q)
        # 同上：空命中会被判定为证据不足并直接拒答，模型不会被调用
        return [EVIDENCE_HIT]

    async def fake_usage(**kwargs):
        return None

    async def fake_load(_agent_id):
        return SimpleNamespace(name="A", organization_id=None), {
            "knowledge_ids": [str(uuid4())],
            "tools": [],
        }

    async def fake_resolve(_agent_id, _user_input, _user_id, _session_id):
        return None, list(REFUND_HISTORY)

    async def fake_build(_agent, _config, _session_id=None):
        return LlmRuntime(llm=chat_llm, model_code="fake", rewrite_llm=rewrite)

    monkeypatch.setattr(service_module, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(service_module, "persist_usage", fake_usage)
    monkeypatch.setattr(service, "_load_agent_config", fake_load)
    monkeypatch.setattr(service, "_resolve_conversation", fake_resolve)
    monkeypatch.setattr(service, "_build_llm", fake_build)

    out = await service.chat(uuid4(), ANAPHORA)

    assert rewrite.calls == 1
    assert retrieved == [REWRITTEN]
    assert chat_llm.seen[-1] == ANAPHORA
    assert all(REWRITTEN not in content for content in chat_llm.seen)
    assert out.reply == "好的"


# ---- 七、默认配置（保证改造前后行为一致）----


def test_config_defaults_are_safe() -> None:
    """默认关闭：不配置 .env 时行为与改造前完全一致。"""
    defaults = Settings(_env_file=None)

    assert defaults.rag_query_rewrite_enabled is False
    assert defaults.rag_query_rewrite_history_rounds == 2
    assert defaults.rag_query_rewrite_timeout > 0
    assert defaults.rag_query_rewrite_max_tokens > 0
    assert defaults.rag_query_rewrite_max_chars > 0
