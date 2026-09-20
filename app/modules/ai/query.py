"""AI 查询改写（Query Rewrite）：多轮对话中「指代 / 省略」问题的检索前补全。

解决的问题：多轮对话里「那这个怎么办？」这类问题本身不含检索所需的实体，
直接拿去 Hybrid 检索必然召回不到正确片段。本模块在**检索之前**结合已有对话历史
把它补全成自足问题（如「企业版退款申请流程是什么？」），**只用于改善检索**。

完整链路（本模块只占其中一步）：

    Conversation Context
        ↓
    Query Rewrite   ← 本模块（可选 / 可降级 / 失败即回退原问题）
        ↓
    Retrieval → Hybrid Search → Reranker → Context → LLM

边界（务必遵守，否则会改变用户真正意图）：
- 改写结果**只**作为检索 query，绝不替换发给模型的 user 消息，也绝不进入最终答案；
- 改写失败 / 超时 / 输出非法 → 回退 `original_query`，检索行为与未启用时完全一致
  （Query Rewrite 是 RAG 的**增强**能力，不是依赖能力）；
- 总开关关闭时零外部调用（连启发式门控都不跑）。

为什么只有一个入口 `apply_query_rewrite()`、不定义 Protocol / registry：
改写实现只有一个（复用 Agent 已绑定的 chat 模型，见 `build_rewrite_llm`），
按项目约定不为单一实现增加抽象层；将来若接入独立小模型，只需新增一个 `process()` 实现。
"""

import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger

from app.core.config import get_settings
from app.modules.ai.chat.messages import chunk_text
from app.modules.ai.reasoning import ReasoningChatOpenAI

# 改写器名称标记（供日志与测试断言）
NOOP = "noop"
LLM = "llm"


class RewriteSource(StrEnum):
    """`QueryRewriteResult.source` 的固定取值：说明「为什么得到这个结果」。"""

    DISABLED = "DISABLED"  # 总开关关闭
    NO_CONTEXT = "NO_CONTEXT"  # 无历史 / 空问题，没有可补全的上下文
    NOT_NEEDED = "NOT_NEEDED"  # 启发式判定不值得改写，或模型判定原问题已自足
    MODEL = "MODEL"  # 模型成功产出改写
    FAILURE = "FAILURE"  # 配置缺失 / 调用失败 / 超时 / 输出非法（已回退原问题）


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """一次查询改写的结果（冻结值对象，仅供检索链路内部使用，不进 API 出参）。"""

    original_query: str
    # 应该拿去检索的 query：改写成功时为改写结果，否则等于 original_query
    rewritten_query: str
    rewritten: bool
    source: str
    elapsed_ms: int | None = None


# ---- 启发式门控（纯函数）----

# 指代信号：出现这些词说明问题本身缺少主语 / 主题，只能靠历史补全
_ANAPHORA = re.compile(r"这个|那个|这些|那些|这种|那种|这里|那里|它|上述|前面提到|刚才|之前说")
# 省略式追问：不含独立主题的跟进提问（"那……呢" / "然后呢" / "怎么办"）
_FOLLOWUP = re.compile(r"怎么办|咋办|怎么弄|怎么搞|然后呢|还有呢|接着呢|呢[？?]?$|^那")


def needs_rewrite(query: str, history: list[tuple[str, str]]) -> bool:
    """轻量启发式门控：判断「是否值得尝试改写」（纯函数）。

    只回答「要不要花钱调模型」这一个问题，**不做改写、不做正确性判断**：
    - 无历史 / 空问题 → False（没有可补全的上下文，首轮也不该多一次 LLM 调用）；
    - 命中指代词，或是无独立主题的追问 → True；
    - 其余（问题已自足，如「企业版退款期限是多少？」）→ False。

    刻意保持简单：误判为「需要改写」的代价只是多一次带超时的调用，
    而误判为「不需要」只是拿不到改写收益 —— 两侧都不会破坏原有检索行为。
    模型侧还有第二道自纠（prompt 要求「已自足则原样输出」）。
    """
    text = (query or "").strip()
    if not text or not history:
        return False
    return bool(_ANAPHORA.search(text) or _FOLLOWUP.search(text))


# ---- 输出清洗与校验（纯函数）----

# 代码块围栏（模型偶尔把结果包在 ``` 里）
_FENCE = re.compile(r"^\s*```[A-Za-z]*\s*|\s*```\s*$")
# 解释性前缀（"改写后：xxx" / "Rewritten: xxx"）
_PREFIX = re.compile(
    r"^\s*(改写后的问题|改写后|改写结果|改写|输出|结果|Rewritten query|Rewritten)\s*[:：]\s*"
)
# 元话语行（这类行是模型的解释，不是改写结果，跳过继续找下一行）
_META = re.compile(r"^\s*(说明|解释|理由|分析|注意|首先|其次|然后|因此|所以|思考|推导|思路)[:：]")
# 首尾引号 / 书名号（模型爱加）
_QUOTES = "\"'“”‘’「」『』"
# 疑问信号：用于「原问题是疑问句 → 改写也必须是疑问句」的一致性校验
_QUESTION_TAILS = ("？", "?")
_QUESTION_WORDS = ("吗", "呢", "怎么", "什么", "多少", "如何", "哪", "为什么", "是否", "几")


def _looks_like_question(text: str) -> bool:
    """是否是疑问句（以问号结尾，或含疑问词）。"""
    return text.endswith(_QUESTION_TAILS) or any(word in text for word in _QUESTION_WORDS)


def sanitize_rewrite(raw: str, original: str) -> str | None:
    """清洗并校验模型输出（纯函数）。**返回 None 表示校验失败 → 调用方必须回退原问题。**

    清洗：去代码块围栏 → 去「改写后：」这类解释性前缀 → 去首尾引号 → 取首个有效行 → 压缩空白。

    校验（任一不过即 None）：
    - 非空，且长度不超过 `rag_query_rewrite_max_chars`；
    - 原问题是疑问句时，改写结果也必须是疑问句 —— 拦住「把问题答成陈述句」这类
      既是答案、又严重偏离原问题的输出（此时用原问题检索比用一个错误查询更安全）。
    """
    if not isinstance(raw, str):
        return None
    limit = get_settings().rag_query_rewrite_max_chars

    for line in raw.splitlines():
        line = _FENCE.sub("", line).strip()
        if not line or _META.match(line):
            continue
        line = _PREFIX.sub("", line).strip().strip(_QUOTES).strip()
        line = " ".join(line.split())
        if not line or (_looks_like_question(original) and not _looks_like_question(line)):
            continue
        # 超长说明模型跑偏了（改写应该比原问题略长，但不会长到 200 字以上）
        return line if len(line) <= limit else None
    return None


def _normalize(text: str) -> str:
    """归一化（压缩空白 + 忽略大小写），用于判断「改写结果是否等于原问题」。"""
    return " ".join((text or "").split()).casefold()


def _identity(query: str, source: str, elapsed_ms: int | None = None) -> QueryRewriteResult:
    """构造「未改写」结果（原问题原样返回）。"""
    return QueryRewriteResult(
        original_query=query,
        rewritten_query=query,
        rewritten=False,
        source=source,
        elapsed_ms=elapsed_ms,
    )


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


# ---- 提示词 ----

REWRITE_PROMPT = (
    "你是检索查询改写助手。用户的问题会被送去知识库检索，"
    "你的唯一任务是把「最新问题」改写成「脱离对话历史也能独立成立」的检索查询。\n"
    "\n"
    "规则：\n"
    "1. 若最新问题包含指代（这个/那个/它/该/上述/刚才说的）、省略，"
    "或是不含独立主题的追问（那……呢 / 怎么办 / 然后呢），**必须**结合对话历史补全出"
    "明确的主语与主题，改写为完整问句；不要因为「不能 100% 确定」就放弃改写，"
    "应选择历史中**最近、最相关**的主题作为指代对象；\n"
    "2. 补全时只能使用对话历史里已出现的实体、主题与限定条件（产品名、功能名、版本等），"
    "不得臆造新的事实、数字、政策或条件；\n"
    "3. 不得回答问题、不得给出解释或建议，输出必须仍然是**疑问句**；\n"
    "4. 不得改变用户意图：疑问类型（是什么 / 怎么做 / 多少）、范围与条件要与原问题一致；\n"
    "5. 只有当最新问题本身已经能独立理解时，才原样输出；\n"
    "6. 保留原问题的语言。\n"
    "\n"
    "示例：\n"
    "历史：用户：企业版退款期限是多少？ / 助手：企业版是 60 天。\n"
    "最新问题：那这个怎么办？\n"
    "改写：企业版退款申请流程是什么？\n"
    "\n"
    "历史：用户：考勤制度里迟到怎么算？ / 助手：当月累计迟到超过 30 分钟按旷工处理。\n"
    "最新问题：然后呢？\n"
    "改写：考勤制度里迟到超过 30 分钟之后会怎么处理？\n"
    "\n"
    "输出要求：只输出一行改写后的问题，不要引号、不要「改写：」等前后缀、不要任何解释。"
)

_NO_HISTORY = "（无历史，首轮对话）"
_ROLE_LABELS = {"user": "用户", "assistant": "助手"}


def _format_history(history: list[tuple[str, str]]) -> str:
    """把历史渲染成「用户: … / 助手: …」多行文本，只取最近 N 轮。

    轮数口径：1 个完整 round = 用户 1 条 + 助手 1 条，故取最后 `history_rounds * 2` 条。
    """
    rounds = get_settings().rag_query_rewrite_history_rounds
    recent = history[-(rounds * 2) :] if rounds > 0 else history
    lines = [
        f"{_ROLE_LABELS.get(role, role)}: {content}" for role, content in recent if content
    ]
    # 用 f-string 拼接而不用 str.format：提示词里出现裸露 `{}` 不会变成 KeyError（messages.py 踩过）
    return "\n".join(lines) or _NO_HISTORY


def _build_prompt(query: str, history_text: str) -> str:
    return f"{REWRITE_PROMPT}\n\n对话历史：\n{history_text}\n\n最新问题：{query}\n\n改写后的问题："


# ---- 改写器实现 ----


def build_rewrite_llm(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> ReasoningChatOpenAI:
    """构造改写专用的 LLM 客户端：复用主 Chat 模型的供应商参数，但预算与超时独立。

    必须走**构造期**参数，不能复用主客户端再传调用期 kwargs，原因有二：
    - `max_retries=0` 无法在调用期设置，而 SDK 默认重试 2 次 —— 实测把 0.5s 超时
      放大到 3.8s（改写串行阻塞首 token，宁可立刻降级也不要重试）；
    - 调用期覆盖 `max_tokens` 会与主客户端的同名参数形成双键风险。

    `temperature=0.0` 保证同一问题、同一历史的改写结果稳定可比对（便于排查）。
    """
    settings = get_settings()
    return ReasoningChatOpenAI(
        model=model,
        api_key=api_key or "not-set",
        base_url=base_url,
        default_headers=default_headers,
        temperature=0.0,
        max_tokens=settings.rag_query_rewrite_max_tokens,
        max_retries=0,
        timeout=settings.rag_query_rewrite_timeout,
    )


class LlmQueryRewriter:
    """基于 LLM 的查询改写器（只负责「调模型 + 解析输出」，不做开关 / 门控判定）。"""

    name = LLM

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def process(self, query: str, history: list[tuple[str, str]]) -> QueryRewriteResult:
        """产出改写结果；输出非法时返回 `source=FAILURE`（调用方据此回退原问题）。"""
        started = time.perf_counter()
        messages = [HumanMessage(content=_build_prompt(query, _format_history(history)))]
        response = await self._llm.ainvoke(messages)
        elapsed_ms = _elapsed_ms(started)

        raw = chunk_text(response)
        cleaned = sanitize_rewrite(raw, query)
        if cleaned is None:
            logger.warning(
                "Query Rewrite 输出非法，回退原问题 原问题={} 原始输出={!r}", query, raw
            )
            return _identity(query, RewriteSource.FAILURE, elapsed_ms)
        # 模型判定原问题已自足 → 原样返回（不是失败，只是没有改写收益）
        if _normalize(cleaned) == _normalize(query):
            return _identity(query, RewriteSource.NOT_NEEDED, elapsed_ms)
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=cleaned,
            rewritten=True,
            source=RewriteSource.MODEL,
            elapsed_ms=elapsed_ms,
        )


class NoopQueryRewriter:
    """空改写器：恒等返回原问题（关闭 / 无上下文 / 不值得改写 / 失败 四种降级场景）。

    `source` 说明本次降级的原因，便于日志与排查。
    """

    name = NOOP

    def __init__(self, source: str = RewriteSource.DISABLED) -> None:
        self.source = source

    async def process(self, query: str, history: list[tuple[str, str]]) -> QueryRewriteResult:
        return _identity(query, self.source)


# ---- 唯一编排入口 ----


async def apply_query_rewrite(
    llm: Any | None,
    query: str,
    history: list[tuple[str, str]] | None,
) -> QueryRewriteResult:
    """查询改写的**唯一编排入口**：门控 → 调用 → 校验 → 降级（保证不抛异常）。

    返回的 `rewritten_query` 才是应拿去检索的 query；调用方**必须**继续用原始
    `user_input` 组装给模型的消息（改写只改善检索，不得改变用户真正意图）。

    以下情况直接短路，**不产生任何外部调用**：
    `rag_query_rewrite_enabled=False` → DISABLED；无历史 / 空问题 → NO_CONTEXT；
    启发式判定不值得改写 → NOT_NEEDED。
    调用失败 / 超时 / 输出非法 → FAILURE 并回退原问题，绝不影响正常 RAG。
    """
    history = history or []
    settings = get_settings()
    if not settings.rag_query_rewrite_enabled:
        return await NoopQueryRewriter(RewriteSource.DISABLED).process(query, history)
    if not (query or "").strip() or not history:
        return await NoopQueryRewriter(RewriteSource.NO_CONTEXT).process(query, history)
    if not needs_rewrite(query, history):
        logger.debug("Query Rewrite 跳过（启发式判定不值得改写）原问题={}", query)
        return await NoopQueryRewriter(RewriteSource.NOT_NEEDED).process(query, history)
    if llm is None:
        # 未配置可用模型（Agent 未绑定且无默认模型）：视为失败并回退，不影响检索
        logger.warning("Query Rewrite 无可用模型，回退原问题 原问题={}", query)
        return await NoopQueryRewriter(RewriteSource.FAILURE).process(query, history)

    rewriter = LlmQueryRewriter(llm)
    started = time.perf_counter()
    try:
        result = await rewriter.process(query, history)
    except Exception as exc:  # noqa: BLE001 - 改写失败不得影响正常 RAG
        logger.warning(
            "Query Rewrite 调用失败，回退原问题 name={} err={!r} 原问题={}",
            rewriter.name,
            exc,
            query,
        )
        return _identity(query, RewriteSource.FAILURE, _elapsed_ms(started))

    if result.rewritten:
        logger.info(
            "Query Rewrite 完成 name={} source={} elapsed_ms={} 原问题={} 改写后={}",
            rewriter.name,
            result.source,
            result.elapsed_ms,
            query,
            result.rewritten_query,
        )
    else:
        logger.debug(
            "Query Rewrite 未改写 name={} source={} 原问题={}", rewriter.name, result.source, query
        )
    return result
