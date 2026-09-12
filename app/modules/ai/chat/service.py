"""AI 对话 Service 层：对话业务规则与流程编排。

核心职责：加载 Agent 配置 → 解析绑定的模型实例/供应商 → 用 LangChain 构造 LLM →
按 config.tools 走智能体（工具调用）或纯 RAG 检索 → 组装上下文 → 调用模型返回回复。

知识库检索策略：**只要 config.knowledge_ids 非空就先检索**（两种模式一致）。
工具模式下检索本可由模型自主决定，但模型经常不调工具就直接回答，导致「配了知识库却没检索」，
故这里统一预检索并把「参考资料」注入 system prompt；knowledge_retrieval 工具仍保留，供模型追问细节。

两条执行路径：
- 同步 `AiChatService.chat()`：一次拿完整回复（Agent 测试运行 / ARQ 后台任务复用）；
- 流式 `stream_chat(ctx)`：模块级异步生成器，产出 SSE 事件（meta / reasoning / tool / delta / error / done）。

设计约束（未来可拆独立 ai-service 的边界）：
- 所有 LLM 调用收敛在本模块，RAG 检索收敛在 retrieval.py，向量库封装在 vector.py；
- 会话与消息的存取统一委托 `ConversationService`（conversation 子包），本模块只管「怎么问模型」；
- 流式生成器**不得使用请求级数据库会话**（HTTP 响应开始后请求会话已归还连接），
  需要检索/落库时用 `AsyncSessionLocal` 开短会话。
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.ai.chat.schema import (
    AiChatOut,
    AiStreamDeltaOut,
    AiStreamDoneOut,
    AiStreamErrorOut,
    AiStreamMetaOut,
    AiStreamReasoningOut,
    AiStreamToolOut,
    AiStreamUsageOut,
)
from app.modules.ai.conversation.service import ConversationService
from app.modules.ai.reasoning import ReasoningChatOpenAI
from app.modules.ai.retrieval import format_context, retrieve_context, retrieve_hits
from app.modules.ai.tools import build_tools

# 纯 RAG 模式注入「参考资料」的提示词模板
REFERENCE_PROMPT = "以下是参考资料，请优先基于这些资料回答用户问题，不要编造资料外的内容：\n\n{context}"

# tool 事件结果摘要最大长度（避免把整篇资料推给前端）
TOOL_OUTPUT_LIMIT = 500

# 流式结束原因
FINISH_STOP = "stop"
FINISH_ERROR = "error"

# 知识库检索的工具名 / 事件名（预检索与 knowledge_retrieval 工具同名，前端展示统一）
KNOWLEDGE_TOOL = "knowledge_retrieval"


# ---- 内部：纯函数工具 ----


def system_prompt_of(config: dict) -> str:
    """从配置提取 system_prompt。"""
    return str(config.get("system_prompt", "") or "")


def compose_system_prompt(system_prompt: str, context: str) -> str:
    """把 system_prompt 与「参考资料」拼成最终系统提示词（两者都可为空）。

    工具模式（create_agent）只接受一个 system_prompt 字符串，参考资料需拼进这里；
    纯 RAG 模式则用 build_messages 单独挂一条 SystemMessage（见下）。
    """
    if not context:
        return system_prompt
    reference = REFERENCE_PROMPT.format(context=context)
    return f"{system_prompt}\n\n{reference}" if system_prompt else reference


def build_messages(
    system_prompt: str,
    context: str,
    history: list[tuple[str, str]],
    user_input: str,
) -> list[BaseMessage]:
    """组装 LLM 消息：system_prompt →（参考资料）→ 多轮历史 → 当前输入。"""
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    if context:
        messages.append(SystemMessage(content=REFERENCE_PROMPT.format(context=context)))
    for role, content in history:
        messages.append(
            HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        )
    messages.append(HumanMessage(content=user_input))
    return messages


def _chunk_text(chunk: Any) -> str:
    """从消息/消息块中提取纯文本（兼容 str 与 content blocks 列表）。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                texts.append(str(block.get("text", "")))
        return "".join(texts)
    return ""


def _chunk_reasoning(chunk: Any) -> str:
    """提取推理增量（ReasoningChatOpenAI 已把它放进 additional_kwargs）。"""
    kwargs = getattr(chunk, "additional_kwargs", None) or {}
    value = kwargs.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _accumulate_usage(usage: dict[str, int], chunk: Any, seen_ids: set[str]) -> None:
    """累加 token 用量（按消息 ID 去重，避免同一消息的多个 chunk 重复计数）。"""
    meta = getattr(chunk, "usage_metadata", None)
    if not isinstance(meta, dict):
        return
    chunk_id = getattr(chunk, "id", None)
    if chunk_id and chunk_id in seen_ids:
        return
    if chunk_id:
        seen_ids.add(chunk_id)
    usage["input_tokens"] += int(meta.get("input_tokens") or 0)
    usage["output_tokens"] += int(meta.get("output_tokens") or 0)
    usage["total_tokens"] += int(meta.get("total_tokens") or 0)


# ---- 流式对话的数据结构 ----


@dataclass
class StreamEvent:
    """SSE 事件：`type` 为事件名，`data` 为事件载荷（ApiOutModel）。"""

    type: str
    data: Any


@dataclass
class ChatStreamContext:
    """流式对话预检上下文（只放纯数据，不持有 ORM 对象与数据库会话）。"""

    agent_id: UUID
    conv_id: UUID | None
    model_code: str
    llm: ChatOpenAI
    config: dict
    history: list[tuple[str, str]]
    user_input: str


# ---- 流式执行（模块级：不依赖请求级数据库会话）----


async def _retrieve_hits_short_session(config: dict, query: str) -> list[dict]:
    """用短会话执行 RAG 检索（进入流式阶段后请求级会话已归还连接）。"""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await retrieve_hits(db, config, query)


async def _save_reply(conv_id: UUID | None, reply: str, reasoning: str = "") -> None:
    """流式结束后落库 assistant 回复（独立短会话；失败只记日志，不影响已推送内容）。"""
    if conv_id is None or (not reply and not reasoning):
        return
    # 延迟导入：避免与 core.database 的导入顺序耦合
    from app.core.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            await ConversationService(db).save_assistant_message(conv_id, reply, reasoning)
    except Exception:  # noqa: BLE001 - 落库失败不应影响流式结果
        logger.exception("AI 流式回复落库失败 conversation={}", conv_id)


async def _stream_rag_mode(
    ctx: ChatStreamContext,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
) -> AsyncIterator[StreamEvent]:
    """纯 RAG 模式：用已检索好的参考资料组装消息，再流式调用模型。"""
    messages = build_messages(system_prompt_of(ctx.config), context, ctx.history, ctx.user_input)
    seen_ids: set[str] = set()
    settings = get_settings()
    async for chunk in ctx.llm.astream(messages, timeout=settings.ai_request_timeout):
        # 推理增量 → reasoning 事件（推理型模型才有；无则静默跳过）
        if reasoning := _chunk_reasoning(chunk):
            reasoning_parts.append(reasoning)
            yield StreamEvent("reasoning", AiStreamReasoningOut(content=reasoning))
        text = _chunk_text(chunk)
        if text:
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        _accumulate_usage(usage, chunk, seen_ids)


async def _stream_tool_mode(
    ctx: ChatStreamContext,
    tools: list,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
) -> AsyncIterator[StreamEvent]:
    """工具模式：走 LangChain 智能体逐 token 流式，并把工具调用过程推给前端。

    `context` 是预先检索好的参考资料（已推过 tool 事件），注入 system prompt，
    保证「绑定了知识库必定检索」；模型仍可继续调用 knowledge_retrieval 工具追问细节。
    """
    from langchain.agents import create_agent

    agent = create_agent(
        ctx.llm, tools, system_prompt=compose_system_prompt(system_prompt_of(ctx.config), context)
    )
    messages = build_messages("", "", ctx.history, ctx.user_input)
    started_tools: set[str] = set()
    seen_ids: set[str] = set()

    # stream_mode="messages"：逐 token 产出 (消息块, 元数据) 二元组
    async for chunk, _meta in agent.astream({"messages": messages}, stream_mode="messages"):
        print('切片', chunk)
        # 工具执行结果 → tool done（附带结果摘要）
        if isinstance(chunk, ToolMessage):
            yield StreamEvent(
                "tool",
                AiStreamToolOut(
                    name=chunk.name or "tool",
                    status="done",
                    output=_chunk_text(chunk)[:TOOL_OUTPUT_LIMIT],
                ),
            )
            continue
        # 模型推理增量 → reasoning 事件（推理型模型才有；与正文/tool 可能交错）
        if reasoning := _chunk_reasoning(chunk):
            reasoning_parts.append(reasoning)
            yield StreamEvent("reasoning", AiStreamReasoningOut(content=reasoning))
        # 模型发起工具调用 → tool running（按调用 ID 去重：同一工具可被多次调用）
        for call in getattr(chunk, "tool_call_chunks", None) or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            key = call.get("id") or name
            if key and name and key not in started_tools:
                started_tools.add(key)
                yield StreamEvent("tool", AiStreamToolOut(name=name, status="running"))
        text = _chunk_text(chunk)
        if text:
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        _accumulate_usage(usage, chunk, seen_ids)


async def stream_chat(ctx: ChatStreamContext) -> AsyncIterator[StreamEvent]:
    """流式执行对话，产出 SSE 事件：meta → (reasoning) → (tool) → delta… → done。

    - 流中途异常不再抛出（响应已开始，改不了状态码）：转 `error` 事件后正常收尾；
    - 客户端中断（CancelledError）不发 done，但会把已生成内容落库；
    - 落库用独立短会话 + 取消屏蔽（shield），保证中断时也能写入部分内容。
    - 绑定知识库则先统一检索（两种模式一致），避免工具模式下模型不调工具就"没检索直接答"。
    """
    parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    finish_reason = FINISH_STOP
    tools = build_tools(ctx.config)

    yield StreamEvent(
        "meta",
        AiStreamMetaOut(agent_id=ctx.agent_id, session_id=ctx.conv_id, model=ctx.model_code),
    )

    try:
        # 预检索：只要绑定了知识库就检索（不受"模型是否愿意调工具"影响）
        context = ""
        if ctx.config.get("knowledge_ids"):
            yield StreamEvent("tool", AiStreamToolOut(name=KNOWLEDGE_TOOL, status="running"))
            hits = await _retrieve_hits_short_session(ctx.config, ctx.user_input)
            context = format_context(hits)
            yield StreamEvent(
                "tool",
                AiStreamToolOut(
                    name=KNOWLEDGE_TOOL, status="done", output=f"命中 {len(hits)} 条参考资料"
                ),
            )

        if tools:
            async for event in _stream_tool_mode(ctx, tools, context, parts, reasoning_parts, usage):
                yield event
        else:
            async for event in _stream_rag_mode(ctx, context, parts, reasoning_parts, usage):
                yield event
    except asyncio.CancelledError:
        # 客户端断开：不发 done（连接已断），交由 finally 落库已生成内容
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转 error 事件
        logger.exception("AI 流式对话失败 agent_id={} model={}", ctx.agent_id, ctx.model_code)
        finish_reason = FINISH_ERROR
        yield StreamEvent("error", AiStreamErrorOut(code=500, message=f"模型调用失败: {exc}"))
    finally:
        # 取消屏蔽：客户端中断后仍要完成落库（不 shield 的话 await 会被立即取消）
        with anyio.CancelScope(shield=True):
            await _save_reply(ctx.conv_id, "".join(parts), "".join(reasoning_parts))

    yield StreamEvent(
        "done",
        AiStreamDoneOut(
            session_id=ctx.conv_id,
            content="".join(parts),
            reasoning="".join(reasoning_parts),
            model=ctx.model_code,
            finish_reason=finish_reason,
            usage=AiStreamUsageOut(**usage),
        ),
    )


class AiChatService:
    """AI 对话服务（同步执行 + 流式预检）。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.agent_repo = AgentRepository(db)
        self.version_repo = AgentVersionRepository(db)
        self.instance_repo = ModelInstanceRepository(db)
        self.provider_repo = ModelProviderRepository(db)
        # 会话与消息存取统一委托会话服务（依赖方向单向：chat → conversation）
        self.conv_service = ConversationService(db)

    # ---- 对话（同步）----
    async def chat(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> AiChatOut:
        """同步执行 Agent 对话：加载配置 → 按 config.tools 走智能体（工具调用）或纯 RAG → 调用模型 → 返回回复。

        - 有 user_id 时自动落库会话（传 session_id 复用，否则新建）；
        - 无 user_id（后台任务）不落库。
        """
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        llm, model_code = await self._build_llm(agent, config, conv_id)
        system_prompt = system_prompt_of(config)

        # 工具模式：config.tools 配置了工具时走 LangChain 智能体（模型自主决定调工具）
        tools = build_tools(config)
        settings = get_settings()
        reasoning = ""
        try:
            if tools:
                from langchain.agents import create_agent

                # 工具模式同样先检索并注入参考资料：模型可能不调工具就直接回答，
                # 导致"配了知识库却没检索"；knowledge_retrieval 工具仍可用于追问细节。
                context = await retrieve_context(self.db, config, user_input)
                agent_graph = create_agent(
                    llm, tools, system_prompt=compose_system_prompt(system_prompt, context)
                )
                msgs = build_messages("", "", history, user_input)
                result = await agent_graph.ainvoke({"messages": msgs})
                last = result["messages"][-1]
                reply = last.content if isinstance(last.content, str) else str(last.content)
                # 汇总所有 AI 消息的推理内容（工具模式下可能分布在多轮中间消息）
                reasoning = "".join(
                    _chunk_reasoning(m) for m in result["messages"] if isinstance(m, AIMessage)
                )
            else:
                # 纯 RAG 模式：无条件检索知识库 → 组装上下文 → 调用模型
                context = await retrieve_context(self.db, config, user_input)
                messages = build_messages(system_prompt, context, history, user_input)
                response = await llm.ainvoke(messages, timeout=settings.ai_request_timeout)
                reply = response.content if hasattr(response, "content") else str(response)
                reasoning = _chunk_reasoning(response)
        except Exception as exc:  # noqa: BLE001 - 统一转业务错误
            logger.exception("LLM 调用失败 agent_id={} model={}", agent_id, model_code)
            raise BizError(f"模型调用失败: {exc}") from exc

        # 落库 assistant 回复
        if conv_id is not None:
            await self.conv_service.save_assistant_message(conv_id, reply, reasoning)

        logger.info(
            "AI chat done agent_id={} model={} user_id={} session={}",
            agent_id,
            model_code,
            user_id,
            conv_id,
        )
        return AiChatOut(
            agent_id=agent_id,
            reply=reply,
            reasoning=reasoning,
            model=model_code,
            async_=False,
            session_id=conv_id,
        )

    # ---- 对话（流式预检）----
    async def prepare_chat(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> ChatStreamContext:
        """流式对话预检：校验 Agent → 读版本快照 → 解析会话（落库 user 消息）→ 构造 LLM。

        必须在「响应开始前」执行（HTTP 层由依赖调用）：此阶段失败会抛 AppError，
        由全局异常处理器返回统一 JSON 错误体；进入流式阶段后失败只能以 error 事件返回。
        """
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        llm, model_code = await self._build_llm(agent, config, conv_id)
        return ChatStreamContext(
            agent_id=agent_id,
            conv_id=conv_id,
            model_code=model_code,
            llm=llm,
            config=config,
            history=history,
            user_input=user_input,
        )

    # ---- ARQ 任务入口（异步）----
    @staticmethod
    async def chat_task(
        agent_id: str,
        user_input: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """ARQ 后台任务：独立会话执行对话，返回结果 dict（供 worker 序列化）。"""
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = AiChatService(db)
            out = await service.chat(
                UUID(agent_id),
                user_input,
                UUID(user_id) if user_id else None,
                UUID(session_id) if session_id else None,
            )
            return out.model_dump(by_alias=True)

    # ---- 内部：加载 / 校验 ----
    async def _load_agent_config(self, agent_id: UUID) -> tuple[Agent, dict]:
        """加载 Agent 并校验可运行，返回 (agent, 当前版本配置快照)。"""
        agent = await self.agent_repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status in (
            AgentStatus.PAUSED.value,
            AgentStatus.STOPPED.value,
            AgentStatus.PENDING.value,
        ):
            raise BizError("Agent 已暂停/停止/待发布，无法运行")
        if not agent.current_version:
            raise BizError("Agent 尚未发布，无法运行")
        # 方案 B：运行时读 current_version 对应的版本快照（发布后生效，修改草稿不影响线上）
        version = await self.version_repo.get_by_version(agent_id, agent.current_version)
        if not version:
            raise BizError("Agent 版本快照不存在")
        return agent, (version.config or {})

    async def _resolve_conversation(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None,
        session_id: UUID | None,
    ) -> tuple[UUID | None, list[tuple[str, str]]]:
        """解析会话与多轮历史：有 user_id 时落库（复用/新建），无 user_id（后台任务）不落库。"""
        if user_id is None:
            return session_id, []
        return await self.conv_service.resolve_for_chat(
            agent_id, user_id, session_id, user_input
        )

    # ---- 内部：构造 LLM ----
    async def _build_llm(
        self, agent: Agent, config: dict, session_id: UUID | None = None
    ) -> tuple[ChatOpenAI, str]:
        """根据 Agent 绑定的模型实例构造 ChatOpenAI（OpenAI 兼容协议）。

        返回 (llm, model_code)。Agent 未绑定模型实例时，回退到配置 ai_default_model。
        opencode 网关兼容：base_url 含 opencode 时需携带 x-opencode-session（取会话 ID 前 8 位）。
        """
        settings = get_settings()
        model_code = settings.ai_default_model
        base_url: str | None = None
        api_key: str | None = None

        if agent.model_id is not None:
            instance = await self.instance_repo.get_by_id(agent.model_id)
            if instance is None:
                raise BizError("Agent 绑定的模型实例不存在")
            if instance.status != 1:
                raise BizError("Agent 绑定的模型实例已停用")
            model_code = instance.code
            provider = await self.provider_repo.get_by_id(instance.provider_id)
            if provider is not None:
                base_url = provider.base_url
                api_key = provider.api_key

        if not model_code:
            raise BizError("Agent 未绑定模型实例，且未配置默认模型")

        # opencode 网关兼容：请求需携带 x-opencode-session（取会话 ID 前 8 位）
        default_headers: dict[str, str] | None = None
        if base_url and "opencode" in base_url:
            default_headers = {"x-opencode-session": session_id.hex[:8] if session_id else ""}

        return (
            ReasoningChatOpenAI(
                model=model_code,
                api_key=api_key or "not-set",
                base_url=base_url,
                default_headers=default_headers,
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
                max_retries=settings.ai_max_retries,
                timeout=settings.ai_request_timeout,
                # 流式请求 token 用量（网关不支持时用 AI_STREAM_USAGE=false 关闭）
                stream_usage=settings.ai_stream_usage,
            ),
            model_code,
        )
