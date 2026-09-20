"""AI 对话 Service 层：Agent 装载校验、模型构造与会话编排。

职责（本模块只负责「怎么问模型」的编排，不处理 SSE 字节流）：
- `chat()`：同步执行一次对话（Agent 测试运行 / ARQ 后台任务复用）；
- `prepare_chat()`：流式对话预检（必须在 HTTP 响应开始前完成，失败仍可返回 JSON 错误体）；
- `chat_task()`：ARQ 后台任务入口。

知识库检索策略：**只要 config.knowledge_ids 非空就先检索**（两种模式一致）。
工具模式下检索本可由模型自主决定，但模型经常不调工具就直接回答，导致「配了知识库却没检索」，
故这里统一预检索并把「参考资料」注入 system prompt；knowledge_retrieval 工具仍保留，供模型追问细节。

两条执行路径（口径必须一致，否则 `done` 事件与历史接口对不上）：
- 同步 `chat()`：一次拿完整回复，步骤/用量在本文件内汇总；
- 流式 `stream_chat(ctx)`：在 `chat/stream.py`，由 `chat/api.py` 的 SSE 端点驱动。

相邻模块：
- `chat/messages.py`：提示词组装与消息/用量解析（纯函数，两条路径共用）；
- `chat/runtime.py`：`LlmRuntime` / `ChatStreamContext` / `TokenTimer` / `StepCollector` 等运行时数据结构；
- `chat/stream.py`：流式执行与旁路落库（`persist_usage` 亦被本模块同步路径复用）。

设计约束（未来可拆独立 ai-service 的边界）：
- 所有 LLM 调用收敛在本模块，RAG 检索收敛在 retrieval.py，向量库封装在 vector.py；
- 会话与消息的存取统一委托 `ConversationService`（conversation 子包）；
- 流式生成器**不得使用请求级数据库会话**（HTTP 响应开始后请求会话已归还连接），
  故流式路径全部收在 `chat/stream.py`，本文件只做「响应开始前」的事。
"""

import time
from decimal import Decimal
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus, UsageCallType
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.ai.chat.messages import (
    accumulate_usage,
    build_messages,
    chunk_reasoning,
    collect_steps_from_messages,
    compose_system_prompt,
    system_prompt_of,
)
from app.modules.ai.chat.runtime import (
    FINISH_ABSTAIN,
    FINISH_STOP,
    KNOWLEDGE_TOOL,
    ChatStreamContext,
    LlmRuntime,
)
from app.modules.ai.chat.schema import AiChatOut
from app.modules.ai.chat.stream import persist_usage
from app.modules.ai.conversation.service import ConversationService
from app.modules.ai.grounding import (
    EvidenceAssessment,
    abstain_message,
    assess_evidence,
    collect_sources,
    grounding_out,
    grounding_prompt_enabled,
    needs_no_evidence_note,
    should_abstain,
)
from app.modules.ai.query import apply_query_rewrite, build_rewrite_llm
from app.modules.ai.reasoning import ReasoningChatOpenAI
from app.modules.ai.retrieval import build_sources, format_context, retrieve_hits
from app.modules.ai.steps import AiStepOut, StepKind, build_timeline
from app.modules.ai.tools import build_tools


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
        - 无 user_id（后台任务）不落库；
        - 证据不足且未配置非知识库工具时直接拒答（固定文案，**不调用 LLM**），
          与流式路径共用 `grounding.py` 的同一套判定。
        """
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        started_at = time.perf_counter()
        runtime = await self._build_llm(agent, config, conv_id)
        llm, model_code = runtime.llm, runtime.model_code
        system_prompt = system_prompt_of(config)

        # 引用编号计数器：与流式路径口径一致（预检索与检索工具共用）
        cited: dict[str, int] = {"n": 0}
        # 工具模式：config.tools 配置了工具时走 LangChain 智能体（模型自主决定调工具）
        tools = build_tools(config, cited)
        settings = get_settings()
        reasoning = ""
        steps: list[dict] = []
        # token 用量（同步路径从 usage_metadata 汇总；网关不返回则全 0）
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        seen_usage_ids: set[str] = set()
        # 证据接地：判定结果 + 提示词开关（与流式路径共用 grounding.py 的同一套函数）
        assessment: EvidenceAssessment | None = None
        abstain = False
        no_evidence_note = False
        finish_reason = FINISH_STOP
        grounding = grounding_prompt_enabled(config)
        try:
            # 预检索：与流式路径保持一致 —— 只要绑定知识库就先检索并注入参考资料，
            # 避免工具模式下模型不调工具就"没检索直接答"；同时记录一条 retrieval 步骤。
            context = ""
            if config.get("knowledge_ids"):
                # 查询改写（可选）：把「那这个怎么办？」这类指代 / 省略问题结合历史补全成
                # 自足问题后再检索。⚠️ 只影响检索用 query —— 后面组装消息仍用原始 user_input
                # （改写绝不允许替换发给模型的消息，也不允许进入最终答案）。
                rewrite = await apply_query_rewrite(runtime.rewrite_llm, user_input, history)
                hits = await retrieve_hits(self.db, config, rewrite.rewritten_query)
                # 证据接地判定：与流式路径共用 grounding.py 的同一套函数，口径必然一致
                assessment = assess_evidence(hits)
                abstain = should_abstain(hits, config)
                no_evidence_note = needs_no_evidence_note(hits, config)
                # 与流式路径口径一致：读改写紧邻，编号请求内全局唯一
                start = cited["n"]
                cited["n"] += len(hits)
                context = format_context(hits, start)
                # 拒答时把判定结论写进步骤摘要（与流式路径的文案一致）
                summary = f"命中 {len(hits)} 条参考资料"
                if abstain:
                    summary = (
                        f"{summary}（相关度不足，已拒答）" if hits else "未命中参考资料（已拒答）"
                    )
                steps.append(
                    {
                        "step_id": uuid4().hex,
                        "kind": StepKind.RETRIEVAL.value,
                        "name": KNOWLEDGE_TOOL,
                        "status": "done",
                        "output": summary,
                        "cost_ms": None,
                        # 预检索在 LLM 调用之前，推理必为空
                        "reasoning_offset": 0,
                        "sources": build_sources(hits, start) or None,
                    }
                )

            if abstain:
                # 证据不足：后端固定文案，**不调用任何 LLM**（与流式路径同一套判定）
                reply = abstain_message()
                finish_reason = FINISH_ABSTAIN
            elif tools:
                from langchain.agents import create_agent

                agent_graph = create_agent(
                    llm,
                    tools,
                    system_prompt=compose_system_prompt(
                        system_prompt,
                        context,
                        grounding=grounding,
                        no_evidence_note=no_evidence_note,
                    ),
                )
                msgs = build_messages("", "", history, user_input)
                result = await agent_graph.ainvoke({"messages": msgs})
                last = result["messages"][-1]
                reply = last.content if isinstance(last.content, str) else str(last.content)
                # 汇总所有 AI 消息的推理内容（工具模式下可能分布在多轮中间消息）
                reasoning = "".join(
                    chunk_reasoning(m) for m in result["messages"] if isinstance(m, AIMessage)
                )
                # 从完整消息序列反推模型发起的工具调用步骤（非流式无单步耗时）
                steps.extend(collect_steps_from_messages(result["messages"]))
                # 汇总多轮 LLM 调用的 token 用量（按消息 ID 去重）
                for message in result["messages"]:
                    accumulate_usage(usage, message, seen_usage_ids)
            else:
                # 纯 RAG 模式：组装上下文 → 调用模型
                messages = build_messages(
                    system_prompt,
                    context,
                    history,
                    user_input,
                    grounding=grounding,
                    no_evidence_note=no_evidence_note,
                )
                response = await llm.ainvoke(messages, timeout=settings.ai_request_timeout)
                reply = response.content if hasattr(response, "content") else str(response)
                reasoning = chunk_reasoning(response)
                accumulate_usage(usage, response, seen_usage_ids)
        except Exception as exc:  # noqa: BLE001 - 统一转业务错误
            logger.exception("LLM 调用失败 agent_id={} model={}", agent_id, model_code)
            raise BizError(f"模型调用失败: {exc}") from exc

        # 落库 assistant 回复（同步路径拿不到「首 token」时刻，两个耗时字段留空）
        message_id: UUID | None = None
        if conv_id is not None:
            message = await self.conv_service.save_assistant_message(
                conv_id, reply, reasoning, None, None, steps or None
            )
            message_id = message.id if message else None

        # 落库用量：无 user_id / 无会话的测试运行也计入（维度字段留空）
        await persist_usage(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            user_id=user_id,
            agent_id=agent_id,
            agent_name=agent.name,
            organization_id=agent.organization_id,
            model_instance_id=runtime.model_instance_id,
            provider_id=runtime.provider_id,
            provider_code=runtime.provider_code,
            model_code=model_code,
            conversation_id=conv_id,
            message_id=message_id,
            input_price=runtime.input_price,
            output_price=runtime.output_price,
            call_type=UsageCallType.TOOL.value if tools else UsageCallType.CHAT.value,
            latency_ms=round((time.perf_counter() - started_at) * 1000),
        )

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
            # 时间线：thinking 片段与动作按位置交错（与流式路径同一纯函数）
            steps=[AiStepOut.model_validate(s) for s in build_timeline(steps, reasoning)],
            # 证据接地审计（与流式 done 事件同一套组装函数，字段必然一致）
            finish_reason=finish_reason,
            grounding=grounding_out(assessment, collect_sources(steps), reply, abstained=abstain),
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

        入口处即打点 `started_at`，作为 `ttft_ms`（首字节耗时）的起点。
        """
        started_at = time.perf_counter()
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        runtime = await self._build_llm(agent, config, conv_id)
        return ChatStreamContext(
            agent_id=agent_id,
            conv_id=conv_id,
            model_code=runtime.model_code,
            llm=runtime.llm,
            config=config,
            history=history,
            user_input=user_input,
            started_at=started_at,
            # 用量统计维度快照（username 由 HTTP 层依赖补填）
            agent_name=agent.name,
            organization_id=agent.organization_id,
            user_id=user_id,
            model_instance_id=runtime.model_instance_id,
            provider_id=runtime.provider_id,
            provider_code=runtime.provider_code,
            input_price=runtime.input_price,
            output_price=runtime.output_price,
            # 查询改写客户端（流式阶段请求级会话已归还，但它不依赖会话，可安全传递）
            rewrite_llm=runtime.rewrite_llm,
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
    ) -> LlmRuntime:
        """根据 Agent 绑定的模型实例构造 ChatOpenAI（OpenAI 兼容协议）。

        返回 `LlmRuntime`：客户端 + 模型 / 供应商快照与单价（供用量统计打点），
        以及查询改写专用客户端（复用同一套供应商参数，但超时 / token 预算独立）。
        Agent 未绑定模型实例时，回退到配置 ai_default_model。
        opencode 网关兼容：base_url 含 opencode 时需携带 x-opencode-session（取会话 ID 前 8 位）。
        """
        settings = get_settings()
        model_code = settings.ai_default_model
        base_url: str | None = None
        api_key: str | None = None
        model_instance_id: UUID | None = None
        provider_id: UUID | None = None
        provider_code: str | None = None
        input_price: Decimal | None = None
        output_price: Decimal | None = None

        if agent.model_id is not None:
            instance = await self.instance_repo.get_by_id(agent.model_id)
            if instance is None:
                raise BizError("Agent 绑定的模型实例不存在")
            if instance.status != 1:
                raise BizError("Agent 绑定的模型实例已停用")
            model_code = instance.code
            model_instance_id = instance.id
            input_price = instance.input_price
            output_price = instance.output_price
            provider = await self.provider_repo.get_by_id(instance.provider_id)
            if provider is not None:
                provider_id = provider.id
                provider_code = provider.code
                base_url = provider.base_url
                api_key = provider.api_key

        if not model_code:
            raise BizError("Agent 未绑定模型实例，且未配置默认模型")

        # opencode 网关兼容：请求需携带 x-opencode-session（取会话 ID 前 8 位）
        default_headers: dict[str, str] | None = None
        if base_url and "opencode" in base_url:
            default_headers = {"x-opencode-session": session_id.hex[:8] if session_id else ""}

        return LlmRuntime(
            llm=ReasoningChatOpenAI(
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
            model_code=model_code,
            model_instance_id=model_instance_id,
            provider_id=provider_id,
            provider_code=provider_code,
            input_price=input_price,
            output_price=output_price,
            # 查询改写客户端：复用上面的 model / base_url / api_key / default_headers，
            # 但超时与 token 预算独立（且 max_retries=0），避免拖慢或影响主 Chat 调用；
            # 开关关闭时它不会被调用（唯一的 gate 在 query.apply_query_rewrite 内）。
            rewrite_llm=build_rewrite_llm(
                model_code,
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            ),
        )
