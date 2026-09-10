"""AI 能力 Service 层：业务规则、流程编排。

核心职责：加载 Agent 配置 → 解析绑定的模型实例/供应商 → 用 LangChain 构造 LLM →
按 config.tools 走智能体（工具调用）或纯 RAG 检索 → 组装上下文 → 调用模型返回回复 → 会话落库。

设计约束（未来可拆独立 ai-service 的边界）：
- 所有 LLM / embedding / 向量检索调用收敛在本模块（service.py + vector.py），不散落在业务模块；
- 通过 repository 读取 Agent / 模型 / 知识库配置（数据访问解耦），不反向依赖业务模块内部实现。
"""

from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.common.pagination import PageParams, PageResult
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.ai.model import AiConversation, AiMessage
from app.modules.ai.repository import ConversationRepository, MessageRepository
from app.modules.ai.schema import (
    AiChatOut,
    ConversationCreate,
    ConversationOut,
    ConversationQuery,
    MessageOut,
)
from app.modules.ai.tools import build_tools


class AiService:
    """AI 对话服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.agent_repo = AgentRepository(db)
        self.version_repo = AgentVersionRepository(db)
        self.instance_repo = ModelInstanceRepository(db)
        self.provider_repo = ModelProviderRepository(db)
        self.conv_repo = ConversationRepository(db)
        self.msg_repo = MessageRepository(db)

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
        agent = await self.agent_repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status in (AgentStatus.PAUSED.value, AgentStatus.STOPPED.value):
            raise BizError("Agent 已暂停/停止，无法运行")
        if agent.current_version == 0:
            raise BizError("Agent 尚未发布，无法运行")

        # 方案 B：运行时读 current_version 对应的版本快照（发布后生效，修改草稿不影响线上）
        version = await self.version_repo.get_by_version(agent_id, agent.current_version)
        if not version:
            raise BizError("Agent 版本快照不存在")
        config = version.config or {}

        # 会话记忆（user_id 存在时落库）—— 提前解析，以便拿到会话 ID 用于 opencode 兼容头
        conv_id: UUID | None = session_id
        history: list[tuple[str, str]] = []
        if user_id is not None:
            conv_id, history = await self._resolve_conversation(
                agent_id, user_id, session_id, user_input
            )

        llm, model_code = await self._build_llm(agent, config, conv_id)
        system_prompt = self._system_prompt(config)

        # 工具模式：config.tools 配置了工具时走 LangChain 智能体（模型自主决定调工具）
        tools = build_tools(self, config)
        settings = get_settings()
        try:
            if tools:
                from langchain.agents import create_agent

                agent = create_agent(llm, tools, system_prompt=system_prompt)
                msgs = [
                    HumanMessage(content=c) if role == "user" else AIMessage(content=c)
                    for role, c in history
                ]
                msgs.append(HumanMessage(content=user_input))
                print('智能体消息：', msgs)
                result = await agent.ainvoke({"messages": msgs})
                print('智能体结果：', result)
                last = result["messages"][-1]
                reply = last.content if isinstance(last.content, str) else str(last.content)
            else:
                # 纯 RAG 模式：无条件检索知识库 → 组装上下文 → 调用模型
                context = await self._retrieve_context(config, user_input)
                print('检索内容：', context)
                messages = [SystemMessage(content=system_prompt)]
                if context:
                    messages.append(
                        SystemMessage(
                            content=f"以下是参考资料，请优先基于这些资料回答用户问题，不要编造资料外的内容：\n\n{context}"
                        )
                    )
                for role, content in history:
                    messages.append(
                        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
                    )
                messages.append(HumanMessage(content=user_input))
                print('大模型消息：', messages)
                response = await llm.ainvoke(messages, timeout=settings.ai_request_timeout)
                print('大模型结果：', response)
                reply = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:  # noqa: BLE001 - 统一转业务错误
            logger.exception("LLM 调用失败 agent_id={} model={}", agent_id, model_code)
            raise BizError(f"模型调用失败: {exc}") from exc

        # 落库 assistant 回复
        if conv_id is not None:
            await self.msg_repo.create(
                AiMessage(conversation_id=conv_id, role="assistant", content=reply)
            )

        logger.info(
            "AI chat done agent_id={} model={} user_id={} session={}",
            agent_id,
            model_code,
            user_id,
            conv_id,
        )
        return AiChatOut(
            agent_id=agent_id, reply=reply, model=model_code, async_=False, session_id=conv_id
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
            service = AiService(db)
            out = await service.chat(
                UUID(agent_id),
                user_input,
                UUID(user_id) if user_id else None,
                UUID(session_id) if session_id else None,
            )
            return out.model_dump(by_alias=True)

    # ---- 会话管理 ----
    async def list_conversations(
        self, query: ConversationQuery, user_id: UUID
    ) -> PageResult[ConversationOut]:
        """分页查询当前用户的会话列表。"""
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.conv_repo.list_page(page_params, user_id, query.agent_id)
        return PageResult(
            list=[ConversationOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    async def create_conversation(
        self, req: ConversationCreate, user_id: UUID
    ) -> ConversationOut:
        """手动创建会话（校验 Agent 存在）。"""
        agent = await self.agent_repo.get_by_id(req.agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        conv = AiConversation(
            agent_id=req.agent_id,
            user_id=user_id,
            title=req.title or "新会话",
            status="active",
        )
        return ConversationOut.model_validate(await self.conv_repo.create(conv))

    async def get_conversation(self, conversation_id: UUID, user_id: UUID) -> ConversationOut:
        """查询会话详情（校验归属）。"""
        conv = await self._get_owned_conversation(conversation_id, user_id)
        return ConversationOut.model_validate(conv)

    async def delete_conversation(self, conversation_id: UUID, user_id: UUID) -> None:
        """删除会话（校验归属，消息级联删除）。"""
        conv = await self._get_owned_conversation(conversation_id, user_id)
        await self.conv_repo.delete(conv)

    async def list_messages(self, conversation_id: UUID, user_id: UUID) -> list[MessageOut]:
        """查询会话消息列表（校验归属，时间升序）。"""
        await self._get_owned_conversation(conversation_id, user_id)
        messages = await self.msg_repo.list_by_conversation(conversation_id)
        return [MessageOut.model_validate(m) for m in messages]

    async def _get_owned_conversation(
        self, conversation_id: UUID, user_id: UUID
    ) -> AiConversation:
        """查询会话并校验归属当前用户。"""
        conv = await self.conv_repo.get_by_id(conversation_id)
        if not conv:
            raise NotFoundError("会话不存在")
        if conv.user_id != user_id:
            raise BizError("会话不属于当前用户")
        return conv

    # ---- 内部：RAG 检索 ----
    async def _retrieve_context(self, config: dict, user_input: str) -> str:
        """RAG 检索：读 config.knowledge_ids → embedding → 检索 Qdrant → 拼参考资料。

        约束：Agent 绑定的多个知识库应使用同一 embedding 模型（用第一个知识库的模型构造查询向量）。
        """
        knowledge_ids = config.get("knowledge_ids") or []
        if not knowledge_ids:
            return ""
        from app.modules.ai import vector
        from app.modules.knowledge.repository import KnowledgeBaseRepository

        kb_repo = KnowledgeBaseRepository(self.db)
        kbs = []
        for kid in knowledge_ids:
            kb = await kb_repo.get_by_id(UUID(kid))
            if kb is not None and kb.status == 1:
                kbs.append(kb)
        if not kbs:
            return ""

        embeddings = await vector.build_embeddings(self.db, kbs[0].embedding_model_id)
        query_vector = await embeddings.aembed_query(user_input)
        hits = await vector.search_chunks(
            query_vector, [kb.id.hex for kb in kbs], get_settings().rag_top_k
        )
        if not hits:
            return ""
        parts = [f"[{i + 1}] {h['content']}" for i, h in enumerate(hits)]
        return "\n\n".join(parts)

    # ---- 内部：会话记忆 ----
    async def _resolve_conversation(
        self,
        agent_id: UUID,
        user_id: UUID,
        session_id: UUID | None,
        user_input: str,
    ) -> tuple[UUID, list[tuple[str, str]]]:
        """解析会话：复用（校验归属）或新建，落库 user 消息，返回 (conv_id, history)。"""
        if session_id is not None:
            conv = await self.conv_repo.get_by_id(session_id)
            if not conv:
                raise NotFoundError("会话不存在")
            if conv.agent_id != agent_id:
                raise BizError("会话不属于该 Agent")
            if conv.user_id != user_id:
                raise BizError("会话不属于当前用户")
            history = await self._load_history(session_id)
        else:
            conv = AiConversation(
                agent_id=agent_id,
                user_id=user_id,
                title=user_input[:50] or "新会话",
                status="active",
            )
            conv = await self.conv_repo.create(conv)
            session_id = conv.id
            history = []
        await self.msg_repo.create(
            AiMessage(conversation_id=session_id, role="user", content=user_input)
        )
        return session_id, history

    async def _load_history(self, conversation_id: UUID) -> list[tuple[str, str]]:
        """加载最近 N 轮历史（user/assistant 对），供多轮上下文组装。"""
        messages = await self.msg_repo.list_by_conversation(conversation_id)
        max_rounds = get_settings().ai_max_history_rounds
        recent = messages[-(max_rounds * 2):]
        return [(m.role, m.content) for m in recent]

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
            ChatOpenAI(
                model=model_code,
                api_key=api_key or "not-set",
                base_url=base_url,
                default_headers=default_headers,
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
                max_retries=settings.ai_max_retries,
                timeout=settings.ai_request_timeout,
            ),
            model_code,
        )

    @staticmethod
    def _system_prompt(config: dict) -> str:
        """从配置提取 system_prompt。"""
        return str(config.get("system_prompt", "") or "")
