"""AI 能力 Service 层：业务规则、流程编排。

核心职责：加载 Agent 配置 → 解析绑定的模型实例/供应商 → 用 LangChain 构造 LLM →
组装 system_prompt → 调用模型返回回复。

设计约束（未来可拆独立 ai-service 的边界）：
- 所有 LLM 调用收敛在本文件，不散落在业务模块；
- 通过 repository 读取 Agent / 模型配置（数据访问解耦），不反向依赖业务模块内部实现。
"""

from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.ai.schema import AiChatOut


class AiService:
    """AI 对话服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.agent_repo = AgentRepository(db)
        self.version_repo = AgentVersionRepository(db)
        self.instance_repo = ModelInstanceRepository(db)
        self.provider_repo = ModelProviderRepository(db)

    # ---- 对话（同步）----
    async def chat(self, agent_id: UUID, user_input: str, user_id: UUID | None = None) -> AiChatOut:
        """同步执行 Agent 对话：加载配置 → 构造 LLM → 调用 → 返回回复。"""
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

        llm, model_code = await self._build_llm(agent, config)
        system_prompt = self._system_prompt(config)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

        settings = get_settings()
        try:
            response = await llm.ainvoke(messages, timeout=settings.ai_request_timeout)
            reply = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:  # noqa: BLE001 - 统一转业务错误
            logger.exception("LLM 调用失败 agent_id={} model={}", agent_id, model_code)
            raise BizError(f"模型调用失败: {exc}") from exc

        logger.info("AI chat done agent_id={} model={} user_id={}", agent_id, model_code, user_id)
        return AiChatOut(agent_id=agent_id, reply=reply, model=model_code, async_=False)

    # ---- ARQ 任务入口（异步）----
    @staticmethod
    async def chat_task(agent_id: str, user_input: str, user_id: str | None = None) -> dict:
        """ARQ 后台任务：独立会话执行对话，返回结果 dict（供 worker 序列化）。"""
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = AiService(db)
            out = await service.chat(UUID(agent_id), user_input, UUID(user_id) if user_id else None)
            return out.model_dump(by_alias=True)

    # ---- 内部：构造 LLM ----
    async def _build_llm(self, agent: Agent, config: dict) -> tuple[ChatOpenAI, str]:
        """根据 Agent 绑定的模型实例构造 ChatOpenAI（OpenAI 兼容协议）。

        返回 (llm, model_code)。Agent 未绑定模型实例时，回退到配置 ai_default_model。
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

        return (
            ChatOpenAI(
                model=model_code,
                api_key=api_key or "not-set",
                base_url=base_url,
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
