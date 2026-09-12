"""AI 会话 Service 层：会话与消息的存取、归属校验与多轮记忆。

职责边界（只管「会话与消息怎么存、能不能给你」，不涉及 LLM / RAG / prompt）：
- 会话 CRUD 与消息查询（`/ai/conversations/*` 接口的业务逻辑）；
- 归属校验：会话必须属于当前登录用户；
- 多轮记忆：解析/新建会话、落库 user 消息、加载最近 N 轮历史、落库 assistant 回复。

调用方：`conversation/api.py`（会话接口）与 `chat/service.py`（对话预检与落库）。
依赖方向单向：chat → conversation。
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.repository import AgentRepository
from app.modules.ai.conversation.schema import (
    ConversationCreate,
    ConversationOut,
    ConversationQuery,
    MessageOut,
)
from app.modules.ai.model import AiConversation, AiMessage
from app.modules.ai.repository import ConversationRepository, MessageRepository
from app.modules.ai.steps import AiStepOut, build_timeline


class ConversationService:
    """AI 会话服务：会话与消息数据访问编排 + 多轮记忆。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.agent_repo = AgentRepository(db)
        self.conv_repo = ConversationRepository(db)
        self.msg_repo = MessageRepository(db)

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
        conv = await self.get_owned(conversation_id, user_id)
        return ConversationOut.model_validate(conv)

    async def delete_conversation(self, conversation_id: UUID, user_id: UUID) -> None:
        """删除会话（校验归属，消息级联删除）。"""
        conv = await self.get_owned(conversation_id, user_id)
        await self.conv_repo.delete(conv)

    async def list_messages(self, conversation_id: UUID, user_id: UUID) -> list[MessageOut]:
        """查询会话消息列表（校验归属，时间升序）。

        `steps` 在这里由「落库动作 + reasoning 全文」**读时派生**为思考/动作统一时间线：
        `thinking` 条目不落库，因此切分规则改进后历史消息会自动跟随，
        第一期写入的老消息（`steps` 里没有 `reasoning_offset`）也立即具备时间线。
        """
        await self.get_owned(conversation_id, user_id)
        messages = await self.msg_repo.list_by_conversation(conversation_id)

        out: list[MessageOut] = []
        for msg in messages:
            item = MessageOut.model_validate(msg)
            if item.role == "assistant" and (msg.steps or msg.reasoning):
                item.steps = [
                    AiStepOut.model_validate(s)
                    for s in build_timeline(msg.steps, msg.reasoning)
                ]
            out.append(item)
        return out

    # ---- 归属校验 ----
    async def get_owned(self, conversation_id: UUID, user_id: UUID) -> AiConversation:
        """查询会话并校验归属当前用户。"""
        conv = await self.conv_repo.get_by_id(conversation_id)
        if not conv:
            raise NotFoundError("会话不存在")
        if conv.user_id != user_id:
            raise BizError("会话不属于当前用户")
        return conv

    # ---- 多轮记忆 ----
    async def resolve_for_chat(
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
            history = await self.load_history(session_id)
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

    async def load_history(self, conversation_id: UUID) -> list[tuple[str, str]]:
        """加载最近 N 轮历史（user/assistant 对），供多轮上下文组装。"""
        messages = await self.msg_repo.list_by_conversation(conversation_id)
        max_rounds = get_settings().ai_max_history_rounds
        recent = messages[-(max_rounds * 2):]
        return [(m.role, m.content) for m in recent]

    async def save_assistant_message(
        self,
        conversation_id: UUID,
        content: str,
        reasoning: str | None = None,
        thinking_ms: int | None = None,
        ttft_ms: int | None = None,
        steps: list[dict] | None = None,
    ) -> None:
        """落库 assistant 回复（会话记忆）。

        `steps` 为工具 / 检索步骤（snake_case dict 列表），落 JSON 列；
        调用方需保证已是可 JSON 序列化的结构（UUID 等需先转 str）。
        """
        await self.msg_repo.create(
            AiMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                reasoning=reasoning,
                thinking_ms=thinking_ms,
                ttft_ms=ttft_ms,
                steps=steps,
            )
        )
